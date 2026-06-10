"""2.7 工具执行网关 (Tool Executor & Action Gateway)

职责边界：系统的"数据面"与"物理手脚"。没有认知能力，只负责把认知引擎
下发的标准数字指令安全、稳定地转化为真实调用。

- 协议转换与鉴权拦截：参数校验/序列化，统一注入鉴权上下文（auth_provider）；
- 基建级容错防御：重试 + 超时 + 限流 + 熔断；当且仅当基建手段无法恢复时，
  向上层抛出清晰的 ToolExecutionError，请求 AI 重新决策；
- 执行态沙盒隔离：sandboxed 技能路由到 SandboxRunner 执行。
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import time
from typing import Any, Callable

from ..config import ResilienceConfig
from ..registry.tool_registry import ToolRegistry
from ..types import ToolExecutionError, ToolInstruction, ToolOutcome
from .resilience import CircuitBreaker, RateLimiter

logger = logging.getLogger(__name__)


class SandboxRunner:
    """高风险操作的沙盒执行占位实现。

    生产环境应替换为真正的隔离运行时（容器/microVM/受限解释器），
    防止恶意或幻觉指令破坏宿主系统。
    """

    def run(self, handler: Callable[..., str], arguments: dict[str, Any]) -> str:
        raise ToolExecutionError(
            handler.__name__, "沙盒运行时未配置，已拒绝执行高风险操作", retryable=False
        )


class ActionGateway:
    def __init__(
        self,
        registry: ToolRegistry,
        config: ResilienceConfig,
        auth_provider: Callable[[], dict[str, str]] | None = None,
        sandbox: SandboxRunner | None = None,
    ):
        self._registry = registry
        self._cfg = config
        self._auth_provider = auth_provider or (lambda: {})
        self._sandbox = sandbox or SandboxRunner()
        self._rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self._breakers: dict[str, CircuitBreaker] = {}
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def _breaker(self, tool_name: str) -> CircuitBreaker:
        if tool_name not in self._breakers:
            self._breakers[tool_name] = CircuitBreaker(
                self._cfg.circuit_failure_threshold, self._cfg.circuit_recovery_time_s
            )
        return self._breakers[tool_name]

    def execute(self, instruction: ToolInstruction) -> ToolOutcome:
        """执行单条指令。基建错误在网关内消化；不可恢复时抛 ToolExecutionError。"""
        name = instruction.tool_name
        try:
            spec = self._registry.get(name)
        except KeyError as e:
            raise ToolExecutionError(name, str(e), retryable=False) from e

        breaker = self._breaker(name)
        if not breaker.allow():
            raise ToolExecutionError(name, "熔断器打开：该工具近期连续失败，已降级", retryable=True)

        # 协议转换：必填项校验（认知引擎产出的指令也需在数据面二次防御）
        required = spec.input_schema.get("required", [])
        missing = [k for k in required if k not in instruction.arguments]
        if missing:
            raise ToolExecutionError(name, f"缺少必填参数: {missing}", retryable=False)

        # 鉴权拦截：将凭证注入执行上下文（业务 handler 按需读取，模型不可见）
        auth_context = self._auth_provider()

        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            self._rate_limiter.acquire()
            try:
                result = self._invoke_with_timeout(spec.sandboxed, spec.handler,
                                                   instruction.arguments, auth_context)
                breaker.record_success()
                return ToolOutcome(tool_use_id=instruction.tool_use_id, content=result)
            except ToolExecutionError:
                breaker.record_failure()
                raise
            except concurrent.futures.TimeoutError as e:
                last_error = e
                breaker.record_failure()
                logger.warning("工具 %s 第 %d 次执行超时", name, attempt + 1)
            except Exception as e:  # noqa: BLE001 — 数据面统一兜底，转译后上抛
                last_error = e
                breaker.record_failure()
                logger.warning("工具 %s 第 %d 次执行失败: %s", name, attempt + 1, e)
            if attempt < self._cfg.max_retries:
                delay = min(
                    self._cfg.retry_base_delay_s * (2 ** attempt) + random.uniform(0, 0.5),
                    self._cfg.retry_max_delay_s,
                )
                time.sleep(delay)

        raise ToolExecutionError(name, f"重试 {self._cfg.max_retries} 次后仍失败: {last_error}", retryable=False)

    def execute_safe(self, instruction: ToolInstruction) -> ToolOutcome:
        """执行并把不可恢复错误转为 is_error 结果，供引擎以 tool_result 回喂模型重新决策。"""
        try:
            return self.execute(instruction)
        except ToolExecutionError as e:
            return ToolOutcome(
                tool_use_id=instruction.tool_use_id,
                content=f"ToolExecutionError: {e}",
                is_error=True,
            )

    def _invoke_with_timeout(
        self,
        sandboxed: bool,
        handler: Callable[..., str],
        arguments: dict[str, Any],
        auth_context: dict[str, str],
    ) -> str:
        def call() -> str:
            if sandboxed:
                return self._sandbox.run(handler, arguments)
            # handler 声明了 auth_context 参数时才注入，避免污染普通技能签名
            if "auth_context" in handler.__code__.co_varnames:
                return handler(**arguments, auth_context=auth_context)
            return handler(**arguments)

        future = self._pool.submit(call)
        return future.result(timeout=self._cfg.tool_timeout_s)
