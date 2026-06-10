"""全局配置：模型 Provider、容错参数、记忆存储路径。

模型后端可配置（环境变量优先）：
- anthropic（默认）：Anthropic 官方 API 或任何 Anthropic 协议兼容网关
- openai_compatible：本地部署模型服务（GLM-V5 via vLLM/sglang 等 OpenAI 兼容端点）

环境变量：
  LRA_PROVIDER       anthropic | openai_compatible
  LRA_BASE_URL       服务地址（如 http://localhost:8000/v1）
  LRA_API_KEY        鉴权 key（本地服务通常任意值，如 EMPTY）
  LRA_ENGINE_MODEL   认知引擎/规划模型名（如 glm-v5）
  LRA_ROUTER_MODEL   路由/记忆筛选/压缩模型名（本地部署可与引擎同一模型）
  LRA_JSON_MODE      =1 时对 openai_compatible 启用 response_format=json_object

能力门控：adaptive thinking、effort、结构化输出、cache_control 是 Anthropic 专有参数，
发给 GLM 等后端会被拒绝 —— 由 capabilities 统一开关，调用方不感知差异。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(path: str | os.PathLike | None = None) -> dict[str, str]:
    """加载 .env 配置文件（KEY=VALUE 格式，# 开头为注释）。

    查找顺序：显式传入 path → 环境变量 LRA_ENV_FILE → 当前工作目录 ./.env。
    已存在的环境变量优先，不会被 .env 覆盖（shell 显式 export 的值最大）。
    在 config 模块导入时自动执行一次。
    """
    candidate = Path(path) if path else Path(os.environ.get("LRA_ENV_FILE", ".env"))
    if not candidate.is_file():
        return {}
    loaded: dict[str, str] = {}
    for lineno, raw in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if not key:
            logger.warning("%s 第 %d 行格式无效，已忽略", candidate, lineno)
            continue
        loaded[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return loaded


load_env_file()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ModelCapabilities:
    """后端支持的 Anthropic 专有能力；不支持的参数一律不发送。"""

    adaptive_thinking: bool = True
    effort: bool = True
    structured_outputs: bool = True     # output_config.format=json_schema
    prompt_caching: bool = True         # system 块 cache_control
    json_mode: bool = False             # OpenAI 兼容端点的 response_format=json_object


def _default_capabilities(provider: str) -> ModelCapabilities:
    if provider == "openai_compatible":
        return ModelCapabilities(
            adaptive_thinking=False, effort=False,
            structured_outputs=False, prompt_caching=False,
            json_mode=_env("LRA_JSON_MODE", "0") == "1",
        )
    return ModelCapabilities()


@dataclass(frozen=True)
class ModelConfig:
    provider: str = field(default_factory=lambda: _env("LRA_PROVIDER", "anthropic"))
    base_url: str | None = field(default_factory=lambda: os.environ.get("LRA_BASE_URL"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("LRA_API_KEY"))
    engine_model: str = field(default_factory=lambda: _env("LRA_ENGINE_MODEL", "claude-opus-4-8"))
    router_model: str = field(default_factory=lambda: _env("LRA_ROUTER_MODEL", "claude-haiku-4-5"))
    engine_max_tokens: int = 16000
    router_max_tokens: int = 1024
    effort: str = "high"
    capabilities: ModelCapabilities | None = None

    @property
    def caps(self) -> ModelCapabilities:
        return self.capabilities or _default_capabilities(self.provider)


@dataclass(frozen=True)
class ResilienceConfig:
    """工具执行网关的基建级容错参数。"""

    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    tool_timeout_s: float = 60.0
    tool_max_workers: int = 8
    rate_limit_per_minute: int = 120
    circuit_failure_threshold: int = 5
    circuit_recovery_time_s: float = 30.0


@dataclass(frozen=True)
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    memory_root: str = field(
        default_factory=lambda: os.environ.get("LRA_MEMORY_ROOT", "data/memory_store")
    )
    # Agentic Loop 单任务最大迭代轮数，防止失控循环
    max_loop_iterations: int = 25
    # 上下文压缩触发阈值（估算 token 数）
    context_compaction_threshold_tokens: int = 150_000


DEFAULT_CONFIG = Config()
