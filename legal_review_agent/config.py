"""全局配置：模型选型、容错参数、记忆存储路径。

模型分层策略（成本/能力分离）：
- 认知引擎 / 任务规划：claude-opus-4-8（adaptive thinking，长链路推理）
- 意图路由 / 记忆相关性筛选：claude-haiku-4-5（低成本、低延迟的粗粒度分类）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    engine_model: str = "claude-opus-4-8"
    router_model: str = "claude-haiku-4-5"
    engine_max_tokens: int = 16000
    router_max_tokens: int = 1024
    effort: str = "high"


@dataclass(frozen=True)
class ResilienceConfig:
    """工具执行网关的基建级容错参数。"""

    max_retries: int = 3
    retry_base_delay_s: float = 1.0
    retry_max_delay_s: float = 30.0
    tool_timeout_s: float = 60.0
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
