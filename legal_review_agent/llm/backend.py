"""模型后端抽象层：把不同 Provider 的协议差异收敛到统一接口。

调用方（路由/规划/组装/引擎）只依赖本模块的归一化类型，不再直接触碰 SDK：
- TextBlock / ToolUseBlock / ThinkingBlock：归一化内容块（与 Anthropic 块字段对齐，
  model_dump 后可直接作为 Anthropic 消息内容回传）；
- LLMResponse：归一化响应（content / stop_reason / usage）；
- ChatBackend.complete()：非流式 + 可选 JSON Schema 约束（路由/规划/压缩/记忆筛选用）；
- ChatBackend.engine_turn()：流式 + 工具调用（认知引擎 Agentic Loop 用），
  文本增量经 on_text 回调推给前端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel

from ..config import ModelConfig


class TextBlock(BaseModel):
    type: str = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = {}


class ThinkingBlock(BaseModel):
    """Anthropic adaptive thinking 块；signature 必须原样保留回传。"""

    type: str = "thinking"
    thinking: str = ""
    signature: str = ""


class LLMResponse(BaseModel):
    content: list[Any]            # TextBlock | ToolUseBlock | ThinkingBlock | 原样 dict
    stop_reason: str = "end_turn"
    usage: dict[str, Any] = {}


class ChatBackend(ABC):
    def __init__(self, config: ModelConfig):
        self._cfg = config
        self.caps = config.caps

    @abstractmethod
    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        thinking: bool = False,
    ) -> LLMResponse:
        """非流式补全。json_schema 非空时尽后端所能约束输出为该 Schema 的 JSON。"""

    @abstractmethod
    def engine_turn(
        self,
        *,
        model: str,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """流式 + 工具调用的一轮引擎请求。

        文本增量经 on_text 回调；推理模型的思考增量经 on_thinking 回调
        （GLM 的 reasoning_content / Claude 的 thinking_delta）。"""


def build_backend(config: ModelConfig) -> ChatBackend:
    if config.provider == "openai_compatible":
        from .openai_backend import OpenAICompatBackend
        return OpenAICompatBackend(config)
    if config.provider == "anthropic":
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(config)
    raise ValueError(f"未知的模型 Provider: {config.provider}（支持 anthropic / openai_compatible）")
