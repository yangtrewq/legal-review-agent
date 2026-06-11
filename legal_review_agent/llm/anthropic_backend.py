"""Anthropic 后端：官方 API 或任何 Anthropic 协议兼容网关（base_url 可配）。

能力按 caps 门控（thinking/effort/structured_outputs/prompt_caching），
因此同一实现也可对接关闭部分能力的 Anthropic 兼容代理。
"""

from __future__ import annotations

from typing import Any, Callable

import anthropic

from ..config import ModelConfig
from ..observability.tracing import serialize
from .backend import ChatBackend, LLMResponse, TextBlock, ThinkingBlock, ToolUseBlock


def _normalize_content(content: list[Any]) -> list[Any]:
    """SDK 块 → 归一化块；未知类型原样保留 dict，保证回传不丢字段。"""
    normalized: list[Any] = []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text":
            normalized.append(TextBlock(text=block.text))
        elif btype == "tool_use":
            normalized.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        elif btype == "thinking":
            normalized.append(ThinkingBlock(thinking=block.thinking, signature=block.signature))
        else:
            normalized.append(serialize(block))
    return normalized


class AnthropicBackend(ChatBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        kwargs: dict[str, Any] = {}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key
        self._client = anthropic.Anthropic(**kwargs)

    def complete(self, *, model, system, messages, max_tokens,
                 json_schema=None, thinking=False) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if thinking and self.caps.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if json_schema is not None and self.caps.structured_outputs:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=serialize(messages),
            **kwargs,
        )
        return LLMResponse(
            content=_normalize_content(response.content),
            stop_reason=response.stop_reason or "end_turn",
            usage=serialize(response.usage) or {},
        )

    def engine_turn(self, *, model, system, messages, tools, max_tokens,
                    on_text: Callable[[str], None] | None = None,
                    on_thinking: Callable[[str], None] | None = None) -> LLMResponse:
        if not self.caps.prompt_caching:
            system = [{k: v for k, v in blk.items() if k != "cache_control"} for blk in system]
        kwargs: dict[str, Any] = {}
        if self.caps.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.caps.effort:
            kwargs["output_config"] = {"effort": self._cfg.effort}
        with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=serialize(messages),
            **kwargs,
        ) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                if on_text is not None and event.delta.type == "text_delta":
                    on_text(event.delta.text)
                elif on_thinking is not None and event.delta.type == "thinking_delta":
                    on_thinking(event.delta.thinking)
            response = stream.get_final_message()
        return LLMResponse(
            content=_normalize_content(response.content),
            stop_reason=response.stop_reason or "end_turn",
            usage=serialize(response.usage) or {},
        )
