"""OpenAI 兼容后端：对接本地部署的模型服务（GLM-V5 via vLLM / sglang 等）。

职责：在统一接口下完成两个方向的协议翻译——
- 出向：Anthropic 风格消息（tool_use/tool_result 块）→ OpenAI chat.completions
  （assistant.tool_calls / role=tool 消息）、工具 Schema input_schema → parameters；
- 入向：OpenAI 响应/流 → 归一化 LLMResponse（TextBlock/ToolUseBlock + stop_reason 映射）。

结构化输出：本地端点不支持 Anthropic 的 json_schema 约束，退化为
系统提示词注入 Schema（+ 可选 response_format=json_object，LRA_JSON_MODE=1），
配合调用方的宽松 JSON 解析（sdk_utils.parse_json）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from ..config import ModelConfig
from .backend import ChatBackend, LLMResponse, TextBlock, ToolUseBlock

logger = logging.getLogger(__name__)

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _bget(block: Any, key: str, default: Any = None) -> Any:
    """兼容归一化 pydantic 块与原始 dict 块的字段读取。"""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object"}),
        },
    } for t in tools]


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic 风格消息序列 → OpenAI 消息序列。"""
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue

        if role == "assistant":
            texts, tool_calls = [], []
            for block in content:
                btype = _bget(block, "type")
                if btype == "text":
                    texts.append(_bget(block, "text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": _bget(block, "id"),
                        "type": "function",
                        "function": {
                            "name": _bget(block, "name"),
                            "arguments": json.dumps(_bget(block, "input", {}), ensure_ascii=False),
                        },
                    })
                # thinking 块为 Anthropic 专有，对 OpenAI 端点不可见
            entry: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
            continue

        # user 消息：tool_result 块拆成 role=tool；text 块合并为普通 user 消息
        texts = []
        for block in content:
            btype = _bget(block, "type")
            if btype == "tool_result":
                result.append({
                    "role": "tool",
                    "tool_call_id": _bget(block, "tool_use_id"),
                    "content": str(_bget(block, "content", "")),
                })
            elif btype == "text":
                texts.append(_bget(block, "text", ""))
            elif btype == "document":
                logger.warning("openai_compatible 后端不支持 document 块，已替换为占位文本")
                texts.append(f"[附件文档 {_bget(block, 'title', '')}：请通过文本方式提供内容]")
        if texts:
            result.append({"role": role, "content": "\n".join(texts)})
    return result


def from_openai_message(message: Any, finish_reason: str | None) -> LLMResponse:
    content: list[Any] = []
    text = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
    if text:
        content.append(TextBlock(text=text))
    for call in (getattr(message, "tool_calls", None)
                 or (message.get("tool_calls") if isinstance(message, dict) else None) or []):
        fn = _bget(call, "function")
        try:
            arguments = json.loads(_bget(fn, "arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        content.append(ToolUseBlock(id=_bget(call, "id") or "", name=_bget(fn, "name") or "",
                                    input=arguments))
    return LLMResponse(
        content=content,
        stop_reason=_STOP_REASON_MAP.get(finish_reason or "stop", "end_turn"),
    )


class OpenAICompatBackend(ChatBackend):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai_compatible 后端需要 openai 包：pip install openai") from e
        if not config.base_url:
            raise ValueError("openai_compatible 后端必须配置 LRA_BASE_URL（如 http://localhost:8000/v1）")
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key or "EMPTY")

    def complete(self, *, model, system, messages, max_tokens,
                 json_schema=None, thinking=False) -> LLMResponse:  # noqa: ARG002 — thinking 由 caps 门控
        if json_schema is not None:
            system = (
                f"{system}\n\n# 输出格式要求\n"
                "只输出一个符合以下 JSON Schema 的 JSON 对象，不要输出任何其他文字、解释或代码围栏：\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
        kwargs: dict[str, Any] = {}
        if json_schema is not None and self.caps.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=to_openai_messages(system, messages),
            **kwargs,
        )
        choice = response.choices[0]
        result = from_openai_message(choice.message, choice.finish_reason)
        if getattr(response, "usage", None):
            result.usage = response.usage.model_dump()
        return result

    def engine_turn(self, *, model, system, messages, tools, max_tokens,
                    on_text: Callable[[str], None] | None = None) -> LLMResponse:
        system_text = "\n\n".join(b.get("text", "") for b in system)
        stream = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=to_openai_messages(system_text, messages),
            tools=to_openai_tools(tools),
            stream=True,
        )
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if delta is None:
                continue
            if delta.content:
                text_parts.append(delta.content)
                if on_text is not None:
                    on_text(delta.content)
            for tc in delta.tool_calls or []:
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

        content: list[Any] = []
        if text_parts:
            content.append(TextBlock(text="".join(text_parts)))
        for index in sorted(calls):
            slot = calls[index]
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            content.append(ToolUseBlock(id=slot["id"] or f"call_{index}",
                                        name=slot["name"], input=arguments))
        return LLMResponse(
            content=content,
            stop_reason=_STOP_REASON_MAP.get(finish_reason or "stop", "end_turn"),
        )
