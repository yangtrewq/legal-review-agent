"""与 Anthropic SDK 响应交互的公共小工具。"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_text(response: Any, default: str = "") -> str:
    """提取响应中全部 text 块并拼接；无 text 块（refusal/纯 thinking）时返回 default。

    替代散落各处的 `next(b.text for b in response.content if b.type == "text")` ——
    后者在响应无文本块时抛 StopIteration（在生成器上下文中变成晦涩的 RuntimeError）。
    """
    texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(texts) if texts else default


def parse_json(text: str) -> Any:
    """宽松 JSON 解析：容忍代码围栏、前后缀文字以及坏前缀（本地模型常见输出形态）。

    例：GLM 在 json_object 模式下偶发输出 '{"' 伪起始符 + 真正对象——
    退化扫描会从每个 '{' 起点依次尝试，取第一个括号配对完整且可解析的对象。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # 退化：从每个 '{' 起点尝试提取可解析的完整 JSON 对象
    found_balanced = False
    for start, ch in enumerate(stripped):
        if ch != "{":
            continue
        candidate = _balanced_object(stripped, start)
        if candidate is None:
            continue
        found_balanced = True
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    if "{" not in stripped:
        raise ValueError(f"输出中不含 JSON 对象: {text[:200]!r}")
    if found_balanced:
        raise ValueError(f"JSON 对象无法解析: {text[:200]!r}")
    raise ValueError(f"JSON 对象不完整: {text[:200]!r}")


def _balanced_object(text: str, start: int) -> str | None:
    """从 start（必须是 '{'）提取括号配对完整的子串；未闭合返回 None。"""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def require_text(response: Any, stage: str) -> str:
    """提取文本，无文本块时抛出带上下文的明确错误（供必须拿到结构化输出的调用方使用）。"""
    text = extract_text(response)
    if not text:
        raise RuntimeError(
            f"[{stage}] 模型未返回文本内容（stop_reason={getattr(response, 'stop_reason', '?')}），无法继续"
        )
    return text
