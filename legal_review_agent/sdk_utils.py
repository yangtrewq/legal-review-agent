"""与 Anthropic SDK 响应交互的公共小工具。"""

from __future__ import annotations

from typing import Any


def extract_text(response: Any, default: str = "") -> str:
    """提取响应中全部 text 块并拼接；无 text 块（refusal/纯 thinking）时返回 default。

    替代散落各处的 `next(b.text for b in response.content if b.type == "text")` ——
    后者在响应无文本块时抛 StopIteration（在生成器上下文中变成晦涩的 RuntimeError）。
    """
    texts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "\n".join(texts) if texts else default


def require_text(response: Any, stage: str) -> str:
    """提取文本，无文本块时抛出带上下文的明确错误（供必须拿到结构化输出的调用方使用）。"""
    text = extract_text(response)
    if not text:
        raise RuntimeError(
            f"[{stage}] 模型未返回文本内容（stop_reason={getattr(response, 'stop_reason', '?')}），无法继续"
        )
    return text
