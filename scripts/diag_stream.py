"""流式输出诊断脚本：直接探测模型端点是否真正流式返回。

用途：前端看不到流式效果时，区分问题出在哪一段——
  端点不流式 / 思考阶段无 content 增量 / SSE 或前端问题。

用法（读取 .env 配置）：
    python scripts/diag_stream.py
    python scripts/diag_stream.py "自定义测试问题"
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from legal_review_agent.config import DEFAULT_CONFIG          # noqa: E402（先加载 .env）
from legal_review_agent.llm.backend import build_backend       # noqa: E402


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "用三句话介绍合同审查的基本流程。"
    cfg = DEFAULT_CONFIG.models
    print(f"Provider : {cfg.provider}")
    print(f"Base URL : {cfg.base_url or '(官方 API)'}")
    print(f"Model    : {cfg.engine_model}")
    print(f"问题     : {question}")
    print("-" * 60)

    backend = build_backend(cfg)
    started = time.monotonic()
    events: list[tuple[float, str, int]] = []   # (相对时刻, 类型, 字符数)

    def on_text(text: str) -> None:
        events.append((time.monotonic() - started, "text", len(text)))
        print(text, end="", flush=True)

    def on_thinking(text: str) -> None:
        events.append((time.monotonic() - started, "thinking", len(text)))
        print(".", end="", flush=True)   # 思考增量以点显示

    response = backend.engine_turn(
        model=cfg.engine_model,
        max_tokens=1024,
        system=[{"type": "text", "text": "你是简洁的助手。"}],
        messages=[{"role": "user", "content": question}],
        tools=[],
        on_text=on_text,
        on_thinking=on_thinking,
    )
    total = time.monotonic() - started

    print("\n" + "-" * 60)
    text_events = [e for e in events if e[1] == "text"]
    think_events = [e for e in events if e[1] == "thinking"]
    stats = response.usage.get("stream_stats", {})
    print(f"总耗时           : {total:.2f}s")
    print(f"text 增量次数    : {len(text_events)}（{sum(e[2] for e in text_events)} 字符）")
    print(f"thinking 增量次数: {len(think_events)}（{sum(e[2] for e in think_events)} 字符）")
    if stats:
        print(f"stream_stats     : {stats}")
    print("-" * 60)

    if len(text_events) <= 1 and not think_events:
        print("【诊断】端点几乎一次性返回 —— 大概率是模型服务/网关没有真正流式")
        print("        （检查 vLLM 启动参数、反向代理是否关闭 buffering：proxy_buffering off）")
    elif think_events and text_events:
        first_text_at = text_events[0][0]
        print(f"【诊断】流式正常。模型先思考 {first_text_at:.1f}s（{len(think_events)} 次增量）后才输出正文 ——")
        print("        前端在此期间会显示『🧠 模型思考中…』指示器（升级前会表现为长时间无输出）")
    elif len(text_events) > 1:
        print("【诊断】端点流式正常。若前端仍无效果，检查浏览器 Network 面板中")
        print("        /api/runs/{id}/events 是否持续收到 text_delta 事件")


if __name__ == "__main__":
    main()
