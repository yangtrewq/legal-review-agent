"""运行时事件流：引擎向前端推送的实时事件（SSE 数据源）。

事件类型约定（type 字段）：
- routing        路由决策结果
- plan           执行图谱（任务清单可视化数据源）
- task_status    子任务状态变更 {task_id, status: running|done}
- text_delta     模型流式文本增量 {text}
- tool_start     工具调用开始 {tool_name, arguments, tool_use_id}
- tool_end       工具调用结束 {tool_name, tool_use_id, is_error, preview}
- checkpoint     HITL 卡点挂起，等待前端处理 {checkpoint_id, kind, question, payload}
- checkpoint_resolved 卡点已处理 {checkpoint_id, action}
- final          最终产出全文 {text}
- error          运行异常 {message}
- done           运行结束（SSE 终止信号）
"""

from __future__ import annotations

import queue
import time
from typing import Any


class EventEmitter:
    """线程安全的事件队列。引擎线程 emit，SSE 协程消费。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        self._queue.put({"type": event_type, "at": time.time(), **(data or {})})

    def close(self) -> None:
        self._queue.put(None)

    def consume(self, poll_timeout_s: float = 0.5):
        """阻塞式迭代事件，收到 close 哨兵后结束。"""
        while True:
            try:
                item = self._queue.get(timeout=poll_timeout_s)
            except queue.Empty:
                yield {"type": "heartbeat"}
                continue
            if item is None:
                return
            yield item


class NoopEmitter(EventEmitter):
    """CLI / 测试场景默认使用。"""

    def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        pass

    def close(self) -> None:
        pass
