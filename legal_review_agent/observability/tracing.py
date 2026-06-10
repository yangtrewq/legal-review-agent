"""链路观测 (Tracing)：记录一次运行的完整执行链路，供调试审阅。

记录内容：
- model_request：每次调用大模型时**完整组装后的提示词**（system / messages / tools）、
  stop_reason、token 用量 —— 用于审阅上下文组装组件的最终产物；
- tool_call：认知引擎下发的标准数字指令、网关回传结果、耗时；
- routing / planning / checkpoint：路由决策、执行图谱、HITL 卡点处理。

Trace 按 run_id 存于内存 TraceStore（环形淘汰），由 server 暴露查询 API。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def serialize(obj: Any) -> Any:
    """把含 SDK Pydantic 对象的消息结构转为可 JSON 化的纯数据。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


@dataclass
class Span:
    span_id: str
    kind: str                  # model_request / tool_call / routing / planning / checkpoint
    name: str
    started_at: float
    ended_at: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "kind": self.kind,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "payload": self.payload,
        }


class Tracer:
    """单次运行的链路记录器，线程安全。"""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.created_at = time.time()
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def start_span(self, kind: str, name: str, **payload: Any) -> Span:
        span = Span(
            span_id=uuid.uuid4().hex[:8],
            kind=kind,
            name=name,
            started_at=time.time(),
            payload=serialize(payload),
        )
        with self._lock:
            self._spans.append(span)
        return span

    def end_span(self, span: Span, **payload: Any) -> None:
        span.ended_at = time.time()
        span.payload.update(serialize(payload))

    def record(self, kind: str, name: str, **payload: Any) -> Span:
        """记录瞬时事件（无持续时间）。"""
        span = self.start_span(kind, name, **payload)
        span.ended_at = span.started_at
        return span

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "spans": [s.to_dict() for s in self._spans],
            }


class NoopTracer(Tracer):
    """CLI / 测试场景默认使用，不保留数据。"""

    def start_span(self, kind: str, name: str, **payload: Any) -> Span:  # noqa: ARG002
        return Span(span_id="noop", kind=kind, name=name, started_at=time.time())

    def end_span(self, span: Span, **payload: Any) -> None:
        span.ended_at = time.time()

    def record(self, kind: str, name: str, **payload: Any) -> Span:
        return self.start_span(kind, name)


class TraceStore:
    """按 run_id 保存最近 N 条 Trace（内存环形淘汰）。"""

    def __init__(self, max_traces: int = 200):
        self._max = max_traces
        self._traces: OrderedDict[str, Tracer] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, tracer: Tracer) -> None:
        with self._lock:
            self._traces[tracer.run_id] = tracer
            while len(self._traces) > self._max:
                self._traces.popitem(last=False)

    def get(self, run_id: str) -> Tracer | None:
        with self._lock:
            return self._traces.get(run_id)

    def list_summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"run_id": t.run_id, "created_at": t.created_at,
                 "span_count": len(t.to_dict()["spans"])}
                for t in reversed(self._traces.values())
            ]
