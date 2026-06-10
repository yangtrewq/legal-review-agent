"""Web 版 HITL 通道：卡点经事件流推送到前端，引擎线程阻塞等待前端回执。

流程：
1. 引擎线程调用 raise_checkpoint → emit("checkpoint") 推给前端（含备选项卡片）→ 阻塞在响应队列上；
2. 前端弹出卡点面板（备选项以卡片展示，点击即回执），法务人员处理后
   POST /api/checkpoints/{id}/resolve；
3. server 调用 resolve() 投递回执，引擎线程恢复执行。

中断支持：等待回执期间以小步轮询同时监听 interrupt_event，
用户点击"暂停"时抛出 RunInterrupted，由引擎挂起并持久化现场。
"""

from __future__ import annotations

import queue
import threading

from ..observability.events import EventEmitter
from ..types import Checkpoint, RunInterrupted
from .hitl import CheckpointResolution, HITLChannel

_POLL_INTERVAL_S = 0.2


class WebHITLChannel(HITLChannel):
    def __init__(
        self,
        emitter: EventEmitter,
        timeout_s: float = 1800.0,
        interrupt_event: threading.Event | None = None,
    ):
        self._emitter = emitter
        self._timeout_s = timeout_s
        self._interrupt_event = interrupt_event
        self._pending: dict[str, queue.Queue[CheckpointResolution]] = {}
        self._lock = threading.Lock()

    def raise_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResolution:
        response: queue.Queue[CheckpointResolution] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[checkpoint.checkpoint_id] = response
        self._emitter.emit("checkpoint", {
            "checkpoint_id": checkpoint.checkpoint_id,
            "kind": checkpoint.kind,
            "question": checkpoint.question,
            "payload": checkpoint.payload,
            "options": [o.to_dict() for o in checkpoint.options],
        })
        try:
            resolution = self._wait(response, checkpoint.checkpoint_id)
        finally:
            with self._lock:
                self._pending.pop(checkpoint.checkpoint_id, None)
        self._emitter.emit("checkpoint_resolved", {
            "checkpoint_id": checkpoint.checkpoint_id, "action": resolution.action,
        })
        return resolution

    def _wait(self, response: queue.Queue, checkpoint_id: str) -> CheckpointResolution:
        """小步轮询等待回执，期间响应中断请求。"""
        waited = 0.0
        while waited < self._timeout_s:
            if self._interrupt_event is not None and self._interrupt_event.is_set():
                self._emitter.emit("checkpoint_resolved", {
                    "checkpoint_id": checkpoint_id, "action": "interrupted",
                })
                raise RunInterrupted
            try:
                return response.get(timeout=_POLL_INTERVAL_S)
            except queue.Empty:
                waited += _POLL_INTERVAL_S
        return CheckpointResolution(action="reject", comment="卡点等待超时，自动驳回")

    def resolve(self, checkpoint_id: str, resolution: CheckpointResolution) -> bool:
        with self._lock:
            response = self._pending.get(checkpoint_id)
        if response is None:
            return False
        response.put(resolution)
        return True

    def pending_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def deliver(self, title: str, content: str) -> None:
        """Web 场景的交付由编排层的 final 事件承担，此处为空实现避免重复事件。"""
