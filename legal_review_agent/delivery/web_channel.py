"""Web 版 HITL 通道：卡点经事件流推送到前端，引擎线程阻塞等待前端回执。

流程：
1. 引擎线程调用 raise_checkpoint → emit("checkpoint") 推给前端 → 阻塞在响应队列上；
2. 前端弹出卡点面板，法务人员处理后 POST /api/checkpoints/{id}/resolve；
3. server 调用 resolve() 投递回执，引擎线程恢复执行。
"""

from __future__ import annotations

import queue
import threading

from ..observability.events import EventEmitter
from ..types import Checkpoint
from .hitl import CheckpointResolution, HITLChannel


class WebHITLChannel(HITLChannel):
    def __init__(self, emitter: EventEmitter, timeout_s: float = 1800.0):
        self._emitter = emitter
        self._timeout_s = timeout_s
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
        })
        try:
            resolution = response.get(timeout=self._timeout_s)
        except queue.Empty:
            resolution = CheckpointResolution(action="reject", comment="卡点等待超时，自动驳回")
        finally:
            with self._lock:
                self._pending.pop(checkpoint.checkpoint_id, None)
        self._emitter.emit("checkpoint_resolved", {
            "checkpoint_id": checkpoint.checkpoint_id, "action": resolution.action,
        })
        return resolution

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
        self._emitter.emit("deliver", {"title": title, "content": content})
