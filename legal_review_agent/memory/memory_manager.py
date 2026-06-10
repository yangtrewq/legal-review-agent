"""2.4 记忆管理组件 (Memory Management)

职责边界：系统的"静态数据库"，解决"哪些信息最重要，需要提取并持久化"。

三层记忆：
- 长记忆/企业知识 (LongTermMemory)：文件持久化的条款基线库摘要、优秀范本、黑名单条款等记忆文件；
- 会话/项目记忆 (SessionMemory)：单次交易多轮博弈历史的关键状态，按 session_id 持久化，防"失忆式反复"；
- 工作区记忆 (WorkingMemory)：单次任务生命周期内的元数据与中间态结论，进程内存即可。

注意：相关性筛选（用低成本模型挑记忆）属于上下文组装组件的职责，本组件只负责存取。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# 进程内按文件路径加锁：防止同 session 并发运行时读改写互相覆盖；
# 跨进程并发需在部署层保证（同一 session 路由到同一实例）
_FILE_LOCKS: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _atomic_write(path: Path, text: str) -> None:
    """临时文件 + rename 原子落盘，避免写一半进程退出产生损坏文件。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class MemoryEntry:
    key: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class LongTermMemory:
    """企业级静态知识的记忆文件存储（追加写 JSONL，按 tag 粗筛）。"""

    def __init__(self, root: Path):
        self._path = root / "long_term.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, entry: MemoryEntry) -> None:
        with _FILE_LOCKS[str(self._path)]:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def scan(self, tags: list[str] | None = None) -> list[MemoryEntry]:
        if not self._path.exists():
            return []
        entries = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            e = MemoryEntry(**data)
            if tags is None or set(tags) & set(e.tags):
                entries.append(e)
        return entries


class SessionMemory:
    """单次交易的多轮博弈历史：关键状态信息按 session 持久化。"""

    def __init__(self, root: Path, session_id: str):
        self.session_id = session_id
        self._path = root / "sessions" / f"{session_id}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _FILE_LOCKS[str(self._path)]
        self._state: dict[str, Any] = (
            json.loads(self._path.read_text(encoding="utf-8"))
            if self._path.exists()
            else {"turns": [], "facts": {}}
        )

    def record_turn(self, role: str, summary: str) -> None:
        with self._lock:
            self._state["turns"].append({"role": role, "summary": summary, "at": time.time()})
            self._flush()

    def set_fact(self, key: str, value: Any) -> None:
        """提取并存储关键状态（如：对方已拒绝的条款、已达成一致的让步）。"""
        with self._lock:
            self._state["facts"][key] = value
            self._flush()

    def facts(self) -> dict[str, Any]:
        return dict(self._state["facts"])

    def recent_turns(self, n: int = 10) -> list[dict[str, Any]]:
        return self._state["turns"][-n:]

    def _flush(self) -> None:
        _atomic_write(self._path, json.dumps(self._state, ensure_ascii=False, indent=2))


class WorkingMemory:
    """单次任务生命周期内的暂存区：元数据、子任务中间产出，供后续节点动态调用。"""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._store)


class MemoryManager:
    def __init__(self, memory_root: str, session_id: str):
        root = Path(memory_root)
        self.long_term = LongTermMemory(root)
        self.session = SessionMemory(root, session_id)
        self.working = WorkingMemory()
