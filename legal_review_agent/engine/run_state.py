"""Agent Loop 运行状态的持久化：支撑中断 (interrupt) 与恢复 (resume)。

中断时引擎把循环现场（对话上下文、激活工具集、已注入技能、迭代位置）落盘；
恢复时按 run_id 取回现场继续执行，可附带用户补充指示。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..types import ExecutionPlan, SubTask


@dataclass
class RunState:
    run_id: str
    session_id: str
    user_text: str
    goal: str
    tasks: list[dict[str, Any]]            # SubTask 的序列化形态
    messages: list[Any] = field(default_factory=list)
    active_tools: list[dict[str, Any]] = field(default_factory=list)
    injected_skills: list[str] = field(default_factory=list)
    iteration: int = 0
    status: str = "running"                # running / suspended / completed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def plan(self) -> ExecutionPlan:
        return ExecutionPlan(goal=self.goal, tasks=[SubTask(**t) for t in self.tasks])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        return cls(**data)


class RunStateStore:
    """按 run_id 落盘 JSON。挂起态可跨进程恢复。"""

    def __init__(self, root: str | Path):
        self._dir = Path(root) / "runs"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self._dir / f"{run_id}.json"

    def save(self, state: RunState) -> None:
        state.updated_at = time.time()
        self._path(state.run_id).write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, run_id: str) -> RunState | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        return RunState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, run_id: str) -> None:
        self._path(run_id).unlink(missing_ok=True)

    def list_suspended(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self._dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") == "suspended":
                result.append({"run_id": data["run_id"], "goal": data["goal"],
                               "user_text": data["user_text"], "updated_at": data["updated_at"]})
        return result
