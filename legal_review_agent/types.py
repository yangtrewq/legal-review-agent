"""跨组件共享的数据契约（控制面与数据面之间的"标准数字指令"）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MacroIntent(str, Enum):
    """网关层宏观意图定性 —— 只做粗粒度分类，不做槽位提取。"""

    LEGAL_REVIEW = "legal_review"      # 常规法律审查 → 主链路
    PURE_RETRIEVAL = "pure_retrieval"  # 单一检索请求 → 旁路
    CHITCHAT = "chitchat"              # 闲聊 → 短路兜底
    OUT_OF_SCOPE = "out_of_scope"      # 超纲 → 短路兜底


@dataclass
class UserInput:
    """多模态输入的统一封装。documents 为 (filename, media_type, base64) 三元组。"""

    text: str
    documents: list[tuple[str, str, str]] = field(default_factory=list)
    session_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    intent: MacroIntent
    confidence: float
    reason: str = ""


@dataclass
class SubTask:
    """规划器输出的细粒度标准子任务（DAG 节点）。"""

    id: str
    name: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    suggested_skills: list[str] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """执行图谱（DAG）—— 交给认知引擎的"施工图纸"。"""

    goal: str
    tasks: list[SubTask]

    def topological_batches(self) -> list[list[SubTask]]:
        """按依赖关系输出可并发批次：同一批次内的子任务可并行执行。"""
        remaining = {t.id: t for t in self.tasks}
        done: set[str] = set()
        batches: list[list[SubTask]] = []
        while remaining:
            batch = [
                t for t in remaining.values()
                if all(dep in done for dep in t.depends_on)
            ]
            if not batch:
                raise ValueError(f"执行图谱存在环或悬空依赖: {sorted(remaining)}")
            for t in batch:
                done.add(t.id)
                del remaining[t.id]
            batches.append(batch)
        return batches


@dataclass
class ToolInstruction:
    """认知引擎下发给执行网关的标准数字指令。"""

    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolOutcome:
    """执行网关回传认知引擎的执行结果。"""

    tool_use_id: str
    content: str
    is_error: bool = False


class ToolExecutionError(Exception):
    """基建手段无法恢复时，网关向认知引擎抛出的明确错误，请求 AI 重新决策。"""

    def __init__(self, tool_name: str, message: str, retryable: bool = False):
        super().__init__(f"[{tool_name}] {message}")
        self.tool_name = tool_name
        self.retryable = retryable


@dataclass
class CheckpointOption:
    """卡点的备选项（前端以卡片展示，点击即回执）。"""

    id: str
    label: str
    description: str = ""
    recommended: bool = False
    # 点击该卡片时回执的 action："approve" / "reject" / "answer" 等
    action: str = "answer"

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "description": self.description,
                "recommended": self.recommended, "action": self.action}


@dataclass
class Checkpoint:
    """HITL 执行态卡点：等待法务人员 Review/修改/驳回的中间结论。"""

    checkpoint_id: str
    kind: str                 # 例如 "risk_confirmation" / "opinion_review" / "ask_user"
    payload: dict[str, Any]
    question: str = ""
    options: list[CheckpointOption] = field(default_factory=list)


class RunInterrupted(Exception):
    """Agent Loop 被用户中断。引擎捕获后持久化运行状态并挂起，等待恢复。"""
