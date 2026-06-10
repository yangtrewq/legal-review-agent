"""2.8 结果反馈与产物交付 (Result Feedback & HITL)

职责边界：系统的"UI 与打印机"。负责与人类的交互确认和最终产物的呈现。

本项目中前端交互统一由 Eureka X（企业级入口 Agent）提供，因此这里定义
标准化对接协议（HITLChannel），由 EurekaXAdapter 实现到企业入口的映射；
本地开发/测试用 ConsoleChannel。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..types import Checkpoint


@dataclass
class CheckpointResolution:
    """法务人员对卡点的处理结果。"""

    action: str                      # "approve" / "modify" / "reject" / "answer"
    payload: dict[str, Any] | None = None  # modify 时为修改后的内容；answer 时为补充参数
    comment: str = ""


class HITLChannel(ABC):
    """与人类交互的标准化通道。Eureka X / 控制台 / 测试桩均实现此接口。"""

    @abstractmethod
    def raise_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResolution:
        """挂起执行流，等待人工处理卡点（Review/修改/驳回，或追问补参）。"""

    @abstractmethod
    def deliver(self, title: str, content: str) -> None:
        """交付最终产物（审查报告/修改意见）。"""


class ConsoleChannel(HITLChannel):
    """本地开发用：在终端中模拟 Eureka X 的卡点交互。"""

    def raise_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResolution:
        print(f"\n=== HITL 卡点 [{checkpoint.kind}] {checkpoint.checkpoint_id} ===")
        if checkpoint.question:
            print(f"问题：{checkpoint.question}")
        print(f"内容：{checkpoint.payload}")
        for i, opt in enumerate(checkpoint.options, start=1):
            mark = "（推荐）" if opt.recommended else ""
            print(f"  [{i}] {opt.label}{mark} — {opt.description}")
        answer = input("请输入序号选择备选项，或直接输入文字回答（回车=approve）：").strip()
        if not answer:
            return CheckpointResolution(action="approve")
        if answer.isdigit() and 1 <= int(answer) <= len(checkpoint.options):
            opt = checkpoint.options[int(answer) - 1]
            return CheckpointResolution(action=opt.action,
                                        payload={"answer": opt.label, "option_id": opt.id})
        return CheckpointResolution(action="answer", payload={"answer": answer})

    def deliver(self, title: str, content: str) -> None:
        print(f"\n===== {title} =====\n{content}\n")


class EurekaXAdapter(HITLChannel):
    """与 Eureka X 企业入口 Agent 的标准化对接。

    对接契约（待联调确认）：
    - raise_checkpoint → Eureka X 卡点工作台 API，阻塞等待法务人员处理回执
    - deliver          → Eureka X 消息/文档交付通道
    """

    def __init__(self, endpoint: str, token: str):
        self._endpoint = endpoint
        self._token = token

    def raise_checkpoint(self, checkpoint: Checkpoint) -> CheckpointResolution:
        raise NotImplementedError("待 Eureka X 卡点工作台 API 联调后实现")

    def deliver(self, title: str, content: str) -> None:
        raise NotImplementedError("待 Eureka X 交付通道 API 联调后实现")
