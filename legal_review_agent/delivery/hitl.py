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
        answer = input("请输入处理意见（直接回车=approve；其他输入视为补充回答/修改意见）：").strip()
        if not answer:
            return CheckpointResolution(action="approve")
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
