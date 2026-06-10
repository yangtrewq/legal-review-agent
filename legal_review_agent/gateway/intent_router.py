"""2.1 意图路由与感知分发网关 (Intent Routing & Perception Gateway)

职责边界：系统的"网关"。只做宏观路由与能力发现，不做槽位提取与参数对齐。
- 宏观意图定性：低成本模型 + 结构化输出，做粗粒度四分类；
- 边界裁决与路由分发：
    * legal_review     → 主链路（任务规划组件）
    * pure_retrieval   → 旁路（单一简单工具，跳过规划）
    * chitchat / out_of_scope → 短路（兜底回复）
"""

from __future__ import annotations

import json

import anthropic

from ..config import Config
from ..types import MacroIntent, RoutingDecision, UserInput

_ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [i.value for i in MacroIntent],
            "description": "宏观意图分类",
        },
        "confidence": {"type": "number", "description": "0~1 置信度"},
        "reason": {"type": "string", "description": "一句话分类依据"},
    },
    "required": ["intent", "confidence", "reason"],
    "additionalProperties": False,
}

_SYSTEM = """你是法律评审系统的意图路由网关。只做粗粒度宏观意图分类，不要提取任何参数。

分类标准：
- legal_review: 涉及合同/协议/条款的审查、风险识别、修改意见等有效审查任务（包括用户上传文档要求审查）
- pure_retrieval: 单纯查询法律知识、法规条文、判例，不涉及对具体文本的审查
- chitchat: 寒暄、闲聊、与法律事务无关的对话
- out_of_scope: 超出法律评审系统能力范围的请求（如要求出庭、代签合同等）"""


class IntentRouter:
    def __init__(self, client: anthropic.Anthropic, config: Config):
        self._client = client
        self._cfg = config

    def route(self, user_input: UserInput, tracer=None) -> RoutingDecision:
        has_docs = f"（用户附带了 {len(user_input.documents)} 份文档）" if user_input.documents else ""
        messages = [{"role": "user", "content": f"{user_input.text}{has_docs}"}]
        span = tracer.start_span(
            "model_request", self._cfg.models.router_model,
            stage="routing", system=_SYSTEM, messages=messages,
        ) if tracer else None
        response = self._client.messages.create(
            model=self._cfg.models.router_model,
            max_tokens=self._cfg.models.router_max_tokens,
            system=_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": _ROUTING_SCHEMA}},
            messages=messages,
        )
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        if tracer:
            tracer.end_span(span, decision=data, usage=response.usage)
        return RoutingDecision(
            intent=MacroIntent(data["intent"]),
            confidence=float(data["confidence"]),
            reason=data.get("reason", ""),
        )

    @staticmethod
    def fallback_reply(decision: RoutingDecision) -> str:
        """短路链路的兜底回复。"""
        if decision.intent == MacroIntent.CHITCHAT:
            return "您好，我是法律评审助手，可以帮您审查合同条款、识别风险点并生成修改意见。请上传合同文档或描述您的审查需求。"
        return "抱歉，该请求超出了法律评审助手的能力范围。我可以协助：合同条款审查、风险点识别、修改意见生成、法律知识检索。"
