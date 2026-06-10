"""法律评审 Agent 总装与生命周期编排。

核心逻辑流：
多模态输入 → [意图路由] → [任务规划] → 按需检视[技能/工具注册]
→ [上下文组装](融合[记忆]) → [Agentic Loop 认知调度] → [工具执行网关]
→ [结果反馈与交付 (HITL)]
"""

from __future__ import annotations

import logging

import anthropic

from .config import DEFAULT_CONFIG, Config
from .context.context_assembler import ContextAssembler
from .delivery.hitl import ConsoleChannel, HITLChannel
from .engine.cognitive_engine import CognitiveEngine
from .executor.action_gateway import ActionGateway
from .gateway.intent_router import IntentRouter
from .memory.memory_manager import MemoryManager
from .observability.events import EventEmitter, NoopEmitter
from .observability.tracing import NoopTracer, Tracer
from .planner.task_planner import TaskPlanner
from .registry.tool_registry import ToolRegistry
from .skills.builtin import register_builtin_skills
from .types import MacroIntent, ToolInstruction, UserInput

logger = logging.getLogger(__name__)


class LegalReviewAgent:
    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        hitl: HITLChannel | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self._cfg = config
        self._client = client or anthropic.Anthropic()
        self._hitl = hitl or ConsoleChannel()
        self._registry = register_builtin_skills(ToolRegistry())
        self._router = IntentRouter(self._client, config)
        self._planner = TaskPlanner(self._client, config, self._registry)
        self._gateway = ActionGateway(self._registry, config.resilience)

    def handle(
        self,
        user_input: UserInput,
        emitter: EventEmitter | None = None,
        tracer: Tracer | None = None,
        hitl: HITLChannel | None = None,
    ) -> str:
        emitter = emitter or NoopEmitter()
        tracer = tracer or NoopTracer()
        hitl = hitl or self._hitl

        # 1. 意图路由：宏观定性 + 边界裁决
        decision = self._router.route(user_input, tracer=tracer)
        logger.info("路由结果: %s (%.2f) %s", decision.intent, decision.confidence, decision.reason)
        tracer.record("routing", decision.intent.value,
                      confidence=decision.confidence, reason=decision.reason)
        emitter.emit("routing", {
            "intent": decision.intent.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
        })

        # 短路链路：闲聊 / 超纲 → 兜底回复
        if decision.intent in (MacroIntent.CHITCHAT, MacroIntent.OUT_OF_SCOPE):
            reply = IntentRouter.fallback_reply(decision)
            emitter.emit("final", {"text": reply})
            hitl.deliver("回复", reply)
            return reply

        # 旁路链路：单一检索 → 直接调用简单工具，跳过复杂规划
        if decision.intent == MacroIntent.PURE_RETRIEVAL:
            instruction = ToolInstruction(
                tool_use_id="bypass-retrieval",
                tool_name="search_legal_knowledge",
                arguments={"query": user_input.text},
            )
            span = tracer.start_span("tool_call", instruction.tool_name,
                                     instruction=instruction.__dict__)
            outcome = self._gateway.execute_safe(instruction)
            tracer.end_span(span, is_error=outcome.is_error, result=outcome.content)
            emitter.emit("final", {"text": outcome.content})
            hitl.deliver("检索结果", outcome.content)
            return outcome.content

        # 主链路：规划 → 组装 → 认知循环 → 交付
        memory = MemoryManager(self._cfg.memory_root, user_input.session_id)
        assembler = ContextAssembler(self._client, self._cfg, memory)
        engine = CognitiveEngine(
            self._client, self._cfg, self._registry,
            self._gateway, assembler, memory, hitl,
            emitter=emitter, tracer=tracer,
        )

        plan = self._planner.plan(user_input, tracer=tracer)
        logger.info("执行图谱: %s, %d 个子任务", plan.goal, len(plan.tasks))
        tracer.record("planning", plan.goal, tasks=[t.__dict__ for t in plan.tasks])
        emitter.emit("plan", {
            "goal": plan.goal,
            "tasks": [t.__dict__ for t in plan.tasks],
        })

        report = engine.run(user_input, plan)
        emitter.emit("final", {"text": report})
        hitl.deliver("审查报告", report)
        return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    agent = LegalReviewAgent()
    print("法律评审 Agent（输入 exit 退出）")
    while True:
        text = input("\n> ").strip()
        if not text or text.lower() == "exit":
            break
        agent.handle(UserInput(text=text))


if __name__ == "__main__":
    main()
