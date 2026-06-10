"""2.6 Agentic Loop 认知循环引擎 (Cognitive Engine)

职责边界：系统的"大脑控制面"。与底层网络调用完全解耦，只负责思考、
状态评估、动态槽位提取与决策下发。

- 状态机流转：Plan -> Execute -> Evaluate 循环。每轮向上下文组装组件取最新数据，
  拿着数据请求模型；模型产出 tool_use（JIT 槽位提取）则下发执行网关；
- JIT 动态发现：按当前子任务的 suggested_skills 从注册表按需加载 Schema；
- 主动追问 (Reverse Prompting)：内置 ask_user 工具，模型发现缺参时挂起执行流，
  经 HITL 通道向用户追问；
- HITL 卡点：requires_checkpoint 的技能执行成功后，结果先过人工卡点再回喂模型。
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import anthropic

from ..config import Config
from ..context.context_assembler import ContextAssembler
from ..delivery.hitl import HITLChannel
from ..memory.memory_manager import MemoryManager
from ..registry.tool_registry import ToolRegistry
from ..types import Checkpoint, ExecutionPlan, ToolInstruction, UserInput
from ..executor.action_gateway import ActionGateway

logger = logging.getLogger(__name__)

# 主动追问工具：模型缺槽位时的标准出口，由引擎拦截并走 HITL 通道，不进执行网关
ASK_USER_TOOL = {
    "name": "ask_user",
    "description": "当上下文中缺少完成当前子任务的必填信息（如合同类型、相对方名称、审查立场）时调用，向用户追问。一次只问一个明确的问题。",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "向用户提出的具体问题"},
            "missing_field": {"type": "string", "description": "缺失的参数名"},
        },
        "required": ["question"],
    },
}


class CognitiveEngine:
    def __init__(
        self,
        client: anthropic.Anthropic,
        config: Config,
        registry: ToolRegistry,
        gateway: ActionGateway,
        assembler: ContextAssembler,
        memory: MemoryManager,
        hitl: HITLChannel,
    ):
        self._client = client
        self._cfg = config
        self._registry = registry
        self._gateway = gateway
        self._assembler = assembler
        self._memory = memory
        self._hitl = hitl

    # ---- 对外入口 ----

    def run(self, user_input: UserInput, plan: ExecutionPlan) -> str:
        """按执行图谱驱动 Agentic Loop，返回最终审查产出文本。"""
        messages = self._bootstrap_messages(user_input, plan)
        tools = self._load_tools(plan)

        final_text = ""
        for iteration in range(self._cfg.max_loop_iterations):
            messages = self._assembler.maybe_compress(messages)
            response = self._call_model(messages, tools)

            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            if response.stop_reason != "tool_use":
                final_text = self._extract_text(response)
                break

            # Execute：下发标准数字指令到执行网关；ask_user 由引擎拦截走 HITL
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "ask_user":
                    tool_results.append(self._handle_ask_user(block))
                else:
                    tool_results.append(self._handle_tool_call(block))
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("达到最大迭代轮数 %d，强制收敛", self._cfg.max_loop_iterations)
            final_text = "审查未在限定步数内完成，请缩小审查范围后重试。"

        self._memory.session.record_turn("assistant", final_text[:500])
        return final_text

    # ---- 内部：上下文 / 工具装配 ----

    def _bootstrap_messages(self, user_input: UserInput, plan: ExecutionPlan) -> list[dict[str, Any]]:
        dynamic_ctx = self._assembler.build_dynamic_context(user_input.text)
        plan_text = json.dumps(
            {"goal": plan.goal, "tasks": [t.__dict__ for t in plan.tasks]},
            ensure_ascii=False, indent=2,
        )
        content: list[dict[str, Any]] = []
        for filename, media_type, data in user_input.documents:
            content.append({
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": data},
                "title": filename,
            })
        content.append({
            "type": "text",
            "text": (
                f"# 审查需求\n{user_input.text}\n\n"
                f"# 执行图谱（按依赖顺序完成全部子任务）\n{plan_text}\n\n"
                + (f"{dynamic_ctx}\n\n" if dynamic_ctx else "")
                + "完成全部子任务后，输出结构化的审查报告。"
            ),
        })
        self._memory.session.record_turn("user", user_input.text[:500])
        return [{"role": "user", "content": content}]

    def _load_tools(self, plan: ExecutionPlan) -> list[dict[str, Any]]:
        """JIT 加载：仅加载规划建议涉及的技能 Schema + ask_user。"""
        suggested = sorted({s for t in plan.tasks for s in t.suggested_skills
                            if s in self._registry.list_names()})
        names = suggested or None  # 规划未给建议时回退到全量（小工具集）
        return self._registry.load_schemas(names) + [ASK_USER_TOOL]

    def _call_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        with self._client.messages.stream(
            model=self._cfg.models.engine_model,
            max_tokens=self._cfg.models.engine_max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self._cfg.models.effort},
            system=self._assembler.build_system(),
            tools=tools,
            messages=messages,
        ) as stream:
            return stream.get_final_message()

    # ---- 内部：工具调用 / 追问 / 卡点 ----

    def _handle_tool_call(self, block: Any) -> dict[str, Any]:
        instruction = ToolInstruction(
            tool_use_id=block.id, tool_name=block.name, arguments=dict(block.input)
        )
        outcome = self._gateway.execute_safe(instruction)

        if not outcome.is_error:
            spec = self._registry.get(block.name)
            self._memory.working.put(f"result:{block.name}", outcome.content[:2000])
            if spec.requires_checkpoint:
                outcome = self._run_checkpoint(block.name, outcome)

        return {
            "type": "tool_result",
            "tool_use_id": outcome.tool_use_id,
            "content": outcome.content,
            "is_error": outcome.is_error,
        }

    def _run_checkpoint(self, tool_name: str, outcome):
        resolution = self._hitl.raise_checkpoint(Checkpoint(
            checkpoint_id=f"ckpt-{uuid.uuid4().hex[:8]}",
            kind=f"review:{tool_name}",
            payload={"tool": tool_name, "result": outcome.content},
        ))
        if resolution.action == "reject":
            outcome.content = f"[人工驳回] {resolution.comment or '法务人员驳回了该结论，请调整后重新执行。'}"
            outcome.is_error = True
        elif resolution.action == "modify" and resolution.payload:
            outcome.content = json.dumps(resolution.payload, ensure_ascii=False)
            self._memory.session.set_fact(f"hitl_modified:{tool_name}", resolution.payload)
        elif resolution.action == "answer" and resolution.payload:
            outcome.content += f"\n[法务人员补充] {resolution.payload.get('answer', '')}"
        return outcome

    def _handle_ask_user(self, block: Any) -> dict[str, Any]:
        question = block.input.get("question", "")
        resolution = self._hitl.raise_checkpoint(Checkpoint(
            checkpoint_id=f"ask-{uuid.uuid4().hex[:8]}",
            kind="ask_user",
            payload=dict(block.input),
            question=question,
        ))
        answer = (resolution.payload or {}).get("answer", "用户未提供补充信息")
        # 追问得到的关键参数沉淀到会话记忆，避免后续轮次"失忆式反复"
        field = block.input.get("missing_field") or question[:40]
        self._memory.session.set_fact(field, answer)
        return {"type": "tool_result", "tool_use_id": block.id, "content": answer}

    @staticmethod
    def _extract_text(response: Any) -> str:
        return "\n".join(b.text for b in response.content if b.type == "text")
