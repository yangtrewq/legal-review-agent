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
import threading
import uuid
from typing import Any

from ..config import Config
from ..context.context_assembler import ContextAssembler, estimate_tokens
from ..delivery.hitl import HITLChannel
from ..llm.backend import ChatBackend
from ..memory.memory_manager import MemoryManager
from ..observability.events import EventEmitter, NoopEmitter
from ..observability.tracing import NoopTracer, Tracer, serialize
from ..registry.tool_registry import ToolRegistry
from ..sdk_utils import extract_text
from ..types import (
    Checkpoint, CheckpointOption, ExecutionPlan, RunInterrupted,
    ToolInstruction, UserInput,
)
from ..executor.action_gateway import ActionGateway
from .run_state import RunState, RunStateStore

logger = logging.getLogger(__name__)

# 主动追问工具：模型缺槽位时的标准出口，由引擎拦截并走 HITL 通道，不进执行网关
ASK_USER_TOOL = {
    "name": "ask_user",
    "description": "当上下文中缺少完成当前子任务的必填信息（如合同类型、相对方名称、审查立场）时调用，向用户追问。一次只问一个明确的问题。尽量给出 2-4 个候选答案（options），用户可直接点选；开放性问题可不给。",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "向用户提出的具体问题"},
            "missing_field": {"type": "string", "description": "缺失的参数名"},
            "options": {
                "type": "array",
                "description": "建议的候选答案，前端以卡片展示供用户点选",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "候选答案（点选后即作为回答）"},
                        "description": {"type": "string", "description": "该选项的含义或影响，一句话"},
                        "recommended": {"type": "boolean", "description": "是否为推荐项，至多一个"},
                    },
                    "required": ["label"],
                },
            },
        },
        "required": ["question"],
    },
}

# 渐进披露元工具：模型自主发现并加载技能（L1+L2），由引擎拦截，不进执行网关
LOAD_SKILL_TOOL = {
    "name": "load_skill",
    "description": "当技能目录中存在适合当前子任务、但本轮尚未加载的技能时调用，按名称加载其完整定义与使用指南。加载后即可直接调用该技能。",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill_name": {"type": "string", "description": "技能目录中列出的技能名"},
        },
        "required": ["skill_name"],
    },
}

# 渐进披露元工具：读取技能的 L3 资源文件（范本、checklist 等）
READ_SKILL_RESOURCE_TOOL = {
    "name": "read_skill_resource",
    "description": "读取某技能附带的资源文件（技能目录或使用指南中列出的范本、核对清单等）。仅在确实需要其内容时调用。",
    "input_schema": {
        "type": "object",
        "properties": {
            "skill_name": {"type": "string", "description": "技能名"},
            "resource_name": {"type": "string", "description": "资源文件名，如 risk_checklist.md"},
        },
        "required": ["skill_name", "resource_name"],
    },
}



class CognitiveEngine:
    def __init__(
        self,
        llm: ChatBackend,
        config: Config,
        registry: ToolRegistry,
        gateway: ActionGateway,
        assembler: ContextAssembler,
        memory: MemoryManager,
        hitl: HITLChannel,
        emitter: EventEmitter | None = None,
        tracer: Tracer | None = None,
        state_store: RunStateStore | None = None,
        interrupt_event: threading.Event | None = None,
    ):
        self._llm = llm
        self._cfg = config
        self._registry = registry
        self._gateway = gateway
        self._assembler = assembler
        self._memory = memory
        self._hitl = hitl
        self._emitter = emitter or NoopEmitter()
        self._tracer = tracer or NoopTracer()
        self._state_store = state_store
        self._interrupt_event = interrupt_event or threading.Event()
        # 任务进度可视化：技能 → 子任务的映射与完成度（启发式：
        # 子任务 suggested_skills 全部成功执行过即视为完成）
        self._skill_to_tasks: dict[str, list[str]] = {}
        self._task_pending_skills: dict[str, set[str]] = {}

    # ---- 对外入口 ----

    def run(self, user_input: UserInput, plan: ExecutionPlan,
            run_id: str | None = None) -> str | None:
        """按执行图谱驱动 Agentic Loop。返回最终产出文本；被中断挂起时返回 None。"""
        state = RunState(
            run_id=run_id or uuid.uuid4().hex[:12],
            session_id=user_input.session_id,
            user_text=user_input.text,
            goal=plan.goal,
            tasks=[t.__dict__ for t in plan.tasks],
        )
        self._init_task_tracking(plan)
        state.messages = self._bootstrap_messages(user_input, plan)
        # 本轮激活的工具集：load_skill 元工具可在运行中追加（会重建一次前缀缓存）
        state.active_tools = self._load_tools(plan)
        return self._execute(state)

    def resume(self, state: RunState, supplement: str | None = None) -> str | None:
        """从挂起状态恢复 Agent Loop。supplement 为用户恢复时的补充指示。

        注意：任务进度可视化按持久化的计划重置重算；已注入的技能指南不会重复注入。
        """
        self._init_task_tracking(state.plan())
        if supplement:
            state.messages.append({
                "role": "user",
                "content": f"<resume_note>运行已恢复。用户补充指示：{supplement}</resume_note>",
            })
        state.status = "running"
        self._tracer.record("run_lifecycle", "resume",
                            run_id=state.run_id, iteration=state.iteration,
                            supplement=supplement or "")
        return self._execute(state)

    def request_interrupt(self) -> None:
        """请求中断：在下一个安全点（迭代边界 / 卡点等待中）挂起运行。"""
        self._interrupt_event.set()

    def _execute(self, state: RunState) -> str | None:
        self._active_tools = state.active_tools
        self._injected_skills: set[str] = set(state.injected_skills)
        messages = state.messages

        final_text = ""
        try:
            for iteration in range(state.iteration, self._cfg.max_loop_iterations):
                if self._interrupt_event.is_set():
                    raise RunInterrupted
                state.iteration = iteration
                messages = self._assembler.maybe_compress(messages)
                state.messages = messages
                response = self._call_model(messages, self._active_tools)

                if response.stop_reason == "pause_turn":
                    messages.append({"role": "assistant", "content": response.content})
                    continue

                if response.stop_reason != "tool_use":
                    final_text = self._extract_text(response)
                    break

                # Execute：下发标准数字指令到执行网关；元工具由引擎拦截
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                try:
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        handler = self._meta_handlers().get(block.name, self._handle_tool_call)
                        tool_results.append(handler(block))
                except RunInterrupted:
                    # 中断幂等处理：已执行工具的结果保留（副作用已发生，恢复后不应重做），
                    # 未执行的补"中断"错误回执，保证 tool_use/tool_result 配对完整
                    self._fill_interrupted_results(response.content, tool_results)
                    messages.append({"role": "user", "content": tool_results})
                    raise
                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning("达到最大迭代轮数 %d，强制收敛", self._cfg.max_loop_iterations)
                final_text = "审查未在限定步数内完成，请缩小审查范围后重试。"
        except RunInterrupted:
            self._suspend(state, messages)
            return None

        state.status = "completed"
        self._complete_all_tasks(state.tasks)
        if self._state_store is not None:
            self._state_store.delete(state.run_id)
        self._memory.session.record_turn("assistant", final_text[:500])
        return final_text

    def _meta_handlers(self) -> dict[str, Any]:
        """元工具分发表：新增元工具时只需在此注册一处。"""
        return {
            "ask_user": self._handle_ask_user,
            "load_skill": self._handle_load_skill,
            "read_skill_resource": self._handle_read_resource,
        }

    @staticmethod
    def _fill_interrupted_results(content: list[Any], tool_results: list[dict[str, Any]]) -> None:
        executed = {r["tool_use_id"] for r in tool_results}
        for block in content:
            if getattr(block, "type", None) == "tool_use" and block.id not in executed:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "[运行被用户中断] 该工具未执行；恢复运行后如仍需要请重新调用。",
                    "is_error": True,
                })

    def _suspend(self, state: RunState, messages: list[Any]) -> None:
        """持久化循环现场并挂起。"""
        self._interrupt_event.clear()
        state.messages = serialize(messages)
        state.active_tools = list(self._active_tools)
        state.injected_skills = sorted(self._injected_skills)
        state.status = "suspended"
        if self._state_store is not None:
            self._state_store.save(state)
        self._tracer.record("run_lifecycle", "interrupt",
                            run_id=state.run_id, iteration=state.iteration)
        self._emitter.emit("interrupted", {
            "run_id": state.run_id, "iteration": state.iteration,
            "resumable": self._state_store is not None,
        })
        logger.info("运行 %s 已在第 %d 轮挂起", state.run_id, state.iteration)

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
                f"# 技能目录（L0，渐进披露）\n{self._registry.catalog()}\n"
                "说明：目录中的技能若本轮未加载，可用 load_skill 按名称加载；"
                "标注了资源的技能可用 read_skill_resource 读取资源文件。\n\n"
                + (f"{dynamic_ctx}\n\n" if dynamic_ctx else "")
                + "完成全部子任务后，输出结构化的审查报告。"
            ),
        })
        self._memory.session.record_turn("user", user_input.text[:500])
        return [{"role": "user", "content": content}]

    def _load_tools(self, plan: ExecutionPlan) -> list[dict[str, Any]]:
        """JIT 加载：仅加载规划建议涉及的技能 Schema（L1）+ 引擎元工具。

        规划遗漏的技能不在此兜底 —— 模型可在运行中通过 load_skill 自主补载。
        """
        suggested = sorted({s for t in plan.tasks for s in t.suggested_skills
                            if s in self._registry.list_names()})
        names = suggested or None  # 规划未给建议时回退到全量（小工具集）
        return self._registry.load_schemas(names) + [
            ASK_USER_TOOL, LOAD_SKILL_TOOL, READ_SKILL_RESOURCE_TOOL,
        ]

    def _call_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):
        system = self._assembler.build_system()
        # 链路观测：记录本次完整组装后的提示词（system / messages / tools）
        span = self._tracer.start_span(
            "model_request", self._cfg.models.engine_model,
            system=system, messages=messages, tools=[t["name"] for t in tools],
            tool_schemas=tools,
        )
        response = self._llm.engine_turn(
            model=self._cfg.models.engine_model,
            max_tokens=self._cfg.models.engine_max_tokens,
            system=system,
            tools=tools,
            messages=messages,
            on_text=lambda text: self._emitter.emit("text_delta", {"text": text}),
        )
        self._tracer.end_span(
            span,
            stop_reason=response.stop_reason,
            usage=response.usage,
            response_content=response.content,
        )
        return response

    # ---- 内部：工具调用 / 追问 / 卡点 ----

    def _init_task_tracking(self, plan: ExecutionPlan) -> None:
        self._skill_to_tasks.clear()
        self._task_pending_skills.clear()
        for task in plan.tasks:
            if task.suggested_skills:
                self._task_pending_skills[task.id] = set(task.suggested_skills)
                for skill in task.suggested_skills:
                    self._skill_to_tasks.setdefault(skill, []).append(task.id)

    def _update_task_progress(self, skill: str, succeeded: bool) -> None:
        # running 事件已在工具下发前发出（_handle_tool_call），此处只负责 done 判定
        if not succeeded:
            return
        for task_id in self._skill_to_tasks.get(skill, []):
            pending = self._task_pending_skills.get(task_id)
            if pending is None:
                continue
            pending.discard(skill)
            if not pending:
                del self._task_pending_skills[task_id]
                self._emitter.emit("task_status", {"task_id": task_id, "status": "done"})

    def _complete_all_tasks(self, plan_tasks: list[dict[str, Any]]) -> None:
        """运行成功收敛时兜底：把仍未标记完成的子任务（含无 suggested_skills、
        启发式无法追踪的任务）统一标记 done，避免前端清单永久卡 pending。"""
        for task in plan_tasks:
            task_id = task.get("id")
            if task_id:
                self._emitter.emit("task_status", {"task_id": task_id, "status": "done"})
        self._task_pending_skills.clear()

    def _handle_tool_call(self, block: Any) -> dict[str, Any]:
        instruction = ToolInstruction(
            tool_use_id=block.id, tool_name=block.name, arguments=dict(block.input)
        )
        self._emitter.emit("tool_start", {
            "tool_use_id": block.id, "tool_name": block.name, "arguments": instruction.arguments,
        })
        for task_id in self._skill_to_tasks.get(block.name, []):
            if task_id in self._task_pending_skills:
                self._emitter.emit("task_status", {"task_id": task_id, "status": "running"})

        span = self._tracer.start_span("tool_call", block.name,
                                       instruction=instruction.__dict__)
        outcome = self._gateway.execute_safe(instruction)
        self._tracer.end_span(span, is_error=outcome.is_error, result=outcome.content)

        if not outcome.is_error:
            spec = self._registry.get(block.name)
            self._memory.working.put(f"result:{block.name}", outcome.content[:2000])
            if spec.requires_checkpoint:
                outcome = self._run_checkpoint(block.name, outcome)

        self._update_task_progress(block.name, succeeded=not outcome.is_error)
        self._emitter.emit("tool_end", {
            "tool_use_id": outcome.tool_use_id, "tool_name": block.name,
            "is_error": outcome.is_error, "preview": outcome.content[:300],
        })
        content = outcome.content + self._maybe_inject_instructions(block.name)
        return {
            "type": "tool_result",
            "tool_use_id": outcome.tool_use_id,
            "content": content,
            "is_error": outcome.is_error,
        }

    # ---- 内部：渐进披露 ----

    def _maybe_inject_instructions(self, skill_name: str) -> str:
        """L2 首调注入：技能在本轮第一次被调用时，把使用指南拼进 tool_result 回喂。

        正文走 messages 尾部而非 system，保持缓存前缀稳定；每技能每轮只注入一次。
        """
        if skill_name in self._injected_skills:
            return ""
        self._injected_skills.add(skill_name)
        try:
            instructions = self._registry.get(skill_name).load_instructions()
        except KeyError:
            return ""
        if not instructions:
            return ""
        self._disclose(skill_name, level="L2", trigger="first_call", text=instructions)
        return (
            f"\n\n<skill_instructions skill=\"{skill_name}\">\n"
            "（该技能本轮首次调用，以下是使用指南，后续调用与结果解读须遵循）\n"
            f"{instructions}\n</skill_instructions>"
        )

    def _handle_load_skill(self, block: Any) -> dict[str, Any]:
        """模型自主发现：加载技能 Schema（L1）入本轮工具集，并返回 L2 指南。"""
        skill_name = block.input.get("skill_name", "")
        try:
            spec = self._registry.get(skill_name)
        except KeyError:
            return self._meta_result(block, f"技能不存在：{skill_name}。可用技能：{', '.join(self._registry.list_names())}", is_error=True)

        appended = False
        if all(t["name"] != skill_name for t in self._active_tools):
            # 追加工具会击穿本轮前缀缓存，记入链路以便审视成本
            self._active_tools.append(spec.to_anthropic_tool())
            appended = True

        instructions = spec.load_instructions() or "（该技能无详细指南）"
        resources = spec.list_resources()
        content = (
            f"技能 {skill_name} 已加载，可直接调用。\n\n## 使用指南\n{instructions}"
            + (f"\n\n## 可读资源（read_skill_resource）\n{', '.join(resources)}" if resources else "")
        )
        self._disclose(skill_name, level="L1+L2", trigger="model_request",
                       text=content, cache_invalidated=appended)
        self._injected_skills.add(skill_name)  # 已随加载给过指南，首调不再重复注入
        return self._meta_result(block, content)

    def _handle_read_resource(self, block: Any) -> dict[str, Any]:
        """L3 资源读取。"""
        skill_name = block.input.get("skill_name", "")
        resource_name = block.input.get("resource_name", "")
        try:
            content = self._registry.get(skill_name).read_resource(resource_name)
        except (KeyError, FileNotFoundError) as e:
            return self._meta_result(block, str(e), is_error=True)
        self._disclose(skill_name, level="L3", trigger="model_request",
                       text=content, resource=resource_name)
        return self._meta_result(block, content)

    def _disclose(self, skill_name: str, level: str, trigger: str,
                  text: str, **extra: Any) -> None:
        est_tokens = estimate_tokens(text)
        self._tracer.record("skill_disclosure", skill_name,
                            level=level, trigger=trigger, est_tokens=est_tokens, **extra)
        self._emitter.emit("skill_disclosure", {
            "skill_name": skill_name, "level": level,
            "trigger": trigger, "est_tokens": est_tokens,
        })

    @staticmethod
    def _meta_result(block: Any, content: str, is_error: bool = False) -> dict[str, Any]:
        return {"type": "tool_result", "tool_use_id": block.id,
                "content": content, "is_error": is_error}

    def _run_checkpoint(self, tool_name: str, outcome):
        checkpoint = Checkpoint(
            checkpoint_id=f"ckpt-{uuid.uuid4().hex[:8]}",
            kind=f"review:{tool_name}",
            payload={"tool": tool_name, "result": outcome.content},
            options=[
                CheckpointOption(id="approve", label="批准", action="approve",
                                 description="确认该结论，继续执行后续子任务", recommended=True),
                CheckpointOption(id="reject", label="驳回", action="reject",
                                 description="驳回该结论，Agent 将调整思路后重做"),
            ],
        )
        span = self._tracer.start_span("checkpoint", checkpoint.kind,
                                       checkpoint_id=checkpoint.checkpoint_id,
                                       payload=checkpoint.payload)
        resolution = self._hitl.raise_checkpoint(checkpoint)
        self._tracer.end_span(span, action=resolution.action, comment=resolution.comment)
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
        options = [
            CheckpointOption(
                id=f"opt-{i}",
                label=o.get("label", ""),
                description=o.get("description", ""),
                recommended=bool(o.get("recommended", False)),
                action="answer",
            )
            for i, o in enumerate(block.input.get("options") or [])
            if o.get("label")
        ]
        checkpoint = Checkpoint(
            checkpoint_id=f"ask-{uuid.uuid4().hex[:8]}",
            kind="ask_user",
            payload=dict(block.input),
            question=question,
            options=options,
        )
        span = self._tracer.start_span("checkpoint", "ask_user",
                                       checkpoint_id=checkpoint.checkpoint_id,
                                       question=question)
        resolution = self._hitl.raise_checkpoint(checkpoint)
        self._tracer.end_span(span, action=resolution.action, payload=resolution.payload)
        answer = (resolution.payload or {}).get("answer", "用户未提供补充信息")
        # 追问得到的关键参数沉淀到会话记忆，避免后续轮次"失忆式反复"
        field = block.input.get("missing_field") or question[:40]
        self._memory.session.set_fact(field, answer)
        return {"type": "tool_result", "tool_use_id": block.id, "content": answer}

    @staticmethod
    def _extract_text(response: Any) -> str:
        return extract_text(response)
