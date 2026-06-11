import threading
from types import SimpleNamespace

import pytest

from legal_review_agent.config import DEFAULT_CONFIG, ResilienceConfig
from legal_review_agent.delivery.hitl import CheckpointResolution
from legal_review_agent.delivery.web_channel import WebHITLChannel
from legal_review_agent.engine.cognitive_engine import CognitiveEngine
from legal_review_agent.engine.run_state import RunState, RunStateStore
from legal_review_agent.executor.action_gateway import ActionGateway
from legal_review_agent.memory.memory_manager import MemoryManager
from legal_review_agent.observability.events import EventEmitter
from legal_review_agent.observability.tracing import Tracer
from legal_review_agent.registry.tool_registry import ToolRegistry
from legal_review_agent.skills.builtin import register_builtin_skills
from legal_review_agent.types import (
    Checkpoint, CheckpointOption, ExecutionPlan, RunInterrupted, SubTask, UserInput,
)

FAST_CFG = ResilienceConfig(max_retries=0, retry_base_delay_s=0, tool_timeout_s=5,
                            rate_limit_per_minute=10_000, circuit_failure_threshold=99,
                            circuit_recovery_time_s=1)


# ---- RunState 持久化 ----

def test_run_state_roundtrip(tmp_path):
    store = RunStateStore(tmp_path)
    state = RunState(
        run_id="r1", session_id="s1", user_text="审查合同", goal="g",
        tasks=[{"id": "t1", "name": "n", "description": "", "depends_on": [],
                "suggested_skills": ["QueryRequirement"]}],
        messages=[{"role": "user", "content": "hi"}],
        active_tools=[{"name": "QueryRequirement"}],
        injected_skills=["IdentifyRisk"],
        iteration=3, status="suspended",
    )
    store.save(state)

    loaded = store.load("r1")
    assert loaded.iteration == 3
    assert loaded.status == "suspended"
    assert loaded.plan().tasks[0].id == "t1"
    assert store.list_suspended()[0]["run_id"] == "r1"

    state.status = "completed"
    store.save(state)
    assert store.list_suspended() == []
    store.delete("r1")
    assert store.load("r1") is None


# ---- 卡点备选项 ----

def test_checkpoint_event_carries_option_cards():
    em = EventEmitter()
    channel = WebHITLChannel(em, timeout_s=5)
    ckpt = Checkpoint(
        checkpoint_id="ck1", kind="review:identify_risk_points", payload={},
        options=[
            CheckpointOption(id="approve", label="批准", action="approve", recommended=True),
            CheckpointOption(id="reject", label="驳回", action="reject"),
        ],
    )
    th = threading.Thread(target=lambda: channel.raise_checkpoint(ckpt))
    th.start()
    for event in em.consume(poll_timeout_s=0.1):
        if event["type"] == "checkpoint":
            assert [o["label"] for o in event["options"]] == ["批准", "驳回"]
            assert event["options"][0]["recommended"] is True
            assert event["options"][0]["action"] == "approve"
            channel.resolve("ck1", CheckpointResolution(action="approve"))
            break
    th.join(timeout=5)


def test_ask_user_builds_options_from_model_input(tmp_path):
    engine, _ = _make_engine(tmp_path)

    class SelectFirst:
        def raise_checkpoint(self, checkpoint):
            assert len(checkpoint.options) == 2
            assert checkpoint.options[0].label == "采购合同"
            assert checkpoint.options[0].recommended
            opt = checkpoint.options[0]
            return CheckpointResolution(action=opt.action,
                                        payload={"answer": opt.label, "option_id": opt.id})

        def deliver(self, title, content):
            pass

    engine._hitl = SelectFirst()
    block = SimpleNamespace(id="tu1", name="ask_user", input={
        "question": "合同类型是什么？",
        "missing_field": "contract_type",
        "options": [
            {"label": "采购合同", "description": "货物采购", "recommended": True},
            {"label": "技术服务合同"},
        ],
    })
    result = engine._handle_ask_user(block)
    assert result["content"] == "采购合同"


# ---- 中断与恢复 ----

def _make_engine(tmp_path, interrupt_event=None, state_store=None):
    registry = register_builtin_skills(ToolRegistry())
    gateway = ActionGateway(registry, FAST_CFG)
    memory = MemoryManager(str(tmp_path), "t")
    engine = CognitiveEngine(
        llm=None, config=DEFAULT_CONFIG, registry=registry,
        gateway=gateway, assembler=None, memory=memory, hitl=None,
        tracer=Tracer("test"), emitter=EventEmitter(),
        state_store=state_store, interrupt_event=interrupt_event,
    )
    return engine, registry


def test_interrupt_during_checkpoint_wait_raises():
    interrupt = threading.Event()
    em = EventEmitter()
    channel = WebHITLChannel(em, timeout_s=30, interrupt_event=interrupt)
    ckpt = Checkpoint(checkpoint_id="ck1", kind="ask_user", payload={})

    result = {}

    def waiter():
        try:
            channel.raise_checkpoint(ckpt)
        except RunInterrupted:
            result["interrupted"] = True

    th = threading.Thread(target=waiter)
    th.start()
    interrupt.set()
    th.join(timeout=5)
    assert result.get("interrupted")


def test_execute_suspends_and_persists_before_model_call(tmp_path, monkeypatch):
    """中断信号已置位时，循环在调用模型前挂起并持久化现场（client 不会被触达）。"""
    interrupt = threading.Event()
    interrupt.set()
    store = RunStateStore(tmp_path)
    engine, _ = _make_engine(tmp_path, interrupt_event=interrupt, state_store=store)

    state = RunState(
        run_id="r-int", session_id="t", user_text="u", goal="g",
        tasks=[], messages=[{"role": "user", "content": "hi"}],
        active_tools=[], iteration=2,
    )
    assert engine._execute(state) is None
    assert not interrupt.is_set()  # 挂起后复位，供恢复后再次中断

    saved = store.load("r-int")
    assert saved.status == "suspended"
    assert saved.iteration == 2
    assert saved.messages == [{"role": "user", "content": "hi"}]

    # 引擎事件流中应有 interrupted 事件
    engine._emitter.close()
    types = [e["type"] for e in engine._emitter.consume()]
    assert "interrupted" in types


def test_resume_appends_supplement_and_completes(tmp_path):
    """恢复时补充指示进入上下文；模型直接收敛则状态清理。"""
    store = RunStateStore(tmp_path)
    engine, _ = _make_engine(tmp_path, state_store=store)

    final = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="审查完成")],
    )
    engine._call_model = lambda messages, tools: final
    engine._assembler = SimpleNamespace(maybe_compress=lambda m: m)
    engine._memory = MemoryManager(str(tmp_path), "t")

    state = RunState(
        run_id="r-res", session_id="t", user_text="u", goal="g",
        tasks=[], messages=[{"role": "user", "content": "hi"}],
        active_tools=[], iteration=1, status="suspended",
    )
    store.save(state)

    report = engine.resume(state, supplement="重点看违约条款")
    assert report == "审查完成"
    assert any("重点看违约条款" in str(m.get("content", "")) for m in state.messages)
    assert store.load("r-res") is None  # 完成后清理挂起状态
