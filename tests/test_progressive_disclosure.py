import json
from types import SimpleNamespace

import pytest

from legal_review_agent.config import DEFAULT_CONFIG, ResilienceConfig
from legal_review_agent.delivery.hitl import CheckpointResolution, HITLChannel
from legal_review_agent.engine.cognitive_engine import CognitiveEngine
from legal_review_agent.executor.action_gateway import ActionGateway
from legal_review_agent.memory.memory_manager import MemoryManager
from legal_review_agent.observability.tracing import Tracer
from legal_review_agent.registry.tool_registry import ToolRegistry
from legal_review_agent.skills.builtin import register_builtin_skills
from legal_review_agent.types import ExecutionPlan


# ---- 注册表：四层披露资产 ----

def test_docs_attached_to_builtin_skills():
    registry = register_builtin_skills(ToolRegistry())
    spec = registry.get("identify_risk_points")
    assert spec.instructions_path is not None
    assert "风险定级标准" in spec.load_instructions()
    assert "risk_checklist.md" in spec.list_resources()
    assert "违约责任" in spec.read_resource("risk_checklist.md")


def test_catalog_contains_l0_metadata():
    registry = register_builtin_skills(ToolRegistry())
    catalog = registry.catalog()
    assert "适用：审查任务开始时拉取比对基准" in catalog
    assert "有详细指南" in catalog
    assert "risk_checklist.md" in catalog


def test_read_resource_rejects_path_traversal():
    registry = register_builtin_skills(ToolRegistry())
    spec = registry.get("identify_risk_points")
    with pytest.raises(FileNotFoundError):
        spec.read_resource("../instructions.md")


def test_instructions_lazy_cached():
    registry = register_builtin_skills(ToolRegistry())
    spec = registry.get("generate_risk_point_opinions")
    first = spec.load_instructions()
    assert first is spec.load_instructions()  # 第二次命中缓存


# ---- 引擎：注入与元工具（不触发真实模型调用） ----

class AutoApprove(HITLChannel):
    def raise_checkpoint(self, checkpoint):
        return CheckpointResolution(action="approve")

    def deliver(self, title, content):
        pass


FAST_CFG = ResilienceConfig(max_retries=0, retry_base_delay_s=0, tool_timeout_s=5,
                            rate_limit_per_minute=10_000, circuit_failure_threshold=99,
                            circuit_recovery_time_s=1)


def make_engine(tmp_path, plan_skills=("fetch_baseline",)):
    registry = register_builtin_skills(ToolRegistry())
    gateway = ActionGateway(registry, FAST_CFG)
    memory = MemoryManager(str(tmp_path), "t")
    engine = CognitiveEngine(
        client=None, config=DEFAULT_CONFIG, registry=registry,
        gateway=gateway, assembler=None, memory=memory, hitl=AutoApprove(),
        tracer=Tracer("test"),
    )
    plan = ExecutionPlan(goal="g", tasks=[])
    engine._init_task_tracking(plan)
    engine._active_tools = registry.load_schemas(list(plan_skills))
    engine._injected_skills = set()
    return engine, registry


def block(name, **inputs):
    return SimpleNamespace(id="tu1", name=name, input=inputs)


def test_l2_injected_on_first_call_only(tmp_path):
    engine, _ = make_engine(tmp_path)
    r1 = engine._handle_tool_call(block("identify_risk_points",
                                        clauses_json="{}", baseline_json="{}"))
    assert "<skill_instructions" in r1["content"]
    assert "风险定级标准" in r1["content"]
    r2 = engine._handle_tool_call(block("identify_risk_points",
                                        clauses_json="{}", baseline_json="{}"))
    assert "<skill_instructions" not in r2["content"]


def test_skill_without_docs_injects_nothing(tmp_path):
    engine, _ = make_engine(tmp_path)
    r = engine._handle_tool_call(block("fetch_baseline", contract_type="采购合同"))
    assert "<skill_instructions" not in r["content"]
    assert json.loads(r["content"])["contract_type"] == "采购合同"


def test_load_skill_appends_tool_and_returns_instructions(tmp_path):
    engine, _ = make_engine(tmp_path, plan_skills=("fetch_baseline",))
    assert all(t["name"] != "identify_risk_points" for t in engine._active_tools)
    r = engine._handle_load_skill(block("load_skill", skill_name="identify_risk_points"))
    assert not r["is_error"]
    assert "已加载" in r["content"] and "风险定级标准" in r["content"]
    assert any(t["name"] == "identify_risk_points" for t in engine._active_tools)
    # 已随加载给过指南，后续首调不再重复注入
    rc = engine._handle_tool_call(block("identify_risk_points",
                                        clauses_json="{}", baseline_json="{}"))
    assert "<skill_instructions" not in rc["content"]


def test_load_skill_unknown_name(tmp_path):
    engine, _ = make_engine(tmp_path)
    r = engine._handle_load_skill(block("load_skill", skill_name="nope"))
    assert r["is_error"] and "可用技能" in r["content"]


def test_read_skill_resource(tmp_path):
    engine, _ = make_engine(tmp_path)
    r = engine._handle_read_resource(block("read_skill_resource",
                                           skill_name="identify_risk_points",
                                           resource_name="risk_checklist.md"))
    assert not r["is_error"] and "违约责任" in r["content"]
    bad = engine._handle_read_resource(block("read_skill_resource",
                                             skill_name="identify_risk_points",
                                             resource_name="../instructions.md"))
    assert bad["is_error"]


def test_disclosure_recorded_in_trace(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine._handle_tool_call(block("identify_risk_points",
                                   clauses_json="{}", baseline_json="{}"))
    spans = engine._tracer.to_dict()["spans"]
    disclosures = [s for s in spans if s["kind"] == "skill_disclosure"]
    assert len(disclosures) == 1
    assert disclosures[0]["payload"]["level"] == "L2"
    assert disclosures[0]["payload"]["trigger"] == "first_call"
    assert disclosures[0]["payload"]["est_tokens"] > 0
