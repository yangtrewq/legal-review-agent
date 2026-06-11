"""Skill 体系契约测试（F2-5 ~ F2-9）：接口定义、Mock 返回、输出契约、可被调度。"""

import json

import pytest

from legal_review_agent.registry.tool_registry import SkillSpec, ToolRegistry
from legal_review_agent.skills.builtin import (
    IDENTIFY_RISK_PROMPT_TEMPLATE, register_builtin_skills,
)

ALL_SKILLS = ["GenerateOpinion", "GenerateSummary", "IdentifyRisk",
              "QueryRequirement", "SearchDocument"]


@pytest.fixture()
def registry():
    return register_builtin_skills(ToolRegistry())


# ---- 通用验收：接口定义完整 + 可被 Planner 调度 ----

def test_all_skills_registered_with_valid_schemas(registry):
    assert registry.list_names() == ALL_SKILLS
    for tool in registry.load_schemas():
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]
        assert "required" in tool["input_schema"]


def test_catalog_exposes_all_skills_for_planner(registry):
    catalog = registry.catalog()
    for name in ALL_SKILLS:
        assert name in catalog  # Planner 能力发现目录（可被正确调度的前提）


def test_checkpoint_gating(registry):
    assert registry.get("IdentifyRisk").requires_checkpoint
    assert registry.get("GenerateOpinion").requires_checkpoint
    assert not registry.get("QueryRequirement").requires_checkpoint
    assert not registry.get("SearchDocument").requires_checkpoint


# ---- F2-5 QueryRequirement ----

def test_query_requirement_contract(registry):
    spec = registry.get("QueryRequirement")
    assert spec.input_schema["required"] == ["user_id", "requirement_id"]

    out = json.loads(spec.handler(user_id="U001", requirement_id="REQ_001"))
    meta = out["contract_metadata"]
    assert set(meta) == {"contract_id", "contract_name", "contract_type",
                         "party_a", "party_b", "sign_date", "amount"}
    assert isinstance(meta["amount"], float)
    assert isinstance(out["business_background"], str)
    assert {"file_id", "file_name", "file_type"} == set(out["attachments"][0])


# ---- F2-6 IdentifyRisk ----

def test_identify_risk_contract(registry):
    spec = registry.get("IdentifyRisk")
    assert spec.input_schema["required"] == ["contract_document"]
    assert "risk_checklist" in spec.input_schema["properties"]

    out = json.loads(spec.handler(contract_document="合同全文…"))
    assert set(out) == {"risk_points", "total_count", "high_risk_count",
                        "medium_risk_count", "low_risk_count"}
    assert out["total_count"] == len(out["risk_points"]) == 3
    assert (out["high_risk_count"] + out["medium_risk_count"]
            + out["low_risk_count"]) == out["total_count"]
    rp = out["risk_points"][0]
    assert set(rp) == {"risk_id", "clause_location", "risk_description",
                       "risk_level", "risk_category", "original_text"}
    assert rp["risk_level"] in ("高", "中", "低")


def test_identify_risk_prompt_template_complete():
    for placeholder in ("{contract_document}", "{risk_checklist}"):
        assert placeholder in IDENTIFY_RISK_PROMPT_TEMPLATE
    assert "risk_level" in IDENTIFY_RISK_PROMPT_TEMPLATE


def test_identify_risk_progressive_docs(registry):
    spec = registry.get("IdentifyRisk")
    assert "Prompt 模板" in spec.load_instructions()
    assert "risk_checklist.md" in spec.list_resources()
    assert "预付款比例" in spec.read_resource("risk_checklist.md")


# ---- F2-7 GenerateOpinion ----

def test_generate_opinion_contract(registry):
    spec = registry.get("GenerateOpinion")
    assert spec.input_schema["required"] == ["risk_point"]

    risk_point = {"risk_id": "RISK_002", "original_text": "原条款文本"}
    out = json.loads(spec.handler(risk_point=risk_point))
    assert set(out) == {"opinion_id", "risk_id", "suggestion_type", "original_clause",
                        "suggested_clause", "reasoning", "legal_basis", "compromise_option"}
    assert out["risk_id"] == "RISK_002"           # 与入参风险点关联
    assert out["original_clause"] == "原条款文本"
    assert out["suggestion_type"] in ("修改", "删除", "新增", "妥协")


# ---- F2-8 GenerateSummary ----

def test_generate_summary_contract(registry):
    spec = registry.get("GenerateSummary")
    assert spec.input_schema["required"] == ["risk_points", "opinions", "contract_metadata"]

    risk_points = [{"risk_description": f"风险{i}"} for i in range(2)]
    out = json.loads(spec.handler(
        risk_points=risk_points, opinions=[{}],
        contract_metadata={"contract_name": "测试合同"},
    ))
    assert set(out) == {"summary_id", "overall_assessment", "risk_summary",
                        "key_risks", "recommendations", "review_conclusion"}
    assert out["risk_summary"] == {"total_count": 2, "resolved_count": 0, "pending_count": 2}
    assert out["key_risks"] == ["风险0", "风险1"]
    # 结论文档格式规范：Markdown 三段式
    for heading in ("# 合同评审结论", "## 整体评估", "## 重点风险", "## 综合建议"):
        assert heading in out["review_conclusion"]
    assert "测试合同" in out["review_conclusion"]


# ---- F2-9 SearchDocument ----

def test_search_document_contract(registry):
    spec = registry.get("SearchDocument")
    assert spec.input_schema["required"] == ["query"]
    assert spec.input_schema["properties"]["document_type"]["enum"] == ["案例", "法规", "模板"]

    out = json.loads(spec.handler(query="预付款比例", document_type="案例", top_k=1))
    assert set(out) == {"results", "total_hits"}
    assert len(out["results"]) == 1               # top_k 生效
    r = out["results"][0]
    assert set(r) == {"doc_id", "doc_title", "doc_type", "relevance_score",
                      "highlight", "source_url"}
    assert 0 <= r["relevance_score"] <= 1


# ---- 注册表通用行为 ----

def test_jit_schema_loading_subset(registry):
    tools = registry.load_schemas(["IdentifyRisk"])
    assert [t["name"] for t in tools] == ["IdentifyRisk"]


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    spec = SkillSpec(name="x", description="d", input_schema={"type": "object"}, handler=lambda: "")
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)
