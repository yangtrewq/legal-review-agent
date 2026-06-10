import json

import pytest

from legal_review_agent.registry.tool_registry import SkillSpec, ToolRegistry
from legal_review_agent.skills.builtin import register_builtin_skills


def test_builtin_skills_registered_with_valid_schemas():
    registry = register_builtin_skills(ToolRegistry())
    names = registry.list_names()
    assert "fetch_baseline" in names
    assert "identify_risk_points" in names
    for tool in registry.load_schemas():
        assert tool["input_schema"]["type"] == "object"
        assert tool["description"]


def test_jit_schema_loading_subset():
    registry = register_builtin_skills(ToolRegistry())
    tools = registry.load_schemas(["fetch_baseline"])
    assert [t["name"] for t in tools] == ["fetch_baseline"]


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    spec = SkillSpec(name="x", description="d", input_schema={"type": "object"}, handler=lambda: "")
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)


def test_skill_handler_runs():
    registry = register_builtin_skills(ToolRegistry())
    out = json.loads(registry.get_handler("fetch_baseline")(contract_type="采购合同"))
    assert out["contract_type"] == "采购合同"
