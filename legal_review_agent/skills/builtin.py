"""内置业务技能：由法务 SOP 沉淀的高内聚功能块。

这里的 handler 是对接企业内部系统的占位实现（PoC）。接入真实系统时，
只需替换 handler 内部逻辑，Schema 契约保持不变 —— 上层组件零改动。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..registry.tool_registry import SkillSpec, ToolRegistry

# 渐进披露文档根目录：docs/<skill_name>/{instructions.md, resources/}
DOCS_ROOT = Path(__file__).resolve().parent / "docs"


def _fetch_baseline(contract_type: str, counterparty: str = "") -> str:
    # TODO: 接入企业《条款基线库》检索服务
    return json.dumps({
        "contract_type": contract_type,
        "counterparty": counterparty,
        "baseline_clauses": [
            {"clause": "违约责任", "baseline": "违约金不超过合同总额的20%"},
            {"clause": "知识产权", "baseline": "交付成果知识产权归甲方所有"},
        ],
    }, ensure_ascii=False)


def _parse_clauses(document_text: str) -> str:
    # TODO: 接入条款解析服务（结构化抽取）
    return json.dumps({
        "clauses": [{"index": 1, "title": "示例条款", "text": document_text[:200]}],
    }, ensure_ascii=False)


def _identify_risk_points(clauses_json: str, baseline_json: str) -> str:
    # TODO: 接入风控比对引擎 / 黑名单条款库
    return json.dumps({
        "risk_points": [
            {"clause": "违约责任", "level": "high", "issue": "违约金上限缺失，偏离基线"},
        ],
    }, ensure_ascii=False)


def _generate_risk_point_opinions(risk_points_json: str, tone: str = "standard") -> str:
    # TODO: 可结合历史优秀审查范本生成修改意见
    risks = json.loads(risk_points_json).get("risk_points", [])
    opinions = [
        {"clause": r["clause"], "opinion": f"建议修订：{r['issue']}，参照基线条款补充上限约定。"}
        for r in risks
    ]
    return json.dumps({"opinions": opinions, "tone": tone}, ensure_ascii=False)


def _search_legal_knowledge(query: str, top_k: int = 5) -> str:
    # TODO: 接入法律知识库 / 法规检索
    return json.dumps({"query": query, "results": [], "top_k": top_k}, ensure_ascii=False)


def register_builtin_skills(registry: ToolRegistry) -> ToolRegistry:
    registry.register(SkillSpec(
        name="fetch_baseline",
        when_to_use="审查任务开始时拉取比对基准",
        description="拉取企业《条款基线库》中指定合同类型的历史基线条款。审查任务的第一步通常需要调用本技能获取比对基准。",
        input_schema={
            "type": "object",
            "properties": {
                "contract_type": {"type": "string", "description": "合同类型，如：采购合同、保密协议、技术服务合同"},
                "counterparty": {"type": "string", "description": "交易相对方名称（可选，用于拉取历史博弈记录）"},
            },
            "required": ["contract_type"],
        },
        handler=_fetch_baseline,
        tags=("review", "baseline"),
    ))
    registry.register(SkillSpec(
        name="parse_clauses",
        when_to_use="拿到合同原文后做逐条结构化",
        description="对合同文本做结构化条款解析，输出条款列表。当上下文中已有合同原文且需要逐条分析时调用。",
        input_schema={
            "type": "object",
            "properties": {
                "document_text": {"type": "string", "description": "合同全文文本"},
            },
            "required": ["document_text"],
        },
        handler=_parse_clauses,
        tags=("review", "parse"),
    ))
    registry.register(SkillSpec(
        name="identify_risk_points",
        when_to_use="条款与基线齐备后做风险比对",
        description="将解析后的条款与基线条款/黑名单条款做风控比对，识别风险点。必须先有 parse_clauses 和 fetch_baseline 的结果。",
        input_schema={
            "type": "object",
            "properties": {
                "clauses_json": {"type": "string", "description": "parse_clauses 输出的条款 JSON"},
                "baseline_json": {"type": "string", "description": "fetch_baseline 输出的基线 JSON"},
            },
            "required": ["clauses_json", "baseline_json"],
        },
        handler=_identify_risk_points,
        tags=("review", "risk"),
        requires_checkpoint=True,  # 确诊风险需法务人员卡点确认
    ))
    registry.register(SkillSpec(
        name="generate_risk_point_opinions",
        when_to_use="风险点经人工确认后生成修改意见",
        description="针对已确认的风险点生成修改意见。调用前风险点应已通过 HITL 确认。",
        input_schema={
            "type": "object",
            "properties": {
                "risk_points_json": {"type": "string", "description": "identify_risk_points 输出且经确认的风险点 JSON"},
                "tone": {"type": "string", "enum": ["standard", "firm", "conciliatory"], "description": "意见措辞风格"},
            },
            "required": ["risk_points_json"],
        },
        handler=_generate_risk_point_opinions,
        tags=("review", "opinion"),
        requires_checkpoint=True,
    ))
    registry.register(SkillSpec(
        name="search_legal_knowledge",
        when_to_use="需要法规依据或单一知识检索时",
        description="检索法律知识库与法规条文。用于单一检索请求旁路，或审查过程中的法规依据补充。",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题"},
                "top_k": {"type": "integer", "description": "返回条数，默认5"},
            },
            "required": ["query"],
        },
        handler=_search_legal_knowledge,
        tags=("retrieval",),
    ))
    registry.attach_docs(DOCS_ROOT)
    return registry
