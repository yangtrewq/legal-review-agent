"""内置业务 Skill（F2-5 ~ F2-9）：Mock 实现 + 完整接口契约。

| 功能ID | Skill            | 能力                               | 实现方式            |
|--------|------------------|------------------------------------|---------------------|
| F2-5   | QueryRequirement | 查询待评审需求（元数据/背景/附件） | Mock，契约完整      |
| F2-6   | IdentifyRisk     | 识别合同风险点（红线批注）         | Mock + Prompt 模板  |
| F2-7   | GenerateOpinion  | 出具风险点意见（修改/删除/新增/妥协）| Mock，输出格式定义 |
| F2-8   | GenerateSummary  | 出具综合评审意见（结构化结论）     | Mock，模板定义      |
| F2-9   | SearchDocument   | 检索相似案例（相关度排序）         | Mock，检索框架      |

接入真实系统时只替换各 handler 内部实现，契约（input_schema / 输出 JSON 结构）
保持不变 —— Planner / 引擎 / 前端零改动。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..registry.tool_registry import SkillSpec, ToolRegistry

# 渐进披露文档根目录：docs/<skill_name>/{instructions.md, resources/}
DOCS_ROOT = Path(__file__).resolve().parent / "docs"


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


# ---- F2-5 QueryRequirement ----

def _query_requirement(user_id: str, requirement_id: str) -> str:
    # TODO: 接入需求管理系统；当前按契约返回 Mock 数据
    return _dumps({
        "contract_metadata": {
            "contract_id": "CONTRACT_001",
            "contract_name": "2026年度IT设备采购合同",
            "contract_type": "采购合同",
            "party_a": "XX科技有限公司",
            "party_b": "YY供应商有限公司",
            "sign_date": "2026-05-01",
            "amount": 5000000.00,
        },
        "business_background": "年度IT设备采购框架协议，包含服务器、网络设备等",
        "attachments": [
            {"file_id": "FILE_001", "file_name": "合同正文.docx", "file_type": "docx"},
            {"file_id": "FILE_002", "file_name": "技术规格书.pdf", "file_type": "pdf"},
        ],
    })


# ---- F2-6 IdentifyRisk ----

# Prompt 模板（接入真实模型识别时使用；完整版见 docs/IdentifyRisk/instructions.md）
IDENTIFY_RISK_PROMPT_TEMPLATE = """你是企业法务风险识别专家。逐条扫描以下合同内容，基于风险检查清单识别风险点。

# 合同内容
{contract_document}

# 风险检查清单
{risk_checklist}

# 输出要求
对每个风险点输出：clause_location（条款位置）、risk_description（风险描述）、
risk_level（高/中/低）、risk_category（风险类别）、original_text（条款原文，逐字引用不得改写）。
只输出符合契约的 JSON，不要输出其他内容。"""


def _identify_risk(contract_document: str, risk_checklist: list | None = None) -> str:
    # TODO: 接入真实模型识别（使用 IDENTIFY_RISK_PROMPT_TEMPLATE）；当前返回 Mock
    return _dumps({
        "risk_points": [
            {
                "risk_id": "RISK_001",
                "clause_location": "第3条第2款",
                "risk_description": "付款条件对甲方不利，预付款比例过高",
                "risk_level": "高",
                "risk_category": "付款风险",
                "original_text": "合同签订后5个工作日内支付合同总额的50%作为预付款",
            },
            {
                "risk_id": "RISK_002",
                "clause_location": "第7条第1款",
                "risk_description": "违约责任不对等，仅约定甲方违约责任",
                "risk_level": "中",
                "risk_category": "违约风险",
                "original_text": "甲方逾期付款的，每日按未付款金额的千分之五支付违约金",
            },
            {
                "risk_id": "RISK_003",
                "clause_location": "第9条",
                "risk_description": "交付成果知识产权归属约定不明",
                "risk_level": "低",
                "risk_category": "知识产权风险",
                "original_text": "项目相关技术资料由双方共享",
            },
        ],
        "total_count": 3,
        "high_risk_count": 1,
        "medium_risk_count": 1,
        "low_risk_count": 1,
    })


# ---- F2-7 GenerateOpinion ----

def _generate_opinion(risk_point: dict, baseline_rules: list | None = None,
                      reference_cases: list | None = None) -> str:
    # TODO: 接入基线规则比对与意见生成；当前返回 Mock
    risk_id = risk_point.get("risk_id", "RISK_001") if isinstance(risk_point, dict) else "RISK_001"
    return _dumps({
        "opinion_id": f"OPINION_{risk_id.split('_')[-1]}",
        "risk_id": risk_id,
        "suggestion_type": "修改",
        "original_clause": risk_point.get("original_text", "") if isinstance(risk_point, dict)
        else "合同签订后5个工作日内支付合同总额的50%作为预付款",
        "suggested_clause": "合同签订后5个工作日内支付合同总额的30%作为预付款，"
                            "剩余款项按交付进度分期支付",
        "reasoning": "降低预付款比例可减少资金占用风险，分期支付与交付进度挂钩可强化履约约束",
        "legal_basis": "《中华人民共和国民法典》合同编相关规定",
        "compromise_option": "如对方不同意降低预付款，可要求增加银行保函等担保措施",
    })


# ---- F2-8 GenerateSummary ----

def _generate_summary(risk_points: list, opinions: list, contract_metadata: dict) -> str:
    # TODO: 接入评审结论模板引擎；当前返回 Mock（计数基于真实入参）
    total = len(risk_points) if isinstance(risk_points, list) else 0
    key_risks = [
        rp.get("risk_description", "") for rp in (risk_points or [])
        if isinstance(rp, dict)
    ][:3] or ["预付款比例过高", "违约责任不对等", "知识产权约定不明"]
    contract_name = (contract_metadata or {}).get("contract_name", "本合同") \
        if isinstance(contract_metadata, dict) else "本合同"
    return _dumps({
        "summary_id": "SUMMARY_001",
        "overall_assessment": "有条件通过",
        "risk_summary": {
            "total_count": total,
            "resolved_count": 0,
            "pending_count": total,
        },
        "key_risks": key_risks,
        "recommendations": [
            "降低预付款比例至30%",
            "平衡双方违约责任条款",
            "明确知识产权归属",
        ],
        "review_conclusion": (
            f"# 合同评审结论\n\n## 整体评估\n《{contract_name}》整体框架合理，"
            f"存在 {total} 项待整改风险点，建议按修改意见调整后签署。\n\n"
            "## 重点风险\n" + "\n".join(f"- {r}" for r in key_risks) + "\n\n"
            "## 综合建议\n按附件修改意见逐项落实，整改完成后可进入用印流程。"
        ),
    })


# ---- F2-9 SearchDocument ----

def _search_document(query: str, document_type: str | None = None, top_k: int = 5) -> str:
    # TODO: 接入检索服务（向量/全文索引）；当前返回 Mock
    results = [
        {
            "doc_id": "DOC_001",
            "doc_title": "2025年类似采购合同评审案例",
            "doc_type": document_type or "案例",
            "relevance_score": 0.92,
            "highlight": "…预付款比例过高，建议调整至30%以下…",
            "source_url": "https://example.com/cases/001",
        },
        {
            "doc_id": "DOC_002",
            "doc_title": "采购合同违约责任条款基线指引",
            "doc_type": document_type or "模板",
            "relevance_score": 0.85,
            "highlight": "…违约责任应当对等约定，逾期付款与逾期交付的违约金比例一致…",
            "source_url": "https://example.com/templates/002",
        },
    ]
    return _dumps({"results": results[: max(int(top_k), 1)], "total_hits": 15})


# ---- 注册 ----

def register_builtin_skills(registry: ToolRegistry) -> ToolRegistry:
    registry.register(SkillSpec(
        name="QueryRequirement",
        description="查询待评审需求，获取合同元数据、业务背景和附件列表。审查任务的入口步骤。",
        when_to_use="审查开始时按需求ID拉取合同元数据与背景",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户标识"},
                "requirement_id": {"type": "string", "description": "需求ID"},
            },
            "required": ["user_id", "requirement_id"],
        },
        handler=_query_requirement,
        tags=("review", "requirement"),
    ))
    registry.register(SkillSpec(
        name="IdentifyRisk",
        description="扫描合同条款，基于风险检查清单识别风险点，输出风险位置、描述、等级（高/中/低）与条款原文（红线批注）。",
        when_to_use="拿到合同文档后做风险扫描",
        input_schema={
            "type": "object",
            "properties": {
                "contract_document": {"type": "string", "description": "合同文档内容或文档ID"},
                "risk_checklist": {"type": "array", "items": {"type": "string"},
                                   "description": "风险检查清单（可选，缺省使用企业默认清单）"},
            },
            "required": ["contract_document"],
        },
        handler=_identify_risk,
        tags=("review", "risk"),
        requires_checkpoint=True,  # 确诊风险需法务人员卡点确认
    ))
    registry.register(SkillSpec(
        name="GenerateOpinion",
        description="针对单个风险点比对企业基线规则，生成修改建议（修改/删除/新增/妥协），输出建议条款、理由、法律依据与妥协方案。",
        when_to_use="风险点经人工确认后逐个出具意见",
        input_schema={
            "type": "object",
            "properties": {
                "risk_point": {"type": "object", "description": "风险点详情（IdentifyRisk 输出的单个 risk_point）"},
                "baseline_rules": {"type": "array", "items": {"type": "string"},
                                   "description": "企业基线规则（可选）"},
                "reference_cases": {"type": "array", "items": {"type": "object"},
                                    "description": "相似参考案例（可选，来自 SearchDocument）"},
            },
            "required": ["risk_point"],
        },
        handler=_generate_opinion,
        tags=("review", "opinion"),
        requires_checkpoint=True,
    ))
    registry.register(SkillSpec(
        name="GenerateSummary",
        description="汇总全部风险点与修改意见，生成含整体评估（通过/有条件通过/不通过）、重点风险与综合建议的结构化评审结论文档。",
        when_to_use="全部风险点处理完毕后出具综合结论",
        input_schema={
            "type": "object",
            "properties": {
                "risk_points": {"type": "array", "items": {"type": "object"},
                                "description": "所有风险点列表"},
                "opinions": {"type": "array", "items": {"type": "object"},
                             "description": "所有修改意见列表"},
                "contract_metadata": {"type": "object", "description": "合同元数据（QueryRequirement 输出）"},
            },
            "required": ["risk_points", "opinions", "contract_metadata"],
        },
        handler=_generate_summary,
        tags=("review", "summary"),
    ))
    registry.register(SkillSpec(
        name="SearchDocument",
        description="按关键词检索相似案例、法规或合同模板，返回相关度排序的文档列表（标题、得分、高亮片段）。",
        when_to_use="需要参考案例/法规依据/模板时，或单一检索请求",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问题描述"},
                "document_type": {"type": "string", "enum": ["案例", "法规", "模板"],
                                  "description": "文档类型（可选）"},
                "top_k": {"type": "integer", "description": "返回数量，默认5"},
            },
            "required": ["query"],
        },
        handler=_search_document,
        tags=("retrieval",),
    ))
    registry.attach_docs(DOCS_ROOT)
    return registry
