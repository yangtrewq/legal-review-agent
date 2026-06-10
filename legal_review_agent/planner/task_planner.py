"""2.2 任务规划组件 (Task Planner)

职责边界：系统的"导航仪"。接收网关确认的合法任务，将审查目标拆解为
细粒度标准子任务并生成 DAG（施工图纸），不涉及底层 API 的具体传参。
"""

from __future__ import annotations

import json

import anthropic

from ..config import Config
from ..sdk_utils import require_text
from ..registry.tool_registry import ToolRegistry
from ..types import ExecutionPlan, SubTask, UserInput

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "description": "本次审查的总目标"},
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "子任务唯一 id，如 t1、t2"},
                    "name": {"type": "string", "description": "子任务短名，如 历史基线拉取"},
                    "description": {"type": "string", "description": "子任务要达成什么"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "依赖的子任务 id 列表；无依赖则为空数组（可并发）",
                    },
                    "suggested_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "建议使用的技能名（来自技能目录），仅作提示不做绑定",
                    },
                },
                "required": ["id", "name", "description", "depends_on", "suggested_skills"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["goal", "tasks"],
    "additionalProperties": False,
}

_SYSTEM_TEMPLATE = """你是法律评审系统的任务规划器。把审查目标拆解为细粒度的标准子任务，并给出有向无环图（DAG）依赖关系。

规划原则：
1. 子任务粒度对齐业务 SOP，例如：[历史基线拉取] -> [条款解析] -> [风控比对] -> [意见生成]
2. 无依赖关系的子任务放在同一并发层（depends_on 为空或仅依赖已完成节点）
3. 只描述"做什么"，不要给出工具的具体传参 —— 参数由执行引擎在运行时动态提取
4. 不要规划系统能力之外的步骤

当前系统可用技能目录（仅供参考 suggested_skills 字段）：
{catalog}"""


class TaskPlanner:
    def __init__(self, client: anthropic.Anthropic, config: Config, registry: ToolRegistry):
        self._client = client
        self._cfg = config
        self._registry = registry

    def plan(self, user_input: UserInput, tracer=None) -> ExecutionPlan:
        system = _SYSTEM_TEMPLATE.format(catalog=self._registry.catalog())
        messages = [{"role": "user", "content": f"审查需求：{user_input.text}"}]
        span = tracer.start_span(
            "model_request", self._cfg.models.engine_model,
            stage="planning", system=system, messages=messages,
        ) if tracer else None
        response = self._client.messages.create(
            model=self._cfg.models.engine_model,
            max_tokens=self._cfg.models.engine_max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            output_config={"format": {"type": "json_schema", "schema": _PLAN_SCHEMA}},
            messages=messages,
        )
        text = require_text(response, stage="planning")
        data = json.loads(text)
        if tracer:
            tracer.end_span(span, plan=data, usage=response.usage)
        plan = ExecutionPlan(
            goal=data["goal"],
            tasks=[SubTask(**t) for t in data["tasks"]],
        )
        plan.topological_batches()  # 校验无环，提前暴露规划错误
        return plan
