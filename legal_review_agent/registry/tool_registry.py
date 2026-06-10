"""2.3 技能与工具注册表 (Skill & Tool Registry)

职责边界：开发态对业务能力的规范化组织，运行时只读的"静态工具说明书（Schema 字典）"。
- 注册表本身不执行任何工具，只维护 Schema（入参说明、必填项）与执行函数的映射；
- 认知引擎在 JIT 节点按需 `load_schemas()` 加载目标 Schema（动态发现，避免全量塞入上下文）；
- 执行网关通过 `get_handler()` 取到物理执行函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SkillSpec:
    """一个高内聚的业务功能块（由业务 SOP 沉淀而来）。"""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]
    tags: tuple[str, ...] = ()
    # 高风险技能（如执行动态代码）由网关路由到沙盒执行
    sandboxed: bool = False
    # 需要 HITL 卡点确认的技能（如生成最终意见）
    requires_checkpoint: bool = False

    def to_anthropic_tool(self) -> dict[str, Any]:
        """转为 Claude Messages API 的 tool 定义。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"技能重复注册: {spec.name}")
        self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        if name not in self._skills:
            raise KeyError(f"未注册的技能: {name}")
        return self._skills[name]

    def get_handler(self, name: str) -> Callable[..., str]:
        return self.get(name).handler

    def list_names(self) -> list[str]:
        return sorted(self._skills)

    def catalog(self) -> str:
        """供规划器/引擎做能力发现的轻量目录（只含名称与描述，不含完整 Schema）。"""
        return "\n".join(
            f"- {s.name}: {s.description}" for s in self._skills.values()
        )

    def load_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """JIT 加载工具 Schema。names 为 None 时加载全部（小规模工具集场景）。"""
        specs = (
            self._skills.values()
            if names is None
            else [self.get(n) for n in names]
        )
        return [s.to_anthropic_tool() for s in specs]
