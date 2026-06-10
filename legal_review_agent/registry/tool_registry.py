"""2.3 技能与工具注册表 (Skill & Tool Registry)

职责边界：开发态对业务能力的规范化组织，运行时只读的"静态工具说明书（Schema 字典）"。
- 注册表本身不执行任何工具，只维护 Schema（入参说明、必填项）与执行函数的映射；
- 认知引擎在 JIT 节点按需 `load_schemas()` 加载目标 Schema（动态发现，避免全量塞入上下文）；
- 执行网关通过 `get_handler()` 取到物理执行函数。

渐进披露的四层资产（按需进入上下文，控制 token 成本）：
- L0 元数据：name + 一行描述 + when_to_use 触发条件（常驻目录，~30 token/技能）
- L1 Schema：input_schema 完整定义（工具纳入本轮 tools 时加载）
- L2 指南正文：详细 SOP / 注意事项（instructions.md，技能首次被调用时由引擎注入）
- L3 资源文件：范本 / checklist（resources/ 目录，模型经 read_skill_resource 显式读取）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
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
    # 渐进披露：L0 触发条件 / L2 正文路径 / L3 资源目录（由 attach_docs 挂载）
    when_to_use: str = ""
    instructions_path: Path | None = None
    resources_dir: Path | None = None
    _instructions_cache: str | None = field(default=None, repr=False, compare=False)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """转为 Claude Messages API 的 tool 定义（L1）。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def load_instructions(self) -> str | None:
        """懒加载 L2 指南正文，仅首次读盘。"""
        if self._instructions_cache is None and self.instructions_path is not None:
            self._instructions_cache = self.instructions_path.read_text(encoding="utf-8")
        return self._instructions_cache

    def list_resources(self) -> list[str]:
        """列出 L3 资源文件名。"""
        if self.resources_dir is None or not self.resources_dir.is_dir():
            return []
        return sorted(p.name for p in self.resources_dir.iterdir() if p.is_file())

    def read_resource(self, resource_name: str) -> str:
        """读取单个 L3 资源，拒绝路径穿越。"""
        if self.resources_dir is None:
            raise FileNotFoundError(f"技能 {self.name} 未配置资源目录")
        target = (self.resources_dir / resource_name).resolve()
        if target.parent != self.resources_dir.resolve() or not target.is_file():
            raise FileNotFoundError(f"技能 {self.name} 不存在资源: {resource_name}")
        return target.read_text(encoding="utf-8")


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
        """L0 目录：名称 + 描述 + 触发条件 + L2/L3 资产提示，供能力发现常驻上下文。"""
        lines = []
        for s in self._skills.values():
            line = f"- {s.name}: {s.description}"
            if s.when_to_use:
                line += f"（适用：{s.when_to_use}）"
            extras = []
            if s.instructions_path is not None:
                extras.append("有详细指南")
            resources = s.list_resources()
            if resources:
                extras.append(f"资源: {', '.join(resources)}")
            if extras:
                line += f" [{'; '.join(extras)}]"
            lines.append(line)
        return "\n".join(lines)

    def load_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """JIT 加载工具 Schema（L1）。names 为 None 时加载全部（小规模工具集场景）。"""
        specs = (
            self._skills.values()
            if names is None
            else [self.get(n) for n in names]
        )
        return [s.to_anthropic_tool() for s in specs]

    def attach_docs(self, docs_root: Path) -> None:
        """挂载渐进披露文档目录。约定结构：

        docs_root/<skill_name>/instructions.md   → L2 指南正文
        docs_root/<skill_name>/resources/*       → L3 资源文件
        """
        if not docs_root.is_dir():
            return
        for skill_dir in docs_root.iterdir():
            if not skill_dir.is_dir() or skill_dir.name not in self._skills:
                if skill_dir.is_dir():
                    logger.warning("技能文档 %s 没有对应的已注册技能，已忽略", skill_dir.name)
                continue
            spec = self._skills[skill_dir.name]
            instructions = skill_dir / "instructions.md"
            if instructions.is_file():
                spec.instructions_path = instructions
            resources = skill_dir / "resources"
            if resources.is_dir():
                spec.resources_dir = resources
