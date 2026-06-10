# 法律评审 Agent (Legal Review Agent)

基于 **Plan-and-Execute 混合架构** 的企业级法律评审 Agent，结合 Just-in-Time (JIT) 动态工具调用与 **控制面/数据面解耦** 设计。

## 核心逻辑流

```
多模态输入
  → [意图路由]            gateway/intent_router.py
  → [任务规划]            planner/task_planner.py      （DAG 施工图纸）
  → 按需检视 [技能/工具注册] registry/tool_registry.py
  → [上下文组装]           context/context_assembler.py （融合 [记忆] memory/）
  → [Agentic Loop 认知调度] engine/cognitive_engine.py   （控制面）
  → [工具执行网关]          executor/action_gateway.py   （数据面：物理调用与容错）
  → [结果反馈与交付 HITL]   delivery/hitl.py             （对接 Eureka X）
```

## 组件映射

| 设计组件 | 模块 | 关键实现 |
|---|---|---|
| 2.1 意图路由网关 | `gateway/intent_router.py` | haiku 低成本模型 + 结构化输出做四分类；主链路/旁路/短路分发，**不做槽位提取** |
| 2.2 任务规划 | `planner/task_planner.py` | opus + adaptive thinking 拆解子任务；`ExecutionPlan.topological_batches()` 输出并发/串行批次并校验无环 |
| 2.3 技能/工具注册表 | `registry/tool_registry.py` + `skills/builtin.py` | 静态 Schema 字典；`load_schemas(names)` 供引擎 JIT 按需加载；`sandboxed` / `requires_checkpoint` 元属性 |
| 2.4 记忆管理 | `memory/memory_manager.py` | 三层：长记忆（JSONL 记忆文件）/ 会话记忆（按 session 持久化关键状态，防失忆式反复）/ 工作区记忆（任务级暂存） |
| 2.5 上下文组装 | `context/context_assembler.py` | 静态系统提示词字节级冻结 + `cache_control` 前缀缓存；haiku 做记忆相关性筛选注入；超阈值触发有损压缩 |
| 2.6 认知循环引擎 | `engine/cognitive_engine.py` | Plan→Execute→Evaluate 状态机；JIT 槽位提取（模型生成 tool_use）；`ask_user` 主动追问（Reverse Prompting）；只下发指令不做物理调用 |
| 2.7 工具执行网关 | `executor/action_gateway.py` + `resilience.py` | 参数校验/鉴权注入；重试（指数退避）+ 超时 + 限流（令牌桶）+ 熔断；不可恢复时抛 `ToolExecutionError` 请求 AI 重新决策；沙盒路由 |
| 2.8 结果反馈与 HITL | `delivery/hitl.py` | `HITLChannel` 标准化接口：`ConsoleChannel`（本地）/ `EurekaXAdapter`（企业入口对接，待联调） |

## 关键设计决策

- **控制面/数据面解耦**：`CognitiveEngine` 只产出 `ToolInstruction`（标准数字指令）并评估 `ToolOutcome`；
  `ActionGateway` 只做物理调用与基建容错，没有任何认知逻辑。两者通过 `types.py` 中的数据契约通信。
- **模型分层**：认知引擎/规划器用 `claude-opus-4-8`（adaptive thinking, effort=high）；
  意图路由、记忆筛选、历史压缩用 `claude-haiku-4-5` 控制成本。
- **HITL 卡点**：`requires_checkpoint=True` 的技能（确诊风险、生成意见）执行成功后，
  结果先经法务人员 Review/修改/驳回，处理结果再回喂模型；驳回会以 `is_error` 形式触发模型重新决策。
- **主动追问**：内置 `ask_user` 工具由引擎拦截走 HITL 通道（不进执行网关），
  追问获得的参数沉淀到会话记忆，后续轮次不再重复询问。
- **前缀缓存友好**：静态系统提示词禁止插入时间戳/会话 ID 等易变内容；动态上下文统一放在 messages 内。

## 快速开始

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
python -m legal_review_agent.app     # 交互式 CLI（本地用 ConsoleChannel 模拟 Eureka X 卡点）
pytest                               # 运行测试（不依赖 API Key）
```

## 接入真实业务

1. **业务技能**：替换 `skills/builtin.py` 中各 handler 的占位实现（基线库、条款解析、风控比对服务）；Schema 契约不变则上层零改动。
2. **Eureka X 对接**：实现 `delivery/hitl.py::EurekaXAdapter` 的卡点工作台与交付通道 API。
3. **鉴权**：向 `ActionGateway(auth_provider=...)` 注入企业凭证提供器（OAuth Token 等），声明了 `auth_context` 参数的 handler 会自动收到凭证。
4. **沙盒**：高风险技能（`sandboxed=True`）需配置 `SandboxRunner` 的真实隔离运行时，默认拒绝执行。
