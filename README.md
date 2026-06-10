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

## 技能渐进披露 (Progressive Disclosure)

四层披露资产，按需进入上下文以控制 token 成本：

| 层级 | 内容 | 何时进上下文 | 实现 |
|---|---|---|---|
| L0 元数据 | name + 描述 + when_to_use 触发条件 | 常驻（规划器目录 + 引擎引导消息） | `SkillSpec.when_to_use`、`ToolRegistry.catalog()` |
| L1 Schema | input_schema 完整定义 | bootstrap 按规划建议加载，或运行中经 `load_skill` 补载 | `load_schemas()`、引擎 `_handle_load_skill` |
| L2 指南正文 | 详细 SOP / 注意事项（instructions.md） | 技能**首次被调用**时注入 tool_result，每轮一次 | 引擎 `_maybe_inject_instructions` |
| L3 资源文件 | 范本 / checklist（resources/） | 模型经 `read_skill_resource` 显式读取 | `SkillSpec.read_resource()`（拒绝路径穿越） |

- **文档目录约定**：`skills/docs/<skill_name>/{instructions.md, resources/*}`，由 `registry.attach_docs()` 挂载，正文懒加载；
- **披露触发三机制**：规划期收窄（suggested_skills）、模型自主发现（`load_skill` 元工具，补上规划遗漏）、首调自动注入（L2）；
- **缓存友好**：L2/L3 一律走 messages 尾部（tool_result），不进 system；`load_skill` 中途追加工具会击穿前缀缓存，该事件以 `cache_invalidated` 标记记入链路；
- **观测**：每次披露记录 `skill_disclosure` Span（层级/触发方式/估算 token），前端聊天流显示披露气泡，调试抽屉可逐条审阅。

## HITL 增强：中断/恢复 与 卡片选项

**Agent Loop 中断与恢复**：
- 安全点中断：`POST /api/runs/{id}/interrupt` 置位中断信号，引擎在**迭代边界**或**卡点等待中**响应；
  执行到一半的工具回合整体回滚（恢复后模型重做），保证上下文完整性；
- 现场持久化：`engine/run_state.py` 把对话上下文、激活工具集、已注入技能、迭代位置落盘
  （`RunStateStore`，JSON，可跨进程恢复）；`GET /api/runs/suspended` 列出可恢复的运行；
- 恢复：`POST /api/runs/{id}/resume`（可附带 `supplement` 补充指示，以 `<resume_note>` 注入上下文）；
  已注入的技能指南不会重复注入；前端「⏸ 暂停 / ▶ 恢复」按钮 + 恢复条交互。

**备选项卡片 (CheckpointOption)**：
- `ask_user` 工具支持模型给出 2-4 个候选答案（label/description/recommended），前端以卡片展示，点击即回执；
- 审查类卡点（`requires_checkpoint` 技能）自动生成「批准（推荐）/ 驳回」标准卡片，
  卡片自带回执动作（approve/reject/answer），点选与自由文本输入并存；
- CLI 的 `ConsoleChannel` 同步支持序号选择备选项。

## 前端交互与链路观测

| 能力 | 模块 | 说明 |
|---|---|---|
| Web 服务端 | `server/app.py` | FastAPI：`POST /api/chat` 发起运行；`GET /api/runs/{id}/events` SSE 事件流；`POST /api/checkpoints/{id}/resolve` HITL 回执；`GET /api/runs/{id}/trace` 链路查询 |
| 前端单页 | `web/index.html` | 零构建：对话流式输出（text_delta 增量渲染）、Plan 任务清单进度可视化（pending/running/done）、工具调用气泡、HITL 卡点弹窗（批准/回答/驳回） |
| 运行时事件流 | `observability/events.py` | `EventEmitter` 线程安全队列：routing / plan / task_status / text_delta / tool_start|end / checkpoint / final / done |
| 链路观测 | `observability/tracing.py` | 按 run_id 记录 Span：**model_request 含完整组装后的 system/messages/tools**、tool_call 含下发指令与执行结果及耗时、routing/planning/checkpoint；前端"链路调试"抽屉可逐 Span 审阅 |
| Web HITL 通道 | `delivery/web_channel.py` | 卡点经 SSE 推前端，引擎线程阻塞等回执；超时自动驳回兜底 |

任务进度采用启发式映射：子任务 `suggested_skills` 全部成功执行即标记完成，引擎逐事件推送 `task_status`。

## 快速开始

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...

# Web 模式（推荐）：对话流式输出 + 任务进度 + 链路调试
uvicorn legal_review_agent.server.app:app --port 8000
# 浏览器打开 http://localhost:8000

python -m legal_review_agent.app     # 或：交互式 CLI（ConsoleChannel 模拟卡点）
pytest                               # 运行测试（不依赖 API Key）
```

## 接入真实业务

1. **业务技能**：替换 `skills/builtin.py` 中各 handler 的占位实现（基线库、条款解析、风控比对服务）；Schema 契约不变则上层零改动。
2. **Eureka X 对接**：实现 `delivery/hitl.py::EurekaXAdapter` 的卡点工作台与交付通道 API。
3. **鉴权**：向 `ActionGateway(auth_provider=...)` 注入企业凭证提供器（OAuth Token 等），声明了 `auth_context` 参数的 handler 会自动收到凭证。
4. **沙盒**：高风险技能（`sandboxed=True`）需配置 `SandboxRunner` 的真实隔离运行时，默认拒绝执行。
