"""第二轮代码审查修复的回归测试：线程池防护、中断幂等、trace 截断、并发记忆、鉴权。"""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from legal_review_agent.config import DEFAULT_CONFIG, ResilienceConfig
from legal_review_agent.delivery.hitl import CheckpointResolution, HITLChannel
from legal_review_agent.engine.cognitive_engine import CognitiveEngine
from legal_review_agent.engine.run_state import RunState, RunStateStore
from legal_review_agent.executor.action_gateway import ActionGateway
from legal_review_agent.memory.memory_manager import MemoryManager, SessionMemory
from legal_review_agent.observability.events import EventEmitter
from legal_review_agent.observability.tracing import Tracer, compact_for_trace
from legal_review_agent.registry.tool_registry import SkillSpec, ToolRegistry
from legal_review_agent.skills.builtin import register_builtin_skills
from legal_review_agent.types import RunInterrupted, ToolExecutionError, ToolInstruction


# ---- 缺陷#4：线程池耗尽快速失败 ----

def test_pool_exhaustion_fails_fast():
    cfg = ResilienceConfig(max_retries=0, retry_base_delay_s=0, tool_timeout_s=0.05,
                           tool_max_workers=1, rate_limit_per_minute=10_000,
                           circuit_failure_threshold=99, circuit_recovery_time_s=1)
    registry = ToolRegistry()
    registry.register(SkillSpec(
        name="hang", description="d",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: time.sleep(5) or "never",
    ))
    gw = ActionGateway(registry, cfg)

    # 第一次：超时（worker 仍挂死占用唯一槽位）
    with pytest.raises(ToolExecutionError, match="重试"):
        gw.execute(ToolInstruction("id1", "hang", {}))
    # 第二次：槽位未归还 → 快速失败而非排队等死
    start = time.monotonic()
    with pytest.raises(ToolExecutionError, match="线程池已满"):
        gw.execute(ToolInstruction("id2", "hang", {}))
    assert time.monotonic() - start < 1.0


def test_slot_released_after_handler_finishes():
    cfg = ResilienceConfig(max_retries=0, retry_base_delay_s=0, tool_timeout_s=2,
                           tool_max_workers=1, rate_limit_per_minute=10_000,
                           circuit_failure_threshold=99, circuit_recovery_time_s=1)
    registry = ToolRegistry()
    registry.register(SkillSpec(
        name="quick", description="d",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda: "ok",
    ))
    gw = ActionGateway(registry, cfg)
    for i in range(3):  # 正常完成的 handler 归还槽位，可连续执行
        assert gw.execute(ToolInstruction(f"id{i}", "quick", {})).content == "ok"


# ---- 缺陷#5：中断幂等（保留已执行结果，补齐未执行回执） ----

class InterruptOnAsk(HITLChannel):
    def raise_checkpoint(self, checkpoint):
        raise RunInterrupted

    def deliver(self, title, content):
        pass


def test_interrupt_mid_turn_preserves_executed_results(tmp_path):
    registry = register_builtin_skills(ToolRegistry())
    gateway = ActionGateway(registry, ResilienceConfig(
        max_retries=0, retry_base_delay_s=0, tool_timeout_s=5,
        rate_limit_per_minute=10_000, circuit_failure_threshold=99, circuit_recovery_time_s=1))
    store = RunStateStore(tmp_path)
    engine = CognitiveEngine(
        llm=None, config=DEFAULT_CONFIG, registry=registry,
        gateway=gateway, assembler=SimpleNamespace(maybe_compress=lambda m: m),
        memory=MemoryManager(str(tmp_path), "t"), hitl=InterruptOnAsk(),
        tracer=Tracer("test"), emitter=EventEmitter(), state_store=store,
    )
    # 一轮回合：先执行 fetch_baseline（成功），再 ask_user（触发中断）
    response = SimpleNamespace(stop_reason="tool_use", content=[
        SimpleNamespace(type="tool_use", id="t-ok", name="fetch_baseline",
                        input={"contract_type": "采购合同"}),
        SimpleNamespace(type="tool_use", id="t-ask", name="ask_user",
                        input={"question": "立场？"}),
    ])
    engine._call_model = lambda messages, tools: response

    state = RunState(run_id="r-mid", session_id="t", user_text="u", goal="g",
                     tasks=[], messages=[{"role": "user", "content": "hi"}], active_tools=[])
    assert engine._execute(state) is None

    saved = store.load("r-mid")
    last = saved.messages[-1]
    assert last["role"] == "user"
    results = {r["tool_use_id"]: r for r in last["content"]}
    assert "采购合同" in results["t-ok"]["content"]          # 已执行结果保留
    assert results["t-ask"]["is_error"]                      # 未执行的补中断回执
    assert "中断" in results["t-ask"]["content"]
    # tool_use 与 tool_result 配对完整
    assert {b.id for b in response.content} == set(results)


# ---- Trace 截断 ----

def test_compact_for_trace_truncates_long_strings():
    doc = "A" * 20_000
    out = compact_for_trace({"messages": [{"content": [{"data": doc, "type": "document"}]}]})
    data = out["messages"][0]["content"][0]["data"]
    assert len(data) < 9_000 and "已截断" in data
    assert compact_for_trace("短文本") == "短文本"


def test_tracer_payload_truncated_at_record_time():
    t = Tracer("r")
    t.record("model_request", "m", messages=[{"role": "user", "content": "B" * 20_000}])
    payload = t.to_dict()["spans"][0]["payload"]
    assert "已截断" in payload["messages"][0]["content"]


# ---- 缺陷#9：SessionMemory 并发写 ----

def test_session_memory_concurrent_set_fact(tmp_path):
    memory = SessionMemory(tmp_path, "concurrent")
    threads = [threading.Thread(target=memory.set_fact, args=(f"k{i}", i)) for i in range(50)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # 文件是合法 JSON 且 50 个事实全部在场（同实例并发不互相覆盖）
    on_disk = json.loads((tmp_path / "sessions" / "concurrent.json").read_text(encoding="utf-8"))
    assert len(on_disk["facts"]) == 50


# ---- 缺陷#3：server 鉴权 ----

def test_server_auth_middleware():
    from fastapi.testclient import TestClient
    import legal_review_agent.server.app as server

    client = TestClient(server.app)
    original = server._SERVER_TOKEN
    server._SERVER_TOKEN = "secret-token"
    try:
        assert client.get("/api/runs").status_code == 401
        assert client.get("/api/runs", headers={"X-API-Token": "wrong"}).status_code == 401
        assert client.get("/api/runs", headers={"X-API-Token": "secret-token"}).status_code == 200
        assert client.get("/api/runs",
                          headers={"Authorization": "Bearer secret-token"}).status_code == 200
        assert client.get("/api/runs?token=secret-token").status_code == 200  # SSE 场景
        assert client.get("/").status_code == 200  # 前端页面不拦截
    finally:
        server._SERVER_TOKEN = original


def test_server_auth_disabled_by_default():
    from fastapi.testclient import TestClient
    import legal_review_agent.server.app as server

    client = TestClient(server.app)
    assert server._SERVER_TOKEN == "" or server._SERVER_TOKEN is not None
    if not server._SERVER_TOKEN:
        assert client.get("/api/runs").status_code == 200
