import pytest

from legal_review_agent.config import ResilienceConfig
from legal_review_agent.executor.action_gateway import ActionGateway
from legal_review_agent.executor.resilience import CircuitBreaker
from legal_review_agent.registry.tool_registry import SkillSpec, ToolRegistry
from legal_review_agent.types import ToolExecutionError, ToolInstruction

FAST_CFG = ResilienceConfig(
    max_retries=2, retry_base_delay_s=0.01, retry_max_delay_s=0.02,
    tool_timeout_s=2.0, rate_limit_per_minute=10_000,
    circuit_failure_threshold=3, circuit_recovery_time_s=0.05,
)


def make_gateway(handler, name="t", sandboxed=False):
    registry = ToolRegistry()
    registry.register(SkillSpec(
        name=name, description="d",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        handler=handler, sandboxed=sandboxed,
    ))
    return ActionGateway(registry, FAST_CFG)


def test_successful_execution():
    gw = make_gateway(lambda x: f"ok:{x}")
    out = gw.execute(ToolInstruction("id1", "t", {"x": "v"}))
    assert out.content == "ok:v" and not out.is_error


def test_missing_required_argument_raises():
    gw = make_gateway(lambda x: "ok")
    with pytest.raises(ToolExecutionError, match="必填参数"):
        gw.execute(ToolInstruction("id1", "t", {}))


def test_retry_then_succeed():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "recovered"

    gw = make_gateway(flaky)
    out = gw.execute(ToolInstruction("id1", "t", {"x": "v"}))
    assert out.content == "recovered" and calls["n"] == 3


def test_exhausted_retries_raise_tool_execution_error():
    def always_fail(x):
        raise RuntimeError("boom")

    gw = make_gateway(always_fail)
    with pytest.raises(ToolExecutionError, match="重试"):
        gw.execute(ToolInstruction("id1", "t", {"x": "v"}))


def test_execute_safe_converts_error_to_outcome():
    gw = make_gateway(lambda x: "ok")
    out = gw.execute_safe(ToolInstruction("id1", "unknown_tool", {}))
    assert out.is_error and "ToolExecutionError" in out.content


def test_sandboxed_skill_blocked_without_runtime():
    gw = make_gateway(lambda x: "ok", sandboxed=True)
    with pytest.raises(ToolExecutionError, match="沙盒"):
        gw.execute(ToolInstruction("id1", "t", {"x": "v"}))


def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=2, recovery_time_s=0.05)
    assert cb.allow()
    cb.record_failure()
    cb.record_failure()
    assert not cb.allow()
    import time
    time.sleep(0.06)
    assert cb.state == CircuitBreaker.HALF_OPEN
    assert cb.allow()
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED
