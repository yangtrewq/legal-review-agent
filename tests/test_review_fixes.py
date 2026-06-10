"""代码审查修复的回归测试：压缩序列化/回合边界、文本提取、auth_context 注入。"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from legal_review_agent.config import Config, ResilienceConfig
from legal_review_agent.context.context_assembler import ContextAssembler
from legal_review_agent.executor.action_gateway import ActionGateway
from legal_review_agent.registry.tool_registry import SkillSpec, ToolRegistry
from legal_review_agent.sdk_utils import extract_text, require_text
from legal_review_agent.types import ToolInstruction


class FakeTextBlock(BaseModel):
    """模拟 SDK 响应内容块（Pydantic 对象，json.dumps 不可直接序列化）。"""

    type: str = "text"
    text: str = "模型输出"


class FakeToolUseBlock(BaseModel):
    type: str = "tool_use"
    id: str = "tu1"
    name: str = "fetch_baseline"
    input: dict = {}


def fake_llm(summary_text="压缩摘要"):
    response = SimpleNamespace(content=[SimpleNamespace(type="text", text=summary_text)])
    return SimpleNamespace(complete=lambda **kw: response)


def make_assembler(threshold):
    cfg = Config(context_compaction_threshold_tokens=threshold)
    return ContextAssembler(fake_llm(), cfg, memory=None)


def tool_result_msg(tool_use_id):
    return {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": "ok"}]}


def assistant_tool_use_msg(tool_use_id):
    return {"role": "assistant", "content": [FakeToolUseBlock(id=tool_use_id)]}


# ---- 缺陷#1：messages 含 SDK 对象时估算不再崩溃 ----

def test_maybe_compress_handles_sdk_objects_below_threshold():
    assembler = make_assembler(threshold=10**9)
    messages = [
        {"role": "user", "content": "审查需求"},
        {"role": "assistant", "content": [FakeTextBlock(), FakeToolUseBlock()]},
        tool_result_msg("tu1"),
    ]
    assert assembler.maybe_compress(messages) is messages  # 无崩溃且未触发压缩


# ---- 缺陷#2：压缩切分对齐回合边界 ----

def test_compress_tail_never_starts_with_orphan_tool_result():
    assembler = make_assembler(threshold=1)
    # 构造 [-4] 位置恰好是 tool_result 的消息序列
    messages = [
        {"role": "user", "content": "审查需求" * 100},
        assistant_tool_use_msg("a"), tool_result_msg("a"),
        assistant_tool_use_msg("b"),
        tool_result_msg("b"),                      # ← len-4：孤立风险点
        {"role": "assistant", "content": [FakeTextBlock()]},
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": [FakeTextBlock()]},
    ]
    result = assembler.maybe_compress(messages)
    assert any("<compressed_history>" in str(m.get("content")) for m in result)
    # 压缩标记之后的保留区（tail）不得含任何孤立 tool_result —— tool_use "b" 与其
    # tool_result 必须同侧（一起被压缩进 middle）
    idx = next(i for i, m in enumerate(result) if "<compressed_history>" in str(m.get("content")))
    assert all(not assembler._is_tool_result_message(m) for m in result[idx + 1:])


def test_compress_gives_up_when_no_safe_boundary():
    assembler = make_assembler(threshold=1)
    messages = [{"role": "user", "content": "x" * 100}] + [tool_result_msg(f"t{i}") for i in range(6)]
    assert assembler.maybe_compress(messages) is messages  # 找不到安全切点 → 放弃压缩


def test_compress_keeps_original_when_summarizer_returns_empty():
    cfg = Config(context_compaction_threshold_tokens=1)
    assembler = ContextAssembler(fake_llm(), cfg, memory=None)
    assembler._llm = SimpleNamespace(complete=lambda **kw: SimpleNamespace(content=[]))
    messages = [{"role": "user", "content": f"消息{i}" * 50} for i in range(10)]
    assert assembler.maybe_compress(messages) is messages


# ---- 缺陷#7：文本提取的健壮性 ----

def test_extract_text_empty_content_returns_default():
    response = SimpleNamespace(content=[], stop_reason="refusal")
    assert extract_text(response) == ""
    assert extract_text(response, default="兜底") == "兜底"


def test_require_text_raises_clear_error():
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking")], stop_reason="refusal")
    with pytest.raises(RuntimeError, match=r"\[routing\].*refusal"):
        require_text(response, stage="routing")


def test_extract_text_joins_multiple_blocks():
    response = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="tool_use"),
        SimpleNamespace(type="text", text="b"),
    ])
    assert extract_text(response) == "a\nb"


# ---- 缺陷#6：auth_context 注入判定 ----

FAST_CFG = ResilienceConfig(max_retries=0, retry_base_delay_s=0, tool_timeout_s=5,
                            rate_limit_per_minute=10_000, circuit_failure_threshold=99,
                            circuit_recovery_time_s=1)


def make_gateway(handler):
    registry = ToolRegistry()
    registry.register(SkillSpec(
        name="t", description="d",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        handler=handler,
    ))
    return ActionGateway(registry, FAST_CFG, auth_provider=lambda: {"token": "secret"})


def test_handler_with_local_var_auth_context_not_injected():
    def handler(x):
        auth_context = "局部变量，不是参数"  # noqa: F841
        return f"ok:{x}"

    out = make_gateway(handler).execute(ToolInstruction("id1", "t", {"x": "v"}))
    assert out.content == "ok:v"


def test_handler_with_param_auth_context_receives_credentials():
    def handler(x, auth_context=None):
        return auth_context["token"]

    out = make_gateway(handler).execute(ToolInstruction("id1", "t", {"x": "v"}))
    assert out.content == "secret"
