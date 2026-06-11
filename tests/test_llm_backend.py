"""模型后端抽象层测试：协议翻译、能力门控、配置加载、宽松 JSON 解析。"""

from types import SimpleNamespace

import pytest

from legal_review_agent.config import Config, ModelConfig
from legal_review_agent.llm.backend import (
    LLMResponse, TextBlock, ThinkingBlock, ToolUseBlock, build_backend,
)
from legal_review_agent.llm.openai_backend import (
    from_openai_message, to_openai_messages, to_openai_tools,
)
from legal_review_agent.sdk_utils import parse_json


# ---- 配置 ----

def test_model_config_from_env(monkeypatch):
    monkeypatch.setenv("LRA_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LRA_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LRA_ENGINE_MODEL", "glm-v5")
    monkeypatch.setenv("LRA_ROUTER_MODEL", "glm-v5")
    cfg = ModelConfig()
    assert cfg.provider == "openai_compatible"
    assert cfg.engine_model == "glm-v5"
    # openai_compatible 默认关闭 Anthropic 专有能力
    assert not cfg.caps.adaptive_thinking
    assert not cfg.caps.effort
    assert not cfg.caps.structured_outputs
    assert not cfg.caps.prompt_caching


def test_anthropic_default_capabilities():
    caps = ModelConfig(provider="anthropic").caps
    assert caps.adaptive_thinking and caps.effort and caps.structured_outputs


def test_build_backend_rejects_unknown_provider():
    with pytest.raises(ValueError, match="未知的模型 Provider"):
        build_backend(ModelConfig(provider="gemini"))


# ---- 出向翻译：Anthropic 风格 → OpenAI ----

def test_to_openai_messages_assistant_tool_use():
    messages = [
        {"role": "user", "content": "审查合同"},
        {"role": "assistant", "content": [
            TextBlock(text="先拉基线"),
            ToolUseBlock(id="tu1", name="fetch_baseline", input={"contract_type": "采购合同"}),
            ThinkingBlock(thinking="...", signature="sig"),  # Anthropic 专有，应被剔除
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "基线JSON"},
        ]},
    ]
    out = to_openai_messages("系统提示", messages)
    assert out[0] == {"role": "system", "content": "系统提示"}
    assert out[1] == {"role": "user", "content": "审查合同"}
    assistant = out[2]
    assert assistant["content"] == "先拉基线"
    assert assistant["tool_calls"][0]["function"]["name"] == "fetch_baseline"
    assert '"采购合同"' in assistant["tool_calls"][0]["function"]["arguments"]
    assert "thinking" not in str(assistant)
    assert out[3] == {"role": "tool", "tool_call_id": "tu1", "content": "基线JSON"}


def test_to_openai_tools_schema_mapping():
    tools = to_openai_tools([{
        "name": "t", "description": "d",
        "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }])
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["properties"]["x"]["type"] == "string"


# ---- 入向翻译：OpenAI → 归一化 ----

def test_from_openai_message_tool_calls():
    message = SimpleNamespace(
        content="我来调用工具",
        tool_calls=[SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="fetch_baseline",
                                     arguments='{"contract_type": "采购合同"}'),
        )],
    )
    resp = from_openai_message(message, "tool_calls")
    assert resp.stop_reason == "tool_use"
    assert resp.content[0].text == "我来调用工具"
    tool = resp.content[1]
    assert tool.type == "tool_use" and tool.input == {"contract_type": "采购合同"}


def test_from_openai_message_stop_reason_mapping():
    msg = SimpleNamespace(content="完成", tool_calls=None)
    assert from_openai_message(msg, "stop").stop_reason == "end_turn"
    assert from_openai_message(msg, "length").stop_reason == "max_tokens"


def test_from_openai_message_bad_tool_arguments():
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
        id="c1", function=SimpleNamespace(name="t", arguments="不是JSON"),
    )])
    resp = from_openai_message(message, "tool_calls")
    assert resp.content[0].input == {}  # 解析失败兜底为空参，交给网关必填校验报错


# ---- 宽松 JSON 解析（本地模型常见输出形态） ----

def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_with_code_fence():
    assert parse_json('```json\n{"intent": "legal_review"}\n```') == {"intent": "legal_review"}


def test_parse_json_with_surrounding_text():
    text = '好的，以下是分类结果：{"intent": "chitchat", "note": "含}括号\\"引号"} 希望有帮助'
    assert parse_json(text)["intent"] == "chitchat"


def test_parse_json_no_object_raises():
    with pytest.raises(ValueError):
        parse_json("完全没有 JSON")


# ---- 归一化块可直接回传 Anthropic（serialize 兼容） ----

def test_normalized_blocks_serializable():
    from legal_review_agent.observability.tracing import serialize
    blocks = [TextBlock(text="t"), ToolUseBlock(id="i", name="n", input={"a": 1}),
              ThinkingBlock(thinking="th", signature="sig")]
    dumped = serialize(blocks)
    assert dumped[0] == {"type": "text", "text": "t"}
    assert dumped[1] == {"type": "tool_use", "id": "i", "name": "n", "input": {"a": 1}}
    assert dumped[2]["signature"] == "sig"


# ---- .env 配置文件加载 ----

def test_load_env_file(tmp_path, monkeypatch):
    from legal_review_agent.config import load_env_file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\n"
        "LRA_PROVIDER=openai_compatible\n"
        'LRA_BASE_URL="http://localhost:8000/v1"\n'
        "LRA_ENGINE_MODEL = glm-v5\n"
        "\n"
        "无等号的坏行\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LRA_PROVIDER", raising=False)
    monkeypatch.delenv("LRA_BASE_URL", raising=False)
    monkeypatch.setenv("LRA_ENGINE_MODEL", "shell优先")  # shell 显式值不被 .env 覆盖

    loaded = load_env_file(env_file)
    assert loaded["LRA_PROVIDER"] == "openai_compatible"
    assert loaded["LRA_BASE_URL"] == "http://localhost:8000/v1"  # 引号被剥离

    import os
    assert os.environ["LRA_PROVIDER"] == "openai_compatible"
    assert os.environ["LRA_ENGINE_MODEL"] == "shell优先"

    monkeypatch.delenv("LRA_PROVIDER", raising=False)
    monkeypatch.delenv("LRA_BASE_URL", raising=False)


def test_load_env_file_missing_returns_empty(tmp_path):
    from legal_review_agent.config import load_env_file
    assert load_env_file(tmp_path / "nonexistent.env") == {}


# ---- 流式消费：reasoning_content（GLM 等推理模型）与统计 ----

def _chunk(content=None, reasoning=None, finish=None, tool_calls=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning,
                            tool_calls=tool_calls)
    return SimpleNamespace(usage=None,
                           choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def test_consume_openai_stream_reasoning_and_text():
    from legal_review_agent.llm.openai_backend import consume_openai_stream
    chunks = [
        _chunk(reasoning="让我想想"),
        _chunk(reasoning="，需要先比对基线"),
        _chunk(content="审查"),
        _chunk(content="完成"),
        _chunk(finish="stop"),
    ]
    texts, thoughts = [], []
    resp = consume_openai_stream(iter(chunks),
                                 on_text=texts.append, on_thinking=thoughts.append)
    assert "".join(texts) == "审查完成"
    assert "".join(thoughts) == "让我想想，需要先比对基线"
    assert resp.content[0].text == "审查完成"
    assert resp.stop_reason == "end_turn"
    stats = resp.usage["stream_stats"]
    assert stats["chunks"] == 5
    assert stats["text_chars"] == 4
    assert stats["reasoning_chars"] == 12


def test_consume_openai_stream_tool_calls_accumulated():
    from legal_review_agent.llm.openai_backend import consume_openai_stream
    tc1 = SimpleNamespace(index=0, id="call_1",
                          function=SimpleNamespace(name="fetch_", arguments='{"contract'))
    tc2 = SimpleNamespace(index=0, id=None,
                          function=SimpleNamespace(name="baseline", arguments='_type": "采购合同"}'))
    chunks = [_chunk(tool_calls=[tc1]), _chunk(tool_calls=[tc2]), _chunk(finish="tool_calls")]
    resp = consume_openai_stream(iter(chunks))
    assert resp.stop_reason == "tool_use"
    tool = resp.content[0]
    assert tool.name == "fetch_baseline"
    assert tool.input == {"contract_type": "采购合同"}
