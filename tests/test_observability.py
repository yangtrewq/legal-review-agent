import threading

from legal_review_agent.delivery.hitl import CheckpointResolution
from legal_review_agent.delivery.web_channel import WebHITLChannel
from legal_review_agent.observability.events import EventEmitter
from legal_review_agent.observability.tracing import Tracer, TraceStore, serialize
from legal_review_agent.types import Checkpoint


def test_tracer_records_spans_with_duration():
    t = Tracer("run1")
    span = t.start_span("model_request", "claude-opus-4-8", system="s", messages=[{"role": "user"}])
    t.end_span(span, stop_reason="end_turn")
    t.record("routing", "legal_review", confidence=0.9)

    data = t.to_dict()
    assert data["run_id"] == "run1"
    assert len(data["spans"]) == 2
    assert data["spans"][0]["payload"]["system"] == "s"
    assert data["spans"][0]["payload"]["stop_reason"] == "end_turn"
    assert data["spans"][0]["duration_ms"] is not None


def test_serialize_handles_unknown_objects():
    class Weird:
        pass
    out = serialize({"a": [Weird()], "b": 1})
    assert out["b"] == 1
    assert isinstance(out["a"][0], str)


def test_trace_store_eviction():
    store = TraceStore(max_traces=2)
    for i in range(3):
        store.add(Tracer(f"r{i}"))
    assert store.get("r0") is None
    assert store.get("r2") is not None
    assert [s["run_id"] for s in store.list_summaries()] == ["r2", "r1"]


def test_emitter_consume_until_close():
    em = EventEmitter()
    em.emit("text_delta", {"text": "你好"})
    em.emit("done")
    em.close()
    events = [e for e in em.consume() if e["type"] != "heartbeat"]
    assert [e["type"] for e in events] == ["text_delta", "done"]


def test_web_hitl_channel_roundtrip():
    em = EventEmitter()
    channel = WebHITLChannel(em, timeout_s=5)
    ckpt = Checkpoint(checkpoint_id="ck1", kind="ask_user", payload={}, question="合同类型？")

    result = {}

    def engine_thread():
        result["resolution"] = channel.raise_checkpoint(ckpt)

    th = threading.Thread(target=engine_thread)
    th.start()
    # 等待卡点事件出现后回执
    for event in em.consume(poll_timeout_s=0.1):
        if event["type"] == "checkpoint":
            assert event["checkpoint_id"] == "ck1"
            assert channel.resolve("ck1", CheckpointResolution(action="answer", payload={"answer": "采购合同"}))
            break
    th.join(timeout=5)
    assert result["resolution"].action == "answer"
    assert result["resolution"].payload == {"answer": "采购合同"}


def test_web_hitl_resolve_unknown_checkpoint():
    channel = WebHITLChannel(EventEmitter(), timeout_s=1)
    assert not channel.resolve("nope", CheckpointResolution(action="approve"))
