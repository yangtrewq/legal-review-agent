"""Web 服务端：前端交互 + 链路观测 API。

端点：
- POST /api/chat                          发起一次运行，返回 run_id
- GET  /api/runs/{run_id}/events          SSE 事件流（流式文本/任务进度/工具调用/卡点）
- POST /api/checkpoints/{checkpoint_id}/resolve   HITL 卡点回执
- GET  /api/runs                          运行列表
- GET  /api/runs/{run_id}/trace           完整链路（含组装后的提示词与工具调用过程）
- GET  /                                  前端单页

启动：uvicorn legal_review_agent.server.app:app --reload
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..app import LegalReviewAgent
from ..delivery.hitl import CheckpointResolution
from ..delivery.web_channel import WebHITLChannel
from ..observability.events import EventEmitter
from ..observability.tracing import Tracer, TraceStore
from ..types import UserInput

logger = logging.getLogger(__name__)

app = FastAPI(title="法律评审 Agent")

_agent = LegalReviewAgent()
_trace_store = TraceStore()
_runs: dict[str, dict] = {}          # run_id → {emitter, hitl, thread}
_runs_lock = threading.Lock()
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"


class ResolveRequest(BaseModel):
    action: str = "approve"            # approve / modify / reject / answer
    payload: dict | None = None
    comment: str = ""


@app.post("/api/chat")
def start_chat(req: ChatRequest) -> dict:
    run_id = uuid.uuid4().hex[:12]
    emitter = EventEmitter()
    hitl = WebHITLChannel(emitter)
    tracer = Tracer(run_id)
    _trace_store.add(tracer)

    def worker() -> None:
        try:
            _agent.handle(
                UserInput(text=req.text, session_id=req.session_id),
                emitter=emitter, tracer=tracer, hitl=hitl,
            )
        except Exception as e:  # noqa: BLE001 — 异常需透传给前端
            logger.exception("运行 %s 失败", run_id)
            emitter.emit("error", {"message": str(e)})
        finally:
            emitter.emit("done")
            emitter.close()

    thread = threading.Thread(target=worker, name=f"run-{run_id}", daemon=True)
    with _runs_lock:
        _runs[run_id] = {"emitter": emitter, "hitl": hitl, "thread": thread}
    thread.start()
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}/events")
def stream_events(run_id: str) -> StreamingResponse:
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run 不存在")

    def event_stream():
        for event in run["emitter"].consume():
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/checkpoints/{checkpoint_id}/resolve")
def resolve_checkpoint(checkpoint_id: str, req: ResolveRequest) -> dict:
    with _runs_lock:
        runs = list(_runs.values())
    for run in runs:
        if run["hitl"].resolve(checkpoint_id, CheckpointResolution(
            action=req.action, payload=req.payload, comment=req.comment,
        )):
            return {"ok": True}
    raise HTTPException(404, "卡点不存在或已处理")


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return _trace_store.list_summaries()


@app.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str) -> dict:
    tracer = _trace_store.get(run_id)
    if tracer is None:
        raise HTTPException(404, "trace 不存在")
    return tracer.to_dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")
