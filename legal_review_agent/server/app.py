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
import os
import secrets
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..app import LegalReviewAgent
from ..delivery.hitl import CheckpointResolution
from ..delivery.web_channel import WebHITLChannel
from ..observability.events import EventEmitter
from ..observability.tracing import Tracer, TraceStore
from ..types import UserInput

logger = logging.getLogger(__name__)

app = FastAPI(title="法律评审 Agent")

# 鉴权：设置 LRA_SERVER_TOKEN 后所有 /api/* 需携带凭证
# （Authorization: Bearer <token> / X-API-Token 头 / ?token= 查询参数，
#   查询参数用于 EventSource 无法自定义请求头的场景）。未设置则不启用（本地开发）。
_SERVER_TOKEN = os.environ.get("LRA_SERVER_TOKEN", "")


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if _SERVER_TOKEN and request.url.path.startswith("/api/"):
        provided = (
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.headers.get("x-api-token", "")
            or request.query_params.get("token", "")
        )
        if not secrets.compare_digest(provided, _SERVER_TOKEN):
            return JSONResponse({"detail": "未授权：缺少或错误的 API Token"}, status_code=401)
    return await call_next(request)


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


def _launch(run_id: str, target) -> None:
    """统一的运行线程封装：异常透传前端，结束时关闭事件流。"""
    emitter = _runs[run_id]["emitter"]

    entry = _runs[run_id]

    def worker() -> None:
        try:
            target()
        except Exception as e:  # noqa: BLE001 — 异常需透传给前端
            logger.exception("运行 %s 失败", run_id)
            emitter.emit("error", {"message": str(e)})
        finally:
            emitter.emit("done")
            emitter.close()
            # 不立即删除：前端可能尚未订阅 SSE；标记完成时间，由 _sweep_runs 延迟清扫
            entry["done_at"] = time.time()

    thread = threading.Thread(target=worker, name=f"run-{run_id}", daemon=True)
    _runs[run_id]["thread"] = thread
    thread.start()


_RUN_RETENTION_S = 600.0


def _sweep_runs() -> None:
    """清扫已完成且超过保留期的运行条目，防止 _runs 无界增长。"""
    now = time.time()
    with _runs_lock:
        stale = [rid for rid, entry in _runs.items()
                 if entry.get("done_at") and now - entry["done_at"] > _RUN_RETENTION_S]
        for rid in stale:
            del _runs[rid]


def _register_run(run_id: str) -> dict:
    _sweep_runs()
    emitter = EventEmitter()
    interrupt_event = threading.Event()
    hitl = WebHITLChannel(emitter, interrupt_event=interrupt_event)
    entry = {"emitter": emitter, "hitl": hitl, "interrupt": interrupt_event, "thread": None}
    with _runs_lock:
        _runs[run_id] = entry
    return entry


@app.post("/api/chat")
def start_chat(req: ChatRequest) -> dict:
    run_id = uuid.uuid4().hex[:12]
    entry = _register_run(run_id)
    tracer = Tracer(run_id)
    _trace_store.add(tracer)

    _launch(run_id, lambda: _agent.handle(
        UserInput(text=req.text, session_id=req.session_id),
        emitter=entry["emitter"], tracer=tracer, hitl=entry["hitl"],
        run_id=run_id, interrupt_event=entry["interrupt"],
    ))
    return {"run_id": run_id}


@app.post("/api/runs/{run_id}/interrupt")
def interrupt_run(run_id: str) -> dict:
    """请求中断：引擎在下一个安全点（迭代边界/卡点等待）挂起并持久化现场。"""
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(404, "run 不存在")
    run["interrupt"].set()
    return {"ok": True}


class ResumeRequest(BaseModel):
    supplement: str | None = None    # 恢复时附带的用户补充指示


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, req: ResumeRequest) -> dict:
    state = _agent.state_store.load(run_id)
    if state is None or state.status != "suspended":
        raise HTTPException(404, "运行不存在或不处于挂起状态")

    entry = _register_run(run_id)  # 新事件流替换旧的，前端重新订阅
    tracer = _trace_store.get(run_id) or Tracer(run_id)
    _trace_store.add(tracer)

    _launch(run_id, lambda: _agent.resume(
        run_id, supplement=req.supplement,
        emitter=entry["emitter"], tracer=tracer, hitl=entry["hitl"],
        interrupt_event=entry["interrupt"],
    ))
    return {"run_id": run_id}


@app.get("/api/runs/suspended")
def list_suspended() -> list[dict]:
    return _agent.state_store.list_suspended()


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
