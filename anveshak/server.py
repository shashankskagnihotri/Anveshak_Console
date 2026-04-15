"""FastAPI application assembly for the browser UI and API endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chat.service import ChatService
from .config import RuntimeConfig

_SHUTDOWN_SENTINEL = object()


async def _await_thread_call(func, *args):
    """Run a blocking callback in the default executor unless shutdown has already begun."""

    try:
        return await asyncio.to_thread(func, *args)
    except RuntimeError as exc:
        if "cannot schedule new futures after shutdown" in str(exc):
            return _SHUTDOWN_SENTINEL
        raise


def build_app(config: RuntimeConfig, service: ChatService) -> FastAPI:
    """Construct the FastAPI app with chat, runtime, and API-call endpoints."""

    app = FastAPI(title="Anveshak Console", version="0.1.0")

    @app.middleware("http")
    async def disable_browser_cache_for_ui(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        if request.url.path == "/":
            response.headers["Clear-Site-Data"] = '"cache"'
        return response

    app.mount("/static", StaticFiles(directory=str(config.static_dir)), name="static")
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        static_version = _static_asset_version(config.static_dir)
        html = (config.static_dir / "index.html").read_text(encoding="utf-8").replace(
            "__STATIC_VERSION__",
            static_version,
        )
        return HTMLResponse(html)

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/runtime/status")
    async def runtime_status() -> JSONResponse:
        return JSONResponse(service.runtime_status())

    @app.get("/api/workspace-index/status")
    async def workspace_index_status() -> JSONResponse:
        return JSONResponse(service.workspace_index_status())

    @app.post("/api/runtime/whisper-warmup")
    async def whisper_warmup() -> JSONResponse:
        return JSONResponse({"scheduled": service.schedule_whisper_prewarm()})

    @app.post("/api/runtime/huggingface-token")
    async def configure_huggingface_token(payload: dict) -> JSONResponse:
        token = str(payload.get("token") or "")
        try:
            return JSONResponse(service.configure_huggingface_token(token))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runtime/events")
    async def runtime_events() -> StreamingResponse:
        async def event_stream():
            last_version = -1
            try:
                while True:
                    payload = await _await_thread_call(service.wait_for_runtime_status_change, last_version, 30.0)
                    if payload is _SHUTDOWN_SENTINEL:
                        break
                    if payload is None:
                        yield ": keep-alive\n\n"
                        continue
                    last_version = int(payload["version"])
                    yield f"event: status\ndata: {json.dumps(payload)}\n\n"
            except asyncio.CancelledError:
                return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/sessions")
    async def create_session() -> JSONResponse:
        session = service.create_session()
        return JSONResponse(session.to_dict())

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> JSONResponse:
        session = service.get_or_create_session(session_id)
        return JSONResponse(session.to_dict())

    @app.post("/api/sessions/{session_id}/microphone-transcription")
    async def transcribe_microphone_recording(session_id: str, file: UploadFile = File(...)) -> JSONResponse:
        session = service.get_or_create_session(session_id)
        upload_paths = await _persist_uploads(config, session.session_id, [file])
        try:
            if not upload_paths:
                raise HTTPException(status_code=400, detail="No microphone recording was uploaded.")
            payload = await _await_thread_call(service.transcribe_microphone_recording, upload_paths[0])
            if payload is _SHUTDOWN_SENTINEL:
                raise HTTPException(status_code=503, detail="The server is shutting down; transcription could not finish.")
            return JSONResponse(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _cleanup_saved_uploads(upload_paths)

    @app.post("/api/sessions/{session_id}/messages")
    async def submit_message(
        session_id: str,
        text: str = Form(""),
        web_mode: Literal["off", "auto", "always"] = Form("auto"),
        media_mode: Literal["safe", "unrestricted"] = Form("safe"),
        files: list[UploadFile] | None = File(default=None),
    ) -> JSONResponse:
        submitted_text = text or ""
        submitted_files = files or []
        if not submitted_text.strip() and not submitted_files:
            raise HTTPException(status_code=400, detail="Provide text or at least one file before sending a message.")
        session = service.get_or_create_session(session_id)
        upload_paths = await _persist_uploads(config, session.session_id, submitted_files)
        attachments = service.save_uploads(session.session_id, upload_paths)
        try:
            run_handle = service.submit_message(
                session_id=session.session_id,
                text=submitted_text,
                attachments=attachments,
                web_mode=web_mode,
                media_mode=media_mode,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 409 if "Finish the current run" in detail else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        return JSONResponse({"run_id": run_handle.run_id, "session_id": session.session_id})

    @app.post("/api/runs/{run_id}/steer")
    async def steer_run(run_id: str, text: str = Form(...)) -> JSONResponse:
        if run_id not in service.runs:
            raise HTTPException(status_code=404, detail="Unknown run")
        try:
            service.steer_run(run_id, text)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse({"status": "queued"})

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        if run_id not in service.runs:
            raise HTTPException(status_code=404, detail="Unknown run")
        handle = service.runs[run_id]

        async def event_stream():
            try:
                while True:
                    event = await _await_thread_call(handle.next_event, 0.5)
                    if event is _SHUTDOWN_SENTINEL:
                        break
                    if event is None:
                        if handle.done:
                            break
                        continue
                    yield event.to_sse()
                    if event.event_type in {"done", "error"}:
                        break
            except asyncio.CancelledError:
                return

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/api-calls")
    async def list_api_calls() -> JSONResponse:
        return JSONResponse({"items": service.list_api_calls()})

    @app.get("/api/api-calls/{call_id}")
    async def get_api_call(call_id: str) -> JSONResponse:
        return JSONResponse(service.get_api_call(call_id))

    @app.post("/api/api-calls")
    async def create_api_call(payload: dict) -> JSONResponse:
        try:
            return JSONResponse(service.create_api_call(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/api-calls/{call_id}")
    async def update_api_call(call_id: str, payload: dict) -> JSONResponse:
        try:
            return JSONResponse(service.update_api_call(call_id, payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/api-calls/{call_id}")
    async def delete_api_call(call_id: str) -> JSONResponse:
        try:
            return JSONResponse(service.delete_api_call(call_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown API call") from exc

    @app.post("/v1/api-calls/{call_ref}/invoke")
    async def invoke_api_call(call_ref: str, payload: dict, request: Request) -> JSONResponse:
        bearer_key = _extract_api_key(request)
        try:
            return JSONResponse(service.invoke_api_call(call_ref, payload, api_key=bearer_key))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


async def _persist_uploads(config: RuntimeConfig, session_id: str, files: list[UploadFile]) -> list[Path]:
    """Store uploads in the cache area before they are normalized into attachments."""

    tmp_root = config.cache_dir / "incoming" / session_id
    tmp_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for upload in files:
        target = tmp_root / (upload.filename or "upload.bin")
        content = await upload.read()
        target.write_bytes(content)
        saved.append(target)
    return saved


def _cleanup_saved_uploads(paths: list[Path]) -> None:
    """Best-effort cleanup for temporary upload files that should not persist."""

    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            continue


def _extract_api_key(request: Request) -> str | None:
    """Read an API key from standard bearer or x-api-key request headers."""

    authorization = request.headers.get("authorization", "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip() or None
    x_api_key = request.headers.get("x-api-key", "").strip()
    return x_api_key or None


def _static_asset_version(static_dir: Path) -> str:
    """Return a simple revision token so browser reloads pick up fresh UI assets."""

    version = 0
    for name in ("index.html", "app.js", "styles.css", "anveshak_logo2.png", "anveshak_favicon.svg"):
        path = static_dir / name
        if not path.exists():
            continue
        try:
            version = max(version, path.stat().st_mtime_ns)
        except Exception:
            continue
    return str(version or 1)
