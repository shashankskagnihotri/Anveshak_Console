"""End-to-end chat orchestration tying together runtime, retrieval, and UI events."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from queue import Empty, Queue
import re
import shutil
from pathlib import Path
from threading import Lock, Thread
import time
from typing import TYPE_CHECKING, Any

from ..api_calls import APICallConfig, APICallManager
from ..config import RuntimeConfig
from ..events import RunHandle
from ..file_parsers import detect_media_kind, extract_text_from_path, parse_document_for_chat, sample_video_frames_for_chat
from ..model_catalog import get_model_profile
from ..run_logging import RunLogger
from ..runtime import RuntimeManager
from ..schema import Attachment, ChatMessage, ChatSession, RetrievedChunk
from ..utils import compact_whitespace, extract_json_object
from ..utils import utc_now_iso

if TYPE_CHECKING:
    from ..modeling.qwen_runner import QwenRunner
    from ..retrieval.active_search import ActiveSearchOrchestrator, SearchPlan
    from ..retrieval.embeddings import QwenEmbeddingModel
    from ..retrieval.memory import ConversationMemory
    from ..retrieval.web import WebIndexer
    from ..retrieval.workspace import WorkspaceIndex
    from ..transcription import WhisperTranscriber


class ChatService:
    """Coordinate sessions, retrieval, model execution, and per-run logging."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.runtime = RuntimeManager(config)
        self.embedder: QwenEmbeddingModel | None = None
        self.workspace_index: WorkspaceIndex | None = None
        self.memory: ConversationMemory | None = None
        self.web_indexer: WebIndexer | None = None
        self.active_search: ActiveSearchOrchestrator | None = None
        self.runner: QwenRunner | None = None
        self.whisper_transcriber: WhisperTranscriber | None = None
        self.api_calls = APICallManager(config)
        self.sessions: dict[str, ChatSession] = {}
        self.runs: dict[str, RunHandle] = {}
        self.run_loggers: dict[str, RunLogger] = {}
        self._lock = Lock()
        self._component_lock = Lock()
        self._prewarm_lock = Lock()
        self._prewarm_started = False
        self._memory_tasks: Queue[dict[str, Any]] = Queue()
        self._memory_idle_grace_seconds = 2.0
        self._workspace_refresh_lock = Lock()
        self._workspace_refresh_requested = False
        self._workspace_refresh_thread: Thread | None = None
        self._whisper_prewarm_lock = Lock()
        self._whisper_prewarm_thread: Thread | None = None
        self._workspace_refresh_status: dict[str, Any] = {
            "enabled": bool(self.config.enable_workspace_indexing),
            "active": False,
            "message": "",
            "detail": "",
            "last_error": "",
            "last_started_at": None,
            "last_completed_at": None,
            "last_duration_seconds": None,
            "last_changed_count": 0,
        }
        if self.config.prepare_runtime_on_start:
            self.runtime.start_async()
            self._start_background_prewarm()
        Thread(target=self._idle_unload_loop, daemon=True).start()
        Thread(target=self._memory_compression_loop, daemon=True).start()
        if self.config.enable_workspace_indexing:
            self.schedule_workspace_refresh(reason="startup")

    def _ensure_components(self) -> None:
        """Instantiate heavy retrieval/model components lazily on first use."""

        with self._component_lock:
            if self.embedder is None:
                from ..retrieval.embeddings import QwenEmbeddingModel

                self.embedder = QwenEmbeddingModel(self.config)
            if self.workspace_index is None:
                from ..retrieval.workspace import WorkspaceIndex

                self.workspace_index = WorkspaceIndex(self.config, self.embedder)
            if self.memory is None:
                from ..retrieval.memory import ConversationMemory

                self.memory = ConversationMemory(self.config.memory_dir, self.embedder)
            if self.web_indexer is None:
                from ..retrieval.web import WebIndexer

                self.web_indexer = WebIndexer(self.config, self.embedder)
            if self.active_search is None:
                from ..retrieval.active_search import ActiveSearchOrchestrator

                self.active_search = ActiveSearchOrchestrator(self.web_indexer)
            if self.runner is None:
                from ..modeling.factory import create_runner

                self.runner = create_runner(self.config)
            if self.whisper_transcriber is None:
                from ..transcription import WhisperTranscriber

                self.whisper_transcriber = WhisperTranscriber(
                    model_name=self.config.whisper_model_name,
                    device=self.config.whisper_device,
                )

    def _ensure_whisper_transcriber(self) -> "WhisperTranscriber":
        """Load the Whisper helper without forcing the full reasoning stack to initialize."""

        with self._component_lock:
            if self.whisper_transcriber is None:
                from ..transcription import WhisperTranscriber

                self.whisper_transcriber = WhisperTranscriber(
                    model_name=self.config.whisper_model_name,
                    device=self.config.whisper_device,
                )
            return self.whisper_transcriber

    def schedule_whisper_prewarm(self) -> bool:
        """Warm Whisper in the background so the first mic transcription starts faster."""

        transcriber = self._ensure_whisper_transcriber()
        if getattr(transcriber, "is_loaded", lambda: False)():
            return False
        with self._whisper_prewarm_lock:
            if self._whisper_prewarm_thread is not None and self._whisper_prewarm_thread.is_alive():
                return False
            self._whisper_prewarm_thread = Thread(target=self._prewarm_whisper_transcriber, daemon=True)
            self._whisper_prewarm_thread.start()
            return True

    def _ensure_workspace_index(self) -> "WorkspaceIndex":
        """Load only the embedding model and workspace index when background indexing needs them."""

        with self._component_lock:
            if self.embedder is None:
                from ..retrieval.embeddings import QwenEmbeddingModel

                self.embedder = QwenEmbeddingModel(self.config)
            if self.workspace_index is None:
                from ..retrieval.workspace import WorkspaceIndex

                self.workspace_index = WorkspaceIndex(self.config, self.embedder)
            return self.workspace_index

    def _start_background_prewarm(self) -> None:
        """Warm the model in the background once runtime assets are ready."""

        with self._prewarm_lock:
            if self._prewarm_started:
                return
            self._prewarm_started = True
        Thread(target=self._prewarm_runner, daemon=True).start()

    def _prewarm_runner(self) -> None:
        """Load the reasoning model asynchronously so the first prompt waits less."""

        try:
            while not self.runtime.wait_until_ready(timeout=0.5):
                continue
            status = self.runtime.status_dict()
            if status.get("phase") in {"error", "auth-required"}:
                return
            self._ensure_components()
            if self.runner is not None:
                self.runner.load()
        except Exception:
            return
        finally:
            with self._prewarm_lock:
                self._prewarm_started = False

    def _start_model_load(self, handle: RunHandle | None = None) -> Thread | None:
        """Kick off model loading early so it can overlap retrieval work."""

        self._ensure_components()
        if self.runner is None or getattr(self.runner, "model", None) is not None:
            return None
        if handle is not None:
            self._announce_model_load(handle)
        worker = Thread(target=self.runner.load, daemon=True)
        worker.start()
        return worker

    def _prewarm_whisper_transcriber(self) -> None:
        """Load Whisper asynchronously without blocking the browser microphone flow."""

        try:
            self._ensure_whisper_transcriber().warmup()
        except Exception:
            return
        finally:
            with self._whisper_prewarm_lock:
                self._whisper_prewarm_thread = None

    def create_session(self) -> ChatSession:
        """Create and persist a new chat session."""

        session = ChatSession()
        self.sessions[session.session_id] = session
        self._save_session(session)
        return session

    def get_or_create_session(self, session_id: str | None = None) -> ChatSession:
        """Load an existing session from disk or create a new one on demand."""

        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        if session_id:
            session_path = self.config.session_dir / f"{session_id}.json"
            if session_path.exists():
                payload = json.loads(session_path.read_text(encoding="utf-8"))
                session = ChatSession.from_dict(payload)
                self.sessions[session.session_id] = session
                return session
        return self.create_session()

    def save_uploads(self, session_id: str, file_paths: list[Path]) -> list[Attachment]:
        """Normalize uploaded files into session-scoped attachment records."""

        target_dir = self.config.uploads_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        attachments: list[Attachment] = []
        for path in file_paths:
            target_path = target_dir / path.name
            if path.resolve() != target_path.resolve():
                shutil.copy2(path, target_path)
            kind = detect_media_kind(target_path)
            source = "microphone" if target_path.name.startswith("microphone-recording-") else "upload"
            attachments.append(Attachment.from_path(target_path, media_kind=kind, source=source))
        return attachments

    def submit_message(
        self,
        *,
        session_id: str,
        text: str,
        attachments: list[Attachment],
        web_mode: str = "auto",
        media_mode: str = "safe",
    ) -> RunHandle:
        """Queue one user message for background processing."""

        session = self.get_or_create_session(session_id)
        normalized_web_mode = _normalize_web_mode(web_mode)
        normalized_media_mode = _normalize_media_mode(media_mode)
        handle = RunHandle(session.session_id)
        with self._lock:
            active_for_session = any(
                existing.session_id == session.session_id and not existing.done for existing in self.runs.values()
            )
            if active_for_session:
                raise ValueError("Finish the current run before sending another prompt.")
            self.runs[handle.run_id] = handle
        run_logger = RunLogger(
            self.config,
            run_id=handle.run_id,
            session_id=session.session_id,
            user_text=text,
            attachments=attachments,
        )
        handle.add_listener(run_logger.record_event)
        self.run_loggers[handle.run_id] = run_logger
        run_logger.record_note(
            "run_queued",
            {
                "attachment_count": len(attachments),
                "web_mode": normalized_web_mode,
                "media_mode": normalized_media_mode,
            },
        )
        worker = Thread(
            target=self._process_message,
            kwargs={
                "session": session,
                "text": text,
                "attachments": attachments,
                "handle": handle,
                "web_mode": normalized_web_mode,
                "media_mode": normalized_media_mode,
            },
            daemon=True,
        )
        worker.start()
        return handle

    def runtime_status(self) -> dict[str, Any]:
        """Return the current checkpoint/runtime preparation status."""

        return self.runtime.status_dict()

    def workspace_index_status(self) -> dict[str, Any]:
        """Expose whether the ambient workspace index is currently refreshing in the background."""

        with self._workspace_refresh_lock:
            return dict(self._workspace_refresh_status)

    def schedule_workspace_refresh(self, *, reason: str = "background") -> bool:
        """Request one non-blocking workspace refresh if ambient indexing is enabled."""

        if not self.config.enable_workspace_indexing:
            return False
        with self._workspace_refresh_lock:
            self._workspace_refresh_requested = True
            if self._workspace_refresh_thread is not None and self._workspace_refresh_thread.is_alive():
                return False
            self._workspace_refresh_thread = Thread(
                target=self._workspace_refresh_loop,
                kwargs={"reason": reason},
                daemon=True,
            )
            self._workspace_refresh_thread.start()
            return True

    def wait_until_model_ready(self) -> None:
        """Block until the configured reasoning model has finished loading."""

        self.ensure_runtime()
        self._ensure_components()
        if self.runner is not None:
            self.runner.load()

    def configure_huggingface_token(self, token: str) -> dict[str, Any]:
        """Accept a Hugging Face token from the UI and retry runtime preparation."""

        payload = self.runtime.configure_huggingface_token(token)
        self._start_background_prewarm()
        return payload

    def transcribe_microphone_recording(self, audio_path: Path) -> dict[str, str]:
        """Transcribe one browser-recorded audio clip into editable chat text."""

        if detect_media_kind(audio_path) != "audio":
            raise ValueError("Microphone transcription expects an audio file.")
        transcriber = self._ensure_whisper_transcriber()
        transcript = compact_whitespace(transcriber.transcribe(audio_path))
        if not transcript:
            raise RuntimeError(f"No transcript was produced for {audio_path.name}.")
        return {
            "attachment_name": audio_path.name,
            "backend": "Whisper",
            "text": transcript,
        }

    def wait_for_runtime_status_change(self, last_version: int, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until a newer runtime status payload is available."""

        self.runtime.start_async()
        return self.runtime.wait_for_status_change(last_version, timeout=timeout)

    def _workspace_refresh_loop(self, *, reason: str) -> None:
        """Continuously process queued workspace refresh requests without blocking chat runs."""

        while True:
            with self._workspace_refresh_lock:
                if not self._workspace_refresh_requested:
                    self._workspace_refresh_thread = None
                    return
                self._workspace_refresh_requested = False
                started_at = utc_now_iso()
                self._workspace_refresh_status.update(
                    {
                        "enabled": True,
                        "active": True,
                        "message": "Refreshing local-file index",
                        "detail": "Anveshak is updating workspace retrieval in the background. You can keep chatting while this runs.",
                        "last_error": "",
                        "last_started_at": started_at,
                    }
                )

            started = time.perf_counter()
            try:
                index = self._ensure_workspace_index()
                changed = index.refresh()
                duration = round(time.perf_counter() - started, 2)
                with self._workspace_refresh_lock:
                    self._workspace_refresh_status.update(
                        {
                            "enabled": True,
                            "active": False,
                            "message": "Local-file index ready",
                            "detail": (
                                f"Indexed {len(changed)} changed local files in the background."
                                if changed
                                else "Local-file index was already up to date."
                            ),
                            "last_error": "",
                            "last_completed_at": utc_now_iso(),
                            "last_duration_seconds": duration,
                            "last_changed_count": len(changed),
                        }
                    )
            except Exception as exc:
                with self._workspace_refresh_lock:
                    self._workspace_refresh_status.update(
                        {
                            "enabled": True,
                            "active": False,
                            "message": "Local-file index refresh failed",
                            "detail": str(exc),
                            "last_error": str(exc),
                            "last_completed_at": utc_now_iso(),
                        }
                    )

    def list_api_calls(self) -> list[dict[str, Any]]:
        """List every saved API-call preset."""

        return self.api_calls.list_calls()

    def get_api_call(self, call_id: str) -> dict[str, Any]:
        """Load one saved API-call preset."""

        return self.api_calls.get_call(call_id).to_dict()

    def create_api_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a new API-call preset from browser form data."""

        created = self.api_calls.create_call(payload)
        self.prepare_api_call(created.call_id)
        return created.to_dict()

    def update_api_call(self, call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Update an existing API-call preset."""

        updated = self.api_calls.update_call(call_id, payload)
        self.prepare_api_call(updated.call_id)
        return updated.to_dict()

    def delete_api_call(self, call_id: str) -> dict[str, Any]:
        """Delete one saved API-call preset and any remembered API-only transcript."""

        removed = self.api_calls.delete_call(call_id)
        session_path = self._api_call_session_path(call_id)
        if session_path.exists():
            session_path.unlink()
        return removed.to_dict()

    def prepare_api_call(self, call_id: str) -> None:
        """Warm the runtime and model path for a newly saved API call in the background."""

        Thread(target=self._prepare_api_call_worker, args=(call_id,), daemon=True).start()

    def invoke_api_call(self, call_ref: str, payload: dict[str, Any], *, api_key: str | None = None) -> dict[str, Any]:
        """Invoke one stored API-call configuration through the local runner."""

        self.ensure_runtime()
        self._ensure_components()
        api_call = self.api_calls.resolve_call(call_ref, api_key=api_key)
        rendered_prompt = self.api_calls.render_prompt(api_call, payload)
        user_query = compact_whitespace(str(payload.get("input", ""))) or compact_whitespace(rendered_prompt)

        recent_messages: list[ChatMessage] = []
        api_session: ChatSession | None = None
        if api_call.instance_mode == "remember":
            api_session = self._load_api_call_session(api_call)
            recent_messages = api_session.recent_messages(self.config.max_recent_turns)

        memory_chunks = []
        if api_call.use_user_context and self.memory is not None:
            memory_chunks = self.memory.retrieve(user_query, self.config.memory_top_k)

        web_chunks = self._run_api_call_web_search(api_call=api_call, user_query=user_query, recent_messages=recent_messages)
        prompt = _compose_api_invoke_prompt(
            rendered_prompt=rendered_prompt,
            response_instructions=api_call.response_instructions,
            recent_messages=recent_messages,
            memory_chunks=memory_chunks,
            web_chunks=web_chunks,
            response_mode=api_call.response_mode,
        )
        system_prompt = api_call.system_prompt or "You are a configured API call for Anveshak Console."

        if api_call.response_mode == "json":
            output: dict[str, Any] | str = _generate_json_response(
                runner=self.runner,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_new_tokens=min(self.config.max_new_tokens, 1536),
            )
        else:
            output = self.runner.generate_text(
                system_prompt=system_prompt,
                user_prompt=prompt,
                attachments=[],
                max_new_tokens=min(self.config.max_new_tokens, 1536),
            )

        if api_session is not None:
            api_session.append(ChatMessage(role="user", text=rendered_prompt))
            assistant_text = json.dumps(output, ensure_ascii=False, indent=2) if isinstance(output, dict) else str(output)
            api_session.append(
                ChatMessage(
                    role="assistant",
                    text=assistant_text,
                    citations=[item.citation() for item in _merge_chunks(web_chunks, memory_chunks, limit=24)],
                )
            )
            self._save_api_call_session(api_call.call_id, api_session)

        self.api_calls.record_invocation(api_call.call_id)
        return {
            "call_id": api_call.call_id,
            "name": api_call.name,
            "configured_model_id": api_call.model_id,
            "runtime_model_id": self.config.model_id,
            "configured_embedding_model_id": api_call.embedding_model_id,
            "runtime_embedding_model_id": self.config.embedding_model_id,
            "response_mode": api_call.response_mode,
            "web_mode": api_call.web_mode,
            "use_user_context": api_call.use_user_context,
            "instance_mode": api_call.instance_mode,
            "citations": [item.citation() for item in _merge_chunks(web_chunks, memory_chunks, limit=24)],
            "output": output,
        }

    def steer_run(self, run_id: str, note: str) -> None:
        """Queue a steering note for an actively generating run."""

        handle = self.runs[run_id]
        if handle.done or handle.phase != "generation":
            raise ValueError("Steering is only available while the model is actively generating an answer.")
        handle.add_steering(note.strip())
        handle.emit("status", phase="steer", text="Steering note received; the current run will adapt.")

    def ensure_runtime(self, handle: RunHandle | None = None) -> None:
        """Wait until model assets are present, streaming progress if requested."""

        self.runtime.start_async()
        while not self.runtime.wait_until_ready(timeout=0.5):
            if handle is not None:
                status = self.runtime.status_dict()
                handle.emit(
                    "status",
                    phase="runtime",
                    text=status["message"],
                    progress=status["progress"],
                )
        status = self.runtime.status_dict()
        if status["phase"] == "error":
            raise RuntimeError(status["error"] or "Runtime preparation failed")

    def _announce_model_load(self, handle: RunHandle) -> None:
        """Emit a user-facing status update before the first model load in a run."""

        if self.runner is not None and self.runner.model is not None:
            return
        profile = get_model_profile(self.config.model_id)
        label = profile.get("label") or self.config.model_id
        handle.emit(
            "status",
            phase="model-load",
            text=(
                f"Connecting to the dedicated {label} server backend"
                if self.config.kimi_server_url and profile.get("preferred_runtime_backend") == "kimi_server"
                else f"Loading {label} into GPU/CPU memory for this run"
            ),
        )

    def _process_message(
        self,
        *,
        session: ChatSession,
        text: str,
        attachments: list[Attachment],
        handle: RunHandle,
        web_mode: str,
        media_mode: str,
    ) -> None:
        """Resolve one user message into retrieval context, answer, and memory update."""

        run_logger = self.run_loggers.get(handle.run_id)
        outcome = "error"
        try:
            self.ensure_runtime(handle)
            self._ensure_components()
            model_load_worker = self._start_model_load(handle)
            if text.strip() == "Obliviate":
                session.clear()
                if self.memory is not None:
                    self.memory.reset()
                self._save_session(session)
                handle.emit("status", phase="memory", text="Long-term memory erased. Starting fresh.")
                handle.emit("done", answer="Memory reset complete.", reasoning="", citations=[])
                outcome = "done"
                return

            user_message = ChatMessage(role="user", text=text, attachments=attachments)
            session.append(user_message)
            self._save_session(session)

            profile = get_model_profile(self.config.model_id)
            has_document_attachments = any(item.media_kind == "document" for item in attachments)
            if has_document_attachments:
                handle.emit(
                    "status",
                    phase="attachments",
                    text="Parsing attached documents into text and visual context",
                )
            if self.config.enable_workspace_indexing:
                self.schedule_workspace_refresh(reason="prompt")

            with ThreadPoolExecutor(max_workers=1) as executor:
                attachment_future = executor.submit(
                    _prepare_attachment_context,
                    attachments,
                    profile,
                    self.config,
                )
                model_attachments, inline_document_chunks, warnings, parsed_document_texts = attachment_future.result()

            if warnings:
                handle.emit("warning", text="\n".join(warnings))

            effective_text, model_attachments, audio_transcriptions = self._apply_audio_transcriptions(
                user_text=text,
                model_attachments=model_attachments,
                profile=profile,
                handle=handle,
                run_logger=run_logger,
            )
            if audio_transcriptions:
                session.messages[-1].text = effective_text
                if session.title == "New Session" and effective_text.strip():
                    session.title = effective_text.strip().splitlines()[0][:80]
                self._save_session(session)

            if attachments:
                file_paths = [Path(item.path) for item in attachments if item.media_kind == "document"]
                if file_paths and self.workspace_index is not None:
                    indexed = self.workspace_index.index_paths(file_paths, parsed_texts=parsed_document_texts)
                    if indexed:
                        handle.emit("status", phase="attachments", text=f"Indexed {len(indexed)} attached documents")

            attempt = 0
            answer_text = ""
            reasoning_text = ""
            citations: list[dict[str, Any]] = []
            media_results_payload: list[dict[str, Any]] = []
            media_warning = ""
            while attempt < 3:
                attempt += 1
                if run_logger is not None:
                    run_logger.record_note("attempt_started", {"attempt": attempt})
                # Each restart pulls in any newly queued steering notes before retrieval and generation.
                steering_notes = handle.consume_steering_notes()
                recent = _prior_recent_messages_for_current_turn(session, self.config.max_recent_turns)
                memory_chunks = self.memory.retrieve(effective_text, self.config.memory_top_k) if self.memory is not None else []
                direct_paths = self.workspace_index.extract_path_mentions(effective_text) if self.workspace_index is not None else []
                direct_file_chunks = self.workspace_index.direct_path_context(direct_paths) if self.workspace_index is not None else []
                retrieved_file_chunks = self.workspace_index.retrieve(effective_text, self.config.file_top_k) if self.workspace_index is not None else []
                file_chunks = _merge_chunks(
                    inline_document_chunks,
                    direct_file_chunks,
                    retrieved_file_chunks,
                    limit=self.config.file_top_k,
                )

                web_chunks = []
                search_plan = None
                search_plan = _build_search_plan(
                    user_query=effective_text,
                    recent_messages=recent,
                    config=self.config,
                    web_mode_override=web_mode,
                )
                if search_plan.enabled:
                    handle.emit("status", phase="planning", text="Planning active web retrieval")
                    handle.emit(
                        "status",
                        phase="planning",
                        text=f"Web plan: {search_plan.rationale}",
                    )
                    web_chunks, _rounds = self.active_search.run(
                        user_query=effective_text,
                        plan=search_plan,
                        model_runner=self.runner,
                        handle=handle,
                    )

                if handle.has_pending_restart():
                    handle.emit("status", phase="restart", text="Restarting with the new steering note")
                    continue

                if model_load_worker is not None and model_load_worker.is_alive():
                    handle.emit("status", phase="model-load", text="Finalizing model load while retrieval results are ready")
                answer = self.runner.stream_answer(
                    user_query=effective_text,
                    attachments=model_attachments,
                    memory_chunks=memory_chunks,
                    file_chunks=file_chunks,
                    web_chunks=web_chunks,
                    recent_messages=recent,
                    steering_notes=steering_notes,
                    handle=handle,
                )

                if handle.has_pending_restart():
                    handle.emit("status", phase="restart", text="Restarting generation with the steering note")
                    continue

                answer_text = answer.answer
                reasoning_text = answer.reasoning
                citations = [item.citation() for item in _merge_chunks(file_chunks, web_chunks, memory_chunks, limit=24)]
                if search_plan is not None and search_plan.enabled and self.web_indexer is not None:
                    handle.emit(
                        "status",
                        phase="web-media",
                        text=(
                            "Curating safe web media previews"
                            if media_mode == "safe"
                            else "Preparing unrestricted web media previews"
                        ),
                    )
                    try:
                        media_items = self.web_indexer.search_media(
                            search_plan.search_queries or [effective_text],
                            profile=profile,
                            runner=self.runner,
                            media_mode=media_mode,
                        )
                    except Exception as exc:
                        handle.emit("warning", text=f"Web media previews could not be prepared: {exc}")
                        media_items = []
                    media_results_payload = [item.to_dict() for item in media_items]
                    media_warning = _media_warning_for_mode(media_mode) if media_mode == "unrestricted" else ""
                break

            assistant_message = ChatMessage(
                role="assistant",
                text=answer_text,
                citations=citations,
                reasoning=reasoning_text,
            )
            session.append(assistant_message)
            self._save_session(session)

            done_payload = {
                "answer": answer_text,
                "reasoning": reasoning_text,
                "citations": citations,
                "media_results": media_results_payload,
                "media_mode": media_mode,
                "media_warning": media_warning,
                "session_id": session.session_id,
            }
            handle.emit("done", **done_payload)
            self._queue_memory_compression(
                run_id=handle.run_id,
                user_text=effective_text,
                assistant_text=answer_text,
                citations=citations,
                recent_messages=session.recent_messages(self.config.max_recent_turns),
                run_logger=run_logger,
            )
            self.run_loggers.pop(handle.run_id, None)
            run_logger = None
            outcome = "done"
        except Exception as exc:  # pragma: no cover - surfaced to UI
            if run_logger is not None:
                run_logger.record_exception(exc)
            handle.emit("error", message=str(exc))
        finally:
            if run_logger is not None:
                run_logger.close(outcome=outcome)
                self.run_loggers.pop(handle.run_id, None)

    def _save_session(self, session: ChatSession) -> None:
        """Persist the full session transcript to disk."""

        session_path = self.config.session_dir / f"{session.session_id}.json"
        session_path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def _model_is_pinned_by_session(self) -> bool:
        """Keep the reasoning model warm while sessions or active runs still exist."""

        with self._lock:
            if self.sessions:
                return True
            return any(not run.done for run in self.runs.values())

    def _has_active_runs(self) -> bool:
        """Report whether any foreground run is still active."""

        with self._lock:
            return any(not run.done for run in self.runs.values())

    def _idle_unload_loop(self) -> None:
        """Background safety loop for unloading the model only when no session pins it."""

        while True:
            time.sleep(10)
            try:
                if self._model_is_pinned_by_session():
                    continue
                if self.runner is not None:
                    self.runner.unload_if_idle()
            except Exception:
                continue

    def _queue_memory_compression(
        self,
        *,
        run_id: str,
        user_text: str,
        assistant_text: str,
        citations: list[dict[str, Any]],
        recent_messages: list[ChatMessage],
        run_logger: RunLogger | None,
    ) -> None:
        """Schedule one long-term memory write outside the foreground answer path."""

        if run_logger is not None:
            run_logger.record_note(
                "memory_compression_queued",
                {
                    "run_id": run_id,
                    "recent_message_count": len(recent_messages),
                },
            )
        self._memory_tasks.put(
            {
                "run_id": run_id,
                "user_text": user_text,
                "assistant_text": assistant_text,
                "citations": citations,
                "recent_messages": recent_messages,
                "run_logger": run_logger,
            }
        )

    def _memory_compression_loop(self) -> None:
        """Process queued long-term memory writes once the foreground chat is idle."""

        while True:
            try:
                task = self._memory_tasks.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._wait_for_memory_idle_window()
                self._compress_memory_task(task)
            except Exception:
                run_logger = task.get("run_logger")
                if run_logger is not None:
                    run_logger.record_exception(RuntimeError("background memory compression failed"))
                    run_logger.close(outcome="done")
            finally:
                self._memory_tasks.task_done()

    def _wait_for_memory_idle_window(self) -> None:
        """Delay background memory writes until the chat has been idle briefly."""

        while True:
            while self._has_active_runs():
                time.sleep(0.1)
            grace_deadline = time.time() + self._memory_idle_grace_seconds
            while time.time() < grace_deadline:
                if self._has_active_runs():
                    break
                time.sleep(0.1)
            else:
                return

    def _compress_memory_task(self, task: dict[str, Any]) -> None:
        """Write one queued exchange into long-term memory and close its run log."""

        run_logger = task.get("run_logger")
        try:
            if self.runner is None or self.memory is None:
                if run_logger is not None:
                    run_logger.record_note(
                        "memory_compression_skipped",
                        {"reason": "runner or memory store is unavailable"},
                    )
                return
            if run_logger is not None:
                run_logger.record_note("memory_compression_started", {"run_id": task["run_id"]})
            note = self.runner.summarize_memory(
                user_text=task["user_text"],
                assistant_text=task["assistant_text"],
                citations=task["citations"],
                recent_messages=task["recent_messages"],
            )
            summary = note.get("summary", "")
            keywords = note.get("keywords", [])
            facts = note.get("facts", [])
            open_loops = note.get("open_loops", [])
            payload = {
                "note_id": f"M-{utc_now_iso()}",
                "label": "Conversation memory",
                "created_at": utc_now_iso(),
                "keywords": keywords,
                "facts": facts,
                "open_loops": open_loops,
            }
            self.memory.add_note(summary, payload)
            if run_logger is not None:
                run_logger.record_note(
                    "memory_compression_finished",
                    {
                        "summary": summary,
                        "keywords": keywords,
                        "facts": facts,
                        "open_loops": open_loops,
                    },
                )
        except Exception as exc:
            if run_logger is not None:
                run_logger.record_note(
                    "memory_compression_failed",
                    {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
        finally:
            if run_logger is not None:
                run_logger.close(outcome="done")

    def _prepare_api_call_worker(self, call_id: str) -> None:
        """Warm the runtime and model path for one saved API call."""

        try:
            self.runtime.start_async()
            self._start_background_prewarm()
            self.ensure_runtime()
            self._ensure_components()
            if self.runner is not None:
                self.runner.load()
        except Exception:
            return

    def _api_call_session_path(self, call_id: str) -> Path:
        """Return the transcript path used for one stateful API call."""

        return self.config.api_session_dir / f"{call_id}.json"

    def _load_api_call_session(self, api_call: APICallConfig) -> ChatSession:
        """Load or initialize the remembered transcript for one API call."""

        path = self._api_call_session_path(api_call.call_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return ChatSession.from_dict(payload)
        return ChatSession(session_id=f"api-{api_call.call_id}", title=api_call.name)

    def _save_api_call_session(self, call_id: str, session: ChatSession) -> None:
        """Persist the remembered transcript for one API call."""

        path = self._api_call_session_path(call_id)
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def _run_api_call_web_search(
        self,
        *,
        api_call: APICallConfig,
        user_query: str,
        recent_messages: list[ChatMessage],
    ) -> list[RetrievedChunk]:
        """Execute live web retrieval for one API invocation when its policy enables it."""

        if self.active_search is None or self.runner is None:
            return []
        plan = _build_search_plan(
            user_query=user_query,
            recent_messages=recent_messages,
            config=self.config,
            web_mode_override=api_call.web_mode,
        )
        if not plan.enabled:
            return []
        handle = RunHandle(f"api-{api_call.call_id}")
        chunks, _rounds = self.active_search.run(
            user_query=user_query,
            plan=plan,
            model_runner=self.runner,
            handle=handle,
        )
        return chunks

    def _apply_audio_transcriptions(
        self,
        *,
        user_text: str,
        model_attachments: list[Attachment],
        profile: dict[str, Any],
        handle: RunHandle,
        run_logger: RunLogger | None,
    ) -> tuple[str, list[Attachment], list[dict[str, str]]]:
        """Transcribe supported audio inputs before the main reasoning step."""

        audio_attachments = [item for item in model_attachments if item.media_kind == "audio"]
        if not audio_attachments:
            return user_text, model_attachments, []
        transcriptions: list[dict[str, str]] = []
        for index, attachment in enumerate(audio_attachments, start=1):
            backend = self._select_audio_transcription_backend(attachment=attachment, profile=profile)
            backend_label = "Gemma" if backend == "gemma" else "Whisper"
            handle.emit(
                "status",
                phase="transcription",
                text=f"Transcribing audio clip {index}/{len(audio_attachments)} with {backend_label}: {attachment.name}",
            )
            transcript = self._transcribe_audio_attachment(attachment=attachment, backend=backend)
            if not transcript:
                handle.emit("warning", text=f"No transcript was produced for {attachment.name}.")
                continue
            payload = {"attachment_name": attachment.name, "backend": backend_label, "text": transcript}
            transcriptions.append(payload)
            handle.emit("transcription", **payload)

        if run_logger is not None and transcriptions:
            run_logger.record_note("audio_transcribed", {"items": transcriptions})

        remaining_attachments = [item for item in model_attachments if item.media_kind != "audio"]
        effective_text = _merge_audio_transcriptions_into_user_text(user_text, transcriptions)
        return effective_text, remaining_attachments, transcriptions

    def _select_audio_transcription_backend(self, *, attachment: Attachment, profile: dict[str, Any]) -> str:
        """Choose whether one audio clip should be transcribed by Gemma or Whisper."""

        if attachment.source == "microphone":
            return "whisper"
        if profile.get("supports_audio"):
            return "gemma"
        return "whisper"

    def _transcribe_audio_attachment(self, *, attachment: Attachment, backend: str) -> str:
        """Route one audio attachment through the requested transcription engine."""

        if backend == "gemma":
            if self.runner is None or not hasattr(self.runner, "transcribe_audio"):
                raise RuntimeError(
                    f'{self.config.model_id} is configured for Gemma audio, but the current runner cannot transcribe audio inputs.'
                )
            return compact_whitespace(self.runner.transcribe_audio(attachment))
        if self.whisper_transcriber is None:
            raise RuntimeError("Whisper is not available for audio transcription.")
        return compact_whitespace(self.whisper_transcriber.transcribe(Path(attachment.path)))


def _merge_chunks(*chunk_groups, limit: int) -> list:
    """Merge retrieved chunks while preserving order and removing duplicates."""

    merged: list = []
    seen: set[str] = set()
    for group in chunk_groups:
        for chunk in group:
            key = chunk.source_id + "|" + chunk.label + "|" + chunk.text[:120]
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= limit:
                return merged
    return merged


def _prior_recent_messages_for_current_turn(session: ChatSession, count: int) -> list[ChatMessage]:
    """Return prior conversation only, excluding the newest user turn being answered now."""

    if count <= 0:
        return []
    if not session.messages:
        return []
    return session.messages[:-1][-count:]


def _prepare_attachment_context(
    attachments: list[Attachment],
    profile: dict[str, Any],
    config: RuntimeConfig,
) -> tuple[list[Attachment], list[RetrievedChunk], list[str], dict[str, str]]:
    """Parse attachments into model-native payloads plus text retrieval context."""

    accepted: list[Attachment] = []
    inline_document_chunks: list[RetrievedChunk] = []
    warnings: list[str] = []
    parsed_texts: dict[str, str] = {}

    unsupported_images: list[str] = []
    unsupported_videos: list[str] = []
    unsupported_binary: list[str] = []
    parser_document_names: list[str] = []
    sampled_video_names: list[str] = []
    failed_video_fallbacks: list[str] = []

    for attachment in attachments:
        if attachment.media_kind == "image":
            if profile.get("supports_images"):
                accepted.append(attachment)
            else:
                unsupported_images.append(attachment.name)
            continue

        if attachment.media_kind == "audio":
            accepted.append(attachment)
            continue

        if attachment.media_kind == "video":
            if profile.get("supports_video"):
                accepted.append(attachment)
            elif profile.get("supports_images"):
                try:
                    frame_paths = sample_video_frames_for_chat(
                        Path(attachment.path),
                        cache_root=config.cache_dir / "video_frame_payloads",
                    )
                except Exception:
                    frame_paths = []

                if frame_paths:
                    sampled_video_names.append(f"{attachment.name} ({len(frame_paths)} sampled frames)")
                    for frame_path in frame_paths:
                        accepted.append(
                            Attachment.from_path(
                                frame_path,
                                media_kind="image",
                                source=f"video-fallback:{attachment.name}",
                            )
                        )
                else:
                    unsupported_videos.append(attachment.name)
                    failed_video_fallbacks.append(attachment.name)
            else:
                unsupported_videos.append(attachment.name)
            continue

        if attachment.media_kind == "document":
            parser_document_names.append(attachment.name)
            text = ""
            extracted_images: list[Attachment] = []
            try:
                # PDFs are treated as both text and visuals so multimodal backends can still see figures.
                parsed = parse_document_for_chat(
                    Path(attachment.path),
                    cache_root=config.cache_dir / "document_payloads",
                    max_images=config.max_pdf_inline_images,
                    max_rendered_pages=config.max_pdf_preview_pages,
                )
                parsed_texts[str(Path(attachment.path).resolve())] = parsed.text
                text = compact_whitespace(parsed.text)[: config.max_inline_file_chars]
                if profile.get("supports_images"):
                    for image_path in parsed.image_paths:
                        extracted_images.append(
                            Attachment.from_path(image_path, media_kind="image", source=f"parsed:{attachment.name}")
                        )
            except Exception:
                # If the richer parser fails, fall back to text extraction instead of dropping the document.
                try:
                    raw_text = extract_text_from_path(Path(attachment.path))
                    parsed_texts[str(Path(attachment.path).resolve())] = raw_text
                    text = compact_whitespace(raw_text)[: config.max_inline_file_chars]
                except Exception:
                    text = ""

            if text:
                inline_document_chunks.append(
                    RetrievedChunk(
                        source_id=f"F-INLINE-{len(inline_document_chunks)+1}",
                        source_kind="file",
                        label=attachment.name,
                        text=text,
                        score=1.0,
                        metadata={"source_path": attachment.path, "label": attachment.name},
                    )
                )
            if extracted_images:
                accepted.extend(extracted_images)
            continue

        unsupported_binary.append(attachment.name)

    model_label = profile.get("label") or profile.get("model_id", "This model")
    if unsupported_images:
        warnings.append(
            f'Warning: the chosen model "{model_label}" does not support image attachments natively for: {", ".join(unsupported_images)}.'
        )
    if unsupported_videos:
        warnings.append(
            f'Warning: the chosen model "{model_label}" does not support video attachments natively for: {", ".join(unsupported_videos)}.'
        )
    if sampled_video_names:
        warnings.append(
            f'Warning: the chosen model "{model_label}" does not support video attachments natively, so Anveshak sampled fallback image frames from: {", ".join(sampled_video_names)}. Important moments may fall between sampled frames, so video-based answers can be incomplete or unreliable.'
        )
    if failed_video_fallbacks:
        warnings.append(
            f'Warning: Anveshak could not extract fallback frames from: {", ".join(failed_video_fallbacks)}. Those video attachments were not sent to the model.'
        )
    if parser_document_names and profile.get("kind") == "text-generation":
        warnings.append(
            f'Note: the chosen model "{model_label}" is text-only, so these files were parsed into text instead of being sent as native attachments: {", ".join(parser_document_names)}.'
        )
    elif parser_document_names:
        warnings.append(
            f'Note: the chosen model "{model_label}" does not natively support document attachments, so these files were parsed into text context with extracted visuals when available: {", ".join(parser_document_names)}.'
        )
    if unsupported_binary:
        warnings.append(
            f'Warning: the chosen model "{model_label}" cannot use these unsupported file types: {", ".join(unsupported_binary)}.'
        )

    return accepted, inline_document_chunks, warnings, parsed_texts


def _normalize_web_mode(value: str | None) -> str:
    """Normalize one per-run web policy value."""

    normalized = compact_whitespace(value or "auto").lower()
    if normalized not in {"off", "auto", "always"}:
        raise ValueError(f"Unsupported web mode: {value}")
    return normalized


def _normalize_media_mode(value: str | None) -> str:
    """Normalize one per-run remote-media safety policy value."""

    normalized = compact_whitespace(value or "safe").lower()
    if normalized not in {"safe", "unrestricted"}:
        raise ValueError(f"Unsupported media mode: {value}")
    return normalized


def _media_warning_for_mode(media_mode: str) -> str:
    """Return the user-facing warning copy for unrestricted remote media previews."""

    if _normalize_media_mode(media_mode) != "unrestricted":
        return ""
    return (
        "Unrestricted web media is enabled. NSFW or graphic images/videos can appear in the results. "
        "User discretion is advised."
    )


def _merge_audio_transcriptions_into_user_text(user_text: str, transcriptions: list[dict[str, str]]) -> str:
    """Blend typed input and audio transcripts into one task string for retrieval and reasoning."""

    cleaned_user_text = user_text.strip()
    transcript_blocks = [
        f'Audio transcription from {item["attachment_name"]} ({item.get("backend", "transcriber")}):\n{item["text"]}'
        for item in transcriptions
        if item.get("text")
    ]
    if not transcript_blocks:
        return cleaned_user_text
    if not cleaned_user_text:
        return "\n\n".join(transcript_blocks).strip()
    return f"{cleaned_user_text}\n\n" + "\n\n".join(transcript_blocks)


def _build_search_plan(
    *,
    user_query: str,
    recent_messages: list[ChatMessage],
    config: RuntimeConfig,
    web_mode_override: str | None = None,
) -> SearchPlan:
    """Convert runtime policy and query hints into a structured search plan."""

    from ..retrieval.active_search import SearchPlan
    from ..retrieval.web import should_use_web

    effective_web_mode = _normalize_web_mode(web_mode_override or config.web_mode)

    if effective_web_mode == "off":
        return SearchPlan(
            enabled=False,
            rationale="web retrieval was disabled for this message",
            search_queries=[],
            max_rounds=0,
        )

    enabled = should_use_web(user_query, effective_web_mode)
    if effective_web_mode == "always":
        enabled = True
    if not enabled:
        return SearchPlan(
            enabled=False,
            rationale="the request looks self-contained, so local context stays primary",
            search_queries=[],
            max_rounds=0,
        )

    search_queries = _derive_search_queries(user_query, recent_messages)
    rationale = (
        "web retrieval was forced by the message setting"
        if effective_web_mode == "always"
        else "the request mentions current or internet-facing information"
    )
    return SearchPlan(
        enabled=True,
        rationale=rationale,
        search_queries=search_queries or [user_query],
        max_rounds=2,
    )


def _derive_search_queries(user_query: str, recent_messages: list[ChatMessage]) -> list[str]:
    """Generate a few focused search queries without requiring a model call."""

    cleaned_query = compact_whitespace(user_query)
    if not cleaned_query:
        return []

    queries = [cleaned_query[:240]]
    recent_user_turns = [
        compact_whitespace(message.text)
        for message in recent_messages[-3:]
        if message.role == "user" and compact_whitespace(message.text)
    ]
    if recent_user_turns:
        combined = compact_whitespace(" ".join(recent_user_turns[-2:]))
        if combined and combined != cleaned_query:
            queries.append(combined[:240])

    keywords = re.findall(r"[A-Za-z0-9][A-Za-z0-9+_.-]{2,}", cleaned_query)
    if keywords:
        keyword_query = " ".join(keywords[:12])
        if keyword_query and keyword_query not in queries:
            queries.append(keyword_query[:240])

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        lowered = query.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(query)
        if len(deduped) >= 3:
            break
    return deduped


def _compose_api_invoke_prompt(
    *,
    rendered_prompt: str,
    response_instructions: str,
    recent_messages: list[ChatMessage],
    memory_chunks: list[RetrievedChunk],
    web_chunks: list[RetrievedChunk],
    response_mode: str,
) -> str:
    """Build the final prompt for one API invocation."""

    sections = [rendered_prompt.strip()]

    if recent_messages:
        history_lines = []
        for message in recent_messages[-6:]:
            label = "User" if message.role == "user" else "Assistant"
            history_lines.append(f"{label}: {compact_whitespace(message.text)[:1200]}")
        if history_lines:
            sections.append("Remembered prior API-call exchanges:\n" + "\n".join(history_lines))

    if memory_chunks:
        memory_lines = [f"- {chunk.label}: {compact_whitespace(chunk.text)[:600]}" for chunk in memory_chunks[:6]]
        sections.append("Relevant long-term user context from Anveshak:\n" + "\n".join(memory_lines))

    if web_chunks:
        web_lines = []
        for chunk in web_chunks[:8]:
            url = chunk.metadata.get("url", "")
            prefix = f"{chunk.label} ({url})" if url else chunk.label
            web_lines.append(f"- {prefix}: {compact_whitespace(chunk.text)[:700]}")
        sections.append("Live web evidence retrieved for this API call:\n" + "\n".join(web_lines))

    if response_instructions.strip():
        sections.append("Response requirements:\n" + response_instructions.strip())

    if response_mode == "json":
        sections.append("Return exactly one valid JSON object and no extra prose.")

    return "\n\n".join(section for section in sections if section)


def _generate_json_response(*, runner, system_prompt: str, user_prompt: str, max_new_tokens: int) -> dict[str, Any]:
    """Generate JSON while preserving the saved API-call system prompt."""

    json_system_prompt = (
        f"{system_prompt.strip()}\n\n"
        "Return only valid JSON. No markdown fences, no prose before or after the JSON."
    ).strip()
    raw = runner.generate_text(
        system_prompt=json_system_prompt,
        user_prompt=user_prompt,
        attachments=[],
        max_new_tokens=max_new_tokens,
    )
    try:
        return extract_json_object(raw)
    except Exception:
        repair_prompt = f"""
The previous attempt was not valid JSON. Repair it and return exactly one valid JSON object.

Original prompt:
{user_prompt}

Previous output:
{raw}
""".strip()
        repaired = runner.generate_text(
            system_prompt=json_system_prompt,
            user_prompt=repair_prompt,
            attachments=[],
            max_new_tokens=max_new_tokens,
        )
        return extract_json_object(repaired)
