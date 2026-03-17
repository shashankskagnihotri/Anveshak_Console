from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from threading import Lock, Thread
import time
from typing import Any

from ..api_calls import APICallManager
from ..config import RuntimeConfig
from ..events import RunHandle
from ..file_parsers import detect_media_kind, extract_text_from_path, parse_document_for_chat
from ..modeling.qwen_runner import QwenRunner
from ..model_catalog import get_model_profile
from ..retrieval.active_search import ActiveSearchOrchestrator, SearchPlan
from ..retrieval.embeddings import QwenEmbeddingModel
from ..retrieval.memory import ConversationMemory
from ..retrieval.web import WebIndexer, should_use_web
from ..retrieval.workspace import WorkspaceIndex
from ..run_logging import RunLogger
from ..runtime import RuntimeManager
from ..schema import Attachment, ChatMessage, ChatSession, RetrievedChunk
from ..utils import compact_whitespace
from ..utils import utc_now_iso


class ChatService:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.runtime = RuntimeManager(config)
        self.embedder = QwenEmbeddingModel(config)
        self.workspace_index = WorkspaceIndex(config, self.embedder)
        self.memory = ConversationMemory(config.memory_dir, self.embedder)
        self.web_indexer = WebIndexer(config, self.embedder)
        self.active_search = ActiveSearchOrchestrator(self.web_indexer)
        self.runner = QwenRunner(config)
        self.api_calls = APICallManager(config)
        self.sessions: dict[str, ChatSession] = {}
        self.runs: dict[str, RunHandle] = {}
        self.run_loggers: dict[str, RunLogger] = {}
        self._lock = Lock()
        if self.config.prepare_runtime_on_start:
            self.runtime.start_async()
        Thread(target=self._idle_unload_loop, daemon=True).start()

    def create_session(self) -> ChatSession:
        session = ChatSession()
        self.sessions[session.session_id] = session
        self._save_session(session)
        return session

    def get_or_create_session(self, session_id: str | None = None) -> ChatSession:
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
        target_dir = self.config.uploads_dir / session_id
        target_dir.mkdir(parents=True, exist_ok=True)
        attachments: list[Attachment] = []
        for path in file_paths:
            target_path = target_dir / path.name
            if path.resolve() != target_path.resolve():
                shutil.copy2(path, target_path)
            kind = detect_media_kind(target_path)
            attachments.append(Attachment.from_path(target_path, media_kind=kind, source="upload"))
        return attachments

    def submit_message(
        self,
        *,
        session_id: str,
        text: str,
        attachments: list[Attachment],
    ) -> RunHandle:
        session = self.get_or_create_session(session_id)
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
            },
        )
        worker = Thread(
            target=self._process_message,
            kwargs={
                "session": session,
                "text": text,
                "attachments": attachments,
                "handle": handle,
            },
            daemon=True,
        )
        worker.start()
        return handle

    def runtime_status(self) -> dict[str, Any]:
        return self.runtime.status_dict()

    def wait_for_runtime_status_change(self, last_version: int, timeout: float | None = None) -> dict[str, Any] | None:
        self.runtime.start_async()
        return self.runtime.wait_for_status_change(last_version, timeout=timeout)

    def list_api_calls(self) -> list[dict[str, Any]]:
        return self.api_calls.list_calls()

    def get_api_call(self, call_id: str) -> dict[str, Any]:
        return self.api_calls.get_call(call_id).to_dict()

    def create_api_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.api_calls.create_call(payload).to_dict()

    def update_api_call(self, call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.api_calls.update_call(call_id, payload).to_dict()

    def invoke_api_call(self, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_runtime()
        return self.api_calls.invoke(api_key, payload, runner=self.runner)

    def steer_run(self, run_id: str, note: str) -> None:
        handle = self.runs[run_id]
        if handle.done or handle.phase != "generation":
            raise ValueError("Steering is only available while the model is actively generating an answer.")
        handle.add_steering(note.strip())
        handle.emit("status", phase="steer", text="Steering note received; the current run will adapt.")

    def ensure_runtime(self, handle: RunHandle | None = None) -> None:
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
        if self.runner.model is not None:
            return
        profile = get_model_profile(self.config.model_id)
        label = profile.get("label") or self.config.model_id
        handle.emit(
            "status",
            phase="model-load",
            text=f"Loading {label} into GPU/CPU memory for this run",
        )

    def _process_message(
        self,
        *,
        session: ChatSession,
        text: str,
        attachments: list[Attachment],
        handle: RunHandle,
    ) -> None:
        run_logger = self.run_loggers.get(handle.run_id)
        outcome = "error"
        try:
            self.ensure_runtime(handle)
            if text.strip() == "Obliviate":
                session.clear()
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
            if any(item.media_kind == "document" for item in attachments):
                handle.emit(
                    "status",
                    phase="attachments",
                    text="Parsing attached documents into text and visual context",
                )
            model_attachments, inline_document_chunks, warnings, parsed_document_texts = _prepare_attachment_context(
                attachments,
                profile,
                self.config,
            )
            if warnings:
                handle.emit("warning", text="\n".join(warnings))

            if self.config.enable_workspace_indexing:
                handle.emit("status", phase="workspace", text="Refreshing local-file index")
                changed = self.workspace_index.refresh()
                if changed:
                    handle.emit("status", phase="workspace", text=f"Indexed {len(changed)} changed local files")

            if attachments:
                file_paths = [Path(item.path) for item in attachments if item.media_kind == "document"]
                if file_paths:
                    indexed = self.workspace_index.index_paths(file_paths, parsed_texts=parsed_document_texts)
                    if indexed:
                        handle.emit("status", phase="attachments", text=f"Indexed {len(indexed)} attached documents")

            attempt = 0
            answer_text = ""
            reasoning_text = ""
            citations: list[dict[str, Any]] = []
            while attempt < 3:
                attempt += 1
                if run_logger is not None:
                    run_logger.record_note("attempt_started", {"attempt": attempt})
                steering_notes = handle.consume_steering_notes()
                recent = session.recent_messages(self.config.max_recent_turns)
                memory_chunks = self.memory.retrieve(text, self.config.memory_top_k)
                direct_paths = self.workspace_index.extract_path_mentions(text)
                direct_file_chunks = self.workspace_index.direct_path_context(direct_paths)
                retrieved_file_chunks = self.workspace_index.retrieve(text, self.config.file_top_k)
                file_chunks = _merge_chunks(
                    inline_document_chunks,
                    direct_file_chunks,
                    retrieved_file_chunks,
                    limit=self.config.file_top_k,
                )

                web_chunks = []
                if self.config.enable_web:
                    handle.emit("status", phase="planning", text="Planning active web retrieval")
                    search_plan = _build_search_plan(
                        user_query=text,
                        recent_messages=recent,
                        config=self.config,
                    )
                    if search_plan.enabled:
                        handle.emit(
                            "status",
                            phase="planning",
                            text=f"Web plan: {search_plan.rationale}",
                        )
                        web_chunks, _rounds = self.active_search.run(
                            user_query=text,
                            plan=search_plan,
                            model_runner=self.runner,
                            handle=handle,
                        )

                if handle.has_pending_restart():
                    handle.emit("status", phase="restart", text="Restarting with the new steering note")
                    continue

                self._announce_model_load(handle)
                answer = self.runner.stream_answer(
                    user_query=text,
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
                break

            assistant_message = ChatMessage(
                role="assistant",
                text=answer_text,
                citations=citations,
                reasoning=reasoning_text,
            )
            session.append(assistant_message)
            self._save_session(session)

            try:
                handle.emit("status", phase="memory", text="Compressing the exchange into long-term memory")
                self._announce_model_load(handle)
                note = self.runner.summarize_memory(
                    user_text=text,
                    assistant_text=answer_text,
                    citations=citations,
                    recent_messages=session.recent_messages(self.config.max_recent_turns),
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
            except Exception as exc:
                if run_logger is not None:
                    run_logger.record_note(
                        "memory_compression_failed",
                        {
                            "type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                    )
                handle.emit(
                    "warning",
                    text=f"Memory compression failed for this run, but the answer was preserved: {exc}",
                )

            handle.emit(
                "done",
                answer=answer_text,
                reasoning=reasoning_text,
                citations=citations,
                session_id=session.session_id,
            )
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
        session_path = self.config.session_dir / f"{session.session_id}.json"
        session_path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")

    def _model_is_pinned_by_session(self) -> bool:
        with self._lock:
            if self.sessions:
                return True
            return any(not run.done for run in self.runs.values())

    def _idle_unload_loop(self) -> None:
        while True:
            time.sleep(10)
            try:
                if self._model_is_pinned_by_session():
                    continue
                self.runner.unload_if_idle()
            except Exception:
                continue


def _merge_chunks(*chunk_groups, limit: int) -> list:
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


def _prepare_attachment_context(
    attachments: list[Attachment],
    profile: dict[str, Any],
    config: RuntimeConfig,
) -> tuple[list[Attachment], list[RetrievedChunk], list[str], dict[str, str]]:
    accepted: list[Attachment] = []
    inline_document_chunks: list[RetrievedChunk] = []
    warnings: list[str] = []
    parsed_texts: dict[str, str] = {}

    unsupported_images: list[str] = []
    unsupported_videos: list[str] = []
    unsupported_binary: list[str] = []
    parser_document_names: list[str] = []

    for attachment in attachments:
        if attachment.media_kind == "image":
            if profile.get("supports_images"):
                accepted.append(attachment)
            else:
                unsupported_images.append(attachment.name)
            continue

        if attachment.media_kind == "video":
            if profile.get("supports_video"):
                accepted.append(attachment)
            else:
                unsupported_videos.append(attachment.name)
            continue

        if attachment.media_kind == "document":
            parser_document_names.append(attachment.name)
            text = ""
            extracted_images: list[Attachment] = []
            try:
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


def _build_search_plan(
    *,
    user_query: str,
    recent_messages: list[ChatMessage],
    config: RuntimeConfig,
) -> SearchPlan:
    if config.web_mode == "off":
        return SearchPlan(
            enabled=False,
            rationale="web retrieval is disabled for this runtime",
            search_queries=[],
            max_rounds=0,
        )

    enabled = should_use_web(user_query, config.web_mode)
    if config.web_mode == "always":
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
        "web retrieval was forced by the runtime setting"
        if config.web_mode == "always"
        else "the request mentions current or internet-facing information"
    )
    return SearchPlan(
        enabled=True,
        rationale=rationale,
        search_queries=search_queries or [user_query],
        max_rounds=2,
    )


def _derive_search_queries(user_query: str, recent_messages: list[ChatMessage]) -> list[str]:
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
