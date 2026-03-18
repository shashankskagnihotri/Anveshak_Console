"""Structured per-run logging for reasoning, status, and lifecycle events."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from .config import RuntimeConfig
from .events import RunEvent
from .schema import Attachment
from .utils import utc_now_iso


class RunLogger:
    """Write one JSONL log file per run without duplicating streamed answer tokens."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        run_id: str,
        session_id: str,
        user_text: str,
        attachments: list[Attachment],
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.session_id = session_id
        self.started_at = perf_counter()
        self._lock = Lock()
        self._closed = False
        timestamp = utc_now_iso().replace(":", "-")
        self.path = config.logs_dir / f"{timestamp}_{run_id}.jsonl"
        self._write(
            "run_started",
            {
                "run_id": run_id,
                "session_id": session_id,
                "model_id": config.model_id,
                "embedding_model_id": config.embedding_model_id,
                "mode": config.mode,
                "web_mode": config.web_mode,
                "seed": config.seed,
                "user_text": user_text,
                "attachments": [self._serialize_attachment(item) for item in attachments],
            },
        )

    def record_event(self, event: RunEvent) -> None:
        """Map streamed run events into the normalized log vocabulary."""

        if self._closed:
            return
        if event.event_type == "token":
            return
        if event.event_type == "status":
            self._write("THINKING", event.payload)
            return
        if event.event_type == "reasoning":
            self._write("REASONING", event.payload)
            return
        self._write(event.event_type, event.payload)

    def record_note(self, event_type: str, payload: dict[str, Any]) -> None:
        """Write an internal note that did not originate from a streamed event."""

        self._write(event_type, payload)

    def record_exception(self, exc: BaseException) -> None:
        """Persist exception metadata without crashing the logger itself."""

        self._write(
            "exception",
            {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        )

    def close(self, *, outcome: str) -> None:
        """Mark the run log as finished once the run reaches a terminal state."""

        if self._closed:
            return
        self._closed = True
        self._write(
            "run_finished",
            {
                "outcome": outcome,
                "elapsed_seconds": round(perf_counter() - self.started_at, 3),
            },
        )

    def _write(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one structured record to the run's JSONL log file."""

        record = {
            "created_at": utc_now_iso(),
            "elapsed_seconds": round(perf_counter() - self.started_at, 3),
            "event_type": event_type,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=self._json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _serialize_attachment(attachment: Attachment) -> dict[str, Any]:
        """Convert an attachment into the compact metadata stored in logs."""

        return {
            "name": attachment.name,
            "path": attachment.path,
            "media_kind": attachment.media_kind,
            "source": attachment.source,
            "size_bytes": attachment.size_bytes,
        }

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Fallback serializer for paths and other non-JSON-native objects."""

        if isinstance(value, Path):
            return str(value)
        return str(value)
