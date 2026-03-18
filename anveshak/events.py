"""Run-event primitives shared by the web UI, terminal UI, and loggers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from queue import Queue
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class RunEvent:
    """A single streamed lifecycle event emitted by a run."""

    event_type: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_sse(self) -> str:
        """Serialize the event in Server-Sent Events format for browser clients."""

        import json

        return f"event: {self.event_type}\ndata: {json.dumps(asdict(self), ensure_ascii=False)}\n\n"


class RunHandle:
    """Coordinate one in-flight user request and expose its streamed events."""

    def __init__(self, session_id: str, *, listeners: list[Callable[[RunEvent], None]] | None = None) -> None:
        self.run_id = uuid4().hex
        self.session_id = session_id
        self._events: Queue[RunEvent] = Queue()
        self._listeners = list(listeners or [])
        self._lock = Lock()
        self._done = False
        self._cancelled = False
        self._restart_requested = False
        self._steering_notes: list[str] = []
        self.phase = "created"

    def emit(self, event_type: str, **payload: Any) -> None:
        """Push a new run event to listeners and downstream consumers."""

        if event_type == "status" and payload.get("phase"):
            self.phase = str(payload["phase"])
        if event_type in {"done", "error"}:
            self._done = True
            self.phase = event_type
        event = RunEvent(event_type=event_type, payload=payload)
        self._events.put(event)
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                continue

    def add_listener(self, listener: Callable[[RunEvent], None]) -> None:
        """Register a callback that should receive every emitted event."""

        self._listeners.append(listener)

    def next_event(self, timeout: float | None = None) -> RunEvent | None:
        """Read the next queued event, returning ``None`` on timeout."""

        try:
            return self._events.get(timeout=timeout)
        except Exception:
            return None

    @property
    def done(self) -> bool:
        """Expose whether the run has reached a terminal event."""

        return self._done

    def cancel(self) -> None:
        """Request immediate cancellation of the active generation loop."""

        with self._lock:
            self._cancelled = True
            self._restart_requested = False

    def should_stop_generation(self) -> bool:
        """Tell the model streamer whether it should halt early."""

        with self._lock:
            return self._cancelled or self._restart_requested

    def add_steering(self, note: str) -> None:
        """Queue a steering note and request a restart with the new guidance."""

        with self._lock:
            self._steering_notes.append(note)
            self._restart_requested = True

    def consume_steering_notes(self) -> list[str]:
        """Drain all queued steering notes for the next generation attempt."""

        with self._lock:
            notes = list(self._steering_notes)
            self._steering_notes.clear()
            self._restart_requested = False
            return notes

    def has_pending_restart(self) -> bool:
        """Report whether a steering-triggered restart has been requested."""

        with self._lock:
            return self._restart_requested
