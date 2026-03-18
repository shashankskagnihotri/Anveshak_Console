"""Core data structures shared across chat, retrieval, and API layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    """Return a timezone-aware ISO8601 timestamp in UTC."""

    return datetime.now(tz=UTC).isoformat()


@dataclass(slots=True)
class Attachment:
    """A file the user uploaded or referenced during a run."""

    path: str
    name: str
    media_kind: str
    source: str
    size_bytes: int | None = None

    @classmethod
    def from_path(cls, path: Path, *, media_kind: str, source: str) -> "Attachment":
        """Build an attachment record from a local file path."""

        stat = path.stat()
        return cls(
            path=str(path.resolve()),
            name=path.name,
            media_kind=media_kind,
            source=source,
            size_bytes=stat.st_size,
        )


@dataclass(slots=True)
class ChatMessage:
    """A persisted message in a chat session transcript."""

    role: str
    text: str
    created_at: str = field(default_factory=utc_now_iso)
    attachments: list[Attachment] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message for session persistence."""

        return {
            "role": self.role,
            "text": self.text,
            "created_at": self.created_at,
            "attachments": [asdict(item) for item in self.attachments],
            "citations": self.citations,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatMessage":
        """Restore a chat message from JSON-friendly session data."""

        return cls(
            role=payload["role"],
            text=payload["text"],
            created_at=payload.get("created_at", utc_now_iso()),
            attachments=[Attachment(**item) for item in payload.get("attachments", [])],
            citations=list(payload.get("citations", [])),
            reasoning=payload.get("reasoning"),
        )


@dataclass(slots=True)
class ChatSession:
    """A conversation transcript and its derived metadata."""

    session_id: str = field(default_factory=lambda: uuid4().hex)
    title: str = "New Session"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    messages: list[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        """Append a new message and keep the session title fresh."""

        self.messages.append(message)
        self.updated_at = utc_now_iso()
        if self.title == "New Session" and message.role == "user" and message.text.strip():
            self.title = message.text.strip().splitlines()[0][:80]

    def recent_messages(self, count: int) -> list[ChatMessage]:
        """Return the newest ``count`` messages from the session."""

        if count <= 0:
            return []
        return self.messages[-count:]

    def clear(self) -> None:
        """Erase the transcript contents while keeping the session shell."""

        self.messages.clear()
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the session for on-disk persistence."""

        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [message.to_dict() for message in self.messages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatSession":
        """Rebuild a session and its messages from persisted JSON data."""

        session = cls(
            session_id=payload["session_id"],
            title=payload.get("title", "New Session"),
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )
        session.messages = [ChatMessage.from_dict(item) for item in payload.get("messages", [])]
        return session


@dataclass(slots=True)
class RetrievedChunk:
    """A ranked retrieval hit from files, memory, or the live web."""

    source_id: str
    source_kind: str
    label: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> dict[str, Any]:
        """Return the lightweight citation payload exposed to the UI."""

        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "label": self.label,
            "metadata": self.metadata,
        }
