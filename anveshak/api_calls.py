"""Saved API-call presets and invocation helpers for Anveshak."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import RuntimeConfig
from .utils import compact_whitespace, utc_now_iso

DEFAULT_INPUT_TEMPLATE = "User input:\n{{input}}\n\nVariables:\n{{json}}"
DEFAULT_RESPONSE_MODE = "text"
DEFAULT_WEB_MODE = "auto"
DEFAULT_INSTANCE_MODE = "independent"


@dataclass(slots=True)
class APICallConfig:
    """Serializable configuration for one reusable API-style workflow."""

    call_id: str
    name: str
    api_key: str
    model_id: str
    embedding_model_id: str
    system_prompt: str
    input_template: str
    response_instructions: str
    response_mode: str
    web_mode: str
    use_user_context: bool
    instance_mode: str
    created_at: str
    updated_at: str
    last_invoked_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize a stored API-call definition to plain JSON data."""

        return {
            "call_id": self.call_id,
            "name": self.name,
            "api_key": self.api_key,
            "model_id": self.model_id,
            "embedding_model_id": self.embedding_model_id,
            "system_prompt": self.system_prompt,
            "input_template": self.input_template,
            "response_instructions": self.response_instructions,
            "response_mode": self.response_mode,
            "web_mode": self.web_mode,
            "use_user_context": self.use_user_context,
            "instance_mode": self.instance_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_invoked_at": self.last_invoked_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "APICallConfig":
        """Restore an API-call definition from persisted JSON with backward compatibility."""

        return cls(
            call_id=payload["call_id"],
            name=payload.get("name", "").strip() or "Untitled API Call",
            api_key=payload["api_key"],
            model_id=payload.get("model_id", ""),
            embedding_model_id=payload.get("embedding_model_id", ""),
            system_prompt=payload.get("system_prompt", ""),
            input_template=payload.get("input_template", DEFAULT_INPUT_TEMPLATE) or DEFAULT_INPUT_TEMPLATE,
            response_instructions=payload.get("response_instructions", ""),
            response_mode=_normalize_response_mode(payload.get("response_mode", DEFAULT_RESPONSE_MODE)),
            web_mode=_normalize_web_mode(payload.get("web_mode", DEFAULT_WEB_MODE)),
            use_user_context=bool(payload.get("use_user_context", False)),
            instance_mode=_normalize_instance_mode(payload.get("instance_mode", DEFAULT_INSTANCE_MODE)),
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", payload.get("created_at", utc_now_iso())),
            last_invoked_at=payload.get("last_invoked_at"),
        )


class APICallManager:
    """Create, persist, update, delete, and resolve saved API-call presets."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = config.api_calls_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def list_calls(self) -> list[dict[str, Any]]:
        """Return every stored API-call configuration as JSON-friendly payloads."""

        calls: list[dict[str, Any]] = []
        for path in self.root.glob("*.json"):
            try:
                calls.append(self.get_call(path.stem).to_dict())
            except Exception:
                continue
        calls.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return calls

    def create_call(self, payload: dict[str, Any]) -> APICallConfig:
        """Create and persist a new API-call configuration."""

        now = utc_now_iso()
        config = APICallConfig(
            call_id=uuid4().hex,
            name=_normalize_name(payload.get("name", "")),
            api_key=self._new_key(),
            model_id=self.config.model_id,
            embedding_model_id=self.config.embedding_model_id,
            system_prompt=payload.get("system_prompt", "").strip(),
            input_template=(payload.get("input_template", DEFAULT_INPUT_TEMPLATE) or DEFAULT_INPUT_TEMPLATE).strip(),
            response_instructions=payload.get("response_instructions", "").strip(),
            response_mode=_normalize_response_mode(payload.get("response_mode", DEFAULT_RESPONSE_MODE)),
            web_mode=_normalize_web_mode(payload.get("web_mode", DEFAULT_WEB_MODE)),
            use_user_context=bool(payload.get("use_user_context", False)),
            instance_mode=_normalize_instance_mode(payload.get("instance_mode", DEFAULT_INSTANCE_MODE)),
            created_at=now,
            updated_at=now,
        )
        self._save(config)
        return config

    def get_call(self, call_id: str) -> APICallConfig:
        """Load one stored API-call configuration by id."""

        path = self._path_for(call_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return APICallConfig.from_dict(payload)

    def update_call(self, call_id: str, payload: dict[str, Any]) -> APICallConfig:
        """Update a saved API-call definition in place."""

        config = self.get_call(call_id)
        config.name = _normalize_name(payload.get("name", config.name))
        config.model_id = self.config.model_id
        config.embedding_model_id = self.config.embedding_model_id
        config.system_prompt = payload.get("system_prompt", config.system_prompt).strip()
        config.input_template = (payload.get("input_template", config.input_template) or DEFAULT_INPUT_TEMPLATE).strip()
        config.response_instructions = payload.get("response_instructions", config.response_instructions).strip()
        config.response_mode = _normalize_response_mode(payload.get("response_mode", config.response_mode))
        config.web_mode = _normalize_web_mode(payload.get("web_mode", config.web_mode))
        config.use_user_context = bool(payload.get("use_user_context", config.use_user_context))
        config.instance_mode = _normalize_instance_mode(payload.get("instance_mode", config.instance_mode))
        config.updated_at = utc_now_iso()
        self._save(config)
        return config

    def delete_call(self, call_id: str) -> APICallConfig:
        """Delete one stored API-call definition and return the removed payload."""

        config = self.get_call(call_id)
        self._path_for(call_id).unlink()
        return config

    def resolve_call(self, call_ref: str, *, api_key: str | None = None) -> APICallConfig:
        """Resolve a saved call definition from either a call id or an API key."""

        if api_key:
            try:
                config = self.get_call(call_ref)
            except FileNotFoundError:
                config = self._find_by_api_key(call_ref)
            if not secrets.compare_digest(config.api_key, api_key):
                raise PermissionError("The provided API key does not match this API call.")
            return config

        try:
            return self.get_call(call_ref)
        except FileNotFoundError:
            return self._find_by_api_key(call_ref)

    def render_prompt(self, config: APICallConfig, payload: dict[str, Any]) -> str:
        """Expand the saved template into the concrete user prompt for one invocation."""

        user_input = payload.get("input", "")
        variables = payload.get("variables", {})
        if not isinstance(variables, dict):
            variables = {"value": variables}
        template_context = {
            "input": str(user_input),
            "json": json.dumps(variables, ensure_ascii=False),
        }
        rendered = config.input_template
        for key, value in template_context.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    def record_invocation(self, call_id: str) -> APICallConfig:
        """Update the last-invoked timestamp after a successful API invocation."""

        config = self.get_call(call_id)
        config.last_invoked_at = utc_now_iso()
        self._save(config)
        return config

    def _find_by_api_key(self, api_key: str) -> APICallConfig:
        """Resolve a saved call definition from its generated API key."""

        for item in self.list_calls():
            if secrets.compare_digest(item["api_key"], api_key):
                return APICallConfig.from_dict(item)
        raise KeyError("Unknown API key")

    def _save(self, config: APICallConfig) -> None:
        """Persist one API-call definition to disk."""

        self._path_for(config.call_id).write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    def _path_for(self, call_id: str) -> Path:
        """Return the on-disk path for one API-call definition."""

        return self.root / f"{call_id}.json"

    @staticmethod
    def _new_key() -> str:
        """Generate a user-facing key for invoking stored API presets."""

        return f"qidx_{secrets.token_urlsafe(24)}"


def _normalize_name(value: str) -> str:
    """Validate and normalize one API-call display name."""

    name = compact_whitespace(value)
    if not name:
        raise ValueError("API call name cannot be empty.")
    return name[:120]


def _normalize_response_mode(value: str) -> str:
    """Restrict response mode to the supported options."""

    normalized = compact_whitespace(value or DEFAULT_RESPONSE_MODE).lower()
    if normalized not in {"text", "json"}:
        raise ValueError(f"Unsupported response mode: {value}")
    return normalized


def _normalize_web_mode(value: str) -> str:
    """Restrict API-call web policy to the supported options."""

    normalized = compact_whitespace(value or DEFAULT_WEB_MODE).lower()
    if normalized not in {"off", "auto", "always"}:
        raise ValueError(f"Unsupported web mode: {value}")
    return normalized


def _normalize_instance_mode(value: str) -> str:
    """Restrict API-call invocation memory behavior to the supported options."""

    normalized = compact_whitespace(value or DEFAULT_INSTANCE_MODE).lower()
    if normalized not in {"independent", "remember"}:
        raise ValueError(f"Unsupported instance mode: {value}")
    return normalized
