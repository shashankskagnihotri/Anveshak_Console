from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import RuntimeConfig
from .schema import Attachment
from .utils import utc_now_iso


@dataclass(slots=True)
class APICallConfig:
    call_id: str
    name: str
    api_key: str
    model_id: str
    system_prompt: str
    input_template: str
    response_instructions: str
    response_mode: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "api_key": self.api_key,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "input_template": self.input_template,
            "response_instructions": self.response_instructions,
            "response_mode": self.response_mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "APICallConfig":
        return cls(**payload)


class APICallManager:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.root = config.api_calls_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def list_calls(self) -> list[dict[str, Any]]:
        calls = [self.get_call(path.stem).to_dict() for path in sorted(self.root.glob("*.json"))]
        return calls

    def create_call(self, payload: dict[str, Any]) -> APICallConfig:
        now = utc_now_iso()
        config = APICallConfig(
            call_id=uuid4().hex,
            name=payload["name"].strip(),
            api_key=self._new_key(),
            model_id=payload.get("model_id") or self.config.model_id,
            system_prompt=payload.get("system_prompt", "").strip(),
            input_template=payload.get("input_template", "{{input}}").strip() or "{{input}}",
            response_instructions=payload.get("response_instructions", "").strip(),
            response_mode=payload.get("response_mode", "text"),
            created_at=now,
            updated_at=now,
        )
        self._save(config)
        return config

    def get_call(self, call_id: str) -> APICallConfig:
        path = self.root / f"{call_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return APICallConfig.from_dict(payload)

    def update_call(self, call_id: str, payload: dict[str, Any]) -> APICallConfig:
        config = self.get_call(call_id)
        config.name = payload.get("name", config.name).strip()
        config.model_id = payload.get("model_id", config.model_id).strip()
        config.system_prompt = payload.get("system_prompt", config.system_prompt)
        config.input_template = payload.get("input_template", config.input_template) or "{{input}}"
        config.response_instructions = payload.get("response_instructions", config.response_instructions)
        config.response_mode = payload.get("response_mode", config.response_mode)
        config.updated_at = utc_now_iso()
        self._save(config)
        return config

    def invoke(self, api_key: str, payload: dict[str, Any], *, runner) -> dict[str, Any]:
        config = self._find_by_api_key(api_key)
        user_input = payload.get("input", "")
        variables = payload.get("variables", {})
        template_context = {"input": user_input, "json": json.dumps(variables, ensure_ascii=False)}
        user_prompt = config.input_template
        for key, value in template_context.items():
            user_prompt = user_prompt.replace(f"{{{{{key}}}}}", str(value))
        if config.response_instructions:
            user_prompt = f"{user_prompt}\n\nResponse requirements:\n{config.response_instructions}"

        text = runner.generate_text(
            system_prompt=config.system_prompt or "You are a configured API call for Anveshak Console.",
            user_prompt=user_prompt,
            attachments=[],
            max_new_tokens=1024,
        )
        return {
            "call_id": config.call_id,
            "name": config.name,
            "model_id": config.model_id,
            "response_mode": config.response_mode,
            "output": text,
        }

    def _find_by_api_key(self, api_key: str) -> APICallConfig:
        for item in self.list_calls():
            if item["api_key"] == api_key:
                return APICallConfig.from_dict(item)
        raise KeyError("Unknown API key")

    def _save(self, config: APICallConfig) -> None:
        path = self.root / f"{config.call_id}.json"
        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def _new_key() -> str:
        return f"qidx_{secrets.token_urlsafe(24)}"
