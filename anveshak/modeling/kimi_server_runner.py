"""OpenAI-compatible served-model backend used by Kimi and other server-hosted models."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..events import RunHandle
from ..schema import Attachment, ChatMessage, RetrievedChunk
from .qwen_runner import GeneratedAnswer, QwenRunner, _ThinkParser


class KimiServerRunner(QwenRunner):
    """Run a server-hosted reasoning model instead of the generic HF in-process path."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.base_url = self._normalized_base_url(config.kimi_server_url)
        self.server_model_name = config.kimi_server_model or self.profile.get("server_model_name") or "kimi-k2"
        self.server_api_key = config.kimi_server_api_key or "anveshak-local"
        self.request_timeout = float(max(config.kimi_server_timeout_seconds, 1))
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.request_timeout, connect=min(self.request_timeout, 5.0)),
        )

    def load(self) -> None:
        """Wait for the configured served-model endpoint to become reachable."""

        with self._load_lock:
            if self.model is not None:
                self.last_used_at = time.time()
                return
            deadline = time.time() + self.request_timeout
            last_error = ""
            while time.time() < deadline:
                try:
                    response = self._client.get("/models", headers=self._headers())
                    response.raise_for_status()
                    self.model = {"backend": "openai_compatible_server", "model": self.server_model_name}
                    self.last_used_at = time.time()
                    return
                except Exception as exc:
                    last_error = str(exc)
                    time.sleep(0.5)
            raise RuntimeError(
                "Configured served-model backend is not reachable. "
                f"Checked {self.base_url}/models for model `{self.server_model_name}`. "
                f"Last error: {last_error or 'unknown error'}"
            )

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachments: list[Attachment],
        max_new_tokens: int,
    ) -> str:
        """Send one non-streaming chat completion request to the served model."""

        with self._inference_lock:
            self.load()
            messages = self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt, attachments=attachments)
            payload = self._chat_payload(messages, max_new_tokens=max_new_tokens, stream=False)
            response = self._client.post("/chat/completions", json=payload, headers=self._headers())
            response.raise_for_status()
            data = response.json()
            self.last_used_at = time.time()
            return self._message_text(data).strip()

    def stream_answer(
        self,
        *,
        user_query: str,
        attachments: list[Attachment],
        memory_chunks: list[RetrievedChunk],
        file_chunks: list[RetrievedChunk],
        web_chunks: list[RetrievedChunk],
        recent_messages: list[ChatMessage],
        steering_notes: list[str],
        handle: RunHandle,
    ) -> GeneratedAnswer:
        """Stream the main answer from the served backend."""

        with self._inference_lock:
            self.load()
            prompt = self._compose_answer_prompt(
                user_query=user_query,
                attachments=attachments,
                memory_chunks=memory_chunks,
                file_chunks=file_chunks,
                web_chunks=web_chunks,
                recent_messages=recent_messages,
                steering_notes=steering_notes,
            )
            messages = self._build_messages(system_prompt=self.SYSTEM_PROMPT, user_prompt=prompt, attachments=attachments)
            payload = self._chat_payload(messages, max_new_tokens=self.config.max_new_tokens, stream=True)

            parser = _ThinkParser()
            handle.phase = "generating"
            handle.emit("status", phase="generation", text="Generating answer with retrieved context")

            with self._client.stream("POST", "/chat/completions", json=payload, headers=self._headers()) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if handle.should_stop_generation():
                        break
                    if not line:
                        continue
                    piece = self._stream_piece(line)
                    if not piece:
                        continue
                    for event_kind, chunk in parser.feed(piece):
                        if not chunk:
                            continue
                        if event_kind == "reasoning":
                            handle.emit("reasoning", text=chunk)
                        else:
                            handle.emit("token", text=chunk)

            self.last_used_at = time.time()
            return parser.finalize()

    def unload(self) -> None:
        """Release the local runner state without stopping the external server."""

        self.model = None
        self.model_config = None
        self.processor = None
        self.tokenizer = None

    @property
    def SYSTEM_PROMPT(self) -> str:
        from .qwen_runner import SYSTEM_PROMPT

        return SYSTEM_PROMPT

    def _headers(self) -> dict[str, str]:
        """Build OpenAI-compatible HTTP headers."""

        return {
            "Authorization": f"Bearer {self.server_api_key}",
            "Content-Type": "application/json",
        }

    def _chat_payload(self, messages: list[dict[str, Any]], *, max_new_tokens: int, stream: bool) -> dict[str, Any]:
        """Translate internal Anveshak chat messages into an OpenAI-compatible payload."""

        return {
            "model": self.server_model_name,
            "messages": [self._server_message(message) for message in messages],
            "stream": stream,
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
        }

    def _server_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize one internal message into the text-only Kimi server schema."""

        return {
            "role": message["role"],
            "content": self._coerce_text(message.get("content", "")),
        }

    def _message_text(self, payload: dict[str, Any]) -> str:
        """Extract assistant text from a non-streaming OpenAI-compatible response."""

        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return self._coerce_text(message.get("content", ""))

    def _stream_piece(self, line: str) -> str:
        """Parse one SSE line from a streaming chat completion response."""

        stripped = line.strip()
        if not stripped.startswith("data:"):
            return ""
        data = stripped[5:].strip()
        if not data or data == "[DONE]":
            return ""
        payload = json.loads(data)
        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        if "content" in delta:
            return self._coerce_text(delta.get("content", ""))
        return self._coerce_text(choices[0].get("text", ""))

    def _coerce_text(self, content: Any) -> str:
        """Flatten OpenAI-style message content into plain text."""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                    continue
                text_payload = item.get("text")
                if isinstance(text_payload, str):
                    parts.append(text_payload)
            return "".join(parts)
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return str(content or "")

    @staticmethod
    def _normalized_base_url(url: str | None) -> str:
        """Normalize a user-provided server URL so requests consistently hit `/v1`."""

        if not url:
            raise RuntimeError(
                "OpenAI-compatible served backend requires a server URL. "
                "Pass --kimi-server-url or set ANVESHAK_KIMI_SERVER_URL."
            )
        stripped = url.rstrip("/")
        if stripped.endswith("/v1"):
            return stripped
        return f"{stripped}/v1"
