from __future__ import annotations

from pathlib import Path

import pytest

from anveshak.chat.service import ChatService
from anveshak.config import RuntimeConfig
from anveshak.retrieval.web import WebIndexer, should_use_web
from anveshak.schema import RetrievedChunk, WebMediaResult


class FakeEmbedder:
    def encode_documents(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            score = float(sum(ord(char) for char in text) % 1000)
            rows.append([score, float(len(text) or 1)])
        return np.asarray(rows, dtype="float32")

    def encode_query(self, text):
        import numpy as np

        score = float(sum(ord(char) for char in text) % 1000)
        return np.asarray([score, float(len(text) or 1)], dtype="float32")


class FakeAnswerRunner:
    model = None

    def load(self):
        return None

    def stream_answer(
        self,
        *,
        user_query,
        attachments,
        memory_chunks,
        file_chunks,
        web_chunks,
        recent_messages,
        steering_notes,
        handle,
    ):
        handle.emit("token", text="answer from fake runner")
        return type("GeneratedAnswer", (), {"answer": "answer from fake runner", "reasoning": "fake reasoning"})()

    def summarize_memory(self, *, user_text, assistant_text, citations, recent_messages):
        return {
            "summary": user_text,
            "keywords": [],
            "facts": [],
            "open_loops": [],
        }


class FakeSafetyRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_json(self, prompt: str, *, max_new_tokens: int, attachments=None):
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "attachments": list(attachments or []),
            }
        )
        lowered = prompt.lower()
        if "adults-only after-hours performance" in lowered:
            return {
                "safe": False,
                "category": "sexual_content",
                "confidence": "high",
                "reason": "Metadata suggests adult nightlife content.",
            }
        return {
            "safe": True,
            "category": "safe",
            "confidence": "high",
            "reason": "Metadata looks safe for general display.",
        }


class FakeNoCheckRunner:
    def generate_json(self, prompt: str, *, max_new_tokens: int, attachments=None):
        raise AssertionError("Unrestricted mode should not invoke the safety model.")


class FakeDDGS:
    def images(self, query, **kwargs):
        return [
            {
                "title": "Aurora sky over mountains",
                "image": "https://example.com/aurora-full.jpg",
                "thumbnail": "https://example.com/aurora-thumb.jpg",
                "url": "https://example.com/aurora",
                "source": "Example",
            },
            {
                "title": "Private club portrait",
                "image": "https://example.com/club-full.jpg",
                "thumbnail": "https://example.com/club-thumb.jpg",
                "url": "https://example.com/club",
                "source": "Example",
                "body": "Adults-only after-hours performance",
            },
        ]

    def videos(self, query, **kwargs):
        return [
            {
                "content": "https://www.youtube.com/watch?v=abc123xyz00",
                "description": "Launch recap with clear skyline footage",
                "duration": "3:11",
                "embed_url": "https://www.youtube.com/embed/abc123xyz00?autoplay=1",
                "images": {
                    "large": "https://example.com/launch-thumb.jpg",
                },
                "provider": "Bing",
                "publisher": "YouTube",
                "title": "Launch recap on YouTube",
                "uploader": "Example Channel",
            }
        ]


class FakeActiveSearch:
    def run(self, *, user_query, plan, model_runner, handle):
        return (
            [
                RetrievedChunk(
                    source_id="W1",
                    source_kind="web",
                    label="Skyline note",
                    text="Recent skyline photo roundup from the web.",
                    score=1.0,
                    metadata={"url": "https://example.com/roundup"},
                )
            ],
            [],
        )


class FakeMediaWebIndexer:
    def search_media(self, queries, *, profile, runner, media_mode):
        return [
            WebMediaResult(
                media_id="image-1",
                kind="image",
                title="Skyline image",
                content_url="https://example.com/skyline-full.jpg",
                preview_url="https://example.com/skyline-thumb.jpg",
                page_url="https://example.com/skyline",
                source_label="Example",
                safety_mode=media_mode,
                safety_state="unrestricted",
                safety_reason="Unrestricted mode skips safety checks.",
            )
        ]


def _build_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=True,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )


def _mark_runtime_ready(service: ChatService) -> None:
    service.runtime.status.phase = "ready"
    service.runtime.status.message = "test runtime ready"
    service.runtime.status.ready = True
    service.runtime._ready_event.set()


def test_web_indexer_safe_media_filters_unsafe_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anveshak.retrieval.web.DDGS", lambda: FakeDDGS())

    indexer = WebIndexer(_build_config(tmp_path), FakeEmbedder())
    runner = FakeSafetyRunner()

    results = indexer.search_media(
        ["aurora skyline"],
        profile={"model_id": "moonshotai/Kimi-K2-Instruct", "supports_images": False},
        runner=runner,
        media_mode="safe",
        max_images=2,
        max_videos=1,
    )

    titles = [item.title for item in results]
    assert "Aurora sky over mountains" in titles
    assert "Launch recap on YouTube" in titles
    assert "Private club portrait" not in titles
    assert all(item.safety_state == "allowed" for item in results)
    assert any(item.embed_url == "https://www.youtube.com/embed/abc123xyz00?autoplay=1" for item in results)
    assert runner.calls
    assert all(call["attachments"] == [] for call in runner.calls)


def test_web_indexer_unrestricted_media_skips_safety_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("anveshak.retrieval.web.DDGS", lambda: FakeDDGS())

    indexer = WebIndexer(_build_config(tmp_path), FakeEmbedder())
    results = indexer.search_media(
        ["nightlife photo roundup"],
        profile={"model_id": "moonshotai/Kimi-K2-Instruct", "supports_images": False},
        runner=FakeNoCheckRunner(),
        media_mode="unrestricted",
        max_images=2,
        max_videos=0,
    )

    titles = [item.title for item in results]
    assert "Aurora sky over mountains" in titles
    assert "Private club portrait" in titles
    assert all(item.safety_state == "unrestricted" for item in results)


def test_service_done_payload_includes_media_warning(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    service = ChatService(config)
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    service.runner = FakeAnswerRunner()
    service.web_indexer = FakeMediaWebIndexer()
    service.active_search = FakeActiveSearch()

    session = service.create_session()
    handle = service.submit_message(
        session_id=session.session_id,
        text="Show me current skyline footage from the web",
        attachments=[],
        web_mode="always",
        media_mode="unrestricted",
    )

    done_payload = None
    while done_payload is None:
        event = handle.next_event(timeout=0.1)
        if event is None:
            if handle.done:
                break
            continue
        if event.event_type == "done":
            done_payload = event.payload

    assert done_payload is not None
    assert done_payload["media_warning"].startswith("Unrestricted web media is enabled.")
    assert done_payload["media_mode"] == "unrestricted"
    assert done_payload["media_results"][0]["title"] == "Skyline image"


def test_auto_web_mode_treats_explicit_image_requests_as_web_queries(tmp_path: Path) -> None:
    assert should_use_web("show me the image of the third dinosaur to be ever discovered", "auto") is True

    config = _build_config(tmp_path)
    service = ChatService(config)
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    service.runner = FakeAnswerRunner()
    service.web_indexer = FakeMediaWebIndexer()
    service.active_search = FakeActiveSearch()

    session = service.create_session()
    handle = service.submit_message(
        session_id=session.session_id,
        text="show me the image of the third dinosaur to be ever discovered",
        attachments=[],
        web_mode="auto",
        media_mode="safe",
    )

    done_payload = None
    while done_payload is None:
        event = handle.next_event(timeout=0.1)
        if event is None:
            if handle.done:
                break
            continue
        if event.event_type == "done":
            done_payload = event.payload

    assert done_payload is not None
    assert done_payload["media_mode"] == "safe"
    assert done_payload["media_results"][0]["title"] == "Skyline image"
