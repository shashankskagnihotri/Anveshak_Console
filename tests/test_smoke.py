from __future__ import annotations

import json
import random
import asyncio
import os
import time
import types
from io import BytesIO
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from transformers import AutoConfig
from transformers.quantizers.quantizers_utils import should_convert_module

from anveshak.chat.service import ChatService, _build_search_plan, _prior_recent_messages_for_current_turn
from anveshak.config import RuntimeConfig
from anveshak.events import RunHandle
from anveshak.file_parsers import parse_document_for_chat
from anveshak.model_catalog import get_model_profile
from anveshak.modeling.factory import create_runner
from anveshak.modeling.kimi_server_runner import KimiServerRunner
from anveshak.modeling import qwen_runner as qwen_runner_module
from anveshak.modeling.qwen_runner import QwenRunner, SYSTEM_PROMPT
from anveshak.retrieval.web import WebIndexer
from anveshak.retrieval.memory import ConversationMemory
from anveshak.runtime import (
    HUGGINGFACE_TOKEN_ALIAS_ENV_VARS,
    HUGGINGFACE_TOKEN_ENV_VAR,
    HuggingFaceAuthRequiredError,
    RuntimeManager,
    _ProgressTracker,
)
from anveshak import server as server_module
from anveshak.utils import apply_global_seed
from anveshak.schema import Attachment, ChatMessage, ChatSession, RetrievedChunk


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


class FakeRunner:
    model = None

    def load(self):
        return None

    def plan_search(self, *, user_query, recent_messages):
        return {"enabled": False, "rationale": "offline test", "search_queries": [], "max_rounds": 1}

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
        handle.emit("token", text="test answer")
        return type("GeneratedAnswer", (), {"answer": "test answer", "reasoning": "test reasoning"})()

    def summarize_memory(self, *, user_text, assistant_text, citations, recent_messages):
        return {
            "summary": f"Memory of: {user_text}",
            "keywords": ["memory"],
            "facts": ["assistant replied"],
            "open_loops": [],
        }


class FakeRunnerBrokenMemory(FakeRunner):
    def summarize_memory(self, *, user_text, assistant_text, citations, recent_messages):
        raise RuntimeError("memory summarizer unavailable")


class FakeAPIRunner:
    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.calls = []
        self.model = object()

    def generate_text(self, *, system_prompt, user_prompt, attachments, max_new_tokens):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "attachments": attachments,
                "max_new_tokens": max_new_tokens,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return "api output"


class FakeMemory:
    def __init__(self, chunks=None) -> None:
        self.chunks = list(chunks or [])
        self.notes = []

    def retrieve(self, text, top_k):
        return self.chunks[:top_k]

    def add_note(self, summary, metadata):
        self.notes.append((summary, metadata))

    def reset(self):
        self.notes.clear()


class BackgroundMemoryRunner(FakeRunner):
    def __init__(self, started: Event, release: Event) -> None:
        self.started = started
        self.release = release

    def summarize_memory(self, *, user_text, assistant_text, citations, recent_messages):
        self.started.set()
        self.release.wait(timeout=2.0)
        return super().summarize_memory(
            user_text=user_text,
            assistant_text=assistant_text,
            citations=citations,
            recent_messages=recent_messages,
        )


class BlockingRunner(FakeRunner):
    def __init__(self, started: Event, release: Event) -> None:
        self.started = started
        self.release = release

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
        handle.emit("status", phase="generation", text="Generating answer with retrieved context")
        self.started.set()
        self.release.wait(timeout=2.0)
        handle.emit("token", text="blocked answer")
        return type("GeneratedAnswer", (), {"answer": "blocked answer", "reasoning": "blocked reasoning"})()


def _mark_runtime_ready(service: ChatService) -> None:
    service.runtime.status.phase = "ready"
    service.runtime.status.message = "test runtime ready"
    service.runtime.status.ready = True
    service.runtime._ready_event.set()


def test_memory_round_trip(tmp_path: Path) -> None:
    memory = ConversationMemory(tmp_path, FakeEmbedder())
    memory.add_note(
        "The user likes citing local files.",
        {
            "note_id": "note-1",
            "label": "Conversation memory",
            "keywords": ["files", "local"],
            "facts": ["prefers file citations"],
            "open_loops": [],
        },
    )
    retrieved = memory.retrieve("How should we cite local files?", top_k=3)
    assert retrieved
    assert retrieved[0].source_kind == "memory"


def test_obliviate_clears_memory(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    _mark_runtime_ready(service)
    service.runner = FakeRunner()
    service.memory = ConversationMemory(config.memory_dir, FakeEmbedder())

    session = service.create_session()
    handle = service.submit_message(session_id=session.session_id, text="remember this", attachments=[])
    while not handle.done:
        handle.next_event(timeout=0.1)

    memory_notes_path = config.memory_dir / "memory_notes.jsonl"
    assert memory_notes_path.exists()

    reset_handle = service.submit_message(session_id=session.session_id, text="Obliviate", attachments=[])
    while not reset_handle.done:
        reset_handle.next_event(timeout=0.1)

    assert not memory_notes_path.exists()
    saved_session = json.loads((config.session_dir / f"{session.session_id}.json").read_text(encoding="utf-8"))
    assert saved_session["messages"] == []


def test_catalog_infers_backend_metadata() -> None:
    deepseek = get_model_profile("deepseek-ai/deepseek-vl2")
    kimi = get_model_profile("moonshotai/Kimi-K2-Instruct")
    mirothinker = get_model_profile("miromind-ai/MiroThinker-1.7")
    qwen = get_model_profile("Qwen/Qwen2.5-VL-72B-Instruct")
    tiny_qwen = get_model_profile("trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration")

    assert deepseek["input_backend"] == "hf_multimodal"
    assert deepseek["supports_images"] is True
    assert kimi["input_backend"] == "text-chat"
    assert kimi["preferred_runtime_backend"] == "kimi_server"
    assert mirothinker["input_backend"] == "text-chat"
    assert mirothinker["supports_images"] is False
    assert qwen["input_backend"] == "qwen_vision"
    assert tiny_qwen["input_backend"] == "text-chat"


def test_system_prompt_mentions_markdown_and_latex_rendering() -> None:
    assert "Markdown" in SYSTEM_PROMPT
    assert "LaTeX" in SYSTEM_PROMPT


def test_pdf_parser_extracts_text_and_images(tmp_path: Path) -> None:
    import fitz
    from PIL import Image

    pdf_path = tmp_path / "sample.pdf"
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (18, 104, 164)).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello PDF world")
    page.insert_image(fitz.Rect(72, 120, 180, 220), stream=png_bytes)
    document.save(str(pdf_path))
    document.close()

    parsed = parse_document_for_chat(
        pdf_path,
        cache_root=tmp_path / "document-cache",
        max_images=4,
        max_rendered_pages=0,
    )

    assert "Hello PDF world" in parsed.text
    assert parsed.image_paths


def test_query_driven_web_plan_avoids_empty_searches(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        web_mode="auto",
        enable_web=True,
        prepare_runtime_on_start=False,
    )
    plan = _build_search_plan(
        user_query="Find the latest arXiv papers about robust pixel-wise prediction attacks.",
        recent_messages=[],
        config=config,
    )

    assert plan.enabled is True
    assert plan.search_queries
    assert plan.search_queries[0].startswith("Find the latest arXiv papers")


def test_api_call_manager_persists_extended_configuration(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)

    created = service.api_calls.create_call(
        {
            "name": "Reviewer decision helper",
            "system_prompt": "Be precise.",
            "input_template": "Input: {{input}}\nVars: {{json}}",
            "response_instructions": "Return concise prose.",
            "response_mode": "text",
            "web_mode": "always",
            "use_user_context": True,
            "instance_mode": "remember",
        }
    )

    reloaded = service.api_calls.get_call(created.call_id)

    assert reloaded.name == "Reviewer decision helper"
    assert reloaded.embedding_model_id == config.embedding_model_id
    assert reloaded.web_mode == "always"
    assert reloaded.use_user_context is True
    assert reloaded.instance_mode == "remember"


def test_api_call_delete_removes_saved_definition(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)
    created = service.api_calls.create_call({"name": "Delete me"})
    remembered = config.api_session_dir / f"{created.call_id}.json"
    remembered.write_text("{}", encoding="utf-8")

    removed = service.delete_api_call(created.call_id)

    assert removed["call_id"] == created.call_id
    assert not (config.api_calls_dir / f"{created.call_id}.json").exists()
    assert not remembered.exists()


def test_service_create_api_call_triggers_background_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)
    prepared = []

    monkeypatch.setattr(service, "prepare_api_call", lambda call_id: prepared.append(call_id))

    created = service.create_api_call({"name": "Warm after save"})

    assert prepared == [created["call_id"]]


def test_service_invoke_api_call_uses_user_context_and_stateful_history(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    service.runner = FakeAPIRunner(["first response", "second response"])
    service.memory = FakeMemory(
        [
            RetrievedChunk(
                source_id="M1",
                source_kind="memory",
                label="Preference note",
                text="The user prefers strong evidence and reviewer-oriented answers.",
                score=1.0,
                metadata={},
            )
        ]
    )

    api_call = service.api_calls.create_call(
        {
            "name": "Remembering helper",
            "use_user_context": True,
            "instance_mode": "remember",
        }
    )

    first = service.invoke_api_call(api_call.call_id, {"input": "First question"}, api_key=api_call.api_key)
    second = service.invoke_api_call(api_call.call_id, {"input": "Second question"}, api_key=api_call.api_key)

    assert first["output"] == "first response"
    assert second["output"] == "second response"
    assert "Relevant long-term user context from Anveshak" in service.runner.calls[0]["user_prompt"]
    assert "Remembered prior API-call exchanges" in service.runner.calls[1]["user_prompt"]
    assert "First question" in service.runner.calls[1]["user_prompt"]
    assert "first response" in service.runner.calls[1]["user_prompt"]


def test_service_invoke_api_call_returns_json_and_web_citations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=True,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    service.runner = FakeAPIRunner(['{"decision":"accept"}'])
    web_chunk = RetrievedChunk(
        source_id="W1",
        source_kind="web",
        label="ICLR reviews",
        text="Recent review trends suggest stronger evidence matters.",
        score=0.9,
        metadata={"url": "https://example.com/reviews"},
    )
    monkeypatch.setattr(service, "_run_api_call_web_search", lambda **kwargs: [web_chunk])

    api_call = service.api_calls.create_call(
        {
            "name": "JSON reviewer",
            "response_mode": "json",
            "web_mode": "always",
        }
    )

    result = service.invoke_api_call(api_call.call_id, {"input": "Assess this paper"}, api_key=api_call.api_key)

    assert result["output"] == {"decision": "accept"}
    assert result["citations"][0]["metadata"]["url"] == "https://example.com/reviews"
    assert "Return exactly one valid JSON object" in service.runner.calls[0]["user_prompt"]


def test_message_override_can_disable_web_search(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        web_mode="always",
        enable_web=True,
        prepare_runtime_on_start=False,
    )

    plan = _build_search_plan(
        user_query="Find the latest ICLR reviews for this topic.",
        recent_messages=[],
        config=config,
        web_mode_override="off",
    )

    assert plan.enabled is False
    assert "disabled" in plan.rationale


def test_message_override_can_force_web_search(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        web_mode="off",
        enable_web=False,
        prepare_runtime_on_start=False,
    )

    plan = _build_search_plan(
        user_query="Explain the transformer architecture at a high level.",
        recent_messages=[],
        config=config,
        web_mode_override="always",
    )

    assert plan.enabled is True
    assert plan.search_queries
    assert plan.rationale == "web retrieval was forced by the message setting"


def test_kimi_config_compat_shim_skips_only_top_level_experts_module() -> None:
    root = Path(__file__).resolve().parents[1]
    config = RuntimeConfig(
        workspace_root=root,
        model_id="moonshotai/Kimi-K2-Instruct",
        embedding_model_id="Qwen/Qwen3-Embedding-0.6B",
        checkpoints_dir=root / "checkpoints",
        api_calls_dir=root / "API_calls",
        prepare_runtime_on_start=False,
    )
    runner = QwenRunner(config)
    source = root / "checkpoints" / "models" / "moonshotai--Kimi-K2-Instruct"

    kimi_config = AutoConfig.from_pretrained(str(source), trust_remote_code=True)
    kimi_config = runner._apply_model_config_compat_shims(kimi_config)
    quantization_config = getattr(kimi_config, "quantization_config")
    modules_to_not_convert = list(quantization_config.get("modules_to_not_convert") or [])

    assert r".*\.experts$" in modules_to_not_convert
    assert should_convert_module("model.layers.12.mlp.experts.0.gate_proj", modules_to_not_convert) is True
    assert should_convert_module("model.layers.12.mlp.experts", modules_to_not_convert) is False


def test_prequantized_fp8_models_auto_allocate_cpu_headroom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        prepare_runtime_on_start=False,
    )
    runner = QwenRunner(config)
    monkeypatch.setattr(runner, "_system_memory_gib", lambda: 1511)
    model_config = types.SimpleNamespace(quantization_config={"quant_method": "fp8"})

    max_memory = runner._resolved_max_memory(model_config)

    assert max_memory == {"cpu": "1447GiB"}


def test_multi_gpu_memory_policy_uses_requested_visible_gpu_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        n_gpus=2,
        max_gpu_memory_gib=70,
        prepare_runtime_on_start=False,
    )
    runner = QwenRunner(config)
    model_config = types.SimpleNamespace(quantization_config={"quant_method": "none"})

    monkeypatch.setattr(qwen_runner_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(qwen_runner_module.torch.cuda, "device_count", lambda: 4)

    max_memory = runner._resolved_max_memory(model_config)

    assert max_memory == {
        0: "70GiB",
        1: "70GiB",
        2: "0GiB",
        3: "0GiB",
    }


def test_large_local_checkpoint_prefers_ram_heavy_loading_when_memory_allows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(runner, "_checkpoint_size_gib", lambda model_source: 959.0)
    monkeypatch.setattr(runner, "_available_memory_gib", lambda: 1467)

    assert runner._can_use_ram_heavy_load_path(str(model_dir)) is True


def test_large_local_checkpoint_keeps_streaming_loading_when_memory_is_tight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(runner, "_checkpoint_size_gib", lambda model_source: 959.0)
    monkeypatch.setattr(runner, "_available_memory_gib", lambda: 900)

    assert runner._can_use_ram_heavy_load_path(str(model_dir)) is False


def test_local_sharded_checkpoints_enable_parallel_loading_workers(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-1-of-2.safetensors").write_bytes(b"a")
    (model_dir / "model-2-of-2.safetensors").write_bytes(b"b")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-1-of-2.safetensors",
                    "model.layers.1.weight": "model-2-of-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert runner._parallel_loading_worker_count(str(model_dir)) == 2


def test_runner_factory_selects_dedicated_kimi_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            created["client"] = self

    monkeypatch.setattr("anveshak.modeling.kimi_server_runner.httpx.Client", FakeClient)
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        kimi_server_url="http://127.0.0.1:9000",
        prepare_runtime_on_start=False,
    )

    runner = create_runner(config)

    assert isinstance(runner, KimiServerRunner)


def test_answer_prompt_prioritizes_current_task_and_compacts_background_context(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)
    attachments = [
        Attachment(path="/tmp/chart.png", name="chart.png", media_kind="image", source="upload"),
        Attachment(path="/tmp/notes.txt", name="notes.txt", media_kind="document", source="upload"),
    ]
    memory_chunks = [
        RetrievedChunk(source_id=f"M{index}", source_kind="memory", label=f"Memory {index}", text="memory " * 200, score=1.0 / index)
        for index in range(1, 6)
    ]
    file_chunks = [
        RetrievedChunk(source_id="F1", source_kind="file", label="report.pdf", text="current file evidence " * 120, score=1.0),
    ]
    web_chunks = [
        RetrievedChunk(source_id="W1", source_kind="web", label="example.com", text="fresh web evidence " * 120, score=1.0),
    ]
    recent_messages = [
        ChatMessage(role="user", text="Older request that should stay in the background."),
        ChatMessage(role="assistant", text="Older answer that should not override the new task."),
    ]

    prompt = runner._compose_answer_prompt(
        user_query="Summarize the attached report and focus on the newest experiment.",
        attachments=attachments,
        memory_chunks=memory_chunks,
        file_chunks=file_chunks,
        web_chunks=web_chunks,
        recent_messages=recent_messages,
        steering_notes=[],
    )

    assert "Primary task for this answer (highest priority):" in prompt
    assert "Task focus rules:" in prompt
    assert "Attachments for this turn:" in prompt
    assert "chart.png (image)" in prompt
    assert prompt.index("Local file context for this task:") < prompt.index("Long-term memory background (supporting only, do not override the current task):")
    assert "M1 Memory 1" in prompt
    assert "M2 Memory 2" in prompt
    assert "M3 Memory 3" in prompt
    assert "M4 Memory 4" not in prompt
    assert "Final focus reminder:" in prompt


def test_prior_recent_messages_for_current_turn_excludes_latest_user_message() -> None:
    session = ChatSession()
    session.append(ChatMessage(role="user", text="First question"))
    session.append(ChatMessage(role="assistant", text="First answer"))
    session.append(ChatMessage(role="user", text="Newest request"))

    recent = _prior_recent_messages_for_current_turn(session, 6)

    assert [message.text for message in recent] == ["First question", "First answer"]


def test_kimi_server_runner_generate_text_uses_openai_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class FakeResponse:
        def __init__(self, payload=None, lines=None):
            self.payload = payload or {}
            self.lines = lines or []

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            yield from self.lines

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.posts = []
            created["client"] = self

        def get(self, path, headers=None):
            return FakeResponse({"data": [{"id": "kimi-k2"}]})

        def post(self, path, json=None, headers=None):
            self.posts.append({"path": path, "json": json, "headers": headers})
            return FakeResponse({"choices": [{"message": {"content": "server answer"}}]})

    monkeypatch.setattr("anveshak.modeling.kimi_server_runner.httpx.Client", FakeClient)
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        kimi_server_url="http://127.0.0.1:9000",
        prepare_runtime_on_start=False,
    )
    runner = KimiServerRunner(config)

    output = runner.generate_text(
        system_prompt="system prompt",
        user_prompt="user prompt",
        attachments=[],
        max_new_tokens=64,
    )

    assert output == "server answer"
    payload = created["client"].posts[0]["json"]
    assert payload["model"] == "kimi-k2"
    assert payload["messages"][0]["content"] == "system prompt"
    assert payload["messages"][1]["content"] == "user prompt"


def test_kimi_server_runner_streams_reasoning_and_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload=None, lines=None):
            self.payload = payload or {}
            self.lines = lines or []

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def iter_lines(self):
            yield from self.lines

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, path, headers=None):
            return FakeResponse({"data": [{"id": "kimi-k2"}]})

        def stream(self, method, path, json=None, headers=None):
            return FakeResponse(
                lines=[
                    'data: {"choices":[{"delta":{"content":"<think>Reaso"}}]}',
                    'data: {"choices":[{"delta":{"content":"ning</think>Answer"}}]}',
                    "data: [DONE]",
                ]
            )

    monkeypatch.setattr("anveshak.modeling.kimi_server_runner.httpx.Client", FakeClient)
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        kimi_server_url="http://127.0.0.1:9000",
        prepare_runtime_on_start=False,
    )
    runner = KimiServerRunner(config)
    handle = RunHandle("session")

    answer = runner.stream_answer(
        user_query="Should I search the web?",
        attachments=[],
        memory_chunks=[],
        file_chunks=[],
        web_chunks=[],
        recent_messages=[],
        steering_notes=[],
        handle=handle,
    )

    assert answer.reasoning == "Reasoning"
    assert answer.answer == "Answer"


def test_web_indexer_caches_search_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=True,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    calls = {"count": 0}

    class FakeDDGS:
        def text(self, query, **kwargs):
            calls["count"] += 1
            return [
                {
                    "href": "https://example.com/paper",
                    "title": "Example Paper",
                    "body": "Example snippet",
                }
            ]

    monkeypatch.setattr("anveshak.retrieval.web.DDGS", lambda: FakeDDGS())
    monkeypatch.setattr(WebIndexer, "_fetch_page", lambda self, url: "Example paper body")

    indexer = WebIndexer(config, FakeEmbedder())
    first = indexer.search_and_retrieve("cached paper query", top_k=2)
    second = indexer.search_and_retrieve("cached paper query", top_k=2)

    assert calls["count"] == 1
    assert first
    assert second


def test_web_indexer_swallow_search_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=True,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )

    class FailingDDGS:
        def text(self, query, **kwargs):
            raise RuntimeError("temporary DDGS failure")

    monkeypatch.setattr("anveshak.retrieval.web.DDGS", lambda: FailingDDGS())

    indexer = WebIndexer(config, FakeEmbedder())
    assert indexer.search_and_retrieve("flaky web search", top_k=2) == []


def test_run_logs_are_written(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    _mark_runtime_ready(service)
    service.runner = FakeRunner()
    service.memory = ConversationMemory(config.memory_dir, FakeEmbedder())

    session = service.create_session()
    handle = service.submit_message(session_id=session.session_id, text="log this run", attachments=[])
    while not handle.done:
        handle.next_event(timeout=0.1)

    log_files = sorted(config.logs_dir.glob("*.jsonl"))
    assert log_files
    records = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
    event_types = [record["event_type"] for record in records]
    assert "run_started" in event_types
    assert "THINKING" in event_types
    assert "done" in event_types
    assert event_types[-1] == "run_finished"


def test_apply_global_seed_reproducible() -> None:
    import torch

    apply_global_seed(7)
    first = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
    )

    apply_global_seed(7)
    second = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_memory_failure_does_not_fail_answer(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    service._memory_idle_grace_seconds = 0.0
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    service.runner = FakeRunnerBrokenMemory()
    service.memory = FakeMemory()

    session = service.create_session()
    handle = service.submit_message(session_id=session.session_id, text="keep the answer", attachments=[])

    event_types: list[str] = []
    while not handle.done:
        event = handle.next_event(timeout=0.1)
        if event is None:
            continue
        event_types.append(event.event_type)

    for _ in range(40):
        log_files = sorted(config.logs_dir.glob("*.jsonl"))
        if log_files:
            records = [json.loads(line) for line in log_files[0].read_text(encoding="utf-8").splitlines()]
            if any(record["event_type"] == "memory_compression_failed" for record in records):
                break
        time.sleep(0.05)
    else:
        pytest.fail("memory compression failure was not recorded in the run log")

    assert "done" in event_types
    assert "error" not in event_types


def test_chat_unlocks_before_background_memory_compression_finishes(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    service._memory_idle_grace_seconds = 0.0
    _mark_runtime_ready(service)
    service._ensure_components = lambda: None
    started = Event()
    release = Event()
    service.runner = BackgroundMemoryRunner(started, release)
    service.memory = FakeMemory()

    session = service.create_session()
    first = service.submit_message(session_id=session.session_id, text="first turn", attachments=[])
    while not first.done:
        first.next_event(timeout=0.1)

    assert started.wait(timeout=2.0) is True

    second = service.submit_message(session_id=session.session_id, text="second turn", attachments=[])

    assert second.run_id != first.run_id
    release.set()
    while not second.done:
        second.next_event(timeout=0.1)


def test_decoder_tokenizer_falls_back_to_auto_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)
    runner.processor = type("Processor", (), {"tokenizer": None})()

    sentinel = object()

    def fake_load_tokenizer(model_source: str):
        return sentinel

    monkeypatch.setattr(runner, "_load_tokenizer", fake_load_tokenizer)

    tokenizer = runner._decoder_tokenizer()

    assert tokenizer is sentinel
    assert runner.tokenizer is sentinel


def test_kimi_prefers_eager_attention_when_runtime_uses_default_sdpa(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        prepare_runtime_on_start=False,
    )
    runner = QwenRunner(config)

    assert runner._resolved_attn_implementation() == "eager"


def test_transformers_compat_shim_restores_legacy_flash_attn_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import_utils = types.SimpleNamespace()
    utils = types.SimpleNamespace(
        import_utils=import_utils,
        is_flash_attn_greater_or_equal=lambda version: version == "2.10",
    )
    fake_transformers = types.SimpleNamespace(utils=utils)

    monkeypatch.setattr(qwen_runner_module, "transformers", fake_transformers)

    qwen_runner_module._apply_transformers_compat_shims()

    assert utils.is_flash_attn_greater_or_equal_2_10() is True
    assert import_utils.is_flash_attn_greater_or_equal_2_10() is True
    assert import_utils.is_torch_fx_available() is True


def test_model_loader_retries_with_eager_attention_on_sdpa_rejection(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    runner = QwenRunner(config)

    class FakeLoader:
        calls = []

        @classmethod
        def from_pretrained(cls, model_source, **kwargs):
            cls.calls.append((model_source, dict(kwargs)))
            if kwargs.get("attn_implementation") == "sdpa":
                raise ValueError(
                    "DeepseekV3ForCausalLM does not support an attention implementation through "
                    "torch.nn.functional.scaled_dot_product_attention yet. "
                    "Please load your model with the argument `attn_implementation=\"eager\"` meanwhile."
                )
            return {"model_source": model_source, "kwargs": kwargs}

    model = runner._load_model_with_compat(
        FakeLoader,
        "moonshotai/Kimi-K2-Instruct",
        {"attn_implementation": "sdpa", "trust_remote_code": True},
    )

    assert len(FakeLoader.calls) == 2
    assert FakeLoader.calls[0][1]["attn_implementation"] == "sdpa"
    assert FakeLoader.calls[1][1]["attn_implementation"] == "eager"
    assert model["kwargs"]["attn_implementation"] == "eager"


def test_kimi_config_compat_shim_backfills_fp8_expert_aliases(tmp_path: Path) -> None:
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / "models" / "moonshotai--Kimi-K2-Instruct"
    if not checkpoint_dir.exists():
        pytest.skip("Kimi checkpoint metadata is not available in this checkout")

    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        prepare_runtime_on_start=False,
    )
    runner = QwenRunner(config)
    raw_config = AutoConfig.from_pretrained(
        str(checkpoint_dir),
        cache_dir=str(tmp_path / "hf-cache"),
        trust_remote_code=True,
    )

    compatible = runner._apply_model_config_compat_shims(raw_config)

    from transformers.integrations.finegrained_fp8 import FP8Expert

    expert = FP8Expert(compatible, [128, 128])

    assert compatible.num_local_experts == compatible.n_routed_experts
    assert compatible.num_experts == compatible.n_routed_experts
    assert compatible.rope_parameters["rope_type"] == "yarn"
    assert compatible.head_dim == compatible.qk_rope_head_dim
    assert expert.num_experts == compatible.n_routed_experts


def test_runtime_progress_tracker_updates_aggregate_bytes(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    manager = RuntimeManager(config)
    tracker = _ProgressTracker(manager, role="assistant-model")
    tracking_class = tracker.make_tqdm_class()

    bytes_bar = tracking_class(
        desc="Downloading (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        disable=True,
        name="huggingface_hub.snapshot_download",
    )
    files_bar = tracking_class(
        desc="Fetching 4 files",
        total=4,
        initial=0,
        disable=True,
    )

    bytes_bar.total = 100
    bytes_bar.refresh()
    bytes_bar.update(25)
    files_bar.update(2)

    status = manager.status_dict()

    assert status["bytes_total"] == 100
    assert status["bytes_downloaded"] == 25
    assert status["files_total"] == 4
    assert status["files_completed"] == 2
    assert status["progress"] == pytest.approx(0.25)


def test_runtime_uses_existing_local_repo_without_hub_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    manager = RuntimeManager(config)
    target_dir = config.models_dir / "example--model"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "model.safetensors").write_bytes(b"test")

    def _unexpected_snapshot_download(*args, **kwargs):
        raise AssertionError("snapshot_download should not run for an existing local repo")

    monkeypatch.setattr("anveshak.runtime.snapshot_download", _unexpected_snapshot_download)

    resolved = manager._ensure_repo_downloaded("example/model", role="assistant-model")

    assert resolved == target_dir
    assert (target_dir / ".download-complete").exists()


def test_runtime_stages_large_repo_to_local_nvme_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    manager = RuntimeManager(config)
    target_dir = config.models_dir / "example--model"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / "model-1-of-2.safetensors").write_bytes(b"a")
    (target_dir / "model-2-of-2.safetensors").write_bytes(b"b")
    (target_dir / ".download-complete").write_text("example/model", encoding="utf-8")

    stage_root = tmp_path / "local-nvme-cache"
    monkeypatch.setattr(manager, "_local_stage_root", lambda: stage_root)
    monkeypatch.setattr(manager, "_should_stage_repo_locally", lambda source_dir: True)

    resolved = manager._ensure_repo_downloaded("example/model", role="assistant-model")

    assert resolved == stage_root / "example--model"
    assert (resolved / "config.json").exists()
    assert (resolved / "model-1-of-2.safetensors").read_bytes() == b"a"
    assert (resolved / ".stage-complete").exists()


def test_runtime_skips_local_kimi_checkpoint_when_dedicated_server_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="moonshotai/Kimi-K2-Instruct",
        kimi_server_url="http://127.0.0.1:9000",
        prepare_runtime_on_start=False,
    )
    manager = RuntimeManager(config)
    calls: list[tuple[str, str]] = []

    def _fake_ensure_repo_downloaded(model_id: str, *, role: str) -> Path:
        calls.append((model_id, role))
        target = tmp_path / role
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(manager, "_ensure_repo_downloaded", _fake_ensure_repo_downloaded)

    manager._prepare_runtime()

    assert config.model_local_path is None
    assert calls == [(config.embedding_model_id, "embedding-model")]


def test_runtime_requires_server_for_server_only_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="example/server-only-model",
        prepare_runtime_on_start=False,
    )
    manager = RuntimeManager(config)
    monkeypatch.setattr(
        "anveshak.runtime.get_model_profile",
        lambda model_id: {
            "label": "Server Only Model",
            "model_id": model_id,
            "kind": "text-generation",
            "input_backend": "text-chat",
            "requires_server_backend": True,
        },
    )

    manager._prepare_runtime()

    status = manager.status_dict()
    assert status["phase"] == "error"
    assert "server-backed reasoning model" in (status.get("error") or "")
    assert "--kimi-server-url" in (status.get("error") or "")


def test_runtime_marks_gated_model_downloads_as_auth_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        model_id="google/gemma-3-27b-it",
        prepare_runtime_on_start=False,
    )
    manager = RuntimeManager(config)

    def _fake_ensure_repo_downloaded(model_id: str, *, role: str) -> Path:
        raise HuggingFaceAuthRequiredError(model_id=model_id, role=role)

    monkeypatch.setattr(manager, "_ensure_repo_downloaded", _fake_ensure_repo_downloaded)

    manager._prepare_runtime()

    status = manager.status_dict()
    assert status["phase"] == "auth-required"
    assert status["huggingface_auth_required"] is True
    assert status["huggingface_auth_model_id"] == config.model_id
    assert HUGGINGFACE_TOKEN_ENV_VAR in (status.get("huggingface_auth_message") or "")


@pytest.mark.parametrize("env_name", (HUGGINGFACE_TOKEN_ENV_VAR, *HUGGINGFACE_TOKEN_ALIAS_ENV_VARS))
def test_runtime_treats_huggingface_token_env_aliases_equivalently_and_retries_after_manual_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    manager = RuntimeManager(config)
    monkeypatch.setenv(env_name, "hf_env_token")
    monkeypatch.setattr("anveshak.runtime.login", lambda *args, **kwargs: None)

    manager._refresh_huggingface_token_from_environment()

    assert manager._active_huggingface_token() == "hf_env_token"
    assert manager.status_dict()["huggingface_token_source"] == f"env:{env_name}"

    starts: list[str] = []
    monkeypatch.setattr(manager.api, "whoami", lambda token: {"name": "tester"})
    monkeypatch.setattr(manager, "start_async", lambda: starts.append("started"))

    try:
        payload = manager.configure_huggingface_token("hf_manual_token")
    finally:
        os.environ.pop(HUGGINGFACE_TOKEN_ENV_VAR, None)
        for alias_name in HUGGINGFACE_TOKEN_ALIAS_ENV_VARS:
            os.environ.pop(alias_name, None)

    assert starts == ["started"]
    assert payload["phase"] == "checking"
    assert payload["huggingface_auth_required"] is False
    assert payload["huggingface_token_source"] == "manual"


def test_server_thread_helper_returns_shutdown_sentinel_on_executor_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raising_to_thread(func, *args):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(server_module.asyncio, "to_thread", _raising_to_thread)

    result = asyncio.run(server_module._await_thread_call(lambda: "ignored"))

    assert result is server_module._SHUTDOWN_SENTINEL


def test_server_thread_helper_preserves_normal_results() -> None:
    result = asyncio.run(server_module._await_thread_call(lambda value: value, "ok"))

    assert result == "ok"


def test_runtime_wait_for_status_change_only_returns_on_change(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    manager = RuntimeManager(config)

    initial = manager.status_dict()

    assert manager.wait_for_status_change(initial["version"], timeout=0.01) is None

    manager._update(message="Downloading model shards")
    changed = manager.wait_for_status_change(initial["version"], timeout=0.01)

    assert changed is not None
    assert changed["message"] == "Downloading model shards"
    assert changed["version"] > initial["version"]


def test_submit_message_rejects_second_active_run(tmp_path: Path) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    _mark_runtime_ready(service)
    started = Event()
    release = Event()
    service.runner = BlockingRunner(started, release)
    service.memory = ConversationMemory(config.memory_dir, FakeEmbedder())

    session = service.create_session()
    handle = service.submit_message(session_id=session.session_id, text="first", attachments=[])

    with pytest.raises(ValueError, match="Finish the current run before sending another prompt"):
        service.submit_message(session_id=session.session_id, text="second", attachments=[])

    release.set()
    while not handle.done:
        handle.next_event(timeout=0.1)


def test_steering_is_rejected_outside_generation_phase(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)
    session = service.create_session()
    handle = RunHandle(session.session_id)
    service.runs[handle.run_id] = handle

    with pytest.raises(ValueError, match="actively generating an answer"):
        service.steer_run(handle.run_id, "change direction")


def test_steering_is_allowed_during_generation_phase(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)
    session = service.create_session()
    handle = RunHandle(session.session_id)
    handle.phase = "generation"
    service.runs[handle.run_id] = handle

    service.steer_run(handle.run_id, "favor recent evidence")

    assert handle.has_pending_restart() is True


def test_model_stays_pinned_while_session_exists(tmp_path: Path) -> None:
    config = RuntimeConfig(workspace_root=tmp_path, prepare_runtime_on_start=False)
    service = ChatService(config)

    assert service._model_is_pinned_by_session() is False

    service.create_session()

    assert service._model_is_pinned_by_session() is True
