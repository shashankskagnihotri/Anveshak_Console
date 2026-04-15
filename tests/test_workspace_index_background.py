from __future__ import annotations

import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from anveshak.chat.service import ChatService
from anveshak.config import RuntimeConfig
from anveshak.retrieval import workspace as workspace_module


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_documents(self, texts):
        items = list(texts)
        self.calls.append(items)
        rows = []
        for text in items:
            rows.append([float(len(text)), float(sum(ord(char) for char in text) % 97 or 1)])
        return np.asarray(rows, dtype=np.float32)

    def encode_query(self, text):
        return np.asarray([float(len(text)), 1.0], dtype=np.float32)


class FakeStore:
    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name
        self.next_id = 1
        self.metadata: dict[str, dict[str, object]] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.index = None

    def reset(self) -> None:
        self.next_id = 1
        self.metadata = {}
        self.vectors = {}
        self.index = None

    def add(self, embeddings, metadatas, *, save: bool = True) -> None:
        for vector, metadata in zip(embeddings, metadatas, strict=True):
            row_id = str(self.next_id)
            self.next_id += 1
            self.metadata[row_id] = metadata
            self.vectors[row_id] = np.asarray(vector, dtype=np.float32)
        self.index = object() if self.metadata else None

    def save(self) -> None:
        return None

    def snapshot_rows(self, predicate=None):
        vectors: list[np.ndarray] = []
        metadatas: list[dict[str, object]] = []
        for row_id in sorted(self.metadata, key=int):
            metadata = self.metadata[row_id]
            if predicate is not None and not predicate(metadata):
                continue
            vectors.append(self.vectors[row_id])
            metadatas.append(metadata)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), []
        return np.vstack(vectors).astype(np.float32), metadatas

    def search(self, embedding, top_k: int):
        return []


class BlockingWorkspaceIndex:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def refresh(self):
        self.started.set()
        self.release.wait(timeout=2.0)
        self.calls += 1
        return ["changed.txt"]


def _activate_in_memory_index(
    self,
    *,
    reused_embeddings,
    reused_metadatas,
    changed_embeddings,
    changed_metadatas,
    manifest,
):
    next_store = FakeStore(self.root, "workspace")
    if reused_metadatas:
        next_store.add(reused_embeddings, reused_metadatas, save=False)
    if changed_metadatas and changed_embeddings is not None:
        next_store.add(changed_embeddings, changed_metadatas, save=False)
    self.store = next_store
    self.manifest = manifest


def test_workspace_refresh_only_embeds_changed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workspace_module, "PersistentFaissStore", FakeStore)
    monkeypatch.setattr(workspace_module.WorkspaceIndex, "_build_and_activate_index", _activate_in_memory_index)

    file_a = tmp_path / "alpha.txt"
    file_b = tmp_path / "beta.txt"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_workspace_indexing=True,
        prepare_runtime_on_start=False,
    )
    embedder = FakeEmbedder()
    index = workspace_module.WorkspaceIndex(config, embedder)

    initial = set(index.refresh())
    assert initial == {str(file_a.resolve()), str(file_b.resolve())}
    assert len(embedder.calls) == 1

    embedder.calls.clear()
    file_a.write_text("alpha updated", encoding="utf-8")
    changed = index.refresh()
    assert changed == [str(file_a.resolve())]
    assert embedder.calls == [["alpha updated"]]

    embedder.calls.clear()
    file_b.unlink()
    changed_after_delete = index.refresh()
    assert changed_after_delete == []
    assert embedder.calls == []
    assert all(
        metadata.get("source_path") != str(file_b.resolve())
        for metadata in index.store.metadata.values()
    )


def test_workspace_refresh_status_runs_in_background(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(
        workspace_root=tmp_path,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )
    service = ChatService(config)
    service.config.enable_workspace_indexing = True
    service._workspace_refresh_status["enabled"] = True

    fake_index = BlockingWorkspaceIndex()
    monkeypatch.setattr(service, "_ensure_workspace_index", lambda: fake_index)

    assert service.schedule_workspace_refresh(reason="test") is True
    assert fake_index.started.wait(timeout=1.0)
    assert service.workspace_index_status()["active"] is True

    fake_index.release.set()
    deadline = time.time() + 2.0
    while time.time() < deadline and service.workspace_index_status()["active"]:
        time.sleep(0.02)

    status = service.workspace_index_status()
    assert status["active"] is False
    assert status["last_changed_count"] == 1
    assert fake_index.calls == 1
