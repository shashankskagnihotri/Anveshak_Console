from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class PersistentFaissStore:
    def __init__(self, root: Path, name: str) -> None:
        self.root = root
        self.name = name
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / f"{name}.faiss"
        self.metadata_path = self.root / f"{name}.json"
        self.dimension: int | None = None
        self.next_id = 1
        self.metadata: dict[str, dict[str, Any]] = {}
        self.index = None
        self._load()

    def _load(self) -> None:
        import faiss

        if self.metadata_path.exists():
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.dimension = payload.get("dimension")
            self.next_id = payload.get("next_id", 1)
            self.metadata = payload.get("metadata", {})

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.dimension is None:
                self.dimension = self.index.d

    def _ensure_index(self, dimension: int) -> None:
        import faiss

        if self.index is not None:
            return
        self.dimension = dimension
        base = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap2(base)

    def add(self, embeddings: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be rank-2")
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")
        if len(metadatas) == 0:
            return

        normalized = embeddings.astype("float32", copy=True)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized /= norms

        self._ensure_index(normalized.shape[1])

        import faiss

        ids = np.arange(self.next_id, self.next_id + len(metadatas), dtype=np.int64)
        self.next_id += len(metadatas)
        self.index.add_with_ids(normalized, ids)
        for row_id, metadata in zip(ids.tolist(), metadatas, strict=True):
            self.metadata[str(row_id)] = metadata
        self.save()

    def search(self, embedding: np.ndarray, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if self.index is None or not self.metadata:
            return []

        vector = embedding.astype("float32", copy=True).reshape(1, -1)
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vector /= norm

        scores, ids = self.index.search(vector, top_k)
        results: list[tuple[float, dict[str, Any]]] = []
        for score, row_id in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if row_id == -1:
                continue
            metadata = self.metadata.get(str(row_id))
            if metadata is None:
                continue
            results.append((float(score), metadata))
        return results

    def save(self) -> None:
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        payload = {
            "dimension": self.dimension,
            "next_id": self.next_id,
            "metadata": self.metadata,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self.dimension = None
        self.next_id = 1
        self.metadata = {}
        self.index = None
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
