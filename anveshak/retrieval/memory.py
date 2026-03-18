"""Persistent long-term memory retrieval built on dense and lexical ranking."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from ..schema import RetrievedChunk
from ..utils import bm25_scores, min_max_normalize, tokenize_for_bm25
from .embeddings import QwenEmbeddingModel
from .vector_store import PersistentFaissStore


class ConversationMemory:
    """Store compact conversation notes and retrieve them for later runs."""

    def __init__(self, root: Path, embedder: QwenEmbeddingModel) -> None:
        self.root = root
        self.embedder = embedder
        self._lock = Lock()
        self.store = PersistentFaissStore(root, "memory")
        self.notes_path = self.root / "memory_notes.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def add_note(self, summary: str, metadata: dict[str, Any]) -> None:
        """Persist a compressed memory note and its dense embedding."""

        summary = summary.strip()
        if not summary:
            return
        with self._lock:
            embedding = self.embedder.encode_documents([summary])
            metadata = {
                **metadata,
                "summary": summary,
            }
            self.store.add(embedding, [metadata])
            with self.notes_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the highest-scoring memory notes for a new user query."""

        if not query.strip():
            return []
        with self._lock:
            all_notes = list(self.store.metadata.values())
            if not all_notes:
                return []
            dense_rows = self.store.search(self.embedder.encode_query(query), min(top_k * 4, max(len(all_notes), top_k)))
            dense_map = {id(item): score for score, item in dense_rows}
            notes = all_notes
            if not notes:
                return []

        dense_scores = min_max_normalize([dense_map.get(id(item), 0.0) for item in notes])
        corpus = [
            tokenize_for_bm25(
                " ".join(
                    [
                        item.get("summary", ""),
                        " ".join(item.get("keywords", [])),
                        " ".join(item.get("facts", [])),
                        " ".join(item.get("open_loops", [])),
                    ]
                )
            )
            for item in notes
        ]
        lexical_scores = bm25_scores(corpus, tokenize_for_bm25(query))
        lexical_scores = min_max_normalize(lexical_scores)

        ranked: list[tuple[float, dict[str, Any]]] = []
        for item, dense_score, lexical_score in zip(notes, dense_scores, lexical_scores, strict=True):
            combined = 0.7 * dense_score + 0.3 * lexical_score
            ranked.append((combined, item))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                source_id=item.get("note_id", f"M{index}"),
                source_kind="memory",
                label=item.get("label", f"Memory {index}"),
                text=item["summary"],
                score=score,
                metadata=item,
            )
            for index, (score, item) in enumerate(ranked[:top_k], start=1)
        ]

    def reset(self) -> None:
        """Delete all persisted memory notes and reset the vector index."""

        with self._lock:
            self.store.reset()
            if self.notes_path.exists():
                self.notes_path.unlink()
