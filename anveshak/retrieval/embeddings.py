"""Embedding adapter shared by file, memory, and web retrieval subsystems."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..config import RuntimeConfig


class QwenEmbeddingModel:
    """Lazy wrapper around the configured embedding model."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._model: Any | None = None

    @property
    def model(self):
        """Load the embedding model on first use and cache it for reuse."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model_source = str(self.config.embedding_local_path) if self.config.embedding_local_path else self.config.embedding_model_id
            self._model = SentenceTransformer(
                model_source,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": self.config.torch_dtype},
                device=self.config.embedding_device,
                cache_folder=str(self.config.hf_cache_dir),
            )
        return self._model

    def encode_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Embed one or more corpus chunks for retrieval-time indexing."""

        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        return self.model.encode(
            items,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a query using the embedding model's query prompt when available."""

        embeddings = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt_name="query",
        )
        return embeddings[0].astype(np.float32)
