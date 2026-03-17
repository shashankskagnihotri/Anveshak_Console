from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import trafilatura
from ddgs import DDGS

from ..chunking import chunk_text
from ..config import RuntimeConfig
from ..schema import RetrievedChunk
from .embeddings import QwenEmbeddingModel
from .vector_store import PersistentFaissStore


class WebIndexer:
    def __init__(self, config: RuntimeConfig, embedder: QwenEmbeddingModel) -> None:
        self.config = config
        self.embedder = embedder
        self.root = config.cache_dir / "web"
        self.root.mkdir(parents=True, exist_ok=True)

    def search_and_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = list(DDGS().text(query, max_results=self.config.max_web_pages_per_query))
        if not results:
            return []

        run_key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
        run_dir = self.root / run_key
        store = PersistentFaissStore(run_dir, "web")

        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for row_index, row in enumerate(results, start=1):
            url = row.get("href") or row.get("url")
            title = row.get("title") or url or f"Result {row_index}"
            if not url:
                continue
            page_text = self._fetch_page(url)
            if not page_text:
                continue
            for chunk in chunk_text(
                page_text[: self.config.max_web_chars_per_page],
                chunk_chars=self.config.web_chunk_chars,
                overlap_chars=self.config.web_chunk_overlap,
                chunk_prefix=url,
            ):
                texts.append(chunk.text)
                metadatas.append(
                    {
                        "source_kind": "web",
                        "url": url,
                        "label": title,
                        "snippet": row.get("body", ""),
                        "chunk_text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                    }
                )

        if not texts:
            return []

        embeddings = self.embedder.encode_documents(texts)
        store.reset()
        store.add(embeddings, metadatas)

        vector = self.embedder.encode_query(query)
        rows = store.search(vector, top_k)
        retrieved: list[RetrievedChunk] = []
        for index, (score, item) in enumerate(rows, start=1):
            retrieved.append(
                RetrievedChunk(
                    source_id=f"W{index}",
                    source_kind="web",
                    label=item["label"],
                    text=item.get("chunk_text", item.get("snippet", "")),
                    score=score,
                    metadata=item,
                )
            )

        return retrieved

    def _fetch_page(self, url: str) -> str:
        cache_path = self.root / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return payload.get("text", "")

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_links=True) or ""
        cache_path.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False), encoding="utf-8")
        return text


def should_use_web(query: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "off":
        return False
    lowered = query.lower()
    keywords = {
        "current",
        "internet",
        "latest",
        "lookup",
        "news",
        "online",
        "recent",
        "search",
        "today",
        "web",
        "website",
        "yesterday",
    }
    return any(keyword in lowered for keyword in keywords) or "http://" in lowered or "https://" in lowered
