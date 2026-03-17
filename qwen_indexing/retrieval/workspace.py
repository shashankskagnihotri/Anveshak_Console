from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import re

from ..chunking import chunk_text
from ..config import DOCUMENT_EXTENSIONS, IGNORED_DIR_NAMES, RuntimeConfig, TEXT_EXTENSIONS
from ..file_parsers import detect_media_kind, extract_text_from_path
from ..schema import RetrievedChunk
from .embeddings import QwenEmbeddingModel
from .vector_store import PersistentFaissStore


class WorkspaceIndex:
    def __init__(self, config: RuntimeConfig, embedder: QwenEmbeddingModel) -> None:
        self.config = config
        self.embedder = embedder
        self.root = config.file_index_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = PersistentFaissStore(self.root, "workspace")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict[str, str | int]]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _iter_candidate_files(self) -> Iterable[Path]:
        for path in self.config.workspace_root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > self.config.max_indexed_file_bytes:
                continue
            yield path

    def refresh(self) -> list[str]:
        changed_paths: list[Path] = []
        seen: set[str] = set()

        for path in self._iter_candidate_files():
            resolved = str(path.resolve())
            seen.add(resolved)
            stat = path.stat()
            fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
            if self.manifest.get(resolved, {}).get("fingerprint") == fingerprint:
                continue
            changed_paths.append(path)

        stale_paths = set(self.manifest) - seen
        if stale_paths:
            self._rebuild_index(skip_paths=stale_paths)

        if changed_paths:
            self._rebuild_index()

        return [str(path) for path in changed_paths]

    def _rebuild_index(self, *, skip_paths: set[str] | None = None) -> None:
        documents: list[tuple[str, dict[str, str]]] = []
        new_manifest: dict[str, dict[str, str | int]] = {}

        for path in self._iter_candidate_files():
            resolved = str(path.resolve())
            if skip_paths and resolved in skip_paths:
                continue
            try:
                text = extract_text_from_path(path)
            except Exception:
                continue
            if not text.strip():
                continue
            chunks = chunk_text(
                text,
                chunk_chars=self.config.workspace_chunk_chars,
                overlap_chars=self.config.workspace_chunk_overlap,
                chunk_prefix=resolved,
            )
            for chunk in chunks:
                documents.append(
                    (
                        chunk.text,
                        {
                            "source_kind": "file",
                            "source_path": resolved,
                            "label": path.name,
                            "chunk_text": chunk.text,
                            "chunk_id": chunk.chunk_id,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                        },
                    )
                )
            stat = path.stat()
            new_manifest[resolved] = {
                "fingerprint": f"{stat.st_mtime_ns}:{stat.st_size}",
                "name": path.name,
                "kind": detect_media_kind(path),
            }

        self.store.reset()
        if documents:
            embeddings = self.embedder.encode_documents([text for text, _ in documents])
            self.store.add(embeddings, [metadata for _, metadata in documents])
        self.manifest = new_manifest
        self._save_manifest()

    def index_paths(self, paths: list[Path], *, parsed_texts: dict[str, str] | None = None) -> list[str]:
        texts: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        indexed: list[str] = []
        parsed_texts = parsed_texts or {}

        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS:
                continue
            try:
                resolved = str(path.resolve())
                text = parsed_texts.get(resolved)
                if text is None:
                    text = extract_text_from_path(path)
            except Exception:
                continue
            if not text.strip():
                continue
            indexed.append(str(path.resolve()))
            for chunk in chunk_text(
                text,
                chunk_chars=self.config.direct_file_chunk_chars,
                overlap_chars=self.config.direct_file_chunk_overlap,
                chunk_prefix=str(path.resolve()),
            ):
                texts.append(chunk.text)
                metadatas.append(
                    {
                        "source_kind": "file",
                        "source_path": str(path.resolve()),
                        "label": path.name,
                        "chunk_text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                )

        if texts:
            embeddings = self.embedder.encode_documents(texts)
            self.store.add(embeddings, metadatas)
        return indexed

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        vector = self.embedder.encode_query(query)
        rows = self.store.search(vector, top_k)
        results: list[RetrievedChunk] = []
        for index, (score, item) in enumerate(rows, start=1):
            source_path = item["source_path"]
            label = item.get("label", Path(source_path).name)
            results.append(
                RetrievedChunk(
                    source_id=f"F{index}",
                    source_kind="file",
                    label=label,
                    text=item.get(
                        "chunk_text",
                        self._excerpt(source_path, int(item["start_char"]), int(item["end_char"])),
                    ),
                    score=score,
                    metadata=item,
                )
            )
        return results

    def direct_path_context(self, paths: list[Path]) -> list[RetrievedChunk]:
        snippets: list[RetrievedChunk] = []
        for index, path in enumerate(paths, start=1):
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS:
                continue
            try:
                text = extract_text_from_path(path)
            except Exception:
                continue
            excerpt = " ".join(text.split())[: self.config.max_inline_file_chars]
            if not excerpt:
                continue
            snippets.append(
                RetrievedChunk(
                    source_id=f"F-DIRECT-{index}",
                    source_kind="file",
                    label=path.name,
                    text=excerpt,
                    score=1.0,
                    metadata={"source_path": str(path.resolve()), "label": path.name},
                )
            )
        return snippets

    def _excerpt(self, source_path: str, start_char: int, end_char: int) -> str:
        path = Path(source_path)
        try:
            text = extract_text_from_path(path)
        except Exception:
            return ""
        cleaned = " ".join(text.split())
        return cleaned[start_char:end_char]

    @staticmethod
    def extract_path_mentions(text: str) -> list[Path]:
        paths: list[Path] = []
        quoted_candidates = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for left, right in quoted_candidates:
            token = left or right
            candidate = Path(token).expanduser()
            if candidate.exists():
                paths.append(candidate.resolve())

        for token in text.split():
            if "/" not in token and "\\" not in token:
                continue
            candidate = Path(token.strip(" ,.;:!?()[]{}<>\"'")).expanduser()
            if candidate.exists():
                paths.append(candidate.resolve())
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(path)
            if resolved in seen:
                continue
            unique.append(path)
            seen.add(resolved)
        return unique
