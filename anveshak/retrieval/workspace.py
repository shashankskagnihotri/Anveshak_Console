"""Local workspace indexing and explicit path-grounded retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Iterable
import re

from ..chunking import chunk_text
from ..config import DOCUMENT_EXTENSIONS, IGNORED_DIR_NAMES, RuntimeConfig, TEXT_EXTENSIONS
from ..file_parsers import detect_media_kind, extract_text_from_path
from ..schema import RetrievedChunk
from .embeddings import QwenEmbeddingModel
from .vector_store import PersistentFaissStore


class WorkspaceIndex:
    """Maintain a searchable index over local text and office documents."""

    def __init__(self, config: RuntimeConfig, embedder: QwenEmbeddingModel) -> None:
        self.config = config
        self.embedder = embedder
        self.root = config.file_index_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = PersistentFaissStore(self.root, "workspace")
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()
        self._refresh_lock = Lock()
        self._swap_lock = Lock()

    def _load_manifest(self) -> dict[str, dict[str, str | int]]:
        """Load the file fingerprint manifest used for incremental refreshes."""

        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self) -> None:
        """Persist the current workspace manifest to disk."""

        self.manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _iter_candidate_files(self) -> Iterable[Path]:
        """Yield local files that should participate in ambient workspace retrieval."""

        for root_text, dir_names, file_names in os.walk(self.config.workspace_root, topdown=True):
            dir_names[:] = [name for name in dir_names if name not in IGNORED_DIR_NAMES]
            root_path = Path(root_text)
            for filename in file_names:
                path = root_path / filename
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
        """Re-index changed files and drop stale entries from the workspace store."""

        with self._refresh_lock:
            with self._swap_lock:
                current_store = self.store
                manifest_snapshot = dict(self.manifest)
            candidate_paths = list(self._iter_candidate_files())
            seen_paths: set[str] = set()
            changed_paths: list[Path] = []

            for path in candidate_paths:
                resolved = str(path.resolve())
                seen_paths.add(resolved)
                try:
                    stat = path.stat()
                except OSError:
                    continue
                fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
                if manifest_snapshot.get(resolved, {}).get("fingerprint") == fingerprint:
                    continue
                changed_paths.append(path)

            stale_paths = set(manifest_snapshot) - seen_paths
            if not changed_paths and not stale_paths:
                return []

            replaced_paths = stale_paths | {str(path.resolve()) for path in changed_paths}
            try:
                reused_embeddings, reused_metadatas = current_store.snapshot_rows(
                    lambda metadata: (
                        str(metadata.get("source_path", "")) in seen_paths
                        and str(metadata.get("source_path", "")) not in replaced_paths
                    )
                )
                changed_texts, changed_metadatas, changed_manifest = self._prepare_documents(changed_paths)
            except RuntimeError:
                self._rebuild_index(paths=candidate_paths)
                return [str(path) for path in changed_paths]
            changed_embeddings = self.embedder.encode_documents(changed_texts) if changed_metadatas else None

            next_manifest = {
                path_text: payload
                for path_text, payload in manifest_snapshot.items()
                if path_text in seen_paths and path_text not in replaced_paths
            }
            next_manifest.update(changed_manifest)
            self._build_and_activate_index(
                reused_embeddings=reused_embeddings,
                reused_metadatas=reused_metadatas,
                changed_embeddings=changed_embeddings,
                changed_metadatas=changed_metadatas,
                manifest=next_manifest,
            )
            return [str(path) for path in changed_paths]

    def _prepare_documents(
        self,
        paths: list[Path],
    ) -> tuple[list[str], list[dict[str, str | int]], dict[str, dict[str, str | int]]]:
        """Read and chunk candidate files into metadata-rich documents plus manifest rows."""

        texts: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        manifest_rows: dict[str, dict[str, str | int]] = {}

        for path in paths:
            resolved = str(path.resolve())
            try:
                text = extract_text_from_path(path)
            except Exception:
                continue
            if not text.strip():
                continue
            # Each chunk keeps enough source metadata to become a citation later.
            chunks = chunk_text(
                text,
                chunk_chars=self.config.workspace_chunk_chars,
                overlap_chars=self.config.workspace_chunk_overlap,
                chunk_prefix=resolved,
            )
            for chunk in chunks:
                texts.append(chunk.text)
                metadatas.append(
                    {
                        "source_kind": "file",
                        "source_path": resolved,
                        "label": path.name,
                        "chunk_text": chunk.text,
                        "chunk_id": chunk.chunk_id,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                )
            try:
                stat = path.stat()
            except OSError:
                continue
            manifest_rows[resolved] = {
                "fingerprint": f"{stat.st_mtime_ns}:{stat.st_size}",
                "name": path.name,
                "kind": detect_media_kind(path),
            }
        return texts, metadatas, manifest_rows

    def _rebuild_index(self, *, paths: list[Path] | None = None) -> None:
        """Recreate the workspace vector store from the current candidate files."""

        paths = list(paths or self._iter_candidate_files())
        texts, metadatas, new_manifest = self._prepare_documents(paths)
        embeddings = self.embedder.encode_documents(texts) if metadatas else None
        self._build_and_activate_index(
            reused_embeddings=None,
            reused_metadatas=[],
            changed_embeddings=embeddings,
            changed_metadatas=metadatas,
            manifest=new_manifest,
        )

    def _build_and_activate_index(
        self,
        *,
        reused_embeddings,
        reused_metadatas: list[dict[str, str | int]],
        changed_embeddings,
        changed_metadatas: list[dict[str, str | int]],
        manifest: dict[str, dict[str, str | int]],
    ) -> None:
        """Build a fresh index off to the side and atomically swap it into place."""

        with TemporaryDirectory(dir=str(self.root), prefix="workspace-build-") as temp_dir:
            build_root = Path(temp_dir)
            next_store = PersistentFaissStore(build_root, "workspace")
            next_store.reset()
            if reused_metadatas:
                next_store.add(reused_embeddings, reused_metadatas, save=False)
            if changed_metadatas and changed_embeddings is not None:
                next_store.add(changed_embeddings, changed_metadatas, save=False)
            if reused_metadatas or changed_metadatas:
                next_store.save()
            manifest_path = build_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            with self._swap_lock:
                for filename in ("workspace.faiss", "workspace.json"):
                    target = self.root / filename
                    built = build_root / filename
                    if built.exists():
                        built.replace(target)
                    elif target.exists():
                        target.unlink()
                manifest_path.replace(self.manifest_path)
                self.store = PersistentFaissStore(self.root, "workspace")
                self.manifest = manifest

    def index_paths(self, paths: list[Path], *, parsed_texts: dict[str, str] | None = None) -> list[str]:
        """Index explicit document attachments without rebuilding the whole workspace."""

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
        """Return the most relevant workspace chunks for a query."""

        if not query.strip():
            return []
        with self._swap_lock:
            store = self.store
        # Avoid loading the embedding model when the workspace index is still empty.
        if store.index is None or not store.metadata:
            return []
        vector = self.embedder.encode_query(query)
        rows = store.search(vector, top_k)
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
        """Read explicitly mentioned file paths into high-priority prompt context."""

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
        """Recover the chunk span text from the original source file."""

        path = Path(source_path)
        try:
            text = extract_text_from_path(path)
        except Exception:
            return ""
        cleaned = " ".join(text.split())
        return cleaned[start_char:end_char]

    @staticmethod
    def extract_path_mentions(text: str) -> list[Path]:
        """Find quoted or path-like local filesystem references in user text."""

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
