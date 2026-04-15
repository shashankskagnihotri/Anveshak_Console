"""Active web search, media discovery, and indexing helpers used during live retrieval."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from PIL import Image

try:  # pragma: no cover - optional dependency is exercised via monkeypatch in tests.
    from ddgs import DDGS as _DDGS
except Exception:  # pragma: no cover - absence is handled gracefully at runtime.
    _DDGS = None

try:  # pragma: no cover - optional dependency is exercised indirectly in integration use.
    import trafilatura as _trafilatura
except Exception:  # pragma: no cover - absence is handled gracefully at runtime.
    _trafilatura = None

from ..chunking import chunk_text
from ..config import RuntimeConfig
from ..schema import Attachment, RetrievedChunk, WebMediaResult
from .embeddings import QwenEmbeddingModel
from .vector_store import PersistentFaissStore

DDGS = _DDGS
trafilatura = _trafilatura

_MEDIA_REQUEST_HEADERS = {
    "User-Agent": "Anveshak Console/0.1 (+https://github.com/shashankskagnihotri/Anveshak_Console)",
}
_MEDIA_DOWNLOAD_LIMIT_BYTES = 6 * 1024 * 1024
_MEDIA_TIMEOUT_SECONDS = 6.0
_MAX_MEDIA_QUERIES = 2
_MAX_MEDIA_CANDIDATES_PER_KIND = 6
_UNSAFE_MEDIA_TERMS = {
    "adult",
    "assault",
    "blood",
    "corpse",
    "death",
    "decapitation",
    "erotic",
    "execution",
    "gore",
    "graphic",
    "kill",
    "murder",
    "nude",
    "nudity",
    "nsfw",
    "porn",
    "pornographic",
    "sex",
    "sexual",
    "shooting",
    "slaughter",
    "suicide",
    "torture",
    "violence",
    "violent",
}
_WEB_MEDIA_REQUEST_PATTERNS = (
    re.compile(r"\bshow me (an? )?(image|images|picture|pictures|photo|photos|video|videos)\b"),
    re.compile(r"\b(find|get|fetch|display|embed|open) (me )?(an? )?(image|images|picture|pictures|photo|photos|video|videos)\b"),
    re.compile(r"\b(image|images|picture|pictures|photo|photos|video|videos|footage|thumbnail)s? of\b"),
    re.compile(r"\b(youtube|vimeo)\b"),
)


class WebIndexer:
    """Fetch, chunk, embed, rank, and safely preview live web results for a query."""

    def __init__(self, config: RuntimeConfig, embedder: QwenEmbeddingModel) -> None:
        self.config = config
        self.embedder = embedder
        self.root = config.cache_dir / "web"
        self.root.mkdir(parents=True, exist_ok=True)
        self.media_root = self.root / "media"
        self.media_root.mkdir(parents=True, exist_ok=True)
        self.media_download_dir = self.media_root / "downloads"
        self.media_download_dir.mkdir(parents=True, exist_ok=True)
        self.media_moderation_dir = self.media_root / "moderation"
        self.media_moderation_dir.mkdir(parents=True, exist_ok=True)

    def search_and_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run a text search query and return ranked web chunks ready for prompting."""

        results = self._search_results(query)
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

    def search_media(
        self,
        queries: list[str],
        *,
        profile: dict[str, Any],
        runner: Any | None,
        media_mode: str = "safe",
        max_images: int = 3,
        max_videos: int = 2,
    ) -> list[WebMediaResult]:
        """Discover a small set of relevant remote images and videos for inline chat previews."""

        if DDGS is None:
            return []

        normalized_queries = [
            query.strip()
            for query in queries[:_MAX_MEDIA_QUERIES]
            if isinstance(query, str) and query.strip()
        ]
        if not normalized_queries:
            return []

        safesearch = "moderate" if media_mode == "safe" else "off"
        candidates: list[WebMediaResult] = []
        for query in normalized_queries:
            image_rows = self._search_media_rows(query=query, kind="images", safesearch=safesearch)
            for row_index, row in enumerate(image_rows, start=1):
                candidate = self._normalize_image_result(row=row, query=query, row_index=row_index)
                if candidate is not None:
                    candidates.append(candidate)

            video_rows = self._search_media_rows(query=query, kind="videos", safesearch=safesearch)
            for row_index, row in enumerate(video_rows, start=1):
                candidate = self._normalize_video_result(row=row, query=query, row_index=row_index)
                if candidate is not None:
                    candidates.append(candidate)

        curated: list[WebMediaResult] = []
        image_count = 0
        video_count = 0
        for candidate in self._dedupe_media_results(candidates):
            if candidate.kind == "image" and image_count >= max_images:
                continue
            if candidate.kind == "video" and video_count >= max_videos:
                continue
            allowed, evaluated = self._evaluate_media_candidate(
                candidate,
                profile=profile,
                runner=runner,
                media_mode=media_mode,
            )
            if not allowed:
                continue
            curated.append(evaluated)
            if evaluated.kind == "image":
                image_count += 1
            else:
                video_count += 1
            if image_count >= max_images and video_count >= max_videos:
                break
        return curated

    def _search_results(self, query: str) -> list[dict[str, Any]]:
        """Fetch text-search results with caching and light retries for stability."""

        cache_path = self.root / f"search-{hashlib.sha1(query.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_results = payload.get("results", [])
                if isinstance(cached_results, list):
                    return cached_results
            except Exception:
                pass

        if DDGS is None:
            return []

        for attempt in range(3):
            try:
                results = list(
                    DDGS().text(
                        query=query,
                        max_results=self.config.max_web_pages_per_query,
                        safesearch="moderate",
                        backend="auto",
                    )
                )
                cache_path.write_text(
                    json.dumps({"query": query, "results": results}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return results
            except Exception:
                if attempt == 2:
                    break
                time.sleep(0.75 * (attempt + 1))
        return []

    def _fetch_page(self, url: str) -> str:
        """Fetch and cache a cleaned representation of a web page."""

        cache_path = self.root / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return payload.get("text", "")
            except Exception:
                pass

        if trafilatura is None:
            return ""

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_links=True) or ""
        cache_path.write_text(json.dumps({"url": url, "text": text}, ensure_ascii=False), encoding="utf-8")
        return text

    def _search_media_rows(self, *, query: str, kind: str, safesearch: str) -> list[dict[str, Any]]:
        """Fetch cached DDGS image or video results for one query."""

        cache_key = hashlib.sha1(f"{kind}|{query}|{safesearch}".encode("utf-8")).hexdigest()
        cache_path = self.media_root / f"{kind}-{cache_key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                rows = payload.get("results", [])
                if isinstance(rows, list):
                    return rows
            except Exception:
                pass

        if DDGS is None:
            return []

        for attempt in range(2):
            try:
                ddgs = DDGS()
                search_method = getattr(ddgs, kind, None)
                if search_method is None:
                    return []
                rows = list(
                    search_method(
                        query=query,
                        safesearch=safesearch,
                        max_results=_MAX_MEDIA_CANDIDATES_PER_KIND,
                        backend="auto",
                    )
                )
                cache_path.write_text(
                    json.dumps({"query": query, "kind": kind, "results": rows}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return rows
            except Exception:
                if attempt == 1:
                    break
                time.sleep(0.5 * (attempt + 1))
        return []

    def _normalize_image_result(self, *, row: dict[str, Any], query: str, row_index: int) -> WebMediaResult | None:
        """Map one DDGS image-search row into the UI-friendly media shape."""

        content_url = _first_str(row, "image", "url", "href")
        preview_url = _first_str(row, "thumbnail", "image", "url")
        if not content_url or not preview_url:
            return None
        page_url = _first_str(row, "url", "href") or content_url
        title = _first_str(row, "title", "name") or f"Image result {row_index}"
        source_label = _first_str(row, "source", "provider") or _domain_from_url(page_url) or "Web"
        snippet = _first_str(row, "body", "description")
        return WebMediaResult(
            media_id=f"image-{hashlib.sha1(f'{query}|{content_url}'.encode('utf-8')).hexdigest()[:12]}",
            kind="image",
            title=title,
            content_url=content_url,
            preview_url=preview_url,
            page_url=page_url,
            snippet=snippet,
            source_label=source_label,
            metadata={
                "query": query,
                "provider": source_label,
                "width": row.get("width"),
                "height": row.get("height"),
                "raw": row,
            },
        )

    def _normalize_video_result(self, *, row: dict[str, Any], query: str, row_index: int) -> WebMediaResult | None:
        """Map one DDGS video-search row into the UI-friendly media shape."""

        content_url = _first_str(row, "content", "url", "href")
        preview_url = (
            _first_str(row, "thumbnail", "image")
            or _nested_str(row, "images", "large")
            or _nested_str(row, "images", "medium")
            or _nested_str(row, "images", "small")
        )
        if not content_url or not preview_url:
            return None
        title = _first_str(row, "title", "name") or f"Video result {row_index}"
        source_label = _first_str(row, "publisher", "provider", "source") or _domain_from_url(content_url) or "Web"
        snippet = _first_str(row, "description", "body")
        embed_url = _first_str(row, "embed_url") or _youtube_embed_url(content_url)
        return WebMediaResult(
            media_id=f"video-{hashlib.sha1(f'{query}|{content_url}'.encode('utf-8')).hexdigest()[:12]}",
            kind="video",
            title=title,
            content_url=content_url,
            preview_url=preview_url,
            page_url=content_url,
            snippet=snippet,
            source_label=source_label,
            embed_url=embed_url,
            metadata={
                "query": query,
                "provider": _first_str(row, "provider"),
                "publisher": _first_str(row, "publisher"),
                "duration": _first_str(row, "duration"),
                "published": _first_str(row, "published"),
                "uploader": _first_str(row, "uploader"),
                "raw": row,
            },
        )

    def _dedupe_media_results(self, items: list[WebMediaResult]) -> list[WebMediaResult]:
        """Remove repeated media results while preserving the earliest useful copy."""

        deduped: list[WebMediaResult] = []
        seen: set[str] = set()
        for item in items:
            key = f"{item.kind}|{item.content_url}|{item.preview_url}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _evaluate_media_candidate(
        self,
        candidate: WebMediaResult,
        *,
        profile: dict[str, Any],
        runner: Any | None,
        media_mode: str,
    ) -> tuple[bool, WebMediaResult]:
        """Apply safe-mode moderation rules before a remote media result reaches the UI."""

        cached = self._load_moderation_cache(candidate=candidate, profile=profile, media_mode=media_mode)
        if cached is not None:
            candidate.safety_mode = cached.get("safety_mode", media_mode)
            candidate.safety_state = cached.get("safety_state", "allowed")
            candidate.safety_reason = cached.get("safety_reason", "")
            candidate.metadata.update(
                {
                    "safety_checked_with": cached.get("safety_checked_with"),
                    "safety_cached": True,
                }
            )
            return candidate.safety_state != "blocked", candidate

        if media_mode != "safe":
            candidate.safety_mode = "unrestricted"
            candidate.safety_state = "unrestricted"
            candidate.safety_reason = "Unrestricted mode skips safety checks."
            candidate.metadata["safety_checked_with"] = "disabled"
            self._store_moderation_cache(candidate, profile=profile, checked_with="disabled")
            return True, candidate

        blocked_term = _unsafe_keyword_hit(candidate)
        if blocked_term:
            candidate.safety_mode = "safe"
            candidate.safety_state = "blocked"
            candidate.safety_reason = f'Blocked by metadata term "{blocked_term}".'
            candidate.metadata["safety_checked_with"] = "metadata-keyword"
            self._store_moderation_cache(candidate, profile=profile, checked_with="metadata-keyword")
            return False, candidate

        checked_with = "metadata"
        verdict: dict[str, Any]
        local_preview = None
        if profile.get("supports_images"):
            local_preview = self._download_preview_image(candidate)
        if local_preview is not None:
            checked_with = "vision"
            verdict = self._classify_visual_safety(candidate=candidate, preview_path=local_preview, runner=runner)
        else:
            verdict = self._classify_metadata_safety(candidate=candidate, runner=runner)

        candidate.safety_mode = "safe"
        candidate.safety_state = "allowed" if verdict.get("safe") else "blocked"
        candidate.safety_reason = str(verdict.get("reason") or "Safety screening finished.")
        candidate.metadata.update(
            {
                "safety_checked_with": checked_with,
                "safety_category": verdict.get("category", "unknown"),
                "safety_confidence": verdict.get("confidence", ""),
            }
        )
        self._store_moderation_cache(candidate, profile=profile, checked_with=checked_with)
        return candidate.safety_state != "blocked", candidate

    def _classify_visual_safety(
        self,
        *,
        candidate: WebMediaResult,
        preview_path: Path,
        runner: Any | None,
    ) -> dict[str, Any]:
        """Ask a multimodal model to inspect a preview image directly for obvious safety issues."""

        if runner is None:
            return {
                "safe": False,
                "category": "unknown",
                "confidence": "low",
                "reason": "The vision safety check was unavailable, so the preview stayed hidden.",
            }
        attachment = Attachment.from_path(preview_path, media_kind="image", source="web-safe-check")
        prompt = f"""
Inspect the attached web-media preview and decide whether it is safe to display inside a general-purpose chat UI for all audiences.

Return JSON with:
- safe: boolean
- category: one of safe, nudity, sexual_content, graphic_violence, violence, disturbing, unknown
- confidence: one of low, medium, high
- reason: short string, at most 18 words

Rules:
- If the preview contains or strongly suggests nudity, sexual content, gore, graphic injury, or explicit violence, set safe to false.
- If you are unsure, set safe to false.
- Use the metadata only as supporting context, not as a substitute for the image itself.

Metadata:
Title: {candidate.title}
Source: {candidate.source_label}
Snippet: {candidate.snippet or "-"}
Link: {candidate.content_url}
""".strip()
        try:
            payload = runner.generate_json(prompt, max_new_tokens=220, attachments=[attachment])
        except Exception:
            return {
                "safe": False,
                "category": "unknown",
                "confidence": "low",
                "reason": "The vision safety check failed, so the preview stayed hidden.",
            }
        return _normalize_safety_payload(payload)

    def _classify_metadata_safety(self, *, candidate: WebMediaResult, runner: Any | None) -> dict[str, Any]:
        """Ask a text-only model to judge safety based only on metadata and source cues."""

        if runner is None:
            return {
                "safe": False,
                "category": "unknown",
                "confidence": "low",
                "reason": "The metadata safety check was unavailable, so the preview stayed hidden.",
            }
        prompt = f"""
Using only the metadata below, decide whether this remote media result is safe to display inside a general-purpose chat UI for all audiences.

Return JSON with:
- safe: boolean
- category: one of safe, nudity, sexual_content, graphic_violence, violence, disturbing, unknown
- confidence: one of low, medium, high
- reason: short string, at most 18 words

Rules:
- You cannot see the pixels. Judge only from the metadata, title, description, source, and URL.
- If the metadata hints at nudity, explicit sexual content, gore, graphic injury, or violence, set safe to false.
- If the metadata is too weak or ambiguous to be sure, set safe to false.

Metadata:
Kind: {candidate.kind}
Title: {candidate.title}
Source: {candidate.source_label}
Snippet: {candidate.snippet or "-"}
Content URL: {candidate.content_url}
Page URL: {candidate.page_url or "-"}
""".strip()
        try:
            payload = runner.generate_json(prompt, max_new_tokens=180)
        except Exception:
            return {
                "safe": False,
                "category": "unknown",
                "confidence": "low",
                "reason": "The metadata safety check failed, so the preview stayed hidden.",
            }
        return _normalize_safety_payload(payload)

    def _download_preview_image(self, candidate: WebMediaResult) -> Path | None:
        """Download one remote preview image so multimodal safe mode can inspect actual pixels."""

        url_candidates: list[str] = []
        if candidate.kind == "image":
            url_candidates.extend([candidate.content_url, candidate.preview_url])
        else:
            url_candidates.append(candidate.preview_url)

        seen: set[str] = set()
        for url in url_candidates:
            if not url or url in seen or not url.startswith(("http://", "https://")):
                continue
            seen.add(url)
            downloaded = self._download_remote_image(url)
            if downloaded is not None:
                return downloaded
        return None

    def _download_remote_image(self, url: str) -> Path | None:
        """Fetch and cache a remote preview image with a small size cap."""

        cache_key = hashlib.sha1(url.encode("utf-8")).hexdigest()
        manifest_path = self.media_download_dir / f"{cache_key}.json"
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                target_path = Path(payload.get("path", ""))
                if target_path.exists():
                    return target_path
            except Exception:
                pass

        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=_MEDIA_TIMEOUT_SECONDS,
                headers=_MEDIA_REQUEST_HEADERS,
            ) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if content_type and not content_type.lower().startswith("image/"):
                        return None
                    buffer = bytearray()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        buffer.extend(chunk)
                        if len(buffer) > _MEDIA_DOWNLOAD_LIMIT_BYTES:
                            return None
                    if not buffer:
                        return None
                    final_url = str(response.url)
        except Exception:
            return None

        extension = _guess_image_extension(url=final_url, content_type=content_type)
        target_path = self.media_download_dir / f"{cache_key}{extension}"
        target_path.write_bytes(buffer)
        try:
            with Image.open(target_path) as image:
                image.verify()
        except Exception:
            target_path.unlink(missing_ok=True)
            return None

        manifest_path.write_text(
            json.dumps({"url": url, "path": str(target_path.resolve())}, ensure_ascii=False),
            encoding="utf-8",
        )
        return target_path

    def _moderation_cache_path(self, *, candidate: WebMediaResult, profile: dict[str, Any], media_mode: str) -> Path:
        """Return the cache path for one safe-mode moderation decision."""

        fingerprint = "|".join(
            [
                profile.get("model_id", "unknown"),
                media_mode,
                candidate.kind,
                candidate.content_url,
                candidate.preview_url,
                candidate.title,
                candidate.snippet,
            ]
        )
        cache_key = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
        return self.media_moderation_dir / f"{cache_key}.json"

    def _load_moderation_cache(
        self,
        *,
        candidate: WebMediaResult,
        profile: dict[str, Any],
        media_mode: str,
    ) -> dict[str, Any] | None:
        """Restore a previous moderation result when the same preview is requested again."""

        cache_path = self._moderation_cache_path(candidate=candidate, profile=profile, media_mode=media_mode)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _store_moderation_cache(
        self,
        candidate: WebMediaResult,
        *,
        profile: dict[str, Any],
        checked_with: str,
    ) -> None:
        """Persist one moderation result so repeated prompts stay fast."""

        media_mode = candidate.safety_mode or "safe"
        cache_path = self._moderation_cache_path(candidate=candidate, profile=profile, media_mode=media_mode)
        cache_path.write_text(
            json.dumps(
                {
                    "safety_mode": candidate.safety_mode,
                    "safety_state": candidate.safety_state,
                    "safety_reason": candidate.safety_reason,
                    "safety_checked_with": checked_with,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def should_use_web(query: str, mode: str) -> bool:
    """Decide whether a prompt likely needs live web retrieval."""

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
    return (
        any(keyword in lowered for keyword in keywords)
        or "http://" in lowered
        or "https://" in lowered
        or _looks_like_web_media_request(lowered)
    )


def _looks_like_web_media_request(lowered_query: str) -> bool:
    """Recognize prompts that explicitly ask Anveshak to fetch a web image or video."""

    return any(pattern.search(lowered_query) for pattern in _WEB_MEDIA_REQUEST_PATTERNS)


def _first_str(row: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string-like value from a row."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def _nested_str(row: dict[str, Any], *keys: str) -> str:
    """Return a nested string value from a dictionary path."""

    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    if value is None:
        return ""
    cleaned = str(value).strip()
    return cleaned


def _domain_from_url(url: str) -> str:
    """Extract a human-friendly hostname label from a URL."""

    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return ""
    return hostname.removeprefix("www.")


def _youtube_embed_url(url: str) -> str | None:
    """Convert common YouTube URLs into an embeddable no-cookie iframe URL."""

    try:
        parsed = urlparse(url)
    except Exception:
        return None
    hostname = (parsed.hostname or "").lower()
    if hostname.endswith("youtu.be"):
        video_id = parsed.path.strip("/")
    elif "youtube.com" in hostname:
        if parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
        else:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        return None
    if not video_id:
        return None
    return f"https://www.youtube-nocookie.com/embed/{video_id}"


def _guess_image_extension(*, url: str, content_type: str) -> str:
    """Choose a useful filename extension for a downloaded remote image."""

    mime = content_type.split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_extension(mime) if mime else None
    if guessed:
        if guessed == ".jpe":
            return ".jpg"
        return guessed
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}:
        return suffix
    return ".jpg"


def _unsafe_keyword_hit(candidate: WebMediaResult) -> str | None:
    """Catch obviously unsafe metadata before invoking a model-based safety check."""

    haystack = " ".join(
        [
            candidate.title,
            candidate.snippet,
            candidate.source_label,
            candidate.content_url,
            candidate.page_url,
        ]
    ).lower()
    for term in sorted(_UNSAFE_MEDIA_TERMS):
        if term in haystack:
            return term
    return None


def _coerce_bool(value: Any) -> bool:
    """Convert JSON-ish booleans into a strict Python boolean."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "safe", "allowed"}
    return bool(value)


def _normalize_safety_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one model-produced safety JSON payload into a stable shape."""

    if not isinstance(payload, dict):
        return {
            "safe": False,
            "category": "unknown",
            "confidence": "low",
            "reason": "The safety checker returned an invalid response.",
        }
    reason = str(payload.get("reason") or "").strip() or "Safety screening finished."
    return {
        "safe": _coerce_bool(payload.get("safe")),
        "category": str(payload.get("category") or "unknown").strip() or "unknown",
        "confidence": str(payload.get("confidence") or "low").strip() or "low",
        "reason": reason,
    }
