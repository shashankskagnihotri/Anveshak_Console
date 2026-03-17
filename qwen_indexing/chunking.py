from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int


def chunk_text(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
    chunk_prefix: str,
) -> list[TextChunk]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be smaller than chunk_chars")

    chunks: list[TextChunk] = []
    start = 0
    step = chunk_chars - overlap_chars

    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_chars)
        window = cleaned[start:end].strip()
        if window:
            digest = sha1(f"{chunk_prefix}:{start}:{end}:{window}".encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=digest,
                    text=window,
                    start_char=start,
                    end_char=end,
                )
            )
        if end >= len(cleaned):
            break
        start += step

    return chunks
