from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def extract_json_object(text: str) -> dict[str, Any]:
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def tokenize_for_bm25(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper == lower:
        return [1.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def compact_whitespace(text: str) -> str:
    return " ".join(text.split())


def apply_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:
        from transformers import set_seed as set_transformers_seed

        set_transformers_seed(seed)
    except Exception:
        pass


def bm25_scores(
    tokenized_corpus: list[list[str]],
    query_tokens: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not tokenized_corpus:
        return []

    doc_lengths = [len(doc) for doc in tokenized_corpus]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    vocab_doc_counts: dict[str, int] = {}
    for doc in tokenized_corpus:
        for token in set(doc):
            vocab_doc_counts[token] = vocab_doc_counts.get(token, 0) + 1

    scores: list[float] = []
    total_docs = len(tokenized_corpus)
    for doc, doc_length in zip(tokenized_corpus, doc_lengths, strict=True):
        term_counts: dict[str, int] = {}
        for token in doc:
            term_counts[token] = term_counts.get(token, 0) + 1
        score = 0.0
        for token in query_tokens:
            df = vocab_doc_counts.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            freq = term_counts.get(token, 0)
            denominator = freq + k1 * (1 - b + b * (doc_length / avgdl if avgdl else 0))
            if denominator == 0:
                continue
            score += idf * ((freq * (k1 + 1)) / denominator)
        scores.append(score)
    return scores
