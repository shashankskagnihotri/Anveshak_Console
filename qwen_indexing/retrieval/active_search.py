from __future__ import annotations

from dataclasses import dataclass, field

from ..events import RunHandle
from ..schema import RetrievedChunk
from .web import WebIndexer


@dataclass(slots=True)
class SearchPlan:
    enabled: bool
    rationale: str
    search_queries: list[str] = field(default_factory=list)
    max_rounds: int = 2


@dataclass(slots=True)
class SearchRound:
    round_index: int
    query: str
    retrieved_chunks: list[RetrievedChunk]


class ActiveSearchOrchestrator:
    def __init__(self, web_indexer: WebIndexer) -> None:
        self.web_indexer = web_indexer

    def run(
        self,
        *,
        user_query: str,
        plan: SearchPlan,
        model_runner,
        handle: RunHandle,
    ) -> tuple[list[RetrievedChunk], list[SearchRound]]:
        if not plan.enabled or not plan.search_queries:
            return [], []

        rounds: list[SearchRound] = []
        gathered: list[RetrievedChunk] = []
        current_queries = list(plan.search_queries)

        for round_index in range(1, max(plan.max_rounds, 1) + 1):
            if not current_queries:
                break
            if handle.has_pending_restart():
                break
            handle.emit(
                "status",
                phase="web-search",
                text=f"Active web search round {round_index}: {' | '.join(current_queries[:3])}",
            )
            round_chunks: list[RetrievedChunk] = []
            for query in current_queries[:3]:
                retrieved = self.web_indexer.search_and_retrieve(query, self.web_indexer.config.web_top_k)
                round_chunks.extend(retrieved)
            deduped = _dedupe_chunks(round_chunks)
            rounds.append(SearchRound(round_index=round_index, query=" | ".join(current_queries[:3]), retrieved_chunks=deduped))
            gathered.extend(deduped)
            gathered = _dedupe_chunks(gathered)[: self.web_indexer.config.web_top_k]

            if round_index >= max(plan.max_rounds, 1):
                break
            if getattr(model_runner, "model", None) is None or not hasattr(model_runner, "plan_follow_up_searches"):
                break

            follow_up = model_runner.plan_follow_up_searches(
                user_query=user_query,
                current_queries=current_queries,
                evidence=gathered,
            )
            current_queries = follow_up
            if not current_queries:
                break

        return gathered, rounds


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
        key = chunk.metadata.get("url", "") + "|" + chunk.text[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped
