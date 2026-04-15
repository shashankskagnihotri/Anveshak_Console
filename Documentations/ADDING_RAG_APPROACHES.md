# Adding RAG Approaches

This guide explains how to extend Anveshak with new retrieval strategies.

## Current Retrieval Layers

Anveshak currently has three major retrieval paths:

- workspace retrieval
  `anveshak/retrieval/workspace.py`
- live web retrieval
  `anveshak/retrieval/web.py`
- long-term memory retrieval
  `anveshak/retrieval/memory.py`

The orchestration layer that decides how these sources are combined lives in:

- `anveshak/chat/service.py`

The active multi-round web loop lives in:

- `anveshak/retrieval/active_search.py`

Inline web image/video previews are also curated in:

- `anveshak/retrieval/web.py`
- `anveshak/static/app.js`

## Where to Add a New RAG Strategy

Use this rule of thumb:

- new retrieval source
  Add a new module under `anveshak/retrieval/`
- new orchestration policy
  Update `anveshak/chat/service.py`
- new chunking strategy
  Update `anveshak/chunking.py` or add a new chunker and wire it in
- new ranking scheme
  Update the relevant retrieval module

## Recommended Pattern

1. Create a retrieval module under `anveshak/retrieval/`
2. Return results as `RetrievedChunk` objects from `anveshak/schema.py`
3. Keep source metadata rich enough for later citations
4. Merge the new chunks inside `anveshak/chat/service.py`
5. Add tests

The system expects retrieval outputs to be shaped like:

- `source_id`
- `source_kind`
- `label`
- `text`
- `score`
- `metadata`

## Example Extension Ideas

- replace DDGS with another search backend
- add arXiv-specific retrieval
- add paper metadata indexing
- add OCR-backed scanned-document retrieval
- add hybrid reranking with a cross-encoder
- add a different active-retrieval planner

## Where to Merge New Evidence

The main merge points are:

- `_merge_chunks(...)` in `anveshak/chat/service.py`
- `_compose_answer_prompt(...)` in `anveshak/modeling/qwen_runner.py`

If you add a new evidence source, you usually need both:

- retrieval-time ranking
- prompt-time rendering
- and, if the source has a user-facing preview, browser rendering plus any required safety policy

## Testing Expectations

For each new RAG strategy, add:

- one unit test for ranking or retrieval behavior
- one orchestration test if the new source changes run control
- documentation in `README.md` if the feature is user-facing

## Keep the Contract Stable

Try not to break:

- `RetrievedChunk`
- citation payloads
- run logging
- browser event streaming

Those are the contracts that the UI and logs depend on.
