# Changing or Adding Embedding Models

This guide explains how the shared embedding backend works and how to change it.

## Current Embedding Path

The shared embedding adapter lives in:

- `anveshak/retrieval/embeddings.py`

It is used by:

- workspace retrieval
- long-term memory retrieval
- live web retrieval

That means one embedding change can affect all RAG paths at once.

## Default Behavior

By default Anveshak uses:

- `Qwen/Qwen3-Embedding-0.6B`

The embedding model is configured through:

- `RuntimeConfig.embedding_model_id`
- CLI flag: `--embedding-model-id`

## How to Swap the Embedding Model

If the new model is still compatible with `sentence-transformers`, you may only need:

1. update the default in `anveshak/config.py`
2. update `README.md`
3. update tests if any expected values depend on embedding behavior

## When More Code Changes Are Needed

You will likely need code changes if the new embedding backend:

- does not use `SentenceTransformer`
- needs a custom prompt format
- needs a different batching strategy
- needs GPU placement instead of CPU
- returns embeddings in a different shape or dtype

In those cases, update:

- `anveshak/retrieval/embeddings.py`

You may want to:

- rename `QwenEmbeddingModel` to a more generic adapter name in a future cleanup
- keep a thin compatibility wrapper so the rest of the codebase does not change

## What Must Stay Stable

The retrieval layers expect:

- `encode_documents(texts)` returning a rank-2 `float32` array
- `encode_query(text)` returning a rank-1 `float32` array
- normalized embeddings for cosine / inner-product retrieval

If you change those assumptions, also update:

- `anveshak/retrieval/vector_store.py`
- retrieval tests

## Good Verification Steps

- run `pytest`
- verify local file retrieval still returns results
- verify memory retrieval still returns results
- verify web retrieval still ranks fetched chunks

## Suggested Follow-Up Docs

If you add a significantly different embedding backend, update:

- `README.md`
- `Documentations/ARCHITECTURE.md`
- `debugging/anveshak_console_jmlr_mloss_report.tex` if the change is important for the software paper
