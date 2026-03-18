# Anveshak Architecture

This document explains how the codebase is organized after the `anveshak` package refactor.

## Top-Level Entry Points

- `anveshak/cli.py`
  Builds the installable `anveshak` command, parses CLI arguments, and launches web, terminal, or both modes.
- `anveshak/__main__.py`
  Enables `python -m anveshak`.
- `main.py`
  Legacy source-checkout wrapper that simply forwards to `anveshak.cli.main()`.

## Core Runtime Flow

1. `anveshak.cli` parses arguments and builds `RuntimeConfig`.
2. `anveshak.chat.service.ChatService` wires together:
   - runtime preparation
   - the reasoning model runner
   - embedding model
   - workspace retrieval
   - memory retrieval
   - active web retrieval
   - API-call presets
   - run logging
3. `anveshak.server.build_app()` exposes the browser UI and JSON/SSE API.
4. `anveshak.terminal.TerminalChat` exposes the terminal REPL.

## Important Packages

- `anveshak/chat/`
  Conversation orchestration, session management, attachment preparation, and run control.
- `anveshak/modeling/`
  The reasoning model adapter layer. This is where model-family-specific parsing and generation logic lives.
- `anveshak/retrieval/`
  Retrieval subsystems for workspace files, live web search, embeddings, memory, and FAISS persistence.
- `anveshak/static/`
  Browser UI assets packaged with the library.

## Files Most Contributors Will Touch

- `anveshak/model_catalog.py`
  Add or edit supported models and modality metadata.
- `anveshak/modeling/qwen_runner.py`
  Add input adapters or generation-time behaviors for model families.
- `anveshak/retrieval/active_search.py`
  Change the active web retrieval loop.
- `anveshak/retrieval/web.py`
  Change search providers or evidence fetching.
- `anveshak/retrieval/memory.py`
  Change long-term memory retrieval and ranking.
- `anveshak/retrieval/embeddings.py`
  Change the embedding backend.
- `anveshak/file_parsers.py`
  Add or improve file/document parsing.
- `tests/`
  Regression and unit tests.

## Local State Layout

- `checkpoints/`
  Downloaded model assets and Hugging Face cache data.
- `context_window/`
  Persistent session transcripts, file index state, uploads, caches, and memory state.
- `API_calls/`
  Stored API-style preset definitions.
- `logs/`
  One JSONL file per run.

## Packaging Notes

- The package import path is now `anveshak`.
- The installable CLI is `anveshak`.
- Static browser assets are loaded from the installed package, not from a hard-coded repo-relative path.

## Recommended Reading Order

If you are new to the repository, read the code in this order:

1. `anveshak/cli.py`
2. `anveshak/config.py`
3. `anveshak/chat/service.py`
4. `anveshak/modeling/qwen_runner.py`
5. `anveshak/retrieval/`
6. `anveshak/server.py`
7. `tests/`
