# Anveshak Console Overview

## What This Project Is

Anveshak Console is a local-first multimodal research assistant for users who want open-weight models, private local files, durable memory, and live web retrieval in one inspectable stack.

It is not only a chat UI. It is a full local application with:

- a browser console
- a terminal console
- persistent sessions
- local file retrieval
- long-term memory
- live web retrieval
- reusable API-call workflows

The project is designed so that the reasoning model, retrieval stack, memory stack, UI, and API surface are all visible in code and can be modified independently.

## What It Can Do

- Run large open Hugging Face models locally
- Use retrieved local-file evidence during answer generation
- Parse PDFs and other document-like files into retrieval-ready text
- Extract page previews or document visuals when the selected model can use images
- Search the web live during a run
- Keep durable long-term user memory across sessions
- Let users steer a run while the model is generating
- Expose saved local workflows through generated API keys
- Keep per-run logs for debugging and reproducibility

## What It Cannot Do

- It does not make every supported model equally fast or equally practical on one machine
- It cannot bypass upstream gated-model restrictions or missing checkpoint access
- It does not magically make text-only models natively understand PDFs as documents; those are parsed into text context
- It does not remove the hardware cost of very large checkpoints
- It does not guarantee identical behavior across every model family, because processor and remote-code behavior differs across upstream repos

## Main Components

### Runtime

`main.py`, `anveshak/config.py`, and `anveshak/runtime.py`

These files handle:

- launch mode selection
- directory layout
- checkpoint preparation
- local staging and cache management
- single-node GPU visibility control
- runtime status tracking
- startup progress reporting to the browser

The runtime layer is responsible for making the rest of the application possible before the first prompt is even sent.

### Chat Orchestration

`anveshak/chat/service.py`

This is the central coordinator. It handles:

- sessions
- attachments
- retrieval
- streaming answers
- steering
- memory updates
- saved API calls

This file is where the end-to-end behavior of a run is decided. It is the place where attachments, retrieval, model execution, logging, session state, steering, and background memory compression come together.

### Model Execution

`anveshak/modeling/factory.py`

Chooses the correct backend for the configured reasoning model.

`anveshak/modeling/qwen_runner.py`

This is the main local runner for Hugging Face model families. It handles:

- tokenizer and processor loading
- prompt composition
- streaming generation
- JSON generation
- search planning
- memory summarization

`anveshak/modeling/kimi_server_runner.py`

This is the dedicated Kimi backend for OpenAI-compatible local servers. It exists because Kimi is better served through a compatible backend than through the slow generic in-process Hugging Face path on the target hardware.

### Retrieval

`anveshak/retrieval/`

This directory contains:

- `workspace.py` for local file indexing and retrieval
- `web.py` for live web search, extraction, and ranking
- `active_search.py` for multi-round search orchestration
- `memory.py` for persistent long-term memory retrieval
- `vector_store.py` for persistent FAISS-backed dense storage

The retrieval layer is not one single RAG module. It is a coordinated collection of subsystems for local files, web evidence, and long-term memory, each with different data lifecycles and ranking behavior.

### API-Call Persistence

`anveshak/api_calls.py`

This layer stores reusable API workflows and their metadata, including:

- generated API keys
- saved reasoning-model snapshot
- saved embedding-model snapshot
- response mode
- internet policy
- `Use User Context`
- invocation memory mode

It also supports listing, editing, deleting, and resolving saved calls during programmatic invocation.

### Interfaces

`anveshak/server.py`, `anveshak/static/`, and `anveshak/terminal.py`

These files power the browser UI, SSE event streams, FastAPI routes, and terminal REPL.

### Observability

`anveshak/run_logging.py`

This subsystem records structured run logs so model loading, retrieval rounds, warnings, failures, and completion state can be inspected after the fact.

## Retrieval And Memory Flow

The system does not rely only on the base model weights.

For a normal chat run:

1. The prompt and attachments are normalized.
2. Documents are parsed into text, and sometimes visuals.
3. The workspace index is refreshed when enabled.
4. Long-term memory is retrieved.
5. Direct local path mentions are pulled in.
6. Semantic local-file retrieval runs.
7. Web retrieval is disabled, automatic, or forced depending on the selected policy.
8. The answer prompt is composed from all retrieved evidence.
9. The model streams the answer.
10. Durable memory compression happens after the answer, in the background.

This last point matters: the chat is unlocked as soon as the answer is done, while long-term memory compaction finishes asynchronously.

## User-Facing Surfaces

### Browser UI

The browser console provides:

- runtime-preparation overlay
- streaming answer tokens
- streamed reasoning trace
- drag-and-drop attachments
- attachment thumbnails/cards
- per-message web-mode selection
- steering during generation
- saved API-call workflows

The main prompt textarea stays editable while a run is active, but the `Send` button only unlocks when the session can accept another prompt.

The steering textarea also stays editable outside the active steering window, but its send button unlocks only while the model is actively generating.

### Terminal UI

The terminal interface provides a simpler SSH-friendly REPL. It supports the same local runtime and retrieval stack, but the browser remains the richer surface for attachments, inline steering, and saved API-call management.

### API Calls

The API-call system is a reusable workflow layer on top of the same local runtime.

Each saved API call stores:

- name
- generated API key
- saved reasoning model snapshot
- saved embedding model snapshot
- system prompt
- input template
- response instructions
- response mode
- internet policy
- `Use User Context`
- invocation memory mode

The browser experience for API calls is split into two screens:

- `API Call Builder`, where a workflow is created or edited
- `Existing API Keys`, where saved keys are listed, copied, edited, or deleted

Saving a call produces a generated key popup with copy support and a link to the usage guide. Deleting a call shows a confirmation dialog that previews the saved configuration before removal.

### Internet Policy

- `No Internet`
- `Auto`
- `Searching the Web`

### User Context

When enabled, the API call can use Anveshak's long-term user memory.

### Invocation Memory

- `Independent`: each API call is a fresh run
- `Remember Calls`: that API call remembers its own prior invocations

### Save-Time Preparation

Saving or updating an API call starts background preparation for the current runtime so the workflow is faster to invoke afterward.

### Invocation Pattern

Use:

```bash
POST /v1/api-calls/<call_id>/invoke
```

with:

```bash
Authorization: Bearer <api_key>
```

and a JSON body:

```json
{
  "input": "Run the saved workflow on this payload.",
  "variables": {"example": true}
}
```

See [`API_CALLS.md`](API_CALLS.md) for the detailed guide.

## API-Call Use Cases

- Stateless structured workflows that should always behave like a fresh request
- Stateful local tools that should remember earlier invocations of the same saved API call
- Controlled evaluations that need a fixed internet policy
- Private assistants that should optionally reuse the same long-term user context as the chat console
- Local application backends that want a stable API key and saved prompt contract over a changing local runtime

## Storage Layout

Important local directories:

- `checkpoints/` for model weights and Hugging Face cache
- `context_window/sessions/` for chat sessions
- `context_window/uploads/` for user-uploaded files
- `context_window/local_files/` for workspace retrieval state
- `context_window/memory/` for durable long-term memory
- `context_window/api_call_sessions/` for remembered API-call histories
- `API_calls/` for saved API-call definitions
- `logs/` for per-run JSONL logs

## Resource Requirements

Resource needs depend heavily on the selected reasoning model.

Users should expect patience-worthy startup and shutdown times. On large local checkpoints, starting a run, loading the model, releasing resources, or exiting the application can take a couple of minutes. That delay is normal for a private local runtime and should not be confused with a crash.

### Practical Baseline

- A modern CUDA-capable GPU is strongly recommended for the flagship local models
- Enough disk space is required for large checkpoint repos and cache files
- Fast local storage helps large-model startup substantially
- Large memory and disk budgets matter more than they do in lightweight chatbot projects because some supported checkpoints are extremely large

### Important Reality

- Small and medium models are much easier to use interactively
- Very large models can take minutes to load
- Some giant checkpoints are only practical with aggressive quantization or a dedicated serving backend
- Kimi K2 is best served through a dedicated compatible server backend rather than the generic in-process Hugging Face path

### Practical Expectations By Feature

- Browser chat is the most feature-complete surface
- Terminal chat is the lightest-weight surface
- Web retrieval adds latency but improves freshness
- Long-term memory improves continuity but requires background summarization work
- Huge multimodal checkpoints can dominate the total runtime cost even when retrieval is efficient

### GPU Selection

- `--n_GPUs <int>` limits Anveshak to that many GPUs on the current node
- if `--n_GPUs` is omitted, Anveshak uses all GPUs visible to the process
- this is a single-node multi-GPU path; the system does not target multi-node orchestration here

## Operational Trade-Offs

- Keeping the model warm improves interactivity but consumes VRAM
- Live web retrieval improves freshness but adds latency
- Persistent memory improves continuity but requires background summarization work
- Large retrieved contexts improve grounding but increase prompt-ingestion cost

## Good Use Cases

- Private paper reading and review
- Local codebase analysis with cited file evidence
- Multimodal research workflows with PDFs and images
- Controlled local-vs-hosted assistant comparisons
- Reusable structured local workflows exposed through API keys

## Less Ideal Use Cases

- Lightweight commodity laptops with very little VRAM
- Use cases requiring broad hosted-tool ecosystems out of the box
- Situations where every model must support every modality identically

## How To Read The Rest Of The Project Docs

- Start here for the high-level system picture
- Read [`API_CALLS.md`](API_CALLS.md) next if you want programmatic usage
- Read the contributor guides in `Documentations/` if you want to extend the architecture

## Recommended Reading Order

1. This file: [`overview.md`](overview.md)
2. API workflows: [`API_CALLS.md`](API_CALLS.md)
3. Contributor extension guides in `Documentations/`
