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
- Fall back to sampled video frames for image-capable models that do not natively support video
- Transcribe browser microphone recordings into editable chat text before send
- Warm Whisper in the background when microphone capture begins
- Search the web live during a run
- Show inline web images or videos inside the chat when live retrieval finds them
- Offer `Safe` and `Unrestricted` policies for inline web media
- Keep durable long-term user memory across sessions
- Let users steer a run while the model is generating
- Render assistant responses as Markdown by default with per-message raw-text fallback and LaTeX support
- Expose saved local workflows through generated API keys
- Detect when a gated Hugging Face model needs user authentication and prompt for a token in the browser UI
- Keep per-run logs for debugging and reproducibility

## What It Cannot Do

- It does not make every supported model equally fast or equally practical on one machine
- It cannot bypass upstream gated-model restrictions or missing checkpoint access
- It does not magically make text-only models natively understand PDFs as documents; those are parsed into text context
- It does not guarantee that sampled fallback video frames capture every important moment of a clip
- It does not remove the hardware cost of very large checkpoints
- It does not guarantee identical behavior across every model family, because processor and remote-code behavior differs across upstream repos

## Hugging Face Authentication For Gated Models

Most Anveshak users do not need a Hugging Face token, because many supported checkpoints are public.

For gated models, the runtime now checks `HUGGINGFACE_HUB_TOKEN` automatically. If that variable is present, Anveshak passes it through to the Hugging Face Hub during checkpoint preparation. `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and `HUGGING_FACE_TOKEN` are treated as equivalent compatibility aliases. Users can set the primary variable in the current shell:

```bash
export HUGGINGFACE_HUB_TOKEN=hf_...
```

Or persist it for future bash sessions by adding the same export line to `~/.bashrc` or `~/.bash_profile`.

If a user chooses a gated model and automatic authentication still is not available, the browser UI opens a modal prompt asking for a personal Hugging Face token, explains that public models do not need one, and links to the official Hugging Face token-management instructions. The runtime then retries checkpoint preparation after the token is submitted.

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

This is the OpenAI-compatible served-model backend. It is currently used for models such as Kimi K2 when they are better served through a compatible endpoint than through the generic in-process Hugging Face path.

`anveshak/transcription.py`

This module wraps OpenAI Whisper for local transcription. It warms lazily, supports browser-recorded PCM WAV clips without requiring `ffmpeg`, and is used by both the microphone workflow and Whisper-backed audio attachments.

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
2. Starting microphone capture also starts background Whisper warm-up, and finished mic clips are transcribed into editable chat text before send.
3. Documents are parsed into text, and sometimes visuals.
4. Video-capable models receive native video attachments, while image-only multimodal models receive sampled fallback frames and a warning about possible missed moments.
5. The workspace index refreshes in the background when enabled, rather than blocking prompt submission.
6. Direct local path mentions are pulled in.
7. Semantic local-file retrieval runs.
8. Long-term memory is retrieved.
9. Web retrieval is disabled, automatic, or forced depending on the selected policy.
10. When web retrieval is active and the prompt warrants it, inline web media previews are curated in `Safe` or `Unrestricted` mode.
11. The answer prompt is composed from all retrieved evidence.
12. The model streams the answer.
13. Durable memory compression happens after the answer, in the background.

This last point matters: the chat is unlocked as soon as the answer is done, while long-term memory compaction finishes asynchronously.

## User-Facing Surfaces

### Browser UI

The browser console provides:

- runtime-preparation overlay
- background local-file-index status in the sidebar
- streaming answer tokens
- streamed reasoning trace
- drag-and-drop attachments
- attachment thumbnails/cards
- per-message web-mode selection
- per-message `Safe` / `Unrestricted` web-media selection
- inline web image/video cards under answers when live retrieval returns them
- microphone recording with editable Whisper transcription before send
- per-answer Markdown / raw-text toggle with LaTeX rendering when Markdown is enabled
- steering during generation
- saved API-call workflows

The main prompt textarea stays editable while a run is active, but the `Send` button only unlocks when the session can accept another prompt. During microphone transcription, `Send` is intentionally disabled until the transcript is ready or the audio is attached as a fallback.

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
- MiroThinker 1.7 is supported as a local text model, but its official checkpoint is extremely large in practice

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
