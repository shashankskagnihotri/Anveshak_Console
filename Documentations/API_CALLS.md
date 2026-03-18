# API Calls In Anveshak Console

This document explains how the API Call Builder works, how to use the generated API keys, what each option means, and how to update or delete old keys safely.

## UI Structure

The API area is split into two user-facing screens:

- `API Call Builder`
  Create a new saved workflow or edit an existing one.
- `Existing API Keys`
  Browse all saved keys, copy them, reopen them in the builder, or delete them.

Saving a call opens a popup that shows:

- the generated API key
- a copy button
- a documentation link
- example invocation guidance

Deleting a key opens a confirmation dialog that previews the saved configuration before removal so the user can see exactly what is being deleted.

## What The Builder Saves

Each saved API call stores:

- A human-readable name
- A generated API key
- The saved reasoning-model snapshot
- The saved embedding-model snapshot
- A system prompt
- An input template
- Response instructions
- A response mode: `text` or `json`
- An internet policy: `No Internet`, `Auto`, or `Searching the Web`
- A `Use User Context` flag
- An invocation memory mode: `Independent` or `Remember Calls`

Saved API-call definitions are written to `./API_calls`.

## Builder Workflow

1. Open `API Calls`.
2. Stay on `API Call Builder` to create a new workflow, or open `Existing API Keys` and click the pencil button to edit one.
3. Fill in the workflow name, prompting fields, response mode, and runtime options.
4. Click `Save API Call`.
5. Copy the generated key from the popup and use the docs link if you need the request format.

Saving or updating a call also starts background preparation for the current runtime so the workflow is faster to invoke afterward.

## What Each Builder Field Means

### Name

A human-readable label for the saved workflow.

### System Prompt

The persistent instruction block applied before user input.

### Input Template

The reusable request template. The builder expands placeholders when the API call is invoked.

### Response Instructions

Extra output-format or style requirements. This is where you can require summaries, bullet lists, or schema-oriented behavior.

### Response Mode

- `text`: return normal model text
- `json`: force a single JSON object and validate the returned structure

### Saved Model Snapshot

The builder records the reasoning model that was active when the call was saved.

### Saved Embedding Snapshot

The builder records the embedding model that was active when the call was saved.

These snapshots make it clear what runtime configuration the saved API call was created against.

## Recommended Invocation Pattern

The recommended endpoint format is:

```bash
POST /v1/api-calls/<call_id>/invoke
```

Pass the generated API key through a bearer header:

```bash
Authorization: Bearer <api_key>
```

You can also pass the key with:

```bash
x-api-key: <api_key>
```

## Request Body

Send JSON like this:

```json
{
  "input": "Summarize the evidence and return the key findings.",
  "variables": {
    "paper_id": 42,
    "audience": "reviewers"
  }
}
```

The builder expands:

- `{{input}}` with the `input` string
- `{{json}}` with the `variables` object serialized to JSON

## Example `curl`

```bash
curl -X POST http://127.0.0.1:8000/v1/api-calls/<call_id>/invoke \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Run the saved workflow on this payload.",
    "variables": {"example": true}
  }'
```

## What The Runtime Options Mean

### Internet Policy

- `No Internet`: the API call is not allowed to use live web retrieval
- `Auto`: the API call decides based on the request
- `Searching the Web`: the API call always performs web retrieval and verification first

### Use User Context

When enabled, the API call can use Anveshak's long-term learned user context from previous conversations.

When disabled, the API call ignores that cross-conversation context.

### Invocation Memory

- `Independent`: each API invocation is treated as a fresh run
- `Remember Calls`: this API call keeps its own prior invocation history and can use it in later invocations

This remembered invocation history is separate from the main user-context memory.

## How Invocation Actually Works

When an API call is invoked, Anveshak:

1. Resolves the saved API-call definition
2. Verifies the supplied API key
3. Applies the saved prompt contract and options
4. Reuses or prepares the current local runtime
5. Applies the saved internet policy
6. Optionally pulls long-term user context if `Use User Context` is enabled
7. Optionally loads prior API-only history if `Remember Calls` is enabled
8. Runs the request through the same local reasoning and retrieval stack used by the console
9. Returns both the output and useful runtime metadata

## Response Modes

### `text`

The API returns plain model output in the `output` field.

### `json`

The API instructs the model to return exactly one JSON object. Anveshak validates and repairs one malformed attempt if necessary before returning the parsed JSON object.

## Response Shape

The invoke endpoint returns metadata plus the generated output. A typical response includes:

- `call_id`
- `name`
- `configured_model_id`
- `runtime_model_id`
- `configured_embedding_model_id`
- `runtime_embedding_model_id`
- `response_mode`
- `web_mode`
- `use_user_context`
- `instance_mode`
- `citations`
- `output`

These metadata fields are useful because a saved API call can preserve one model snapshot while being invoked later against a runtime that may have changed.

## Existing API Keys Page

The `Existing API Keys` screen is the management surface for saved workflows.

It supports:

- copying an existing key
- reopening the configuration in the builder
- deleting the saved definition
- inspecting the saved options before deletion

This page is meant to act like a proper control panel for local API workflows rather than a hidden config file browser.

## How To Delete Old API Keys

1. Open `API Calls` from the top-right menu.
2. Open `Existing API Keys`.
3. Find the key you want to remove.
4. Click the trash button.
5. Inspect the deletion popup carefully.
6. Confirm deletion.

Deleting a saved API key removes:

- the saved API-call definition
- any remembered invocation history for that API call

## Notes On Model Snapshots

The builder shows:

- the current runtime models on the right side
- the saved model and embedding snapshots inside the API-call definition

When you save or update a call, the saved snapshots are refreshed to the current runtime values.

## Good Use Cases

- A fixed JSON-producing workflow for a local application
- A reusable paper-review or summarization endpoint
- A private assistant endpoint that should use saved user context
- A stateful API-backed local agent that should remember earlier invocations
- Controlled experiments where internet use must be forced, allowed automatically, or disabled completely
