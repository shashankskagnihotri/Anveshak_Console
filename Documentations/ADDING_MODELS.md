# Adding Models

This guide explains how to add a new reasoning model to Anveshak.

## Step 1: Add Metadata to the Catalog

Edit `anveshak/model_catalog.py`.

Each entry in `MODEL_CATALOG` should include:

- `label`
- `model_id`
- `family`
- `kind`
- `input_backend`
- `supports_text`
- `supports_images`
- `supports_video`
- `supports_native_documents`
- `notes`

Optional fields when needed:

- `preferred_runtime_backend`
- `server_model_name`
- `requires_server_backend`

Example fields:

- `kind`
  Use `text-generation`, `image-text-to-text`, or `video-text-to-text`.
- `input_backend`
  Use one of:
  - `text-chat`
  - `qwen_vision`
  - `hf_multimodal`

## Step 2: Make Sure the Backend Matches the Model Family

The backend in `model_catalog.py` decides how the model is prompted inside `anveshak/modeling/qwen_runner.py`.

Use:

- `text-chat`
  For text-only causal language models.
- `qwen_vision`
  For Qwen multimodal models that expect the Qwen vision preprocessing path.
- `hf_multimodal`
  For multimodal Hugging Face processors that support `apply_chat_template(...)` or processor-driven media inputs.

Optional runtime-backend metadata:

- `preferred_runtime_backend`
  Use this when the model should prefer a dedicated served backend over the default local loader.
- `server_model_name`
  Default model name to send to the OpenAI-compatible server when the served backend is used.
- `requires_server_backend`
  Use this for models that are intentionally exposed through Anveshak but are not downloadable local Hugging Face checkpoints.

If your new model needs a new backend style, add it in:

- `anveshak/modeling/qwen_runner.py`

You will usually need to update:

- `_build_messages(...)`
- `_prepare_inputs(...)`
- a new backend-specific helper such as `_prepare_<backend>_inputs(...)`

If the model is server-backed rather than locally downloadable, also verify the runtime gating in:

- `anveshak/runtime.py`

## Step 3: Verify Modality Support

The UI warning system relies on the modality flags in `model_catalog.py`.

If the flags are wrong:

- unsupported files may be sent to the model incorrectly
- supported files may be hidden behind unnecessary warnings

Be explicit and conservative.

## Step 4: Test the Model Path

Add at least one test or catalog assertion in `tests/`:

- backend inference
- modality flags
- any family-specific parsing behavior

Useful starting points:

- `tests/test_smoke.py`
- `tests/test_cli.py`

## Step 5: Update Public Docs

Update:

- `README.md`
- `debugging/anveshak_console_jmlr_mloss_report.tex` if the new model materially changes the paper's model discussion

## Checklist

- model catalog entry added
- backend choice verified
- modality flags verified
- prompt/input path tested
- docs updated
