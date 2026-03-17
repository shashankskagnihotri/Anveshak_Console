<p align="center">
  <img src="logo/anveshak_logo2.png" alt="Anveshak Console logo" width="220" />
</p>

# Anveshak Console

> Private multimodal reasoning with local memory and live retrieval.
>
> TL;DR: A private multimodal research console for local reasoning, live web retrieval, long-term memory, and API-style workflows.

<p align="center">
  <a href="https://www.youtube.com/watch?v=qFrBSUA1BPo">
    <img src="https://img.youtube.com/vi/qFrBSUA1BPo/maxresdefault.jpg" alt="Watch the Anveshak Console demo on YouTube" width="100%" />
  </a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=qFrBSUA1BPo">Watch the demo on YouTube</a>
</p>

Anveshak Console is built for research workflows that need strong local control without giving up the feel of a modern frontier assistant. It runs open multimodal models on your own GPU, keeps checkpoints and memory on your own machine, retrieves fresh web evidence during a run, and exposes the whole stack in code so the system stays inspectable, reproducible, and extensible.

## Why This Project Matters

Serious research work often involves unpublished notes, drafts, PDFs, experiments, and iterative reasoning traces that should not leave your machine. Anveshak Console is designed for that boundary: local model execution, local file access, local memory, live retrieval, and a clean interface that still feels practical for day-to-day use. Its API-call layer also makes it easier to build controlled interfaces around local open-source models, which creates a clearer path for direct comparison between open models that do not natively ship with internet capabilities and closed-source systems that do.

## What It Does

- Runs large open Hugging Face models locally, with `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` as the default assistant model.
- Uses live web retrieval during a run instead of answering from stale model weights alone.
- Maintains long-term memory in `context_window/` and resets it with `Obliviate`.
- Reads local files, direct file paths, and browser uploads without shipping them to a hosted service.
- Supports browser chat, terminal chat, drag-and-drop attachments, inline steering, and saved API-style call presets.
- Writes per-run structured logs in `logs/` and starts reproducibly with `--seed`.

## Model Stack

Default assistant model:

```text
Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
```

Default embedding model:

```text
Qwen/Qwen3-Embedding-0.6B
```

Included model options:

- `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4`
- `Qwen/Qwen2.5-VL-72B-Instruct`
- `OpenGVLab/InternVL3_5-38B`
- `google/gemma-3-27b-it`
- `meta-llama/Llama-3.2-90B-Vision-Instruct`
- `llava-hf/llava-onevision-qwen2-72b-ov-hf`
- `llava-hf/LLaVA-NeXT-Video-34B-hf`
- `moonshotai/Kimi-K2-Instruct`
- `deepseek-ai/deepseek-vl2`
- `swiss-ai/Apertus-70B-Instruct-2509`

## Install

Create and activate the environment:

```bash
conda create -n qwen_index python=3.12
conda activate qwen_index
```

Install CUDA PyTorch:

```bash
python -m pip install --ignore-installed torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -U setuptools
```

Install the project dependencies:

```bash
python -m pip install --no-build-isolation -r requirements.txt
```

Notes:

- `transformers` is installed from `main`.
- `gptqmodel` may compile extensions during install or first use.
- `PyMuPDF` is used for PDF text and visual extraction.

## Launch

Web UI (Recommended):

```bash
python -W ignore main.py --mode web --open-browser
```

Terminal mode:

```bash
python -W ignore main.py
```

Terminal and browser together:

```bash
python -W ignore main.py --mode both --open-browser
```

## Useful Arguments

```bash
python -W ignore main.py \
  --mode web \
  --model-id Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --embedding-model-id Qwen/Qwen3-Embedding-0.6B \
  --web-mode auto \
  --seed 0 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-gpu-memory-gib 90 \
  --max-cpu-memory-gib 220 \
  --torch-dtype auto \
  --attn-implementation sdpa
```

Key flags:

- `--mode terminal|web|both`
- `--web-mode auto|always|off`
- `--model-id <huggingface-model-id>`
- `--embedding-model-id <huggingface-model-id>`
- `--seed <int>`
- `--max-gpu-memory-gib <int>`
- `--max-cpu-memory-gib <int>`
- `--open-browser`

## Run Behavior

- Only one chat prompt can run at a time for a given session.
- Steering is enabled only while the assistant is actively generating an answer.
- Images and videos are passed directly to multimodal models when the selected backend supports them.
- PDFs are parsed with PyMuPDF into text plus extracted visuals.
- Text-like and office-style documents are parsed into retrieval context.
- Text-only models receive parsed document text instead of fake multimodal attachments.
- Unsupported modalities are reported directly inside the chat.

To erase long-term memory and clear the current session:

```text
Obliviate
```

## Runtime Layout

Anveshak Console stores local state in:

- `checkpoints/` for model weights and Hugging Face cache data
- `context_window/` for sessions, uploads, caches, memory, and file-index state
- `API_calls/` for saved API-style call configurations
- `logs/` for per-run structured logs

## Reproducibility

Startup is seeded by default with:

```text
--seed 0
```

This applies deterministic seeding across Python, NumPy, PyTorch, and Transformers where supported.

## Performance Notes

The default 122B GPTQ path is intentionally ambitious for a single-node H100 setup.

- Initial model load can take minutes.
- GPTQ kernel compilation can happen on first use in a fresh process.
- Large retrieved context increases prompt-ingestion time before answer tokens appear.
- Idle unloading is used to reclaim VRAM when the model has been inactive long enough.

The runtime overlay and per-run logs are meant to make that cost visible instead of opaque.

## Quick Smoke Test

For a fast functional check before downloading the full default stack:

```bash
python -W ignore main.py \
  --model-id trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration \
  --embedding-model-id sentence-transformers/all-MiniLM-L6-v2 \
  --mode web \
  --open-browser \
  --seed 0
```

## Python Package Name

The import path remains:

```text
qwen_indexing
```

## License

The code in this repository is licensed under:

```text
AGPL-3.0-or-later
```

See [LICENSE](LICENSE) for the full text.

Model weights, external model cards, and third-party components keep their own licenses and usage terms. Using this repo does not relicense those upstream assets.

All third-party trademarks, service marks, model names, logos, brands, datasets, and copyrighted materials referenced by this project remain the property of their respective owners. This repository does not claim ownership over any such third-party intellectual property.

## Citation

If you use this repo, please cite:

Text citation:

```text
[Add plain-text citation here]
```

BibTeX:

```bibtex
% Add BibTeX citation here
```
