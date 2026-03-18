"""Model metadata that drives backend selection and modality checks."""

from __future__ import annotations


MODEL_CATALOG = [
    {
        "label": "Qwen3.5 122B GPTQ",
        "model_id": "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
        "family": "Qwen",
        "kind": "image-text-to-text",
        "input_backend": "qwen_vision",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Default flagship local Qwen option in this project.",
    },
    {
        "label": "Qwen2.5 VL 72B",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "family": "Qwen",
        "kind": "image-text-to-text",
        "input_backend": "qwen_vision",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Large open Qwen VLM alternative.",
    },
    {
        "label": "InternVL3.5 38B",
        "model_id": "OpenGVLab/InternVL3_5-38B",
        "family": "InternVL",
        "kind": "image-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Strong open multimodal reasoning model.",
    },
    {
        "label": "Gemma 3 27B",
        "model_id": "google/gemma-3-27b-it",
        "family": "Gemma",
        "kind": "image-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Open Gemma multimodal instruction model.",
    },
    {
        "label": "Llama 3.2 90B Vision",
        "model_id": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "family": "Llama",
        "kind": "image-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Large Llama vision model.",
    },
    {
        "label": "LLaVA OneVision 72B",
        "model_id": "llava-hf/llava-onevision-qwen2-72b-ov-hf",
        "family": "LLaVA",
        "kind": "image-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Large LLaVA OneVision model.",
    },
    {
        "label": "LLaVA NeXT Video 34B",
        "model_id": "llava-hf/LLaVA-NeXT-Video-34B-hf",
        "family": "LLaVA",
        "kind": "video-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": True,
        "supports_native_documents": False,
        "notes": "Video-capable LLaVA option.",
    },
    {
        "label": "Kimi K2 Instruct",
        "model_id": "moonshotai/Kimi-K2-Instruct",
        "family": "Kimi",
        "kind": "text-generation",
        "input_backend": "text-chat",
        "preferred_runtime_backend": "kimi_server",
        "server_model_name": "kimi-k2",
        "preferred_attn_implementation": "eager",
        "supports_text": True,
        "supports_images": False,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Text-only open model; best served through a dedicated Kimi engine such as a local OpenAI-compatible vLLM, SGLang, or KTransformers deployment.",
    },
    {
        "label": "DeepSeek VL2",
        "model_id": "deepseek-ai/deepseek-vl2",
        "family": "DeepSeek",
        "kind": "image-text-to-text",
        "input_backend": "hf_multimodal",
        "supports_text": True,
        "supports_images": True,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Added as the closest verified public Hugging Face match for the requested 'DeepLab VLM Instruct'; no public VLM by that exact DeepLab name was found.",
    },
    {
        "label": "Apertus 70B",
        "model_id": "swiss-ai/Apertus-70B-Instruct-2509",
        "family": "Apertus",
        "kind": "text-generation",
        "input_backend": "text-chat",
        "supports_text": True,
        "supports_images": False,
        "supports_video": False,
        "supports_native_documents": False,
        "notes": "Text-only Apertus open model.",
    },
]


def get_model_profile(model_id: str) -> dict:
    """Return the known or inferred capability profile for a model id."""

    for item in MODEL_CATALOG:
        if item["model_id"] == model_id:
            return item
    lowered = model_id.lower()
    if "video" in lowered:
        return {
            "label": model_id,
            "model_id": model_id,
            "kind": "video-text-to-text",
            "input_backend": "hf_multimodal",
            "supports_text": True,
            "supports_images": True,
            "supports_video": True,
            "supports_native_documents": False,
        }
    if any(token in lowered for token in ["qwen2.5-vl", "qwen2_5_vl", "qwen3-vl", "qwen3_vl", "qwen3.5-vl", "qwen3_5_vl"]):
        return {
            "label": model_id,
            "model_id": model_id,
            "kind": "image-text-to-text",
            "input_backend": "qwen_vision",
            "supports_text": True,
            "supports_images": True,
            "supports_video": False,
            "supports_native_documents": False,
        }
    if any(token in lowered for token in ["vision", "vl", "llava", "internvl", "gemma-3", "deepseek"]):
        return {
            "label": model_id,
            "model_id": model_id,
            "kind": "image-text-to-text",
            "input_backend": "hf_multimodal",
            "supports_text": True,
            "supports_images": True,
            "supports_video": False,
            "supports_native_documents": False,
        }
    return {
        "label": model_id,
        "model_id": model_id,
        "kind": "text-generation",
        "input_backend": "text-chat",
        "supports_text": True,
        "supports_images": False,
        "supports_video": False,
        "supports_native_documents": False,
    }
