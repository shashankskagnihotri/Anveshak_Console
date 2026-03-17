from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Iterable

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    GPTQConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from ..config import RuntimeConfig
from ..events import RunHandle
from ..model_catalog import get_model_profile
from ..schema import Attachment, ChatMessage, RetrievedChunk
from ..utils import compact_whitespace, extract_json_object


SYSTEM_PROMPT = """You are Anveshak Console, a private multimodal research assistant with local memory and live retrieval.

Always ground your answer using the supplied context in this priority order:
1. Explicit user attachments and directly referenced local files.
2. Active web evidence.
3. Long-term memory notes.
4. Recent conversation history.

Rules:
- Cite sources inline as [F#], [W#], or [M#] when you rely on them.
- If evidence is missing or conflicting, say so clearly.
- Do not invent web citations that were not provided.
- Keep reasoning internal inside <think>...</think> and place the final answer outside it.
"""


JSON_SYSTEM_PROMPT = """Return only valid JSON. No markdown fences, no prose before or after the JSON."""


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    reasoning: str


class _RunStoppingCriteria(StoppingCriteria):
    def __init__(self, handle: RunHandle) -> None:
        self.handle = handle

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, D401
        return self.handle.should_stop_generation()


class _ThinkParser:
    def __init__(self) -> None:
        self.buffer = ""
        self.in_think = False
        self.reasoning_parts: list[str] = []
        self.answer_parts: list[str] = []

    def feed(self, piece: str) -> list[tuple[str, str]]:
        self.buffer += piece
        events: list[tuple[str, str]] = []

        while self.buffer:
            if self.in_think:
                marker = self.buffer.find("</think>")
                if marker == -1:
                    chunk = self.buffer
                    self.buffer = ""
                    if chunk:
                        self.reasoning_parts.append(chunk)
                        events.append(("reasoning", chunk))
                    break
                chunk = self.buffer[:marker]
                if chunk:
                    self.reasoning_parts.append(chunk)
                    events.append(("reasoning", chunk))
                self.buffer = self.buffer[marker + len("</think>") :]
                self.in_think = False
                continue

            marker = self.buffer.find("<think>")
            if marker == -1:
                chunk = self.buffer
                self.buffer = ""
                if chunk:
                    self.answer_parts.append(chunk)
                    events.append(("answer", chunk))
                break

            prefix = self.buffer[:marker]
            if prefix:
                self.answer_parts.append(prefix)
                events.append(("answer", prefix))
            self.buffer = self.buffer[marker + len("<think>") :]
            self.in_think = True

        return events

    def finalize(self) -> GeneratedAnswer:
        if self.buffer:
            if self.in_think:
                self.reasoning_parts.append(self.buffer)
            else:
                self.answer_parts.append(self.buffer)
        return GeneratedAnswer(
            answer="".join(self.answer_parts).strip(),
            reasoning="".join(self.reasoning_parts).strip(),
        )


class QwenRunner:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.profile = get_model_profile(config.model_id)
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_used_at = 0.0

    def load(self) -> None:
        if self.model is not None and (self.processor is not None or self.tokenizer is not None):
            self.last_used_at = time.time()
            return
        model_source = self._model_source()
        is_gptq = self._is_gptq_checkpoint(model_source)

        dtype = self._resolve_dtype()
        max_memory = {}
        if torch.cuda.is_available() and self.config.max_gpu_memory_gib:
            max_memory[0] = f"{self.config.max_gpu_memory_gib}GiB"
        if self.config.max_cpu_memory_gib:
            max_memory["cpu"] = f"{self.config.max_cpu_memory_gib}GiB"

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "offload_folder": str(self.config.offload_dir),
            "offload_state_dict": True,
            "attn_implementation": self.config.attn_implementation,
            "low_cpu_mem_usage": True,
            "max_memory": max_memory or None,
            "cache_dir": str(self.config.hf_cache_dir),
            "trust_remote_code": True,
        }
        quantization_override = self._gptq_backend_override(model_source)
        if quantization_override is not None and not is_gptq:
            model_kwargs["quantization_config"] = quantization_override

        kind = self.profile.get("kind", "text-generation")
        if kind == "text-generation":
            self.tokenizer = self._load_tokenizer(model_source)
            self.processor = None
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_source,
                cache_dir=str(self.config.hf_cache_dir),
                trust_remote_code=True,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = self._load_tokenizer(model_source)
                try:
                    self.processor.tokenizer = self.tokenizer
                except Exception:
                    pass
        if is_gptq:
            self.model = self._load_gptq_model(model_source)
        elif kind == "text-generation":
            self.model = self._load_model_with_compat(AutoModelForCausalLM, model_source, model_kwargs)
        else:
            self.model = self._load_model_with_compat(AutoModelForImageTextToText, model_source, model_kwargs)

        decoder = self._decoder_tokenizer()
        if getattr(decoder, "pad_token", None) is None and getattr(decoder, "eos_token", None) is not None:
            decoder.pad_token = decoder.eos_token
        self.model.eval()
        self.last_used_at = time.time()

    def plan_search(self, *, user_query: str, recent_messages: list[ChatMessage]) -> dict:
        prompt = f"""
Decide whether live internet search is needed for the user's request.

Return JSON with:
- enabled: boolean
- rationale: short string
- search_queries: array of up to 3 focused search queries
- max_rounds: integer from 1 to 3

Recent conversation:
{self._render_recent_messages(recent_messages)}

User request:
{user_query}
""".strip()
        return self.generate_json(prompt, max_new_tokens=400)

    def plan_follow_up_searches(
        self,
        *,
        user_query: str,
        current_queries: list[str],
        evidence: list[RetrievedChunk],
    ) -> list[str]:
        if not evidence:
            return []
        prompt = f"""
You are controlling an active web retrieval loop.
Given the user query, the previous search queries, and the current evidence, decide whether one more search round is needed.

Return JSON with:
- continue_search: boolean
- follow_up_queries: array of up to 2 new queries

User query:
{user_query}

Previous queries:
{json.dumps(current_queries, ensure_ascii=False)}

Current evidence:
{self._render_chunks(evidence)}
""".strip()
        payload = self.generate_json(prompt, max_new_tokens=300)
        if not payload.get("continue_search"):
            return []
        return [item for item in payload.get("follow_up_queries", []) if isinstance(item, str) and item.strip()]

    def summarize_memory(
        self,
        *,
        user_text: str,
        assistant_text: str,
        citations: list[dict],
        recent_messages: list[ChatMessage],
    ) -> dict:
        prompt = f"""
Compress this exchange into a durable long-term memory note using a LightMem-style compact memory entry.

Return JSON with:
- summary: string no more than {self.config.memory_note_max_words} words
- keywords: array of up to 12 short keywords
- facts: array of short factual bullets
- open_loops: array of unresolved tasks or follow-ups

Recent conversation:
{self._render_recent_messages(recent_messages)}

Latest user message:
{user_text}

Latest assistant reply:
{assistant_text}

Citations:
{json.dumps(citations, ensure_ascii=False)}
""".strip()
        return self.generate_json(prompt, max_new_tokens=500)

    def generate_json(self, prompt: str, *, max_new_tokens: int) -> dict:
        raw = self.generate_text(
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=prompt,
            attachments=[],
            max_new_tokens=max_new_tokens,
        )
        try:
            return extract_json_object(raw)
        except Exception:
            repair_prompt = f"""
The previous attempt was not valid JSON. Repair it and return only one valid JSON object.

Original prompt:
{prompt}

Previous output:
{raw}
""".strip()
            repaired = self.generate_text(
                system_prompt=JSON_SYSTEM_PROMPT,
                user_prompt=repair_prompt,
                attachments=[],
                max_new_tokens=max_new_tokens,
            )
            return extract_json_object(repaired)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachments: list[Attachment],
        max_new_tokens: int,
    ) -> str:
        self.load()
        messages = self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt, attachments=attachments)
        inputs = self._prepare_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], generated_ids, strict=True)
        ]
        outputs = self._decoder_tokenizer().batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_used_at = time.time()
        return outputs[0].strip()

    def stream_answer(
        self,
        *,
        user_query: str,
        attachments: list[Attachment],
        memory_chunks: list[RetrievedChunk],
        file_chunks: list[RetrievedChunk],
        web_chunks: list[RetrievedChunk],
        recent_messages: list[ChatMessage],
        steering_notes: list[str],
        handle: RunHandle,
    ) -> GeneratedAnswer:
        self.load()
        prompt = self._compose_answer_prompt(
            user_query=user_query,
            memory_chunks=memory_chunks,
            file_chunks=file_chunks,
            web_chunks=web_chunks,
            recent_messages=recent_messages,
            steering_notes=steering_notes,
        )
        messages = self._build_messages(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, attachments=attachments)
        inputs = self._prepare_inputs(messages)

        streamer = TextIteratorStreamer(
            self._decoder_tokenizer(),
            skip_prompt=True,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([_RunStoppingCriteria(handle)]),
        }
        error_queue: Queue[BaseException] = Queue(maxsize=1)

        def _run_generation() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except BaseException as exc:  # pragma: no cover - surfaced via the main thread
                error_queue.put(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=_run_generation, daemon=True)
        worker.start()

        parser = _ThinkParser()
        handle.phase = "generating"
        handle.emit("status", phase="generation", text="Generating answer with retrieved context")
        for piece in streamer:
            for event_kind, chunk in parser.feed(piece):
                if not chunk:
                    continue
                if event_kind == "reasoning":
                    handle.emit("reasoning", text=chunk)
                else:
                    handle.emit("token", text=chunk)

        worker.join()
        try:
            raise error_queue.get_nowait()
        except Empty:
            pass
        self.last_used_at = time.time()
        return parser.finalize()

    def unload_if_idle(self) -> None:
        if self.model is None:
            return
        if self.config.model_idle_unload_seconds <= 0:
            return
        if time.time() - self.last_used_at < self.config.model_idle_unload_seconds:
            return
        self.unload()

    def unload(self) -> None:
        if self.model is None and self.processor is None and self.tokenizer is None:
            return
        self.model = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _render_recent_messages(self, messages: Iterable[ChatMessage]) -> str:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"{message.role.upper()}: {compact_whitespace(message.text)}")
        return "\n".join(rendered) if rendered else "(none)"

    def _render_chunks(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(none)"
        lines: list[str] = []
        for chunk in chunks:
            lines.append(f"{chunk.source_id} {chunk.label}: {compact_whitespace(chunk.text)[:1200]}")
        return "\n\n".join(lines)

    def _compose_answer_prompt(
        self,
        *,
        user_query: str,
        memory_chunks: list[RetrievedChunk],
        file_chunks: list[RetrievedChunk],
        web_chunks: list[RetrievedChunk],
        recent_messages: list[ChatMessage],
        steering_notes: list[str],
    ) -> str:
        steering_block = "\n".join(f"- {note}" for note in steering_notes) if steering_notes else "(none)"
        return f"""
User request:
{user_query}

Steering notes received while thinking:
{steering_block}

Recent conversation:
{self._render_recent_messages(recent_messages)}

Long-term memory:
{self._render_chunks(memory_chunks)}

Local file context:
{self._render_chunks(file_chunks)}

Active web evidence:
{self._render_chunks(web_chunks)}
""".strip()

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachments: list[Attachment],
    ) -> list[dict]:
        backend = self.profile.get("input_backend", "text-chat")
        if backend == "text-chat":
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        user_content: list[dict] = []
        for attachment in attachments:
            if attachment.media_kind == "image":
                if backend == "qwen_vision":
                    user_content.append({"type": "image", "image": attachment.path})
                else:
                    user_content.append({"type": "image", "path": attachment.path})
            elif attachment.media_kind == "video":
                if backend == "qwen_vision":
                    user_content.append({"type": "video", "video": attachment.path})
                else:
                    user_content.append({"type": "video", "path": attachment.path})
        user_content.append({"type": "text", "text": user_prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _prepare_inputs(self, messages: list[dict]) -> dict:
        backend = self.profile.get("input_backend", "text-chat")
        if backend == "text-chat":
            return self._prepare_text_inputs(messages)
        if backend == "qwen_vision":
            return self._prepare_qwen_inputs(messages)
        return self._prepare_hf_multimodal_inputs(messages)

    def _prepare_text_inputs(self, messages: list[dict]) -> dict:
        tokenizer = self._decoder_tokenizer()
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            prompt = self._fallback_text_prompt(messages)
            inputs = tokenizer(prompt, return_tensors="pt")
        return self._move_inputs_to_device(inputs)

    def _prepare_qwen_inputs(self, messages: list[dict]) -> dict:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        kwargs = {
            "text": [text],
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        inputs = self.processor(**kwargs)
        return self._move_inputs_to_device(inputs)

    def _prepare_hf_multimodal_inputs(self, messages: list[dict]) -> dict:
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            return self._move_inputs_to_device(inputs)
        except Exception:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs = self._load_image_inputs(messages)
            video_inputs = self._load_video_inputs(messages)
            kwargs = {
                "text": [text],
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs:
                kwargs["images"] = image_inputs
            if video_inputs:
                kwargs["videos"] = video_inputs
            inputs = self.processor(**kwargs)
            return self._move_inputs_to_device(inputs)

    def _move_inputs_to_device(self, inputs) -> dict:
        prepared = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.primary_device)
            else:
                prepared[key] = value
        return prepared

    def _decoder_tokenizer(self):
        tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = self._load_tokenizer(self._model_source())
            self.tokenizer = tokenizer
            if self.processor is not None and getattr(self.processor, "tokenizer", None) is None:
                try:
                    self.processor.tokenizer = tokenizer
                except Exception:
                    pass
        if tokenizer is None:
            raise RuntimeError(f"No tokenizer is available for {self.config.model_id}")
        return tokenizer

    def _load_tokenizer(self, model_source: str):
        return AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(self.config.hf_cache_dir),
            trust_remote_code=True,
        )

    def _model_source(self) -> str:
        return str(self.config.model_local_path) if self.config.model_local_path else self.config.model_id

    def _load_model_with_compat(self, loader, model_source: str, model_kwargs: dict):
        try:
            return loader.from_pretrained(model_source, **model_kwargs)
        except TypeError as exc:
            if "attn_implementation" not in str(exc):
                raise
            compatible = dict(model_kwargs)
            compatible.pop("attn_implementation", None)
            return loader.from_pretrained(model_source, **compatible)

    def _load_gptq_model(self, model_source: str):
        from gptqmodel import GPTQModel

        load_kwargs = {
            "backend": "torch",
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                load_kwargs["device"] = "cuda:0"
            else:
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device"] = "cpu"

        model = GPTQModel.from_quantized(model_source, **load_kwargs)
        if self.tokenizer is not None:
            model.tokenizer = self.tokenizer
        if self.processor is not None:
            model.processor = self.processor
        return model

    def _is_gptq_checkpoint(self, model_source: str) -> bool:
        config_path = Path(model_source) / "config.json"
        if not config_path.exists():
            return False
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        quant_config = payload.get("quantization_config")
        if not isinstance(quant_config, dict):
            return False
        return str(quant_config.get("quant_method", "")).lower() == "gptq"

    def _gptq_backend_override(self, model_source: str) -> GPTQConfig | None:
        config_path = Path(model_source) / "config.json"
        if not config_path.exists():
            return None
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        quant_config = payload.get("quantization_config")
        if not isinstance(quant_config, dict):
            return None
        if str(quant_config.get("quant_method", "")).lower() != "gptq":
            return None
        merged = dict(quant_config)
        merged["backend"] = "torch"
        return GPTQConfig.from_dict_optimum(merged)

    def _fallback_text_prompt(self, messages: list[dict]) -> str:
        blocks: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                text = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
            else:
                text = str(content)
            blocks.append(f"{message.get('role', 'user').upper()}: {text}")
        blocks.append("ASSISTANT:")
        return "\n\n".join(blocks)

    def _load_image_inputs(self, messages: list[dict]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for media_path in self._iter_media_paths(messages, media_type="image"):
            with Image.open(media_path) as image:
                images.append(image.convert("RGB"))
        return images

    def _load_video_inputs(self, messages: list[dict]) -> list[list]:
        videos: list[list] = []
        for media_path in self._iter_media_paths(messages, media_type="video"):
            videos.append(self._sample_video_frames(media_path))
        return videos

    def _iter_media_paths(self, messages: list[dict], *, media_type: str) -> Iterable[Path]:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") != media_type:
                    continue
                candidate = item.get("path") or item.get(media_type) or item.get("url")
                if candidate:
                    yield Path(candidate)

    def _sample_video_frames(self, media_path: Path, *, max_frames: int = 8) -> list:
        from decord import VideoReader, cpu
        import numpy as np

        reader = VideoReader(str(media_path), ctx=cpu(0))
        frame_count = len(reader)
        if frame_count <= 0:
            return []
        indices = np.linspace(0, frame_count - 1, num=min(max_frames, frame_count), dtype=int).tolist()
        batch = reader.get_batch(indices).asnumpy()
        return [frame for frame in batch]

    def _resolve_dtype(self):
        mapping = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.config.torch_dtype]
