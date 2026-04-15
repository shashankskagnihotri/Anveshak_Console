"""Model loading and generation adapters for the supported reasoning models."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Iterable

import torch
import transformers
from PIL import Image
from transformers import (
    AutoConfig,
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
1. The latest user request for this run.
2. This turn's explicit attachments and directly referenced local files.
3. Retrieved local file context relevant to the latest request.
4. Active web evidence.
5. Long-term memory notes, only when directly relevant.
6. Prior conversation history, only when directly relevant.

Rules:
- Solve the newest user request first.
- Treat long-term memory and old conversation as supporting background, not as the main task.
- Never let stale memory, previous goals, or prior assumptions override the current request, current files, or newer evidence.
- If the latest request conflicts with older memory or prior conversation, follow the latest request and mention the conflict if it materially matters.
- If files are attached for this turn, inspect and use them before leaning on background memory.
- Cite sources inline as [F#], [W#], or [M#] when you rely on them.
- The console renders final answers as Markdown by default. You may use concise Markdown structure and LaTeX math with $...$ or $$...$$ when it helps.
- When live web retrieval is active, the console can append curated inline web images or videos beneath your answer. Do not claim the interface cannot show web media; instead mention that relevant previews can appear below when available.
- If evidence is missing or conflicting, say so clearly.
- Do not invent web citations that were not provided.
- Keep reasoning internal inside <think>...</think> and place the final answer outside it.
"""


JSON_SYSTEM_PROMPT = """Return only valid JSON. No markdown fences, no prose before or after the JSON."""

AUDIO_TRANSCRIPTION_SYSTEM_PROMPT = """You are a precise speech transcription assistant."""

AUDIO_TRANSCRIPTION_PROMPT = """Transcribe the attached audio exactly as spoken in its original language.

Rules:
- Return only the transcription text.
- Do not add markdown, timestamps, speaker labels, or commentary.
- Use digits when numbers are clearly spoken as numbers.
"""


def _apply_transformers_compat_shims() -> None:
    """Restore helper names that older remote-code checkpoints still import."""

    utils_module = getattr(transformers, "utils", None)
    import_utils_module = getattr(utils_module, "import_utils", None)
    if utils_module is None:
        return

    if not hasattr(utils_module, "is_flash_attn_greater_or_equal_2_10"):
        comparator = getattr(utils_module, "is_flash_attn_greater_or_equal", None)
        if comparator is None and import_utils_module is not None:
            comparator = getattr(import_utils_module, "is_flash_attn_greater_or_equal", None)
        if comparator is not None:
            def _is_flash_attn_greater_or_equal_2_10() -> bool:
                try:
                    return bool(comparator("2.10"))
                except TypeError:
                    return bool(comparator(version="2.10"))
                except Exception:
                    return False

            utils_module.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10
            if import_utils_module is not None and not hasattr(import_utils_module, "is_flash_attn_greater_or_equal_2_10"):
                import_utils_module.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10

    if import_utils_module is not None and not hasattr(import_utils_module, "is_torch_fx_available"):
        def _is_torch_fx_available() -> bool:
            try:
                import torch.fx  # noqa: F401
            except Exception:
                return False
            return True

        import_utils_module.is_torch_fx_available = _is_torch_fx_available


@dataclass(slots=True)
class GeneratedAnswer:
    """Split model output into the visible answer and internal reasoning trace."""

    answer: str
    reasoning: str


class _RunStoppingCriteria(StoppingCriteria):
    """Stop generation when the current run has been cancelled or restarted."""

    def __init__(self, handle: RunHandle) -> None:
        self.handle = handle

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001, D401
        return self.handle.should_stop_generation()


class _ThinkParser:
    """Separate streamed `<think>` content from the final answer text."""

    def __init__(self) -> None:
        self.buffer = ""
        self.in_think = False
        self.reasoning_parts: list[str] = []
        self.answer_parts: list[str] = []

    def feed(self, piece: str) -> list[tuple[str, str]]:
        """Consume streamed text and emit reasoning or answer fragments."""

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
        """Flush any buffered text into the final structured answer object."""

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
    """Load the configured reasoning model and execute its planning/generation calls."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.profile = get_model_profile(config.model_id)
        self.model_config = None
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_used_at = 0.0
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def load(self) -> None:
        """Load the configured reasoning model, tokenizer, and processor on demand."""

        with self._load_lock:
            if self.model is not None and (self.processor is not None or self.tokenizer is not None):
                self.last_used_at = time.time()
                return
            _apply_transformers_compat_shims()
            model_source = self._model_source()
            is_gptq = self._is_gptq_checkpoint(model_source)
            model_config = self._load_model_config(model_source)
            self.model_config = model_config

            dtype = self._resolve_dtype()
            max_memory = self._resolved_max_memory(model_config)
            ram_heavy_load = self._can_use_ram_heavy_load_path(model_source)

            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": "auto",
                "offload_folder": str(self.config.offload_dir),
                "offload_state_dict": not ram_heavy_load,
                "attn_implementation": self._resolved_attn_implementation(),
                "low_cpu_mem_usage": not ram_heavy_load,
                "max_memory": max_memory,
                "cache_dir": str(self.config.hf_cache_dir),
                "config": model_config,
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
                # Multimodal models often need a processor for both tokenization and media preprocessing.
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
            with self._parallel_loading_environment(model_source):
                if is_gptq:
                    self.model = self._load_gptq_model(model_source)
                elif kind == "text-generation":
                    self.model = self._load_model_with_compat(AutoModelForCausalLM, model_source, model_kwargs)
                else:
                    loader = self._resolve_multimodal_loader()
                    self.model = self._load_model_with_compat(loader, model_source, model_kwargs)

            decoder = self._decoder_tokenizer()
            if getattr(decoder, "pad_token", None) is None and getattr(decoder, "eos_token", None) is not None:
                decoder.pad_token = decoder.eos_token
            self.model.eval()
            self.last_used_at = time.time()

    def plan_search(self, *, user_query: str, recent_messages: list[ChatMessage]) -> dict:
        """Ask the reasoning model whether live internet search is worthwhile."""

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
        """Ask the model whether another retrieval round should be executed."""

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
        """Compress the latest exchange into a compact persistent memory note."""

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

    def generate_json(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        attachments: list[Attachment] | None = None,
    ) -> dict:
        """Generate JSON and attempt one repair pass if the first response is invalid."""

        payload_attachments = list(attachments or [])
        raw = self.generate_text(
            system_prompt=JSON_SYSTEM_PROMPT,
            user_prompt=prompt,
            attachments=payload_attachments,
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
                attachments=payload_attachments,
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
        """Run one non-streaming text generation call."""

        with self._inference_lock:
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

    def transcribe_audio(self, attachment: Attachment, *, max_new_tokens: int = 512) -> str:
        """Convert one supported audio attachment into plain text before the main task."""

        if not self.profile.get("supports_audio"):
            raise RuntimeError(f'{self.config.model_id} does not support native audio inputs in Anveshak.')
        if attachment.media_kind != "audio":
            raise ValueError(f"Expected an audio attachment, received {attachment.media_kind!r}.")
        return compact_whitespace(
            self.generate_text(
                system_prompt=AUDIO_TRANSCRIPTION_SYSTEM_PROMPT,
                user_prompt=AUDIO_TRANSCRIPTION_PROMPT,
                attachments=[attachment],
                max_new_tokens=max_new_tokens,
            )
        )

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
        """Stream the main assistant answer while separating reasoning from output."""

        with self._inference_lock:
            self.load()
            prompt = self._compose_answer_prompt(
                user_query=user_query,
                attachments=attachments,
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
        """Unload the model after a long idle period when session pinning allows it."""

        if self.model is None:
            return
        if self.config.model_idle_unload_seconds <= 0:
            return
        if time.time() - self.last_used_at < self.config.model_idle_unload_seconds:
            return
        self.unload()

    def unload(self) -> None:
        """Drop the loaded model state and free cached GPU memory."""

        if self.model is None and self.processor is None and self.tokenizer is None:
            return
        self.model = None
        self.model_config = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _render_recent_messages(
        self,
        messages: Iterable[ChatMessage],
        *,
        max_messages: int | None = None,
        max_chars_per_message: int = 1200,
    ) -> str:
        """Render the recent transcript into a prompt-friendly text block."""

        rendered: list[str] = []
        items = list(messages)
        if max_messages is not None and max_messages > 0:
            items = items[-max_messages:]
        for message in items:
            rendered.append(
                f"{message.role.upper()}: {compact_whitespace(message.text)[:max_chars_per_message]}"
            )
        return "\n".join(rendered) if rendered else "(none)"

    def _render_chunks(
        self,
        chunks: list[RetrievedChunk],
        *,
        max_chunks: int | None = None,
        max_chars_per_chunk: int = 1200,
    ) -> str:
        """Render retrieved evidence chunks into one prompt section."""

        if not chunks:
            return "(none)"
        lines: list[str] = []
        items = chunks[:max_chunks] if max_chunks is not None else chunks
        for chunk in items:
            lines.append(f"{chunk.source_id} {chunk.label}: {compact_whitespace(chunk.text)[:max_chars_per_chunk]}")
        return "\n\n".join(lines)

    def _render_attachment_summary(self, attachments: list[Attachment], *, max_items: int = 8) -> str:
        """Render the current-turn attachments into a short focus summary."""

        if not attachments:
            return "(none)"
        lines = [f"- {attachment.name} ({attachment.media_kind})" for attachment in attachments[:max_items]]
        if len(attachments) > max_items:
            lines.append(f"- ... and {len(attachments) - max_items} more attachment(s)")
        return "\n".join(lines)

    def _compose_answer_prompt(
        self,
        *,
        user_query: str,
        attachments: list[Attachment],
        memory_chunks: list[RetrievedChunk],
        file_chunks: list[RetrievedChunk],
        web_chunks: list[RetrievedChunk],
        recent_messages: list[ChatMessage],
        steering_notes: list[str],
    ) -> str:
        """Assemble the grounded answer prompt from every retrieval source."""

        steering_block = "\n".join(f"- {note}" for note in steering_notes) if steering_notes else "(none)"
        attachment_summary = self._render_attachment_summary(attachments)
        file_context = self._render_chunks(file_chunks, max_chunks=self.config.file_top_k, max_chars_per_chunk=1200)
        web_context = self._render_chunks(web_chunks, max_chunks=self.config.web_top_k, max_chars_per_chunk=900)
        recent_context = self._render_recent_messages(recent_messages, max_messages=4, max_chars_per_message=500)
        memory_context = self._render_chunks(memory_chunks, max_chunks=3, max_chars_per_chunk=420)
        return f"""
Primary task for this answer (highest priority):
{user_query}

Task focus rules:
- Answer the latest user request above.
- Start from this turn's attachments and local file context before using background memory.
- Use long-term memory only when it directly helps with the current task.
- Ignore stale or unrelated background context.

Attachments for this turn:
{attachment_summary}

Steering notes received while thinking:
{steering_block}

Local file context for this task:
{file_context}

Active web evidence for this task:
{web_context}

Prior conversation background (only if directly relevant):
{recent_context}

Long-term memory background (supporting only, do not override the current task):
{memory_context}

Final focus reminder:
Answer the latest user request for this run. Prefer the current task, current attachments, current local files, and newer evidence over older memory.
""".strip()

    def _build_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attachments: list[Attachment],
    ) -> list[dict]:
        """Build backend-specific chat messages with text and media attachments."""

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
            elif attachment.media_kind == "audio":
                user_content.append({"type": "audio", "audio": attachment.path})
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
        """Dispatch chat messages to the correct tokenizer/processor path."""

        backend = self.profile.get("input_backend", "text-chat")
        if backend == "text-chat":
            return self._prepare_text_inputs(messages)
        if backend == "qwen_vision":
            return self._prepare_qwen_inputs(messages)
        return self._prepare_hf_multimodal_inputs(messages)

    def _prepare_text_inputs(self, messages: list[dict]) -> dict:
        """Prepare a plain text-chat request for text-only models."""

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
        """Prepare multimodal inputs using Qwen's vision processor path."""

        from qwen_vl_utils import process_vision_info

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
        """Prepare multimodal inputs for non-Qwen Hugging Face processors."""

        has_audio_inputs = self._message_contains_media(messages, media_type="audio")
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
            if has_audio_inputs:
                raise RuntimeError(
                    "The current processor could not prepare local audio inputs. "
                    "Gemma audio support requires a Transformers build with tokenized multimodal chat-template audio support."
                )
            # Some processors only expose a string chat template and need media passed separately.
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
        """Move tensor inputs onto the primary execution device."""

        prepared = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.primary_device)
            else:
                prepared[key] = value
        return prepared

    def _decoder_tokenizer(self):
        """Return the tokenizer used to decode and chat-template model output."""

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
        """Load a tokenizer directly from the configured model source."""

        return AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=str(self.config.hf_cache_dir),
            trust_remote_code=True,
        )

    def _model_source(self) -> str:
        """Prefer the downloaded local checkpoint path when one exists."""

        return str(self.config.model_local_path) if self.config.model_local_path else self.config.model_id

    def _load_model_config(self, model_source: str):
        """Load and normalize model config metadata before model instantiation."""

        config = AutoConfig.from_pretrained(
            model_source,
            cache_dir=str(self.config.hf_cache_dir),
            trust_remote_code=True,
        )
        return self._apply_model_config_compat_shims(config)

    def _apply_model_config_compat_shims(self, config):
        """Backfill aliases that newer Transformers integrations expect from older remote configs."""

        n_routed_experts = getattr(config, "n_routed_experts", None)
        if n_routed_experts is not None:
            if not hasattr(config, "num_local_experts"):
                setattr(config, "num_local_experts", n_routed_experts)
            if not hasattr(config, "num_experts"):
                setattr(config, "num_experts", n_routed_experts)
            attribute_map = getattr(config, "attribute_map", None)
            if isinstance(attribute_map, dict):
                attribute_map.setdefault("num_local_experts", "n_routed_experts")
                attribute_map.setdefault("num_experts", "n_routed_experts")
            elif attribute_map is None:
                try:
                    config.attribute_map = {
                        "num_local_experts": "n_routed_experts",
                        "num_experts": "n_routed_experts",
                    }
                except Exception:
                    pass

        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and not hasattr(config, "rope_parameters"):
            rope_parameters = dict(rope_scaling)
            rope_type = rope_parameters.pop("type", None)
            if rope_type is not None and "rope_type" not in rope_parameters:
                rope_parameters["rope_type"] = rope_type
            if getattr(config, "rope_theta", None) is not None:
                rope_parameters.setdefault("rope_theta", config.rope_theta)
            setattr(config, "rope_parameters", rope_parameters)

        if not hasattr(config, "head_dim") and getattr(config, "qk_rope_head_dim", None) is not None:
            setattr(config, "head_dim", config.qk_rope_head_dim)
        if (
            not hasattr(config, "qk_head_dim")
            and getattr(config, "qk_rope_head_dim", None) is not None
            and getattr(config, "qk_nope_head_dim", None) is not None
        ):
            setattr(config, "qk_head_dim", config.qk_rope_head_dim + config.qk_nope_head_dim)

        quantization_config = getattr(config, "quantization_config", None)
        if self._should_preserve_fp8_modulelist_experts(config, quantization_config):
            expert_skip_pattern = r".*\.experts$"
            if isinstance(quantization_config, dict):
                merged_quantization_config = dict(quantization_config)
                modules_to_not_convert = list(merged_quantization_config.get("modules_to_not_convert") or [])
                if expert_skip_pattern not in modules_to_not_convert:
                    modules_to_not_convert.append(expert_skip_pattern)
                merged_quantization_config["modules_to_not_convert"] = modules_to_not_convert
                setattr(config, "quantization_config", merged_quantization_config)
            else:
                modules_to_not_convert = list(getattr(quantization_config, "modules_to_not_convert", None) or [])
                if expert_skip_pattern not in modules_to_not_convert:
                    modules_to_not_convert.append(expert_skip_pattern)
                setattr(quantization_config, "modules_to_not_convert", modules_to_not_convert)
        return config

    def _should_preserve_fp8_modulelist_experts(self, config, quantization_config) -> bool:
        """Keep DeepSeek-v3-style expert ModuleLists intact when loading pre-quantized FP8 checkpoints."""

        if quantization_config is None or getattr(config, "n_routed_experts", None) is None:
            return False
        if isinstance(quantization_config, dict):
            quant_method = str(quantization_config.get("quant_method", ""))
        else:
            quant_method = str(getattr(quantization_config, "quant_method", ""))
        if quant_method.lower() != "fp8":
            return False

        model_type = str(getattr(config, "model_type", "")).lower()
        class_name = config.__class__.__name__.lower()
        if model_type in {"kimi_k2", "deepseek_v3"} or "deepseekv3" in class_name:
            return True

        auto_map = getattr(config, "auto_map", None)
        if isinstance(auto_map, dict):
            return any("modeling_deepseek" in str(reference).lower() for reference in auto_map.values())
        return False

    def _resolved_attn_implementation(self) -> str:
        """Honor per-model attention preferences when the runtime uses the default backend."""

        preferred = self.profile.get("preferred_attn_implementation")
        if preferred and self.config.attn_implementation == "sdpa":
            return str(preferred)
        return self.config.attn_implementation

    def _resolved_max_memory(self, model_config) -> dict | None:
        """Choose a practical max-memory policy, especially for large pre-quantized FP8 checkpoints."""

        max_memory = {}
        detected_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if detected_gpu_count:
            configured_gpu_count = self._configured_gpu_count(detected_gpu_count)
            if self.config.n_gpus is not None or self.config.max_gpu_memory_gib:
                for device_index in range(detected_gpu_count):
                    if device_index >= configured_gpu_count:
                        max_memory[device_index] = "0GiB"
                        continue
                    gpu_memory_gib = self.config.max_gpu_memory_gib or self._device_total_memory_gib(device_index)
                    if gpu_memory_gib is not None:
                        max_memory[device_index] = f"{gpu_memory_gib}GiB"

        cpu_memory_gib = self.config.max_cpu_memory_gib
        if cpu_memory_gib is None and self._is_prequantized_fp8_checkpoint(model_config):
            detected = self._system_memory_gib()
            if detected is not None:
                cpu_memory_gib = max(int(detected - 64), 64)

        if cpu_memory_gib:
            max_memory["cpu"] = f"{cpu_memory_gib}GiB"
        return max_memory or None

    def _configured_gpu_count(self, detected_gpu_count: int) -> int:
        """Clamp the requested GPU count to the number of GPUs visible to this process."""

        if detected_gpu_count <= 0:
            return 0
        if self.config.n_gpus is None:
            return detected_gpu_count
        return max(1, min(self.config.n_gpus, detected_gpu_count))

    def _device_total_memory_gib(self, device_index: int) -> int | None:
        """Return the rounded total memory for one visible CUDA device."""

        try:
            total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
        except Exception:
            return None
        if total_bytes <= 0:
            return None
        gib = total_bytes / float(1024**3)
        return max(1, int(gib))

    def _is_prequantized_fp8_checkpoint(self, model_config) -> bool:
        """Detect checkpoints that advertise native fine-grained FP8 metadata."""

        quantization_config = getattr(model_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = str(quantization_config.get("quant_method", ""))
        else:
            quant_method = str(getattr(quantization_config, "quant_method", ""))
        return quant_method.lower() == "fp8"

    def _system_memory_gib(self) -> int | None:
        """Return the host RAM capacity in GiB when the platform exposes it."""

        if not hasattr(os, "sysconf"):
            return None
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):
            return None
        if page_size <= 0 or page_count <= 0:
            return None
        return int((page_size * page_count) / (1024**3))

    def _available_memory_gib(self) -> int | None:
        """Return currently available host RAM in GiB when the platform exposes it."""

        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            try:
                for line in meminfo_path.read_text(encoding="utf-8").splitlines():
                    if not line.startswith("MemAvailable:"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        break
                    return int(int(parts[1]) / (1024**2))
            except Exception:
                pass
        return self._system_memory_gib()

    def _checkpoint_weight_files(self, model_source: str) -> list[Path]:
        """Return the local checkpoint tensor files when the model source is a directory."""

        source_path = Path(model_source)
        if not source_path.exists() or not source_path.is_dir():
            return []

        index_path = source_path / "model.safetensors.index.json"
        if index_path.exists():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                weight_map = payload.get("weight_map")
                if isinstance(weight_map, dict):
                    resolved = []
                    seen = set()
                    for filename in weight_map.values():
                        if not isinstance(filename, str) or filename in seen:
                            continue
                        seen.add(filename)
                        candidate = source_path / filename
                        if candidate.exists():
                            resolved.append(candidate)
                    if resolved:
                        return sorted(resolved, key=lambda path: path.name)

        safetensors_files = sorted(source_path.glob("*.safetensors"))
        if safetensors_files:
            return safetensors_files
        return sorted(source_path.glob("*.bin"))

    def _checkpoint_size_gib(self, model_source: str) -> float | None:
        """Estimate the on-disk checkpoint size for a local model directory."""

        weight_files = self._checkpoint_weight_files(model_source)
        if not weight_files:
            return None
        total_bytes = 0
        for weight_file in weight_files:
            try:
                total_bytes += weight_file.stat().st_size
            except OSError:
                return None
        return total_bytes / float(1024**3)

    def _can_use_ram_heavy_load_path(self, model_source: str) -> bool:
        """Decide whether the host has enough free RAM to avoid streaming/offloaded loading."""

        checkpoint_size_gib = self._checkpoint_size_gib(model_source)
        available_memory_gib = self._available_memory_gib()
        if checkpoint_size_gib is None or available_memory_gib is None:
            return False

        reserve_gib = max(32, min(128, int(checkpoint_size_gib * 0.1)))
        return available_memory_gib >= int(checkpoint_size_gib + reserve_gib)

    def _parallel_loading_worker_count(self, model_source: str) -> int | None:
        """Choose a reasonable worker count for sharded local checkpoints."""

        shard_count = len(self._checkpoint_weight_files(model_source))
        if shard_count <= 1:
            return None
        cpu_count = os.cpu_count() or 1
        return max(2, min(shard_count, cpu_count, 16))

    @contextmanager
    def _parallel_loading_environment(self, model_source: str):
        """Enable Transformers' parallel shard loading for large local checkpoints."""

        worker_count = self._parallel_loading_worker_count(model_source)
        if worker_count is None:
            yield
            return

        previous_enable = os.environ.get("HF_ENABLE_PARALLEL_LOADING")
        previous_workers = os.environ.get("HF_PARALLEL_LOADING_WORKERS")
        os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
        os.environ["HF_PARALLEL_LOADING_WORKERS"] = str(worker_count)
        try:
            yield
        finally:
            if previous_enable is None:
                os.environ.pop("HF_ENABLE_PARALLEL_LOADING", None)
            else:
                os.environ["HF_ENABLE_PARALLEL_LOADING"] = previous_enable
            if previous_workers is None:
                os.environ.pop("HF_PARALLEL_LOADING_WORKERS", None)
            else:
                os.environ["HF_PARALLEL_LOADING_WORKERS"] = previous_workers

    def _load_model_with_compat(self, loader, model_source: str, model_kwargs: dict):
        """Retry model loading without unsupported kwargs when older loaders require it."""

        try:
            return loader.from_pretrained(model_source, **model_kwargs)
        except ValueError as exc:
            if not self._should_retry_with_eager_attention(exc, model_kwargs):
                raise
            compatible = dict(model_kwargs)
            compatible["attn_implementation"] = "eager"
            return loader.from_pretrained(model_source, **compatible)
        except TypeError as exc:
            if "attn_implementation" not in str(exc):
                raise
            compatible = dict(model_kwargs)
            compatible.pop("attn_implementation", None)
            return loader.from_pretrained(model_source, **compatible)

    def _should_retry_with_eager_attention(self, exc: ValueError, model_kwargs: dict) -> bool:
        """Detect model-loader errors that should fall back from SDPA to eager attention."""

        requested = str(model_kwargs.get("attn_implementation") or "").lower()
        if requested != "sdpa":
            return False
        message = str(exc).lower()
        if "scaled_dot_product_attention" not in message:
            return False
        eager_hints = (
            'attn_implementation="eager"',
            "attn_implementation='eager'",
            "attn_implementation=`eager`",
            "attn_implementation=\"eager\"",
            "attn_implementation='eager'",
            "load your model with the argument `attn_implementation=\"eager\"`",
            "does not support an attention implementation",
        )
        return any(hint in message for hint in eager_hints)

    def _load_gptq_model(self, model_source: str):
        """Load GPTQ checkpoints through GPTQModel instead of generic HF loaders."""

        from gptqmodel import GPTQModel

        load_kwargs = {
            "backend": "torch",
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            configured_gpu_count = self._configured_gpu_count(torch.cuda.device_count())
            if configured_gpu_count <= 1:
                load_kwargs["device"] = "cuda:0"
            else:
                load_kwargs["device_map"] = "auto"
                max_memory = self._resolved_max_memory(self.model_config)
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["device"] = "cpu"

        model = GPTQModel.from_quantized(model_source, **load_kwargs)
        if self.tokenizer is not None:
            model.tokenizer = self.tokenizer
        if self.processor is not None:
            model.processor = self.processor
        return model

    def _is_gptq_checkpoint(self, model_source: str) -> bool:
        """Inspect a checkpoint config to determine whether it is GPTQ-quantized."""

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
        """Build a GPTQConfig override when the checkpoint advertises GPTQ metadata."""

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
        """Fallback plain-text prompt builder when chat templating is unavailable."""

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
        """Load referenced image attachments into PIL images for processors."""

        images: list[Image.Image] = []
        for media_path in self._iter_media_paths(messages, media_type="image"):
            with Image.open(media_path) as image:
                images.append(image.convert("RGB"))
        return images

    def _load_video_inputs(self, messages: list[dict]) -> list[list]:
        """Sample frames from referenced video attachments for multimodal backends."""

        videos: list[list] = []
        for media_path in self._iter_media_paths(messages, media_type="video"):
            videos.append(self._sample_video_frames(media_path))
        return videos

    def _iter_media_paths(self, messages: list[dict], *, media_type: str) -> Iterable[Path]:
        """Yield media attachment paths from a multimodal message payload."""

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
        """Sample a small set of evenly spaced frames from a video attachment."""

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
        """Map CLI dtype strings to the torch values expected by loaders."""

        mapping = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.config.torch_dtype]

    def _resolve_multimodal_loader(self):
        """Pick the Hugging Face multimodal loader required by the active model profile."""

        loader_kind = str(self.profile.get("hf_loader") or "").lower()
        if loader_kind == "multimodal-lm":
            loader = getattr(transformers, "AutoModelForMultimodalLM", None)
            if loader is None:
                raise RuntimeError(
                    "This Transformers installation does not expose AutoModelForMultimodalLM, which is required for Gemma 4 multimodal models."
                )
            return loader
        return AutoModelForImageTextToText

    def _message_contains_media(self, messages: list[dict], *, media_type: str) -> bool:
        """Tell whether a multimodal chat payload includes one specific media type."""

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == media_type:
                    return True
        return False
