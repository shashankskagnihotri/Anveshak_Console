"""Speech-transcription helpers shared by chat entrypoints."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
import wave

import numpy as np
import torch

from .utils import compact_whitespace

_WHISPER_SAMPLE_RATE = 16_000


class WhisperTranscriber:
    """Lazy wrapper around the official OpenAI Whisper package."""

    def __init__(self, *, model_name: str = "turbo", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def is_loaded(self) -> bool:
        """Return whether the Whisper checkpoint is already resident in memory."""

        return self._model is not None

    def warmup(self) -> None:
        """Load the Whisper checkpoint ahead of the first real transcription request."""

        self._ensure_model()

    def transcribe(self, audio_path: Path) -> str:
        """Return one plain-text transcript from the provided local audio file."""

        model = self._ensure_model()
        use_fp16 = self.device.startswith("cuda") and torch.cuda.is_available()
        audio_input: str | np.ndarray = str(audio_path)
        if audio_path.suffix.lower() == ".wav":
            audio_input = self._load_wav_audio_samples(audio_path)
        try:
            with self._inference_lock:
                result = model.transcribe(audio_input, task="transcribe", fp16=use_fp16, verbose=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Whisper transcription requires the `ffmpeg` command-line tool to be installed and available on PATH."
            ) from exc
        except Exception as exc:
            if "ffmpeg" in str(exc).lower():
                raise RuntimeError(
                    "Whisper transcription requires the `ffmpeg` command-line tool to be installed and available on PATH."
                ) from exc
            raise
        return compact_whitespace(str(result.get("text", "")))

    def _ensure_model(self):
        """Load the requested Whisper checkpoint once and reuse it across turns."""

        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Whisper transcription requires the `openai-whisper` package. "
                    "Install it with `pip install -U openai-whisper`."
                ) from exc
            self._model = whisper.load_model(self.model_name, device=self.device)
            return self._model

    def _load_wav_audio_samples(self, audio_path: Path) -> np.ndarray:
        """Decode browser-recorded PCM WAV clips without depending on ffmpeg."""

        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                channel_count = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                raw_frames = wav_file.readframes(frame_count)
        except wave.Error as exc:
            raise RuntimeError(f"Whisper could not parse the WAV audio in {audio_path.name}.") from exc

        if sample_width != 2:
            raise RuntimeError(
                f"Whisper currently expects 16-bit PCM WAV audio for direct microphone transcription, but {audio_path.name} uses {sample_width * 8}-bit samples."
            )

        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
        if not samples.size:
            return samples
        samples /= 32768.0
        if channel_count > 1:
            samples = samples.reshape(-1, channel_count).mean(axis=1)
        if sample_rate != _WHISPER_SAMPLE_RATE:
            samples = self._resample_audio(samples, source_rate=sample_rate, target_rate=_WHISPER_SAMPLE_RATE)
        return samples.astype(np.float32, copy=False)

    @staticmethod
    def _resample_audio(samples: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
        """Resample one mono waveform with lightweight linear interpolation."""

        if source_rate <= 0 or target_rate <= 0 or samples.size == 0:
            return samples.astype(np.float32, copy=False)
        if source_rate == target_rate:
            return samples.astype(np.float32, copy=False)
        duration_seconds = samples.shape[0] / float(source_rate)
        target_length = max(1, int(round(duration_seconds * target_rate)))
        source_positions = np.arange(samples.shape[0], dtype=np.float32)
        target_positions = np.linspace(0, samples.shape[0] - 1, num=target_length, dtype=np.float32)
        return np.interp(target_positions, source_positions, samples).astype(np.float32, copy=False)
