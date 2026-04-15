from __future__ import annotations

import types
from pathlib import Path
from threading import Event
import wave

from fastapi.testclient import TestClient
import numpy as np

from anveshak import server as server_module
from anveshak.chat.service import ChatService
from anveshak.config import RuntimeConfig
from anveshak.transcription import WhisperTranscriber


class FakeWhisperTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.paths: list[Path] = []
        self.loaded = False
        self.warmup_event = Event()

    def transcribe(self, audio_path: Path) -> str:
        self.paths.append(audio_path)
        return self.text

    def warmup(self) -> None:
        self.loaded = True
        self.warmup_event.set()

    def is_loaded(self) -> bool:
        return self.loaded


def _build_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        workspace_root=tmp_path,
        enable_web=False,
        enable_workspace_indexing=False,
        prepare_runtime_on_start=False,
    )


def test_transcribe_microphone_recording_returns_whisper_payload(tmp_path: Path) -> None:
    service = ChatService(_build_config(tmp_path))
    fake_transcriber = FakeWhisperTranscriber("hello from whisper")
    service.whisper_transcriber = fake_transcriber

    audio_path = tmp_path / "microphone-recording.wav"
    audio_path.write_bytes(b"RIFFtest")

    payload = service.transcribe_microphone_recording(audio_path)

    assert payload == {
        "attachment_name": "microphone-recording.wav",
        "backend": "Whisper",
        "text": "hello from whisper",
    }
    assert fake_transcriber.paths == [audio_path]


def test_schedule_whisper_prewarm_loads_transcriber_in_background(tmp_path: Path) -> None:
    service = ChatService(_build_config(tmp_path))
    fake_transcriber = FakeWhisperTranscriber("hello from whisper")
    service.whisper_transcriber = fake_transcriber

    scheduled = service.schedule_whisper_prewarm()

    assert scheduled is True
    assert fake_transcriber.warmup_event.wait(timeout=2.0)
    assert fake_transcriber.is_loaded() is True
    assert service.schedule_whisper_prewarm() is False


def test_whisper_transcriber_decodes_browser_wav_without_ffmpeg(tmp_path: Path) -> None:
    audio_path = tmp_path / "microphone-recording.wav"
    sample_rate = 24_000
    seconds = 0.4
    sample_count = int(sample_rate * seconds)
    waveform = (0.3 * np.sin(np.linspace(0, np.pi * 8, sample_count, dtype=np.float32)) * 32767).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform.tobytes())

    captured: dict[str, object] = {}

    class FakeModel:
        def transcribe(self, audio_input, **kwargs):
            captured["audio_input"] = audio_input
            captured["kwargs"] = kwargs
            return {"text": "decoded without ffmpeg"}

    transcriber = WhisperTranscriber()
    transcriber._model = FakeModel()

    text = transcriber.transcribe(audio_path)

    assert text == "decoded without ffmpeg"
    assert isinstance(captured["audio_input"], np.ndarray)
    assert captured["audio_input"].dtype == np.float32
    assert captured["audio_input"].ndim == 1
    assert captured["audio_input"].size > 0
    assert captured["kwargs"]["task"] == "transcribe"


def test_submit_message_endpoint_accepts_attachment_only_payload(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    service = ChatService(config)
    captured: dict[str, object] = {}

    def fake_submit_message(*, session_id, text, attachments, web_mode, media_mode):
        captured["session_id"] = session_id
        captured["text"] = text
        captured["attachments"] = attachments
        captured["web_mode"] = web_mode
        captured["media_mode"] = media_mode
        return types.SimpleNamespace(run_id="run-attachment-only", session_id=session_id)

    service.submit_message = fake_submit_message  # type: ignore[method-assign]

    client = TestClient(server_module.build_app(config, service))
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        data={"web_mode": "auto", "media_mode": "safe"},
        files=[("files", ("clip.wav", b"RIFFtest", "audio/wav"))],
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-attachment-only"
    assert captured["text"] == ""
    attachments = captured["attachments"]
    assert isinstance(attachments, list)
    assert len(attachments) == 1
    assert attachments[0].media_kind == "audio"
    assert captured["media_mode"] == "safe"


def test_microphone_transcription_endpoint_returns_transcript(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    service = ChatService(config)
    captured: dict[str, Path] = {}

    def fake_transcribe_microphone_recording(path: Path):
        captured["path"] = path
        return {"attachment_name": path.name, "backend": "Whisper", "text": "edited transcript"}

    service.transcribe_microphone_recording = fake_transcribe_microphone_recording  # type: ignore[method-assign]

    client = TestClient(server_module.build_app(config, service))
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post(
        f"/api/sessions/{session_id}/microphone-transcription",
        files={"file": ("microphone-recording.wav", b"RIFFtest", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "attachment_name": "microphone-recording.wav",
        "backend": "Whisper",
        "text": "edited transcript",
    }
    assert captured["path"].name == "microphone-recording.wav"


def test_whisper_warmup_endpoint_schedules_background_load(tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    service = ChatService(config)
    fake_transcriber = FakeWhisperTranscriber("hello from whisper")
    service.whisper_transcriber = fake_transcriber

    client = TestClient(server_module.build_app(config, service))

    response = client.post("/api/runtime/whisper-warmup")

    assert response.status_code == 200
    assert response.json() == {"scheduled": True}
    assert fake_transcriber.warmup_event.wait(timeout=2.0)
