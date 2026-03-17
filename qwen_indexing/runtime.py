from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm

from .config import RuntimeConfig
from .model_catalog import MODEL_CATALOG


def _slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "--").replace(":", "--")


@dataclass(slots=True)
class RuntimeStatus:
    phase: str = "starting"
    message: str = "Preparing runtime"
    current_asset: str = ""
    current_file: str = ""
    files_total: int = 0
    files_completed: int = 0
    bytes_total: int = 0
    bytes_downloaded: int = 0
    ready: bool = False
    error: str = ""

    @property
    def progress(self) -> float:
        if self.bytes_total <= 0:
            if self.files_total <= 0:
                return 0.0
            return min(self.files_completed / self.files_total, 1.0)
        return min(self.bytes_downloaded / self.bytes_total, 1.0)


class RuntimeManager:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._status_changed = threading.Condition(self._lock)
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_version = 0
        self.status = RuntimeStatus()
        self.api = HfApi()

    def start_async(self) -> None:
        with self._lock:
            if self.status.ready and self._ready_event.is_set():
                return
            if self._thread and self._thread.is_alive():
                return
            self._ready_event.clear()
            self._thread = threading.Thread(target=self._prepare_runtime, daemon=True)
            self._thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._ready_event.wait(timeout)

    def status_dict(self) -> dict[str, Any]:
        with self._lock:
            payload = self._snapshot_locked()
        return payload

    def wait_for_status_change(self, last_version: int, timeout: float | None = None) -> dict[str, Any] | None:
        with self._status_changed:
            changed = self._status_changed.wait_for(lambda: self._status_version != last_version, timeout=timeout)
            if not changed:
                return None
            return self._snapshot_locked()

    def _prepare_runtime(self) -> None:
        try:
            self._update(phase="checking", message="Checking local model assets")
            model_path = self._ensure_repo_downloaded(self.config.model_id, role="assistant-model")
            embedding_path = self._ensure_repo_downloaded(self.config.embedding_model_id, role="embedding-model")
            self.config.model_local_path = model_path
            self.config.embedding_local_path = embedding_path
            self._update(phase="ready", message="Runtime is ready", ready=True, current_asset="", current_file="")
            self._ready_event.set()
        except Exception as exc:
            self._update(phase="error", message="Runtime preparation failed", error=str(exc), ready=False)
            self._ready_event.set()

    def _ensure_repo_downloaded(self, model_id: str, *, role: str) -> Path:
        candidate = Path(model_id)
        if candidate.exists():
            return candidate.resolve()

        target_dir = self.config.models_dir / _slugify_model_id(model_id)
        complete_marker = target_dir / ".download-complete"
        if complete_marker.exists():
            return target_dir

        self._update(phase="planning-download", current_asset=role, message=f"Preparing download for {model_id}")
        dry_run = snapshot_download(
            model_id,
            local_dir=target_dir,
            cache_dir=self.config.hf_cache_dir,
            dry_run=True,
        )
        total_bytes = 0
        files_total = 0
        for item in dry_run:
            files_total += 1
            total_bytes += int(getattr(item, "size_on_disk", None) or getattr(item, "size", None) or 0)

        self._update(
            phase="downloading",
            current_asset=role,
            message=f"Downloading {model_id}",
            current_file="Starting download",
            files_total=files_total,
            files_completed=0,
            bytes_total=total_bytes,
            bytes_downloaded=0,
        )

        tracker = _ProgressTracker(self, role=role)
        tracking_class = tracker.make_tqdm_class()
        snapshot_download(
            model_id,
            local_dir=target_dir,
            cache_dir=self.config.hf_cache_dir,
            max_workers=4,
            tqdm_class=tracking_class,
        )

        target_dir.mkdir(parents=True, exist_ok=True)
        complete_marker.write_text(model_id, encoding="utf-8")
        self._update(
            phase="downloaded",
            current_asset=role,
            message=f"Downloaded {model_id}",
            current_file="Completed",
            files_completed=files_total,
            bytes_downloaded=max(total_bytes, self.status.bytes_downloaded),
        )
        return target_dir

    def _update(self, **changes: Any) -> None:
        with self._status_changed:
            changed = False
            for key, value in changes.items():
                if getattr(self.status, key) == value:
                    continue
                setattr(self.status, key, value)
                changed = True
            if not changed:
                return
            self._status_version += 1
            self._status_changed.notify_all()

    def _snapshot_locked(self) -> dict[str, Any]:
        payload = asdict(self.status)
        payload["progress"] = self.status.progress
        payload["version"] = self._status_version
        payload["model_id"] = self.config.model_id
        payload["embedding_model_id"] = self.config.embedding_model_id
        payload["checkpoints_dir"] = str(self.config.checkpoints_dir)
        payload["available_models"] = MODEL_CATALOG
        return payload


class _ProgressTracker:
    def __init__(self, manager: RuntimeManager, *, role: str) -> None:
        self.manager = manager
        self.role = role
        self._lock = threading.Lock()
        self._bars: dict[int, dict[str, Any]] = {}
        self._baseline_files_total = manager.status.files_total
        self._baseline_bytes_total = manager.status.bytes_total

    def make_tqdm_class(self):
        tracker = self

        class TrackingTqdm(tqdm):
            def __init__(self, *args, **kwargs):
                name = kwargs.pop("name", "")
                if name and not kwargs.get("desc"):
                    kwargs["desc"] = str(name)
                desc = str(kwargs.get("desc", "") or "")
                unit = str(kwargs.get("unit", "") or "")
                super().__init__(*args, **kwargs)
                tracker._register_bar(self, name=name, desc=desc, unit=unit)

            def update(self, n=1):
                super().update(n)
                tracker._sync_bar(self, increment=float(n or 0))

            def refresh(self, *args, **kwargs):
                result = super().refresh(*args, **kwargs)
                tracker._sync_bar(self)
                return result

            def set_description(self, desc=None, refresh=True):
                result = super().set_description(desc=desc, refresh=refresh)
                tracker._sync_bar(self)
                return result

            def close(self):
                try:
                    tracker._sync_bar(self, closed=True)
                finally:
                    return super().close()

        return TrackingTqdm

    def _register_bar(self, bar: tqdm, *, name: str, desc: str, unit: str) -> None:
        with self._lock:
            self._bars[id(bar)] = {
                "name": name,
                "desc": desc,
                "total": float(bar.total or 0),
                "n": float(bar.n or 0),
                "kind": self._classify_bar(name=name, desc=desc, unit=unit),
                "closed": False,
            }
        self._emit_status()

    def _sync_bar(self, bar: tqdm, *, closed: bool = False, increment: float | None = None) -> None:
        with self._lock:
            state = self._bars.get(id(bar))
            if state is None:
                return
            state["desc"] = str(getattr(bar, "desc", "") or state["desc"])
            state["total"] = float(bar.total or 0)
            if increment is not None and getattr(bar, "disable", False):
                state["n"] = float(state["n"] or 0) + increment
            else:
                state["n"] = float(bar.n or 0)
            if closed:
                state["closed"] = True
        self._emit_status()

    def _emit_status(self) -> None:
        with self._lock:
            bytes_bar = self._select_bar("bytes")
            files_bar = self._select_bar("files")

            files_total = self._baseline_files_total
            files_completed = 0
            if files_bar is not None:
                files_total = max(files_total, int(math.floor(files_bar["total"] or 0)))
                files_completed = min(
                    max(0, int(math.floor(files_bar["n"] or 0))),
                    files_total or max(0, int(math.floor(files_bar["n"] or 0))),
                )

            bytes_total = self._baseline_bytes_total
            bytes_downloaded = 0
            current_file = "Downloading"
            if bytes_bar is not None:
                bytes_total = max(bytes_total, int(math.floor(bytes_bar["total"] or 0)))
                bytes_downloaded = max(0, int(math.floor(bytes_bar["n"] or 0)))
                current_file = bytes_bar["desc"] or current_file
            elif files_bar is not None:
                current_file = files_bar["desc"] or current_file

        self.manager._update(
            phase="downloading",
            current_asset=self.role,
            current_file=current_file,
            files_total=files_total,
            files_completed=files_completed,
            bytes_total=bytes_total,
            bytes_downloaded=bytes_downloaded,
        )

    def _select_bar(self, kind: str) -> dict[str, Any] | None:
        candidates = [state for state in self._bars.values() if state["kind"] == kind]
        if not candidates:
            return None
        live = [state for state in candidates if not state["closed"]]
        ordered = live or candidates
        return max(ordered, key=lambda state: (state["total"], state["n"]))

    @staticmethod
    def _classify_bar(*, name: str, desc: str, unit: str) -> str:
        lowered_desc = desc.lower()
        if name == "huggingface_hub.snapshot_download" or unit == "B" or lowered_desc.startswith("downloading"):
            return "bytes"
        if lowered_desc.startswith("fetching"):
            return "files"
        return "other"
