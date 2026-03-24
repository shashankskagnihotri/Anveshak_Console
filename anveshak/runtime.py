"""Prepare checkpoints locally and stream runtime readiness to the UI."""

from __future__ import annotations

import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, login, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from tqdm.auto import tqdm

from .config import RuntimeConfig
from .model_catalog import MODEL_CATALOG, get_model_profile


HUGGINGFACE_TOKEN_ENV_VAR = "HUGGINGFACE_HUB_TOKEN"
HUGGINGFACE_TOKEN_ALIAS_ENV_VARS = (
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGING_FACE_TOKEN",
)
HUGGINGFACE_TOKEN_GUIDE_URL = "https://huggingface.co/docs/hub/main/security-tokens"
HUGGINGFACE_TOKEN_SETTINGS_URL = "https://huggingface.co/settings/tokens"


def _slugify_model_id(model_id: str) -> str:
    """Map a model id to a filesystem-friendly directory name."""

    return model_id.replace("/", "--").replace(":", "--")


def _looks_like_local_model_repo(path: Path) -> bool:
    """Recognize an already-populated local model directory without re-contacting the hub."""

    if not path.exists() or not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    if not children:
        return False

    metadata_files = {
        "config.json",
        "generation_config.json",
        "modules.json",
        "preprocessor_config.json",
        "processor_config.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    artifact_files = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.model",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
    }
    has_metadata = any((path / name).exists() for name in metadata_files)
    has_artifacts = any((path / name).exists() for name in artifact_files)
    if not has_artifacts:
        has_artifacts = any(path.glob("*.safetensors")) or any(path.glob("*.bin")) or any(path.glob("*.pt"))
    return has_metadata and has_artifacts


def _repo_file_inventory(path: Path) -> tuple[list[Path], int]:
    """Return all regular files inside a model repo and their total size in bytes."""

    files: list[Path] = []
    total_bytes = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        files.append(candidate)
        try:
            total_bytes += candidate.stat().st_size
        except OSError:
            continue
    return files, total_bytes


def _free_bytes(path: Path) -> int | None:
    """Return free bytes available on the filesystem that backs ``path``."""

    try:
        stats = os.statvfs(path)
    except OSError:
        return None
    return int(stats.f_bavail * stats.f_frsize)


@dataclass(slots=True)
class RuntimeStatus:
    """Current checkpoint-preparation state exposed to the UI and logs."""

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
    huggingface_auth_required: bool = False
    huggingface_auth_message: str = ""
    huggingface_auth_model_id: str = ""

    @property
    def progress(self) -> float:
        """Return normalized progress from either bytes or file counts."""

        if self.bytes_total <= 0:
            if self.files_total <= 0:
                return 0.0
            return min(self.files_completed / self.files_total, 1.0)
        return min(self.bytes_downloaded / self.bytes_total, 1.0)


class RuntimeManager:
    """Prepare local assets once and broadcast status updates to consumers."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._status_changed = threading.Condition(self._lock)
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._status_version = 0
        self.status = RuntimeStatus()
        self.api = HfApi()
        self._huggingface_token: str | None = None
        self._huggingface_token_source: str = ""

    def start_async(self) -> None:
        """Spawn the background preparation thread if it is not already running."""

        with self._lock:
            if self.status.ready and self._ready_event.is_set():
                return
            if self._thread and self._thread.is_alive():
                return
            self._ready_event.clear()
            self._thread = threading.Thread(target=self._prepare_runtime, daemon=True)
            self._thread.start()

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the runtime has finished preparing or a timeout expires."""

        return self._ready_event.wait(timeout)

    def status_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the current runtime state."""

        with self._lock:
            payload = self._snapshot_locked()
        return payload

    def wait_for_status_change(self, last_version: int, timeout: float | None = None) -> dict[str, Any] | None:
        """Block until a new runtime status version is available."""

        with self._status_changed:
            changed = self._status_changed.wait_for(lambda: self._status_version != last_version, timeout=timeout)
            if not changed:
                return None
            return self._snapshot_locked()

    def _prepare_runtime(self) -> None:
        """Download or resolve both reasoning and embedding model assets."""

        try:
            self._refresh_huggingface_token_from_environment()
            self._update(
                phase="checking",
                message="Checking local model assets",
                ready=False,
                error="",
                huggingface_auth_required=False,
                huggingface_auth_message="",
                huggingface_auth_model_id="",
            )
            profile = get_model_profile(self.config.model_id)
            if profile.get("preferred_runtime_backend") == "kimi_server" and self.config.kimi_server_url:
                model_path = None
                self._update(
                    phase="checking",
                    current_asset="assistant-model",
                    message="Using the configured OpenAI-compatible served backend",
                    current_file="Skipping local reasoning checkpoint staging",
                )
            elif profile.get("requires_server_backend"):
                label = profile.get("label") or self.config.model_id
                raise RuntimeError(
                    f"{label} is configured as a server-backed reasoning model. "
                    "Pass --kimi-server-url for an OpenAI-compatible endpoint and, if needed, "
                    "--kimi-server-model for the served model name."
                )
            else:
                model_path = self._ensure_repo_downloaded(self.config.model_id, role="assistant-model")
            embedding_path = self._ensure_repo_downloaded(self.config.embedding_model_id, role="embedding-model")
            self.config.model_local_path = model_path
            self.config.embedding_local_path = embedding_path
            self._update(
                phase="ready",
                message="Runtime is ready",
                ready=True,
                current_asset="",
                current_file="",
                huggingface_auth_required=False,
                huggingface_auth_message="",
                huggingface_auth_model_id="",
            )
            self._ready_event.set()
        except HuggingFaceAuthRequiredError as exc:
            self._update(
                phase="auth-required",
                message=f"Hugging Face token required for {exc.label}",
                current_asset=exc.role,
                current_file="Waiting for Hugging Face token",
                error=str(exc),
                ready=False,
                huggingface_auth_required=True,
                huggingface_auth_message=exc.user_message,
                huggingface_auth_model_id=exc.model_id,
            )
            self._ready_event.set()
        except Exception as exc:
            self._update(
                phase="error",
                message="Runtime preparation failed",
                error=str(exc),
                ready=False,
                huggingface_auth_required=False,
                huggingface_auth_message="",
                huggingface_auth_model_id="",
            )
            self._ready_event.set()

    def configure_huggingface_token(self, token: str) -> dict[str, Any]:
        """Accept a user token from the UI and retry runtime preparation."""

        normalized = token.strip()
        if not normalized:
            raise ValueError("Enter a Hugging Face token before continuing.")
        self._activate_huggingface_token(normalized, source="manual", validate=True, persist=True)
        self._update(
            phase="checking",
            message="Hugging Face token accepted. Retrying checkpoint access",
            current_file="Reconnecting to Hugging Face",
            error="",
            ready=False,
            huggingface_auth_required=False,
            huggingface_auth_message="",
            huggingface_auth_model_id="",
        )
        self.start_async()
        return self.status_dict()

    def _ensure_repo_downloaded(self, model_id: str, *, role: str) -> Path:
        """Resolve a local path for a model id, downloading it when required."""

        candidate = Path(model_id)
        if candidate.exists():
            return self._resolve_runtime_repo(candidate.resolve(), model_id=model_id, role=role)

        target_dir = self.config.models_dir / _slugify_model_id(model_id)
        complete_marker = target_dir / ".download-complete"
        if complete_marker.exists():
            return self._resolve_runtime_repo(target_dir, model_id=model_id, role=role)
        if _looks_like_local_model_repo(target_dir):
            complete_marker.write_text(model_id, encoding="utf-8")
            return self._resolve_runtime_repo(target_dir, model_id=model_id, role=role)

        self._update(phase="planning-download", current_asset=role, message=f"Preparing download for {model_id}")
        token = self._active_huggingface_token()
        try:
            dry_run = snapshot_download(
                model_id,
                local_dir=target_dir,
                cache_dir=self.config.hf_cache_dir,
                dry_run=True,
                token=token,
            )
        except Exception as exc:
            self._raise_if_huggingface_auth_required(exc, model_id=model_id, role=role)
            raise
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
        try:
            snapshot_download(
                model_id,
                local_dir=target_dir,
                cache_dir=self.config.hf_cache_dir,
                max_workers=4,
                tqdm_class=tracking_class,
                token=token,
            )
        except Exception as exc:
            self._raise_if_huggingface_auth_required(exc, model_id=model_id, role=role)
            raise

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
        return self._resolve_runtime_repo(target_dir, model_id=model_id, role=role)

    def _resolve_runtime_repo(self, source_dir: Path, *, model_id: str, role: str) -> Path:
        """Return the repo path the runtime should actually load from."""

        if not self._should_stage_repo_locally(source_dir):
            return source_dir
        return self._stage_repo_locally(source_dir, model_id=model_id, role=role)

    def _local_stage_root(self) -> Path:
        """Return the persistent local-SSD cache directory for large model repos."""

        override = os.environ.get("ANVESHAK_LOCAL_MODEL_CACHE_DIR")
        if override:
            return Path(override).expanduser().resolve()
        return Path("/tmp/anveshak-model-cache").resolve()

    def _should_stage_repo_locally(self, source_dir: Path) -> bool:
        """Decide whether a large repo should be copied from remote storage to local NVMe."""

        if not _looks_like_local_model_repo(source_dir):
            return False

        stage_root = self._local_stage_root()
        try:
            source_dir.relative_to(stage_root)
        except ValueError:
            pass
        else:
            return False

        try:
            if source_dir.stat().st_dev == stage_root.stat().st_dev:
                return False
        except OSError:
            return False

        _, repo_bytes = _repo_file_inventory(source_dir)
        if repo_bytes < 64 * 1024**3:
            return False

        free_bytes = _free_bytes(stage_root.parent if not stage_root.exists() else stage_root)
        if free_bytes is None:
            return False
        return free_bytes >= repo_bytes + 64 * 1024**3

    def _stage_repo_locally(self, source_dir: Path, *, model_id: str, role: str) -> Path:
        """Copy a large repo to the local NVMe cache and reuse it across restarts."""

        stage_root = self._local_stage_root()
        stage_root.mkdir(parents=True, exist_ok=True)
        staged_dir = stage_root / _slugify_model_id(model_id)
        complete_marker = staged_dir / ".stage-complete"
        marker_payload = f"{model_id}\n{source_dir}\n"
        if complete_marker.exists() and _looks_like_local_model_repo(staged_dir):
            try:
                if complete_marker.read_text(encoding="utf-8") == marker_payload:
                    return staged_dir
            except OSError:
                pass

        source_files, total_bytes = _repo_file_inventory(source_dir)
        files_total = len(source_files)
        if files_total == 0:
            return source_dir

        staged_dir.mkdir(parents=True, exist_ok=True)
        existing_bytes = 0
        existing_files = 0
        copy_plan: list[tuple[Path, Path, int]] = []
        for source_file in source_files:
            relative = source_file.relative_to(source_dir)
            target_file = staged_dir / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                source_stat = source_file.stat()
            except OSError:
                continue
            source_size = int(source_stat.st_size)
            if target_file.exists():
                try:
                    target_stat = target_file.stat()
                except OSError:
                    target_stat = None
                if target_stat is not None and int(target_stat.st_size) == source_size:
                    existing_files += 1
                    existing_bytes += source_size
                    continue
            copy_plan.append((source_file, target_file, source_size))

        self._update(
            phase="staging-local-cache",
            current_asset=role,
            message=f"Staging {model_id} to local NVMe cache",
            current_file="Preparing local copy",
            files_total=files_total,
            files_completed=existing_files,
            bytes_total=total_bytes,
            bytes_downloaded=existing_bytes,
        )

        if copy_plan:
            copied_files = existing_files
            copied_bytes = existing_bytes
            workers = max(1, min(len(copy_plan), 4))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._copy_repo_file, source_file, target_file): (source_file, target_file, size)
                    for source_file, target_file, size in copy_plan
                }
                for future in as_completed(futures):
                    source_file, target_file, size = futures[future]
                    future.result()
                    copied_files += 1
                    copied_bytes += size
                    self._update(
                        phase="staging-local-cache",
                        current_asset=role,
                        current_file=str(target_file.relative_to(staged_dir)),
                        files_total=files_total,
                        files_completed=copied_files,
                        bytes_total=total_bytes,
                        bytes_downloaded=copied_bytes,
                    )

        complete_marker.write_text(marker_payload, encoding="utf-8")
        self._update(
            phase="staged-local-cache",
            current_asset=role,
            message=f"Using local NVMe cache for {model_id}",
            current_file="Completed",
            files_total=files_total,
            files_completed=files_total,
            bytes_total=total_bytes,
            bytes_downloaded=total_bytes,
        )
        return staged_dir

    @staticmethod
    def _copy_repo_file(source_file: Path, target_file: Path) -> None:
        """Copy one repo artifact while preserving metadata."""

        shutil.copy2(source_file, target_file)

    def _update(self, **changes: Any) -> None:
        """Apply status changes only when they actually differ from the last state."""

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
        """Build the public status payload while holding the manager lock."""

        payload = asdict(self.status)
        payload["progress"] = self.status.progress
        payload["version"] = self._status_version
        payload["model_id"] = self.config.model_id
        payload["embedding_model_id"] = self.config.embedding_model_id
        payload["checkpoints_dir"] = str(self.config.checkpoints_dir)
        payload["huggingface_token_env_var"] = HUGGINGFACE_TOKEN_ENV_VAR
        payload["huggingface_token_alias_env_var"] = HUGGINGFACE_TOKEN_ALIAS_ENV_VARS[0]
        payload["huggingface_token_alias_env_vars"] = list(HUGGINGFACE_TOKEN_ALIAS_ENV_VARS)
        payload["huggingface_token_guide_url"] = HUGGINGFACE_TOKEN_GUIDE_URL
        payload["huggingface_token_settings_url"] = HUGGINGFACE_TOKEN_SETTINGS_URL
        payload["huggingface_token_source"] = self._huggingface_token_source
        payload["available_models"] = MODEL_CATALOG
        return payload

    def _refresh_huggingface_token_from_environment(self) -> None:
        """Best-effort pickup of a token configured in the shell environment."""

        if self._huggingface_token_source == "manual" and self._huggingface_token:
            return

        for env_name in (HUGGINGFACE_TOKEN_ENV_VAR, *HUGGINGFACE_TOKEN_ALIAS_ENV_VARS):
            candidate = os.environ.get(env_name, "").strip()
            if not candidate:
                continue
            if candidate == self._huggingface_token and self._huggingface_token_source == f"env:{env_name}":
                return
            self._activate_huggingface_token(candidate, source=f"env:{env_name}", validate=False, persist=True)
            return

    def _activate_huggingface_token(
        self,
        token: str,
        *,
        source: str,
        validate: bool,
        persist: bool,
    ) -> None:
        """Store one token for the current process and optionally persist it locally."""

        normalized = token.strip()
        if not normalized:
            raise ValueError("Hugging Face tokens cannot be empty.")
        if validate:
            try:
                self.api.whoami(token=normalized)
            except Exception as exc:
                raise ValueError(
                    "That Hugging Face token could not be validated. Make sure it is a personal user token "
                    "with access to the requested gated model."
                ) from exc

        self._huggingface_token = normalized
        self._huggingface_token_source = source
        os.environ[HUGGINGFACE_TOKEN_ENV_VAR] = normalized
        for env_name in HUGGINGFACE_TOKEN_ALIAS_ENV_VARS:
            os.environ[env_name] = normalized

        if not persist:
            return
        try:
            login(token=normalized, add_to_git_credential=False, skip_if_logged_in=False)
        except Exception:
            # The in-process token still works even if persistent login is unavailable.
            return

    def _active_huggingface_token(self) -> str | None:
        """Return the current token if Anveshak has one available."""

        if self._huggingface_token:
            return self._huggingface_token
        for env_name in (HUGGINGFACE_TOKEN_ENV_VAR, *HUGGINGFACE_TOKEN_ALIAS_ENV_VARS):
            candidate = os.environ.get(env_name, "").strip()
            if candidate:
                return candidate
        return None

    def _raise_if_huggingface_auth_required(self, exc: Exception, *, model_id: str, role: str) -> None:
        """Translate gated Hub failures into a browser-friendly auth prompt state."""

        if isinstance(exc, GatedRepoError):
            raise HuggingFaceAuthRequiredError(model_id=model_id, role=role) from exc

        if isinstance(exc, HfHubHTTPError):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            message = str(exc).lower()
            if status_code in {401, 403} or "gated" in message or "authorization" in message:
                raise HuggingFaceAuthRequiredError(model_id=model_id, role=role) from exc


class HuggingFaceAuthRequiredError(RuntimeError):
    """Raised when the selected repo appears to need authenticated Hugging Face access."""

    def __init__(self, *, model_id: str, role: str) -> None:
        profile = get_model_profile(model_id)
        label = profile.get("label") or model_id
        user_message = (
            f"{label} appears to require a Hugging Face user token before it can be downloaded. "
            f"Set {HUGGINGFACE_TOKEN_ENV_VAR} in your terminal or bash profile, or enter a token in the browser popup. "
            "Anveshak will retry automatically after a token is provided."
        )
        super().__init__(user_message)
        self.model_id = model_id
        self.role = role
        self.label = label
        self.user_message = user_message


class _ProgressTracker:
    """Translate Hugging Face download tqdm bars into one aggregate status payload."""

    def __init__(self, manager: RuntimeManager, *, role: str) -> None:
        self.manager = manager
        self.role = role
        self._lock = threading.Lock()
        self._bars: dict[int, dict[str, Any]] = {}
        self._baseline_files_total = manager.status.files_total
        self._baseline_bytes_total = manager.status.bytes_total

    def make_tqdm_class(self):
        """Wrap tqdm so snapshot_download progress can update the browser overlay."""

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
