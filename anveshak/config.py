"""Runtime configuration and filesystem layout for Anveshak Console."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
DEFAULT_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

TEXT_EXTENSIONS = {
    ".bib",
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".ipynb",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".md",
    ".mjs",
    ".pdf",
    ".php",
    ".py",
    ".rb",
    ".rst",
    ".sh",
    ".sql",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

DOCUMENT_EXTENSIONS = {
    ".docx",
    ".pptx",
    ".xlsx",
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}

VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

IGNORED_DIR_NAMES = {
    "API_calls",
    ".cache",
    ".git",
    ".hg",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "checkpoints",
    "context_window",
    "logs",
    "node_modules",
    "venv",
}


@dataclass(slots=True)
class RuntimeConfig:
    """Collect tunable runtime settings and derived workspace paths."""

    workspace_root: Path
    model_id: str = DEFAULT_MODEL_ID
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    mode: str = "terminal"
    enable_web: bool = True
    web_mode: str = "auto"
    enable_workspace_indexing: bool = True
    open_browser: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    memory_top_k: int = 6
    file_top_k: int = 8
    web_top_k: int = 8
    max_new_tokens: int = 2048
    memory_note_max_words: int = 180
    workspace_chunk_chars: int = 1800
    workspace_chunk_overlap: int = 250
    direct_file_chunk_chars: int = 2200
    direct_file_chunk_overlap: int = 300
    web_chunk_chars: int = 1500
    web_chunk_overlap: int = 250
    max_indexed_file_bytes: int = 10 * 1024 * 1024
    max_inline_file_chars: int = 14_000
    max_web_pages_per_query: int = 6
    max_web_chars_per_page: int = 20_000
    max_recent_turns: int = 8
    embedding_batch_size: int = 8
    embedding_device: str = "cpu"
    max_pdf_inline_images: int = 8
    max_pdf_preview_pages: int = 2
    torch_dtype: str = "auto"
    attn_implementation: str = "sdpa"
    prepare_runtime_on_start: bool = True
    model_idle_unload_seconds: int = 180
    n_gpus: int | None = None
    max_gpu_memory_gib: int | None = None
    max_cpu_memory_gib: int | None = None
    kimi_server_url: str | None = None
    kimi_server_model: str | None = None
    kimi_server_api_key: str | None = None
    kimi_server_timeout_seconds: int = 30
    hf_cache_dir: Path | None = None
    checkpoints_dir: Path | None = None
    api_calls_dir: Path | None = None
    seed: int = 0
    model_local_path: Path | None = None
    embedding_local_path: Path | None = None
    whisper_model_name: str = "turbo"
    whisper_device: str = "cpu"

    context_dir: Path = field(init=False)
    memory_dir: Path = field(init=False)
    file_index_dir: Path = field(init=False)
    session_dir: Path = field(init=False)
    api_session_dir: Path = field(init=False)
    uploads_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    offload_dir: Path = field(init=False)
    static_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Resolve all user-facing directories once at startup."""
        self.workspace_root = self.workspace_root.expanduser().resolve()
        self.context_dir = self.workspace_root / "context_window"
        self.memory_dir = self.context_dir / "memory"
        self.file_index_dir = self.context_dir / "local_files"
        self.session_dir = self.context_dir / "sessions"
        self.api_session_dir = self.context_dir / "api_call_sessions"
        self.uploads_dir = self.context_dir / "uploads"
        self.cache_dir = self.context_dir / "cache"
        self.logs_dir = self.workspace_root / "logs"
        self.offload_dir = self.cache_dir / "offload"
        self.static_dir = Path(__file__).resolve().parent / "static"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = self.workspace_root / "checkpoints"
        if self.api_calls_dir is None:
            self.api_calls_dir = self.workspace_root / "API_calls"
        self.models_dir = self.checkpoints_dir / "models"
        if self.hf_cache_dir is None:
            self.hf_cache_dir = self.checkpoints_dir / "huggingface"

    def ensure_directories(self) -> None:
        """Create every workspace directory Anveshak expects to write into."""
        for directory in (
            self.context_dir,
            self.memory_dir,
            self.file_index_dir,
            self.session_dir,
            self.api_session_dir,
            self.uploads_dir,
            self.cache_dir,
            self.logs_dir,
            self.offload_dir,
            self.checkpoints_dir,
            self.models_dir,
            self.api_calls_dir,
            self.hf_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
