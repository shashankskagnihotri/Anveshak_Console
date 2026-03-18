"""Command-line entrypoints for the installable ``anveshak`` executable."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import webbrowser
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser used by ``anveshak`` and ``main.py``."""

    parser = argparse.ArgumentParser(
        prog="anveshak",
        description="Run Anveshak Console with local reasoning, live retrieval, and persistent memory.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("terminal", "web", "both"),
        help="Run mode. Defaults to terminal when omitted.",
    )
    parser.add_argument("--mode", dest="legacy_mode", choices=("terminal", "web", "both"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--embedding-model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--web-mode", choices=("auto", "always", "off"), default="auto")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--n_GPUs", dest="n_gpus", type=int)
    parser.add_argument("--max-gpu-memory-gib", type=int)
    parser.add_argument("--max-cpu-memory-gib", type=int)
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--kimi-server-url")
    parser.add_argument("--kimi-server-model")
    parser.add_argument("--kimi-server-api-key")
    parser.add_argument("--kimi-server-timeout-seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments while supporting both new subcommands and legacy ``--mode``."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command and args.legacy_mode and args.command != args.legacy_mode:
        parser.error("Pass either the positional command or --mode, not conflicting values for both.")
    args.mode = args.command or args.legacy_mode or "terminal"
    return args


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    """Translate parsed CLI arguments into the shared runtime configuration object."""

    from .config import RuntimeConfig

    return RuntimeConfig(
        workspace_root=args.workspace_root,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        mode=args.mode,
        web_mode=args.web_mode,
        enable_web=args.web_mode != "off",
        open_browser=args.open_browser,
        host=args.host,
        port=args.port,
        n_gpus=args.n_gpus,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        kimi_server_url=args.kimi_server_url or os.environ.get("ANVESHAK_KIMI_SERVER_URL"),
        kimi_server_model=args.kimi_server_model or os.environ.get("ANVESHAK_KIMI_SERVER_MODEL"),
        kimi_server_api_key=args.kimi_server_api_key or os.environ.get("ANVESHAK_KIMI_SERVER_API_KEY"),
        kimi_server_timeout_seconds=args.kimi_server_timeout_seconds,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
    )


def run_from_args(args: argparse.Namespace) -> None:
    """Launch the requested interface mode from already-parsed CLI arguments."""

    try:
        configured_gpu_count = _configure_visible_cuda_devices(args.n_gpus)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if configured_gpu_count is not None:
        args.n_gpus = configured_gpu_count

    print(f"Starting Anveshak Console in {args.mode} mode...", flush=True)

    import uvicorn

    from .chat.service import ChatService
    from .server import build_app
    from .terminal import TerminalChat
    from .utils import apply_global_seed

    apply_global_seed(args.seed)
    config = build_runtime_config(args)
    service = ChatService(config)
    web_url = f"http://{args.host}:{args.port}"

    if args.mode in {"web", "both"}:
        app = build_app(config, service)
        if args.open_browser:
            Thread(target=_open_browser_when_ready, args=(web_url,), daemon=True).start()

        if args.mode == "web":
            uvicorn.run(app, host=args.host, port=args.port)
            return

        server_thread = Thread(
            target=uvicorn.run,
            kwargs={"app": app, "host": args.host, "port": args.port},
            daemon=True,
        )
        server_thread.start()
        print(f"Web server starting on {web_url}", flush=True)

    TerminalChat(service).run()


def _configure_visible_cuda_devices(requested_gpu_count: int | None) -> int | None:
    """Restrict the process to the requested number of single-node GPUs before torch is imported."""

    if requested_gpu_count is None:
        return None
    if requested_gpu_count < 1:
        raise ValueError("--n_GPUs must be at least 1.")

    available_devices = _discover_visible_cuda_devices()
    if available_devices is not None:
        if requested_gpu_count > len(available_devices):
            raise ValueError(
                f"--n_GPUs={requested_gpu_count} was requested, but only {len(available_devices)} visible GPU(s) were found."
            )
        selected_devices = available_devices[:requested_gpu_count]
    else:
        selected_devices = [str(index) for index in range(requested_gpu_count)]

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_devices)
    return len(selected_devices)


def _discover_visible_cuda_devices() -> list[str] | None:
    """Return the currently visible single-node CUDA device identifiers when they can be discovered."""

    visible_devices = _parse_visible_cuda_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices is not None:
        return visible_devices

    proc_gpu_root = Path("/proc/driver/nvidia/gpus")
    if proc_gpu_root.exists():
        gpu_entries = sorted(entry for entry in proc_gpu_root.iterdir() if entry.is_dir())
        if gpu_entries:
            return [str(index) for index, _ in enumerate(gpu_entries)]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if result.returncode != 0:
        return None

    gpu_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not gpu_lines:
        return None
    return [str(index) for index, _ in enumerate(gpu_lines)]


def _parse_visible_cuda_devices(raw_value: str | None) -> list[str] | None:
    """Parse CUDA_VISIBLE_DEVICES while preserving the user's existing visibility mask."""

    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized or normalized == "-1":
        return []
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _open_browser_when_ready(url: str) -> None:
    """Delay browser launch until the FastAPI server reports a healthy status."""

    health_url = f"{url}/api/health"
    for _ in range(120):
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``anveshak`` and ``python -m anveshak``."""

    run_from_args(parse_args(argv))
