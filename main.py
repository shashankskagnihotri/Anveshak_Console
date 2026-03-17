from __future__ import annotations

import argparse
import time
import webbrowser
from pathlib import Path
from threading import Thread
from urllib.request import urlopen

import uvicorn

from qwen_indexing.chat.service import ChatService
from qwen_indexing.config import RuntimeConfig
from qwen_indexing.server import build_app
from qwen_indexing.terminal import TerminalChat
from qwen_indexing.utils import apply_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Anveshak Console, private multimodal reasoning with local memory and live retrieval."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--embedding-model-id", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--mode", choices=("terminal", "web", "both"), default="terminal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--web-mode", choices=("auto", "always", "off"), default="auto")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--max-gpu-memory-gib", type=int)
    parser.add_argument("--max-cpu-memory-gib", type=int)
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_global_seed(args.seed)
    config = RuntimeConfig(
        workspace_root=Path.cwd(),
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        mode=args.mode,
        web_mode=args.web_mode,
        enable_web=args.web_mode != "off",
        open_browser=args.open_browser,
        host=args.host,
        port=args.port,
        max_gpu_memory_gib=args.max_gpu_memory_gib,
        max_cpu_memory_gib=args.max_cpu_memory_gib,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
    )
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

    TerminalChat(service).run()


def _open_browser_when_ready(url: str) -> None:
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


if __name__ == "__main__":
    main()
