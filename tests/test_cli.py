from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from anveshak.cli import _configure_visible_cuda_devices, build_runtime_config, parse_args


def test_cli_positional_mode_is_supported(tmp_path: Path) -> None:
    args = parse_args(
        [
            "web",
            "--workspace-root",
            str(tmp_path),
            "--port",
            "8123",
            "--seed",
            "4",
        ]
    )

    config = build_runtime_config(args)

    assert args.mode == "web"
    assert config.workspace_root == tmp_path.resolve()
    assert config.port == 8123
    assert config.seed == 4


def test_cli_legacy_mode_flag_is_still_supported(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--mode",
            "both",
            "--workspace-root",
            str(tmp_path),
            "--host",
            "0.0.0.0",
        ]
    )

    config = build_runtime_config(args)

    assert args.mode == "both"
    assert config.host == "0.0.0.0"


def test_cli_accepts_kimi_server_backend_options(tmp_path: Path) -> None:
    args = parse_args(
        [
            "web",
            "--workspace-root",
            str(tmp_path),
            "--model-id",
            "moonshotai/Kimi-K2-Instruct",
            "--kimi-server-url",
            "http://127.0.0.1:9000",
            "--kimi-server-model",
            "kimi-k2-local",
            "--kimi-server-api-key",
            "secret",
            "--kimi-server-timeout-seconds",
            "45",
        ]
    )

    config = build_runtime_config(args)

    assert config.kimi_server_url == "http://127.0.0.1:9000"
    assert config.kimi_server_model == "kimi-k2-local"
    assert config.kimi_server_api_key == "secret"
    assert config.kimi_server_timeout_seconds == 45


def test_cli_accepts_requested_gpu_count(tmp_path: Path) -> None:
    args = parse_args(
        [
            "web",
            "--workspace-root",
            str(tmp_path),
            "--n_GPUs",
            "2",
        ]
    )

    config = build_runtime_config(args)

    assert config.n_gpus == 2


def test_configure_visible_cuda_devices_respects_existing_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")

    configured = _configure_visible_cuda_devices(2)

    assert configured == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5"


def test_configure_visible_cuda_devices_rejects_too_many_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(ValueError, match="only 1 visible GPU"):
        _configure_visible_cuda_devices(2)


def test_cli_rejects_conflicting_mode_inputs(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "web",
                "--mode",
                "terminal",
                "--workspace-root",
                str(tmp_path),
            ]
        )


def test_runtime_config_uses_packaged_static_assets(tmp_path: Path) -> None:
    args = parse_args(["terminal", "--workspace-root", str(tmp_path)])
    config = build_runtime_config(args)

    assert config.static_dir.name == "static"
    assert (config.static_dir / "index.html").exists()
    assert (config.static_dir / "anveshak_logo2.png").exists()


def test_python_module_entrypoint_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "anveshak", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "anveshak" in result.stdout.lower()
