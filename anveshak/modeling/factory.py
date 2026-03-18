"""Factory helpers for selecting the right reasoning runner backend."""

from __future__ import annotations

from ..config import RuntimeConfig
from ..model_catalog import get_model_profile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen_runner import QwenRunner


def create_runner(config: RuntimeConfig) -> QwenRunner:
    """Create the best available runner for the configured reasoning model."""

    profile = get_model_profile(config.model_id)
    if profile.get("preferred_runtime_backend") == "kimi_server" and config.kimi_server_url:
        from .kimi_server_runner import KimiServerRunner

        return KimiServerRunner(config)
    from .qwen_runner import QwenRunner

    return QwenRunner(config)
