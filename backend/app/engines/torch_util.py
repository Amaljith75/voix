"""Shared helpers for PyTorch engines. Importing this module does not import torch."""
from __future__ import annotations

import importlib.util

from ..config import get_settings


def missing(*modules: str) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def ai_runtime_hint(names: list[str]) -> str:
    return (f"AI runtime is not installed (missing: {', '.join(names)}). "
            "Use the VOIX-AI build, or run: pip install -r requirements-ai.txt")


def device() -> str:
    """The device PyTorch engines run on: VOIX_DEVICE, or CUDA when present."""
    wanted = get_settings().device
    import torch

    if wanted == "cpu":
        return "cpu"
    if wanted.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("VOIX_DEVICE asks for CUDA but PyTorch sees no CUDA GPU")
        return wanted
    return "cuda" if torch.cuda.is_available() else "cpu"


def safe_torch_load(path, map_location="cpu"):
    """Load a checkpoint with the restricted unpickler: tensors and plain data only, never code."""
    import torch

    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"{getattr(path, 'name', path)} could not be loaded with the safe (weights-only) loader: {exc}"
        ) from exc
