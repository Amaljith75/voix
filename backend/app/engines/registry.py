"""Plugin registry for engine adapters.

Add an engine by writing one module with a ``BaseTTSEngine`` subclass decorated
with ``@register``, and importing it in ``app/engines/__init__.py``. Nothing
else in the application changes.

Built in: Piper. Every other architecture is reported as "no inference
adapter" instead of pretending it can speak.
"""
from __future__ import annotations

from .base import BaseTTSEngine

_ADAPTERS: dict[str, type[BaseTTSEngine]] = {}

#: What the detector can identify, and where each stands on the roadmap.
KNOWN_ARCHITECTURES: dict[str, dict] = {
    "piper": {"label": "Piper", "kind": "tts", "planned": "built in"},
    "vits": {"label": "VITS", "kind": "tts", "planned": "Phase 3"},
    "gpt_sovits": {"label": "GPT-SoVITS", "kind": "tts", "planned": "Phase 3"},
    "xtts": {"label": "Coqui XTTS", "kind": "tts", "planned": "Phase 3 (non-commercial license)"},
    "qwen3_tts": {"label": "Qwen3-TTS", "kind": "tts", "planned": "built in (optional, heavy)"},
    "rvc": {"label": "RVC", "kind": "voice_conversion", "planned": "built in as a voice changer"},
    "so_vits_svc": {"label": "so-vits-svc", "kind": "voice_conversion", "planned": "not planned"},
}


def register(cls: type[BaseTTSEngine]) -> type[BaseTTSEngine]:
    if not cls.architecture:
        raise ValueError("adapter must set `architecture`")
    _ADAPTERS[cls.architecture] = cls
    return cls


def get_adapter(architecture: str) -> type[BaseTTSEngine] | None:
    return _ADAPTERS.get(architecture)


def inference_status(architecture: str) -> dict:
    """Honest answer to: can VOIX run this architecture right now?"""
    if architecture == "rvc":
        from ..vc.service import status

        state = status()
        return {"ready": state["ready"], "reason": state["reason"]}
    adapter = _ADAPTERS.get(architecture)
    if adapter is None:
        planned = KNOWN_ARCHITECTURES.get(architecture, {}).get("planned")
        return {
            "ready": False,
            "reason": "No inference adapter is installed for this architecture."
            + (f" Planned: {planned}." if planned else ""),
        }
    ok, why = adapter.is_available()
    return {"ready": ok, "reason": "" if ok else why}


def describe_engines() -> list[dict]:
    out = []
    for arch, meta in KNOWN_ARCHITECTURES.items():
        adapter = _ADAPTERS.get(arch)
        out.append(
            {
                "architecture": arch,
                **meta,
                "adapter_installed": adapter is not None or arch == "rvc",
                "true_streaming": bool(adapter and adapter.supports_true_streaming),
                **inference_status(arch),
            }
        )
    return out
