"""Voice changer: applies an imported RVC model to speech."""
from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import psutil
from sqlalchemy.orm import Session

from .. import runtime_assets
from ..config import get_settings
from ..db import VoiceModel
from ..engines.rvc import converter

MAX_SECONDS = 600
#: free RAM the voice changer wants before loading its networks, in MB
MIN_FREE_RAM_MB = 2500
#: callbacks that free memory held by TTS engines when RAM is short
low_memory_hooks: list = []
_runtime: converter.RVCRuntime | None = None
_voices: OrderedDict[str, converter.RVCVoice] = OrderedDict()
_lock = threading.RLock()


class VCRequestError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


def status() -> dict:
    ok, why = converter.available()
    have = runtime_assets.installed(get_settings(), "rvc")
    reason = why if not ok else "" if have else "Voice changer files are not downloaded yet (about 355 MB, one time)."
    return {"ready": ok and have, "libraries": ok, "runtime_installed": have, "reason": reason,
            "size_mb": runtime_assets.BUNDLES["rvc"]["size_mb"]}


def clean_options(raw: dict | None) -> dict | None:
    """Validate voice-changer settings from a request or a character."""
    if not raw or not raw.get("model_id"):
        return None
    try:
        return {
            "model_id": str(raw["model_id"]),
            "pitch": float(np.clip(float(raw.get("pitch", 0)), -24, 24)),
            "index_rate": float(np.clip(float(raw.get("index_rate", 0.5)), 0, 1)),
            "protect": float(np.clip(float(raw.get("protect", 0.33)), 0, 0.5)),
        }
    except (TypeError, ValueError) as exc:
        raise VCRequestError(f"invalid voice changer settings: {exc}") from exc


def check(session: Session, options: dict) -> VoiceModel:
    model = session.get(VoiceModel, options["model_id"])
    if not model:
        raise VCRequestError("voice changer model does not exist", 404)
    if model.architecture != "rvc":
        raise VCRequestError(f"'{model.name}' is {model.architecture}, not an RVC voice changer model")
    state = status()
    if not state["ready"]:
        raise VCRequestError(state["reason"], 409)
    return model


def _voice(model: VoiceModel) -> converter.RVCVoice:
    global _runtime
    with _lock:
        if (_runtime is None or model.id not in _voices) and psutil.virtual_memory().available // 1024**2 < MIN_FREE_RAM_MB:
            for hook in low_memory_hooks:  # e.g. unload Qwen3-TTS; it reloads on its next use
                hook()
            gc.collect()
        if _runtime is None:
            _runtime = converter.RVCRuntime(runtime_assets.bundle_dir(get_settings(), "rvc"))
        if model.id in _voices:
            _voices.move_to_end(model.id)
            return _voices[model.id]
        voice = converter.RVCVoice(Path(model.storage_path), _runtime)
        _voices[model.id] = voice
        while len(_voices) > 2:
            _voices.popitem(last=False)
        return voice


def release() -> None:
    """Drop every loaded voice-changer network (called before a heavy engine loads)."""
    global _runtime
    with _lock:
        _voices.clear()
        _runtime = None


def unload(model_id: str) -> None:
    with _lock:
        _voices.pop(model_id, None)


def convert(model: VoiceModel, samples: np.ndarray, sample_rate: int, options: dict) -> tuple[np.ndarray, int, list[str]]:
    if samples.size / sample_rate > MAX_SECONDS:
        raise VCRequestError(f"audio is longer than {MAX_SECONDS // 60} minutes")
    voice = _voice(model)
    out, rate = voice.convert(samples, sample_rate, pitch=options["pitch"], index_rate=options["index_rate"],
                              protect=options["protect"])
    warnings = [f"{model.name}: {voice.index_error}"] if voice.index_error and options["index_rate"] > 0 else []
    return out, rate, warnings


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read WAV / MP3 / FLAC / OGG into float32 mono."""
    try:
        import soundfile
    except ImportError as exc:
        raise VCRequestError("audio upload needs the soundfile package") from exc
    try:
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise VCRequestError(f"could not read this audio file ({exc}). Use WAV, MP3, FLAC or OGG.") from exc
    if data.shape[0] < rate // 10:
        raise VCRequestError("audio file is empty or too short")
    return data.mean(axis=1), int(rate)
