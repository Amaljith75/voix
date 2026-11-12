"""Qwen3-TTS adapter (Alibaba, Apache-2.0): an LLM-style TTS with preset
speakers and natural-language style control ("emotion").

Heavy: on a 2-core CPU it produces about one second of audio every five
seconds, and it does not speak Malayalam. It generates whole utterances, so
it is served as chunked generation like every other engine here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import AudioResult, BaseTTSEngine, EngineError, ParameterSpec
from .registry import register

LANGUAGES = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German",
             "fr": "French", "ru": "Russian", "pt": "Portuguese", "es": "Spanish", "it": "Italian"}


def style_instruction(emotion: str) -> str:
    """Turn a short cue ('laughing', 'sad') into the instruction sentence the model expects."""
    cue = " ".join(emotion.strip().split())
    if not cue:
        return ""
    if len(cue.split()) > 3 or cue.endswith((".", "!")):
        return cue  # already a full instruction
    if cue.split()[0].lower().endswith("ing"):
        return f"Speak while {cue.lower()}."
    return f"Speak in a {cue.lower()} tone."


@register
class Qwen3TTSEngine(BaseTTSEngine):
    architecture = "qwen3_tts"
    supports_true_streaming = False
    min_free_ram_mb = 4500  # measured: the 0.6B model peaks near 4 GB while loading on CPU

    def __init__(self) -> None:
        self._model = None
        self._speakers: list[str] = []

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import qwen_tts  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            from ..config import missing_feature_hint

            return False, missing_feature_hint(f"Qwen3-TTS runtime is not installed ({exc.name}). Run: pip install -r requirements-qwen.txt")
        return True, ""

    @classmethod
    def parameters_for(cls, details: dict) -> list[ParameterSpec]:
        return [ParameterSpec("emotion", "Emotion / style", "text", "")]

    def validate_model(self, model_dir: Path) -> list[str]:
        return [] if (model_dir / "config.json").is_file() else ["config.json not found; the model download is incomplete"]

    def load_model(self, model_dir: Path, device: str = "cpu") -> None:
        problems = self.validate_model(model_dir)
        if problems:
            raise EngineError("; ".join(problems))
        import torch
        from qwen_tts import Qwen3TTSModel

        try:
            self._model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=device, dtype=torch.float32)
            self._speakers = list(self._model.get_supported_speakers() or [])
        except Exception as exc:
            raise EngineError(f"Qwen3-TTS could not load: {exc}") from exc

    def unload_model(self) -> None:
        self._model = None

    def generate_speech(self, text: str, language: str, **params) -> AudioResult:
        if self._model is None:
            raise EngineError("model is not loaded")
        if language not in LANGUAGES:
            raise EngineError(f"Qwen3-TTS does not speak '{language}'")
        speaker = str(params.get("speaker") or (self._speakers[0] if self._speakers else ""))
        if speaker not in self._speakers:
            raise EngineError(f"unknown Qwen3-TTS speaker '{speaker}'")
        try:
            wavs, rate = self._model.generate_custom_voice(
                text=text, language=LANGUAGES[language], speaker=speaker,
                instruct=style_instruction(str(params.get("emotion") or "")),
            )
        except Exception as exc:
            raise EngineError(f"Qwen3-TTS synthesis failed: {exc}") from exc
        return AudioResult(np.asarray(wavs[0], dtype=np.float32), int(rate))

    def get_model_info(self) -> dict:
        return {"speakers": self._speakers, "streaming": "chunked (sentence by sentence)"}

    def get_supported_languages(self) -> list[str]:
        return list(LANGUAGES)

    def get_supported_parameters(self) -> list[ParameterSpec]:
        return self.parameters_for({})
