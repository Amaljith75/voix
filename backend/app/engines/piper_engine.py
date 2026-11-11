"""Piper adapter: VITS voices exported to ONNX, run on CPU with ONNX Runtime.

Piper synthesizes one sentence at a time and returns it whole. That is not
streaming, so ``supports_true_streaming`` stays False and VOIX serves it through
sentence-chunked generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import AudioResult, BaseTTSEngine, EngineError, ParameterSpec
from .registry import register


def find_piper_files(model_dir: Path) -> tuple[Path, Path] | None:
    for onnx in sorted(model_dir.rglob("*.onnx")):
        config = onnx.with_name(onnx.name + ".json")
        if config.is_file():
            return onnx, config
    return None


@register
class PiperEngine(BaseTTSEngine):
    architecture = "piper"
    supports_true_streaming = False

    def __init__(self) -> None:
        self._voice = None
        self._config: dict = {}
        self._path: Path | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
            import piper  # noqa: F401
        except ImportError as exc:
            return False, f"Piper runtime is not installed ({exc.name}). Run: pip install piper-tts"
        return True, ""

    def validate_model(self, model_dir: Path) -> list[str]:
        found = find_piper_files(model_dir)
        if not found:
            return ["no .onnx file with a matching .onnx.json config"]
        try:
            config = json.loads(found[1].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [f"config is not readable JSON: {exc}"]
        return [f"config is missing '{key}'" for key in ("audio", "phoneme_id_map") if key not in config]

    def load_model(self, model_dir: Path, device: str = "cpu") -> None:
        problems = self.validate_model(model_dir)
        if problems:
            raise EngineError("; ".join(problems))
        onnx, config = find_piper_files(model_dir)  # type: ignore[misc]
        from piper import PiperVoice

        try:
            self._voice = PiperVoice.load(str(onnx), config_path=str(config), use_cuda=device == "cuda")
        except Exception as exc:
            raise EngineError(f"Piper could not load {onnx.name}: {exc}") from exc
        self._config = json.loads(config.read_text(encoding="utf-8"))
        self._path = onnx

    def unload_model(self) -> None:
        self._voice = None

    def generate_speech(self, text: str, language: str, **params) -> AudioResult:
        if self._voice is None:
            raise EngineError("model is not loaded")
        from piper import SynthesisConfig

        speed = float(params.get("speed", 1.0))
        base = self._voice.config
        config = SynthesisConfig(
            length_scale=float(base.length_scale) / max(speed, 0.1),
            noise_scale=float(params.get("variability", base.noise_scale)),
            noise_w_scale=float(base.noise_w_scale),
            speaker_id=int(params["speaker"]) if "speaker" in params and self.num_speakers > 1 else None,
            normalize_audio=False,  # VOIX applies one gain to the whole response instead
        )
        try:
            parts = [chunk.audio_float_array for chunk in self._voice.synthesize(text, syn_config=config)]
        except Exception as exc:
            raise EngineError(f"Piper synthesis failed: {exc}") from exc
        if not parts:
            raise EngineError("Piper returned no audio for this text")
        return AudioResult(np.concatenate(parts).astype(np.float32), int(base.sample_rate))

    @property
    def num_speakers(self) -> int:
        return int(self._config.get("num_speakers") or 1)

    def get_model_info(self) -> dict:
        return {
            "file": self._path.name if self._path else None,
            "sample_rate": (self._config.get("audio") or {}).get("sample_rate"),
            "quality": (self._config.get("audio") or {}).get("quality"),
            "num_speakers": self.num_speakers,
            "streaming": "chunked (sentence by sentence)",
        }

    def get_supported_languages(self) -> list[str]:
        code = (self._config.get("language") or {}).get("code") or (self._config.get("espeak") or {}).get("voice") or ""
        return [code.replace("-", "_").split("_")[0].lower()] if code else []

    def get_supported_parameters(self) -> list[ParameterSpec]:
        return list(self.parameters_for(self.num_speakers))

    @staticmethod
    def parameters_for(num_speakers: int = 1) -> list[ParameterSpec]:
        specs = [
            ParameterSpec("speed", "Speed", "float", 1.0, 0.5, 2.0),
            ParameterSpec("variability", "Variability", "float", 0.667, 0.2, 1.0),
        ]
        if num_speakers > 1:
            specs.append(ParameterSpec("speaker", "Speaker", "int", 0, 0, num_speakers - 1))
        return specs
