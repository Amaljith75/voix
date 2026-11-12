"""Qwen3-TTS adapter (Alibaba Qwen, Apache-2.0), through the ``qwen-tts`` package.

Three model types share one adapter; ``config.json`` says which one a folder is:

* ``custom_voice`` - built-in speakers (Ryan, Aiden, Vivian ...). The 1.7B model
  also follows an instruction ("speak cheerfully"); the 0.6B model ignores it.
* ``voice_design`` - the voice itself is described in words, plus style.
* ``base`` - clones a voice from a short reference clip and its transcript.

Qwen3-TTS speaks Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish and Italian. **Not Malayalam** - for Malayalam a character
keeps a Piper voice, and RVC can give both languages the same timbre.

Weights load from the local folder only (no network, no remote code). The
modelling code comes from the installed ``qwen-tts`` package.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .base import AudioResult, BaseTTSEngine, EngineError, ParameterSpec
from .registry import register
from .torch_util import ai_runtime_hint, missing

#: VOIX language code <-> Qwen language name
LANGUAGES = {
    "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french",
    "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian",
}
_NAME_TO_CODE = {v: k for k, v in LANGUAGES.items()}


def find_root(model_dir: Path) -> Path | None:
    for cfg in sorted(model_dir.rglob("config.json"), key=lambda p: len(p.parts)):
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("model_type") == "qwen3_tts":
            return cfg.parent
    return None


def read_config(model_dir: Path) -> dict:
    root = find_root(model_dir)
    return json.loads((root / "config.json").read_text(encoding="utf-8")) if root else {}


def describe_config(cfg: dict) -> dict:
    talker = cfg.get("talker_config") or {}
    langs = sorted({_NAME_TO_CODE[n.lower()] for n in (talker.get("codec_language_id") or {}) if n.lower() in _NAME_TO_CODE})
    model_type = cfg.get("tts_model_type") or "unknown"
    size = str(cfg.get("tts_model_size") or "")
    return {
        "model_type": model_type,
        "size": {"0b6": "0.6B", "1b7": "1.7B"}.get(size, size or "unknown"),
        "speakers": sorted((talker.get("spk_id") or {}).keys()),
        "languages": langs or sorted(LANGUAGES),
        # qwen-tts drops `instruct` for the 0.6B CustomVoice model; Base has no instruction input
        "follows_instructions": model_type == "voice_design" or (model_type == "custom_voice" and size != "0b6"),
    }


@register
class Qwen3TTSEngine(BaseTTSEngine):
    architecture = "qwen3_tts"
    supports_true_streaming = False

    def __init__(self) -> None:
        self._model = None
        self._info: dict = {}
        self._device = "cpu"
        self._prompts: dict[tuple, object] = {}
        self._lock = threading.Lock()

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        gone = missing("torch", "transformers", "qwen_tts", "librosa", "soundfile")
        return (False, ai_runtime_hint(gone)) if gone else (True, "")

    @property
    def follows_instructions(self) -> bool:
        return bool(self._info.get("follows_instructions"))

    def validate_model(self, model_dir: Path) -> list[str]:
        root = find_root(model_dir)
        if root is None:
            return ["no config.json with model_type qwen3_tts"]
        problems = []
        if not (root / "speech_tokenizer" / "config.json").is_file():
            problems.append("speech_tokenizer/ folder (the 12 Hz audio tokenizer) is missing")
        if not any(root.glob("*.safetensors")):
            problems.append("model weights (*.safetensors) are missing")
        if not ((root / "vocab.json").is_file() or (root / "tokenizer.json").is_file()):
            problems.append("text tokenizer files (vocab.json / merges.txt) are missing")
        return problems

    def load_model(self, model_dir: Path, device: str = "cpu") -> None:
        problems = self.validate_model(model_dir)
        if problems:
            raise EngineError("; ".join(problems))
        import torch
        from qwen_tts import Qwen3TTSModel

        from .torch_util import device as pick_device

        root = find_root(model_dir)
        self._device = pick_device()
        dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        try:
            self._model = Qwen3TTSModel.from_pretrained(str(root), device_map=self._device, dtype=dtype)
        except Exception as exc:
            raise EngineError(f"Qwen3-TTS could not load: {exc}") from exc
        self._info = describe_config(json.loads((root / "config.json").read_text(encoding="utf-8")))

    def unload_model(self) -> None:
        self._model = None
        self._prompts.clear()

    def _clone_prompt(self, ref_audio: str, ref_text: str):
        key = (ref_audio, ref_text)
        with self._lock:
            if key not in self._prompts:
                import soundfile as sf

                wav, sr = sf.read(ref_audio, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                self._prompts[key] = self._model.create_voice_clone_prompt(
                    ref_audio=(wav, sr), ref_text=ref_text or None, x_vector_only_mode=not ref_text,
                )
            return self._prompts[key]

    def generate_speech(self, text: str, language: str, **params) -> AudioResult:
        if self._model is None:
            raise EngineError("model is not loaded")
        name = LANGUAGES.get(language)
        if name is None:
            raise EngineError(f"Qwen3-TTS cannot speak '{language}'. Supported: {', '.join(sorted(LANGUAGES))}.")
        lang = name.capitalize()
        instruct = str(params.get("instruct") or "").strip()
        gen = {k: params[k] for k in ("temperature", "top_p") if k in params}
        kind = self._info.get("model_type")
        try:
            if kind == "custom_voice":
                speaker = str(params.get("speaker") or (self._info["speakers"] or [""])[0])
                wavs, sr = self._model.generate_custom_voice(
                    text=text, language=lang, speaker=speaker, instruct=instruct or None, **gen)
            elif kind == "voice_design":
                description = str(params.get("voice_description") or "").strip()
                full = " ".join(p for p in (description, instruct) if p) or "A clear, friendly voice."
                wavs, sr = self._model.generate_voice_design(text=text, language=lang, instruct=full, **gen)
            elif kind == "base":
                ref = params.get("ref_audio")
                if not ref or not Path(str(ref)).is_file():
                    raise EngineError("this Qwen3-TTS Base voice has no reference clip yet. Upload one on the voice.")
                prompt = self._clone_prompt(str(ref), str(params.get("ref_text") or ""))
                wavs, sr = self._model.generate_voice_clone(text=text, language=lang, voice_clone_prompt=prompt, **gen)
            else:
                raise EngineError(f"unknown Qwen3-TTS model type '{kind}'")
        except EngineError:
            raise
        except Exception as exc:
            raise EngineError(f"Qwen3-TTS synthesis failed: {exc}") from exc
        samples = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        speed = float(params.get("speed", 1.0))
        if abs(speed - 1.0) > 0.02 and samples.size:
            import librosa

            samples = librosa.effects.time_stretch(samples, rate=speed).astype(np.float32)
        return AudioResult(samples, int(sr))

    def get_model_info(self) -> dict:
        return {**self._info, "device": self._device, "streaming": "chunked (sentence by sentence)"}

    def get_supported_languages(self) -> list[str]:
        return list(self._info.get("languages") or LANGUAGES)

    def get_supported_parameters(self) -> list[ParameterSpec]:
        return self.parameters_for()

    @staticmethod
    def parameters_for(num_speakers: int = 1) -> list[ParameterSpec]:
        return [
            ParameterSpec("speed", "Speed", "float", 1.0, 0.5, 2.0),
            ParameterSpec("temperature", "Expressiveness", "float", 0.9, 0.3, 1.3),
        ]
