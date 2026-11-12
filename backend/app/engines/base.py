"""Adapter contract for TTS engines.

Every architecture gets its own adapter. Nothing outside ``app.engines`` may
import an engine's libraries directly; the rest of VOIX talks to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ParameterSpec:
    """A generation control the UI may show for this engine, and only for this engine."""

    name: str
    label: str
    type: str  # "float" | "int" | "text"
    default: float | int | str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


@dataclass
class AudioResult:
    samples: "object"  # numpy float32 mono array
    sample_rate: int


class EngineError(RuntimeError):
    """Raised for any load or inference failure. Never swallowed."""


class BaseTTSEngine(ABC):
    #: architecture id as produced by the detector, e.g. "piper"
    architecture: str = ""
    #: True only if the model emits audio incrementally while it is still
    #: synthesizing. Engines that synthesize a whole utterance must leave this
    #: False; VOIX then serves them through sentence-chunked generation and
    #: labels the response as chunked, not streaming.
    supports_true_streaming: bool = False
    #: Free RAM this engine needs to load, in MB. Engines that set it are loaded
    #: alone: everything else is unloaded first, and loading is refused with a
    #: clear error if the machine still does not have this much free.
    min_free_ram_mb: int = 0

    @classmethod
    def parameters_for(cls, details: dict) -> list[ParameterSpec]:
        """Controls for a model of this architecture, known without loading it."""
        return []

    @classmethod
    @abstractmethod
    def is_available(cls) -> tuple[bool, str]:
        """Whether this adapter's runtime dependencies are installed, and why not."""

    @abstractmethod
    def load_model(self, model_dir: Path, device: str = "cpu") -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @abstractmethod
    def generate_speech(self, text: str, language: str, **params) -> AudioResult: ...

    def generate_stream(self, text: str, language: str, **params) -> Iterator[AudioResult]:
        raise EngineError(f"{self.architecture} does not support true streaming")

    @abstractmethod
    def get_model_info(self) -> dict: ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]: ...

    @abstractmethod
    def get_supported_parameters(self) -> list[ParameterSpec]: ...

    @abstractmethod
    def validate_model(self, model_dir: Path) -> list[str]:
        """Return a list of problems; empty means the package is loadable."""
