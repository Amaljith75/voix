"""Loads engines on demand and keeps only a few in memory."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

from .base import BaseTTSEngine, BaseVoiceConverter, EngineError
from .registry import get_adapter


class EngineManager:
    def __init__(self, max_loaded: int = 2, device: str = "cpu") -> None:
        self.max_loaded = max(1, max_loaded)
        self.device = device
        self._loaded: OrderedDict[str, BaseTTSEngine | BaseVoiceConverter] = OrderedDict()
        self._lock = threading.RLock()
        self.load_seconds: dict[str, float] = {}

    def get(self, model_id: str, architecture: str, model_dir: Path) -> BaseTTSEngine | BaseVoiceConverter:
        with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            adapter = get_adapter(architecture)
            if adapter is None:
                raise EngineError(f"no inference adapter for architecture '{architecture}'")
            ok, why = adapter.is_available()
            if not ok:
                raise EngineError(why)
            engine = adapter()
            started = time.perf_counter()
            engine.load_model(model_dir, self.device)
            self.load_seconds[model_id] = round(time.perf_counter() - started, 3)
            self._loaded[model_id] = engine
            while len(self._loaded) > self.max_loaded:
                _, evicted = self._loaded.popitem(last=False)
                evicted.unload_model()
            return engine

    def unload(self, model_id: str) -> None:
        with self._lock:
            engine = self._loaded.pop(model_id, None)
            if engine:
                engine.unload_model()

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._loaded)
