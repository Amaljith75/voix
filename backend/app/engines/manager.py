"""Loads engines on demand and keeps only a few in memory."""
from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from pathlib import Path

import psutil

from .base import BaseTTSEngine, EngineError
from .registry import get_adapter


class EngineManager:
    def __init__(self, max_loaded: int = 2, device: str = "cpu") -> None:
        self.max_loaded = max(1, max_loaded)
        self.device = device
        self._loaded: OrderedDict[str, BaseTTSEngine] = OrderedDict()
        self._lock = threading.RLock()
        self.load_seconds: dict[str, float] = {}
        #: callbacks that free memory held outside this manager (the voice changer)
        self.release_hooks: list = []

    def _make_room(self, adapter: type[BaseTTSEngine]) -> None:
        """Heavy engines load alone, and only if the machine has the memory."""
        for model_id in list(self._loaded):
            self._loaded.pop(model_id).unload_model()
        for hook in self.release_hooks:
            hook()
        gc.collect()
        free = psutil.virtual_memory().available // 1024**2
        if free < adapter.min_free_ram_mb:
            raise EngineError(
                f"not enough free memory to load {adapter.architecture}: it needs about "
                f"{adapter.min_free_ram_mb} MB free and this machine has {free} MB. Close other programs and retry."
            )

    def get(self, model_id: str, architecture: str, model_dir: Path) -> BaseTTSEngine:
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
            if adapter.min_free_ram_mb:
                self._make_room(adapter)
            engine = adapter()
            started = time.perf_counter()
            engine.load_model(model_dir, self.device)
            self.load_seconds[model_id] = round(time.perf_counter() - started, 3)
            self._loaded[model_id] = engine
            while len(self._loaded) > self.max_loaded:
                _, evicted = self._loaded.popitem(last=False)
                evicted.unload_model()
            return engine

    def unload_all(self) -> None:
        with self._lock:
            for model_id in list(self._loaded):
                self._loaded.pop(model_id).unload_model()
        gc.collect()

    def unload(self, model_id: str) -> None:
        with self._lock:
            engine = self._loaded.pop(model_id, None)
            if engine:
                engine.unload_model()

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._loaded)
