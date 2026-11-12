"""Runtime settings, read from VOIX_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GB = 1024**3


def missing_feature_hint(pip_hint: str) -> str:
    """How to get an optional engine: a pip command from source, a different download for the exe."""
    import sys

    if getattr(sys, "frozen", False):
        return "Not included in this download. Get VOIX-full from the GitHub releases page."
    return pip_hint
BACKEND_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    api_key: str | None
    max_upload_bytes: int
    max_extracted_bytes: int
    max_zip_entries: int
    max_compression_ratio: int
    cors_origins: tuple[str, ...]
    cache_enabled: bool = True
    max_loaded_models: int = 2
    max_text_chars: int = 20000

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.staging_dir, self.audio_dir, self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    env = os.environ.get
    data_dir = Path(env("VOIX_DATA_DIR", str(BACKEND_ROOT / "data"))).resolve()
    return Settings(
        data_dir=data_dir,
        database_url=env("VOIX_DATABASE_URL", f"sqlite:///{data_dir / 'voix.db'}"),
        api_key=env("VOIX_API_KEY") or None,
        max_upload_bytes=int(env("VOIX_MAX_UPLOAD_BYTES", 2 * GB)),
        max_extracted_bytes=int(env("VOIX_MAX_EXTRACTED_BYTES", 4 * GB)),
        max_zip_entries=int(env("VOIX_MAX_ZIP_ENTRIES", 2000)),
        max_compression_ratio=int(env("VOIX_MAX_COMPRESSION_RATIO", 200)),
        cors_origins=tuple(
            o.strip()
            for o in env("VOIX_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        ),
        cache_enabled=env("VOIX_CACHE", "1") not in ("0", "false", "off"),
        max_loaded_models=int(env("VOIX_MAX_LOADED_MODELS", 2)),
    )
