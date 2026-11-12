"""Helper models that engines need besides the voice itself.

RVC needs a content encoder (HuBERT / ContentVec) and a pitch extractor
(RMVPE). Automatic emotion detection needs a small text classifier. They are
downloaded once from Hugging Face into ``<data>/support/<id>`` - either from
Settings, or automatically the first time an engine needs them.
"""
from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path, PurePosixPath

from . import hub
from .config import Settings, get_settings

SUPPORT: dict[str, dict] = {
    "hubert_base": {
        "label": "HuBERT / ContentVec content encoder",
        "purpose": "RVC: turns speech into timbre-free content features. Required for every RVC voice.",
        "repo": "lj1995/VoiceConversionWebUI",
        "files": ["hubert_base/config.json", "hubert_base/preprocessor_config.json", "hubert_base/pytorch_model.bin"],
        "strip": "hubert_base/",
        "size_mb": 362, "license": "MIT (RVC project)",
    },
    "rmvpe": {
        "label": "RMVPE pitch extractor",
        "purpose": "RVC: tracks the speaker's pitch so the converted voice keeps the intonation.",
        "repo": "lj1995/VoiceConversionWebUI",
        "files": ["rmvpe.pt"],
        "size_mb": 173, "license": "MIT (RVC project)",
    },
    "emotion": {
        "label": "Emotion classifier (English)",
        "purpose": "Reads each English sentence and picks an emotion when emotion is set to auto.",
        "repo": None,  # comes from Settings.emotion_model
        "patterns": ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                     "special_tokens_map.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"],
        "weights": ["model.safetensors", "pytorch_model.bin"],
        "size_mb": 330, "license": "see the model card",
    },
}

_locks: dict[str, threading.Lock] = {k: threading.Lock() for k in SUPPORT}


def path(support_id: str, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.support_dir / support_id


def _marker(support_id: str, settings: Settings) -> Path:
    return path(support_id, settings) / ".complete"


def installed(support_id: str, settings: Settings | None = None) -> bool:
    return _marker(support_id, settings or get_settings()).is_file()


def repo_of(support_id: str, settings: Settings) -> str:
    return SUPPORT[support_id]["repo"] or settings.emotion_model


def describe(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    return [
        {"id": k, "label": v["label"], "purpose": v["purpose"], "repo": repo_of(k, settings) or "(disabled)",
         "size_mb": v["size_mb"], "license": v["license"], "installed": installed(k, settings)}
        for k, v in SUPPORT.items()
    ]


def _plan(support_id: str, settings: Settings) -> tuple[str, list[str]]:
    spec = SUPPORT[support_id]
    repo = repo_of(support_id, settings)
    if not repo:
        raise hub.HubError("automatic emotion is turned off (VOIX_EMOTION_MODEL is empty)", 409)
    if "files" in spec:
        return repo, list(spec["files"])
    names = {f["path"] for f in hub.list_files(repo, "main", hub.get_token(settings))}
    files = [p for p in spec["patterns"] if p in names]
    weight = next((w for w in spec["weights"] if w in names), None)
    if not weight or "config.json" not in files:
        raise hub.HubError(f"{repo} is not a transformers text-classification model VOIX can load", 422)
    return repo, files + [weight]


def ensure(support_id: str, settings: Settings | None = None, progress=None) -> Path:
    """Return the folder of a support model, downloading it first if needed."""
    settings = settings or get_settings()
    if support_id not in SUPPORT:
        raise KeyError(support_id)
    with _locks[support_id]:
        if installed(support_id, settings):
            return path(support_id, settings)
        work = settings.staging_dir / f"support-{uuid.uuid4().hex}"
        try:
            try:
                repo, files = _plan(support_id, settings)
                hub.fetch_files(repo, "main", files, work, settings=settings, progress=progress)
            except hub.HubError as exc:
                hint = " Or pick the FCPE pitch tracker, which needs no download." if support_id == "rmvpe" else ""
                raise hub.HubError(
                    f"{SUPPORT[support_id]['label']} ({SUPPORT[support_id]['size_mb']} MB) is not installed and could not "
                    f"be downloaded: {exc.message} Install it on the Hugging Face page when online.{hint}", exc.status
                ) from exc
            strip = SUPPORT[support_id].get("strip")
            final = path(support_id, settings)
            shutil.rmtree(final, ignore_errors=True)
            final.mkdir(parents=True)
            for rel in files:
                name = rel[len(strip):] if strip and rel.startswith(strip) else rel
                target = final / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(work / PurePosixPath(rel)), target)
            (final / "SOURCE").write_text(f"https://huggingface.co/{repo}\n", encoding="utf-8")
            _marker(support_id, settings).write_text("ok", encoding="utf-8")
            return final
        finally:
            shutil.rmtree(work, ignore_errors=True)


def remove(support_id: str, settings: Settings | None = None) -> None:
    shutil.rmtree(path(support_id, settings or get_settings()), ignore_errors=True)
