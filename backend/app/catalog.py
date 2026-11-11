"""Starter voices VOIX can download for you.

Only voices whose files and model cards were checked are listed. Each still
goes through the normal inspection pipeline after download.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import VoiceModel
from .importer.service import ImportRejected, install_package

_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_PAGE = "https://huggingface.co/rhasspy/piper-voices/tree/main"
_UNKNOWN = ("unknown", "unknown")

CATALOG: list[dict] = [
    {
        "id": "ml_IN-meera-medium", "name": "Meera", "language": "ml", "gender": "Female", "size_mb": 60,
        "path": "ml/ml_IN/meera/medium", "license": _UNKNOWN,
        "license_note": "Trained on the Indic TTS Malayalam corpus. Its model card only points to the dataset for licence terms, so commercial use is unconfirmed.",
    },
    {
        "id": "ml_IN-arjun-medium", "name": "Arjun", "language": "ml", "gender": "Male", "size_mb": 60,
        "path": "ml/ml_IN/arjun/medium", "license": _UNKNOWN,
        "license_note": "Trained on the Indic TTS Malayalam corpus. Its model card only points to the dataset for licence terms, so commercial use is unconfirmed.",
    },
    {
        "id": "en_US-ljspeech-medium", "name": "LJ", "language": "en", "gender": "Female", "size_mb": 61,
        "path": "en/en_US/ljspeech/medium", "license": ("Public domain dataset (LJ Speech)", "permissive"),
        "license_note": "Model card lists the training data as public domain. American accent.",
    },
    {
        "id": "en_GB-alba-medium", "name": "Alba", "language": "en", "gender": "Female", "size_mb": 60,
        "path": "en/en_GB/alba/medium", "license": ("CC BY 4.0 dataset", "permissive"),
        "license_note": "Model card lists the training data as CC BY 4.0 (attribution required). Scottish accent.",
    },
]


def describe(session: Session) -> list[dict]:
    installed = set(session.scalars(select(VoiceModel.name)).all())
    return [
        {k: v for k, v in entry.items() if k not in ("path", "license")}
        | {"license": entry["license"][0], "license_status": entry["license"][1],
           "source_url": f"{_PAGE}/{entry['path']}", "installed": entry["id"] in installed}
        for entry in CATALOG
    ]


def _download(url: str, dest, limit: int, digest=None) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "VOIX"})
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:  # noqa: S310 - fixed https host
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ImportRejected("download exceeds the size limit", status=413)
            if digest:
                digest.update(chunk)
            out.write(chunk)
    return size


def install(session: Session, settings: Settings, entry_id: str) -> VoiceModel:
    entry = next((e for e in CATALOG if e["id"] == entry_id), None)
    if entry is None:
        raise ImportRejected("unknown catalog voice", status=404)
    if session.scalar(select(VoiceModel).where(VoiceModel.name == entry_id)):
        raise ImportRejected(f"'{entry_id}' is already installed", status=409)

    work = settings.staging_dir / uuid.uuid4().hex
    folder = work / "files"
    folder.mkdir(parents=True)
    digest = hashlib.sha256()
    try:
        files = []
        for name in (f"{entry_id}.onnx", f"{entry_id}.onnx.json", "MODEL_CARD"):
            try:
                size = _download(f"{_BASE}/{entry['path']}/{name}", folder / name, settings.max_upload_bytes,
                                 digest if name.endswith(".onnx") else None)
            except ImportRejected:
                raise
            except Exception as exc:
                raise ImportRejected(f"could not download {name}: {exc}. Check the internet connection.", status=502) from exc
            files.append({"path": name, "size": size})
        return install_package(
            session, settings, folder, files, [], name=entry_id, filename=f"catalog:{entry_id}",
            sha256=digest.hexdigest(), source_url=f"{_PAGE}/{entry['path']}",
            license_override=entry["license"],
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
