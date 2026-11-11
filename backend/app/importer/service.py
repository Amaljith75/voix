"""Model import pipeline: upload -> hash -> safe extract -> inspect -> detect -> install."""
from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import Settings
from ..db import VoiceModel, audit
from ..engines.registry import inference_status
from ..inspection.detector import detect_package
from ..inspection.license import detect_license
from ..security.zip_safety import UnsafeArchive, safe_extract

_CHUNK = 1024 * 1024


class ImportRejected(Exception):
    def __init__(self, message: str, status: int = 422, report: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.report = report or {}


def clean_name(raw: str) -> str:
    stem = Path(raw or "model").name
    stem = re.sub(r"\.zip$", "", stem, flags=re.I)
    stem = re.sub(r"[^\w\s.\-()]+", " ", stem, flags=re.UNICODE).strip()
    return stem[:120] or "model"


def _receive(stream: BinaryIO, dest: Path, limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(dest, "wb") as out:
        while chunk := stream.read(_CHUNK):
            size += len(chunk)
            if size > limit:
                raise ImportRejected(f"upload exceeds the {limit // 1024**2} MB limit", status=413)
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest(), size


def import_zip(
    session: Session, settings: Settings, stream: BinaryIO, filename: str, name: str | None = None
) -> VoiceModel:
    job = uuid.uuid4().hex
    work = settings.staging_dir / job
    work.mkdir(parents=True)
    zip_path = work / "upload.zip"
    extracted = work / "files"
    try:
        sha256, _ = _receive(stream, zip_path, settings.max_upload_bytes)

        dup = session.scalar(select(VoiceModel).where(VoiceModel.sha256 == sha256))
        if dup:
            raise ImportRejected(
                f"this exact archive is already installed as '{dup.name}' v{dup.version}", status=409
            )

        try:
            result = safe_extract(zip_path, extracted, settings)
        except UnsafeArchive as exc:
            audit(session, "model.import.rejected", filename=filename, sha256=sha256, reason=str(exc))
            session.commit()
            raise ImportRejected(f"archive rejected: {exc}") from exc

        detection, reports = detect_package(extracted, result.files)
        unsafe = [r for r in reports if r.verdict == "unsafe"]
        if unsafe:
            report = {"checkpoints": [r.summary() for r in unsafe]}
            audit(
                session, "model.import.rejected", filename=filename, sha256=sha256,
                reason="unsafe pickle globals",
                globals=[g for r in unsafe for g in r.unsafe_globals],
            )
            session.commit()
            raise ImportRejected(
                "checkpoint would execute code when loaded ("
                + ", ".join(g for r in unsafe for g in r.unsafe_globals)
                + "). Nothing was installed.",
                report=report,
            )

        license_name, license_status, license_file = detect_license(extracted, result.files)
        model_name = clean_name(name or filename)
        version = (
            session.scalar(select(func.max(VoiceModel.version)).where(VoiceModel.name == model_name)) or 0
        ) + 1

        model_id = uuid.uuid4().hex
        final = settings.models_dir / model_id
        shutil.move(str(extracted), final)

        warnings = list(detection.warnings)
        if license_status == "unknown":
            warnings.append(
                "License unknown. Being downloadable does not grant commercial rights; "
                "confirm the license and record the source before using this in SCOL."
            )
        elif license_status == "noncommercial":
            warnings.append(f"{license_name}: not usable in a commercial product.")

        model = VoiceModel(
            id=model_id,
            name=model_name,
            version=version,
            architecture=detection.architecture,
            kind=detection.kind,
            completeness=detection.completeness,
            safety="caution" if any(r.verdict == "caution" for r in reports) else "safe",
            size_bytes=result.total_bytes,
            sha256=sha256,
            languages=detection.languages,
            dependencies=detection.dependencies,
            required_config=detection.required_config,
            missing=detection.missing,
            device_support=detection.device_support,
            license=license_name,
            license_status=license_status,
            files=result.files,
            report={
                "detection": detection.to_dict(),
                "checkpoints": [r.summary() for r in reports],
                "skipped_files": result.skipped,
                "license_source": license_file,
                "warnings": warnings,
                "original_filename": filename,
            },
            storage_path=str(final),
        )
        session.add(model)
        audit(
            session, "model.import", model_id=model_id, name=model_name, version=version,
            architecture=detection.architecture, sha256=sha256,
        )
        session.commit()
        return model
    finally:
        shutil.rmtree(work, ignore_errors=True)


def serialize(model: VoiceModel, full: bool = False) -> dict:
    data = {
        "id": model.id,
        "name": model.name,
        "version": model.version,
        "architecture": model.architecture,
        "architecture_label": model.report.get("detection", {}).get("architecture_label", model.architecture),
        "kind": model.kind,
        "standalone_tts": model.report.get("detection", {}).get("standalone_tts", False),
        "completeness": model.completeness,
        "safety": model.safety,
        "active": model.active,
        "size_bytes": model.size_bytes,
        "languages": model.languages,
        "language_note": model.report.get("detection", {}).get("language_note", ""),
        "dependencies": model.dependencies,
        "required_config": model.required_config,
        "missing": model.missing,
        "device_support": model.device_support,
        "license": model.license,
        "license_status": model.license_status,
        "source_url": model.source_url,
        "tags": model.tags,
        "inference": inference_status(model.architecture),
        "warnings": model.report.get("warnings", []),
        "created_at": model.created_at.isoformat(),
    }
    if full:
        data["files"] = model.files
        data["sha256"] = model.sha256
        data["report"] = model.report
    return data
