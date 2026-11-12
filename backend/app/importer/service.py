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
from ..db import Voice, VoiceModel, audit
from ..engines.registry import get_adapter, inference_status
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

        return install_package(
            session, settings, extracted, result.files, result.skipped,
            name=clean_name(name or filename), filename=filename, sha256=sha256,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install_package(
    session: Session, settings: Settings, folder: Path, files: list[dict], skipped: list[dict], *,
    name: str, filename: str, sha256: str, source_url: str = "",
    license_override: tuple[str, str] | None = None,
) -> VoiceModel:
    """Inspect an extracted folder and install it. Shared by ZIP import and the voice catalog."""
    detection, reports = detect_package(folder, files)
    unsafe = [r for r in reports if r.verdict == "unsafe"]
    if unsafe:
        bad = [g for r in unsafe for g in r.unsafe_globals]
        audit(session, "model.import.rejected", filename=filename, sha256=sha256,
              reason="unsafe pickle globals", globals=bad)
        session.commit()
        raise ImportRejected(
            "checkpoint would execute code when loaded (" + ", ".join(bad) + "). Nothing was installed.",
            report={"checkpoints": [r.summary() for r in unsafe]},
        )

    detected = detect_license(folder, files)
    license_name, license_status = license_override or detected[:2]
    license_file = ("model card" if source_url.startswith("https://huggingface.co") else "voice catalog") \
        if license_override else detected[2]
    version = (session.scalar(select(func.max(VoiceModel.version)).where(VoiceModel.name == name)) or 0) + 1

    model_id = uuid.uuid4().hex
    final = settings.models_dir / model_id
    shutil.move(str(folder), final)

    model = VoiceModel(
        id=model_id, name=name, version=version,
        architecture=detection.architecture, kind=detection.kind, completeness=detection.completeness,
        safety="caution" if any(r.verdict == "caution" for r in reports) else "safe",
        size_bytes=sum(f["size"] for f in files), sha256=sha256,
        languages=detection.languages, dependencies=detection.dependencies,
        required_config=detection.required_config, missing=detection.missing,
        device_support=detection.device_support,
        license=license_name, license_status=license_status, source_url=source_url,
        files=files,
        report={
            "detection": detection.to_dict(),
            "checkpoints": [r.summary() for r in reports],
            "skipped_files": skipped,
            "license_source": license_file,
            "warnings": list(detection.warnings),
            "original_filename": filename,
        },
        storage_path=str(final),
    )
    session.add(model)
    audit(session, "model.import", model_id=model_id, name=name, version=version,
          architecture=detection.architecture, sha256=sha256)
    register_voice(session, model)
    session.commit()
    return model


def _new_voice_id(session: Session, name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:60] or "voice"
    voice_id, n = base, 2
    while session.get(Voice, voice_id) or voice_id in {v.id for v in session.new if isinstance(v, Voice)}:
        voice_id, n = f"{base}_{n}", n + 1
    return voice_id


def register_voice(session: Session, model: VoiceModel) -> Voice | None:
    """A complete TTS model with a working adapter becomes one or more selectable voices.

    Qwen3-TTS CustomVoice: one voice per built-in speaker. VoiceDesign: one voice
    with a default description you can edit. Base (cloning): none until a
    reference clip is uploaded (POST /voices).
    """
    if model.kind != "tts" or model.completeness != "complete" or get_adapter(model.architecture) is None:
        return None
    details = (model.report.get("detection") or {}).get("details") or {}
    label = model.report["detection"].get("architecture_label", model.architecture)
    specs: list[tuple[str, str, dict]] = []
    if model.architecture == "qwen3_tts":
        qtype = details.get("qwen_model_type")
        if qtype == "custom_voice":
            specs = [(f"{model.name} {spk.replace('_', ' ').title()}", f"{label} {details.get('size', '')} · built-in speaker",
                      {"speaker": spk}) for spk in details.get("speakers") or []]
        elif qtype == "voice_design":
            specs = [(f"{model.name} designed", f"{label} {details.get('size', '')} · voice described in words",
                      {"voice_description": "A warm, clear, friendly young female teacher speaking to a child."})]
    else:
        specs = [(model.name, label + (f" · {details['quality']} quality" if details.get("quality") else ""), {})]
    first = None
    for name, description, voice_settings in specs:
        voice = Voice(id=_new_voice_id(session, name), name=name, model_id=model.id, languages=model.languages,
                      description=description, settings=voice_settings)
        session.add(voice)
        audit(session, "voice.create", voice_id=voice.id, model_id=model.id)
        first = first or voice
    model.active = True
    return first


_PERMISSIVE = {"apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0", "cc-by-3.0",
               "cc-by-sa-4.0", "unlicense", "isc", "openrail", "creativeml-openrail-m", "openrail++"}


def hub_license(name: str) -> tuple[str, str] | None:
    """Licence declared on a Hugging Face model card, mapped to VOIX's status."""
    key = (name or "").lower()
    if not key or key in ("unknown", "other"):
        return None
    if "nc" in key.split("-") or "noncommercial" in key or "non-commercial" in key:
        return name, "noncommercial"
    if key in _PERMISSIVE:
        return name, "permissive" if "openrail" not in key else "conditional"
    return name, "conditional"


def install_from_hub(session: Session, settings: Settings, *, repo: str, revision: str, paths: list[str],
                     name: str | None, license_name: str = "", progress=None) -> VoiceModel:
    """Download chosen files from a Hugging Face repository and install them like a ZIP import."""
    from .. import hub

    work = settings.staging_dir / f"hub-{uuid.uuid4().hex}"
    folder = work / "files"
    try:
        files, sha256 = hub.fetch_files(repo, revision, paths, folder, settings=settings, progress=progress)
        dup = session.scalar(select(VoiceModel).where(VoiceModel.sha256 == sha256))
        if dup:
            raise ImportRejected(f"these exact files are already installed as '{dup.name}' v{dup.version}", status=409)
        skipped: list[dict] = []
        for zip_file in [f for f in files if f["path"].lower().endswith(".zip")]:
            zpath = folder / zip_file["path"]
            target = zpath.with_suffix("")
            try:
                result = safe_extract(zpath, target, settings)
            except UnsafeArchive as exc:
                raise ImportRejected(f"{zip_file['path']} rejected: {exc}") from exc
            skipped += result.skipped
            zpath.unlink()
        files = [{"path": str(p.relative_to(folder)).replace("\\", "/"), "size": p.stat().st_size}
                 for p in sorted(folder.rglob("*")) if p.is_file()]
        default_name = paths[0].rsplit("/", 1)[-1].rsplit(".", 1)[0] if len(paths) == 1 else repo.split("/")[-1]
        return install_package(
            session, settings, folder, files, skipped, name=clean_name(name or default_name),
            filename=f"hf:{repo}@{revision}", sha256=sha256,
            source_url=f"https://huggingface.co/{repo}/tree/{revision}", license_override=hub_license(license_name),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def license_warnings(model: VoiceModel) -> list[str]:
    if model.license_status == "unknown":
        return ["License unknown. Being downloadable does not grant commercial rights; "
                "confirm the license and record the source before using this in SCOL."]
    if model.license_status == "noncommercial":
        return [f"{model.license}: not usable in a commercial product."]
    return []


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
        # licence warnings follow the current licence status; older rows stored them inline
        "warnings": [w for w in model.report.get("warnings", []) if not w.startswith(("License unknown", model.license + ":"))]
        + license_warnings(model),
        "created_at": model.created_at.isoformat(),
    }
    if full:
        data["files"] = model.files
        data["sha256"] = model.sha256
        data["report"] = model.report
    return data
