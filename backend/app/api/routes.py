"""REST API, version 1. Provider-independent: it accepts text, never an LLM."""
from __future__ import annotations

import hmac
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
import uuid
from sqlalchemy.orm import Session

from .. import __version__
from ..config import Settings, get_settings
from .. import catalog, runtime_assets
from ..audio.processing import mp3_available
from ..db import AuditLog, Character, Generation, Voice, VoiceModel, audit, get_session
from ..engines.registry import describe_engines
from ..hardware import detect_hardware
from ..importer.service import ImportRejected, import_zip, serialize
from ..tts import service as tts
from ..vc import service as vc

router = APIRouter(prefix="/api/v1")


def require_key(
    x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)
) -> None:
    """Guards every write. Enforced whenever VOIX_API_KEY is set."""
    if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(401, "missing or invalid X-API-Key")


# ---------------------------------------------------------------- system

@router.get("/health")
def health(session: Session = Depends(get_session)):
    engines = describe_engines()
    return {
        "status": "ok",
        "version": __version__,
        "models": len(session.scalars(select(VoiceModel.id)).all()),
        "tts_ready": any(e["ready"] and e["kind"] == "tts" for e in engines),
    }


@router.get("/system/hardware")
def hardware():
    return detect_hardware()


@router.get("/system/config")
def config(settings: Settings = Depends(get_settings)):
    return {
        "auth_required": settings.api_key is not None,
        "max_upload_mb": settings.max_upload_bytes // 1024**2,
        "max_extracted_mb": settings.max_extracted_bytes // 1024**2,
        "max_zip_entries": settings.max_zip_entries,
        "database": settings.database_url.split(":", 1)[0],
        "cache_enabled": settings.cache_enabled,
        "mp3": mp3_available()[0],
        "loaded_models": tts.manager().loaded(),
    }


@router.get("/system/audit")
def audit_log(limit: int = 50, session: Session = Depends(get_session)):
    rows = session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 200))).all()
    return [{"at": r.at.isoformat(), "action": r.action, "detail": r.detail} for r in rows]


@router.get("/engines")
def engines():
    return describe_engines()


# ---------------------------------------------------------------- models

def _get_model(session: Session, model_id: str) -> VoiceModel:
    model = session.get(VoiceModel, model_id)
    if not model:
        raise HTTPException(404, "model not found")
    return model


@router.get("/models")
def list_models(session: Session = Depends(get_session)):
    rows = session.scalars(select(VoiceModel).order_by(VoiceModel.created_at.desc())).all()
    return [serialize(m) for m in rows]


@router.get("/models/{model_id}")
def get_model(model_id: str, session: Session = Depends(get_session)):
    return serialize(_get_model(session, model_id), full=True)


@router.post("/models/import", status_code=201, dependencies=[Depends(require_key)])
def import_model(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(415, "upload a .zip archive")
    try:
        model = import_zip(session, settings, file.file, file.filename or "model.zip", name)
    except ImportRejected as exc:
        return JSONResponse({"detail": exc.message, "report": exc.report}, status_code=exc.status)
    return serialize(model, full=True)


class ModelUpdate(BaseModel):
    active: bool | None = None
    license: str | None = Field(default=None, max_length=120)
    license_status: str | None = Field(default=None, pattern="^(permissive|conditional|noncommercial|unknown)$")
    source_url: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = None
    languages: list[str] | None = None


@router.patch("/models/{model_id}", dependencies=[Depends(require_key)])
def update_model(model_id: str, body: ModelUpdate, session: Session = Depends(get_session)):
    model = _get_model(session, model_id)
    changes = body.model_dump(exclude_none=True)
    if changes.get("active") and model.completeness != "complete":
        raise HTTPException(
            409, "only complete, identified models can be activated: " + "; ".join(model.missing or ["architecture unknown"])
        )
    if "tags" in changes:
        changes["tags"] = sorted({t.strip()[:40] for t in changes["tags"] if t.strip()})[:20]
    if "languages" in changes:
        changes["languages"] = sorted({c.strip().lower()[:8] for c in changes["languages"] if c.strip()})
    for key, value in changes.items():
        setattr(model, key, value)
    audit(session, "model.update", model_id=model_id, changes=changes)
    session.commit()
    return serialize(model, full=True)


@router.delete("/models/{model_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_model(
    model_id: str, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)
):
    model = _get_model(session, model_id)
    path = Path(model.storage_path).resolve()
    if path.is_relative_to(settings.models_dir.resolve()) and path != settings.models_dir.resolve():
        shutil.rmtree(path, ignore_errors=True)
    for voice in session.scalars(select(Voice).where(Voice.model_id == model_id)).all():
        for character in session.scalars(select(Character)).all():
            if character.voice_ml == voice.id:
                character.voice_ml = None
            if character.voice_en == voice.id:
                character.voice_en = None
        session.delete(voice)
    for character in session.scalars(select(Character)).all():
        if (character.rvc or {}).get("model_id") == model_id:
            character.rvc = None
    tts.manager().unload(model_id)
    vc.unload(model_id)
    session.delete(model)
    audit(session, "model.delete", model_id=model_id, name=model.name)
    session.commit()


# ---------------------------------------------------------------- voices

def _voice(session: Session, v: Voice) -> dict:
    model = session.get(VoiceModel, v.model_id)
    status = tts.inference_status(model.architecture) if model else {"ready": False, "reason": "model was deleted"}
    return {
        "id": v.id, "name": v.name, "description": v.description, "model_id": v.model_id,
        "languages": v.languages, "architecture": model.architecture if model else None,
        "license": model.license if model else "unknown",
        "license_status": model.license_status if model else "unknown",
        "ready": status["ready"], "reason": status["reason"],
        "parameters": tts.voice_parameters(model) if model else [],
        "streaming": "chunked",
    }


@router.get("/voices")
def list_voices(session: Session = Depends(get_session)):
    return [_voice(session, v) for v in session.scalars(select(Voice).order_by(Voice.name)).all()]


@router.get("/voices/{voice_id}")
def get_voice(voice_id: str, session: Session = Depends(get_session)):
    voice = session.get(Voice, voice_id)
    if not voice:
        raise HTTPException(404, "voice not found")
    return _voice(session, voice)


@router.get("/catalog")
def voice_catalog(session: Session = Depends(get_session)):
    return catalog.describe(session)


@router.post("/catalog/{entry_id}/install", status_code=201, dependencies=[Depends(require_key)])
def install_catalog_voice(
    entry_id: str, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)
):
    try:
        model = catalog.install(session, settings, entry_id)
    except ImportRejected as exc:
        return JSONResponse({"detail": exc.message, "report": exc.report}, status_code=exc.status)
    return serialize(model, full=True)


# ---------------------------------------------------------------- characters

class CharacterIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    image_url: str = Field(default="", max_length=500)
    voice_ml: str | None = None
    voice_en: str | None = None
    default_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    presets: dict = Field(default_factory=dict)
    #: optional voice changer: {"model_id": "<RVC model id>", "pitch": 0, "index_rate": 0.5}
    rvc: dict | None = None


def _character(c: Character) -> dict:
    return {
        "id": c.id, "name": c.name, "description": c.description, "image_url": c.image_url,
        "voice_ml": c.voice_ml, "voice_en": c.voice_en, "default_speed": c.default_speed,
        "presets": c.presets, "rvc": c.rvc,
    }


def _check_voices(session: Session, body: CharacterIn) -> None:
    for vid in (body.voice_ml, body.voice_en):
        if vid and not session.get(Voice, vid):
            raise HTTPException(422, f"voice '{vid}' does not exist")
    try:
        body.rvc = vc.clean_options(body.rvc)
    except vc.VCRequestError as exc:
        raise HTTPException(exc.status, exc.message) from exc
    if body.rvc:
        model = session.get(VoiceModel, body.rvc["model_id"])
        if not model or model.architecture != "rvc":
            raise HTTPException(422, "rvc.model_id must be an imported RVC model")


@router.get("/characters")
def list_characters(session: Session = Depends(get_session)):
    return [_character(c) for c in session.scalars(select(Character).order_by(Character.name)).all()]


@router.post("/characters", status_code=201, dependencies=[Depends(require_key)])
def create_character(body: CharacterIn, session: Session = Depends(get_session)):
    cid = re.sub(r"[^a-z0-9]+", "_", body.name.lower()).strip("_")[:60] or "character"
    if session.get(Character, cid):
        raise HTTPException(409, f"character id '{cid}' already exists")
    _check_voices(session, body)
    character = Character(id=cid, **body.model_dump())
    session.add(character)
    audit(session, "character.create", character_id=cid)
    session.commit()
    return _character(character)


@router.put("/characters/{character_id}", dependencies=[Depends(require_key)])
def update_character(character_id: str, body: CharacterIn, session: Session = Depends(get_session)):
    character = session.get(Character, character_id)
    if not character:
        raise HTTPException(404, "character not found")
    _check_voices(session, body)
    for key, value in body.model_dump().items():
        setattr(character, key, value)
    audit(session, "character.update", character_id=character_id)
    session.commit()
    return _character(character)


@router.delete("/characters/{character_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_character(character_id: str, session: Session = Depends(get_session)):
    character = session.get(Character, character_id)
    if not character:
        raise HTTPException(404, "character not found")
    session.delete(character)
    audit(session, "character.delete", character_id=character_id)
    session.commit()


# ---------------------------------------------------------------- tts

class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    voice_id: str | None = None
    character_id: str | None = None
    language: str = Field(default="auto", pattern="^(en|ml|auto)$")
    output_format: str = Field(default="mp3", pattern="^(mp3|wav)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    params: dict[str, float] = Field(default_factory=dict)
    #: "laughing", "sad", "excited" or a full instruction. Only engines with emotion control act on it.
    emotion: str | None = Field(default=None, max_length=200)
    #: voice changer after synthesis: {"model_id": "<RVC model id>", "pitch": 0, "index_rate": 0.5}
    rvc: dict | None = None
    use_cache: bool = True


def _job(job: Generation, cached: bool = False) -> dict:
    data = {
        "job_id": job.id, "status": job.status, "cached": cached, "language": job.language,
        "kind": job.kind or "tts", "rvc": job.rvc,
        "voice_id": job.voice_id, "character_id": job.character_id, "output_format": job.output_format,
        "text": job.text, "params": job.params, "warnings": job.warnings,
        "created_at": job.created_at.isoformat(),
    }
    if job.status == "completed":
        data |= {
            "audio_id": job.id, "audio_url": f"/api/v1/audio/{job.id}",
            "audio_seconds": job.audio_seconds, "generation_seconds": job.generation_seconds,
            "sample_rate": job.sample_rate, "chunks": job.chunks,
            # Piper returns whole sentences, so this is chunked generation, not streaming.
            "mode": "chunked",
        }
    if job.status == "failed":
        data["error"] = job.error
    return data


@router.post("/tts/generate", status_code=202, dependencies=[Depends(require_key)])
def tts_generate(
    body: TTSRequest, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)
):
    """Queue a generation. Poll GET /tts/jobs/{job_id} until status is completed or failed."""
    if body.output_format == "mp3" and not mp3_available()[0]:
        raise HTTPException(422, mp3_available()[1])
    speed = body.speed
    if body.character_id and "speed" not in body.model_fields_set:
        character = session.get(Character, body.character_id)
        speed = character.default_speed if character else speed
    try:
        job, cached = tts.submit(
            session, settings, text=body.text, voice_id=body.voice_id, character_id=body.character_id,
            language=body.language, output_format=body.output_format,
            params={"speed": speed, **body.params, **({"emotion": body.emotion.strip()} if body.emotion and body.emotion.strip() else {})},
            use_cache=body.use_cache, rvc=body.rvc,
        )
    except tts.TTSRequestError as exc:
        raise HTTPException(exc.status, exc.message) from exc
    return _job(job, cached)


def _get_job(session: Session, job_id: str) -> Generation:
    job = session.get(Generation, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.get("/tts/jobs/{job_id}")
def tts_job(job_id: str, session: Session = Depends(get_session)):
    return _job(_get_job(session, job_id))


@router.get("/audio/{audio_id}")
def get_audio(audio_id: str, format: str | None = None, download: bool = False,
              session: Session = Depends(get_session)):
    job = _get_job(session, audio_id)
    if job.status != "completed" or not Path(job.audio_path).is_file():
        raise HTTPException(404, "audio not available")
    fmt = format or job.output_format
    if fmt not in ("wav", "mp3"):
        raise HTTPException(422, "format must be wav or mp3")
    try:
        data = tts.audio_bytes(job, fmt)
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    headers = {"Content-Disposition": f'attachment; filename="voix-{job.id[:8]}.{fmt}"'} if download else {}
    return Response(data, media_type="audio/mpeg" if fmt == "mp3" else "audio/wav", headers=headers)


# ---------------------------------------------------------------- voice changer

@router.post("/vc/convert", status_code=202, dependencies=[Depends(require_key)])
def convert_audio_file(
    file: UploadFile = File(...), model_id: str = Form(...), pitch: float = Form(0.0),
    index_rate: float = Form(0.5), protect: float = Form(0.33),
    session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
):
    """Re-voice an uploaded WAV / MP3 / FLAC / OGG with an RVC model. Poll /tts/jobs/{job_id}."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".wav", ".mp3", ".flac", ".ogg"):
        raise HTTPException(415, "upload a WAV, MP3, FLAC or OGG file")
    upload = settings.staging_dir / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with open(upload, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > 100 * 1024 * 1024:
                out.close()
                upload.unlink(missing_ok=True)
                raise HTTPException(413, "audio file is larger than 100 MB")
            out.write(chunk)
    try:
        job = tts.submit_conversion(
            session, upload_path=upload, filename=Path(file.filename or "audio").name,
            rvc={"model_id": model_id, "pitch": pitch, "index_rate": index_rate, "protect": protect},
        )
    except tts.TTSRequestError as exc:
        upload.unlink(missing_ok=True)
        raise HTTPException(exc.status, exc.message) from exc
    return _job(job)


# ---------------------------------------------------------------- runtime bundles

def _runtime(settings: Settings, session: Session) -> list[dict]:
    rvc_state = vc.status()
    qwen_ok, qwen_why = tts.get_adapter("qwen3_tts").is_available()
    qwen_installed = session.scalar(select(VoiceModel.id).where(VoiceModel.architecture == "qwen3_tts")) is not None
    return [
        {"id": "rvc", "label": "Voice changer (RVC)", "size_mb": rvc_state["size_mb"],
         "available": rvc_state["libraries"], "installed": rvc_state["runtime_installed"],
         "note": "Needed to use imported RVC character voices. One download.",
         "reason": "" if rvc_state["libraries"] else rvc_state["reason"], **runtime_assets.state("rvc")},
        {"id": "qwen3", "label": "Qwen3-TTS (English, emotion control)", "size_mb": 2400,
         "available": qwen_ok, "installed": qwen_installed,
         "note": "Nine preset voices with emotion control. No Malayalam. About 5x slower than real time on a 2-core CPU.",
         "reason": "" if qwen_ok else qwen_why, **runtime_assets.state("qwen3")},
    ]


@router.get("/runtime")
def runtime_bundles(session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    return _runtime(settings, session)


@router.post("/runtime/{bundle}/install", status_code=202, dependencies=[Depends(require_key)])
def install_runtime(bundle: str, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    entry = next((b for b in _runtime(settings, session) if b["id"] == bundle), None)
    if entry is None:
        raise HTTPException(404, "unknown runtime bundle")
    if not entry["available"]:
        raise HTTPException(409, entry["reason"])
    if entry["installed"]:
        raise HTTPException(409, "already installed")
    work = (lambda: runtime_assets.install(settings, "rvc")) if bundle == "rvc" else (lambda: runtime_assets.install_qwen(settings))
    runtime_assets.start(bundle, work)
    return {"id": bundle, **runtime_assets.state(bundle)}


# ---------------------------------------------------------------- history

@router.get("/history")
def history(q: str = "", voice_id: str = "", language: str = "", kind: str = "", limit: int = 50,
            session: Session = Depends(get_session)):
    query = select(Generation).order_by(Generation.created_at.desc()).limit(min(max(limit, 1), 200))
    if q:
        query = query.where(Generation.text.contains(q))
    if voice_id:
        query = query.where(Generation.voice_id == voice_id)
    if language:
        query = query.where(Generation.language == language)
    if kind == "conversion":
        query = query.where(Generation.kind == "conversion")
    return [_job(j) for j in session.scalars(query).all()]


@router.delete("/history/{job_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_history(job_id: str, session: Session = Depends(get_session)):
    job = _get_job(session, job_id)
    tts.delete_audio(job)
    session.delete(job)
    audit(session, "generation.delete", job_id=job_id)
    session.commit()
