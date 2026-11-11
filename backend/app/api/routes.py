"""REST API, version 1. Provider-independent: it accepts text, never an LLM."""
from __future__ import annotations

import hmac
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import Settings, get_settings
from ..db import AuditLog, Character, Voice, VoiceModel, audit, get_session
from ..engines.registry import describe_engines
from ..hardware import detect_hardware
from ..importer.service import ImportRejected, import_zip, serialize

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
    session.delete(model)
    audit(session, "model.delete", model_id=model_id, name=model.name)
    session.commit()


# ---------------------------------------------------------------- voices

@router.get("/voices")
def list_voices(session: Session = Depends(get_session)):
    rows = session.scalars(select(Voice).order_by(Voice.name)).all()
    return [
        {"id": v.id, "name": v.name, "description": v.description, "model_id": v.model_id, "languages": v.languages}
        for v in rows
    ]


# ---------------------------------------------------------------- characters

class CharacterIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    image_url: str = Field(default="", max_length=500)
    voice_ml: str | None = None
    voice_en: str | None = None
    default_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    presets: dict = Field(default_factory=dict)


def _character(c: Character) -> dict:
    return {
        "id": c.id, "name": c.name, "description": c.description, "image_url": c.image_url,
        "voice_ml": c.voice_ml, "voice_en": c.voice_en, "default_speed": c.default_speed,
        "presets": c.presets,
    }


def _check_voices(session: Session, body: CharacterIn) -> None:
    for vid in (body.voice_ml, body.voice_en):
        if vid and not session.get(Voice, vid):
            raise HTTPException(422, f"voice '{vid}' does not exist")


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
    language: str = "en"
    output_format: str = Field(default="mp3", pattern="^(mp3|wav)$")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@router.post("/tts/generate", dependencies=[Depends(require_key)])
def tts_generate(body: TTSRequest):
    """The request contract is final; synthesis arrives with the first engine adapter."""
    if not any(e["ready"] and e["kind"] == "tts" for e in describe_engines()):
        raise HTTPException(
            501,
            "No TTS engine adapter is installed yet (Phase 2). The request was valid, "
            "but VOIX will not return placeholder audio.",
        )
    raise HTTPException(501, "generation pipeline not implemented")
