"""REST API, version 1. Provider-independent: it accepts text, never an LLM."""
from __future__ import annotations

import hmac
import json
import re
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__
from ..config import Settings, get_settings
from .. import catalog, emotion, hub, support
from ..audio.processing import mp3_available
from ..db import AuditLog, Character, Generation, Voice, VoiceModel, audit, get_session, new_session
from ..engines.registry import describe_engines
from ..hardware import detect_hardware
from ..importer.service import ImportRejected, import_zip, install_from_hub, serialize
from ..tts import service as tts

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
        "conversion_ready": any(e["ready"] and e["kind"] == "voice_conversion" for e in engines),
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
        "device": settings.device,
        "max_hub_mb": settings.max_hub_bytes // 1024**2,
        "max_convert_mb": settings.max_convert_bytes // 1024**2,
        "hf_token": hub.token_status(settings),
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
def list_models(kind: str = "", session: Session = Depends(get_session)):
    query = select(VoiceModel).order_by(VoiceModel.created_at.desc())
    if kind:
        query = query.where(VoiceModel.kind == kind)
    out = []
    for m in session.scalars(query).all():
        data = serialize(m)
        if m.kind == "voice_conversion":
            data["parameters"] = tts.voice_parameters(m)
        out.append(data)
    return out


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
    for character in session.scalars(select(Character).where(Character.rvc_model_id == model_id)).all():
        character.rvc_model_id = None
    tts.manager().unload(model_id)
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
        "settings": {k: v for k, v in (v.settings or {}).items() if k != "ref_audio"}
        | ({"has_reference": True} if (v.settings or {}).get("ref_audio") else {}),
        "follows_instructions": bool(((model.report.get("detection") or {}).get("details") or {}).get("follows_instructions"))
        if model else False,
        "qwen_model_type": ((model.report.get("detection") or {}).get("details") or {}).get("qwen_model_type") if model else None,
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


class VoiceIn(BaseModel):
    model_id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    #: engine settings: Qwen CustomVoice {"speaker"}, VoiceDesign {"voice_description"}, Piper {"speaker": n}
    settings: dict = Field(default_factory=dict)


_VOICE_SETTING_KEYS = {"speaker", "voice_description", "ref_text", "speed", "variability", "temperature"}


def _clean_voice_settings(model: VoiceModel, raw: dict) -> dict:
    out = {k: v for k, v in raw.items() if k in _VOICE_SETTING_KEYS}
    details = (model.report.get("detection") or {}).get("details") or {}
    if model.architecture == "qwen3_tts" and details.get("qwen_model_type") == "custom_voice":
        speakers = details.get("speakers") or []
        if str(out.get("speaker", "")).lower() not in speakers:
            raise HTTPException(422, f"speaker must be one of: {', '.join(speakers)}")
        out["speaker"] = str(out["speaker"]).lower()
    if isinstance(out.get("voice_description"), str):
        out["voice_description"] = out["voice_description"][:1000]
    return out


@router.post("/voices", status_code=201, dependencies=[Depends(require_key)])
def create_voice(body: VoiceIn, session: Session = Depends(get_session)):
    """Add another voice on an installed TTS model (a Qwen speaker, a designed voice, a clone)."""
    from ..importer.service import _new_voice_id

    model = _get_model(session, body.model_id)
    if model.kind != "tts":
        raise HTTPException(422, "voices are made from text-to-speech models; RVC models are used by characters")
    voice = Voice(id=_new_voice_id(session, body.name), name=body.name, description=body.description,
                  model_id=model.id, languages=model.languages, settings=_clean_voice_settings(model, body.settings))
    session.add(voice)
    audit(session, "voice.create", voice_id=voice.id, model_id=model.id)
    session.commit()
    return _voice(session, voice)


class VoiceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)
    settings: dict | None = None


@router.patch("/voices/{voice_id}", dependencies=[Depends(require_key)])
def update_voice(voice_id: str, body: VoiceUpdate, session: Session = Depends(get_session)):
    voice = session.get(Voice, voice_id)
    if not voice:
        raise HTTPException(404, "voice not found")
    if body.name is not None:
        voice.name = body.name
    if body.description is not None:
        voice.description = body.description
    if body.settings is not None:
        model = _get_model(session, voice.model_id)
        keep = {"ref_audio": voice.settings["ref_audio"]} if (voice.settings or {}).get("ref_audio") else {}
        voice.settings = {**keep, **_clean_voice_settings(model, body.settings)}
    audit(session, "voice.update", voice_id=voice_id)
    session.commit()
    return _voice(session, voice)


@router.post("/voices/{voice_id}/reference", dependencies=[Depends(require_key)])
def upload_reference(voice_id: str, file: UploadFile = File(...), ref_text: str = Form(default=""),
                     session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    """Reference clip (3-15 s) and its exact transcript for a Qwen3-TTS Base (cloning) voice."""
    from ..audio.processing import decode_audio, encode_wav

    voice = session.get(Voice, voice_id)
    if not voice:
        raise HTTPException(404, "voice not found")
    dest = settings.uploads_dir / f"ref-{voice_id}-{uuid.uuid4().hex[:8]}"
    _receive_audio(file, dest, 25 * 1024**2)
    try:
        samples, rate = decode_audio(dest)
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)
    seconds = samples.size / rate
    if not 2 <= seconds <= 30:
        raise HTTPException(422, f"reference clip should be 3-15 seconds long (this one is {seconds:.1f} s)")
    wav = settings.uploads_dir / f"ref-{voice_id}.wav"
    wav.write_bytes(encode_wav(samples, rate))
    voice.settings = {**(voice.settings or {}), "ref_audio": str(wav.relative_to(settings.data_dir)),
                      "ref_text": ref_text.strip()[:1000]}
    audit(session, "voice.reference", voice_id=voice_id, seconds=round(seconds, 2))
    session.commit()
    return _voice(session, voice)


@router.delete("/voices/{voice_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_voice(voice_id: str, session: Session = Depends(get_session)):
    voice = session.get(Voice, voice_id)
    if not voice:
        raise HTTPException(404, "voice not found")
    for character in session.scalars(select(Character)).all():
        if character.voice_ml == voice_id:
            character.voice_ml = None
        if character.voice_en == voice_id:
            character.voice_en = None
    session.delete(voice)
    audit(session, "voice.delete", voice_id=voice_id)
    session.commit()


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
    #: optional stable id (e.g. the id SCOL already uses); derived from the name when empty
    id: str | None = Field(default=None, pattern=r"^[a-z0-9_]{1,60}$")
    style: str = Field(default="", max_length=600)
    rvc_model_id: str | None = None
    rvc_settings: dict = Field(default_factory=dict)
    emotion_mode: str = Field(default="auto", max_length=20)
    default_emotion: str = Field(default="neutral", max_length=20)
    emotion_strength: float = Field(default=1.0, ge=0.0, le=2.0)


def _character(c: Character) -> dict:
    return {
        "id": c.id, "name": c.name, "description": c.description, "image_url": c.image_url,
        "voice_ml": c.voice_ml, "voice_en": c.voice_en, "default_speed": c.default_speed,
        "presets": c.presets, "style": c.style or "", "rvc_model_id": c.rvc_model_id,
        "rvc_settings": c.rvc_settings or {}, "emotion_mode": c.emotion_mode or "auto",
        "default_emotion": c.default_emotion or "neutral",
        "emotion_strength": c.emotion_strength if c.emotion_strength is not None else 1.0,
    }


_RVC_KEYS = {"pitch_shift", "index_rate", "protect", "rms_mix", "f0_method", "speaker"}


def _check_voices(session: Session, body: CharacterIn) -> None:
    for vid in (body.voice_ml, body.voice_en):
        if vid and not session.get(Voice, vid):
            raise HTTPException(422, f"voice '{vid}' does not exist")
    if body.rvc_model_id:
        model = session.get(VoiceModel, body.rvc_model_id)
        if not model or model.kind != "voice_conversion":
            raise HTTPException(422, f"'{body.rvc_model_id}' is not an installed voice-conversion (RVC) model")
    if body.emotion_mode not in ("auto", "none") and emotion.resolve(body.emotion_mode) is None:
        raise HTTPException(422, "emotion_mode must be auto, none or an emotion name")
    if emotion.resolve(body.default_emotion) is None:
        raise HTTPException(422, f"unknown default_emotion '{body.default_emotion}'")
    unknown = set(body.rvc_settings) - _RVC_KEYS
    if unknown:
        raise HTTPException(422, f"unknown rvc_settings: {', '.join(sorted(unknown))}")


def _character_fields(body: CharacterIn) -> dict:
    data = body.model_dump(exclude={"id"})
    data["default_emotion"] = emotion.resolve(body.default_emotion)
    if body.emotion_mode not in ("auto", "none"):
        data["emotion_mode"] = emotion.resolve(body.emotion_mode)
    return data


@router.get("/characters")
def list_characters(session: Session = Depends(get_session)):
    return [_character(c) for c in session.scalars(select(Character).order_by(Character.name)).all()]


@router.post("/characters", status_code=201, dependencies=[Depends(require_key)])
def create_character(body: CharacterIn, session: Session = Depends(get_session)):
    cid = body.id or re.sub(r"[^a-z0-9]+", "_", body.name.lower()).strip("_")[:60] or "character"
    if session.get(Character, cid):
        raise HTTPException(409, f"character id '{cid}' already exists")
    _check_voices(session, body)
    character = Character(id=cid, **_character_fields(body))
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
    for key, value in _character_fields(body).items():
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
    #: auto | none | an emotion name. Omit to use the character's setting (none for plain voices).
    #: Inline tags such as "[happy] Well done!" always win over this.
    emotion: str | None = Field(default=None, max_length=20)
    emotion_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    use_cache: bool = True


def _job(job: Generation, cached: bool = False) -> dict:
    data = {
        "job_id": job.id, "kind": job.kind or "tts", "status": job.status, "cached": cached, "language": job.language,
        "voice_id": job.voice_id, "character_id": job.character_id, "output_format": job.output_format,
        "text": job.text, "params": job.params, "warnings": job.warnings,
        "created_at": job.created_at.isoformat(),
    }
    if job.status == "completed":
        data |= {
            "audio_id": job.id, "audio_url": f"/api/v1/audio/{job.id}",
            "audio_seconds": job.audio_seconds, "generation_seconds": job.generation_seconds,
            "sample_rate": job.sample_rate, "chunks": job.chunks,
            # Engines return whole sentences, so this is chunked generation, not streaming.
            "mode": "chunked",
            "segments": job.segments or [],
            "model_ids": job.model_ids or [],
        }
    if job.status == "failed":
        data["error"] = job.error
    return data


def _tts_params(body: TTSRequest, session: Session) -> dict:
    speed = body.speed
    if body.character_id and "speed" not in body.model_fields_set:
        character = session.get(Character, body.character_id)
        speed = character.default_speed if character else speed
    params: dict = {"speed": speed, **body.params}
    if body.emotion is not None:
        params["emotion"] = body.emotion.strip().lower()
    if body.emotion_strength is not None:
        params["emotion_strength"] = body.emotion_strength
    return params


def _submit(body: TTSRequest, session: Session, settings: Settings) -> tuple[Generation, bool]:
    if body.output_format == "mp3" and not mp3_available()[0]:
        raise HTTPException(422, mp3_available()[1])
    try:
        return tts.submit(
            session, settings, text=body.text, voice_id=body.voice_id, character_id=body.character_id,
            language=body.language, output_format=body.output_format,
            params=_tts_params(body, session), use_cache=body.use_cache,
        )
    except tts.TTSRequestError as exc:
        raise HTTPException(exc.status, exc.message) from exc


def _wait(job_id: str, seconds: float) -> Generation:
    """Poll the job in a fresh session until it finishes or the wait runs out."""
    deadline = time.monotonic() + seconds
    while True:
        with new_session() as fresh:
            job = fresh.get(Generation, job_id)
            if job is None or job.status in ("completed", "failed") or time.monotonic() >= deadline:
                return job
        time.sleep(0.2)


@router.post("/tts/generate", status_code=202, dependencies=[Depends(require_key)])
def tts_generate(
    body: TTSRequest, wait: float = Query(default=0, ge=0, le=300,
                                          description="seconds to wait for the job to finish before answering"),
    session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
):
    """Queue a generation. Poll GET /tts/jobs/{job_id} until status is completed or failed, or pass ?wait=60."""
    job, cached = _submit(body, session, settings)
    if wait and not cached and job.status not in ("completed", "failed"):
        job = _wait(job.id, wait)
    return _job(job, cached)


@router.post("/tts/speak", dependencies=[Depends(require_key)],
             responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}})
def tts_speak(body: TTSRequest, timeout: float = Query(default=120, ge=1, le=600),
              session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    """One call, audio back: what a tutor app or an LLM tool should use.

    The response body is the audio. Headers carry the job id, duration and the
    emotion timeline (X-VOIX-Segments, JSON) so an avatar can react in sync.
    """
    job, _ = _submit(body, session, settings)
    if job.status != "completed":
        job = _wait(job.id, timeout)
    if job.status == "failed":
        raise HTTPException(500, job.error)
    if job.status != "completed":
        raise HTTPException(504, f"not finished after {timeout:.0f} s; poll /api/v1/tts/jobs/{job.id}")
    data = tts.audio_bytes(job, body.output_format)
    timeline = [{"emotion": s.get("emotion"), "start": s.get("start"), "end": s.get("end")} for s in job.segments or []]
    headers = {
        "X-VOIX-Job-Id": job.id, "X-VOIX-Audio-Seconds": str(job.audio_seconds),
        "X-VOIX-Segments": json.dumps(timeline, separators=(",", ":"))[:7000],
        "Access-Control-Expose-Headers": "X-VOIX-Job-Id, X-VOIX-Audio-Seconds, X-VOIX-Segments",
    }
    return Response(data, media_type="audio/mpeg" if body.output_format == "mp3" else "audio/wav", headers=headers)


class BatchRequest(BaseModel):
    items: list[TTSRequest] = Field(min_length=1, max_length=500)


@router.post("/tts/batch", status_code=202, dependencies=[Depends(require_key)])
def tts_batch(body: BatchRequest, session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    """Queue many lines at once (e.g. every lesson line for every tutor). Each item is a /tts/generate body."""
    out = []
    for i, item in enumerate(body.items):
        try:
            job, cached = _submit(item, session, settings)
            out.append(_job(job, cached))
        except HTTPException as exc:
            out.append({"index": i, "status": "rejected", "error": exc.detail})
    return {"jobs": out, "queued": sum(1 for j in out if j.get("status") in ("queued", "processing")),
            "cached": sum(1 for j in out if j.get("cached")), "rejected": sum(1 for j in out if j.get("status") == "rejected")}


# ---------------------------------------------------------------- voice conversion (speech to speech)

_AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".webm"}


def _receive_audio(file: UploadFile, dest: Path, limit: int) -> int:
    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"audio file is larger than {limit // 1024**2} MB")
            out.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "empty file")
    return size


@router.post("/convert", status_code=202, dependencies=[Depends(require_key)])
def convert_audio(
    file: UploadFile = File(..., description="speech to re-voice: mp3, wav, flac, ogg (m4a/aac/webm need FFmpeg)"),
    model_id: str | None = Form(default=None, description="an installed RVC model"),
    character_id: str | None = Form(default=None, description="or a character: its RVC model and settings"),
    pitch_shift: float | None = Form(default=None, ge=-24, le=24),
    index_rate: float | None = Form(default=None, ge=0, le=1),
    protect: float | None = Form(default=None, ge=0, le=0.5),
    rms_mix: float | None = Form(default=None, ge=0, le=1),
    f0_method: str | None = Form(default=None, pattern="^(rmvpe|fcpe)$"),
    output_format: str = Form(default="mp3", pattern="^(mp3|wav)$"),
    session: Session = Depends(get_session), settings: Settings = Depends(get_settings),
):
    """Change the voice of an uploaded recording with an RVC model. Poll /tts/jobs/{job_id}."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _AUDIO_SUFFIXES:
        raise HTTPException(415, f"upload one of: {', '.join(sorted(_AUDIO_SUFFIXES))}")
    if output_format == "mp3" and not mp3_available()[0]:
        raise HTTPException(422, mp3_available()[1])
    dest = settings.uploads_dir / f"src-{uuid.uuid4().hex}{suffix}"
    _receive_audio(file, dest, settings.max_convert_bytes)
    params = {k: v for k, v in {"pitch_shift": pitch_shift, "index_rate": index_rate, "protect": protect,
                                "rms_mix": rms_mix, "f0_method": f0_method}.items() if v is not None}
    try:
        job = tts.submit_convert(session, settings, source=dest, filename=file.filename or "audio", model_id=model_id,
                                 character_id=character_id, params=params, output_format=output_format)
    except tts.TTSRequestError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(exc.status, exc.message) from exc
    audit(session, "convert.submit", job_id=job.id, model_id=job.model_ids)
    session.commit()
    return _job(job)


# ---------------------------------------------------------------- emotion

@router.get("/emotions")
def emotions():
    """The emotion vocabulary, the inline tag syntax, and whether the classifier is installed."""
    return emotion.describe()


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    language: str = Field(default="auto", pattern="^(en|ml|auto)$")
    emotion: str = Field(default="auto", max_length=20)
    default_emotion: str = Field(default="neutral", max_length=20)


@router.post("/emotions/analyze")
def analyze_emotions(body: AnalyzeRequest):
    """Preview which emotion each sentence would get, without generating audio."""
    from ..text.processing import detect_language

    lang = detect_language(emotion.strip_tags(body.text)) if body.language == "auto" else body.language
    plan = tts.Plan(lang, {"ml": "", "en": ""}, True, [])
    mode = body.emotion if body.emotion in ("auto", "none") else (emotion.resolve(body.emotion) or "auto")
    items = tts._plan_chunks(body.text, plan, mode)
    tts._decide_emotions(items, mode, emotion.resolve(body.default_emotion) or "neutral")
    return {"language": lang, "classifier": emotion.classifier_status(), "segments": items}


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


# ---------------------------------------------------------------- history

@router.get("/history")
def history(q: str = "", voice_id: str = "", language: str = "", kind: str = "", limit: int = 50,
            session: Session = Depends(get_session)):
    query = select(Generation).order_by(Generation.created_at.desc()).limit(min(max(limit, 1), 200))
    if kind:
        query = query.where(Generation.kind == kind)
    if q:
        query = query.where(Generation.text.contains(q))
    if voice_id:
        query = query.where(Generation.voice_id == voice_id)
    if language:
        query = query.where(Generation.language == language)
    return [_job(j) for j in session.scalars(query).all()]


@router.delete("/history/{job_id}", status_code=204, dependencies=[Depends(require_key)])
def delete_history(job_id: str, session: Session = Depends(get_session)):
    job = _get_job(session, job_id)
    tts.delete_audio(job)
    session.delete(job)
    audit(session, "generation.delete", job_id=job_id)
    session.commit()


# ---------------------------------------------------------------- Hugging Face

def _hub_error(exc: hub.HubError) -> HTTPException:
    return HTTPException(exc.status, exc.message)


@router.get("/hub/inspect")
def hub_inspect(repo: str, settings: Settings = Depends(get_settings)):
    """List a Hugging Face repository and suggest installable packages (RVC voices, Qwen3-TTS, Piper, ZIPs)."""
    try:
        return hub.inspect_repo(repo, settings)
    except hub.HubError as exc:
        raise _hub_error(exc) from exc


class HubInstall(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200)
    revision: str = Field(default="main", max_length=100)
    files: list[str] = Field(min_length=1, max_length=2000)
    name: str | None = Field(default=None, max_length=120)
    license: str = Field(default="", max_length=120)


@router.post("/hub/install", status_code=202, dependencies=[Depends(require_key)])
def hub_install(body: HubInstall, settings: Settings = Depends(get_settings)):
    """Download the chosen files in the background, then inspect and install them. Poll /hub/tasks/{id}."""
    try:
        repo, _, _ = hub.parse_repo(body.repo_id)
        listing = {f["path"]: f["size"] for f in hub.list_files(repo, body.revision, hub.get_token(settings))}
    except hub.HubError as exc:
        raise _hub_error(exc) from exc
    unknown = [f for f in body.files if f not in listing]
    if unknown:
        raise HTTPException(422, f"not in the repository: {', '.join(unknown[:5])}")
    blocked = [f for f in body.files if not hub.is_data_file(f)]
    if blocked:
        raise HTTPException(422, f"VOIX never downloads code or unknown file types: {', '.join(blocked[:5])}")
    total = sum(listing[f] for f in body.files)
    if total > settings.max_hub_bytes:
        raise HTTPException(413, f"selection is {total // 1024**2} MB; the limit is {settings.max_hub_bytes // 1024**2} MB")

    def work(progress):
        with new_session() as session:
            model = install_from_hub(session, settings, repo=repo, revision=body.revision, paths=body.files,
                                     name=body.name, license_name=body.license, progress=progress)
            return {"model_id": model.id, "name": model.name, "architecture": model.architecture,
                    "kind": model.kind, "completeness": model.completeness}

    return hub.start_task(f"{repo}: {body.name or ', '.join(body.files[:2])}", total, work)


@router.get("/hub/tasks")
def hub_tasks():
    return hub.tasks()


@router.get("/hub/tasks/{task_id}")
def hub_task(task_id: str):
    found = hub.task(task_id)
    if not found:
        raise HTTPException(404, "task not found (tasks are kept in memory until VOIX restarts)")
    return found


class TokenIn(BaseModel):
    token: str | None = Field(default=None, max_length=200)


@router.get("/hub/token")
def hub_token_status(settings: Settings = Depends(get_settings)):
    """Whether a Hugging Face token is configured. The token itself is never returned."""
    return hub.token_status(settings)


@router.put("/hub/token", dependencies=[Depends(require_key)])
def hub_token_save(body: TokenIn, settings: Settings = Depends(get_settings)):
    """Save (or with null, remove) the Hugging Face token used for gated and private repositories."""
    try:
        hub.save_token(settings, body.token)
    except hub.HubError as exc:
        raise _hub_error(exc) from exc
    return hub.token_status(settings)


# ---------------------------------------------------------------- support models

@router.get("/support")
def support_models(settings: Settings = Depends(get_settings)):
    return support.describe(settings)


@router.post("/support/{support_id}/install", status_code=202, dependencies=[Depends(require_key)])
def support_install(support_id: str, settings: Settings = Depends(get_settings)):
    if support_id not in support.SUPPORT:
        raise HTTPException(404, "unknown support model")
    spec = support.SUPPORT[support_id]

    def work(progress):
        support.ensure(support_id, settings, progress=progress)
        if support_id == "emotion":
            emotion.reset_classifier()
        return {"support_id": support_id, "installed": True}

    return hub.start_task(spec["label"], spec["size_mb"] * 1024**2, work)


@router.delete("/support/{support_id}", status_code=204, dependencies=[Depends(require_key)])
def support_remove(support_id: str, settings: Settings = Depends(get_settings)):
    if support_id not in support.SUPPORT:
        raise HTTPException(404, "unknown support model")
    support.remove(support_id, settings)
    if support_id == "emotion":
        emotion.reset_classifier()
