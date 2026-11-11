"""Generation pipeline and job queue.

text -> normalise -> language runs -> sentence chunks -> engine -> trim/join/level -> WAV

Jobs run on a single background worker so two requests never fight over the
CPU or load the same model twice. Every failure is recorded on the job.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..audio import processing as audio
from ..config import Settings, get_settings
from ..db import Character, Generation, Voice, VoiceModel, new_session
from ..engines.manager import EngineManager
from ..engines.registry import get_adapter, inference_status
from ..text.processing import chunk_text, detect_language, normalize

PIPELINE_VERSION = 1


class TTSRequestError(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.status = status


@dataclass
class Plan:
    language: str
    voices: dict[str, str]  # language -> voice_id
    mixed: bool
    warnings: list[str]


_manager: EngineManager | None = None
_executor: ThreadPoolExecutor | None = None


def manager() -> EngineManager:
    global _manager
    if _manager is None:
        _manager = EngineManager(max_loaded=get_settings().max_loaded_models)
    return _manager


def _pool() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voix-tts")
    return _executor


def recover_interrupted_jobs(session: Session) -> None:
    """Jobs left queued or running by a previous process can never finish."""
    session.execute(
        update(Generation)
        .where(Generation.status.in_(("queued", "processing")))
        .values(status="failed", error="interrupted: the server stopped before this job finished")
    )
    session.commit()


def _usable_voice(session: Session, voice_id: str) -> tuple[Voice, VoiceModel]:
    voice = session.get(Voice, voice_id)
    if not voice:
        raise TTSRequestError(f"voice '{voice_id}' does not exist", 404)
    model = session.get(VoiceModel, voice.model_id)
    if not model:
        raise TTSRequestError(f"voice '{voice_id}' points at a model that was deleted", 409)
    status = inference_status(model.architecture)
    if not status["ready"]:
        raise TTSRequestError(f"voice '{voice_id}' cannot generate: {status['reason']}", 409)
    return voice, model


def make_plan(session: Session, *, text: str, voice_id: str | None, character_id: str | None, language: str) -> Plan:
    if language not in ("en", "ml", "auto"):
        raise TTSRequestError("language must be 'en', 'ml' or 'auto'")
    lang = detect_language(text) if language == "auto" else language
    warnings: list[str] = []

    if character_id:
        character = session.get(Character, character_id)
        if not character:
            raise TTSRequestError(f"character '{character_id}' does not exist", 404)
        voices = {k: v for k, v in (("ml", character.voice_ml), ("en", character.voice_en)) if v}
        if not voices:
            raise TTSRequestError(f"character '{character_id}' has no voice assigned", 409)
        for code in ("ml", "en"):
            if code not in voices:
                other = next(iter(voices.values()))
                voices[code] = other
                if code == lang:
                    warnings.append(f"Character has no {code} voice; '{other}' was used and will mispronounce it.")
        mixed = voices["ml"] != voices["en"]
    elif voice_id:
        voices = {"ml": voice_id, "en": voice_id}
        mixed = False
    else:
        raise TTSRequestError("send voice_id or character_id")

    for vid in set(voices.values()):
        voice, _ = _usable_voice(session, vid)
        if not mixed and voice.languages and lang not in voice.languages:
            warnings.append(
                f"Voice '{vid}' is a {'/'.join(voice.languages)} voice; {lang} text will be mispronounced."
            )
    return Plan(lang, voices, mixed, warnings)


def cache_key(text: str, plan: Plan, params: dict, session: Session) -> str:
    models = sorted(session.get(Voice, v).model_id for v in set(plan.voices.values()))  # type: ignore[union-attr]
    blob = json.dumps(
        {"v": PIPELINE_VERSION, "text": text, "lang": plan.language, "voices": plan.voices, "models": models,
         "mixed": plan.mixed, "params": params},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def submit(session: Session, settings: Settings, *, text: str, voice_id: str | None, character_id: str | None,
           language: str, output_format: str, params: dict, use_cache: bool = True) -> tuple[Generation, bool]:
    """Create a job (or return a cached one). Returns (job, served_from_cache)."""
    text = text.strip()
    if not text:
        raise TTSRequestError("text is empty")
    plan = make_plan(session, text=text, voice_id=voice_id, character_id=character_id, language=language)
    key = cache_key(text, plan, params, session)

    if use_cache and settings.cache_enabled:
        hit = session.scalar(
            select(Generation)
            .where(Generation.cache_key == key, Generation.status == "completed")
            .order_by(Generation.created_at.desc())
        )
        if hit and Path(hit.audio_path).is_file():
            return hit, True

    job = Generation(
        text=text, language=plan.language, voice_id=voice_id, character_id=character_id, params=params,
        output_format=output_format, cache_key=key, warnings=plan.warnings,
    )
    session.add(job)
    session.commit()
    _pool().submit(_run, job.id, plan)
    return job, False


def _run(job_id: str, plan: Plan) -> None:
    settings = get_settings()
    session = new_session()
    try:
        job = session.get(Generation, job_id)
        if job is None:
            return
        job.status = "processing"
        session.commit()
        started = time.perf_counter()
        try:
            samples, rate, chunk_count, model_ids = synthesize(session, job.text, plan, job.params)
            path = settings.audio_dir / f"{job.id}.wav"
            path.write_bytes(audio.encode_wav(samples, rate))
            job.audio_path = str(path)
            job.sample_rate = rate
            job.audio_seconds = round(samples.size / rate, 3)
            job.chunks = chunk_count
            job.model_ids = model_ids
            job.status = "completed"
        except Exception as exc:  # recorded on the job, never swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        job.generation_seconds = round(time.perf_counter() - started, 3)
        session.commit()
    finally:
        session.close()


def synthesize(session: Session, text: str, plan: Plan, params: dict) -> tuple[np.ndarray, int, int, list[str]]:
    """Run the whole pipeline in the calling thread and return float32 mono audio."""
    chunks = chunk_text(normalize(text, plan.language), plan.language, mixed=plan.mixed)
    if not chunks:
        raise TTSRequestError("nothing speakable in the text")

    engines = {}
    for vid in {plan.voices[c.language] for c in chunks}:
        voice, model = _usable_voice(session, vid)
        engines[vid] = (manager().get(model.id, model.architecture, Path(model.storage_path)), voice, model)

    target_rate = None
    pieces: list[np.ndarray] = []
    for chunk in chunks:
        engine, voice, _ = engines[plan.voices[chunk.language]]
        allowed = {p.name for p in engine.get_supported_parameters()}
        call = {**voice.settings, **{k: v for k, v in params.items() if k in allowed}}
        result = engine.generate_speech(chunk.text, chunk.language, **call)
        target_rate = target_rate or result.sample_rate
        samples = audio.resample(np.asarray(result.samples, dtype=np.float32), result.sample_rate, target_rate)
        samples = audio.fade_edges(audio.trim_silence(samples, target_rate), target_rate)
        if samples.size:
            pieces.append(samples)
            if chunk.pause_ms:
                pieces.append(audio.silence(chunk.pause_ms, target_rate))
    if not pieces:
        raise RuntimeError("the engine produced only silence")
    joined = audio.normalize_peak(np.concatenate(pieces))
    return joined, int(target_rate), len(chunks), sorted({m.id for _, _, m in engines.values()})


def audio_bytes(job: Generation, fmt: str) -> bytes:
    """WAV is the stored master; MP3 is encoded from it once and kept beside it."""
    wav_path = Path(job.audio_path)
    if fmt == "wav":
        return wav_path.read_bytes()
    mp3_path = wav_path.with_suffix(".mp3")
    if not mp3_path.is_file():
        mp3_path.write_bytes(audio.wav_to_mp3(wav_path.read_bytes()))
    return mp3_path.read_bytes()


def delete_audio(job: Generation) -> None:
    if job.audio_path:
        for path in (Path(job.audio_path), Path(job.audio_path).with_suffix(".mp3")):
            path.unlink(missing_ok=True)


def voice_parameters(model: VoiceModel) -> list[dict]:
    """Controls the studio may show for a voice, straight from its adapter."""
    adapter = get_adapter(model.architecture)
    if adapter is None:
        return []
    details = (model.report.get("detection") or {}).get("details") or {}
    specs = adapter.parameters_for(int(details.get("num_speakers") or 1)) if hasattr(adapter, "parameters_for") else []
    return [
        {"name": s.name, "label": s.label, "type": s.type, "default": s.default, "min": s.minimum, "max": s.maximum}
        for s in specs
    ]
