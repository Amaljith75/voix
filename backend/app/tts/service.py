"""Generation pipeline and job queue.

text -> emotion tags -> normalise -> language runs -> sentence chunks
     -> emotion per chunk -> TTS engine (instruction or prosody)
     -> trim / join -> optional RVC character timbre (pitch follows emotion) -> level -> WAV

Voice conversion of uploaded audio (MP3, WAV ...) runs on the same queue.

Jobs run on a single background worker so two requests never fight over the
CPU or load the same model twice. Every failure is recorded on the job.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .. import emotion as emo
from ..audio import processing as audio
from ..config import Settings, get_settings
from ..db import Character, Generation, Voice, VoiceModel, new_session
from ..engines.manager import EngineManager
from ..engines.registry import get_adapter, inference_status
from ..text.processing import chunk_text, detect_language, normalize

PIPELINE_VERSION = 2
PART_PAUSE_MS = 170
MAX_CONVERT_SECONDS = 15 * 60


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
    character: dict = field(default_factory=dict)  # snapshot of the character's voice settings


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


def usable_converter(session: Session, model_id: str) -> VoiceModel:
    model = session.get(VoiceModel, model_id)
    if not model:
        raise TTSRequestError(f"voice-conversion model '{model_id}' does not exist", 404)
    if model.kind != "voice_conversion":
        raise TTSRequestError(f"model '{model.name}' is {model.architecture}, not a voice-conversion (RVC) model")
    status = inference_status(model.architecture)
    if not status["ready"]:
        raise TTSRequestError(f"'{model.name}' cannot run: {status['reason']}", 409)
    return model


def character_snapshot(c: Character) -> dict:
    return {
        "id": c.id, "style": c.style or "", "rvc_model_id": c.rvc_model_id, "rvc_settings": c.rvc_settings or {},
        "emotion_mode": c.emotion_mode or "auto", "default_emotion": c.default_emotion or "neutral",
        "emotion_strength": c.emotion_strength if c.emotion_strength is not None else 1.0,
    }


def make_plan(session: Session, *, text: str, voice_id: str | None, character_id: str | None, language: str) -> Plan:
    if language not in ("en", "ml", "auto"):
        raise TTSRequestError("language must be 'en', 'ml' or 'auto'")
    lang = detect_language(emo.strip_tags(text)) if language == "auto" else language
    warnings: list[str] = []
    snapshot: dict = {}

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
        snapshot = character_snapshot(character)
        if character.rvc_model_id:
            usable_converter(session, character.rvc_model_id)
    elif voice_id:
        voices = {"ml": voice_id, "en": voice_id}
        mixed = False
    else:
        raise TTSRequestError("send voice_id or character_id")

    for vid in set(voices.values()):
        voice, model = _usable_voice(session, vid)
        if not mixed and voice.languages and lang not in voice.languages:
            warnings.append(
                f"Voice '{vid}' is a {'/'.join(voice.languages)} voice; {lang} text will be mispronounced."
            )
    return Plan(lang, voices, mixed, warnings, snapshot)


def cache_key(text: str, plan: Plan, params: dict, session: Session) -> str:
    models = sorted(session.get(Voice, v).model_id for v in set(plan.voices.values()))  # type: ignore[union-attr]
    settings = {v: session.get(Voice, v).settings for v in set(plan.voices.values())}  # type: ignore[union-attr]
    blob = json.dumps(
        {"v": PIPELINE_VERSION, "text": text, "lang": plan.language, "voices": plan.voices, "models": models,
         "voice_settings": settings, "mixed": plan.mixed, "params": params, "character": plan.character},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def submit(session: Session, settings: Settings, *, text: str, voice_id: str | None, character_id: str | None,
           language: str, output_format: str, params: dict, use_cache: bool = True) -> tuple[Generation, bool]:
    """Create a job (or return a cached one). Returns (job, served_from_cache)."""
    text = text.strip()
    if not text:
        raise TTSRequestError("text is empty")
    mode = params.get("emotion")
    if mode is not None and mode not in ("auto", "none") and emo.resolve(str(mode)) is None:
        raise TTSRequestError(f"unknown emotion '{mode}'. Use auto, none, or one of: {', '.join(emo.EMOTIONS)}")
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


def _finish(job: Generation, samples: np.ndarray, rate: int, settings: Settings) -> None:
    path = settings.audio_dir / f"{job.id}.wav"
    path.write_bytes(audio.encode_wav(samples, rate))
    job.audio_path = str(path)
    job.sample_rate = rate
    job.audio_seconds = round(samples.size / rate, 3)
    job.status = "completed"


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
            result = synthesize(session, job.text, plan, job.params)
            job.chunks = len(result.segments)
            job.model_ids = result.model_ids
            job.segments = result.segments
            job.warnings = list(job.warnings or []) + result.warnings
            _finish(job, result.samples, result.rate, settings)
        except Exception as exc:  # recorded on the job, never swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
        job.generation_seconds = round(time.perf_counter() - started, 3)
        session.commit()
    finally:
        session.close()


@dataclass
class Synthesis:
    samples: np.ndarray
    rate: int
    segments: list[dict]
    model_ids: list[str]
    warnings: list[str]


def _voice_call_settings(voice: Voice) -> dict:
    """A voice's stored settings, with its reference clip (Qwen Base) as an absolute path."""
    out = dict(voice.settings or {})
    if out.get("ref_audio"):
        out["ref_audio"] = str(get_settings().data_dir / out["ref_audio"])
    return out


def _plan_chunks(text: str, plan: Plan, mode: str) -> list[dict]:
    """Emotion parts -> chunks, each with the emotion a tag gave it (or None)."""
    parts = emo.parse_tags(text, keep_tags=mode != "none")
    items: list[dict] = []
    for i, part in enumerate(parts):
        chunks = chunk_text(normalize(part.text, plan.language), plan.language, mixed=plan.mixed)
        for j, c in enumerate(chunks):
            pause = c.pause_ms
            if j == len(chunks) - 1 and i < len(parts) - 1:
                pause = part.pause_ms or PART_PAUSE_MS
            elif j == len(chunks) - 1:
                pause = part.pause_ms
            items.append({"text": c.text, "language": c.language, "pause_ms": pause, "emotion": part.emotion,
                          "source": "tag" if part.emotion else None})
    return items


def _decide_emotions(items: list[dict], mode: str, default: str, implicit: bool = False) -> None:
    open_items = [it for it in items if it["emotion"] is None]
    if not open_items:
        return
    if mode == "auto":
        found = emo.detect([it["text"] for it in open_items], [it["language"] for it in open_items])
        for it, (name, method, score) in zip(open_items, found):
            if name == "neutral" and default != "neutral":
                it["emotion"], it["source"] = default, "default"
            else:
                it["emotion"], it["source"], it["confidence"] = name, method, score
    elif mode == "none":
        for it in open_items:
            it["emotion"], it["source"] = "neutral", "off"
    else:
        name = emo.resolve(mode) or "neutral"
        for it in open_items:
            it["emotion"], it["source"] = name, "default" if implicit else "request"


def synthesize(session: Session, text: str, plan: Plan, params: dict) -> Synthesis:
    """Run the whole pipeline in the calling thread and return float32 mono audio."""
    char = plan.character
    # Tags always count unless the caller says emotion "none". Untagged text gets the request's
    # emotion, else the character's mode, else stays neutral.
    requested = params.get("emotion") or (char.get("emotion_mode") if char else None)
    mode = str(requested or "neutral")
    strength = float(params.get("emotion_strength", char.get("emotion_strength", 1.0) if char else 1.0))
    style = char.get("style", "") if char else ""
    default_emotion = (char.get("default_emotion") if char else None) or "neutral"

    items = _plan_chunks(text, plan, mode)
    if not items:
        raise TTSRequestError("nothing speakable in the text")
    _decide_emotions(items, mode, default_emotion, implicit=requested is None)

    engines = {}
    for vid in {plan.voices[it["language"]] for it in items}:
        voice, model = _usable_voice(session, vid)
        engines[vid] = (manager().get(model.id, model.architecture, Path(model.storage_path)), voice, model)

    warnings: list[str] = []
    target_rate = None
    pieces: list[np.ndarray] = []
    position = 0
    segments: list[dict] = []
    for it in items:
        engine, voice, _ = engines[plan.voices[it["language"]]]
        allowed = {p.name for p in engine.get_supported_parameters()}
        call = {**_voice_call_settings(voice), **{k: v for k, v in params.items() if k in allowed}}
        look = emo.render(it["emotion"], strength, style)
        rendered = []
        if "speed" in allowed:
            call["speed"] = min(max(float(call.get("speed", 1.0)) * look.speed, 0.5), 2.0)
            rendered.append("speed")
        if "variability" in allowed and look.variability:
            call["variability"] = min(max(float(call.get("variability", 0.667)) + look.variability, 0.2), 1.0)
            rendered.append("variability")
        if getattr(engine, "follows_instructions", False) and look.instruct:
            call["instruct"] = look.instruct
            rendered.insert(0, "instruction")
        result = engine.generate_speech(it["text"], it["language"], **call)
        target_rate = target_rate or result.sample_rate
        samples = audio.resample_hq(np.asarray(result.samples, dtype=np.float32), result.sample_rate, target_rate)
        samples = audio.fade_edges(audio.trim_silence(samples, target_rate), target_rate)
        if samples.size:
            pieces.append(samples)
            segments.append({"text": it["text"], "language": it["language"], "emotion": it["emotion"],
                             "source": it["source"], "confidence": it.get("confidence"),
                             "rendered_by": rendered, "pitch": look.pitch,
                             "start": round(position / target_rate, 3),
                             "end": round((position + samples.size) / target_rate, 3)})
            position += samples.size
            if it["pause_ms"]:
                gap = audio.silence(it["pause_ms"], target_rate)
                pieces.append(gap)
                position += gap.size
    if not pieces:
        raise RuntimeError("the engine produced only silence")
    joined = np.concatenate(pieces)
    rate = int(target_rate)
    model_ids = {m.id for _, _, m in engines.values()}

    if char and char.get("rvc_model_id"):
        rvc = usable_converter(session, char["rvc_model_id"])
        converter = manager().get(rvc.id, rvc.architecture, Path(rvc.storage_path))
        allowed = {p.name for p in converter.get_supported_parameters()}
        rvc_params = {k: v for k, v in (char.get("rvc_settings") or {}).items() if k in allowed}
        track = [(s["start"], s["end"], s["pitch"]) for s in segments if s["pitch"]]
        converted = converter.convert(joined, rate, pitch_track=track, **rvc_params)
        scale = converted.samples.size / max(joined.size, 1) * rate / converted.sample_rate
        for s in segments:
            s["start"], s["end"] = round(s["start"] * scale, 3), round(s["end"] * scale, 3)
            if s["pitch"]:
                s["rendered_by"] = s["rendered_by"] + ["pitch"]
        joined, rate = converted.samples, converted.sample_rate
        model_ids.add(rvc.id)
    if any(not s["rendered_by"] and s["emotion"] != "neutral" for s in segments):
        warnings.append("This voice has no way to express emotion (no speed, style or pitch control).")
    return Synthesis(audio.normalize_peak(joined), rate, segments, sorted(model_ids), warnings)


# ---------------------------------------------------------------- voice conversion of uploaded audio

def submit_convert(session: Session, settings: Settings, *, source: Path, filename: str, model_id: str | None,
                   character_id: str | None, params: dict, output_format: str) -> Generation:
    character = None
    if character_id:
        character = session.get(Character, character_id)
        if not character:
            raise TTSRequestError(f"character '{character_id}' does not exist", 404)
        if not model_id:
            model_id = character.rvc_model_id
            params = {**(character.rvc_settings or {}), **params}
    if not model_id:
        raise TTSRequestError("choose an RVC model (model_id) or a character that has one")
    model = usable_converter(session, model_id)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps({"convert": digest, "model": model.id, "params": params},
                                    sort_keys=True).encode()).hexdigest()
    job = Generation(kind="convert", text=filename[:300], language="", character_id=character_id, params=params,
                     output_format=output_format, cache_key=key, source_path=str(source), model_ids=[model.id])
    session.add(job)
    session.commit()
    _pool().submit(_run_convert, job.id, model.id)
    return job


def _run_convert(job_id: str, model_id: str) -> None:
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
            samples, rate = audio.decode_audio(Path(job.source_path))
            if samples.size / rate > MAX_CONVERT_SECONDS:
                raise TTSRequestError(f"audio is longer than {MAX_CONVERT_SECONDS // 60} minutes")
            model = usable_converter(session, model_id)
            converter = manager().get(model.id, model.architecture, Path(model.storage_path))
            allowed = {p.name for p in converter.get_supported_parameters()}
            result = converter.convert(samples, rate, **{k: v for k, v in job.params.items() if k in allowed})
            job.chunks = 1
            job.segments = [{"text": job.text, "start": 0.0, "end": round(result.samples.size / result.sample_rate, 3)}]
            _finish(job, audio.normalize_peak(result.samples), result.sample_rate, settings)
        except Exception as exc:  # recorded on the job, never swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
        job.generation_seconds = round(time.perf_counter() - started, 3)
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------- audio files

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
    paths = []
    if job.audio_path:
        paths += [Path(job.audio_path), Path(job.audio_path).with_suffix(".mp3")]
    if job.source_path:
        paths.append(Path(job.source_path))
    for path in paths:
        path.unlink(missing_ok=True)


def voice_parameters(model: VoiceModel) -> list[dict]:
    """Controls the studio may show for a voice or converter, straight from its adapter."""
    adapter = get_adapter(model.architecture)
    if adapter is None or not hasattr(adapter, "parameters_for"):
        return []
    details = (model.report.get("detection") or {}).get("details") or {}
    if model.kind == "voice_conversion":
        from ..engines.rvc.engine import RVCEngine

        has_index = bool(details.get("index_files"))
        methods = ("rmvpe", "fcpe") if adapter.is_available()[0] and _fcpe() else ("rmvpe",)
        specs = RVCEngine.parameters_for(has_index=has_index, pitch_methods=methods)
    else:
        specs = adapter.parameters_for(int(details.get("num_speakers") or 1))
    return [
        {"name": s.name, "label": s.label, "type": s.type, "default": s.default, "min": s.minimum, "max": s.maximum,
         "choices": list(s.choices)}
        for s in specs
    ]


def _fcpe() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchfcpe") is not None
