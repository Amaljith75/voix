"""Database setup: SQLite for local development, PostgreSQL via VOIX_DATABASE_URL."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class VoiceModel(Base):
    """An imported model package (one ZIP = one row = one version)."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    architecture: Mapped[str] = mapped_column(String(40), index=True)
    # tts | voice_conversion | unknown
    kind: Mapped[str] = mapped_column(String(30))
    # complete | incomplete | unknown
    completeness: Mapped[str] = mapped_column(String(20))
    # safe | caution
    safety: Mapped[str] = mapped_column(String(20))
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    languages: Mapped[list] = mapped_column(JSON, default=list)
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    required_config: Mapped[list] = mapped_column(JSON, default=list)
    missing: Mapped[list] = mapped_column(JSON, default=list)
    device_support: Mapped[dict] = mapped_column(JSON, default=dict)
    license: Mapped[str] = mapped_column(String(120), default="unknown")
    # permissive | conditional | noncommercial | unknown
    license_status: Mapped[str] = mapped_column(String(20), default="unknown")
    source_url: Mapped[str] = mapped_column(String(500), default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    files: Mapped[list] = mapped_column(JSON, default=list)
    report: Mapped[dict] = mapped_column(JSON, default=dict)
    storage_path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Voice(Base):
    """A selectable voice: a TTS-capable model plus speaker settings. Filled from Phase 2."""

    __tablename__ = "voices"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    model_id: Mapped[str] = mapped_column(String(32), index=True)
    languages: Mapped[list] = mapped_column(JSON, default=list)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Character(Base):
    """A tutor character. Its ID is stable for the mobile app; voices behind it can change."""

    __tablename__ = "characters"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    image_url: Mapped[str] = mapped_column(String(500), default="")
    voice_ml: Mapped[str | None] = mapped_column(String(80), nullable=True)
    voice_en: Mapped[str | None] = mapped_column(String(80), nullable=True)
    default_speed: Mapped[float] = mapped_column(Float, default=1.0)
    presets: Mapped[dict] = mapped_column(JSON, default=dict)
    #: How the character speaks, in words. Qwen3-TTS (1.7B / VoiceDesign) follows it.
    style: Mapped[str] = mapped_column(Text, default="")
    #: Optional RVC model that gives every line (Malayalam and English) this character's timbre.
    rvc_model_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rvc_settings: Mapped[dict] = mapped_column(JSON, default=dict)
    #: auto | none | <emotion>: what untagged text gets
    emotion_mode: Mapped[str] = mapped_column(String(20), default="auto")
    default_emotion: Mapped[str] = mapped_column(String(20), default="neutral")
    emotion_strength: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Generation(Base):
    """One synthesis job and its audio. Doubles as the generation history."""

    __tablename__ = "generations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # queued | processing | completed | failed
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    # tts | convert
    kind: Mapped[str] = mapped_column(String(12), default="tts")
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8))
    voice_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    character_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    model_ids: Mapped[list] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    output_format: Mapped[str] = mapped_column(String(8), default="mp3")
    cache_key: Mapped[str] = mapped_column(String(64), index=True)
    audio_path: Mapped[str] = mapped_column(String(500), default="")
    sample_rate: Mapped[int] = mapped_column(Integer, default=0)
    audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    generation_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    #: per chunk: text, language, emotion, how it was chosen, start/end seconds
    segments: Mapped[list] = mapped_column(JSON, default=list)
    #: conversion jobs: the uploaded source audio
    source_path: Mapped[str] = mapped_column(String(500), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    action: Mapped[str] = mapped_column(String(60), index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


_SessionLocal: sessionmaker[Session] | None = None


def init_db() -> None:
    global _SessionLocal
    settings = get_settings()
    settings.ensure_dirs()
    args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    engine = create_engine(settings.database_url, connect_args=args)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    _SessionLocal = sessionmaker(engine, expire_on_commit=False)


def _add_missing_columns(engine) -> None:
    """Additive migration: databases made by an older VOIX get the new columns.

    Only ever adds nullable / defaulted columns; never drops or rewrites data.
    """
    existing = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in inspect(engine).get_table_names()}
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            for column in table.columns:
                if table.name not in existing or column.name in existing[table.name]:
                    continue
                kind = column.type.compile(engine.dialect)
                default = None
                if column.default is not None:
                    arg = column.default.arg
                    default = arg(None) if callable(arg) else arg  # SQLAlchemy wraps callables to take a context
                if isinstance(column.type, JSON):
                    literal = "'{}'" if isinstance(default, dict) else "'[]'"
                elif isinstance(default, str):
                    literal = "'" + default.replace("'", "''") + "'"
                elif isinstance(default, bool):
                    literal = "1" if default else "0"
                elif isinstance(default, (int, float)):
                    literal = str(default)
                else:
                    literal = None
                clause = f" DEFAULT {literal}" if literal is not None else ""
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {kind}{clause}'))


def new_session() -> Session:
    """A session for background workers (caller closes it)."""
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    return _SessionLocal()


def get_session() -> Iterator[Session]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    with _SessionLocal() as session:
        yield session


def audit(session: Session, action: str, **detail) -> None:
    session.add(AuditLog(action=action, detail=detail))
