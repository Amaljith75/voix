"""Database setup: SQLite for local development, PostgreSQL via VOIX_DATABASE_URL."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, Integer, String, Text, create_engine
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


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
    _SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_session() -> Iterator[Session]:
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    with _SessionLocal() as session:
        yield session


def audit(session: Session, action: str, **detail) -> None:
    session.add(AuditLog(action=action, detail=detail))
