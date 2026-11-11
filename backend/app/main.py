"""VOIX API entrypoint: ``uvicorn app.main:app --reload``."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api.routes import router
from .config import get_settings
from .db import init_db


def create_app() -> FastAPI:
    settings = get_settings()
    init_db()
    app = FastAPI(
        title="VOIX",
        version=__version__,
        description="Self-hosted, provider-independent text-to-speech platform.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
