"""VOIX API entrypoint: ``uvicorn app.main:app --reload``."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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

    # Desktop / single-server mode: serve the built studio from the same port.
    frontend = os.environ.get("VOIX_FRONTEND_DIR")
    if frontend and (Path(frontend) / "index.html").is_file():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="studio")
    return app


app = create_app()
