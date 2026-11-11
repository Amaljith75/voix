# VOIX

**Text in. Voice out.**

A self-hosted, provider-independent text-to-speech platform. VOIX is the voice
engine for SCOL: it takes text from any source (Gemini, Claude, GPT, Qwen, a
local LLM, a form field) and will speak it with voice models you own and run.

This repository is at **Phase 1 — Foundation**. Read "What works today" before
assuming anything else.

## What works today

| Area | Status |
| --- | --- |
| FastAPI backend, SQLite (PostgreSQL via `VOIX_DATABASE_URL`) | Working, tested |
| Secure ZIP import (zip-slip, symlinks, bombs, size limits, no script extraction) | Working, tested |
| Safe checkpoint inspection: `.pth` / `.ckpt` scanned without unpickling | Working, tested |
| Architecture detection: Piper, RVC, GPT-SoVITS, VITS, XTTS, so-vits-svc, index-only, unknown | Working, tested |
| TTS vs voice-conversion labelling, completeness, dependencies, license flags | Working |
| Model library, versioning, activate / deactivate, delete, audit log | Working |
| Characters with stable IDs and separate Malayalam / English voice slots | Working (no voices to assign yet) |
| Studio web UI (9 sections), API playground, OpenAPI docs at `/api/docs` | Working |
| **Speech generation** | **Not implemented.** `POST /api/v1/tts/generate` returns 501. No engine adapter is installed, and VOIX does not return placeholder audio. |
| Streaming, batch jobs, history, caching, rate limits, quotas | Not implemented (Phases 2-6) |

Verified so far: 34 automated tests; a real Piper Malayalam voice
(`ml_IN-meera-medium`, 60 MB, from `rhasspy/piper-voices`) imported end to end
and identified as complete Piper TTS; a checkpoint written by real
`torch.save` scanned correctly. Detection of RVC / GPT-SoVITS / VITS / XTTS is
tested against structural fixtures that carry the real tensor names, **not yet
against a real community model** — send one ZIP from your Discord library to
close that gap.

## Run it

Requires Python 3.11+ and Node 20+.

```bash
# API  -> http://127.0.0.1:8000  (docs at /api/docs)
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

# Studio -> http://127.0.0.1:5173
cd frontend
npm install
npm run dev
```

Tests: `cd backend && pytest`.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VOIX_API_KEY` | unset | When set, every write needs an `X-API-Key` header. **Set it before exposing the API beyond localhost.** |
| `VOIX_DATA_DIR` | `backend/data` | Models, staging area, SQLite file |
| `VOIX_DATABASE_URL` | SQLite in data dir | e.g. `postgresql+psycopg://...` |
| `VOIX_MAX_UPLOAD_BYTES` | 2 GB | Upload cap |
| `VOIX_MAX_EXTRACTED_BYTES` | 4 GB | Extracted-size cap |
| `VOIX_CORS_ORIGINS` | Vite dev origins | Comma-separated |

## How an import is checked

1. The upload is streamed to a staging folder with a size cap and hashed (SHA-256). An identical archive is refused as a duplicate.
2. Every ZIP entry is validated: no absolute paths, no `..`, no symlinks, no encrypted entries, bounded count, bounded size, bounded compression ratio. Real bytes are counted while extracting, so a lying header cannot expand past the limit.
3. Only data files are extracted (weights, indexes, JSON/YAML, text, reference audio). `.py`, `.exe`, `.sh` and anything else are listed in the report and never written to disk.
4. Checkpoints are **never unpickled**. `pickletools.genops` decodes the opcode stream to recover tensor names and the globals the pickle would import. A checkpoint that references `os.system`, `subprocess`, `builtins.eval` and the like is rejected and nothing is installed.
5. The architecture is decided from tensor names and sidecar configs, with the evidence recorded. A `.pth` extension decides nothing.
6. RVC is labelled **voice conversion**. It has no text input and cannot do TTS alone.

## Layout

```
backend/app/
  security/zip_safety.py      safe extraction
  inspection/checkpoint.py    pickle opcode scanner (no torch needed)
  inspection/detector.py      architecture detection + evidence
  inspection/license.py       license detection from shipped files
  importer/service.py         import pipeline
  engines/base.py             BaseTTSEngine adapter contract
  engines/registry.py         adapter plugin registry
  hardware.py                 CPU / RAM / GPU / FFmpeg detection
  api/routes.py               REST API v1
frontend/src/                 React + TypeScript + Vite + Tailwind studio
docs/                         architecture, engine recommendation, roadmap
```

See [docs/ENGINES.md](docs/ENGINES.md) for the Malayalam/English engine
recommendation and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design
and roadmap.
