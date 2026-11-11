# VOIX

**Text in. Voice out.**

A self-hosted, provider-independent text-to-speech platform. VOIX is the voice
engine for SCOL: it takes text from any source (Gemini, Claude, GPT, Qwen, a
local LLM, a form field) and will speak it with voice models you own and run.

This repository is at **Phase 2 — first working engine**. Read "What works
today" before assuming anything else.

## What works today

| Area | Status |
| --- | --- |
| **Speech generation with Piper** (CPU, no GPU needed) | Working. Malayalam and English voices, WAV and MP3 |
| Starter voices: one-click download of 2 Malayalam and 2 English Piper voices | Working |
| Text pipeline: English numbers, decimals, maths symbols, units; Malayalam symbols and units; sentence chunking for long text | Working, tested |
| Mixed Malayalam + English: a character switches between its two voices by script | Working |
| Jobs API (`/tts/generate` -> `/tts/jobs/{id}` -> `/audio/{id}`), cache, generation history | Working |
| Studio: generate, waveform player, download, regenerate, per-model controls | Working |
| Secure ZIP import, safe checkpoint inspection, architecture detection | Working, tested |
| Characters with stable IDs and separate Malayalam / English voices | Working |
| True audio streaming | **No.** Piper returns whole sentences. Responses are labelled `"mode": "chunked"`. A streamed endpoint is Phase 5. |
| RVC, GPT-SoVITS, XTTS, Indic Parler-TTS, Chatterbox, Qwen3-TTS | **No adapter.** They are detected and labelled, not run. All need a GPU to be usable. |
| Batch jobs, rate limits, quotas, per-client keys, signed audio URLs | Not implemented (Phase 5) |

Verified: 55 automated tests, including real synthesis with a real Piper
Malayalam voice checked for duration, level and format. The Windows build runs
those tests, then starts `VOIX.exe`, downloads a voice, and generates Malayalam
speech before a release is published.

**Not verified: how it sounds.** Nobody has listened to the Malayalam output
and judged the pronunciation. That is a human check, and it decides whether
these voices are good enough for SCOL.

### Limits worth knowing

- Malayalam digits are read by Piper's phonemizer (espeak-ng), not by VOIX.
- English words inside Malayalam text are only pronounced well when a
  *character* with both voices is used; a single Malayalam voice will mangle them.
- The two English starter voices have American and Scottish accents. There is
  no Indian-English Piper voice in the catalog.
- The Malayalam starter voices have an unconfirmed licence (see
  [docs/ENGINES.md](docs/ENGINES.md)).

### Licence of the bundled engine

`piper-tts` is GPL-3.0-or-later and is bundled inside `VOIX.exe`. Running VOIX
as a server for SCOL is unaffected (the app talks to it over HTTP). Publishing
the exe means VOIX itself must be offered under GPL-compatible terms; this
repository does not yet declare a licence.

## Download for Windows

Get **VOIX.exe** from the [latest release](https://github.com/Amaljith75/voix/releases/latest)
and double-click it. A console window opens (that is the server) and the studio
opens in your browser at `http://127.0.0.1:8000`. Close the window to stop.

- Your models and database are kept in a `voix-data` folder next to the exe.
- Windows SmartScreen may warn because the exe is unsigned: More info -> Run anyway.
- To use it as a server for other devices on your Wi-Fi:
  `set VOIX_API_KEY=choose-a-secret` then `VOIX.exe --host 0.0.0.0`, and open
  `http://<this-PC's-IP>:8000` from the other device.

The exe is built by GitHub Actions (`.github/workflows/build-windows.yml`) on
every push to `main` that changes the app, and published as a new release.

## Run it from source

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
  engines/piper_engine.py     Piper adapter (ONNX Runtime, CPU)
  engines/manager.py          on-demand loading, LRU unload
  text/processing.py          normalisation, language runs, sentence chunks
  audio/processing.py         trim, join, level, WAV / MP3
  tts/service.py              generation pipeline, job queue, cache
  catalog.py                  starter voices
  api/routes.py               REST API v1
frontend/src/                 React + TypeScript + Vite + Tailwind studio
docs/                         architecture, engine recommendation, roadmap
```

See [docs/ENGINES.md](docs/ENGINES.md) for the Malayalam/English engine
recommendation and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design
and roadmap.
