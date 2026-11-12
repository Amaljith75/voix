# VOIX

**Text in. Voice out.**

A self-hosted, provider-independent text-to-speech platform. VOIX is the voice
engine for SCOL: it takes text from any source (Gemini, Claude, GPT, Qwen, a
local LLM, a form field) and will speak it with voice models you own and run.

This repository is at **Phase 3 — voice changer, emotions and a second engine**.

## What works today

| Area | Status |
| --- | --- |
| **Piper** text-to-speech (CPU) | Working. Malayalam and English, WAV and MP3 |
| **Voice changer (RVC)**: imported `.pth` + `.index` character models applied after any TTS voice | Working. Real conversion tested |
| **Audio-file voice changer**: upload MP3 / WAV / FLAC / OGG, get it back in a character's voice | Working |
| **Characters** with a Malayalam voice, an English voice and an optional voice changer | Working |
| **Qwen3-TTS** (optional download): 9 English voices with emotion control | Working, slow on CPU |
| **Emotion cues**: `emotion` field or `[laughing]`, `[sad]` in the text | Acted by Qwen3-TTS; skipped (not read aloud) by Piper |
| Text pipeline, jobs API, cache, history, secure import, architecture detection | Working, tested |
| Malayalam with emotion | **No.** No engine that runs here speaks Malayalam with emotion control |
| True audio streaming | **No.** Responses are `"mode": "chunked"` |
| GPT-SoVITS, XTTS, Indic Parler-TTS, Chatterbox | Detected, not run |

Verified: 66 automated tests, including real Piper synthesis and real RVC
conversion. The Windows build reruns them on Windows, then starts the packaged
app and generates speech through the voice changer and Qwen3-TTS before
publishing. **Not verified: how it sounds.** Pronunciation and voice quality
need a human listening check.

### Speed on a 2-core CPU without a GPU

| Pipeline | 5 s of speech takes |
| --- | --- |
| Piper | about 1 s |
| Piper + voice changer | about 6-10 s |
| Qwen3-TTS | about 25-30 s (plus loading the first time) |

Qwen3-TTS needs about 4.5 GB of free RAM. VOIX unloads the voice changer
before loading it, and refuses with a clear message if memory is still short.

See [docs/API.md](docs/API.md) for calling VOIX from the SCOL app.

### Licence of the bundled engine

`piper-tts` is GPL-3.0-or-later and is bundled inside `VOIX.exe`. Running VOIX
as a server for SCOL is unaffected (the app talks to it over HTTP). Publishing
the exe means VOIX itself must be offered under GPL-compatible terms; this
repository does not yet declare a licence.

## Download for Windows

From the [latest release](https://github.com/Amaljith75/voix/releases/latest):

- **VOIX-full-windows.zip** - everything (Piper, voice changer, Qwen3-TTS). Unzip, open the
  `VOIX` folder, double-click `VOIX.exe`.
- **VOIX.exe** - small single file, Piper voices only.

The voice changer files (355 MB) and Qwen3-TTS (2.4 GB) are downloaded from
Settings -> Optional downloads the first time you need them. A console window opens (that is the server) and the studio
opens in your browser at `http://127.0.0.1:8000`. Close the window to stop.

- Your models and database are kept in a `voix-data` folder next to the exe. To move to a new version, copy that folder next to the new `VOIX.exe`.
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

Optional engines from source: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements-rvc.txt` (voice changer) or `-r requirements-qwen.txt` (both).

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
  engines/qwen_engine.py      Qwen3-TTS adapter (optional)
  engines/rvc/                RVC voice changer (vendor/ is MIT code from RVC-Project)
  engines/manager.py          on-demand loading, LRU unload, memory guard
  vc/service.py               voice changer service, audio upload decoding
  runtime_assets.py           optional downloads (RVC runtime, Qwen3-TTS)
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
