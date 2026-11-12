# VOIX

**Text in. Voice out.**

A self-hosted, provider-independent text-to-speech platform. VOIX is the voice
engine for SCOL: it takes text from any source (Gemini, Claude, GPT, Qwen, a
local LLM, a form field) and will speak it with voice models you own and run.

This repository is at **Phase 3: character voices and emotion**. Read "What
works today" before assuming anything else.

## What works today

| Area | Status |
| --- | --- |
| **Piper** TTS (CPU): Malayalam and English, WAV and MP3 | Working, tested with a real voice in CI |
| **RVC character voices** (`.pth` + `.index`, v1 and v2, with or without pitch) | Working. Runs after TTS so every line, Malayalam or English, gets the character's timbre. Needs the AI runtime |
| **Voice Changer**: upload an MP3/WAV/FLAC/OGG and re-voice it with an RVC model or a character | Working. Needs the AI runtime |
| **Qwen3-TTS** (CustomVoice, VoiceDesign, Base/cloning; 0.6B and 1.7B) | Adapter, detection and voices done. **Not run against real weights yet** (see below). English and 9 other languages, no Malayalam |
| **Emotion**: inline tags `[happy] … [calm] …`, auto detection per sentence, per-character defaults | Working. How audible it is depends on the engine (table below) |
| **Hugging Face installer**: paste a repo, pick RVC voices / Qwen models / Piper voices / ZIPs, install with progress | Working, tested with a mocked Hub. Token support for private and gated repos |
| Tutor API for SCOL: `/tts/speak` (audio back + emotion timeline), `/tts/batch`, `?wait=` on generate | Working. See [docs/SCOL_INTEGRATION.md](docs/SCOL_INTEGRATION.md) |
| Text pipeline, mixed Malayalam + English characters, jobs, cache, history, secure ZIP import | Working, tested |
| True audio streaming | **No.** Engines return whole sentences; responses say `"mode": "chunked"` |
| Rate limits, quotas, per-client keys, signed audio URLs | Not implemented |

Verified: 75 automated tests. The RVC tests run the real vendored RVC network
and conversion code with tiny random-weight models (so they prove loading,
shapes, timing and the API, not how a voice sounds). The light build runs 70
of them; the rest need the AI runtime.

**Not verified here** (the build machine could not reach Hugging Face): a real
community RVC checkpoint, the real HuBERT and RMVPE weights, and real
Qwen3-TTS weights. The code follows the upstream implementations exactly
(RVC's own pipeline, `qwen-tts` 0.1.1's API) and fails loudly if anything does
not fit, but the first real run is on your machine. Also not verified: how
anything *sounds*. That needs a person listening.

### How emotion is expressed

| Engine | What an emotion changes | Effect |
| --- | --- | --- |
| Qwen3-TTS 1.7B CustomVoice / VoiceDesign | A spoken-style instruction ("Speak with excitement…") plus the character's style | Clearly audible |
| Qwen3-TTS 0.6B, Qwen3-TTS Base | Speed only (the model ignores instructions) | Subtle |
| Piper | Speed and variability | Subtle |
| RVC stage of a character | Pitch per sentence (+2 semitones excited, −1.5 sad …) | Subtle to clear |

Each generated sentence reports its emotion, what chose it (`tag`, `classifier`,
`rules`, `default`, `request`) and what expressed it, with start/end times.

Automatic detection uses a Hugging Face classifier when installed
(`j-hartmann/emotion-english-distilroberta-base`, English; change with
`VOIX_EMOTION_MODEL`), otherwise cue-word rules for English and Malayalam.

### Limits worth knowing

- Qwen3-TTS has no Malayalam. Malayalam stays on Piper; give the character an
  RVC voice so both languages sound like the same person.
- On a laptop CPU: Piper is faster than real time, RVC is roughly real time,
  Qwen3-TTS is slow (seconds per sentence for 0.6B, much more for 1.7B). For
  lessons, pre-generate with `/tts/batch`. The first job after start-up also
  loads the models.
- Malayalam digits are read by Piper's phonemizer (espeak-ng), not by VOIX.
- Most community RVC voices declare no licence and many imitate real people.
  VOIX records the source and flags it; deciding is yours.
- The Malayalam starter voices have an unconfirmed licence (see
  [docs/ENGINES.md](docs/ENGINES.md)).

### Licence of the bundled engine

`piper-tts` is GPL-3.0-or-later and is bundled inside `VOIX.exe`. Running VOIX
as a server for SCOL is unaffected (the app talks to it over HTTP). Publishing
the exe means VOIX itself must be offered under GPL-compatible terms; this
repository does not yet declare a licence.

The AI runtime adds: RVC model code vendored in `backend/app/engines/rvc/infer_pack`
(MIT, licence file beside it), `qwen-tts` (Apache-2.0), PyTorch (BSD), Transformers
(Apache-2.0), FAISS (MIT), torchfcpe (MIT). Model weights carry their own licences:
Qwen3-TTS weights are Apache-2.0; RVC's HuBERT and RMVPE come from the MIT-licensed
RVC project; community voices usually declare nothing.

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

## The AI runtime (RVC, Qwen3-TTS, automatic emotion)

`VOIX.exe` is the light build: Piper only. **VOIX-AI** (a zip in the same
release, unzip and run `VOIX-AI\VOIX.exe`) adds PyTorch and runs RVC,
Qwen3-TTS and the emotion classifier on CPU or an NVIDIA GPU. From source:

```bash
pip install -r requirements.txt -r requirements-ai.txt
pip install --no-deps qwen-tts==0.1.1
```

Then, in the studio:

1. **Hugging Face** page → *Support models* → get *HuBERT* and *RMVPE* (needed
   by every RVC voice; fetched automatically on first use otherwise). Optionally
   the *emotion classifier*. Add a Hugging Face token if your repos are private.
2. Paste a repository, e.g. your RVC voice repo or
   `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, press *Look inside*, install what you need.
3. **Characters** → give a tutor its Malayalam and English voices, its RVC
   voice, a speaking style and emotion defaults.
4. **Voice Studio** to try it with `[happy]`-style tags, **Voice Changer** to
   re-voice an MP3.

`VOIX_DEVICE` (`auto`, `cpu`, `cuda`) chooses where PyTorch engines run.

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
| `VOIX_HF_TOKEN` / `HF_TOKEN` | unset | Hugging Face token (or save one in the studio) |
| `VOIX_MAX_HUB_BYTES` | 20 GB | Largest Hugging Face download |
| `VOIX_MAX_CONVERT_BYTES` | 100 MB | Largest audio upload for the Voice Changer |
| `VOIX_DEVICE` | `auto` | `cpu` or `cuda` for RVC and Qwen3-TTS |
| `VOIX_EMOTION_MODEL` | `j-hartmann/emotion-english-distilroberta-base` | Emotion classifier; empty turns it off |

## How an import is checked

1. The upload is streamed to a staging folder with a size cap and hashed (SHA-256). An identical archive is refused as a duplicate.
2. Every ZIP entry is validated: no absolute paths, no `..`, no symlinks, no encrypted entries, bounded count, bounded size, bounded compression ratio. Real bytes are counted while extracting, so a lying header cannot expand past the limit.
3. Only data files are extracted (weights, indexes, JSON/YAML, text, reference audio). `.py`, `.exe`, `.sh` and anything else are listed in the report and never written to disk.
4. Checkpoints are **never unpickled**. `pickletools.genops` decodes the opcode stream to recover tensor names and the globals the pickle would import. A checkpoint that references `os.system`, `subprocess`, `builtins.eval` and the like is rejected and nothing is installed.
5. The architecture is decided from tensor names and sidecar configs, with the evidence recorded. A `.pth` extension decides nothing.
6. RVC is labelled **voice conversion**. It has no text input and cannot do TTS alone; VOIX runs it after a TTS voice, or on uploaded audio.
7. Hugging Face installs go through the same checks. Only data files are downloaded (weights, indexes, configs, audio): never `.py` or executables, and models are never loaded with remote code. PyTorch checkpoints are loaded with `weights_only=True`.

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
  engines/rvc/                RVC adapter; infer_pack/ is vendored RVC model code (MIT)
  engines/qwen3_engine.py     Qwen3-TTS adapter (qwen-tts package)
  emotion/                    tags, detection, per-engine rendering
  hub.py                      Hugging Face client and background installs
  support.py                  HuBERT, RMVPE, emotion classifier downloads
  engines/manager.py          on-demand loading, LRU unload
  text/processing.py          normalisation, language runs, sentence chunks
  audio/processing.py         trim, join, level, WAV / MP3
  tts/service.py              generation pipeline, job queue, cache
  catalog.py                  starter voices
  api/routes.py               REST API v1
frontend/src/                 React + TypeScript + Vite + Tailwind studio
clients/python/voix_client.py one-file client (speak, batch) for SCOL tools
docs/                         architecture, engine recommendation, roadmap
```

See [docs/ENGINES.md](docs/ENGINES.md) for the Malayalam/English engine
recommendation and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design
and roadmap.
