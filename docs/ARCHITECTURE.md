# Architecture

```
LLM (any provider)  --text-->  VOIX API  --audio-->  SCOL mobile app (Expo)
                                  |
        +-------------+-----------+-----------+--------------+
        |             |                       |              |
   Model importer   Registries          Engine manager    Storage
   zip_safety       models / voices /   adapters behind   models, audio,
   checkpoint scan  characters          BaseTTSEngine     database
   detector
```

VOIX never calls an LLM. The only input is text plus a voice or character ID.

## Rules the code holds to

- **Characters are indirection.** The mobile app sends `character_id`. A
  character points at a Malayalam voice and an English voice; a voice points at
  a model. Any link can be swapped without an app release.
- **One adapter per architecture.** `engines/base.py` defines
  `BaseTTSEngine` (`load_model`, `unload_model`, `generate_speech`,
  `generate_stream`, `get_model_info`, `get_supported_languages`,
  `get_supported_parameters`, `validate_model`). Adding an engine is one module
  with `@register`. `supports_true_streaming` stays `False` unless the model
  emits audio while still synthesizing; everything else is served as
  sentence-chunked generation and labelled that way.
- **Parameters come from the adapter.** The studio renders only what
  `get_supported_parameters()` returns for the selected voice.
- **Untrusted files are data.** Nothing uploaded is executed or unpickled.
  Loading in Phase 2 goes through architecture-specific loaders
  (ONNX Runtime for Piper; `torch.load(weights_only=True)` for PyTorch).
- **Failures are loud.** Import rejections return the reason and are written to
  the audit log. Unimplemented endpoints return 501, not fake success.

## Roadmap

| Phase | Scope | State |
| --- | --- | --- |
| 1 Foundation | Backend, database, secure importer, inspection, detection, model library, studio shell | Done |
| 2 First engine | Piper adapter, text normalisation, mixed-script routing, sentence chunking, WAV/MP3, jobs, cache, history, starter voices | Done. Open: a listening review of Malayalam pronunciation |
| 3 Universal models | Indic Parler-TTS, GPT-SoVITS, VITS adapters; RVC post-stage; model lifecycle and VRAM management | |
| 4 Studio | Voice library, real generation controls, waveform player, history | |
| 5 API + SCOL | Jobs, batch, chunked streaming, API keys per client, rate limits, quotas, signed audio URLs, Expo test app | |
| 6 Production | Redis/Celery workers, PostgreSQL, S3 storage, Docker + NVIDIA toolkit, benchmarks, security review | |

## Known gaps

- Detection for RVC, GPT-SoVITS, VITS and XTTS is tested on structural
  fixtures, not on real community checkpoints.
- A single shared API key; no per-client keys, rate limiting or quotas.
- No ONNX graph inspection: an `.onnx` file without a Piper sidecar config is
  reported as unknown.
- Voice-cloning consent records are not modelled yet. Required before any
  custom voice is created.
