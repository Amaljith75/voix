"""Helper model files an engine needs that are not part of any voice.

They are downloaded once, on request, into ``<data>/runtime``.
"""
from __future__ import annotations

from pathlib import Path

from .catalog import _download
from .config import Settings
from .importer.service import ImportRejected

_RVC = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
BUNDLES: dict[str, dict] = {
    "rvc": {
        "label": "Voice changer runtime (ContentVec + RMVPE)",
        "size_mb": 355,
        "source": "https://huggingface.co/lj1995/VoiceConversionWebUI",
        "files": {
            "hubert_base/config.json": f"{_RVC}/hubert_base/config.json",
            "hubert_base/preprocessor_config.json": f"{_RVC}/hubert_base/preprocessor_config.json",
            "hubert_base/pytorch_model.bin": f"{_RVC}/hubert_base/pytorch_model.bin",
            "rmvpe.pt": f"{_RVC}/rmvpe.pt",
        },
    },
}


def bundle_dir(settings: Settings, bundle: str) -> Path:
    return settings.runtime_dir / bundle


def installed(settings: Settings, bundle: str) -> bool:
    folder = bundle_dir(settings, bundle)
    return all((folder / name).is_file() for name in BUNDLES[bundle]["files"])


def install(settings: Settings, bundle: str) -> None:
    if bundle not in BUNDLES:
        raise ImportRejected("unknown runtime bundle", status=404)
    folder = bundle_dir(settings, bundle)
    for name, url in BUNDLES[bundle]["files"].items():
        target = folder / name
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        try:
            _download(url, partial, 1024**3)
        except ImportRejected:
            raise
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise ImportRejected(f"could not download {name}: {exc}. Check the internet connection.", status=502) from exc
        partial.replace(target)


# ---------------------------------------------------------------- background installs

import threading  # noqa: E402

_state: dict[str, dict] = {}
_state_lock = threading.Lock()

QWEN_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
QWEN_SPEAKERS = {
    "ryan": "Male, native English", "aiden": "Male, native English",
    "serena": "Female, native Chinese", "vivian": "Female, native Chinese",
    "uncle_fu": "Male, native Chinese", "dylan": "Male, Beijing accent", "eric": "Male, Sichuan accent",
    "ono_anna": "Female, native Japanese", "sohee": "Female, native Korean",
}


def state(bundle: str) -> dict:
    with _state_lock:
        return dict(_state.get(bundle, {"state": "idle", "error": ""}))


def start(bundle: str, work) -> bool:
    """Run ``work()`` on a background thread. Returns False if it is already running."""
    with _state_lock:
        if _state.get(bundle, {}).get("state") == "installing":
            return False
        _state[bundle] = {"state": "installing", "error": ""}

    def run() -> None:
        try:
            work()
            outcome = {"state": "idle", "error": ""}
        except Exception as exc:  # surfaced through GET /runtime
            outcome = {"state": "error", "error": getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"}
        with _state_lock:
            _state[bundle] = outcome

    threading.Thread(target=run, name=f"voix-install-{bundle}", daemon=True).start()
    return True


def install_qwen(settings: Settings) -> None:
    """Download Qwen3-TTS from Hugging Face (public, Apache-2.0) and register its speakers as voices."""
    import uuid

    from huggingface_hub import snapshot_download

    from .db import Voice, VoiceModel, audit, new_session

    model_id = uuid.uuid4().hex
    folder = settings.models_dir / model_id
    try:
        snapshot_download(QWEN_REPO, local_dir=str(folder))
    except Exception as exc:
        import shutil

        shutil.rmtree(folder, ignore_errors=True)
        raise ImportRejected(f"could not download {QWEN_REPO}: {exc}", status=502) from exc
    files = [{"path": str(p.relative_to(folder)).replace("\\", "/"), "size": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file() and ".cache" not in p.parts]
    with new_session() as session:
        session.add(VoiceModel(
            id=model_id, name="Qwen3-TTS 0.6B CustomVoice", version=1, architecture="qwen3_tts", kind="tts",
            completeness="complete", safety="safe", active=True, size_bytes=sum(f["size"] for f in files),
            sha256=model_id, languages=["en"],
            dependencies=["qwen-tts", "torch", "transformers"], required_config=[], missing=[],
            device_support={"cpu": "slow (about 5x slower than real time on 2 cores)", "gpu": "recommended (CUDA)"},
            license="Apache-2.0", license_status="permissive",
            source_url=f"https://huggingface.co/{QWEN_REPO}", files=files, storage_path=str(folder),
            report={"detection": {
                "architecture_label": "Qwen3-TTS", "standalone_tts": True, "confidence": "high", "details": {},
                "evidence": [f"downloaded by VOIX from {QWEN_REPO}"],
                "language_note": "Qwen3-TTS speaks 10 languages. Malayalam is not one of them. "
                                 "VOIX registers its voices for English only.",
            }, "checkpoints": [], "skipped_files": [], "warnings": [], "original_filename": QWEN_REPO},
        ))
        for speaker, note in QWEN_SPEAKERS.items():
            voice_id = f"qwen_{speaker}"
            if not session.get(Voice, voice_id):
                session.add(Voice(id=voice_id, name=f"Qwen · {speaker.replace('_', ' ').title()}", model_id=model_id,
                                  description=f"Qwen3-TTS · {note} · emotion control", languages=["en"],
                                  settings={"speaker": speaker}))
        audit(session, "model.import", model_id=model_id, name="Qwen3-TTS 0.6B CustomVoice", architecture="qwen3_tts")
        session.commit()
