"""Voice changer (RVC), emotion cues, optional engines and the database upgrade.

Tests marked ``real_rvc`` run a genuine RVC model. Set VOIX_TEST_RVC_DIR to a
folder holding an RVC ``.pth`` (and optionally its ``.index``) and
VOIX_TEST_RVC_ASSETS to a folder with ``hubert_base/`` and ``rmvpe.pt``. They
are skipped otherwise, never faked.
"""
from __future__ import annotations

import io
import os
import shutil
import sqlite3
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.engines.qwen_engine import style_instruction
from app.text.processing import split_directions, strip_stage_directions

from . import conftest as fx
from .conftest import upload
from .test_tts import VOICE_DIR, _install_real_voice, _wait

RVC_DIR = os.environ.get("VOIX_TEST_RVC_DIR")
RVC_ASSETS = os.environ.get("VOIX_TEST_RVC_ASSETS")
real_rvc = pytest.mark.skipif(not (RVC_DIR and RVC_ASSETS and VOICE_DIR), reason="set VOIX_TEST_RVC_DIR, VOIX_TEST_RVC_ASSETS and VOIX_TEST_PIPER_DIR")


# ------------------------------------------------------------ emotion cues

def test_split_directions():
    assert split_directions("Hello there.") == [("", "Hello there.")]
    assert split_directions("Hi. [laughing] So funny! [Sad] Oh no. (sighs) Fine.") == [
        ("", "Hi."), ("laughing", "So funny!"), ("sad", "Oh no."), ("sighs", "Fine."),
    ]
    assert split_directions("[laughing]") == []


def test_brackets_that_are_not_cues_survive():
    text = "Add 2 [two] apples, see note [1], and (a + b) too."
    assert strip_stage_directions(text)[1] == ["two"]  # a single bracketed word is treated as a cue
    assert "[1]" in strip_stage_directions(text)[0] and "(a + b)" in strip_stage_directions(text)[0]


def test_style_instruction():
    assert style_instruction("laughing") == "Speak while laughing."
    assert style_instruction("Sad") == "Speak in a sad tone."
    assert style_instruction("bursts out laughing") == "Speak in a bursts out laughing tone."
    assert style_instruction("Warm and encouraging, like a teacher.") == "Warm and encouraging, like a teacher."
    assert style_instruction("  ") == ""


# ------------------------------------------------------------ DSP pieces

def test_highpass_and_resample():
    converter = pytest.importorskip("app.engines.rvc.converter")
    pytest.importorskip("soxr")
    rate = 16000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    filtered = converter.highpass(tone + 0.5, rate)
    assert abs(float(filtered.mean())) < 1e-3
    assert np.abs(filtered - tone).max() < 0.05
    assert converter.resample(tone, rate, 40000).size == 40000


def test_mel_filterbank_shape_and_normalisation():
    pytest.importorskip("torch")
    from app.engines.rvc.vendor.rmvpe import mel_filterbank

    bank = mel_filterbank(16000, 1024, 128, 30, 8000)
    assert bank.shape == (128, 513) and (bank >= 0).all()
    assert all(bank[i].sum() > 0 for i in range(128))
    assert np.argmax(bank[0]) < np.argmax(bank[60]) < np.argmax(bank[127])


def test_rvc_checkpoint_cannot_run_code(tmp_path):
    """The loader used for RVC weights refuses anything but tensors and plain values."""
    torch = pytest.importorskip("torch")
    converter = pytest.importorskip("app.engines.rvc.converter")
    from .test_inspection import _Exploit

    marker = tmp_path / "pwned"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    torch.save({"weight": {}, "config": [], "x": _Exploit(marker)}, model_dir / "evil.pth")
    with pytest.raises(converter.RVCError, match="could not be loaded safely"):
        converter.RVCVoice(model_dir, converter.RVCRuntime(tmp_path))
    assert not marker.exists()


# ------------------------------------------------------------ API without models

def test_runtime_bundles_listed(client):
    bundles = {b["id"]: b for b in client.get("/api/v1/runtime").json()}
    assert set(bundles) == {"rvc", "qwen3"}
    assert bundles["rvc"]["installed"] is False and bundles["rvc"]["state"] == "idle"
    assert client.post("/api/v1/runtime/nope/install").status_code == 404


def test_voice_changer_validation(client):
    rvc = upload(client, fx.rvc_zip(), "MyVoice.zip").json()
    piper = upload(client, fx.piper_zip(), "ml_IN-test.zip").json()
    assert rvc["inference"]["ready"] is False  # runtime files are not downloaded in this test

    post = lambda body: client.post("/api/v1/tts/generate", json={"text": "hi", "voice_id": "ml_in_test", **body})  # noqa: E731
    assert post({"rvc": {"model_id": "missing"}}).status_code == 404
    assert post({"rvc": {"model_id": piper["id"]}}).status_code == 422
    not_ready = post({"rvc": {"model_id": rvc["id"]}})
    assert not_ready.status_code == 409

    assert client.post("/api/v1/characters", json={"name": "A", "rvc": {"model_id": piper["id"]}}).status_code == 422
    made = client.post("/api/v1/characters", json={"name": "B", "rvc": {"model_id": rvc["id"], "pitch": 40}})
    assert made.status_code == 201 and made.json()["rvc"]["pitch"] == 24.0  # clamped
    assert client.delete(f"/api/v1/models/{rvc['id']}").status_code == 204
    assert client.get("/api/v1/characters").json()[0]["rvc"] is None

    files = {"file": ("a.m4a", b"x", "audio/mp4")}
    assert client.post("/api/v1/vc/convert", files=files, data={"model_id": "x"}).status_code == 415
    files = {"file": ("a.wav", b"x", "audio/wav")}
    assert client.post("/api/v1/vc/convert", files=files, data={"model_id": "missing"}).status_code == 404


def test_old_database_is_upgraded(settings):
    """A database created by VOIX 0.2 gains the new columns and keeps its rows."""
    from app.db import init_db, new_session, Character, Generation

    settings.ensure_dirs()
    db = sqlite3.connect(settings.data_dir / "voix.db")
    db.executescript("""
        CREATE TABLE characters (id VARCHAR(80) PRIMARY KEY, name VARCHAR(200), description TEXT, image_url VARCHAR(500),
            voice_ml VARCHAR(80), voice_en VARCHAR(80), default_speed FLOAT, presets JSON, created_at DATETIME);
        INSERT INTO characters VALUES ('old', 'Old', '', '', NULL, NULL, 1.0, '{}', '2026-11-11 00:00:00');
        CREATE TABLE generations (id VARCHAR(32) PRIMARY KEY, status VARCHAR(20), text TEXT, language VARCHAR(8),
            voice_id VARCHAR(80), character_id VARCHAR(80), model_ids JSON, params JSON, output_format VARCHAR(8),
            cache_key VARCHAR(64), audio_path VARCHAR(500), sample_rate INTEGER, audio_seconds FLOAT,
            generation_seconds FLOAT, chunks INTEGER, warnings JSON, error TEXT, created_at DATETIME);
        INSERT INTO generations VALUES ('g1', 'completed', 'hi', 'en', 'v', NULL, '[]', '{}', 'wav', 'k', '', 22050, 1.0, 0.1, 1, '[]', '', '2026-11-11 00:00:00');
    """)
    db.commit()
    db.close()
    init_db()
    with new_session() as session:
        assert session.get(Character, "old").rvc is None
        job = session.get(Generation, "g1")
        assert job.kind is None and job.rvc is None and job.text == "hi"


def test_emotion_cues_are_not_read_aloud_by_piper(client):
    if not VOICE_DIR:
        pytest.skip("set VOIX_TEST_PIPER_DIR to a real Piper voice folder")
    voice_id = _install_real_voice(client)
    plain = _wait(client, client.post("/api/v1/tts/generate", json={"text": "Okay, listen.", "voice_id": voice_id, "language": "en"}).json()["job_id"])
    cued = _wait(client, client.post("/api/v1/tts/generate", json={"text": "[laughing] Okay, listen.", "voice_id": voice_id, "language": "en"}).json()["job_id"])
    assert cued["status"] == "completed" and any("emotion" in w for w in cued["warnings"])
    assert abs(cued["audio_seconds"] - plain["audio_seconds"]) < 0.4  # the cue added no spoken words


# ------------------------------------------------------------ real RVC

def _install_real_rvc(client, settings) -> str:
    shutil.copytree(RVC_ASSETS, settings.runtime_dir / "rvc", dirs_exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path in Path(RVC_DIR).iterdir():
            if path.suffix in (".pth", ".index"):
                zf.write(path, path.name)
    model = upload(client, buf.getvalue(), "real-rvc.zip").json()
    assert model["architecture"] == "rvc" and model["inference"]["ready"], model
    return model["id"]


def _wav(client, job) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(client.get(job["audio_url"], params={"format": "wav"}).content)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768, wav.getframerate()


@real_rvc
def test_real_tts_through_voice_changer_and_file_conversion(client, settings):
    rvc_id = _install_real_rvc(client, settings)
    voice_id = _install_real_voice(client)
    body = {"text": "നമസ്കാരം കുട്ടികളേ! ഇന്ന് നമുക്ക് ശാസ്ത്രം പഠിക്കാം.", "voice_id": voice_id, "output_format": "wav"}
    plain = _wait(client, client.post("/api/v1/tts/generate", json=body).json()["job_id"])
    changed = _wait(client, client.post("/api/v1/tts/generate", json={**body, "rvc": {"model_id": rvc_id, "pitch": 0}}).json()["job_id"], timeout=300)
    assert changed["status"] == "completed", changed
    assert changed["job_id"] != plain["job_id"] and changed["rvc"]["model_id"] == rvc_id
    assert abs(changed["audio_seconds"] - plain["audio_seconds"]) < 0.3  # same words, same timing
    samples, rate = _wav(client, changed)
    original, original_rate = _wav(client, plain)
    assert rate != original_rate or not np.allclose(samples[:1000], original[:1000])  # audibly different signal
    assert np.sqrt(np.mean(samples**2)) > 0.02 and 0.85 < np.abs(samples).max() < 0.9

    files = {"file": ("speech.wav", client.get(plain["audio_url"], params={"format": "wav"}).content, "audio/wav")}
    job = client.post("/api/v1/vc/convert", files=files, data={"model_id": rvc_id, "pitch": "4"})
    assert job.status_code == 202
    done = _wait(client, job.json()["job_id"], timeout=300)
    assert done["status"] == "completed" and done["kind"] == "conversion", done
    assert abs(done["audio_seconds"] - plain["audio_seconds"]) < 0.3
    assert [j["job_id"] for j in client.get("/api/v1/history", params={"kind": "conversion"}).json()] == [done["job_id"]]
    assert not any(settings.staging_dir.iterdir())  # the upload was removed

    bad = {"file": ("noise.mp3", b"this is not audio", "audio/mpeg")}
    failed = _wait(client, client.post("/api/v1/vc/convert", files=bad, data={"model_id": rvc_id}).json()["job_id"])
    assert failed["status"] == "failed" and "could not read" in failed["error"]
