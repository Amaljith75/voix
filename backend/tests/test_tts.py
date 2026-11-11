"""Text pipeline, audio assembly and generation.

Tests marked ``real_voice`` synthesize speech with a genuine Piper voice. Point
VOIX_TEST_PIPER_DIR at a folder holding ``<voice>.onnx`` and ``<voice>.onnx.json``
to run them; they are skipped otherwise, never faked.
"""
from __future__ import annotations

import io
import os
import time
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.audio import processing as audio
from app.text.processing import chunk_text, detect_language, normalize, number_to_words, split_language_runs

from . import conftest as fx
from .conftest import upload

VOICE_DIR = os.environ.get("VOIX_TEST_PIPER_DIR")
real_voice = pytest.mark.skipif(not VOICE_DIR, reason="set VOIX_TEST_PIPER_DIR to a real Piper voice folder")


# ------------------------------------------------------------ text

@pytest.mark.parametrize("n,words", [
    (0, "zero"), (13, "thirteen"), (21, "twenty-one"), (100, "one hundred"), (101, "one hundred and one"),
    (1250, "one thousand two hundred and fifty"), (1000000, "one million"), (2005, "two thousand and five"),
])
def test_number_to_words(n, words):
    assert number_to_words(n) == words


def test_english_normalisation():
    out = normalize("Dr. Rao has 1,250 books. 3.14 is pi, 5 + 7 = 12, 40% of 25 km at 30°C.", "en")
    assert "Doctor Rao has one thousand two hundred and fifty books" in out
    assert "three point one four" in out and "five plus seven equals twelve" in out
    assert "forty percent" in out and "twenty-five kilometres" in out and "thirty degrees Celsius" in out
    assert not any(ch.isdigit() for ch in out)


def test_hyphen_is_only_a_minus_between_numbers():
    assert "nine minus four" in normalize("9 - 4", "en")
    assert "minus" not in normalize("a well-known fact - really", "en")


def test_malayalam_normalisation_keeps_digits_for_the_phonemizer():
    out = normalize("5 + 7 = 12. 25 km ദൂരം, 40% കുട്ടികൾ", "ml")
    assert out == "5 പ്ലസ് 7 സമം 12. 25 കിലോമീറ്റർ ദൂരം, 40 ശതമാനം കുട്ടികൾ"


def test_zero_width_joiner_survives():
    word = "അവന്‍"  # chillu written with ZWJ
    assert "‍" in normalize(word, "ml")


def test_language_detection_and_runs():
    assert detect_language("നമസ്കാരം കുട്ടികളേ hello") == "ml"
    assert detect_language("Hello children, നമസ്കാരം") == "en"
    runs = split_language_runs("നമസ്കാരം! Today we learn 3 things. ഇത് പ്രധാനമാണ്.", "ml")
    assert [r[0] for r in runs] == ["ml", "en", "ml"]
    assert runs[1][1] == "Today we learn 3 things."


def test_chunking_keeps_order_and_limits_length():
    text = normalize("First sentence. Second one! Third?\nNew paragraph " + "word, " * 200, "en")
    chunks = chunk_text(text, "en", mixed=False, max_chars=120)
    assert [c.text for c in chunks[:3]] == ["First sentence.", "Second one!", "Third?"]
    assert chunks[2].pause_ms > chunks[0].pause_ms  # paragraph break is longer
    assert all(len(c.text) <= 120 for c in chunks) and chunks[-1].pause_ms == 0
    assert "".join(c.text for c in chunks).count("word") == 200


def test_decimal_points_do_not_split_sentences():
    assert len(chunk_text("Pi is 3.14 roughly. Yes.", "ml", mixed=False)) == 2


# ------------------------------------------------------------ audio

def _tone(seconds=0.5, rate=22050, level=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32)


def test_trim_join_and_level():
    rate = 22050
    padded = np.concatenate([audio.silence(300, rate), _tone(), audio.silence(300, rate)])
    trimmed = audio.trim_silence(padded, rate)
    assert 0.5 <= trimmed.size / rate < 0.56
    assert audio.trim_silence(audio.silence(100, rate), rate).size == 0
    assert abs(float(np.abs(audio.normalize_peak(_tone(level=0.1))).max()) - 0.89) < 1e-3
    assert audio.resample(_tone(1.0, 22050), 22050, 16000).size == 16000


def test_wav_and_mp3_encoding():
    wav_bytes = audio.encode_wav(_tone(1.0), 22050)
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (22050, 1, 2, 22050)
    assert audio.mp3_available()[0]
    mp3 = audio.wav_to_mp3(wav_bytes)
    assert len(mp3) > 2000 and (mp3[:3] == b"ID3" or mp3[0] == 0xFF)


# ------------------------------------------------------------ generation API

def _wait(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/tts/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_request_validation(client):
    post = lambda body: client.post("/api/v1/tts/generate", json=body)  # noqa: E731
    assert post({"text": "hi"}).status_code == 422
    assert post({"text": "hi", "voice_id": "nope"}).status_code == 404
    assert post({"text": "hi", "character_id": "nope"}).status_code == 404
    client.post("/api/v1/characters", json={"name": "Empty"})
    assert post({"text": "hi", "character_id": "empty"}).status_code == 409
    assert post({"text": "hi", "voice_id": "x", "language": "fr"}).status_code == 422


def test_rvc_never_becomes_a_voice(client):
    upload(client, fx.rvc_zip(), "MyVoice.zip")
    assert client.get("/api/v1/voices").json() == []


def test_broken_model_fails_loudly_and_is_recorded(client):
    """A Piper-shaped package whose .onnx is not a real model must fail the job, not return audio."""
    upload(client, fx.piper_zip(), "ml_IN-test.zip")
    voices = client.get("/api/v1/voices").json()
    assert [v["id"] for v in voices] == ["ml_in_test"] and voices[0]["parameters"][0]["name"] == "speed"
    res = client.post("/api/v1/tts/generate", json={"text": "നമസ്കാരം", "voice_id": "ml_in_test"})
    assert res.status_code == 202
    job = _wait(client, res.json()["job_id"])
    assert job["status"] == "failed" and "EngineError" in job["error"]
    assert client.get(f"/api/v1/audio/{job['job_id']}").status_code == 404


def _install_real_voice(client) -> str:
    folder = Path(VOICE_DIR)  # type: ignore[arg-type]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path in folder.iterdir():
            if path.name.endswith((".onnx", ".onnx.json")):
                zf.write(path, path.name)
    model = upload(client, buf.getvalue(), "real-voice.zip").json()
    assert model["architecture"] == "piper" and model["inference"]["ready"]
    return client.get("/api/v1/voices").json()[0]["id"]


@real_voice
def test_real_generation_wav_mp3_cache_history(client, settings):
    voice_id = _install_real_voice(client)
    body = {"text": "നമസ്കാരം കുട്ടികളേ! ഇന്ന് നമുക്ക് ശാസ്ത്രം പഠിക്കാം.\nHello children.", "voice_id": voice_id, "output_format": "wav"}
    job = _wait(client, client.post("/api/v1/tts/generate", json=body).json()["job_id"])
    assert job["status"] == "completed", job
    assert job["chunks"] == 3 and job["mode"] == "chunked" and 2.0 < job["audio_seconds"] < 30

    wav_bytes = client.get(job["audio_url"]).content
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        assert wav.getframerate() == job["sample_rate"]
    assert abs(samples.size / job["sample_rate"] - job["audio_seconds"]) < 0.01
    assert 0.85 < np.abs(samples).max() / 32767 < 0.9 and np.sqrt(np.mean((samples / 32767.0) ** 2)) > 0.02

    mp3 = client.get(job["audio_url"], params={"format": "mp3", "download": True})
    assert mp3.headers["content-type"] == "audio/mpeg" and len(mp3.content) > 5000
    assert "attachment" in mp3.headers["content-disposition"]

    again = client.post("/api/v1/tts/generate", json=body).json()
    assert again["cached"] is True and again["job_id"] == job["job_id"]
    fresh = client.post("/api/v1/tts/generate", json={**body, "use_cache": False}).json()
    assert fresh["cached"] is False and fresh["job_id"] != job["job_id"]
    _wait(client, fresh["job_id"])

    slow = _wait(client, client.post("/api/v1/tts/generate", json={**body, "speed": 0.6}).json()["job_id"])
    assert slow["audio_seconds"] > job["audio_seconds"] * 1.2

    assert len(client.get("/api/v1/history").json()) == 3
    assert len(client.get("/api/v1/history", params={"q": "Hello"}).json()) == 3
    assert client.get("/api/v1/history", params={"q": "zzz"}).json() == []
    assert client.delete(f"/api/v1/history/{job['job_id']}").status_code == 204
    assert client.get(job["audio_url"]).status_code == 404
    assert not (settings.audio_dir / f"{job['job_id']}.wav").exists()


@real_voice
def test_character_routes_by_language_and_survives_model_delete(client):
    voice_id = _install_real_voice(client)
    client.post("/api/v1/characters", json={"name": "Teacher", "voice_ml": voice_id})
    res = client.post("/api/v1/tts/generate", json={"text": "Good morning", "character_id": "teacher", "language": "en"}).json()
    assert any("no en voice" in w for w in res["warnings"])
    assert _wait(client, res["job_id"])["status"] == "completed"

    model_id = client.get("/api/v1/voices").json()[0]["model_id"]
    assert client.delete(f"/api/v1/models/{model_id}").status_code == 204
    assert client.get("/api/v1/voices").json() == []
    assert client.get("/api/v1/characters").json()[0]["voice_ml"] is None
