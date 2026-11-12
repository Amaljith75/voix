"""Phase 3: emotion, characters with an RVC stage, voice conversion, Hugging Face, Qwen3-TTS.

The TTS engine here is a test adapter (``fake_tts``) that makes tones, so the
pipeline can be checked without downloading a voice. The RVC tests use the
real vendored RVC network and the real conversion code with a *tiny randomly
initialised* model and HuBERT: they prove loading, shapes, timing and the API,
not how a voice sounds. They are skipped when the AI runtime (torch) is absent.
"""
from __future__ import annotations

import importlib.util
import io
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from app import emotion, hub
from app.engines.base import AudioResult, BaseTTSEngine, ParameterSpec
from app.engines.registry import register

from . import conftest as fx
from .conftest import upload

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "scipy", "librosa", "torchfcpe"))
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="AI runtime (torch, transformers, torchfcpe) not installed")
CALLS: list[dict] = []


@register
class FakeTTS(BaseTTSEngine):
    """Speaks a tone, 60 ms per character. Follows instructions like Qwen 1.7B does."""

    architecture = "fake_tts"
    follows_instructions = True

    @classmethod
    def is_available(cls):
        return True, ""

    def load_model(self, model_dir, device="cpu"):
        pass

    def unload_model(self):
        pass

    def validate_model(self, model_dir):
        return []

    def generate_speech(self, text, language, **params):
        CALLS.append({"text": text, "language": language, **params})
        sr = 16000
        n = int(sr * 0.06 * max(len(text), 3) / float(params.get("speed", 1.0)))
        t = np.arange(n) / sr
        return AudioResult((0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    def get_model_info(self):
        return {}

    def get_supported_languages(self):
        return ["en", "ml"]

    def get_supported_parameters(self):
        return self.parameters_for()

    @staticmethod
    def parameters_for(num_speakers: int = 1):
        return [ParameterSpec("speed", "Speed", "float", 1.0, 0.5, 2.0),
                ParameterSpec("variability", "Variability", "float", 0.667, 0.2, 1.0)]


def add_fake_voice(voice_id: str = "fake", languages=("en", "ml")) -> str:
    from app.db import Voice, VoiceModel, new_session

    with new_session() as s:
        model = VoiceModel(name=voice_id, architecture="fake_tts", kind="tts", completeness="complete", safety="safe",
                           size_bytes=1, sha256=voice_id, storage_path="/nonexistent", languages=list(languages),
                           report={"detection": {"details": {}}})
        s.add(model)
        s.flush()
        s.add(Voice(id=voice_id, name=voice_id, model_id=model.id, languages=list(languages)))
        s.commit()
    return voice_id


def wait_job(client, job_id: str, seconds: float = 60) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/api/v1/tts/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


# ------------------------------------------------------------ emotion (unit)

def test_tags_split_text_and_unknown_brackets_stay():
    parts = emotion.parse_tags("[happy] Well done! [calm] Look at [1] the next. [pause 800ms] Ready? [Excited]Go!")
    assert [(p.text, p.emotion, p.pause_ms) for p in parts] == [
        ("Well done!", "happy", 0), ("Look at [1] the next.", "calm", 800), ("Ready?", "calm", 0), ("Go!", "excited", 0)]
    assert emotion.strip_tags("[cheerful] Hi [sad] bye [pause] end") == "Hi bye end"
    assert emotion.resolve("Gentle") == "calm" and emotion.resolve("banana") is None


def test_rules_read_cues_in_english_and_malayalam():
    found = [e for e, method, _ in emotion.detect(
        ["Great job, you got it!", "Oh no, that is not right.", "Not yet, and that is okay.", "Why do plants need light?",
         "Remember to carry the one.", "സാരമില്ല, ഒന്നുകൂടി നോക്കാം.", "അടിപൊളി!", "The sum is ten."],
        ["en"] * 5 + ["ml", "ml", "en"])]
    assert found == ["happy", "sad", "encouraging", "curious", "serious", "encouraging", "excited", "neutral"]


def test_render_scales_with_strength_and_carries_style():
    look = emotion.render("excited", 0.5, "a playful game host")
    assert look.speed == pytest.approx(1.05) and look.pitch == pytest.approx(1.0)
    assert "excitement" in look.instruct and "playful game host" in look.instruct
    assert emotion.render("excited", 0).speed == 1.0


# ------------------------------------------------------------ generation with emotion

def test_generate_with_tags_auto_and_segments(client):
    add_fake_voice()
    CALLS.clear()
    res = client.post("/api/v1/tts/generate?wait=30", json={
        "text": "Why is the sky blue? [pause 500ms] [excited] You did it! Remember the rule.",
        "voice_id": "fake", "emotion": "auto", "output_format": "wav"})
    job = res.json()
    assert job["status"] == "completed", job
    segs = job["segments"]
    assert [s["emotion"] for s in segs] == ["curious", "excited", "excited"]
    assert [s["source"] for s in segs] == ["rules", "tag", "tag"]
    assert segs[1]["rendered_by"][0] == "instruction"
    assert all(s["end"] > s["start"] for s in segs) and segs[1]["start"] - segs[0]["end"] >= 0.45
    assert "excitement" in CALLS[1]["instruct"] and CALLS[1]["speed"] == pytest.approx(1.1)
    assert CALLS[1]["variability"] > 0.667


def test_emotion_none_strips_tags_and_fixed_emotion_applies(client):
    add_fake_voice()
    CALLS.clear()
    job = client.post("/api/v1/tts/generate?wait=30", json={
        "text": "[happy] Hello there.", "voice_id": "fake", "emotion": "none", "output_format": "wav"}).json()
    assert job["segments"][0]["emotion"] == "neutral" and "instruct" not in CALLS[0]
    assert CALLS[0]["text"] == "Hello there."
    job = client.post("/api/v1/tts/generate?wait=30", json={
        "text": "Hello there.", "voice_id": "fake", "emotion": "calm", "output_format": "wav"}).json()
    assert job["segments"][0]["emotion"] == "calm" and job["segments"][0]["source"] == "request"
    bad = client.post("/api/v1/tts/generate", json={"text": "x", "voice_id": "fake", "emotion": "banana"})
    assert bad.status_code == 422


def test_speak_returns_audio_with_timeline_headers(client):
    add_fake_voice()
    res = client.post("/api/v1/tts/speak", json={"text": "[happy] Well done! Now try the next one.",
                                                 "voice_id": "fake", "output_format": "wav"})
    assert res.status_code == 200 and res.headers["content-type"] == "audio/wav"
    assert wav_seconds(res.content) > 1.0
    timeline = json.loads(res.headers["x-voix-segments"])
    # no emotion field: tags still count, untagged text stays neutral
    assert [t["emotion"] for t in timeline] == ["happy", "happy"] and res.headers["x-voix-job-id"]
    plain = client.post("/api/v1/tts/generate?wait=30", json={"text": "Great job!", "voice_id": "fake",
                                                              "output_format": "wav"}).json()
    assert plain["segments"][0]["emotion"] == "neutral" and plain["segments"][0]["source"] == "default"


def test_batch_queues_many_and_reports_rejects(client):
    add_fake_voice()
    res = client.post("/api/v1/tts/batch", json={"items": [
        {"text": "One.", "voice_id": "fake", "output_format": "wav"},
        {"text": "Two.", "voice_id": "fake", "output_format": "wav"},
        {"text": "Three.", "voice_id": "missing", "output_format": "wav"}]}).json()
    assert res["rejected"] == 1 and res["jobs"][2]["error"]
    for job in res["jobs"][:2]:
        assert wait_job(client, job["job_id"])["status"] == "completed"


def test_emotion_endpoints(client):
    body = client.get("/api/v1/emotions").json()
    assert "happy" in [e["id"] for e in body["emotions"]] and body["classifier"]["installed"] is False
    res = client.post("/api/v1/emotions/analyze", json={"text": "[calm] Breathe. Great job!", "emotion": "auto"}).json()
    assert [(s["emotion"], s["source"]) for s in res["segments"]] == [("calm", "tag"), ("calm", "tag")]


def test_character_with_scol_id_style_and_emotion_defaults(client):
    add_fake_voice()
    res = client.post("/api/v1/characters", json={
        "id": "minnu", "name": "Mio", "voice_en": "fake", "voice_ml": "fake",
        "style": "a warm, gentle storyteller talking to a young child", "emotion_mode": "auto",
        "default_emotion": "gentle", "emotion_strength": 0.5})
    assert res.status_code == 201, res.text
    assert res.json()["id"] == "minnu" and res.json()["default_emotion"] == "calm"
    CALLS.clear()
    job = client.post("/api/v1/tts/generate?wait=30", json={"text": "The answer is ten.", "character_id": "minnu",
                                                           "output_format": "wav"}).json()
    seg = job["segments"][0]
    assert seg["emotion"] == "calm" and seg["source"] == "default"
    assert "warm, gentle storyteller" in CALLS[0]["instruct"]
    assert client.post("/api/v1/characters", json={"name": "X", "rvc_model_id": "nope"}).status_code == 422


# ------------------------------------------------------------ RVC (real network code, tiny random weights)

TINY_RVC = [1025, 32, 64, 64, 128, 2, 2, 3, 0, "1", [3], [[1, 3, 5]], [10, 10, 2, 2], 64, [16, 16, 4, 4], 4, 32, 40000]


@pytest.fixture
def rvc_runtime(settings):
    return build_tiny_rvc(settings.support_dir)


def build_tiny_rvc(support_dir: Path) -> bytes:
    """Install a tiny HuBERT as the support model and return an RVC package as ZIP bytes."""
    import torch
    from torch import nn
    from transformers import HubertConfig, HubertModel

    from app.engines.rvc.infer_pack import models

    class H(HubertModel):
        def __init__(self, c):
            super().__init__(c)
            self.final_proj = nn.Linear(c.hidden_size, c.classifier_proj_size)

    sup = support_dir / "hubert_base"
    cfg = HubertConfig(hidden_size=768, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
                       conv_dim=(32,) * 7, classifier_proj_size=256)
    torch.manual_seed(0)
    H(cfg).save_pretrained(sup)
    (sup / "preprocessor_config.json").write_text(json.dumps({"do_normalize": False}))
    (sup / ".complete").write_text("ok")

    net = models.SynthesizerTrnMs768NSFsid(*TINY_RVC, is_half=False)
    weights = {k: v.half() for k, v in net.state_dict().items() if not k.startswith("enc_q")}
    buf = io.BytesIO()
    torch.save({"weight": weights, "config": TINY_RVC, "info": "1epoch", "sr": "40k", "f0": 1, "version": "v2"}, buf)
    return fx.make_zip({"Tiko/Tiko.pth": buf.getvalue()})


def tone_wav(seconds: float = 2.0, sr: int = 22050) -> bytes:
    import soundfile as sf

    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * k * (190 + 30 * np.sin(2 * np.pi * t)) * t) / k for k in range(1, 6)) * 0.2
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


@needs_torch
def test_rvc_import_is_ready_with_parameters(client, rvc_runtime):
    model = upload(client, rvc_runtime, "Tiko.zip").json()
    assert model["architecture"] == "rvc" and model["inference"]["ready"] is True
    listed = client.get("/api/v1/models?kind=voice_conversion").json()
    assert [p["name"] for p in listed[0]["parameters"]][:2] == ["pitch_shift", "protect"]


@needs_torch
def test_convert_uploaded_audio(client, rvc_runtime):
    model = upload(client, rvc_runtime, "Tiko.zip").json()
    res = client.post("/api/v1/convert", data={"model_id": model["id"], "pitch_shift": "3", "f0_method": "fcpe",
                                               "output_format": "wav"},
                      files={"file": ("lesson.wav", tone_wav(2.0), "audio/wav")})
    assert res.status_code == 202, res.text
    job = wait_job(client, res.json()["job_id"], 120)
    assert job["status"] == "completed", job.get("error")
    assert job["kind"] == "convert" and job["sample_rate"] == 40000
    audio = client.get(f"/api/v1/audio/{job['job_id']}?format=wav").content
    assert abs(wav_seconds(audio) - 2.0) < 0.1
    assert client.post("/api/v1/convert", data={"model_id": model["id"]},
                       files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 415


@needs_torch
def test_character_rvc_stage_follows_emotion_pitch(client, rvc_runtime):
    add_fake_voice()
    model = upload(client, rvc_runtime, "Tiko.zip").json()
    res = client.post("/api/v1/characters", json={
        "id": "tiko", "name": "Leo", "voice_en": "fake", "voice_ml": "fake", "rvc_model_id": model["id"],
        "rvc_settings": {"pitch_shift": 2, "f0_method": "fcpe"}, "emotion_mode": "auto"})
    assert res.status_code == 201, res.text
    job = client.post("/api/v1/tts/generate?wait=120", json={
        "text": "[excited] Boom! Correct! [calm] Now slowly, the next one.", "character_id": "tiko",
        "output_format": "wav"}).json()
    assert job["status"] == "completed", job.get("error")
    assert model["id"] in job["model_ids"]
    assert job["sample_rate"] == 40000
    assert job["segments"][0]["rendered_by"][-1] == "pitch" and job["segments"][0]["pitch"] > 0
    assert job["segments"][-1]["pitch"] < 0
    # deleting the RVC model unlinks it from the character
    assert client.delete(f"/api/v1/models/{model['id']}").status_code == 204
    assert client.get("/api/v1/characters").json()[0]["rvc_model_id"] is None


@needs_torch
def test_rvc_rejects_training_checkpoints(settings, tmp_path):
    import torch

    from app.engines.base import EngineError
    from app.engines.rvc.engine import RVCEngine

    torch.save({"model": {"enc_p.emb_phone.weight": torch.zeros(2)}, "iteration": 5}, tmp_path / "G_2333.pth")
    with pytest.raises(EngineError, match="training checkpoints"):
        RVCEngine().load_model(tmp_path)


# ------------------------------------------------------------ Hugging Face

def test_parse_repo_accepts_ids_and_links():
    assert hub.parse_repo("Owner/Voices") == ("Owner/Voices", "main", "")
    assert hub.parse_repo("https://huggingface.co/Owner/Voices/tree/main/Tutors/Leo") == ("Owner/Voices", "main", "Tutors/Leo")
    with pytest.raises(hub.HubError):
        hub.parse_repo("https://evil.example/Owner/Voices")
    with pytest.raises(hub.HubError):
        hub.parse_repo("../etc")


def test_group_suggests_rvc_pairs_zips_qwen_and_piper():
    files = [
        {"path": "Leo/Leo.pth", "size": 55_000_000}, {"path": "Leo/added_IVF1200_Flat_nprobe_1_Leo_v2.index", "size": 200_000_000},
        {"path": "Leo/total_fea.npy", "size": 1}, {"path": "Mio.zip", "size": 250_000_000},
        {"path": "pack/Ruby.pth", "size": 1}, {"path": "pack/Ollie.pth", "size": 1}, {"path": "pack/added_Ruby_v2.index", "size": 2},
        {"path": "pack/G_2333.pth", "size": 1},
        {"path": "qwen/config.json", "size": 1}, {"path": "qwen/model.safetensors", "size": 1},
        {"path": "qwen/speech_tokenizer/config.json", "size": 1},
        {"path": "en_US-x.onnx", "size": 1}, {"path": "en_US-x.onnx.json", "size": 1}, {"path": "app.py", "size": 1},
    ]
    groups = {(g["kind"], g["label"]): g for g in hub._group(files, "")}
    assert groups[("rvc", "Leo · leo")]["files"] == ["Leo/Leo.pth", "Leo/added_IVF1200_Flat_nprobe_1_Leo_v2.index"]
    assert groups[("rvc", "pack · ruby")]["has_index"] and not groups[("rvc", "pack · ollie")]["has_index"]
    assert ("rvc", "pack · g_2333") not in groups
    assert ("zip", "Mio.zip") in groups and ("piper", "en_US-x") in groups
    assert "qwen/speech_tokenizer/config.json" in groups[("qwen3_tts", "Qwen3-TTS model · qwen")]["files"]
    assert not hub.is_data_file("app.py") and not hub.is_data_file(".git/config") and hub.is_data_file("a/b.index")


def test_install_from_hub_runs_the_normal_inspection(client, settings, monkeypatch):
    rvc = fx.torch_zip_bytes(fx.RVC)
    blobs = {"Leo/Leo.pth": rvc, "Leo/added_Leo_v2.index": b"\0" * 64}

    def fake_list(repo, revision="main", token=None):
        return [{"path": p, "size": len(b)} for p, b in blobs.items()]

    def fake_download(repo, path, dest, *, revision="main", token=None, limit, progress=None, digest=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blobs[path])
        if digest is not None:
            digest.update(blobs[path])
        if progress:
            progress(len(blobs[path]))
        return len(blobs[path])

    monkeypatch.setattr(hub, "list_files", fake_list)
    monkeypatch.setattr(hub, "download", fake_download)
    res = client.post("/api/v1/hub/install", json={"repo_id": "someone/tutors", "files": list(blobs), "name": "Leo",
                                                   "license": "mit"})
    assert res.status_code == 202, res.text
    task_id = res.json()["id"]
    for _ in range(100):
        task = client.get(f"/api/v1/hub/tasks/{task_id}").json()
        if task["status"] != "running":
            break
        time.sleep(0.05)
    assert task["status"] == "completed", task["error"]
    model = client.get(f"/api/v1/models/{task['result']['model_id']}").json()
    assert model["architecture"] == "rvc" and model["name"] == "Leo"
    assert model["license"] == "mit" and model["license_status"] == "permissive"
    assert model["source_url"] == "https://huggingface.co/someone/tutors/tree/main"
    assert task["done_bytes"] == sum(len(b) for b in blobs.values())
    # the same files again are a duplicate
    again = client.post("/api/v1/hub/install", json={"repo_id": "someone/tutors", "files": list(blobs)}).json()
    for _ in range(100):
        t = client.get(f"/api/v1/hub/tasks/{again['id']}").json()
        if t["status"] != "running":
            break
        time.sleep(0.05)
    assert t["status"] == "failed" and "already installed" in t["error"]
    assert client.post("/api/v1/hub/install", json={"repo_id": "someone/tutors", "files": ["run.py"]}).status_code == 422


def test_token_is_stored_but_never_returned(client, settings, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("VOIX_HF_TOKEN", raising=False)
    assert client.get("/api/v1/hub/token").json() == {"set": False, "source": None}
    assert client.put("/api/v1/hub/token", json={"token": "nope"}).status_code == 422
    token = "hf_" + "a" * 34
    body = client.put("/api/v1/hub/token", json={"token": token}).json()
    assert body == {"set": True, "source": "settings"} and token not in json.dumps(client.get("/api/v1/system/config").json())
    assert hub.get_token(settings) == token
    req = hub._request("https://huggingface.co/api/models/x/y", token)
    assert req.unredirected_hdrs["Authorization"] == f"Bearer {token}" and "Authorization" not in req.headers
    assert "Authorization" not in hub._request("https://cdn-lfs.hf.co/file", token).unredirected_hdrs
    client.put("/api/v1/hub/token", json={"token": None})
    assert hub.get_token(settings) is None


def test_support_models_are_listed(client):
    ids = {s["id"]: s for s in client.get("/api/v1/support").json()}
    assert set(ids) == {"hubert_base", "rmvpe", "emotion"} and not ids["rmvpe"]["installed"]
    assert ids["emotion"]["repo"] == "j-hartmann/emotion-english-distilroberta-base"


# ------------------------------------------------------------ Qwen3-TTS detection

def safetensors_bytes(names: list[str]) -> bytes:
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names))


def qwen_zip(model_type: str = "custom_voice", size: str = "1b7", tokenizer: bool = True) -> bytes:
    config = {"model_type": "qwen3_tts", "tts_model_type": model_type, "tts_model_size": size,
              "talker_config": {"spk_id": {"ryan": 1, "aiden": 2} if model_type == "custom_voice" else {},
                                "codec_language_id": {"english": 1, "chinese": 2, "japanese": 3}}}
    entries = {
        "Qwen3-TTS/config.json": json.dumps(config), "Qwen3-TTS/model.safetensors": safetensors_bytes(["talker.w"]),
        "Qwen3-TTS/vocab.json": "{}", "Qwen3-TTS/merges.txt": "", "Qwen3-TTS/generation_config.json": "{}",
        "Qwen3-TTS/LICENSE": "Apache License\nVersion 2.0, January 2004",
    }
    if tokenizer:
        entries |= {"Qwen3-TTS/speech_tokenizer/config.json": json.dumps({"model_type": "qwen3_tts_tokenizer_12hz"}),
                    "Qwen3-TTS/speech_tokenizer/model.safetensors": safetensors_bytes(["decoder.w"])}
    return fx.make_zip(entries)


def test_qwen_custom_voice_import_creates_one_voice_per_speaker(client):
    model = upload(client, qwen_zip(), "Qwen3-TTS-12Hz-1.7B-CustomVoice.zip").json()
    assert model["architecture"] == "qwen3_tts" and model["completeness"] == "complete"
    assert model["languages"] == ["en", "ja", "zh"] and "Malayalam" in model["language_note"]
    voices = {v["id"]: v for v in client.get("/api/v1/voices").json()}
    assert {v["settings"]["speaker"] for v in voices.values()} == {"ryan", "aiden"}
    assert all(v["follows_instructions"] for v in voices.values())
    made = client.post("/api/v1/voices", json={"model_id": model["id"], "name": "Ryan slow", "settings": {"speaker": "Ryan"}})
    assert made.status_code == 201 and made.json()["settings"]["speaker"] == "ryan"
    bad = client.post("/api/v1/voices", json={"model_id": model["id"], "name": "x", "settings": {"speaker": "nobody"}})
    assert bad.status_code == 422


def test_qwen_small_model_is_flagged_and_incomplete_package_reported(client):
    small = upload(client, qwen_zip(size="0b6"), "small.zip").json()
    assert any("does not follow style instructions" in w for w in small["warnings"])
    broken = upload(client, qwen_zip(model_type="base", tokenizer=False), "broken.zip").json()
    assert broken["completeness"] == "incomplete" and any("speech_tokenizer" in m for m in broken["missing"])
    tok = upload(client, fx.make_zip({"config.json": json.dumps({"model_type": "qwen3_tts_tokenizer_12hz"}),
                                      "model.safetensors": safetensors_bytes(["x"])}), "tok.zip").json()
    assert tok["architecture"] == "qwen3_tts_tokenizer"


def test_qwen_base_voice_takes_a_reference_clip(client, settings):
    pytest.importorskip("soundfile")
    model = upload(client, qwen_zip(model_type="base"), "base.zip").json()
    assert client.get("/api/v1/voices").json() == []  # cloning model: no voice until a reference exists
    voice = client.post("/api/v1/voices", json={"model_id": model["id"], "name": "Teacher clone"}).json()
    res = client.post(f"/api/v1/voices/{voice['id']}/reference", data={"ref_text": "Hello children."},
                      files={"file": ("ref.wav", tone_wav(4.0), "audio/wav")})
    assert res.status_code == 200, res.text
    assert res.json()["settings"] == {"ref_text": "Hello children.", "has_reference": True}
    short = client.post(f"/api/v1/voices/{voice['id']}/reference", files={"file": ("r.wav", tone_wav(0.5), "audio/wav")})
    assert short.status_code == 422


def test_old_database_gets_new_columns(tmp_path, monkeypatch):
    import sqlite3

    from app.config import get_settings

    data = tmp_path / "old"
    data.mkdir()
    con = sqlite3.connect(data / "voix.db")
    con.execute("create table characters (id varchar(80) primary key, name varchar(200), description text, "
                "image_url varchar(500), voice_ml varchar(80), voice_en varchar(80), default_speed float, "
                "presets json, created_at datetime)")
    con.execute("insert into characters (id, name, description, image_url, default_speed, presets) "
                "values ('minnu', 'Mio', '', '', 1.0, '{}')")
    con.commit()
    con.close()
    monkeypatch.setenv("VOIX_DATA_DIR", str(data))
    get_settings.cache_clear()
    try:
        from app.db import Character, init_db, new_session

        init_db()
        init_db()  # idempotent
        with new_session() as s:
            c = s.get(Character, "minnu")
            assert (c.style, c.rvc_settings, c.emotion_mode, c.emotion_strength) == ("", {}, "auto", 1.0)
    finally:
        get_settings.cache_clear()
