"""Shared fixtures.

The checkpoint fixtures are structural stand-ins: real torch-format containers
whose pickles carry the same key and tensor names as real RVC / GPT-SoVITS /
VITS / XTTS checkpoints, with no weights. They exist to test detection and
safety. They are not models and nothing here tests audio output.
"""
from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def torch_zip_bytes(obj, protocol: int = 2) -> bytes:
    """A torch.save-style ZIP container around ``obj`` (no tensors)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(obj, protocol=protocol))
        zf.writestr("archive/version", "3\n")
    return buf.getvalue()


def state(*keys: str) -> dict:
    return {k: 0 for k in keys}


RVC = {
    "weight": state(
        "enc_p.emb_phone.weight", "enc_p.emb_pitch.weight", "dec.m_source.l_linear.weight",
        "flow.flows.0.pre.weight", "emb_g.weight",
    ),
    "config": [1025, 32, 192], "info": "300epoch", "sr": "40k", "f0": 1, "version": "v2",
}
SOVITS = {
    "weight": state("enc_p.ssl_proj.weight", "quantizer.vq.layers.0._codebook.embed", "ref_enc.spectral.0.fc.weight", "dec.conv_pre.weight"),
    "config": {"data": {"sampling_rate": 32000}}, "info": "e8",
}
GPT = {
    "weight": state("model.ar_text_embedding.word_embeddings.weight", "model.ar_audio_embedding.word_embeddings.weight", "model.h.layers.0.linear1.weight"),
    "config": {"model": {"n_layer": 24}}, "info": "GPT-e15",
}
VITS = {
    "model": state("enc_p.emb.weight", "dp.flows.0.logs", "dec.conv_pre.weight", "flow.flows.0.pre.weight"),
    "iteration": 1000, "learning_rate": 0.0002,
}
XTTS = {"model": state("gpt.gpt.h.0.attn.c_attn.weight", "hifigan_decoder.waveform_decoder.conv_pre.weight")}

PIPER_CONFIG = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "espeak": {"voice": "ml"},
    "language": {"code": "ml_IN"},
    "num_speakers": 1,
    "phoneme_id_map": {"_": [0], "a": [14]},
    "piper_version": "1.0.0",
}


def make_zip(entries: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def rvc_zip() -> bytes:
    return make_zip({
        "MyVoice/MyVoice.pth": torch_zip_bytes(RVC),
        "MyVoice/added_IVF256_Flat_nprobe_1_MyVoice_v2.index": b"\x00" * 64,
    })


def piper_zip() -> bytes:
    return make_zip({
        "ml_IN-test-medium.onnx": b"\x08\x07fake-onnx-bytes",
        "ml_IN-test-medium.onnx.json": json.dumps(PIPER_CONFIG),
        "LICENSE": "Apache License\nVersion 2.0, January 2004",
    })


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("VOIX_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("VOIX_API_KEY", raising=False)
    monkeypatch.delenv("VOIX_DATABASE_URL", raising=False)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def client(settings):
    from app.main import create_app

    return TestClient(create_app())


def upload(client: TestClient, data: bytes, filename: str = "model.zip", **kw):
    return client.post("/api/v1/models/import", files={"file": (filename, data, "application/zip")}, **kw)


@pytest.fixture
def write(tmp_path):
    def _write(name: str, data: bytes) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    return _write
