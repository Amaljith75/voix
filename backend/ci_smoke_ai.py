"""CI smoke test for the packaged VOIX-AI app: RVC must really run inside the frozen build.

Usage: python ci_smoke_ai.py <data-dir-of-the-running-app> <api-base-url>

Installs a tiny random-weight HuBERT + RVC model (the same ones the tests use),
imports the RVC model through the API, converts a tone with FCPE, and checks
the result. Proves torch, transformers and the vendored RVC code work frozen;
says nothing about voice quality.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tests.test_v3 import build_tiny_rvc  # noqa: E402


def call(url: str, data: bytes | None = None, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - local test server
        return json.loads(resp.read())


def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for k, (name, data) in files.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    data_dir, api = Path(sys.argv[1]), sys.argv[2].rstrip("/")
    engines = {e["architecture"]: e for e in call(f"{api}/engines")}
    for arch in ("piper", "rvc", "qwen3_tts"):
        assert engines[arch]["ready"], f"{arch} not ready in the frozen app: {engines[arch]['reason']}"
    zipped = build_tiny_rvc(data_dir / "support")
    body, ctype = multipart({}, {"file": ("Tiny.zip", zipped)})
    model = call(f"{api}/models/import", body, {"Content-Type": ctype})
    assert model["architecture"] == "rvc" and model["inference"]["ready"], model

    sr = 22050
    t = np.arange(sr * 2) / sr
    wav = io.BytesIO()
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr, format="WAV")
    body, ctype = multipart({"model_id": model["id"], "f0_method": "fcpe", "output_format": "wav"},
                            {"file": ("tone.wav", wav.getvalue())})
    job = call(f"{api}/convert", body, {"Content-Type": ctype})
    for _ in range(300):
        job = call(f"{api}/tts/jobs/{job['job_id']}")
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(1)
    assert job["status"] == "completed", job
    assert abs(job["audio_seconds"] - 2.0) < 0.15 and job["sample_rate"] == 40000, job
    print(f"RVC conversion inside the frozen app: {job['audio_seconds']} s in {job['generation_seconds']} s")


if __name__ == "__main__":
    main()
