from __future__ import annotations

import json
import os
import struct

import pytest

from app.inspection.checkpoint import inspect_checkpoint
from app.inspection.detector import detect_package
from app.inspection.license import detect_license

from . import conftest as fx


class _Exploit:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _package(tmp_path, entries: dict[str, bytes | str]):
    files = []
    for name, data in entries.items():
        path = tmp_path / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        files.append({"path": name, "size": path.stat().st_size})
    return detect_package(tmp_path / "pkg", files)


# ------------------------------------------------------------ checkpoint scanner

def test_scanner_recovers_keys_without_executing(write):
    report = inspect_checkpoint(write("rvc.pth", fx.torch_zip_bytes(fx.RVC)))
    assert report.format == "torch_zip"
    assert report.verdict == "safe"
    assert "enc_p.emb_phone.weight" in report.strings
    assert {"weight", "sr", "f0", "version"} <= set(report.strings)


@pytest.mark.parametrize("protocol", [2, 4])
def test_malicious_pickle_is_flagged_and_never_runs(write, tmp_path, protocol):
    marker = tmp_path / "pwned"
    payload = {"weight": fx.RVC["weight"], "boom": _Exploit(marker)}
    report = inspect_checkpoint(write("evil.pth", fx.torch_zip_bytes(payload, protocol=protocol)))
    assert report.verdict == "unsafe"
    assert any("system" in g for g in report.unsafe_globals)
    assert not marker.exists()


def test_safetensors_header_is_read(write):
    header = json.dumps({"__metadata__": {}, "enc_p.emb.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    report = inspect_checkpoint(write("m.safetensors", struct.pack("<Q", len(header)) + header + b"\x00" * 4))
    assert report.format == "safetensors" and report.verdict == "safe"
    assert report.strings == ["enc_p.emb.weight"]


def test_gpt_sovits_patched_magic_is_handled(write):
    data = bytearray(fx.torch_zip_bytes(fx.SOVITS))
    data[:2] = b"05"
    report = inspect_checkpoint(write("s.pth", bytes(data)))
    assert report.format == "torch_zip" and report.has("enc_p.ssl_proj")


def test_garbage_is_unknown_not_safe(write):
    assert inspect_checkpoint(write("x.pth", b"\x00\x01garbage")).verdict == "unknown"


def test_real_torch_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "real.pth"
    torch.save({"weight": {"enc_p.emb_phone.weight": torch.zeros(2, 3), "dec.m_source.l_linear.weight": torch.ones(1)}, "sr": "40k", "version": "v2"}, path)
    report = inspect_checkpoint(path)
    assert report.verdict == "safe", report.globals
    assert report.storage_count == 2
    assert report.has("enc_p.emb_phone")


# ------------------------------------------------------------ architecture detection

def test_rvc_is_voice_conversion_not_tts(tmp_path):
    det, _ = _package(tmp_path, {"v.pth": fx.torch_zip_bytes(fx.RVC), "added_v2.index": b"\x00"})
    assert (det.architecture, det.kind, det.standalone_tts) == ("rvc", "voice_conversion", False)
    assert det.languages == []
    assert any("not text-to-speech" in w for w in det.warnings)
    assert det.details["index_files"] == ["added_v2.index"]


def test_gpt_sovits_needs_both_halves(tmp_path):
    det, _ = _package(tmp_path / "a", {"s.pth": fx.torch_zip_bytes(fx.SOVITS)})
    assert det.architecture == "gpt_sovits" and det.completeness == "incomplete" and not det.standalone_tts
    assert any("GPT weights" in m for m in det.missing)

    det, _ = _package(tmp_path / "b", {"s.pth": fx.torch_zip_bytes(fx.SOVITS), "g.ckpt": fx.torch_zip_bytes(fx.GPT)})
    assert det.completeness == "complete" and det.standalone_tts
    assert "ml" not in det.languages


def test_piper_reads_language_from_config(tmp_path):
    det, _ = _package(tmp_path, {"v.onnx": b"onnx", "v.onnx.json": json.dumps(fx.PIPER_CONFIG)})
    assert (det.architecture, det.kind, det.completeness) == ("piper", "tts", "complete")
    assert det.languages == ["ml"] and det.details["sample_rate"] == 22050


def test_vits_and_xtts(tmp_path):
    det, _ = _package(tmp_path / "v", {"G_1000.pth": fx.torch_zip_bytes(fx.VITS)})
    assert det.architecture == "vits" and det.completeness == "incomplete"

    det, _ = _package(tmp_path / "x", {
        "model.pth": fx.torch_zip_bytes(fx.XTTS), "vocab.json": "{}",
        "config.json": json.dumps({"model": "xtts", "languages": ["en", "hi", "zh-cn"]}),
    })
    assert det.architecture == "xtts" and det.completeness == "complete"
    assert "ml" not in det.languages
    assert any("non-commercial" in w for w in det.warnings)


def test_index_alone_and_unknown(tmp_path):
    det, _ = _package(tmp_path / "i", {"added.index": b"\x00"})
    assert det.architecture == "faiss_index" and det.kind == "unknown"

    det, _ = _package(tmp_path / "u", {"mystery.pth": fx.torch_zip_bytes({"foo": {"layer.weight": 0}})})
    assert det.architecture == "unknown" and not det.standalone_tts


def test_license_detection(tmp_path):
    root = tmp_path / "l"
    root.mkdir()
    (root / "README.md").write_text("license: cc-by-nc-4.0")
    assert detect_license(root, [{"path": "README.md", "size": 1}])[1] == "noncommercial"
    assert detect_license(root, [{"path": "model.pth", "size": 1}])[1] == "unknown"
