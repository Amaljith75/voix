from __future__ import annotations

import io
import stat
import zipfile
from dataclasses import replace

import pytest

from app.security.zip_safety import UnsafeArchive, normalize_member, safe_extract

from .conftest import make_zip


def _extract(write, tmp_path, settings, data: bytes):
    return safe_extract(write("in.zip", data), tmp_path / "out", settings)


@pytest.mark.parametrize("name", ["../evil.pth", "a/../../evil.pth", "/etc/passwd", "C:\\x.pth", "..\\x.pth", "a\x00b"])
def test_rejects_unsafe_member_names(name):
    with pytest.raises(UnsafeArchive):
        normalize_member(name)


def test_traversal_archive_is_rejected_and_writes_nothing_outside(write, tmp_path, settings):
    with pytest.raises(UnsafeArchive):
        _extract(write, tmp_path, settings, make_zip({"../escaped.pth": b"x"}))
    assert not (tmp_path / "escaped.pth").exists()


def test_symlink_entry_is_rejected(write, tmp_path, settings):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link.pth")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(UnsafeArchive, match="symlink"):
        _extract(write, tmp_path, settings, buf.getvalue())


def test_decompression_bomb_is_rejected(write, tmp_path, settings):
    bomb = make_zip({"big.pth": b"\x00" * (20 * 1024 * 1024)})
    assert len(bomb) < 200_000
    with pytest.raises(UnsafeArchive, match="compression ratio"):
        _extract(write, tmp_path, settings, bomb)


def test_size_and_entry_limits(write, tmp_path, settings):
    with pytest.raises(UnsafeArchive, match="too many entries"):
        _extract(write, tmp_path, replace(settings, max_zip_entries=2), make_zip({f"{i}.json": "{}" for i in range(3)}))
    with pytest.raises(UnsafeArchive, match="exceeds the limit"):
        _extract(write, tmp_path, replace(settings, max_extracted_bytes=10), make_zip({"a.json": "x" * 100}))


def test_scripts_and_binaries_are_never_extracted(write, tmp_path, settings):
    result = _extract(write, tmp_path, settings, make_zip({
        "model.pth": b"weights", "infer.py": "import os", "run.exe": b"MZ", "setup.sh": "rm -rf /",
        "__MACOSX/._model.pth": b"junk",
    }))
    assert [f["path"] for f in result.files] == ["model.pth"]
    assert {s["path"] for s in result.skipped} == {"infer.py", "run.exe", "setup.sh"}
    assert sorted(p.name for p in (tmp_path / "out").rglob("*") if p.is_file()) == ["model.pth"]


def test_not_a_zip(write, tmp_path, settings):
    with pytest.raises(UnsafeArchive, match="not a valid ZIP"):
        _extract(write, tmp_path, settings, b"definitely not a zip")
