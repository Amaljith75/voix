"""Safe extraction of untrusted model ZIPs.

Defends against path traversal (zip-slip), symlink entries, encrypted entries,
decompression bombs and oversized archives. Only data-file extensions on the
allowlist are written to disk; scripts and binaries inside an archive are
never extracted and never executed.
"""
from __future__ import annotations

import os
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ..config import Settings

ALLOWED_EXTENSIONS = {
    # weights / indexes
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".bin", ".index", ".npy", ".model",
    # configuration and docs
    ".json", ".yaml", ".yml", ".txt", ".md", ".csv", ".list",
    # reference audio (GPT-SoVITS / XTTS speaker prompts)
    ".wav", ".flac", ".mp3", ".ogg",
}
# Extensionless documentation files that carry license and provenance information.
ALLOWED_BARE_NAMES = {"license", "licence", "copying", "notice", "readme", "model_card"}
IGNORED_PARTS = {"__MACOSX", ".DS_Store", "Thumbs.db", "desktop.ini"}
_DRIVE = re.compile(r"^[A-Za-z]:")
_CHUNK = 1024 * 1024
_RATIO_MIN_SIZE = 10 * 1024 * 1024


class UnsafeArchive(Exception):
    """The archive must be rejected as a whole."""


@dataclass
class ExtractionResult:
    files: list[dict] = field(default_factory=list)  # {path, size}
    skipped: list[dict] = field(default_factory=list)  # {path, reason}

    @property
    def total_bytes(self) -> int:
        return sum(f["size"] for f in self.files)


def normalize_member(name: str) -> PurePosixPath:
    """Return a safe relative path for a ZIP member, or raise UnsafeArchive."""
    if "\x00" in name:
        raise UnsafeArchive(f"null byte in entry name: {name!r}")
    cleaned = name.replace("\\", "/")
    if cleaned.startswith("/") or _DRIVE.match(cleaned):
        raise UnsafeArchive(f"absolute path in archive: {name!r}")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        raise UnsafeArchive(f"empty entry name: {name!r}")
    if any(p == ".." for p in parts):
        raise UnsafeArchive(f"path traversal in archive: {name!r}")
    if any(len(p) > 255 for p in parts) or len(parts) > 12:
        raise UnsafeArchive(f"entry path too long or too deep: {name!r}")
    return PurePosixPath(*parts)


def safe_extract(zip_path: Path, dest: Path, settings: Settings) -> ExtractionResult:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    result = ExtractionResult()
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise UnsafeArchive("not a valid ZIP archive") from exc

    with zf:
        infos = zf.infolist()
        if len(infos) > settings.max_zip_entries:
            raise UnsafeArchive(f"too many entries ({len(infos)} > {settings.max_zip_entries})")
        if sum(i.file_size for i in infos) > settings.max_extracted_bytes:
            raise UnsafeArchive("declared uncompressed size exceeds the limit")

        total = 0
        for info in infos:
            if info.is_dir():
                continue
            rel = normalize_member(info.filename)
            if any(p in IGNORED_PARTS or p.startswith("._") for p in rel.parts):
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise UnsafeArchive(f"symlink entry rejected: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise UnsafeArchive(f"encrypted entry rejected: {info.filename!r}")
            allowed = rel.suffix.lower() in ALLOWED_EXTENSIONS or (
                not rel.suffix and rel.name.lower() in ALLOWED_BARE_NAMES
            )
            if not allowed:
                result.skipped.append(
                    {"path": str(rel), "reason": "file type not allowed; not extracted"}
                )
                continue
            if (
                info.file_size > _RATIO_MIN_SIZE
                and info.compress_size > 0
                and info.file_size / info.compress_size > settings.max_compression_ratio
            ):
                raise UnsafeArchive(f"suspicious compression ratio: {info.filename!r}")

            target = (root / rel).resolve()
            if not target.is_relative_to(root):
                raise UnsafeArchive(f"entry escapes extraction directory: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)

            written = 0
            with zf.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(_CHUNK):
                    written += len(chunk)
                    total += len(chunk)
                    # Never trust the header: count the real bytes.
                    if written > info.file_size or total > settings.max_extracted_bytes:
                        raise UnsafeArchive("archive expands beyond its declared size or the limit")
                    out.write(chunk)
            os.chmod(target, 0o640)
            result.files.append({"path": str(rel), "size": written})

    if not result.files:
        raise UnsafeArchive("archive contains no usable model files")
    return result
