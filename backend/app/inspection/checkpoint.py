"""Inspect untrusted checkpoints WITHOUT deserializing them.

A PyTorch .pth/.pt/.ckpt file is a pickle, and unpickling runs arbitrary code.
This module never calls pickle.load or torch.load. It walks the pickle opcode
stream with ``pickletools.genops`` (which only decodes, it does not execute) to
recover two things:

* the strings in the pickle - dictionary keys and state-dict tensor names,
  which is what identifies the architecture, and
* the globals the pickle would import - which is what makes it dangerous.

No PyTorch install is needed for any of this.
"""
from __future__ import annotations

import json
import pickletools
import struct
import zipfile
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

MAX_PICKLE_BYTES = 256 * 1024 * 1024
MAX_STRINGS = 20000
_STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING", "STRING",
}

_SAFE_EXACT = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch._utils", "_rebuild_parameter"),
    ("torch._utils", "_rebuild_parameter_with_state"),
    ("torch.nn.parameter", "Parameter"),
    ("torch", "Size"),
    ("torch", "device"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
}
_SAFE_BUILTINS = {
    "set", "frozenset", "dict", "list", "tuple", "int", "float", "bool",
    "str", "bytes", "bytearray", "complex", "slice", "range",
}
_DANGEROUS_MODULES = {
    "os", "posix", "nt", "subprocess", "sys", "shutil", "socket", "runpy", "importlib",
    "pickle", "_pickle", "ctypes", "multiprocessing", "threading", "webbrowser", "http",
    "urllib", "requests", "pty", "commands", "code", "codeop", "operator", "functools",
    "platform", "tempfile", "asyncio", "signal", "zipimport", "pip", "torch.hub",
    "torch.serialization", "torch.storage",
}


@dataclass
class CheckpointReport:
    path: str
    format: str = "unknown"  # torch_zip | torch_legacy | safetensors | unknown
    strings: list[str] = field(default_factory=list)
    globals: list[str] = field(default_factory=list)
    unsafe_globals: list[str] = field(default_factory=list)
    unrecognized_globals: list[str] = field(default_factory=list)
    storage_count: int = 0
    verdict: str = "unknown"  # safe | caution | unsafe | unknown
    notes: list[str] = field(default_factory=list)

    def has(self, *needles: str) -> bool:
        """True if every needle occurs as a substring of some recovered string."""
        return all(any(n in s for s in self.strings) for n in needles)

    def summary(self) -> dict:
        d = asdict(self)
        d["string_count"] = len(self.strings)
        d["strings"] = self.strings[:60]
        return d


def _classify(module: str, name: str) -> str:
    if (module, name) in _SAFE_EXACT:
        return "safe"
    if module == "torch" and (name.endswith("Storage") or name.endswith("Tensor")):
        return "safe"
    if module in ("builtins", "__builtin__"):
        return "safe" if name in _SAFE_BUILTINS else "unsafe"
    top = module.split(".")[0]
    if module in _DANGEROUS_MODULES or top in _DANGEROUS_MODULES:
        return "unsafe"
    return "unrecognized"


def _scan_pickle(stream, report: CheckpointReport) -> None:
    """Decode one pickle from ``stream`` and record its strings and globals."""
    seen = set(report.strings)
    recent: deque[str] = deque(maxlen=2)
    memo: dict[int, str] = {}
    memo_next = 0
    last_string: str | None = None

    for opcode, arg, _pos in pickletools.genops(stream):
        name = opcode.name
        if name in _STRING_OPS and isinstance(arg, str):
            last_string = arg
            recent.append(arg)
            if arg not in seen and len(report.strings) < MAX_STRINGS:
                seen.add(arg)
                report.strings.append(arg)
            continue
        if name == "MEMOIZE":
            if last_string is not None:
                memo[memo_next] = last_string
            memo_next += 1
        elif name in ("BINPUT", "LONG_BINPUT", "PUT"):
            if last_string is not None:
                memo[int(arg)] = last_string
            memo_next = max(memo_next, int(arg) + 1)
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            if int(arg) in memo:
                recent.append(memo[int(arg)])
        elif name in ("GLOBAL", "INST"):
            module, _, attr = str(arg).partition(" ")
            _record_global(report, module, attr)
        elif name == "STACK_GLOBAL":
            if len(recent) == 2:
                _record_global(report, recent[0], recent[1])
            else:
                _record_global(report, "<unresolved>", "<unresolved>")
        elif name in ("BINPERSID", "PERSID"):
            report.storage_count += 1
        last_string = None


def _record_global(report: CheckpointReport, module: str, attr: str) -> None:
    ref = f"{module}.{attr}"
    if ref in report.globals:
        return
    report.globals.append(ref)
    kind = "unsafe" if module == "<unresolved>" else _classify(module, attr)
    if kind == "unsafe":
        report.unsafe_globals.append(ref)
    elif kind == "unrecognized":
        report.unrecognized_globals.append(ref)


class _PatchedHead:
    """File wrapper that presents the first two bytes as b'PK'.

    Some GPT-SoVITS releases overwrite the ZIP magic of their .pth files with a
    version marker. The rest of the container is a normal torch ZIP.
    """

    def __init__(self, raw):
        self._raw = raw

    def read(self, n: int = -1) -> bytes:
        pos = self._raw.tell()
        data = self._raw.read(n)
        if pos < 2 and data:
            head = b"PK"[pos:2]
            data = head[: len(data)] + data[len(head):]
        return data

    def seek(self, *a):
        return self._raw.seek(*a)

    def tell(self):
        return self._raw.tell()

    def seekable(self):
        return True


def _inspect_torch_zip(fh, report: CheckpointReport) -> None:
    report.format = "torch_zip"
    with zipfile.ZipFile(fh) as zf:
        pkl = [n for n in zf.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
        if not pkl:
            report.format = "unknown"
            report.notes.append("ZIP container without data.pkl; not a torch checkpoint")
            return
        info = zf.getinfo(pkl[0])
        if info.file_size > MAX_PICKLE_BYTES:
            report.notes.append("pickle too large to inspect")
            return
        with zf.open(info) as stream:
            _scan_pickle(stream, report)


def _inspect_torch_legacy(fh, report: CheckpointReport) -> None:
    """Pre-1.6 torch.save: several pickles back to back, then raw tensor data."""
    report.format = "torch_legacy"
    for _ in range(5):
        try:
            _scan_pickle(fh, report)
        except Exception:
            break


def _inspect_safetensors(fh, report: CheckpointReport) -> None:
    report.format = "safetensors"
    (length,) = struct.unpack("<Q", fh.read(8))
    if length > 100 * 1024 * 1024:
        report.notes.append("safetensors header too large")
        return
    header = json.loads(fh.read(length))
    report.strings = [k for k in header if k != "__metadata__"][:MAX_STRINGS]
    report.storage_count = len(report.strings)


def inspect_checkpoint(path: Path, rel: str | None = None) -> CheckpointReport:
    report = CheckpointReport(path=rel or path.name)
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(0)
            if path.suffix.lower() == ".safetensors":
                _inspect_safetensors(fh, report)
            elif head[:2] == b"PK":
                _inspect_torch_zip(fh, report)
            elif head[2:4] == b"\x03\x04":
                report.notes.append("non-standard ZIP magic (GPT-SoVITS style header)")
                _inspect_torch_zip(_PatchedHead(fh), report)
            elif head[:1] == b"\x80":
                _inspect_torch_legacy(fh, report)
            else:
                report.notes.append("unrecognized container; could not inspect")
    except Exception as exc:  # report, never hide
        report.notes.append(f"inspection failed: {type(exc).__name__}: {exc}")

    if report.unsafe_globals:
        report.verdict = "unsafe"
    elif report.format == "unknown":
        report.verdict = "unknown"
    elif report.unrecognized_globals:
        report.verdict = "caution"
    else:
        report.verdict = "safe"
    return report
