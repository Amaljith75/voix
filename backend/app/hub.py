"""Hugging Face Hub client: list a repository, download files, install models.

Standard library only (urllib), so it works in the light Piper-only build too.

* The token comes from ``VOIX_HF_TOKEN`` / ``HF_TOKEN`` or the one saved in
  Settings. It is sent to huggingface.co only: file downloads redirect to a CDN
  and the Authorization header is deliberately not forwarded there.
* Every download is streamed with a size cap and hashed.
* Downloaded packages go through the same inspection as a ZIP import
  (pickle scan, architecture detection, licence detection). Nothing is executed:
  no ``trust_remote_code``, and ``.py`` files are never downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .config import Settings, get_settings

HF = "https://huggingface.co"
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}/[\w.\-]{1,96}$")
#: Files VOIX will fetch from a repository. Code and executables never are.
DATA_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".index", ".onnx", ".json", ".yaml", ".yml", ".txt",
    ".model", ".tiktoken", ".wav", ".flac", ".mp3", ".ogg", ".zip", ".md",
}
NAMED_FILES = {"MODEL_CARD", "LICENSE", "LICENCE", "README", "NOTICE", "merges.txt", "vocab.json"}
_CHUNK = 1024 * 1024


class HubError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------- token

def _token_file(settings: Settings) -> Path:
    return settings.data_dir / "hf_token"


def get_token(settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    env = os.environ.get("VOIX_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if env:
        return env.strip()
    path = _token_file(settings)
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def token_status(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    source = "env" if (os.environ.get("VOIX_HF_TOKEN") or os.environ.get("HF_TOKEN")) else (
        "settings" if _token_file(settings).is_file() else None)
    return {"set": source is not None, "source": source}


def save_token(settings: Settings, token: str | None) -> None:
    path = _token_file(settings)
    if not token:
        path.unlink(missing_ok=True)
        return
    if not re.fullmatch(r"hf_[A-Za-z0-9]{20,}", token.strip()):
        raise HubError("that does not look like a Hugging Face token (it starts with hf_)", 422)
    path.write_text(token.strip(), encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


# ---------------------------------------------------------------- http

def parse_repo(raw: str) -> tuple[str, str, str]:
    """Accept 'owner/name', a full URL, or a URL to a folder. Returns (repo, revision, subfolder)."""
    raw = (raw or "").strip()
    revision, sub = "main", ""
    if raw.startswith(("http://", "https://")):
        parts = urllib.parse.urlparse(raw)
        if parts.netloc not in ("huggingface.co", "www.huggingface.co"):
            raise HubError("only huggingface.co links are supported", 422)
        bits = [b for b in parts.path.split("/") if b]
        if bits and bits[0] in ("models",):
            bits = bits[1:]
        if len(bits) < 2:
            raise HubError("link does not name a repository", 422)
        raw = "/".join(bits[:2])
        if len(bits) >= 4 and bits[2] in ("tree", "blob", "resolve"):
            revision = bits[3]
            sub = "/".join(bits[4:])
    if not REPO_ID.match(raw):
        raise HubError("repository must look like owner/name", 422)
    return raw, revision, sub


def _request(url: str, token: str | None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "VOIX"})
    if token and urllib.parse.urlparse(url).netloc == "huggingface.co":
        # unredirected: the CDN a download redirects to never sees the token
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def _open(url: str, token: str | None, timeout: int = 60):
    try:
        return urllib.request.urlopen(_request(url, token), timeout=timeout)  # noqa: S310 - fixed https host
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            hint = " Add a Hugging Face token in Settings." if not token else " Your token has no access to it."
            raise HubError(f"Hugging Face refused access ({exc.code}).{hint} Gated models must be accepted on their page first.", 403) from exc
        if exc.code == 404:
            raise HubError("not found on Hugging Face (check the repository name, revision and path)", 404) from exc
        raise HubError(f"Hugging Face returned HTTP {exc.code}", 502) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HubError(f"could not reach huggingface.co: {getattr(exc, 'reason', exc)}. Check the internet connection.", 502) from exc


def list_files(repo: str, revision: str = "main", token: str | None = None) -> list[dict]:
    """Every file in the repository: [{path, size}]."""
    url = f"{HF}/api/models/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true&expand=false"
    files: list[dict] = []
    for _ in range(50):  # pagination guard
        with _open(url, token) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            link = resp.headers.get("Link", "")
        for item in body:
            if item.get("type") == "file":
                size = (item.get("lfs") or {}).get("size", item.get("size", 0))
                files.append({"path": item["path"], "size": int(size or 0)})
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', link)
        if not nxt:
            break
        url = nxt.group(1)
    return files


def is_data_file(path: str) -> bool:
    p = PurePosixPath(path)
    if any(part.startswith(".") for part in p.parts):
        return False
    return p.suffix.lower() in DATA_SUFFIXES or p.name in NAMED_FILES


def download(repo: str, path: str, dest: Path, *, revision: str = "main", token: str | None = None,
             limit: int, progress=None, digest=None) -> int:
    url = f"{HF}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(path)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    tmp = dest.with_name(dest.name + ".part")
    with _open(url, token, timeout=120) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(_CHUNK):
            size += len(chunk)
            if size > limit:
                raise HubError(f"{path} is larger than the {limit // 1024**2} MB limit", 413)
            if digest is not None:
                digest.update(chunk)
            out.write(chunk)
            if progress:
                progress(len(chunk))
    tmp.replace(dest)
    return size


# ---------------------------------------------------------------- inspect

def _group(files: list[dict], sub: str) -> list[dict]:
    """Suggest installable packages inside a repository."""
    if sub:
        files = [f for f in files if f["path"] == sub or f["path"].startswith(sub.rstrip("/") + "/")]
    by_dir: dict[str, list[dict]] = {}
    for f in files:
        by_dir.setdefault(str(PurePosixPath(f["path"]).parent), []).append(f)
    paths = {f["path"] for f in files}
    out: list[dict] = []

    # Qwen3-TTS / any transformers-style folder that has its own speech tokenizer
    for d, items in by_dir.items():
        names = {PurePosixPath(i["path"]).name for i in items}
        prefix = "" if d == "." else d + "/"
        if "config.json" in names and any(p.startswith(prefix + "speech_tokenizer/") for p in paths):
            group = [f for f in files if f["path"].startswith(prefix) and is_data_file(f["path"])]
            out.append({"kind": "qwen3_tts", "label": f"Qwen3-TTS model{(' · ' + d) if prefix else ''}",
                        "files": [g["path"] for g in group], "size": sum(g["size"] for g in group)})

    for d, items in by_dir.items():
        # Piper voices
        for i in items:
            if i["path"].endswith(".onnx") and i["path"] + ".json" in paths:
                pair = [i, next(f for f in files if f["path"] == i["path"] + ".json")]
                out.append({"kind": "piper", "label": PurePosixPath(i["path"]).stem, "files": [p["path"] for p in pair],
                            "size": sum(p["size"] for p in pair)})
        # RVC: one .pth per character, with the .index beside it (prefer added_*)
        pths = [i for i in items if i["path"].lower().endswith(".pth")
                and not re.match(r"^(D|G)_\d+\.pth$|^(f0)?[DG]\d+k\.pth$", PurePosixPath(i["path"]).name)]
        indexes = [i for i in items if i["path"].lower().endswith(".index")]
        for pth in pths:
            stem = PurePosixPath(pth["path"]).stem.lower()
            idx = None
            if len(pths) == 1 and indexes:
                idx = sorted(indexes, key=lambda x: ("added" not in x["path"], -x["size"]))[0]
            else:
                idx = next((x for x in indexes if stem in x["path"].lower()), None)
            group = [pth] + ([idx] if idx else [])
            label = stem if d == "." else f"{d} · {stem}"
            out.append({"kind": "rvc", "label": label, "files": [g["path"] for g in group],
                        "size": sum(g["size"] for g in group), "has_index": bool(idx)})
        # Zipped packages (very common for RVC character voices)
        for i in items:
            if i["path"].lower().endswith(".zip"):
                out.append({"kind": "zip", "label": PurePosixPath(i["path"]).name, "files": [i["path"]], "size": i["size"]})
    return out


def inspect_repo(raw: str, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    repo, revision, sub = parse_repo(raw)
    token = get_token(settings)
    files = list_files(repo, revision, token)
    license_, gated = "unknown", False
    try:
        with _open(f"{HF}/api/models/{repo}", token) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
        license_ = (meta.get("cardData") or {}).get("license") or next(
            (t.split(":", 1)[1] for t in meta.get("tags", []) if t.startswith("license:")), "unknown")
        gated = bool(meta.get("gated"))
    except HubError:
        pass
    return {
        "repo_id": repo, "revision": revision, "subfolder": sub, "license": license_, "gated": gated,
        "files": [f | {"fetchable": is_data_file(f["path"])} for f in files],
        "candidates": _group(files, sub),
    }


# ---------------------------------------------------------------- background install tasks

_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="voix-hub")
    return _pool


def task(task_id: str) -> dict | None:
    with _tasks_lock:
        t = _tasks.get(task_id)
        return dict(t) if t else None


def tasks() -> list[dict]:
    with _tasks_lock:
        return sorted((dict(t) for t in _tasks.values()), key=lambda t: t["started"], reverse=True)[:30]


def _update(task_id: str, **changes) -> None:
    with _tasks_lock:
        _tasks[task_id].update(changes)


def start_task(label: str, total: int, work) -> dict:
    """Run ``work(report)`` in the background. ``report(n)`` adds downloaded bytes."""
    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {"id": task_id, "label": label, "status": "running", "done_bytes": 0,
                           "total_bytes": total, "result": None, "error": "", "started": time.time()}

    def report(n: int) -> None:
        with _tasks_lock:
            _tasks[task_id]["done_bytes"] += n

    def run() -> None:
        try:
            result = work(report)
            _update(task_id, status="completed", result=result)
        except Exception as exc:  # recorded on the task, never swallowed
            message = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
            _update(task_id, status="failed", error=message)

    _executor().submit(run)
    return task(task_id)  # type: ignore[return-value]


def fetch_files(repo: str, revision: str, paths: list[str], dest: Path, *, settings: Settings,
                progress=None) -> tuple[list[dict], str]:
    """Download the chosen files into ``dest`` keeping their relative paths. Returns (files, sha256)."""
    token = get_token(settings)
    digest = hashlib.sha256()
    out = []
    for path in sorted(paths):
        if not is_data_file(path):
            raise HubError(f"{path}: VOIX only downloads model data (weights, indexes, configs, audio), never code", 422)
        rel = PurePosixPath(path)
        if rel.is_absolute() or ".." in rel.parts:
            raise HubError(f"unsafe path {path}", 422)
        size = download(repo, path, dest / rel, revision=revision, token=token,
                        limit=settings.max_hub_bytes, progress=progress, digest=digest)
        out.append({"path": str(rel), "size": size})
    if sum(f["size"] for f in out) > settings.max_hub_bytes:
        raise HubError("selection is larger than VOIX_MAX_HUB_BYTES", 413)
    return out, digest.hexdigest()


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
