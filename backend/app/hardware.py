"""Hardware detection without importing PyTorch."""
from __future__ import annotations

import platform
import shutil
import subprocess

import psutil


def _nvidia_gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "vram_total_mb": int(parts[1]), "vram_used_mb": int(parts[2])})
    return gpus


def detect_hardware() -> dict:
    mem = psutil.virtual_memory()
    gpus = _nvidia_gpus()
    ffmpeg = shutil.which("ffmpeg") is not None
    warnings = []
    if not gpus:
        warnings.append(
            "No NVIDIA GPU detected. CPU engines (Piper) will run in real time; "
            "GPU-class engines (GPT-SoVITS, XTTS, Indic Parler-TTS, RVC) will be slow or unusable."
        )
    if mem.total < 8 * 1024**3:
        warnings.append("Less than 8 GB RAM: load one model at a time.")
    if not ffmpeg:
        warnings.append("FFmpeg not found on PATH. MP3 export will be unavailable until it is installed.")
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(),
        "cpu_threads": psutil.cpu_count(),
        "ram_total_mb": mem.total // 1024**2,
        "ram_available_mb": mem.available // 1024**2,
        "gpus": gpus,
        "cuda_available": bool(gpus),
        "ffmpeg": ffmpeg,
        "recommended_device": "cuda" if gpus else "cpu",
        "warnings": warnings,
    }
