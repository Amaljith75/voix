"""Audio assembly: trim, join, level, and encode to WAV / MP3."""
from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np


def trim_silence(samples: np.ndarray, sample_rate: int, threshold: float = 0.004, keep_ms: int = 15) -> np.ndarray:
    """Drop near-silent lead-in and tail so pauses between chunks are the ones we choose."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples[:0]
    keep = int(sample_rate * keep_ms / 1000)
    return samples[max(loud[0] - keep, 0): loud[-1] + keep + 1]


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resampling, used only when two voices in one response differ in rate."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    length = int(round(samples.size * target_rate / source_rate))
    positions = np.linspace(0, samples.size - 1, length)
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def silence(ms: int, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * ms / 1000), dtype=np.float32)


def fade_edges(samples: np.ndarray, sample_rate: int, ms: int = 4) -> np.ndarray:
    """Tiny fades so joined chunks never click."""
    n = min(int(sample_rate * ms / 1000), samples.size // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        samples = samples.copy()
        samples[:n] *= ramp
        samples[-n:] *= ramp[::-1]
    return samples


def normalize_peak(samples: np.ndarray, peak: float = 0.89) -> np.ndarray:
    """One gain for the whole response, so every chunk keeps the same loudness."""
    top = float(np.abs(samples).max()) if samples.size else 0.0
    return samples if top < 1e-6 else (samples * (peak / top)).astype(np.float32)


def to_int16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(to_int16(samples).tobytes())
    return buffer.getvalue()


def mp3_available() -> tuple[bool, str]:
    try:
        import lameenc  # noqa: F401
        return True, "lameenc"
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        return True, "ffmpeg"
    return False, "MP3 export needs the lameenc package or FFmpeg on PATH"


def wav_to_mp3(wav_bytes: bytes, bitrate_kbps: int = 128) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        pcm = wav.readframes(wav.getnframes())
    try:
        import lameenc
    except ImportError:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError(mp3_available()[1]) from None
        done = subprocess.run(
            [exe, "-loglevel", "error", "-f", "wav", "-i", "pipe:0", "-b:a", f"{bitrate_kbps}k", "-f", "mp3", "pipe:1"],
            input=wav_bytes, capture_output=True, timeout=300,
        )
        if done.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace')[:300]}")
        return done.stdout
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    encoder.silence()
    return bytes(encoder.encode(pcm)) + bytes(encoder.flush())
