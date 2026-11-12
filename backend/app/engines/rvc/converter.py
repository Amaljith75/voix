"""RVC voice conversion on CPU: speech in, the same speech in another timbre out.

RVC has no text input. VOIX uses it in two ways: as a second stage after a
real TTS engine, and to re-voice an uploaded audio file.

The network definitions under ``vendor/`` come from RVC-Project (MIT). The
pipeline below follows the reference implementation, minus half precision,
CUDA graphs and the scipy/librosa dependencies.

Checkpoints are loaded with ``weights_only=True``, so a .pth can only supply
tensors and plain values; it cannot run code.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

SAMPLE_RATE_IN = 16000
WINDOW = 160  # samples per content frame at 16 kHz
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 5, 30, 32  # seconds; the reference low-memory profile
SR_NAMES = {"32k": 32000, "40k": 40000, "48k": 48000}


class RVCError(RuntimeError):
    pass


def available() -> tuple[bool, str]:
    missing = []
    for module in ("torch", "transformers", "soxr"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        from ...config import missing_feature_hint

        return False, missing_feature_hint("Voice changer needs: " + ", ".join(missing) + " (pip install -r requirements-rvc.txt)")
    return True, ""


def highpass(audio: np.ndarray, sample_rate: int, cutoff: float = 48.0) -> np.ndarray:
    """Zero-phase removal of DC and rumble below ``cutoff`` Hz."""
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    spectrum *= np.clip((freqs - cutoff / 2) / (cutoff / 2), 0.0, 1.0)
    return np.fft.irfft(spectrum, n=audio.size).astype(np.float32)


def resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return audio.astype(np.float32)
    import soxr

    return soxr.resample(audio.astype(np.float32), source, target).astype(np.float32)


class RVCRuntime:
    """The two shared helper networks: ContentVec/HuBERT (what is said) and RMVPE (pitch)."""

    def __init__(self, assets_dir: Path, device: str = "cpu") -> None:
        self.assets_dir = assets_dir
        self.device = device
        self._hubert = None
        self._rmvpe = None
        self._lock = threading.Lock()

    def hubert(self):
        with self._lock:
            if self._hubert is None:
                import torch
                from torch import nn
                from transformers import HubertModel

                class HubertModelWithFinalProj(HubertModel):
                    def __init__(self, config):
                        super().__init__(config)
                        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

                folder = self.assets_dir / "hubert_base"
                if not (folder / "config.json").is_file():
                    raise RVCError("voice-conversion runtime files are not installed")
                model = HubertModelWithFinalProj.from_pretrained(
                    str(folder), local_files_only=True, torch_dtype=torch.float32
                )
                self._hubert = model.to(self.device).eval()
            return self._hubert

    def rmvpe(self):
        with self._lock:
            if self._rmvpe is None:
                import torch

                from .vendor.rmvpe import RMVPE

                path = self.assets_dir / "rmvpe.pt"
                if not path.is_file():
                    raise RVCError("voice-conversion runtime files are not installed")
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._rmvpe = RMVPE(state, device=self.device)
            return self._rmvpe

    def features(self, audio16k: np.ndarray, version: str):
        import torch

        model = self.hubert()
        with torch.no_grad():
            source = torch.from_numpy(audio16k).float().view(1, -1).to(self.device)
            if version == "v1":
                out = model(input_values=source, output_hidden_states=True, return_dict=True)
                return model.final_proj(out.hidden_states[9])
            return model(input_values=source, return_dict=True).last_hidden_state


def find_rvc_files(model_dir: Path) -> tuple[Path | None, Path | None]:
    # G_*.pth / D_*.pth are training checkpoints (no inference config); they cannot be used directly.
    weights = sorted(p for p in model_dir.rglob("*.pth") if not p.name.startswith(("D_", "G_")))
    indexes = sorted(model_dir.rglob("*.index"), key=lambda p: (not p.name.startswith("added"), p.name))
    return (weights[0] if weights else None), (indexes[0] if indexes else None)


class RVCVoice:
    """One imported RVC model (.pth, optionally with its .index)."""

    def __init__(self, model_dir: Path, runtime: RVCRuntime) -> None:
        import torch

        from .vendor import models

        self.runtime = runtime
        weights, self.index_path = find_rvc_files(model_dir)
        if weights is None:
            if any(model_dir.rglob("G_*.pth")):
                raise RVCError(
                    "this package only has a training checkpoint (G_*.pth). Export the small inference "
                    ".pth from RVC (the one named after the voice) and import that instead."
                )
            raise RVCError("no RVC .pth file in this model")
        try:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RVCError(f"{weights.name} could not be loaded safely: {exc}") from exc
        if not isinstance(ckpt, dict) or "weight" not in ckpt or "config" not in ckpt:
            raise RVCError(f"{weights.name} is not an RVC inference checkpoint (no 'weight' / 'config')")

        config = list(ckpt["config"])
        config[-3] = ckpt["weight"]["emb_g.weight"].shape[0]  # number of speakers
        self.target_rate = SR_NAMES.get(config[-1], config[-1]) if isinstance(config[-1], str) else int(config[-1])
        self.f0 = int(ckpt.get("f0", 1)) == 1
        self.version = str(ckpt.get("version", "v1"))
        cls = {
            ("v1", True): models.SynthesizerTrnMs256NSFsid,
            ("v1", False): models.SynthesizerTrnMs256NSFsid_nono,
            ("v2", True): models.SynthesizerTrnMs768NSFsid,
            ("v2", False): models.SynthesizerTrnMs768NSFsid_nono,
        }.get((self.version, self.f0))
        if cls is None:
            raise RVCError(f"unsupported RVC version '{self.version}'")
        net = cls(*config, is_half=False)
        del net.enc_q  # training-only posterior encoder
        net.load_state_dict(ckpt["weight"], strict=False)
        self.net = net.eval().float().to(runtime.device)
        self._index = None
        self._index_vectors = None
        self.index_error = ""

    def _load_index(self) -> None:
        if self._index is not None or self.index_path is None or self.index_error:
            return
        try:
            import faiss

            self._index = faiss.read_index(str(self.index_path))
            self._index_vectors = self._index.reconstruct_n(0, self._index.ntotal)
        except Exception as exc:  # reported to the caller as a warning, conversion continues
            self._index = self._index_vectors = None
            self.index_error = f"index not used: {type(exc).__name__}: {exc}"

    def info(self) -> dict:
        return {
            "version": self.version, "f0": self.f0, "sample_rate": self.target_rate,
            "index": self.index_path.name if self.index_path else None,
        }

    # ------------------------------------------------------------------ conversion

    def convert(self, audio: np.ndarray, sample_rate: int, *, pitch: float = 0.0, index_rate: float = 0.5,
                protect: float = 0.33, speaker: int = 0) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample rate) in this model's voice."""
        import torch

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = resample(np.asarray(audio, dtype=np.float32), sample_rate, SAMPLE_RATE_IN)
        if audio.size < SAMPLE_RATE_IN // 10:
            raise RVCError("audio is too short to convert")
        peak = float(np.abs(audio).max())
        if peak > 1.0:
            audio = audio / peak
        audio = highpass(audio, SAMPLE_RATE_IN)

        use_index = index_rate > 0 and self.index_path is not None
        if use_index:
            self._load_index()

        t_pad = SAMPLE_RATE_IN * X_PAD
        t_pad_tgt = self.target_rate * X_PAD
        t_query, t_center, t_max = SAMPLE_RATE_IN * X_QUERY, SAMPLE_RATE_IN * X_CENTER, SAMPLE_RATE_IN * X_MAX

        # Long audio is cut at the quietest point near every 30 s so memory stays bounded.
        cuts: list[int] = []
        if audio.size + WINDOW > t_max:
            framed = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
            energy = np.convolve(np.abs(framed), np.ones(WINDOW, dtype=np.float32), mode="valid")[: audio.size]
            for t in range(t_center, audio.size, t_center):
                lo, hi = t - t_query, min(t + t_query, energy.size)
                if lo < hi:
                    cuts.append(lo + int(np.argmin(energy[lo:hi])))

        padded = np.pad(audio, (t_pad, t_pad), mode="reflect")
        p_len = padded.size // WINDOW
        sid = torch.tensor([speaker], device=self.runtime.device).long()

        coarse = continuous = None
        if self.f0:
            coarse_np, continuous_np = self._pitch(padded, p_len, pitch)
            coarse = torch.tensor(coarse_np[:p_len], device=self.runtime.device).unsqueeze(0).long()
            continuous = torch.tensor(continuous_np[:p_len], device=self.runtime.device).unsqueeze(0).float()

        pieces = []
        start = 0
        for cut in cuts + [None]:
            if cut is None:
                segment = padded[start:]
                sl = slice(start // WINDOW, None)
            else:
                cut = cut // WINDOW * WINDOW
                segment = padded[start: cut + 2 * t_pad + WINDOW]
                sl = slice(start // WINDOW, (cut + 2 * t_pad) // WINDOW)
            out = self._segment(
                segment,
                coarse[:, sl] if coarse is not None else None,
                continuous[:, sl] if continuous is not None else None,
                sid, index_rate if use_index else 0.0, protect,
            )
            pieces.append(out[t_pad_tgt: -t_pad_tgt])
            if cut is not None:
                start = cut
        result = np.concatenate(pieces).astype(np.float32)
        top = float(np.abs(result).max()) if result.size else 0.0
        if top > 0.99:
            result *= 0.99 / top
        return result, self.target_rate

    def _pitch(self, padded: np.ndarray, p_len: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
        f0 = self.runtime.rmvpe().infer_from_audio(padded, 0.03)
        f0 = f0 * (2 ** (shift / 12))
        if f0.size < p_len:
            f0 = np.pad(f0, (0, p_len - f0.size))
        mel_min, mel_max = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
        mel = 1127 * np.log(1 + f0 / 700)
        mel[mel > 0] = (mel[mel > 0] - mel_min) * 254 / (mel_max - mel_min) + 1
        mel = np.clip(mel, 1, 255)
        return np.rint(mel).astype(np.int64), f0.astype(np.float32)

    def _segment(self, audio16k, coarse, continuous, sid, index_rate, protect) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        feats = self.runtime.features(audio16k, self.version)
        has_pitch = coarse is not None and continuous is not None
        feats0 = feats.clone() if protect < 0.5 and has_pitch else None

        if index_rate > 0 and self._index is not None and self._index_vectors is not None:
            query = feats[0].cpu().numpy().astype(np.float32)
            score, ix = self._index.search(query, k=8)
            weight = np.square(1 / np.maximum(score, 1e-9))
            weight /= weight.sum(axis=1, keepdims=True)
            nearest = np.sum(self._index_vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = torch.from_numpy(nearest.astype(np.float32)).unsqueeze(0).to(feats.device) * index_rate \
                + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if feats0 is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = min(audio16k.shape[0] // WINDOW, feats.shape[1])
        if has_pitch:
            p_len = min(p_len, coarse.shape[1])
            coarse, continuous = coarse[:, :p_len], continuous[:, :p_len]
        feats = feats[:, :p_len]
        if feats0 is not None:
            # keep some of the original consonants and breaths where there is no pitch
            mix = torch.where(continuous > 0, 1.0, protect).unsqueeze(-1)
            feats = feats * mix + feats0[:, :p_len] * (1 - mix)

        lengths = torch.tensor([p_len], device=feats.device).long()
        with torch.no_grad():
            if has_pitch:
                out = self.net.infer(feats, lengths, coarse, continuous, sid)[0]
            else:
                out = self.net.infer(feats, lengths, sid)[0]
        return out[0, 0].float().cpu().numpy()
