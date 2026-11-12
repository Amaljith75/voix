"""RVC voice conversion (v1 and v2, with or without pitch).

The pipeline follows RVC's own inference (infer/vc/pipeline.py, MIT):

  16 kHz -> 48 Hz high-pass -> reflect pad -> split long audio at quiet points
  -> HuBERT features -> optional .index retrieval blend -> x2 upsample
  -> pitch (RMVPE or FCPE) -> consonant protection -> RVC synthesizer -> RMS mix

Two choices differ from today's upstream on purpose:
* unvoiced frames stay unvoiced (pitch 0), the classic behaviour every
  community model was trained and tuned with;
* the pitch shift can change over time (``pitch_track``), so VOIX can give each
  emotion segment of a sentence its own pitch offset.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from ...audio.processing import resample_hq
from ..base import AudioResult, BaseVoiceConverter, EngineError, ParameterSpec
from ..registry import register
from ..torch_util import ai_runtime_hint, missing, safe_torch_load

SR = 16000
WINDOW = 160
# CPU / small-GPU segmentation constants from RVC (x_pad, x_query, x_center, x_max)
X_PAD, X_QUERY, X_CENTER, X_MAX = 1, 6, 38, 41
F0_MIN, F0_MAX = 50.0, 1100.0
F0_MEL_MIN = 1127 * np.log(1 + F0_MIN / 700)
F0_MEL_MAX = 1127 * np.log(1 + F0_MAX / 700)


def find_rvc_files(model_dir: Path) -> tuple[list[Path], Path | None]:
    pths = sorted(model_dir.rglob("*.pth"), key=lambda p: p.stat().st_size, reverse=True)
    indexes = sorted(model_dir.rglob("*.index"), key=lambda p: ("added" not in p.name.lower(), -p.stat().st_size))
    return pths, (indexes[0] if indexes else None)


def high_pass(audio: np.ndarray) -> np.ndarray:
    from scipy import signal

    b, a = signal.butter(N=5, Wn=48, btype="high", fs=SR)
    return signal.filtfilt(b, a, audio)


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    """RMS every half second, centred, like librosa.feature.rms(frame=sr//2*2, hop=sr//2)."""
    frame, hop = sr // 2 * 2, sr // 2
    padded = np.pad(x, (frame // 2, frame // 2))
    n = 1 + (padded.size - frame) // hop if padded.size >= frame else 1
    return np.array([np.sqrt(np.mean(padded[i * hop: i * hop + frame] ** 2) + 1e-12) for i in range(n)])


def mix_rms(source: np.ndarray, sr_source: int, output: np.ndarray, sr_output: int, rate: float) -> np.ndarray:
    """Blend the loudness envelope of the input into the output (rate=1 keeps the output's own)."""
    if rate >= 1 or output.size == 0:
        return output
    pos = np.linspace(0, 1, output.size)
    r1 = frame_rms(source, sr_source)
    r2 = np.maximum(frame_rms(output, sr_output), 1e-6)
    r1 = np.interp(pos, np.linspace(0, 1, r1.size), r1)
    r2 = np.interp(pos, np.linspace(0, 1, r2.size), r2)
    return output * (np.power(r1, 1 - rate) * np.power(r2, rate - 1))


@register
class RVCEngine(BaseVoiceConverter):
    architecture = "rvc"

    def __init__(self) -> None:
        self.net_g = None
        self.version = "v1"
        self.if_f0 = 1
        self.tgt_sr = 40000
        self.n_speakers = 1
        self.device = "cpu"
        self.pth: Path | None = None
        self.index_path: Path | None = None
        self._index = None
        self._index_vectors = None
        self._index_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        gone = missing("torch", "transformers", "scipy", "librosa")
        return (False, ai_runtime_hint(gone)) if gone else (True, "")

    def validate_model(self, model_dir: Path) -> list[str]:
        pths, _ = find_rvc_files(model_dir)
        return [] if pths else ["no .pth weights file"]

    def load_model(self, model_dir: Path, device: str = "cpu") -> None:
        from . import content
        from ..torch_util import device as pick_device

        pths, self.index_path = find_rvc_files(model_dir)
        if not pths:
            raise EngineError("no .pth weights file in this RVC package")
        cpt, chosen = None, None
        for path in pths:
            try:
                data = safe_torch_load(path)
            except RuntimeError as exc:
                raise EngineError(str(exc)) from exc
            if isinstance(data, dict) and "weight" in data and "config" in data:
                cpt, chosen = data, path
                break
        if cpt is None:
            raise EngineError(
                "no inference checkpoint found. RVC needs the small exported .pth (with 'weight' and 'config'), "
                "not the G_/D_ training checkpoints."
            )
        self.device = pick_device()
        self._build(cpt)
        self.pth = chosen
        content.hubert(self.device)  # fetch / load the shared encoder now, not mid-request

    def _build(self, cpt: dict) -> None:
        from .infer_pack import models

        config = list(cpt["config"])
        weights = cpt["weight"]
        self.version = str(cpt.get("version", "v1"))
        self.if_f0 = int(cpt.get("f0", 1))
        config[-3] = int(weights["emb_g.weight"].shape[0])
        self.n_speakers = config[-3]
        self.tgt_sr = int(config[-1]) if not isinstance(config[-1], str) else models.sr2sr[config[-1]]
        cls = {
            ("v1", 1): models.SynthesizerTrnMs256NSFsid, ("v1", 0): models.SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): models.SynthesizerTrnMs768NSFsid, ("v2", 0): models.SynthesizerTrnMs768NSFsid_nono,
        }.get((self.version, self.if_f0))
        if cls is None:
            raise EngineError(f"unsupported RVC checkpoint (version={self.version}, f0={self.if_f0})")
        try:
            net_g = cls(*config, is_half=False)
        except TypeError as exc:
            raise EngineError(f"checkpoint config does not match RVC {self.version}: {exc}") from exc
        if hasattr(net_g, "enc_q"):
            del net_g.enc_q
        result = net_g.load_state_dict(weights, strict=False)
        bad_missing = [k for k in result.missing_keys if not k.startswith("enc_q.")]
        bad_extra = [k for k in result.unexpected_keys if not k.startswith("enc_q.")]
        if bad_missing or bad_extra:
            raise EngineError(
                f"weights do not fit an RVC {self.version} model "
                f"(missing {bad_missing[:3]}, unexpected {bad_extra[:3]})"
            )
        self.net_g = net_g.float().to(self.device).eval()

    def unload_model(self) -> None:
        self.net_g = None
        self._index = self._index_vectors = None

    # ------------------------------------------------------------ index

    def _load_index(self):
        if self.index_path is None:
            return None, None
        with self._index_lock:
            if self._index is None:
                try:
                    import faiss
                except ImportError as exc:
                    raise EngineError("this voice has an .index file; reading it needs faiss-cpu") from exc
                index = faiss.read_index(str(self.index_path))
                self._index_vectors = index.reconstruct_n(0, index.ntotal)
                self._index = index
            return self._index, self._index_vectors

    # ------------------------------------------------------------ pitch

    def _f0(self, audio_pad: np.ndarray, p_len: int, shift: np.ndarray, method: str):
        from . import content

        f0 = content.pitch_extractor(method, self.device).infer_from_audio(audio_pad).astype(np.float64)
        if f0.size < p_len:
            f0 = np.pad(f0, (0, p_len - f0.size))
        f0 = f0[:p_len] * np.power(2.0, shift[:p_len] / 12.0)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        voiced = f0_mel > 0
        f0_mel[voiced] = (f0_mel[voiced] - F0_MEL_MIN) * 254 / (F0_MEL_MAX - F0_MEL_MIN) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return np.rint(f0_mel).astype(np.int64), f0bak.astype(np.float32)

    @staticmethod
    def shift_curve(p_len: int, pad_frames: int, base: float, track) -> np.ndarray:
        """Semitone offset for every 10 ms frame of the padded input."""
        shift = np.full(p_len, float(base))
        for start, end, semis in track or []:
            a = max(0, pad_frames + int(float(start) * SR / WINDOW))
            b = min(p_len, pad_frames + int(float(end) * SR / WINDOW))
            shift[a:b] += float(semis)
        if track:
            kernel = np.ones(9) / 9  # ~90 ms glide instead of a jump between segments
            shift = np.convolve(np.pad(shift, 4, mode="edge"), kernel, mode="valid")
        return shift

    # ------------------------------------------------------------ inference

    def _vc(self, sid, audio0, pitch, pitchf, index_rate, protect):
        import torch
        import torch.nn.functional as F

        from . import content

        feats = content.features(audio0, self.version, self.device)
        use_protect = protect < 0.5 and pitch is not None
        feats0 = feats.clone() if use_protect else None
        index, vectors = self._load_index() if index_rate > 0 else (None, None)
        if index is not None:
            npy = feats[0].cpu().numpy().astype("float32")
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / np.maximum(score, 1e-9))
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(vectors[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if use_protect:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // WINDOW
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None:
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
        if use_protect:
            mask = pitchf.clone()
            mask[pitchf > 0] = 1
            mask[pitchf < 1] = protect
            mask = mask.unsqueeze(-1)
            feats = feats[:, :mask.shape[1]] * mask + feats0[:, :mask.shape[1]] * (1 - mask)
        lengths = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch is not None:
                out = self.net_g.infer(feats, lengths, pitch, pitchf, sid)[0]
            else:
                out = self.net_g.infer(feats, lengths, sid)[0]
        return out[0, 0].float().cpu().numpy()

    def convert(self, samples, sample_rate: int, **params) -> AudioResult:
        import torch

        if self.net_g is None:
            raise EngineError("model is not loaded")
        pitch_shift = float(params.get("pitch_shift", 0))
        index_rate = float(params.get("index_rate", 0.75)) if self.index_path else 0.0
        protect = float(params.get("protect", 0.33))
        rms_mix = float(params.get("rms_mix", 0.25))
        method = str(params.get("f0_method", "rmvpe"))
        speaker = int(params.get("speaker", 0))
        track = params.get("pitch_track") or []

        audio = resample_hq(np.asarray(samples, dtype=np.float32), sample_rate, SR).astype(np.float64)
        if audio.size < SR // 10:
            raise EngineError("audio is too short to convert (under 0.1 s)")
        peak = np.abs(audio).max() / 0.95
        if peak > 1:
            audio /= peak
        audio = high_pass(audio)

        t_pad, t_pad_tgt = SR * X_PAD, self.tgt_sr * X_PAD
        t_query, t_center, t_max = SR * X_QUERY, SR * X_CENTER, SR * X_MAX
        audio_pad = np.pad(audio, (WINDOW // 2, WINDOW // 2), mode="reflect")
        cuts: list[int] = []
        if audio_pad.shape[0] > t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(WINDOW):
                audio_sum += np.abs(audio_pad[i: i - WINDOW])
            for t in range(t_center, audio.shape[0], t_center):
                window = audio_sum[t - t_query: t + t_query]
                cuts.append(t - t_query + int(np.argmin(window)))

        audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // WINDOW
        sid = torch.tensor([min(max(speaker, 0), self.n_speakers - 1)], device=self.device).long()
        pitch = pitchf = None
        if self.if_f0 == 1:
            shift = self.shift_curve(p_len, t_pad // WINDOW, pitch_shift, track)
            coarse, f0 = self._f0(audio_pad, p_len, shift, method)
            pitch = torch.tensor(coarse, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(f0, device=self.device).unsqueeze(0).float()

        pieces, s, t = [], 0, None
        for t in cuts:
            t = t // WINDOW * WINDOW
            seg = audio_pad[s: t + t_pad * 2 + WINDOW]
            p = pitch[:, s // WINDOW: (t + t_pad * 2) // WINDOW] if pitch is not None else None
            pf = pitchf[:, s // WINDOW: (t + t_pad * 2) // WINDOW] if pitch is not None else None
            pieces.append(self._vc(sid, seg, p, pf, index_rate, protect)[t_pad_tgt: -t_pad_tgt])
            s = t
        seg = audio_pad[t:] if t is not None else audio_pad
        p = (pitch[:, t // WINDOW:] if t is not None else pitch) if pitch is not None else None
        pf = (pitchf[:, t // WINDOW:] if t is not None else pitchf) if pitch is not None else None
        pieces.append(self._vc(sid, seg, p, pf, index_rate, protect)[t_pad_tgt: -t_pad_tgt])

        out = np.concatenate(pieces).astype(np.float64)
        out = mix_rms(audio, SR, out, self.tgt_sr, rms_mix)
        top = np.abs(out).max() / 0.99
        if top > 1:
            out /= top
        return AudioResult(out.astype(np.float32), self.tgt_sr)

    # ------------------------------------------------------------ description

    def get_model_info(self) -> dict:
        return {
            "file": self.pth.name if self.pth else None, "index": self.index_path.name if self.index_path else None,
            "version": self.version, "pitch": bool(self.if_f0), "sample_rate": self.tgt_sr,
            "speakers": self.n_speakers, "device": self.device,
        }

    def get_supported_parameters(self) -> list[ParameterSpec]:
        return self.parameters_for(has_index=self.index_path is not None)

    @staticmethod
    def parameters_for(has_index: bool = True, pitch_methods: tuple[str, ...] = ("rmvpe", "fcpe")) -> list[ParameterSpec]:
        specs = [
            ParameterSpec("pitch_shift", "Pitch shift (semitones)", "float", 0, -24, 24),
            ParameterSpec("protect", "Protect consonants", "float", 0.33, 0.0, 0.5),
            ParameterSpec("rms_mix", "Loudness from source", "float", 0.25, 0.0, 1.0),
            ParameterSpec("f0_method", "Pitch tracker", "choice", "rmvpe", choices=pitch_methods),
        ]
        if has_index:
            specs.insert(1, ParameterSpec("index_rate", "Index (timbre detail)", "float", 0.75, 0.0, 1.0))
        return specs
