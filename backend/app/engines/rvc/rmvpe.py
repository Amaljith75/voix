"""RMVPE pitch extractor: the vendored network plus a small CPU/CUDA inference wrapper.

The checkpoint is loaded with the weights-only loader, so it cannot run code.
"""
from __future__ import annotations

import numpy as np


class RMVPE:
    def __init__(self, model_path, device: str = "cpu"):
        import torch
        from librosa.filters import mel

        from ..torch_util import safe_torch_load
        from .infer_pack.rmvpe_net import E2E

        self.device = device
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(safe_torch_load(model_path))
        self.model = model.eval().float().to(device)
        self.mel_basis = torch.from_numpy(
            mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        ).float().to(device)
        self.window = torch.hann_window(1024).to(device)
        cents = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents, (4, 4))

    def _mel(self, audio):
        import torch

        fft = torch.stft(audio, n_fft=1024, hop_length=160, win_length=1024, window=self.window,
                         center=True, return_complex=True)
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=1e-5))

    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """16 kHz mono audio -> f0 in Hz per 10 ms frame (0 = unvoiced)."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device).unsqueeze(0)
            mel = self._mel(x)
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self.model(mel)[:, :n_frames].squeeze(0).cpu().numpy()
        return self._decode(hidden, thred)

    def _decode(self, salience: np.ndarray, thred: float) -> np.ndarray:
        center = np.argmax(salience, axis=1) + 4
        padded = np.pad(salience, ((0, 0), (4, 4)))
        idx = center[:, None] + np.arange(-4, 5)[None, :]
        window = np.take_along_axis(padded, idx, axis=1)
        cents = np.sum(window * self.cents_mapping[idx], 1) / np.maximum(np.sum(window, 1), 1e-9)
        cents[np.max(padded, axis=1) <= thred] = 0
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0
        return f0
