"""Shared helpers for RVC: the HuBERT/ContentVec encoder and the pitch extractors.

They are big (HuBERT ~360 MB) and identical for every RVC voice, so one copy
per device is kept and shared by all loaded RVC models.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from ... import support

_lock = threading.Lock()
_hubert: dict[str, tuple] = {}
_pitch: dict[tuple[str, str], object] = {}


def hubert(device: str):
    """(model, normalize_input) for the shared content encoder."""
    with _lock:
        if device not in _hubert:
            import torch
            from torch import nn
            from transformers import HubertModel

            class HubertModelWithFinalProj(HubertModel):
                def __init__(self, config):
                    super().__init__(config)
                    self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

            folder = support.ensure("hubert_base")
            model = HubertModelWithFinalProj.from_pretrained(str(folder), local_files_only=True,
                                                             dtype=torch.float32)
            normalize = False
            pre = folder / "preprocessor_config.json"
            if pre.is_file():
                normalize = bool(json.loads(pre.read_text(encoding="utf-8")).get("do_normalize", False))
            _hubert[device] = (model.to(device).eval(), normalize)
        return _hubert[device]


def features(audio16: np.ndarray, version: str, device: str):
    """RVC v1: layer 9 + final_proj (256-D). RVC v2: last layer (768-D). Shape (1, T, C)."""
    import torch

    model, normalize = hubert(device)
    x = torch.from_numpy(np.ascontiguousarray(audio16, dtype=np.float32)).to(device).view(1, -1)
    if normalize:
        x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
    with torch.no_grad():
        if version == "v1":
            out = model(input_values=x, output_hidden_states=True, return_dict=True)
            return model.final_proj(out.hidden_states[9])
        return model(input_values=x, return_dict=True).last_hidden_state


class _FCPE:
    def __init__(self, device: str):
        from torchfcpe import spawn_bundled_infer_model

        self.model = spawn_bundled_infer_model(device)
        self.device = device

    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.006) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).unsqueeze(-1).to(self.device)
        n_frames = audio.shape[0] // 160 + 1
        with torch.no_grad():
            f0 = self.model.infer(x, sr=16000, decoder_mode="local_argmax", threshold=thred,
                                  f0_min=50, f0_max=1100, interp_uv=False, output_interp_target_length=n_frames)
        return f0.squeeze().cpu().numpy().astype(np.float64)


def pitch_methods() -> list[str]:
    import importlib.util

    return ["rmvpe"] + (["fcpe"] if importlib.util.find_spec("torchfcpe") else [])


def pitch_extractor(method: str, device: str):
    with _lock:
        key = (method, device)
        if key not in _pitch:
            if method == "rmvpe":
                from .rmvpe import RMVPE

                _pitch[key] = RMVPE(support.ensure("rmvpe") / "rmvpe.pt", device)
            elif method == "fcpe":
                _pitch[key] = _FCPE(device)
            else:
                raise ValueError(f"unknown pitch method '{method}' (use one of {pitch_methods()})")
        return _pitch[key]
