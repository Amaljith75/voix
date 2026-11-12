"""Identify what an extracted model package actually is.

File extensions prove nothing: a .pth can be RVC, GPT-SoVITS, VITS, XTTS or
something unrelated, and an .index file is only a FAISS retrieval index. The
detector looks at state-dict tensor names recovered by the safe checkpoint
scanner and at JSON/YAML sidecar configs, and records the evidence for every
conclusion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .checkpoint import CheckpointReport, inspect_checkpoint

CHECKPOINT_EXT = {".pth", ".pt", ".ckpt", ".safetensors", ".bin"}
MAX_CONFIG_BYTES = 5 * 1024 * 1024

ARCHITECTURE_LABELS = {
    "piper": "Piper (VITS / ONNX)",
    "qwen3_tts": "Qwen3-TTS",
    "qwen3_tts_tokenizer": "Qwen3-TTS speech tokenizer only",
    "xtts": "Coqui XTTS",
    "gpt_sovits": "GPT-SoVITS",
    "vits": "VITS",
    "rvc": "RVC (voice conversion)",
    "so_vits_svc": "so-vits-svc (voice conversion)",
    "faiss_index": "FAISS index only",
    "unknown": "Unknown",
}


@dataclass
class Detection:
    architecture: str = "unknown"
    kind: str = "unknown"  # tts | voice_conversion | unknown
    completeness: str = "unknown"  # complete | incomplete | unknown
    standalone_tts: bool = False
    confidence: str = "low"
    evidence: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    language_note: str = ""
    dependencies: list[str] = field(default_factory=list)
    required_config: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    device_support: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["architecture_label"] = ARCHITECTURE_LABELS.get(self.architecture, self.architecture)
        return d


def _load_config(path: Path):
    if path.stat().st_size > MAX_CONFIG_BYTES:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)  # safe_load: no object construction
    except Exception:
        return None


def _lang_family(code: str) -> str:
    return code.replace("-", "_").split("_")[0].lower()


def detect_package(root: Path, files: list[dict]) -> tuple[Detection, list[CheckpointReport]]:
    """Inspect every file under ``root`` and classify the package."""
    paths = [root / f["path"] for f in files]
    reports = [
        inspect_checkpoint(p, p.relative_to(root).as_posix())
        for p in paths
        if p.suffix.lower() in CHECKPOINT_EXT
    ]
    configs = {
        p.relative_to(root).as_posix(): cfg
        for p in paths
        if p.suffix.lower() in {".json", ".yaml", ".yml"}
        and isinstance(cfg := _load_config(p), dict)
    }
    names = [p.relative_to(root).as_posix() for p in paths]
    onnx = [n for n in names if n.lower().endswith(".onnx")]
    index_files = [n for n in names if n.lower().endswith(".index")]

    det = (
        _piper(onnx, configs)
        or _qwen3(names, configs)
        or _xtts(names, reports, configs)
        or _gpt_sovits(names, reports)
        or _rvc(reports, index_files)
        or _so_vits_svc(reports)
        or _vits(reports, configs)
        or _index_only(index_files, reports, onnx)
        or _unknown(names, reports, onnx)
    )

    cautions = [r.path for r in reports if r.verdict == "caution"]
    if cautions:
        det.warnings.append(
            "Checkpoint references classes outside the safe allowlist "
            f"({', '.join(cautions)}); it can only be loaded by a restricted, weights-only loader."
        )
    return det, reports


def _piper(onnx, configs) -> Detection | None:
    for name in onnx:
        cfg = configs.get(name + ".json")
        if not cfg or "phoneme_id_map" not in cfg:
            continue
        lang = cfg.get("language") or {}
        code = lang.get("code") or (cfg.get("espeak") or {}).get("voice") or ""
        audio = cfg.get("audio") or {}
        return Detection(
            architecture="piper",
            kind="tts",
            completeness="complete",
            standalone_tts=True,
            confidence="high",
            evidence=[
                f"{name} has a Piper sidecar config ({name}.json) with phoneme_id_map",
                f"espeak voice: {(cfg.get('espeak') or {}).get('voice', 'n/a')}",
            ],
            languages=[_lang_family(code)] if code else [],
            language_note=(
                f"Language declared by the voice config: {code or 'not declared'}. "
                "Declared, not yet verified by the VOIX pronunciation suite."
            ),
            dependencies=["piper-tts", "onnxruntime", "espeak-ng (phonemizer)"],
            required_config=[name + ".json"],
            device_support={"cpu": True, "gpu": "optional (onnxruntime-gpu)"},
            details={
                "sample_rate": audio.get("sample_rate"),
                "quality": audio.get("quality"),
                "num_speakers": cfg.get("num_speakers"),
                "piper_version": cfg.get("piper_version"),
            },
        )
    if onnx and not any(n + ".json" in configs for n in onnx):
        return None
    return None


def _qwen3(names, configs) -> Detection | None:
    root_cfg = next(((k, c) for k, c in sorted(configs.items(), key=lambda kv: kv[0].count("/"))
                     if c.get("model_type") == "qwen3_tts"), None)
    if root_cfg is None:
        tok = next((k for k, c in configs.items() if str(c.get("model_type", "")).startswith("qwen3_tts_tokenizer")), None)
        if tok is None:
            return None
        return Detection(
            architecture="qwen3_tts_tokenizer", kind="unknown", completeness="incomplete", confidence="high",
            evidence=[f"{tok} declares a Qwen3-TTS speech tokenizer"],
            missing=["a Qwen3-TTS model (CustomVoice, VoiceDesign or Base). Every model repository already "
                     "includes this tokenizer in speech_tokenizer/, so this package is not needed on its own."],
        )
    from ..engines.qwen3_engine import describe_config

    path, cfg = root_cfg
    prefix = path[: -len("config.json")]
    info = describe_config(cfg)
    base = set(names)
    missing = []
    if not any(n.startswith(prefix + "speech_tokenizer/") and n.endswith("config.json") for n in base):
        missing.append("speech_tokenizer/ (the 12 Hz audio tokenizer)")
    if not any(n.startswith(prefix) and n.count("/") == prefix.count("/") and n.endswith(".safetensors") for n in base):
        missing.append("model weights (*.safetensors)")
    if not ({prefix + "vocab.json", prefix + "tokenizer.json"} & base):
        missing.append("text tokenizer (vocab.json + merges.txt)")
    kinds = {"custom_voice": "built-in speakers", "voice_design": "voice described in words", "base": "voice cloning from a reference clip"}
    return Detection(
        architecture="qwen3_tts", kind="tts",
        completeness="incomplete" if missing else "complete", standalone_tts=not missing, confidence="high",
        evidence=[f"{path} declares model_type qwen3_tts, tts_model_type {info['model_type']}, size {info['size']}"],
        languages=info["languages"],
        language_note="Qwen3-TTS speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, "
                      "Spanish and Italian. It has no Malayalam; pair it with a Malayalam Piper voice in a character.",
        dependencies=["qwen-tts", "torch", "transformers", "librosa"],
        required_config=["config.json", "speech_tokenizer/", "generation_config.json"],
        missing=missing,
        device_support={"cpu": "works, slow (several seconds per sentence)", "gpu": "recommended (CUDA, 4-8 GB VRAM)"},
        warnings=[] if info["follows_instructions"] else [
            "This model does not follow style instructions, so emotions are rendered only through speed and pitch. "
            "Use the 1.7B CustomVoice or VoiceDesign model for emotional delivery."],
        details={"qwen_model_type": info["model_type"], "size": info["size"], "speakers": info["speakers"],
                 "follows_instructions": info["follows_instructions"], "summary": kinds.get(info["model_type"], "")},
    )


def _xtts(names, reports, configs) -> Detection | None:
    cfg = next(
        (c for c in configs.values() if str(c.get("model", "")).lower().startswith("xtts")), None
    )
    weights = next((r for r in reports if r.has("gpt.gpt.") and r.has("hifigan_decoder.")), None)
    if not cfg and not weights:
        return None
    base = {n.rsplit("/", 1)[-1].lower() for n in names}
    missing = []
    if not weights:
        missing.append("model checkpoint with gpt.* and hifigan_decoder.* weights")
    if not cfg:
        missing.append("config.json")
    if "vocab.json" not in base:
        missing.append("vocab.json (tokenizer)")
    langs = [_lang_family(c) for c in (cfg or {}).get("languages", [])]
    return Detection(
        architecture="xtts",
        kind="tts",
        completeness="incomplete" if missing else "complete",
        standalone_tts=not missing,
        confidence="high" if cfg and weights else "medium",
        evidence=[
            e
            for e in (
                "config.json declares model: xtts" if cfg else "",
                f"{weights.path} holds gpt.* and hifigan_decoder.* tensors" if weights else "",
            )
            if e
        ],
        languages=sorted(set(langs)),
        language_note=(
            "XTTS-v2 covers 17 languages and Malayalam is not one of them. "
            "Do not use it for Malayalam."
        ),
        dependencies=["coqui-tts", "torch", "torchaudio"],
        required_config=["config.json", "vocab.json", "speaker reference audio (6s+)"],
        missing=missing,
        device_support={"cpu": "slow", "gpu": "recommended (CUDA, ~4 GB VRAM)"},
        warnings=[
            "XTTS-v2 base weights are under the Coqui Public Model License, which is "
            "non-commercial. Fine-tunes inherit it. Not usable in a commercial product "
            "without a separate license."
        ],
    )


def _gpt_sovits(names, reports) -> Detection | None:
    sovits = next(
        (r for r in reports if r.has("enc_p.ssl_proj") or (r.has("quantizer.") and r.has("ref_enc."))),
        None,
    )
    gpt = next(
        (r for r in reports if r.has("ar_text_embedding") or r.has("ar_audio_embedding")), None
    )
    if not sovits and not gpt:
        return None
    missing = []
    if not sovits:
        missing.append("SoVITS weights (.pth with enc_p.ssl_proj / quantizer tensors)")
    if not gpt:
        missing.append("GPT weights (.ckpt with ar_text_embedding tensors)")
    has_ref = any(n.lower().endswith((".wav", ".flac", ".mp3", ".ogg")) for n in names)
    return Detection(
        architecture="gpt_sovits",
        kind="tts",
        completeness="incomplete" if missing else "complete",
        standalone_tts=not missing,
        confidence="high",
        evidence=[
            e
            for e in (
                f"{sovits.path}: SoVITS decoder tensors (enc_p.ssl_proj / quantizer / ref_enc)"
                if sovits
                else "",
                f"{gpt.path}: autoregressive GPT tensors (ar_text_embedding)" if gpt else "",
            )
            if e
        ],
        languages=["zh", "en", "ja", "ko", "yue"],
        language_note=(
            "Languages are those of the GPT-SoVITS text frontend (Chinese, English, Japanese, "
            "Korean, Cantonese). There is no Malayalam frontend, so this is not Malayalam-capable."
        ),
        dependencies=[
            "GPT-SoVITS inference code",
            "torch",
            "chinese-hubert-base (SSL features)",
            "chinese-roberta-wwm-ext-large (BERT features)",
        ],
        required_config=[
            "reference audio clip (3-10 s) of the target speaker",
            "exact transcript of the reference clip",
        ],
        missing=missing,
        device_support={"cpu": "slow", "gpu": "recommended (CUDA, ~4-6 GB VRAM)"},
        warnings=[]
        if has_ref
        else ["No reference audio in the package; one must be supplied at inference time."],
    )


def _rvc(reports, index_files) -> Detection | None:
    rep = next((r for r in reports if r.has("enc_p.emb_phone")), None)
    if not rep:
        return None
    f0 = rep.has("m_source") or rep.has("emb_pitch")
    v2 = any(s == "v2" for s in rep.strings) or any("_v2" in i for i in index_files)
    return Detection(
        architecture="rvc",
        kind="voice_conversion",
        completeness="complete",
        standalone_tts=False,
        confidence="high",
        evidence=[
            f"{rep.path}: enc_p.emb_phone.* - the encoder input is HuBERT/ContentVec speech "
            "features, not text",
            "pitch-conditioned (f0) generator" if f0 else "no f0 conditioning found",
            f"retrieval index present: {', '.join(index_files)}"
            if index_files
            else "no .index retrieval file (optional)",
        ],
        languages=[],
        language_note=(
            "RVC has no text frontend and no language of its own. It re-voices existing speech, "
            "so the language is whatever the source audio is spoken in."
        ),
        dependencies=[
            "RVC inference code",
            "torch",
            "ContentVec / HuBERT base (content encoder)",
            "RMVPE (pitch extractor)",
            "faiss-cpu (only if the .index file is used)",
        ],
        required_config=["source speech audio to convert", "pitch shift (semitones)"],
        device_support={"cpu": "slow", "gpu": "recommended (CUDA, ~2-4 GB VRAM)"},
        warnings=[
            "This is a voice-conversion model, not text-to-speech. It cannot turn text into "
            "audio by itself. In VOIX it can only run as a second stage: a real TTS engine "
            "speaks the text, then RVC converts that speech to this timbre."
        ],
        details={"f0": f0, "version": "v2" if v2 else "v1 or unspecified", "index_files": index_files},
    )


def _so_vits_svc(reports) -> Detection | None:
    rep = next((r for r in reports if r.has("emb_uv") or r.has("enc_p.f0_emb")), None)
    if not rep:
        return None
    return Detection(
        architecture="so_vits_svc",
        kind="voice_conversion",
        completeness="incomplete",
        standalone_tts=False,
        confidence="medium",
        evidence=[f"{rep.path}: f0 / unvoiced embeddings typical of so-vits-svc"],
        language_note="Singing/speech voice conversion; follows the source audio language.",
        dependencies=["so-vits-svc inference code", "torch", "ContentVec / HuBERT"],
        required_config=["config.json for the model", "source audio to convert"],
        device_support={"cpu": "slow", "gpu": "recommended"},
        warnings=["Voice-conversion model, not text-to-speech."],
    )


def _vits(reports, configs) -> Detection | None:
    orig = next((r for r in reports if r.has("enc_p.emb.weight") and r.has("dp.")), None)
    coqui = next(
        (r for r in reports if r.has("text_encoder.emb.weight") and r.has("duration_predictor.")),
        None,
    )
    rep = orig or coqui
    if not rep:
        return None
    cfg = next((c for c in configs.values() if "data" in c or "characters" in c or "symbols" in c), None)
    missing = [] if cfg else ["config.json with the symbol set / text cleaners used in training"]
    return Detection(
        architecture="vits",
        kind="tts",
        completeness="incomplete" if missing else "complete",
        standalone_tts=not missing,
        confidence="high",
        evidence=[
            f"{rep.path}: text embedding (" + ("enc_p.emb" if orig else "text_encoder.emb")
            + ") plus a duration predictor - the encoder input is text symbols",
            "Coqui-TTS layout" if coqui and not orig else "original VITS layout",
        ],
        language_note=(
            "A VITS checkpoint speaks only the language(s) it was trained on. The language "
            "cannot be read from the weights; set it from the model's source and verify it."
        ),
        dependencies=["torch", "VITS inference code (or coqui-tts)", "matching phonemizer / cleaners"],
        required_config=["config.json (symbols, cleaners, sample rate)"],
        missing=missing,
        device_support={"cpu": True, "gpu": "optional (CUDA)"},
    )


def _index_only(index_files, reports, onnx) -> Detection | None:
    if not index_files or reports or onnx:
        return None
    return Detection(
        architecture="faiss_index",
        kind="unknown",
        completeness="incomplete",
        confidence="high",
        evidence=[f"only FAISS index file(s) found: {', '.join(index_files)}"],
        missing=["model weights (.pth)"],
        warnings=["An .index file is a feature-retrieval index for RVC, not a model."],
    )


def _unknown(names, reports, onnx) -> Detection:
    det = Detection(evidence=[f"{len(names)} file(s); no known architecture signature matched"])
    if onnx:
        det.warnings.append("ONNX file without a Piper .onnx.json sidecar; cannot identify it safely.")
    if not reports and not onnx:
        det.completeness = "incomplete"
        det.missing.append("model weights")
    return det
