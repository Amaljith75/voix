"""Emotion for speech: inline tags, automatic detection, and per-engine rendering.

Where emotion comes from, in order of priority:

1. **Inline tags** written into the text, e.g. by the tutor LLM:
   ``[happy] Well done! [calm] Now let us look at the next one. [pause] Ready?``
   A tag applies until the next tag. ``[pause]`` / ``[pause 800ms]`` inserts silence.
   Unknown bracketed words are left alone, so "[1]" or "[sic]" stay text.
2. The request's ``emotion``: a fixed emotion for untagged text, ``auto`` to
   detect it per sentence, or ``none`` to switch emotion off (tags are removed).
3. The character's default emotion.

Automatic detection uses a Hugging Face text-classification model when it is
installed (Settings -> Support models -> Emotion classifier, default
``j-hartmann/emotion-english-distilroberta-base``, English only). Without it, or
for Malayalam, a small rule set reads cue words and punctuation. Both report
which one decided, so nobody mistakes a rule for a model.

How each engine can express an emotion (honestly):

* Qwen3-TTS 1.7B CustomVoice / VoiceDesign: a spoken-style instruction. Real.
* Piper and Qwen 0.6B / Base: only speed and variability. Subtle.
* RVC stage (characters): a pitch offset per segment. Subtle.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from .. import support

#: name -> how it is rendered. speed is a multiplier, pitch in semitones (RVC stage),
#: variability is added to Piper's noise scale.
EMOTIONS: dict[str, dict] = {
    "neutral": {"label": "Neutral", "instruct": "", "speed": 1.0, "pitch": 0.0, "variability": 0.0},
    "happy": {"label": "Happy", "instruct": "Speak in a happy, cheerful, smiling tone.",
              "speed": 1.04, "pitch": 1.0, "variability": 0.08},
    "excited": {"label": "Excited", "instruct": "Speak with excitement and high energy, upbeat and bouncy.",
                "speed": 1.1, "pitch": 2.0, "variability": 0.12},
    "encouraging": {"label": "Encouraging", "instruct": "Speak warmly and encouragingly, like a proud, supportive teacher.",
                    "speed": 1.0, "pitch": 0.5, "variability": 0.05},
    "curious": {"label": "Curious", "instruct": "Speak with curiosity and wonder, as if asking a fascinating question.",
                "speed": 1.0, "pitch": 1.0, "variability": 0.08},
    "surprised": {"label": "Surprised", "instruct": "Speak with surprise and amazement.",
                  "speed": 1.06, "pitch": 2.0, "variability": 0.12},
    "calm": {"label": "Calm", "instruct": "Speak calmly and gently, soft and unhurried.",
             "speed": 0.92, "pitch": -0.5, "variability": -0.08},
    "sad": {"label": "Sad", "instruct": "Speak softly with a sad, sympathetic tone.",
            "speed": 0.9, "pitch": -1.5, "variability": -0.05},
    "serious": {"label": "Serious", "instruct": "Speak in a clear, serious, focused tone.",
                "speed": 0.96, "pitch": -0.5, "variability": -0.08},
    "angry": {"label": "Angry", "instruct": "Speak in an angry, frustrated tone.",
              "speed": 1.05, "pitch": -1.0, "variability": 0.05},
    "scared": {"label": "Scared", "instruct": "Speak in a nervous, frightened tone.",
               "speed": 1.08, "pitch": 1.5, "variability": 0.1},
}
ALIASES = {
    "cheerful": "happy", "joy": "happy", "joyful": "happy", "glad": "happy", "smile": "happy", "smiling": "happy",
    "energetic": "excited", "thrilled": "excited", "playful": "excited",
    "proud": "encouraging", "supportive": "encouraging", "warm": "encouraging", "kind": "encouraging",
    "wonder": "curious", "thinking": "curious", "question": "curious", "inquisitive": "curious",
    "amazed": "surprised", "surprise": "surprised", "wow": "surprised",
    "gentle": "calm", "soft": "calm", "soothing": "calm", "relaxed": "calm", "patient": "calm",
    "sorry": "sad", "sympathetic": "sad", "sadness": "sad",
    "firm": "serious", "focused": "serious", "important": "serious",
    "anger": "angry", "mad": "angry", "fear": "scared", "afraid": "scared", "nervous": "scared",
    "normal": "neutral", "plain": "neutral",
}
#: Hugging Face classifier labels -> VOIX emotions. Tuned for a children's tutor:
#: anger and disgust become "serious", fear becomes a reassuring "calm".
CLASSIFIER_MAP = {
    "joy": "happy", "happiness": "happy", "love": "happy", "optimism": "happy", "surprise": "surprised",
    "sadness": "sad", "anger": "serious", "disgust": "serious", "fear": "calm", "neutral": "neutral",
}

_TAG = re.compile(r"\[\s*([A-Za-z][A-Za-z_\- ]{0,30}?)\s*(?:[:=]?\s*(\d+(?:\.\d+)?)\s*(ms|s)?)?\s*\]")


def resolve(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in EMOTIONS:
        return key
    return ALIASES.get(key)


@dataclass
class Part:
    text: str
    emotion: str | None  # None: decide later (auto / default)
    pause_ms: int = 0  # silence after this part


def parse_tags(text: str, keep_tags: bool = True) -> list[Part]:
    """Split text at emotion tags. Unknown [words] stay in the text.

    With ``keep_tags=False`` emotion tags are removed but ignored; pauses still apply.
    """
    parts: list[Part] = []
    current: str | None = None
    pos = 0

    def add(segment: str, pause: int = 0) -> None:
        segment = segment.strip()
        if segment:
            parts.append(Part(segment, current, pause))
        elif pause and parts:
            parts[-1].pause_ms += pause

    for m in _TAG.finditer(text):
        word, number, unit = m.group(1).strip().lower(), m.group(2), m.group(3)
        if word in ("pause", "break"):
            ms = 600 if number is None else int(float(number) * (1000 if unit == "s" else 1))
            add(text[pos:m.start()], min(max(ms, 50), 5000))
            pos = m.end()
            continue
        emo = resolve(word) if number is None else None
        if emo is None:
            continue  # not ours: leave it in the text
        add(text[pos:m.start()])
        if keep_tags:
            current = emo
        pos = m.end()
    add(text[pos:])
    return parts


def strip_tags(text: str) -> str:
    return " ".join(p.text for p in parse_tags(text, keep_tags=False) if p.text)


# ---------------------------------------------------------------- automatic detection

_RULES = [
    ("excited", re.compile(r"\b(yay|hooray|woohoo|awesome|fantastic|amazing|incredible|boom|let's go|champion)\b|"
                           r"(അടിപൊളി|കലക്കി|സൂപ്പർ|ഗംഭീരം|ഹായ്)", re.I)),
    ("sad", re.compile(r"\b(sorry|oh no|unfortunately|sad|lost|miss(ed)?)\b|(ക്ഷമിക്ക|സങ്കട|വിഷമ)", re.I)),
    ("encouraging", re.compile(r"\b(you can|try again|keep going|almost|not yet|don't worry|it's okay|that's okay|"
                               r"let's try|you've got this)\b|(സാരമില്ല|ഒന്നുകൂടി|ശ്രമിക്ക|പേടിക്കേണ്ട)", re.I)),
    ("surprised", re.compile(r"\b(wow|whoa|ooh|oh my|really\?|guess what)\b|(അമ്പോ|ഹോ!|ഓ,)", re.I)),
    ("happy", re.compile(r"\b(well done|great|good job|wonderful|lovely|correct|right|yes|nice|brilliant|"
                         r"proud|happy|glad|perfect|excellent)\b|(നന്നായി|മിടുക്ക|ശരിയാണ്|സന്തോഷ|അഭിമാന)", re.I)),
    ("serious", re.compile(r"\b(careful|remember|important|note that|be sure|never|always|rule)\b|"
                           r"(ശ്രദ്ധിക്ക|ഓർക്കുക|പ്രധാന)", re.I)),
    ("calm", re.compile(r"\b(relax|slowly|gently|calm|breathe|step by step|take your time)\b|"
                        r"(പതുക്കെ|ശാന്ത|സമയമെടുത്ത്)", re.I)),
]


def rule_emotion(text: str) -> str:
    for name, pattern in _RULES:
        if pattern.search(text):
            return name
    stripped = text.strip()
    if stripped.endswith("?"):
        return "curious"
    if stripped.endswith("!"):
        return "happy"
    return "neutral"


_classifier = None
_classifier_lock = threading.Lock()
_classifier_error = ""


def classifier_status() -> dict:
    from ..config import get_settings

    settings = get_settings()
    model = settings.emotion_model
    return {
        "model": model or None,
        "installed": bool(model) and support.installed("emotion", settings),
        "loaded": _classifier is not None,
        "error": _classifier_error,
        "languages": ["en"],
    }


def _get_classifier():
    global _classifier, _classifier_error
    from ..config import get_settings

    settings = get_settings()
    if not settings.emotion_model or not support.installed("emotion", settings):
        return None
    with _classifier_lock:
        if _classifier is None and not _classifier_error:
            try:
                from transformers import pipeline

                _classifier = pipeline("text-classification", model=str(support.path("emotion", settings)),
                                       top_k=None, device=-1)
            except Exception as exc:  # recorded and reported in /emotions, falls back to rules
                _classifier_error = f"{type(exc).__name__}: {exc}"
        return _classifier


def detect(texts: list[str], languages: list[str]) -> list[tuple[str, str, float]]:
    """(emotion, method, confidence) for each text. Never raises."""
    out: list[tuple[str, str, float] | None] = [None] * len(texts)
    clf = None
    english = [i for i, lang in enumerate(languages) if lang == "en" and texts[i].strip()]
    if english:
        clf = _get_classifier()
    if clf is not None:
        try:
            results = clf([texts[i][:512] for i in english], truncation=True)
            for i, scores in zip(english, results):
                top = max(scores, key=lambda s: s["score"])
                label = CLASSIFIER_MAP.get(str(top["label"]).lower(), "neutral")
                if top["score"] < 0.45:
                    label = "neutral"
                # the classifier has no "encouraging"/"curious"; keep a rule hit for those two
                rule = rule_emotion(texts[i])
                if label == "neutral" and rule in ("encouraging", "curious"):
                    out[i] = (rule, "rules", 0.5)
                else:
                    out[i] = (label, "classifier", round(float(top["score"]), 3))
        except Exception:
            pass
    return [o or (rule_emotion(t), "rules", 0.5) for o, t in zip(out, texts)]


# ---------------------------------------------------------------- rendering

@dataclass
class Rendering:
    emotion: str
    speed: float = 1.0
    pitch: float = 0.0
    variability: float = 0.0
    instruct: str = ""
    notes: list[str] = field(default_factory=list)


def render(emotion: str, strength: float = 1.0, style: str = "") -> Rendering:
    spec = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    s = max(0.0, min(float(strength), 2.0))
    instruct = " ".join(x for x in (spec["instruct"], f"Voice and manner: {style.strip()}." if style.strip() else "") if x)
    return Rendering(
        emotion=emotion,
        speed=1.0 + (spec["speed"] - 1.0) * s,
        pitch=spec["pitch"] * s,
        variability=spec["variability"] * s,
        instruct=instruct,
    )


def describe() -> dict:
    return {
        "emotions": [{"id": k, "label": v["label"], "instruct": v["instruct"]} for k, v in EMOTIONS.items()],
        "aliases": ALIASES,
        "tag_syntax": "[happy] text ... [calm] text ... [pause] or [pause 800ms]",
        "modes": ["auto", "none"] + list(EMOTIONS),
        "classifier": classifier_status(),
    }


def reset_classifier() -> None:
    """Forget the loaded classifier (after it is installed, replaced or removed)."""
    global _classifier, _classifier_error
    with _classifier_lock:
        _classifier, _classifier_error = None, ""
