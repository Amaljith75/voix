"""Text preparation before synthesis: normalisation, language runs, sentence chunks.

Scope, stated plainly:
* English: numbers, decimals, percentages, common maths symbols, units and a
  few abbreviations are expanded to words.
* Malayalam: maths symbols and units are expanded to Malayalam words. Digits are
  left as digits on purpose - the Piper phonemizer (espeak-ng) reads them out in
  Malayalam, which is more reliable than a hand-written number grammar.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MALAYALAM = re.compile(r"[ഀ-ൿ]")
LATIN = re.compile(r"[A-Za-z]")

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {number_to_words(rest)}" if rest else "")
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            joiner = " and " if 0 < rest < 100 else " "
            return f"{number_to_words(head)} {name}" + (f"{joiner}{number_to_words(rest)}" if rest else "")
    return str(n)


def _spoken_number(token: str) -> str:
    token = token.replace(",", "")
    whole, _, frac = token.partition(".")
    if len(whole) > 15:
        return " ".join(_ONES[int(d)] for d in whole)
    words = number_to_words(int(whole))
    if frac:
        words += " point " + " ".join(_ONES[int(d)] for d in frac)
    return words


_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_UNITS = {
    "en": {"km": "kilometres", "cm": "centimetres", "mm": "millimetres", "kg": "kilograms", "ml": "millilitres",
           "°C": "degrees Celsius", "°F": "degrees Fahrenheit"},
    "ml": {"km": "കിലോമീറ്റർ", "cm": "സെന്റിമീറ്റർ", "mm": "മില്ലിമീറ്റർ", "kg": "കിലോഗ്രാം", "ml": "മില്ലിലിറ്റർ",
           "°C": "ഡിഗ്രി സെൽഷ്യസ്", "°F": "ഡിഗ്രി ഫാരൻഹീറ്റ്"},
}
_SYMBOLS = {
    "en": {"+": "plus", "−": "minus", "×": "times", "÷": "divided by", "=": "equals", "%": "percent",
           "<": "is less than", ">": "is greater than", "&": "and"},
    "ml": {"+": "പ്ലസ്", "−": "മൈനസ്", "×": "ഗുണം", "÷": "ഹരണം", "=": "സമം", "%": "ശതമാനം",
           "<": "ചെറുത്", ">": "വലുത്", "&": "ഉം"},
}
_ABBREVIATIONS = {"Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Prof.": "Professor",
                  "St.": "Saint", "vs.": "versus", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is"}


def normalize(text: str, language: str) -> str:
    """Make text speakable. ``language`` is the dominant language: 'en' or 'ml'."""
    lang = "ml" if language == "ml" else "en"
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[​﻿]", "", text)  # keep ZWJ/ZWNJ: Malayalam chillu and conjunct forms need them
    text = re.sub(r"[“”„]", '"', text).replace("’", "'").replace("‘", "'")
    text = re.sub(r"[–—]", ", ", text)
    for short, full in _ABBREVIATIONS.items():
        text = text.replace(short, full)
    for unit, word in _UNITS[lang].items():
        text = re.sub(rf"(\d)\s?{re.escape(unit)}(?![A-Za-z])", rf"\1 {word}", text)
    # a hyphen between spaces or digits is a minus sign; inside a word it is not
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " − ", text)
    text = re.sub(r"\s-\s(?=\d)", " − ", text)
    text = re.sub(r"\s-\s", ", ", text)
    text = re.sub(r"(?<=\d)\s*[*x]\s*(?=\d)", " × ", text)
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " ÷ ", text)
    for symbol, word in _SYMBOLS[lang].items():
        text = text.replace(symbol, f" {word} ")
    if lang == "en":
        text = _NUMBER.sub(lambda m: _spoken_number(m.group()), text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Dominant language of ``text``: 'ml' if it has more Malayalam than Latin letters."""
    return "ml" if len(MALAYALAM.findall(text)) > len(LATIN.findall(text)) else "en"


def split_language_runs(text: str, default: str) -> list[tuple[str, str]]:
    """Split mixed Malayalam/English text into (language, text) runs.

    Digits, punctuation and spaces carry no language and stay with the run
    they sit in. Single stray words do not start a run of their own unless
    they are written in the other script.
    """
    runs: list[tuple[str, list[str]]] = []
    current = default
    for token in re.findall(r"\S+|\s+", text):
        lang = "ml" if MALAYALAM.search(token) else "en" if LATIN.search(token) else None
        if lang and lang != current and runs:
            current = lang
            runs.append((current, []))
        elif not runs:
            current = lang or default
            runs.append((current, []))
        runs[-1][1].append(token)
    return [(lang, "".join(parts).strip()) for lang, parts in runs if "".join(parts).strip()]


@dataclass(frozen=True)
class Chunk:
    text: str
    language: str
    #: silence to insert after this chunk, in milliseconds
    pause_ms: int


_SENTENCE_END = re.compile(r"(?<=[.!?।…])[\"')\]]*\s+")
_SOFT_BREAK = re.compile(r"(?<=[,;:])\s+")


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    buffer = ""
    for part in _SOFT_BREAK.split(sentence):
        while len(part) > limit:  # no punctuation to break on: fall back to the last space
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append((buffer + " " + part[:cut]).strip() if buffer else part[:cut])
            buffer, part = "", part[cut:].strip()
        if buffer and len(buffer) + len(part) + 1 > limit:
            pieces.append(buffer)
            buffer = part
        else:
            buffer = f"{buffer} {part}".strip()
    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_text(text: str, language: str, *, mixed: bool, max_chars: int = 320,
               sentence_pause_ms: int = 170, paragraph_pause_ms: int = 420, switch_pause_ms: int = 60) -> list[Chunk]:
    """Turn normalised text into ordered chunks small enough for one model call."""
    chunks: list[Chunk] = []
    for paragraph in [p.strip() for p in text.split("\n") if p.strip()]:
        runs = split_language_runs(paragraph, language) if mixed else [(language, paragraph)]
        for r, (lang, run) in enumerate(runs):
            sentences = [s for s in _SENTENCE_END.split(run) if s.strip()]
            for s, sentence in enumerate(sentences):
                pieces = _split_long(sentence.strip(), max_chars)
                for p, piece in enumerate(pieces):
                    last_piece = p == len(pieces) - 1
                    last_sentence = s == len(sentences) - 1
                    pause = 90 if not last_piece else sentence_pause_ms if not last_sentence else switch_pause_ms if r < len(runs) - 1 else paragraph_pause_ms
                    chunks.append(Chunk(piece, lang, pause))
    if chunks:
        chunks[-1] = Chunk(chunks[-1].text, chunks[-1].language, 0)
    return chunks
