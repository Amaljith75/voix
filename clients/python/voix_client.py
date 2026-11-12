"""Minimal VOIX client. Standard library only: copy this one file into any project.

    from voix_client import Voix
    voix = Voix("http://127.0.0.1:8000", api_key="...")       # api_key only if VOIX_API_KEY is set
    audio, timeline = voix.speak("[happy] Well done! Now try the next one.", character="minnu")
    open("line.mp3", "wb").write(audio)

``timeline`` is a list of {emotion, start, end} so an avatar can change its face
in sync with the voice. For many lines at once use ``batch`` and ``wait``.

VOIX never calls an LLM. Generate the tutor's words with any model (Gemini,
Claude, GPT ...), optionally asking it to add emotion tags (see TUTOR_PROMPT),
then pass the text here.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

#: Add this to the tutor LLM's system prompt so its replies carry emotion tags.
TUTOR_PROMPT = """When you write what the tutor says aloud, you may start a sentence with one emotion tag in
square brackets to set how it is spoken: [happy], [excited], [encouraging], [curious], [surprised], [calm],
[sad], [serious]. A tag lasts until the next tag. Use [pause] for a short silence. Use tags sparingly, only
where the feeling changes, and never inside a word. Example: "[excited] You got it! [calm] Now, let us try
one that is a little harder." Do not use any other kind of markup."""


class VoixError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"VOIX {status}: {message}")
        self.status = status


class Voix:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None, timeout: float = 300):
        self.base = base_url.rstrip("/") + "/api/v1"
        self.api_key = api_key
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None) -> tuple[bytes, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - caller-chosen VOIX URL
                return resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except ValueError:
                pass
            raise VoixError(exc.code, str(detail)) from exc

    def _json(self, method: str, path: str, body: dict | None = None):
        return json.loads(self._call(method, path, body)[0] or b"null")

    @staticmethod
    def _request(text: str, character: str | None, voice: str | None, language: str, fmt: str,
                 emotion: str | None, **extra) -> dict:
        if not (character or voice):
            raise ValueError("pass character= or voice=")
        body = {"text": text, "language": language, "output_format": fmt, **extra}
        body["character_id" if character else "voice_id"] = character or voice
        if emotion is not None:
            body["emotion"] = emotion
        return body

    def speak(self, text: str, *, character: str | None = None, voice: str | None = None, language: str = "auto",
              fmt: str = "mp3", emotion: str | None = None, **extra) -> tuple[bytes, list[dict]]:
        """Generate and return (audio bytes, emotion timeline). Blocks until the audio is ready."""
        body = self._request(text, character, voice, language, fmt, emotion, **extra)
        audio, headers = self._call("POST", f"/tts/speak?timeout={int(self.timeout)}", body)
        segments = headers.get("X-VOIX-Segments") or headers.get("x-voix-segments") or "[]"
        return audio, json.loads(segments)

    def batch(self, lines: list[dict]) -> list[dict]:
        """Queue many lines. Each item: {text, character_id|voice_id, language?, output_format?, emotion?}."""
        out = []
        for start in range(0, len(lines), 500):
            out += self._json("POST", "/tts/batch", {"items": lines[start:start + 500]})["jobs"]
        return out

    def wait(self, job_id: str, poll: float = 0.5) -> dict:
        while True:
            job = self._json("GET", f"/tts/jobs/{job_id}")
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(poll)

    def audio(self, job_id: str, fmt: str = "mp3") -> bytes:
        return self._call("GET", f"/audio/{job_id}?format={fmt}")[0]

    def characters(self) -> list[dict]:
        return self._json("GET", "/characters")

    def emotions(self) -> dict:
        return self._json("GET", "/emotions")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Speak a line with a VOIX character")
    ap.add_argument("text")
    ap.add_argument("--character")
    ap.add_argument("--voice")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key")
    ap.add_argument("--out", default="voix.mp3")
    args = ap.parse_args()
    audio, timeline = Voix(args.url, args.key).speak(args.text, character=args.character, voice=args.voice)
    with open(args.out, "wb") as fh:
        fh.write(audio)
    print(f"wrote {args.out} ({len(audio)} bytes)")
    for seg in timeline:
        print(f"  {seg['start']:6.2f}-{seg['end']:6.2f}s  {seg['emotion']}")
