# Using VOIX as the voice of the SCOL tutors

Status: **VOIX side ready, SCOL not changed.** Nothing in the SCOL repository
uses VOIX yet. This page is the plan and the contract for when it does.

```
 lesson notes ──► LLM (Gemini / Claude / GPT / ...) ──► tutor text with [emotion] tags
                                                              │
                                                              ▼
                              VOIX  POST /api/v1/tts/speak  {text, character_id}
                                                              │
     Piper (Malayalam) or Qwen3-TTS (English) ──► RVC: the character's own timbre
                                                              │
                                     audio  +  emotion timeline (for the avatar)
```

VOIX never calls an LLM, so the LLM can be swapped at any time. SCOL (or its
build tools) calls the LLM, then sends the text to VOIX with the tutor's ID.

## 1. One VOIX character per SCOL tutor

Create each character with the **same ID** SCOL already uses in
`content/_app/tutors.json`, and copy the tutor's `style` sentence. Qwen3-TTS
1.7B follows that sentence; the others use it as documentation.

| SCOL id | Name | `style` (from tutors.json) | Suggested emotion default |
| --- | --- | --- | --- |
| `minnu` | Mio | a warm, gentle storyteller talking to a young child, unhurried and smiling | `calm`, strength 0.8 |
| `tiko` | Leo | an excited, playful game host talking to a young child, upbeat and bouncy | `happy`, strength 1.3 |
| `pompy` | Yuki | a calm, patient, soothing friend talking slowly to a young child | `calm`, strength 0.6 |
| `zizu` | Ruby | a curious, wonder-filled young explorer talking to a young child, bright and questioning | `curious`, strength 1.0 |
| `koko` | Ollie | a funny, cheeky, good-humoured friend talking to a young child, with a playful grin in the voice | `happy`, strength 1.0 |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/characters -H "Content-Type: application/json" -d '{
  "id": "tiko", "name": "Leo",
  "voice_ml": "ml_in_arjun_medium", "voice_en": "qwen3_tts_12hz_1_7b_customvoice_aiden",
  "style": "an excited, playful game host talking to a young child, upbeat and bouncy",
  "rvc_model_id": "<id of the Leo RVC model>",
  "rvc_settings": {"pitch_shift": 0, "index_rate": 0.75, "protect": 0.33, "f0_method": "rmvpe"},
  "emotion_mode": "auto", "default_emotion": "happy", "emotion_strength": 1.3
}'
```

Why both a TTS voice and an RVC voice: RVC only re-voices speech, it cannot
read text. The TTS voice says the words (Piper for Malayalam, Qwen3-TTS or
Piper for English); RVC then gives **both languages the same character
timbre**. Use `pitch_shift` when the TTS voice and the RVC voice differ a lot
(for example +12 for a male TTS voice under a child RVC voice).

## 2. Ask the LLM for emotion tags

Add this to the tutor's system prompt (it is also `TUTOR_PROMPT` in
`clients/python/voix_client.py`):

> When you write what the tutor says aloud, you may start a sentence with one
> emotion tag in square brackets: [happy], [excited], [encouraging], [curious],
> [surprised], [calm], [sad], [serious]. A tag lasts until the next tag. Use
> [pause] for a short silence. Use tags sparingly, only where the feeling
> changes. Do not use any other markup.

Untagged text is handled by the character's `emotion_mode` (`auto` detects per
sentence). Send `"emotion": "none"` to ignore tags entirely.

## 3. Call VOIX

**One line, audio back** (live "ask the tutor"):

```http
POST /api/v1/tts/speak
{"text": "[excited] You got it! [calm] Now a harder one.", "character_id": "tiko", "output_format": "mp3"}
```

The body is the MP3. Headers:

| Header | Meaning |
| --- | --- |
| `X-VOIX-Job-Id` | Job id; `GET /api/v1/tts/jobs/{id}` has full details, `GET /api/v1/audio/{id}` the file again |
| `X-VOIX-Audio-Seconds` | Duration |
| `X-VOIX-Segments` | JSON list `[{emotion, start, end}]`, for the avatar's face |

**Many lines** (what `tools/build_voice.py` does today with Gemini):

```http
POST /api/v1/tts/batch
{"items": [{"text": "...", "character_id": "minnu", "language": "ml"}, ...]}   (up to 500 per call)
```

Then poll `GET /api/v1/tts/jobs/{job_id}` and download `GET /api/v1/audio/{job_id}?format=mp3`.
Identical requests are served from the cache, so re-running costs nothing.
There is no daily quota: the limit is this machine's speed.

Python (standard library only): `clients/python/voix_client.py`

```python
from voix_client import Voix
voix = Voix("http://127.0.0.1:8000")
audio, timeline = voix.speak(llm_reply, character="tiko")
```

Dart / Flutter (SCOL app), when it is time:

```dart
final res = await http.post(Uri.parse('$voixUrl/api/v1/tts/speak'),
    headers: {'Content-Type': 'application/json', if (key != null) 'X-API-Key': key},
    body: jsonEncode({'text': reply, 'character_id': tutor.id, 'output_format': 'mp3'}));
final timeline = jsonDecode(res.headers['x-voix-segments'] ?? '[]');
await player.play(BytesSource(res.bodyBytes));
```

## 4. Before children hear it

- **Licences and consent.** Most community RVC character voices declare no
  licence, and many imitate real people or copyrighted characters. VOIX
  records the source and flags unknown licences; decide per voice. A
  commercial children's app should use voices you made or have rights to.
- **Speed on a laptop CPU.** Piper: faster than real time. RVC: about real
  time per CPU core count (a 3 s line takes a few seconds); the first job after
  start-up also loads the models. Qwen3-TTS on CPU takes several seconds per
  sentence (0.6B) or much longer (1.7B). For live use, pre-generate with
  `/tts/batch`, or run VOIX on a machine with an NVIDIA GPU.
- **Expose safely.** Set `VOIX_API_KEY` before letting phones reach VOIX over
  the network. The app must never contain the LLM key; the same applies to the
  VOIX key in a public build (call VOIX from your own server instead).
- **Listen.** Nobody has yet judged the Malayalam pronunciation or the RVC
  results by ear. Do that with real tutor lines before switching SCOL over.
