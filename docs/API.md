# Using VOIX from the SCOL app

VOIX is a plain HTTP API. It takes text and returns audio; it does not care
which LLM wrote the text. Base URL when VOIX runs on your PC and the phone is
on the same Wi-Fi: `http://<PC-IP>:8000/api/v1` (start VOIX with
`--host 0.0.0.0` and set `VOIX_API_KEY`). Interactive docs: `/api/docs`.

Every write needs the header `X-API-Key: <your key>` once `VOIX_API_KEY` is set.

## The flow

1. `POST /tts/generate` - returns a `job_id` straight away (`202`).
2. `GET /tts/jobs/{job_id}` - poll until `status` is `completed` or `failed`.
3. `GET /audio/{job_id}?format=mp3` - the audio file.

A repeated request with the same text, character and settings is answered
from cache: step 1 already returns `status: "completed"` with `cached: true`.

## Request

```json
POST /api/v1/tts/generate
{
  "text": "നമസ്കാരം കുട്ടികളേ! Today we learn about planets.",
  "character_id": "friendly_teacher",
  "language": "auto",
  "output_format": "mp3"
}
```

| Field | Meaning |
| --- | --- |
| `text` | Up to 20,000 characters. Malayalam, English or mixed. |
| `character_id` | A tutor character. It picks the Malayalam or English voice per sentence and applies the character's voice changer. **Use this from SCOL.** |
| `voice_id` | Alternative to `character_id`: one specific voice. |
| `language` | `ml`, `en` or `auto` (default). |
| `speed` | 0.5 - 2.0. Defaults to the character's speed. Piper voices only. |
| `emotion` | `"happy"`, `"sad"`, `"laughing"` or a sentence. Only acted by engines with emotion control (Qwen3-TTS). Cues inside the text such as `[laughing]` work the same way. Other engines skip them and say so in `warnings`. |
| `rvc` | Optional voice changer for this request: `{"model_id": "...", "pitch": 0, "index_rate": 0.5}`. Overrides the character's. |
| `use_cache` | `false` forces a fresh generation. |

## Response

```json
{ "job_id": "9f38c1e7...", "status": "queued", "cached": false }
```

```json
GET /api/v1/tts/jobs/9f38c1e7...
{
  "job_id": "9f38c1e7...", "status": "completed",
  "audio_url": "/api/v1/audio/9f38c1e7...",
  "audio_seconds": 4.78, "generation_seconds": 10.07,
  "mode": "chunked", "warnings": []
}
```

`mode` is always `chunked`: the audio is complete when the job completes.
There is no streamed endpoint yet. A failed job has `status: "failed"` and an
`error` string; it never returns placeholder audio.

## Expo / React Native example

```ts
const VOIX = "http://192.168.1.20:8000/api/v1";
const headers = { "Content-Type": "application/json", "X-API-Key": VOIX_KEY };

export async function speak(text: string, characterId: string): Promise<string> {
  let job = await (await fetch(`${VOIX}/tts/generate`, {
    method: "POST", headers,
    body: JSON.stringify({ text, character_id: characterId, output_format: "mp3" }),
  })).json();

  while (job.status === "queued" || job.status === "processing") {
    await new Promise((r) => setTimeout(r, 500));
    job = await (await fetch(`${VOIX}/tts/jobs/${job.job_id}`)).json();
  }
  if (job.status !== "completed") throw new Error(job.error ?? "generation failed");
  return `${VOIX}/audio/${job.job_id}?format=mp3`;   // give this URL to expo-audio
}
```

Characters are created in the studio (Characters page) or with
`POST /characters`. The app only ever stores the `character_id`; the voices
and the voice changer behind it can be changed in VOIX without an app update.

## Other endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /characters`, `GET /voices` | What the app can ask for |
| `POST /vc/convert` (multipart: `file`, `model_id`, `pitch`) | Re-voice an audio file with an RVC model; poll the returned job |
| `GET /history`, `DELETE /history/{id}` | Past generations |
| `GET /health` | Is the server up, is a TTS engine ready |

## How long it takes (measured on a 2-core CPU, no GPU)

| Pipeline | Time for 5 s of speech |
| --- | --- |
| Piper only | about 1 s |
| Piper + voice changer (RVC) | about 6 - 10 s |
| Qwen3-TTS | about 25 - 30 s, plus 10 - 30 s the first time it loads |
| Qwen3-TTS + voice changer | about 35 s |

For a live tutor, Piper alone is the only one that feels instant. With the
voice changer, generate the answer sentence by sentence or pre-generate fixed
lesson audio; the cache makes repeated lines free.
