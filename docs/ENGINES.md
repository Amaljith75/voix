# Engine recommendation for Malayalam and English

Status: research only. **No engine has been run or listened to yet.** Language
support below is what each project declares; VOIX will not mark a voice as
Malayalam-capable until it passes the pronunciation suite in Phase 2.

## Recommendation

**Start with Piper.** Add **Indic Parler-TTS** as the higher-quality GPU engine
in Phase 3. Use **RVC only as an optional second stage** to put a community
voice's timbre on top of real TTS output.

## Why

| Engine | Malayalam | English | Runs on | License position | Role in VOIX |
| --- | --- | --- | --- | --- | --- |
| Piper | Yes, per-voice models (`ml_IN`) | Yes | CPU, real time | Engine: the archived `rhasspy/piper` is MIT. Check the licence of the maintained fork before bundling it. Voices are licensed individually. | Phase 2, first engine |
| Indic Parler-TTS (AI4Bharat) | Yes (declared) | Yes | GPU, slow | Apache-2.0 | Phase 3, quality engine |
| GPT-SoVITS | No text frontend for Malayalam | Yes | GPU | MIT code; each fine-tune has its own terms | Phase 3, English / cloning only |
| Coqui XTTS-v2 | No (17 languages, Malayalam not among them) | Yes | GPU | Coqui Public Model License: non-commercial | Not for SCOL production |
| RVC | n/a — converts speech, not text | n/a | GPU preferred | Varies per model, usually undeclared | Optional post-TTS timbre stage |

Piper comes first because it is the only candidate that already runs on the
development machine (no GPU needed), has ready Malayalam voices, and has a
small, stable inference path (ONNX Runtime). It gets a real end-to-end
pipeline working before any GPU money is spent.

## Malayalam voices to evaluate in Phase 2

| Voice | Format | Licence note |
| --- | --- | --- |
| IMaSC Malayalam TTS | Piper-compatible | Released under Apache-2.0. Best first candidate for commercial use. |
| `ml_IN-meera-medium`, `ml_IN-arjun-medium` (rhasspy/piper-voices) | Piper | Trained on the Indic TTS Malayalam corpus; the model card defers to the dataset's own licence. **Treat as unknown until that licence is read.** VOIX flags it as such on import. |

## About the Discord voice library

Community `.pth` + `.index` ZIPs are almost always **RVC**. RVC re-voices
existing speech; it has no text input. Those models cannot be SCOL's TTS on
their own. The workable path is: Piper (or Parler) speaks the Malayalam text,
then RVC converts that audio to the character's timbre. That doubles latency
and compute, and inherits the pronunciation of the first stage.

Two things to settle before relying on them: most community voice models
publish no licence, and many are clones of real people made without consent.
Neither is acceptable in a commercial children's product. VOIX records
provenance and flags unknown licences so this is decided per model, in the
open.

## Sources

- Piper overview and licence: https://tts.ai/voices/piper/
- IMaSC Malayalam TTS: https://thennal10.github.io/projects/malayalam_tts
- Indic TTS Malayalam corpus card: https://huggingface.co/datasets/thennal/indic_tts_ml
- Indic Parler-TTS: https://huggingface.co/ai4bharat/indic-parler-tts-pretrained
- XTTS-v2 model card and CPML: https://huggingface.co/coqui/XTTS-v2
