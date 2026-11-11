import { Download, Eraser, Mic2, Play, RotateCcw, Sparkles } from "lucide-react";
import { useState } from "react";
import type { Engine, Voice } from "../api";
import { api, languageName } from "../api";
import { Badge, Button, Card, CardTitle, Empty, Notice, PageHeader, Waveform, cx, useLoad } from "../components/ui";

const SAMPLES = {
  en: "Hello children! Today we are going to learn about planets.",
  ml: "നമസ്കാരം കുട്ടികളേ! ഇന്ന് നമുക്ക് ശാസ്ത്രം പഠിക്കാം.",
};

export default function VoiceStudio() {
  const [text, setText] = useState("");
  const [voiceId, setVoiceId] = useState<string | null>(null);
  const voices = useLoad(() => api<Voice[]>("/voices"));
  const engines = useLoad(() => api<Engine[]>("/engines"));
  const ready = engines.data?.some((e) => e.ready && e.kind === "tts") ?? false;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const voice = voices.data?.find((v) => v.id === voiceId) ?? null;
  const canGenerate = ready && !!voice && text.trim().length > 0;
  const blocked = !ready
    ? "No TTS engine is installed yet. Generation is enabled in Phase 2, when the first engine adapter lands."
    : !voice
      ? "Choose a voice to generate."
      : !text.trim()
        ? "Type some text to generate."
        : "";

  return (
    <>
      <PageHeader title="Voice Studio" subtitle="Write, pick a voice, generate. Malayalam and English." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-4">
          <Card>
            <CardTitle
              action={
                <div className="flex flex-wrap gap-2">
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText(SAMPLES.en)}>
                    <Sparkles size={13} /> English sample
                  </Button>
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText(SAMPLES.ml)}>
                    <Sparkles size={13} /> മലയാളം sample
                  </Button>
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText("")} disabled={!text}>
                    <Eraser size={13} /> Clear
                  </Button>
                </div>
              }
            >
              Text
            </CardTitle>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste text here…"
              aria-label="Text to speak"
              className="h-64 w-full resize-y rounded-2xl bg-soft p-4 text-[15px] leading-relaxed outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10"
            />
            <div className="mt-3 flex gap-4 text-xs font-medium text-muted">
              <span>{words} words</span>
              <span>{[...text].length} characters</span>
            </div>
          </Card>

          <section className="rounded-[28px] bg-ink p-6 text-white shadow-card">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wider text-white/50">Audio workspace</p>
              <span className="text-xs text-white/50">{voice ? voice.name : "No voice selected"}</span>
            </div>
            <Waveform className="my-6" />
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="lime" disabled={!canGenerate} className="!h-12 !px-6 !text-sm">
                <Play size={16} fill="currentColor" /> Generate
              </Button>
              <Button disabled className="!bg-white/10"><RotateCcw size={15} /> Regenerate</Button>
              <Button disabled className="!bg-white/10"><Download size={15} /> WAV</Button>
              <Button disabled className="!bg-white/10"><Download size={15} /> MP3</Button>
            </div>
            {blocked && <p className="mt-4 text-[13px] text-white/60">{blocked}</p>}
          </section>
        </div>

        <div className="space-y-4">
          <Card>
            <CardTitle>Voice</CardTitle>
            {voices.error && <Notice tone="red">{voices.error}</Notice>}
            {voices.data?.length === 0 && (
              <Empty icon={<Mic2 size={20} />} title="No voices yet" action={<a href="#/manager" className="text-[13px] font-semibold underline underline-offset-4">Import a model</a>}>
                A voice is created from a TTS-capable model once an engine can load it.
              </Empty>
            )}
            <div className="space-y-2">
              {voices.data?.map((v) => (
                <button
                  key={v.id}
                  onClick={() => setVoiceId(v.id)}
                  className={cx("flex w-full items-center gap-3 rounded-2xl p-3 text-left transition", voiceId === v.id ? "bg-ink text-white" : "bg-soft hover:bg-line")}
                >
                  <span className="grid h-10 w-10 place-items-center rounded-full bg-lime font-semibold text-ink">{v.name[0]}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold">{v.name}</span>
                    <span className="block truncate text-xs opacity-60">{v.languages.map(languageName).join(", ")}</span>
                  </span>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle action={<Badge>Per model</Badge>}>Generation controls</CardTitle>
            <p className="text-[13px] leading-relaxed text-muted">
              Controls appear after you choose a voice, and only the ones its engine really supports. Speed, pitch, temperature and emotion are not shown for models that ignore them.
            </p>
          </Card>
        </div>
      </div>
    </>
  );
}
