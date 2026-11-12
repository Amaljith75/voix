import { Eraser, Mic2, Play, RotateCcw, Smile, Sparkles, Users } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Character, EmotionInfo, Job, Voice } from "../api";
import { api, generate, languageName } from "../api";
import { Player } from "../components/Player";
import { Badge, Button, Card, CardTitle, Empty, Notice, PageHeader, Waveform, cx, useLoad } from "../components/ui";

const SAMPLES = {
  en: "[excited] Hello children! Today we are going to learn about planets. [curious] Do you know which planet is the biggest? [pause] [happy] It is Jupiter!",
  ml: "നമസ്കാരം കുട്ടികളേ! ഇന്ന് നമുക്ക് ശാസ്ത്രം പഠിക്കാം.",
};
type Target = { kind: "voice" | "character"; id: string };

export default function VoiceStudio() {
  const [text, setText] = useState("");
  const [target, setTarget] = useState<Target | null>(null);
  const [language, setLanguage] = useState("auto");
  const [params, setParams] = useState<Record<string, number>>({});
  const [emotion, setEmotion] = useState("auto");
  const [strength, setStrength] = useState(1);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const emotions = useLoad(() => api<EmotionInfo>("/emotions"));
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const voices = useLoad(() => api<Voice[]>("/voices"));
  const characters = useLoad(() => api<Character[]>("/characters"));

  const usableCharacters = (characters.data ?? []).filter((c) => c.voice_ml || c.voice_en);
  const voice = target?.kind === "voice" ? voices.data?.find((v) => v.id === target.id) : undefined;
  const character = target?.kind === "character" ? usableCharacters.find((c) => c.id === target.id) : undefined;
  const controls = (voice?.parameters ?? (character ? [{ name: "speed", label: "Speed", type: "float", default: character.default_speed, min: 0.5, max: 2 }] : []))
    .filter((p) => p.type !== "choice");

  useEffect(() => {
    if (!target && voices.data?.length) setTarget({ kind: "voice", id: voices.data[0].id });
  }, [voices.data, target]);
  useEffect(() => {
    setParams(Object.fromEntries(controls.map((p) => [p.name, Number(p.default)])));
    if (character) setEmotion(character.emotion_mode || "auto");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.kind, target?.id]);

  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const blocked = !target ? "Choose a voice or character." : voice && !voice.ready ? voice.reason : !text.trim() ? "Type some text to generate." : "";

  async function run(useCache: boolean) {
    if (!target || blocked) return;
    setBusy(true);
    setError(null);
    const { speed = 1, ...rest } = params;
    try {
      const result = await generate(
        {
          text, language, [target.kind === "voice" ? "voice_id" : "character_id"]: target.id, speed, params: rest,
          use_cache: useCache, output_format: "wav", emotion, emotion_strength: strength,
        },
        setJob,
      );
      if (result.status === "failed") setError(result.error ?? "Generation failed.");
    } catch (e) {
      setJob(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function insertTag(tag: string) {
    const el = textRef.current;
    const at = el?.selectionStart ?? text.length;
    const insert = `[${tag}] `;
    setText((t) => t.slice(0, at) + insert + t.slice(at));
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(at + insert.length, at + insert.length);
    });
  }

  const expressive = voice ? voice.follows_instructions : character ? true : false;
  const row = (active: boolean) => cx("flex w-full items-center gap-3 rounded-2xl p-3 text-left transition", active ? "bg-ink text-white" : "bg-soft hover:bg-line");

  return (
    <>
      <PageHeader title="Voice Studio" subtitle="Write, pick a voice, generate. Malayalam and English." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-4">
          <Card>
            <CardTitle
              action={
                <div className="flex flex-wrap gap-2">
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText(SAMPLES.en)}><Sparkles size={13} /> English sample</Button>
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText(SAMPLES.ml)}><Sparkles size={13} /> മലയാളം sample</Button>
                  <Button variant="ghost" className="!h-8 !px-3 !text-xs" onClick={() => setText("")} disabled={!text}><Eraser size={13} /> Clear</Button>
                </div>
              }
            >
              Text
            </CardTitle>
            <textarea
              ref={textRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste text here…"
              aria-label="Text to speak"
              maxLength={20000}
              className="h-64 w-full resize-y rounded-2xl bg-soft p-4 text-[15px] leading-relaxed outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10"
            />
            <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs font-medium text-muted">
              <span className="mr-2">{words} words · {[...text].length} characters</span>
              <span className="mr-1">Insert tag:</span>
              {["happy", "excited", "calm", "curious", "encouraging", "sad", "serious", "pause"].map((t) => (
                <button key={t} onClick={() => insertTag(t)} className="rounded-full bg-soft px-2.5 py-1 font-mono text-[11px] text-ink hover:bg-line">[{t}]</button>
              ))}
            </div>
          </Card>

          <section className="rounded-[28px] bg-ink p-6 text-white shadow-card">
            <div className="mb-5 flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wider text-white/50">Audio workspace</p>
              <span className="text-xs text-white/50">{voice?.name ?? character?.name ?? "Nothing selected"}</span>
            </div>
            {job?.status === "completed" ? <Player job={job} /> : <Waveform active={busy} className="my-2" />}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <Button variant="lime" disabled={!!blocked || busy} onClick={() => run(true)} className="!h-12 !px-6 !text-sm">
                <Play size={16} fill="currentColor" /> {busy ? (job?.status === "queued" ? "Queued…" : "Generating…") : "Generate"}
              </Button>
              <Button disabled={!job || busy || !!blocked} onClick={() => run(false)} className="!bg-white/10 hover:!bg-white/20"><RotateCcw size={15} /> Regenerate</Button>
              {job?.status === "completed" && (
                <span className="text-xs text-white/50">
                  {job.audio_seconds?.toFixed(1)} s of audio in {job.generation_seconds?.toFixed(1)} s · {job.chunks} chunk{job.chunks === 1 ? "" : "s"}
                  {job.cached ? " · from cache" : ""}
                </span>
              )}
            </div>
            {blocked && <p className="mt-4 text-[13px] text-white/60">{blocked}</p>}
            {error && <p className="mt-4 rounded-2xl bg-red-500/15 px-4 py-3 text-[13px] text-red-200">{error}</p>}
            {job?.warnings.map((w) => <p key={w} className="mt-3 rounded-2xl bg-amber-400/15 px-4 py-3 text-[13px] text-amber-100">{w}</p>)}
            {job?.status === "completed" && !!job.segments?.length && (
              <ol className="mt-5 space-y-1.5">
                {job.segments.map((s, i) => (
                  <li key={i} className="flex items-start gap-3 rounded-2xl bg-white/5 px-3 py-2 text-[13px]">
                    <span className="w-20 shrink-0 font-mono text-[11px] text-white/40">{s.start.toFixed(1)}–{s.end.toFixed(1)} s</span>
                    <span className="min-w-0 flex-1 text-white/80">{s.text}</span>
                    <span className="shrink-0 rounded-full bg-lime/20 px-2 py-0.5 text-[11px] font-semibold text-lime" title={`chosen by: ${s.source}; expressed by: ${(s.rendered_by ?? []).join(", ") || "nothing"}`}>
                      {s.emotion}{s.source === "tag" ? "" : ` · ${s.source}`}
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>

        <div className="space-y-4">
          <Card>
            <CardTitle action={<a href="#/voices" className="text-xs font-semibold text-muted hover:text-ink">Get voices</a>}>Voice</CardTitle>
            {voices.error && <Notice tone="red">{voices.error}</Notice>}
            {voices.data?.length === 0 && (
              <Empty icon={<Mic2 size={20} />} title="No voices yet" action={<a href="#/voices" className="text-[13px] font-semibold underline underline-offset-4">Add a starter voice</a>}>
                Download a Malayalam or English voice in Voice Library.
              </Empty>
            )}
            <div className="space-y-2">
              {voices.data?.map((v) => (
                <button key={v.id} onClick={() => setTarget({ kind: "voice", id: v.id })} className={row(target?.kind === "voice" && target.id === v.id)}>
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-lime font-semibold text-ink">{v.name[0].toUpperCase()}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold">{v.name}</span>
                    <span className="block truncate text-xs opacity-60">{v.languages.map(languageName).join(", ") || "Language not declared"}</span>
                  </span>
                </button>
              ))}
              {usableCharacters.length > 0 && <p className="pt-3 text-xs font-semibold text-muted">Characters</p>}
              {usableCharacters.map((c) => (
                <button key={c.id} onClick={() => setTarget({ kind: "character", id: c.id })} className={row(target?.kind === "character" && target.id === c.id)}>
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white text-ink"><Users size={16} /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold">{c.name}</span>
                    <span className="block truncate text-xs opacity-60">Switches voice by language</span>
                  </span>
                </button>
              ))}
            </div>
          </Card>

          <Card>
            <CardTitle action={<Badge>{voice ? "From this model" : character ? "Character" : "Per model"}</Badge>}>Generation controls</CardTitle>
            <label className="block text-xs font-semibold text-muted">
              Language
              <select value={language} onChange={(e) => setLanguage(e.target.value)} className="mt-1.5 h-10 w-full rounded-full bg-soft px-4 text-[13px] font-semibold text-ink outline-none">
                <option value="auto">Detect automatically</option>
                <option value="ml">Malayalam</option>
                <option value="en">English</option>
              </select>
            </label>
            {controls.length === 0 && <p className="mt-4 text-[13px] text-muted">Choose a voice to see the controls its engine supports.</p>}
            {controls.map((p) => (
              <label key={p.name} className="mt-4 block text-xs font-semibold text-muted">
                <span className="flex justify-between">
                  {p.label}
                  <span className="text-ink">{Number(params[p.name] ?? p.default).toFixed(p.type === "int" ? 0 : 2)}{p.name === "speed" ? "×" : ""}</span>
                </span>
                <input
                  type="range" min={p.min ?? 0} max={p.max ?? 1} step={p.type === "int" ? 1 : 0.05}
                  value={Number(params[p.name] ?? p.default)}
                  onChange={(e) => setParams((prev) => ({ ...prev, [p.name]: Number(e.target.value) }))}
                  className="mt-2 w-full accent-ink"
                />
              </label>
            ))}
          </Card>

          <Card>
            <CardTitle action={<Badge tone={expressive ? "lime" : "neutral"}>{voice ? (expressive ? "Follows style" : "Speed only") : "Character"}</Badge>}>
              <span className="inline-flex items-center gap-2"><Smile size={16} /> Emotion</span>
            </CardTitle>
            <label className="block text-xs font-semibold text-muted">
              Untagged text
              <select value={emotion} onChange={(e) => setEmotion(e.target.value)} className="mt-1.5 h-10 w-full rounded-full bg-soft px-4 text-[13px] font-semibold text-ink outline-none">
                <option value="auto">Detect per sentence</option>
                <option value="none">Off (also ignore tags)</option>
                {emotions.data?.emotions.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
              </select>
            </label>
            <label className="mt-4 block text-xs font-semibold text-muted">
              <span className="flex justify-between">Strength <span className="text-ink">{strength.toFixed(2)}</span></span>
              <input type="range" min={0} max={2} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} className="mt-2 w-full accent-ink" />
            </label>
            <p className="mt-4 text-xs leading-relaxed text-muted">
              Tags like <code className="rounded bg-soft px-1">[happy]</code> in the text always win.{" "}
              {voice && !expressive && "This voice can only show emotion through speed and variability, so the effect is subtle. Qwen3-TTS 1.7B (CustomVoice / VoiceDesign) follows real emotional instructions."}
              {voice && expressive && "This voice follows spoken-style instructions, so emotions are clearly audible."}
              {character && "A character with an RVC voice also shifts pitch with the emotion."}
              {emotions.data && !emotions.data.classifier.installed && <> Automatic detection is using simple rules; get the <a href="#/huggingface" className="underline">emotion classifier</a> for English.</>}
            </p>
          </Card>
        </div>
      </div>
    </>
  );
}
