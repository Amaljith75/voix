import { FileAudio, Upload, Wand2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { Character, ConversionModel, Job, Parameter } from "../api";
import { api, formatBytes, uploadForm, waitJob } from "../api";
import { Player } from "../components/Player";
import { Badge, Button, Card, CardTitle, Empty, Notice, PageHeader, Waveform, cx, useLoad } from "../components/ui";

const ACCEPT = ".mp3,.wav,.flac,.ogg,.oga,.opus,.m4a,.aac,.webm,audio/*";

export function ParamControl({ p, value, onChange }: { p: Parameter; value: number | string; onChange: (v: number | string) => void }) {
  if (p.type === "choice")
    return (
      <label className="mt-4 block text-xs font-semibold text-muted">
        {p.label}
        <select value={String(value)} onChange={(e) => onChange(e.target.value)} className="mt-1.5 h-10 w-full rounded-full bg-soft px-4 text-[13px] font-semibold text-ink outline-none">
          {p.choices?.map((c) => <option key={c} value={c}>{c.toUpperCase()}</option>)}
        </select>
      </label>
    );
  const step = p.name === "pitch_shift" ? 1 : p.type === "int" ? 1 : 0.05;
  return (
    <label className="mt-4 block text-xs font-semibold text-muted">
      <span className="flex justify-between">
        {p.label}
        <span className="text-ink">{Number(value).toFixed(step === 1 ? 0 : 2)}</span>
      </span>
      <input type="range" min={p.min ?? 0} max={p.max ?? 1} step={step} value={Number(value)} onChange={(e) => onChange(Number(e.target.value))} className="mt-2 w-full accent-ink" />
    </label>
  );
}

export default function VoiceChanger() {
  const models = useLoad(() => api<ConversionModel[]>("/models?kind=voice_conversion"));
  const characters = useLoad(() => api<Character[]>("/characters"));
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState<{ kind: "model" | "character"; id: string } | null>(null);
  const [params, setParams] = useState<Record<string, number | string>>({});
  const [job, setJob] = useState<Job | null>(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const rvcCharacters = (characters.data ?? []).filter((c) => c.rvc_model_id);
  const modelId = target?.kind === "model" ? target.id : rvcCharacters.find((c) => c.id === target?.id)?.rvc_model_id;
  const model = models.data?.find((m) => m.id === modelId);
  const controls = model?.parameters ?? [];

  useEffect(() => {
    if (!target && models.data?.length) setTarget({ kind: "model", id: models.data[0].id });
  }, [models.data, target]);
  useEffect(() => {
    const base = Object.fromEntries(controls.map((p) => [p.name, p.default]));
    const character = target?.kind === "character" ? rvcCharacters.find((c) => c.id === target.id) : undefined;
    setParams({ ...base, ...(character?.rvc_settings ?? {}) });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target?.kind, target?.id, model?.id]);

  const blocked = !file ? "Choose an audio file." : !target ? "Choose an RVC voice." : model && !model.inference.ready ? model.inference.reason : "";

  async function run() {
    if (blocked || !file || !target) return;
    setBusy(true);
    setError(null);
    setJob(null);
    setProgress(0);
    const form = new FormData();
    form.append("file", file);
    form.append(target.kind === "model" ? "model_id" : "character_id", target.id);
    form.append("output_format", "wav");
    Object.entries(params).forEach(([k, v]) => form.append(k, String(v)));
    try {
      const first = await uploadForm<Job>("/convert", form, setProgress);
      const done = await waitJob(first, setJob);
      if (done.status === "failed") setError(done.error ?? "Conversion failed.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const row = (active: boolean) => cx("flex w-full items-center gap-3 rounded-2xl p-3 text-left transition", active ? "bg-ink text-white" : "bg-soft hover:bg-line");

  return (
    <>
      <PageHeader title="Voice Changer" subtitle="Upload a recording (MP3, WAV, FLAC, OGG) and re-voice it with an RVC character voice. The words and timing stay; the timbre changes." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-4">
          <Card>
            <CardTitle>Source audio</CardTitle>
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-3xl border border-dashed border-line px-6 py-10 text-center hover:bg-soft"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}>
              <span className="mb-3 grid h-12 w-12 place-items-center rounded-full bg-soft">{file ? <FileAudio size={20} /> : <Upload size={20} />}</span>
              <span className="text-sm font-semibold">{file ? file.name : "Drop an audio file here, or click to choose"}</span>
              <span className="mt-1 text-xs text-muted">{file ? formatBytes(file.size) : "Clean speech works best: one speaker, little music or echo. Up to 15 minutes."}</span>
              <input type="file" accept={ACCEPT} className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
          </Card>

          <section className="rounded-[28px] bg-ink p-6 text-white shadow-card">
            <div className="mb-5 flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-wider text-white/50">Converted audio</p>
              <span className="text-xs text-white/50">{model?.name ?? "No voice selected"}</span>
            </div>
            {job?.status === "completed" ? <Player job={job} /> : <Waveform active={busy} className="my-2" />}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <Button variant="lime" disabled={!!blocked || busy} onClick={run} className="!h-12 !px-6 !text-sm">
                <Wand2 size={16} /> {busy ? (progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : job?.status === "queued" ? "Queued…" : "Converting…") : "Convert voice"}
              </Button>
              {job?.status === "completed" && <span className="text-xs text-white/50">{job.audio_seconds?.toFixed(1)} s converted in {job.generation_seconds?.toFixed(1)} s</span>}
            </div>
            {blocked && <p className="mt-4 text-[13px] text-white/60">{blocked}</p>}
            {error && <p className="mt-4 rounded-2xl bg-red-500/15 px-4 py-3 text-[13px] text-red-200">{error}</p>}
          </section>
        </div>

        <div className="space-y-4">
          <Card>
            <CardTitle action={<a href="#/huggingface" className="text-xs font-semibold text-muted hover:text-ink">Get RVC voices</a>}>RVC voice</CardTitle>
            {models.data?.length === 0 && (
              <Empty icon={<Wand2 size={20} />} title="No RVC voices yet">Install a character voice (.pth + .index) from Hugging Face, or import its ZIP in Model Manager.</Empty>
            )}
            <div className="space-y-2">
              {models.data?.map((m) => (
                <button key={m.id} onClick={() => setTarget({ kind: "model", id: m.id })} className={row(target?.kind === "model" && target.id === m.id)}>
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-lime font-semibold text-ink">{m.name[0]?.toUpperCase()}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold">{m.name}</span>
                    <span className="block truncate text-xs opacity-60">{m.inference.ready ? "RVC" : "Not ready"} · {formatBytes(m.size_bytes)}</span>
                  </span>
                </button>
              ))}
              {rvcCharacters.length > 0 && <p className="pt-3 text-xs font-semibold text-muted">Characters</p>}
              {rvcCharacters.map((c) => (
                <button key={c.id} onClick={() => setTarget({ kind: "character", id: c.id })} className={row(target?.kind === "character" && target.id === c.id)}>
                  <span className="grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white font-semibold text-ink">{c.name[0]?.toUpperCase()}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-semibold">{c.name}</span>
                    <span className="block truncate text-xs opacity-60">Uses the character's settings</span>
                  </span>
                </button>
              ))}
            </div>
          </Card>
          <Card>
            <CardTitle action={<Badge>RVC</Badge>}>Conversion controls</CardTitle>
            {controls.length === 0 && <p className="text-[13px] text-muted">Choose a voice to see its controls.</p>}
            {controls.map((p) => <ParamControl key={p.name} p={p} value={params[p.name] ?? p.default} onChange={(v) => setParams((prev) => ({ ...prev, [p.name]: v }))} />)}
            {controls.length > 0 && (
              <div className="mt-4"><Notice>Pitch shift: +12 makes a man's recording suit a child or female voice, −12 the other way. Protect keeps consonants clear; lower it if breaths sound robotic.</Notice></div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
