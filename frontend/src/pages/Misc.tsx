import { Check, Download, History as HistoryIcon, Mic2, Plus, RotateCcw, Search, Trash2 } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import type { CatalogVoice, Job, Model, Voice } from "../api";
import { api, generate, languageName, uploadForm } from "../api";
import { LicenseBadge } from "../components/ModelBits";
import { Player } from "../components/Player";
import { Badge, Button, Card, CardTitle, Empty, LoadState, Notice, PageHeader, useLoad } from "../components/ui";

const input = "h-11 w-full rounded-2xl bg-soft px-4 text-sm outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10";

/** Make another voice on a Qwen3-TTS model: a built-in speaker, a described voice, or a clone. */
function QwenVoiceForm({ onDone }: { onDone: () => void }) {
  const models = useLoad(() => api<Model[]>("/models"));
  const qwen = (models.data ?? []).filter((m) => m.architecture === "qwen3_tts" && m.completeness === "complete");
  const [modelId, setModelId] = useState("");
  const full = useLoad(() => (modelId ? api<Model>(`/models/${modelId}`) : Promise.resolve(null)), [modelId]);
  const details = (full.data?.report?.detection.details ?? {}) as { qwen_model_type?: string; speakers?: string[]; size?: string };
  const [name, setName] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [description, setDescription] = useState("");
  const [refFile, setRefFile] = useState<File | null>(null);
  const [refText, setRefText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  if (!qwen.length) return null;
  const type = details.qwen_model_type;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const settings = type === "custom_voice" ? { speaker } : type === "voice_design" ? { voice_description: description } : {};
      const voice = await api<Voice>("/voices", { method: "POST", body: JSON.stringify({ model_id: modelId, name, settings }) });
      if (type === "base" && refFile) {
        const form = new FormData();
        form.append("file", refFile);
        form.append("ref_text", refText);
        await uploadForm(`/voices/${voice.id}/reference`, form, () => {});
      }
      setName("");
      onDone();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="mt-4">
      <CardTitle>New Qwen3-TTS voice</CardTitle>
      <form onSubmit={submit} className="grid gap-3 md:grid-cols-2">
        <select required value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Qwen3-TTS model" className={input}>
          <option value="">Choose a Qwen3-TTS model</option>
          {qwen.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
        <input required value={name} onChange={(e) => setName(e.target.value)} placeholder="Voice name, e.g. Leo English" aria-label="Voice name" className={input} />
        {type === "custom_voice" && (
          <select required value={speaker} onChange={(e) => setSpeaker(e.target.value)} aria-label="Speaker" className={input}>
            <option value="">Built-in speaker</option>
            {details.speakers?.map((s) => <option key={s} value={s}>{s.replace("_", " ")}</option>)}
          </select>
        )}
        {type === "voice_design" && (
          <textarea required value={description} onChange={(e) => setDescription(e.target.value)} aria-label="Voice description"
            placeholder="Describe the voice: e.g. a bright, playful young boy, quick and bouncy, slightly high pitch" className={`${input} h-20 py-3 md:col-span-2`} />
        )}
        {type === "base" && (
          <>
            <input required type="file" accept="audio/*,.mp3,.wav,.flac,.ogg" onChange={(e) => setRefFile(e.target.files?.[0] ?? null)} aria-label="Reference clip" className={`${input} py-2.5`} />
            <input value={refText} onChange={(e) => setRefText(e.target.value)} placeholder="Exact words spoken in the clip (better cloning)" aria-label="Reference transcript" className={input} />
            <p className="text-xs text-muted md:col-span-2">A clean 3-15 second clip of one speaker. Only clone a voice you have the right to use.</p>
          </>
        )}
        {error && <div className="md:col-span-2"><Notice tone="red">{error}</Notice></div>}
        <Button type="submit" disabled={busy || !modelId || !name.trim()} className="md:col-span-2"><Plus size={15} /> {busy ? "Saving…" : "Add voice"}</Button>
      </form>
    </Card>
  );
}

export function VoiceLibraryPage() {
  const voices = useLoad(() => api<Voice[]>("/voices"));
  const catalog = useLoad(() => api<CatalogVoice[]>("/catalog"));
  const [installing, setInstalling] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function install(entry: CatalogVoice) {
    setInstalling(entry.id);
    setError(null);
    try {
      await api(`/catalog/${entry.id}/install`, { method: "POST" });
      voices.reload();
      catalog.reload();
    } catch (e) {
      setError(`${entry.name}: ${(e as Error).message}`);
    } finally {
      setInstalling(null);
    }
  }

  return (
    <>
      <PageHeader title="Voice Library" subtitle="Voices you can generate with. Each one is a text-to-speech model VOIX can run on this machine." />
      <LoadState error={voices.error} loading={!voices.data && !voices.error} />
      {voices.data?.length === 0 && (
        <Card><Empty icon={<Mic2 size={20} />} title="No voices installed">Add a starter voice below, or import a Piper voice ZIP in Model Manager.</Empty></Card>
      )}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {voices.data?.map((v) => (
          <Card key={v.id}>
            <div className="flex items-start justify-between">
              <span className="grid h-12 w-12 place-items-center rounded-full bg-lime text-lg font-bold">{v.name[0].toUpperCase()}</span>
              <Badge tone={v.ready ? "lime" : "amber"}>{v.ready ? "Ready" : "Not ready"}</Badge>
            </div>
            <p className="mt-4 truncate font-semibold">{v.name}</p>
            <p className="truncate font-mono text-[11px] text-muted">{v.id}</p>
            <p className="mt-1 text-[13px] text-muted">{v.description}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {v.languages.map((l) => <Badge key={l} tone="dark">{languageName(l)}</Badge>)}
              <LicenseBadge model={v} />
            </div>
            {!v.ready && <p className="mt-3 text-xs text-amber-800">{v.reason}</p>}
            <button onClick={async () => { if (window.confirm(`Delete voice "${v.name}"?`)) { await api(`/voices/${v.id}`, { method: "DELETE" }); voices.reload(); } }}
              className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-red-700 hover:underline"><Trash2 size={12} /> Delete voice</button>
          </Card>
        ))}
      </div>
      <QwenVoiceForm onDone={voices.reload} />

      <Card className="mt-4">
        <CardTitle>Starter voices</CardTitle>
        <p className="-mt-2 mb-4 text-[13px] text-muted">Downloaded from the public Piper voice collection on Hugging Face, then inspected like any other import. About 60 MB each.</p>
        {error && <div className="mb-3"><Notice tone="red">{error}</Notice></div>}
        <LoadState error={catalog.error} loading={!catalog.data && !catalog.error} />
        <ul className="divide-y divide-line">
          {catalog.data?.map((c) => (
            <li key={c.id} className="flex flex-wrap items-center gap-x-4 gap-y-2 py-4">
              <div className="min-w-0 flex-1 basis-64">
                <p className="text-sm font-semibold">{c.name} <span className="font-medium text-muted">· {languageName(c.language)} · {c.gender}</span></p>
                <p className="mt-0.5 text-xs text-muted">{c.license_note}</p>
              </div>
              <LicenseBadge model={c} />
              {c.installed ? (
                <Badge tone="lime"><Check size={12} /> Installed</Badge>
              ) : (
                <Button onClick={() => install(c)} disabled={installing !== null} className="!h-9">
                  <Download size={14} /> {installing === c.id ? "Downloading…" : `Get · ${c.size_mb} MB`}
                </Button>
              )}
            </li>
          ))}
        </ul>
      </Card>
    </>
  );
}

export function HistoryPage() {
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const history = useLoad(() => api<Job[]>(`/history?q=${encodeURIComponent(query)}&language=${language}`), [query, language]);

  async function remove(job: Job) {
    if (!window.confirm("Delete this generation and its audio?")) return;
    try {
      await api(`/history/${job.job_id}`, { method: "DELETE" });
      history.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function regenerate(job: Job) {
    setError(null);
    try {
      const { speed = 1, ...rest } = job.params;
      const result = await generate({
        text: job.text, language: job.language, speed, params: rest, use_cache: false, output_format: "wav",
        ...(job.character_id ? { character_id: job.character_id } : { voice_id: job.voice_id }),
      });
      if (result.status === "failed") setError(result.error ?? "Generation failed.");
      history.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Generation History" subtitle="Replay, download, regenerate and search past audio." />
      <div className="mb-4 flex flex-wrap gap-2">
        <label className="flex h-10 min-w-0 flex-1 items-center gap-2 rounded-full bg-white px-4 sm:max-w-sm">
          <Search size={15} className="text-muted" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search text" aria-label="Search history" className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted" />
        </label>
        <select value={language} onChange={(e) => setLanguage(e.target.value)} aria-label="Filter by language" className="h-10 rounded-full bg-white px-4 text-[13px] font-semibold outline-none">
          <option value="">All languages</option>
          <option value="ml">Malayalam</option>
          <option value="en">English</option>
        </select>
      </div>
      {error && <div className="mb-3"><Notice tone="red">{error}</Notice></div>}
      <Card>
        <LoadState error={history.error} loading={!history.data && !history.error} />
        {history.data?.length === 0 && (
          <Empty icon={<HistoryIcon size={20} />} title={query || language ? "Nothing matches" : "Nothing generated yet"}>
            {query || language ? "Try a different search or filter." : "Audio you generate in Voice Studio or through the API is listed here."}
          </Empty>
        )}
        <ul className="divide-y divide-line">
          {history.data?.map((j) => (
            <li key={j.job_id} className="py-4">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
                <button className="min-w-0 flex-1 basis-64 text-left" onClick={() => setOpen(open === j.job_id ? null : j.job_id)} disabled={j.status !== "completed"}>
                  <span className="line-clamp-2 text-sm font-medium">{j.text}</span>
                  <span className="mt-1 block text-xs text-muted">
                    {new Date(j.created_at).toLocaleString()} · {j.character_id ?? j.voice_id} · {languageName(j.language)}
                    {j.status === "completed" && ` · ${j.audio_seconds?.toFixed(1)} s`}
                  </span>
                </button>
                <Badge tone={j.status === "completed" ? "lime" : j.status === "failed" ? "red" : "amber"}>{j.status}</Badge>
                <button aria-label="Regenerate" onClick={() => regenerate(j)} className="grid h-9 w-9 place-items-center rounded-full bg-soft hover:bg-line"><RotateCcw size={15} /></button>
                <button aria-label="Delete" onClick={() => remove(j)} className="grid h-9 w-9 place-items-center rounded-full bg-red-50 text-red-700 hover:bg-red-100"><Trash2 size={15} /></button>
              </div>
              {j.status === "failed" && <p className="mt-2 text-xs text-red-700">{j.error}</p>}
              {open === j.job_id && j.status === "completed" && <div className="mt-4 rounded-3xl bg-soft p-4"><Player job={j} dark={false} /></div>}
            </li>
          ))}
        </ul>
      </Card>
    </>
  );
}
