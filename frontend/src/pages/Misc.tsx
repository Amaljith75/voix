import { Check, Download, History as HistoryIcon, Mic2, RotateCcw, Search, Trash2 } from "lucide-react";
import { useState } from "react";
import type { CatalogVoice, Job, Voice } from "../api";
import { api, generate, languageName } from "../api";
import { LicenseBadge } from "../components/ModelBits";
import { Player } from "../components/Player";
import { Badge, Button, Card, CardTitle, Empty, LoadState, Notice, PageHeader, useLoad } from "../components/ui";

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
          </Card>
        ))}
      </div>

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
      const { speed = 1, emotion = null, ...rest } = job.params;
      const result = await generate({
        text: job.text, language: job.language, speed, emotion, params: rest, rvc: job.rvc, use_cache: false, output_format: "wav",
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
                    {new Date(j.created_at).toLocaleString()} · {j.kind === "conversion" ? "voice changer" : `${j.character_id ?? j.voice_id} · ${languageName(j.language)}`}
                    {j.kind !== "conversion" && j.rvc ? " · voice changer" : ""}
                    {j.status === "completed" && ` · ${j.audio_seconds?.toFixed(1)} s`}
                  </span>
                </button>
                <Badge tone={j.status === "completed" ? "lime" : j.status === "failed" ? "red" : "amber"}>{j.status}</Badge>
                {j.kind !== "conversion" && <button aria-label="Regenerate" onClick={() => regenerate(j)} className="grid h-9 w-9 place-items-center rounded-full bg-soft hover:bg-line"><RotateCcw size={15} /></button>}
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
