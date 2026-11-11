import { Plus, Trash2, Users } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import type { Character, Voice } from "../api";
import { api } from "../api";
import { Badge, Button, Card, CardTitle, Empty, LoadState, Notice, PageHeader, useLoad } from "../components/ui";

const field = "h-11 w-full rounded-2xl bg-soft px-4 text-sm outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10";

export default function Characters() {
  const characters = useLoad(() => api<Character[]>("/characters"));
  const voices = useLoad(() => api<Voice[]>("/voices"));
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [speed, setSpeed] = useState(1);
  const [voiceMl, setVoiceMl] = useState("");
  const [voiceEn, setVoiceEn] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function create(e: FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api("/characters", {
        method: "POST",
        body: JSON.stringify({ name, description, default_speed: speed, voice_ml: voiceMl || null, voice_en: voiceEn || null }),
      });
      setName("");
      setDescription("");
      setSpeed(1);
      characters.reload();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function remove(c: Character) {
    if (!window.confirm(`Delete character "${c.name}"? The mobile app will no longer resolve the ID ${c.id}.`)) return;
    try {
      await api(`/characters/${c.id}`, { method: "DELETE" });
      characters.reload();
    } catch (err) {
      setError((err as Error).message);
    }
  }

  const voiceSelect = (label: string, value: string, set: (v: string) => void) => (
    <label className="block text-xs font-semibold text-muted">
      {label}
      <select value={value} onChange={(e) => set(e.target.value)} className={`${field} mt-1.5 font-medium text-ink`} disabled={!voices.data?.length}>
        <option value="">{voices.data?.length ? "Not assigned" : "No voices available yet"}</option>
        {voices.data?.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
      </select>
    </label>
  );

  return (
    <>
      <PageHeader title="Characters" subtitle="Tutor characters keep a stable ID for the SCOL app. The voices behind them can change at any time." />
      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <Card>
          <CardTitle>All characters</CardTitle>
          <LoadState error={characters.error} loading={!characters.data && !characters.error} />
          {characters.data?.length === 0 && (
            <Empty icon={<Users size={20} />} title="No characters yet">Create the first one, for example a friendly teacher or a funny robot.</Empty>
          )}
          <ul className="grid gap-3 md:grid-cols-2">
            {characters.data?.map((c) => (
              <li key={c.id} className="rounded-3xl bg-soft p-4">
                <div className="flex items-start gap-3">
                  <span className="grid h-12 w-12 shrink-0 place-items-center rounded-full bg-lime text-lg font-bold">{c.name[0]?.toUpperCase()}</span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-semibold">{c.name}</p>
                    <p className="truncate font-mono text-[11px] text-muted">{c.id}</p>
                  </div>
                  <button aria-label={`Delete ${c.name}`} onClick={() => remove(c)} className="grid h-8 w-8 place-items-center rounded-full bg-white text-red-700 hover:bg-red-50">
                    <Trash2 size={14} />
                  </button>
                </div>
                {c.description && <p className="mt-3 text-[13px] text-ink/70">{c.description}</p>}
                <div className="mt-3 flex flex-wrap gap-1.5">
                  <Badge tone={c.voice_ml ? "lime" : "neutral"}>Malayalam · {c.voice_ml ?? "no voice"}</Badge>
                  <Badge tone={c.voice_en ? "lime" : "neutral"}>English · {c.voice_en ?? "no voice"}</Badge>
                  <Badge>{c.default_speed.toFixed(2)}× speed</Badge>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <CardTitle>New character</CardTitle>
          <form onSubmit={create} className="space-y-3">
            <input required value={name} onChange={(e) => setName(e.target.value)} placeholder="Name, e.g. Friendly Teacher" aria-label="Character name" className={field} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Short description" aria-label="Description" className={`${field} h-24 resize-none py-3`} />
            {voiceSelect("Malayalam voice", voiceMl, setVoiceMl)}
            {voiceSelect("English voice", voiceEn, setVoiceEn)}
            <label className="block text-xs font-semibold text-muted">
              Default speed · {speed.toFixed(2)}×
              <input type="range" min={0.5} max={2} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="mt-2 w-full accent-ink" />
            </label>
            {error && <Notice tone="red">{error}</Notice>}
            <Button type="submit" className="w-full" disabled={!name.trim()}><Plus size={15} /> Create character</Button>
          </form>
        </Card>
      </div>
    </>
  );
}
