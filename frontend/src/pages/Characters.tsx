import { Pencil, Plus, Trash2, Users, X } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import type { Character, ConversionModel, EmotionInfo, Voice } from "../api";
import { api } from "../api";
import { Badge, Button, Card, CardTitle, Empty, LoadState, Notice, PageHeader, useLoad } from "../components/ui";
import { ParamControl } from "./VoiceChanger";

const field = "h-11 w-full rounded-2xl bg-soft px-4 text-sm outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10";

type Form = {
  id: string; name: string; description: string; style: string; default_speed: number;
  voice_ml: string; voice_en: string; rvc_model_id: string; rvc_settings: Record<string, number | string>;
  emotion_mode: string; default_emotion: string; emotion_strength: number;
};
const EMPTY: Form = {
  id: "", name: "", description: "", style: "", default_speed: 1, voice_ml: "", voice_en: "", rvc_model_id: "",
  rvc_settings: {}, emotion_mode: "auto", default_emotion: "neutral", emotion_strength: 1,
};
const fromCharacter = (c: Character): Form => ({
  id: c.id, name: c.name, description: c.description, style: c.style ?? "", default_speed: c.default_speed,
  voice_ml: c.voice_ml ?? "", voice_en: c.voice_en ?? "", rvc_model_id: c.rvc_model_id ?? "", rvc_settings: c.rvc_settings ?? {},
  emotion_mode: c.emotion_mode ?? "auto", default_emotion: c.default_emotion ?? "neutral", emotion_strength: c.emotion_strength ?? 1,
});

export default function Characters() {
  const characters = useLoad(() => api<Character[]>("/characters"));
  const voices = useLoad(() => api<Voice[]>("/voices"));
  const rvc = useLoad(() => api<ConversionModel[]>("/models?kind=voice_conversion"));
  const emotions = useLoad(() => api<EmotionInfo>("/emotions"));
  const [form, setForm] = useState<Form>(EMPTY);
  const [editing, setEditing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  const rvcModel = rvc.data?.find((m) => m.id === form.rvc_model_id);

  async function save(e: FormEvent) {
    e.preventDefault();
    setError(null);
    const body = {
      ...form, id: editing ? undefined : form.id.trim() || undefined,
      voice_ml: form.voice_ml || null, voice_en: form.voice_en || null, rvc_model_id: form.rvc_model_id || null,
      rvc_settings: form.rvc_model_id ? form.rvc_settings : {},
    };
    try {
      await api(editing ? `/characters/${editing}` : "/characters", { method: editing ? "PUT" : "POST", body: JSON.stringify(body) });
      setForm(EMPTY);
      setEditing(null);
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

  const voiceSelect = (label: string, key: "voice_ml" | "voice_en", lang: string) => (
    <label className="block text-xs font-semibold text-muted">
      {label}
      <select value={form[key]} onChange={(e) => set(key, e.target.value)} className={`${field} mt-1.5 font-medium text-ink`} disabled={!voices.data?.length}>
        <option value="">{voices.data?.length ? "Not assigned" : "No voices available yet"}</option>
        {voices.data?.map((v) => (
          <option key={v.id} value={v.id}>{v.name}{v.languages.length && !v.languages.includes(lang) ? " (other language)" : ""}</option>
        ))}
      </select>
    </label>
  );

  return (
    <>
      <PageHeader
        title="Characters"
        subtitle="Tutor characters keep a stable ID for the SCOL app. Each one has a Malayalam and an English voice, an optional RVC voice that gives both the same timbre, a speaking style and emotion settings."
      />
      <div className="grid gap-4 xl:grid-cols-[1fr_400px]">
        <Card>
          <CardTitle>All characters</CardTitle>
          <LoadState error={characters.error} loading={!characters.data && !characters.error} />
          {characters.data?.length === 0 && (
            <Empty icon={<Users size={20} />} title="No characters yet">Create the first one. Use the same ID the app already uses, e.g. minnu or tiko.</Empty>
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
                  <button aria-label={`Edit ${c.name}`} onClick={() => { setEditing(c.id); setForm(fromCharacter(c)); }} className="grid h-8 w-8 place-items-center rounded-full bg-white hover:bg-line">
                    <Pencil size={14} />
                  </button>
                  <button aria-label={`Delete ${c.name}`} onClick={() => remove(c)} className="grid h-8 w-8 place-items-center rounded-full bg-white text-red-700 hover:bg-red-50">
                    <Trash2 size={14} />
                  </button>
                </div>
                {c.description && <p className="mt-3 text-[13px] text-ink/70">{c.description}</p>}
                {c.style && <p className="mt-2 text-[12px] italic text-ink/60">“{c.style}”</p>}
                <div className="mt-3 flex flex-wrap gap-1.5">
                  <Badge tone={c.voice_ml ? "lime" : "neutral"}>Malayalam · {c.voice_ml ?? "no voice"}</Badge>
                  <Badge tone={c.voice_en ? "lime" : "neutral"}>English · {c.voice_en ?? "no voice"}</Badge>
                  <Badge tone={c.rvc_model_id ? "sky" : "neutral"}>RVC · {rvc.data?.find((m) => m.id === c.rvc_model_id)?.name ?? (c.rvc_model_id ? "model" : "none")}</Badge>
                  <Badge>Emotion · {c.emotion_mode}</Badge>
                  <Badge>{c.default_speed.toFixed(2)}× speed</Badge>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <CardTitle action={editing && <button onClick={() => { setEditing(null); setForm(EMPTY); }} className="inline-flex items-center gap-1 text-xs font-semibold text-muted"><X size={13} /> Cancel</button>}>
            {editing ? `Edit ${editing}` : "New character"}
          </CardTitle>
          <form onSubmit={save} className="space-y-3">
            <input required value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="Name, e.g. Mio" aria-label="Character name" className={field} />
            {!editing && (
              <input value={form.id} onChange={(e) => set("id", e.target.value.toLowerCase())} pattern="[a-z0-9_]{1,60}" placeholder="ID (optional), e.g. minnu" aria-label="Character ID" className={`${field} font-mono`} />
            )}
            <textarea value={form.description} onChange={(e) => set("description", e.target.value)} placeholder="Short description" aria-label="Description" className={`${field} h-16 resize-none py-3`} />
            <textarea value={form.style} onChange={(e) => set("style", e.target.value)} placeholder="Speaking style, e.g. a warm, gentle storyteller talking to a young child" aria-label="Speaking style" className={`${field} h-16 resize-none py-3`} />
            {voiceSelect("Malayalam voice", "voice_ml", "ml")}
            {voiceSelect("English voice", "voice_en", "en")}
            <label className="block text-xs font-semibold text-muted">
              Default speed · {form.default_speed.toFixed(2)}×
              <input type="range" min={0.5} max={2} step={0.05} value={form.default_speed} onChange={(e) => set("default_speed", Number(e.target.value))} className="mt-2 w-full accent-ink" />
            </label>

            <div className="rounded-2xl border border-line p-4">
              <label className="block text-xs font-semibold text-muted">
                Character voice (RVC)
                <select value={form.rvc_model_id} onChange={(e) => setForm((f) => ({ ...f, rvc_model_id: e.target.value, rvc_settings: {} }))} className={`${field} mt-1.5 font-medium text-ink`}>
                  <option value="">None: use the TTS voices as they are</option>
                  {rvc.data?.map((m) => <option key={m.id} value={m.id}>{m.name}{m.inference.ready ? "" : " (not ready)"}</option>)}
                </select>
              </label>
              {rvcModel && rvcModel.parameters.map((p) => (
                <ParamControl key={p.name} p={p} value={form.rvc_settings[p.name] ?? p.default} onChange={(v) => set("rvc_settings", { ...form.rvc_settings, [p.name]: v })} />
              ))}
              {!rvc.data?.length && <p className="mt-2 text-[11px] text-muted">Install an RVC voice on the <a className="underline" href="#/huggingface">Hugging Face</a> page to give this character its own timbre.</p>}
            </div>

            <div className="rounded-2xl border border-line p-4">
              <label className="block text-xs font-semibold text-muted">
                Emotion for untagged text
                <select value={form.emotion_mode} onChange={(e) => set("emotion_mode", e.target.value)} className={`${field} mt-1.5 font-medium text-ink`}>
                  <option value="auto">Detect per sentence</option>
                  <option value="none">Off</option>
                  {emotions.data?.emotions.map((e) => <option key={e.id} value={e.id}>Always {e.label.toLowerCase()}</option>)}
                </select>
              </label>
              <label className="mt-3 block text-xs font-semibold text-muted">
                When nothing is detected
                <select value={form.default_emotion} onChange={(e) => set("default_emotion", e.target.value)} className={`${field} mt-1.5 font-medium text-ink`}>
                  {emotions.data?.emotions.map((e) => <option key={e.id} value={e.id}>{e.label}</option>)}
                </select>
              </label>
              <label className="mt-3 block text-xs font-semibold text-muted">
                Strength · {form.emotion_strength.toFixed(2)}
                <input type="range" min={0} max={2} step={0.05} value={form.emotion_strength} onChange={(e) => set("emotion_strength", Number(e.target.value))} className="mt-2 w-full accent-ink" />
              </label>
            </div>
            {error && <Notice tone="red">{error}</Notice>}
            <Button type="submit" className="w-full" disabled={!form.name.trim()}><Plus size={15} /> {editing ? "Save changes" : "Create character"}</Button>
          </form>
        </Card>
      </div>
    </>
  );
}
