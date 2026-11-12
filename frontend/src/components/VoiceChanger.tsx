import { Download } from "lucide-react";
import { useEffect } from "react";
import type { Model, RuntimeBundle, RvcOptions } from "../api";
import { api } from "../api";
import { Button, Notice, useLoad } from "./ui";

const select = "h-10 w-full rounded-full bg-soft px-4 text-[13px] font-semibold text-ink outline-none";

/** Download button for an optional engine bundle, with live status. */
export function BundleInstall({ id, compact = false, onReady }: { id: string; compact?: boolean; onReady?: () => void }) {
  const bundles = useLoad(() => api<RuntimeBundle[]>("/runtime"));
  const bundle = bundles.data?.find((b) => b.id === id);
  const installing = bundle?.state === "installing";

  useEffect(() => {
    if (!installing) return;
    const timer = setInterval(bundles.reload, 2500);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [installing]);
  useEffect(() => {
    if (bundle?.installed) onReady?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle?.installed]);

  if (!bundle || bundle.installed) return null;
  const start = async () => {
    try {
      await api(`/runtime/${id}/install`, { method: "POST" });
    } finally {
      bundles.reload();
    }
  };
  const size = bundle.size_mb >= 1000 ? `${(bundle.size_mb / 1000).toFixed(1)} GB` : `${bundle.size_mb} MB`;
  return (
    <div className="space-y-2">
      {!compact && <p className="text-[13px] text-muted">{bundle.note}</p>}
      {!bundle.available ? (
        <Notice>{bundle.reason}</Notice>
      ) : (
        <Button onClick={start} disabled={installing} className="w-full">
          <Download size={14} /> {installing ? `Downloading ${size}…` : `Download ${bundle.label} · ${size}`}
        </Button>
      )}
      {bundle.state === "error" && <Notice tone="red">{bundle.error}</Notice>}
    </div>
  );
}

/** Pick an imported RVC model and its settings. ``value`` is null when the changer is off. */
export function VoiceChangerControls({ value, onChange, allowNone = true }: {
  value: RvcOptions | null; onChange: (v: RvcOptions | null) => void; allowNone?: boolean;
}) {
  const models = useLoad(() => api<Model[]>("/models"));
  const rvc = (models.data ?? []).filter((m) => m.architecture === "rvc");
  const notReady = rvc.find((m) => !m.inference.ready);

  if (models.data && rvc.length === 0) {
    return (
      <p className="text-[13px] leading-relaxed text-muted">
        No RVC models yet. Import a character ZIP (.pth with its .index) in{" "}
        <a href="#/manager" className="font-semibold text-ink underline underline-offset-4">Model Manager</a>.
      </p>
    );
  }
  return (
    <div className="space-y-4">
      {notReady && <BundleInstall id="rvc" onReady={models.reload} />}
      <label className="block text-xs font-semibold text-muted">
        Character voice
        <select
          value={value?.model_id ?? ""}
          onChange={(e) => onChange(e.target.value ? { model_id: e.target.value, pitch: value?.pitch ?? 0, index_rate: value?.index_rate ?? 0.5 } : null)}
          className={`${select} mt-1.5`}
        >
          {allowNone && <option value="">Off (use the voice as it is)</option>}
          {!allowNone && !value && <option value="">Choose a model</option>}
          {rvc.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
        </select>
      </label>
      {value && (
        <>
          <label className="block text-xs font-semibold text-muted">
            <span className="flex justify-between">Pitch <span className="text-ink">{value.pitch > 0 ? "+" : ""}{value.pitch} semitones</span></span>
            <input type="range" min={-12} max={12} step={1} value={value.pitch} onChange={(e) => onChange({ ...value, pitch: Number(e.target.value) })} className="mt-2 w-full accent-ink" />
          </label>
          <label className="block text-xs font-semibold text-muted">
            <span className="flex justify-between">Index strength <span className="text-ink">{value.index_rate.toFixed(2)}</span></span>
            <input type="range" min={0} max={1} step={0.05} value={value.index_rate} onChange={(e) => onChange({ ...value, index_rate: Number(e.target.value) })} className="mt-2 w-full accent-ink" />
          </label>
          <p className="text-xs text-muted">Raise the pitch for a child or female character on a male voice, lower it for the reverse. Conversion takes about as long as the audio itself on this CPU.</p>
        </>
      )}
    </div>
  );
}
