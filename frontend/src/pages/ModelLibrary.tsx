import { Library, Search } from "lucide-react";
import { useState } from "react";
import type { Model } from "../api";
import { api, formatBytes } from "../api";
import { KindBadge, Languages, LicenseBadge, ModelDrawer } from "../components/ModelBits";
import { Badge, Button, Empty, LoadState, PageHeader, cx, useLoad } from "../components/ui";

const FILTERS = [
  { id: "all", label: "All" },
  { id: "tts", label: "Text-to-speech" },
  { id: "voice_conversion", label: "Voice conversion" },
  { id: "unknown", label: "Unidentified" },
] as const;

export default function ModelLibrary() {
  const { data, error } = useLoad(() => api<Model[]>("/models"));
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>("all");
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const q = query.trim().toLowerCase();
  const shown = (data ?? []).filter(
    (m) =>
      (filter === "all" || m.kind === filter) &&
      (!q || [m.name, m.architecture_label, ...m.tags, ...m.languages].some((s) => s.toLowerCase().includes(q))),
  );

  return (
    <>
      <PageHeader
        title="Model Library"
        subtitle="Every imported package, labelled by what it actually is."
        action={<a href="#/manager"><Button>Import model</Button></a>}
      />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            className={cx("rounded-full px-4 py-2 text-[13px] font-semibold transition", filter === f.id ? "bg-ink text-white" : "bg-white text-ink/60 hover:text-ink")}
          >
            {f.label}
          </button>
        ))}
        <label className="ml-auto flex h-10 w-full items-center gap-2 rounded-full bg-white px-4 sm:w-64">
          <Search size={15} className="text-muted" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models" aria-label="Search models" className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted" />
        </label>
      </div>

      <LoadState error={error} loading={!data && !error} />
      {data && shown.length === 0 && (
        <div className="rounded-[28px] bg-white p-6 shadow-card">
          <Empty icon={<Library size={20} />} title={data.length ? "No models match" : "No models installed"}>
            {data.length ? "Try a different filter or search." : "Import a ZIP in Model Manager to see it here."}
          </Empty>
        </div>
      )}
      <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
        {shown.map((m) => (
          <button key={m.id} onClick={() => setOpen(m.id)} className="rounded-[28px] bg-white p-5 text-left shadow-card transition hover:-translate-y-0.5">
            <div className="flex items-start justify-between gap-3">
              <span className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-ink text-sm font-bold text-lime">{m.name[0]?.toUpperCase()}</span>
              <span className={cx("mt-1 flex items-center gap-1.5 text-[11px] font-semibold", m.active ? "text-lime-deep" : "text-muted")}>
                <span className={cx("h-1.5 w-1.5 rounded-full", m.active ? "bg-lime-deep" : "bg-muted/50")} />
                {m.active ? "Active" : "Inactive"}
              </span>
            </div>
            <p className="mt-4 truncate text-base font-semibold">{m.name}</p>
            <p className="text-xs text-muted">{m.architecture_label} · v{m.version} · {formatBytes(m.size_bytes)}</p>
            <div className="mt-4 flex flex-wrap gap-1.5">
              <KindBadge model={m} />
              <LicenseBadge model={m} />
              {!m.inference.ready && <Badge>No adapter yet</Badge>}
            </div>
            <p className="mt-4 border-t border-line pt-3 text-xs">
              <span className="text-muted">Languages · </span>
              <Languages model={m} />
            </p>
          </button>
        ))}
      </div>
      {open && <ModelDrawer id={open} onClose={() => setOpen(null)} />}
    </>
  );
}
