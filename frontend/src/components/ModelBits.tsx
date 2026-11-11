import { AlertTriangle, CheckCircle2, X } from "lucide-react";
import type { ReactNode } from "react";
import type { Model } from "../api";
import { api, formatBytes, languageName } from "../api";
import { Badge, LoadState, Notice, useLoad } from "./ui";

export function KindBadge({ model }: { model: Model }) {
  if (model.kind === "tts") {
    return model.completeness === "complete" ? <Badge tone="lime">Text-to-speech</Badge> : <Badge tone="amber">TTS · incomplete</Badge>;
  }
  if (model.kind === "voice_conversion") return <Badge tone="sky">Voice conversion</Badge>;
  return <Badge>Unidentified</Badge>;
}

export function LicenseBadge({ model }: { model: Model }) {
  const tone = { permissive: "lime", conditional: "amber", noncommercial: "red", unknown: "amber" }[model.license_status] as "lime" | "amber" | "red";
  return <Badge tone={tone}>{model.license_status === "unknown" ? "License unknown" : model.license}</Badge>;
}

export function Languages({ model }: { model: Model }) {
  if (model.languages.length === 0) {
    return <span className="text-muted">{model.kind === "voice_conversion" ? "Follows source audio" : "Not declared"}</span>;
  }
  return <span>{model.languages.map(languageName).join(", ")}</span>;
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[130px_1fr] gap-3 border-b border-line py-2.5 text-[13px] last:border-0">
      <dt className="text-muted">{label}</dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

const List = ({ items }: { items: string[] }) =>
  items.length ? (
    <ul className="space-y-1">
      {items.map((i) => (
        <li key={i}>{i}</li>
      ))}
    </ul>
  ) : (
    <span className="text-muted">None</span>
  );

/** Slide-over with the full inspection report for one model. */
export function ModelDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const { data: m, error } = useLoad(() => api<Model>(`/models/${id}`), [id]);
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-ink/30 backdrop-blur-[2px]" onClick={onClose}>
      <aside
        role="dialog"
        aria-label="Model inspection report"
        className="h-full w-full max-w-xl overflow-y-auto bg-white p-6 shadow-2xl sm:rounded-l-[28px]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Inspection report</p>
            <h2 className="truncate text-xl font-semibold">{m?.name ?? "…"}</h2>
          </div>
          <button aria-label="Close" onClick={onClose} className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-soft hover:bg-line">
            <X size={16} />
          </button>
        </div>
        <LoadState error={error} loading={!m} />
        {m && (
          <div className="space-y-5">
            <div className="flex flex-wrap gap-2">
              <KindBadge model={m} />
              <Badge tone="dark">{m.architecture_label}</Badge>
              <LicenseBadge model={m} />
              {m.safety === "caution" && <Badge tone="amber">Restricted loading only</Badge>}
            </div>

            <Notice tone={m.inference.ready ? "lime" : "amber"}>
              <strong className="font-semibold">{m.inference.ready ? "Ready for inference." : "Cannot generate speech yet."}</strong>{" "}
              {m.inference.reason}
            </Notice>
            {m.warnings.map((w) => (
              <Notice key={w} tone={w.includes("not usable") || w.includes("not text-to-speech") ? "red" : "amber"}>
                <AlertTriangle size={14} className="mr-1.5 inline -translate-y-px" />
                {w}
              </Notice>
            ))}

            <dl>
              <Row label="Standalone TTS">{m.standalone_tts ? "Yes" : "No"}</Row>
              <Row label="Languages">
                <Languages model={m} />
                {m.language_note && <p className="mt-1 text-xs text-muted">{m.language_note}</p>}
              </Row>
              <Row label="Size">{formatBytes(m.size_bytes)}</Row>
              <Row label="Version">v{m.version}</Row>
              <Row label="CPU / GPU">
                CPU: {String(m.device_support.cpu ?? "unknown")} · GPU: {String(m.device_support.gpu ?? "unknown")}
              </Row>
              <Row label="Dependencies"><List items={m.dependencies} /></Row>
              <Row label="Needs at inference"><List items={m.required_config} /></Row>
              {m.missing.length > 0 && <Row label="Missing"><List items={m.missing} /></Row>}
              <Row label="Evidence"><List items={m.report?.detection.evidence ?? []} /></Row>
              <Row label="Checkpoint scan">
                {m.report?.checkpoints.length ? (
                  <ul className="space-y-1.5">
                    {m.report.checkpoints.map((c) => (
                      <li key={c.path}>
                        <CheckCircle2 size={13} className="mr-1 inline -translate-y-px text-lime-deep" />
                        <span className="font-mono text-xs">{c.path}</span> · {c.format} · {c.verdict} · {c.storage_count} tensors
                      </li>
                    ))}
                  </ul>
                ) : (
                  <span className="text-muted">No pickle checkpoints in this package</span>
                )}
              </Row>
              <Row label="Files">
                <ul className="space-y-1 font-mono text-xs">
                  {m.files?.map((f) => (
                    <li key={f.path}>
                      {f.path} <span className="text-muted">({formatBytes(f.size)})</span>
                    </li>
                  ))}
                </ul>
              </Row>
              {!!m.report?.skipped_files.length && (
                <Row label="Not extracted"><List items={m.report.skipped_files.map((s) => s.path)} /></Row>
              )}
              <Row label="SHA-256"><span className="font-mono text-xs">{m.sha256}</span></Row>
            </dl>
          </div>
        )}
      </aside>
    </div>
  );
}
