import { FileArchive, ShieldCheck, Trash2, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import type { Config, Model } from "../api";
import { ApiError, api, formatBytes, importModel } from "../api";
import { KindBadge, LicenseBadge, ModelDrawer } from "../components/ModelBits";
import { Button, Card, CardTitle, LoadState, Notice, PageHeader, cx, useLoad } from "../components/ui";

type Upload = { state: "idle" } | { state: "uploading"; name: string; progress: number } | { state: "done"; model: Model } | { state: "error"; message: string };

const LICENSES = ["unknown", "permissive", "conditional", "noncommercial"] as const;

export default function ModelManager() {
  const models = useLoad(() => api<Model[]>("/models"));
  const config = useLoad(() => api<Config>("/system/config"));
  const [upload, setUpload] = useState<Upload>({ state: "idle" });
  const [dragging, setDragging] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function send(file: File | undefined) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) return setUpload({ state: "error", message: "Only .zip archives can be imported." });
    setUpload({ state: "uploading", name: file.name, progress: 0 });
    try {
      const model = await importModel(file, (progress) => setUpload({ state: "uploading", name: file.name, progress }));
      setUpload({ state: "done", model });
      models.reload();
    } catch (e) {
      setUpload({ state: "error", message: e instanceof ApiError ? e.message : "Upload failed." });
    }
  }

  async function patch(id: string, body: object) {
    setRowError(null);
    try {
      await api(`/models/${id}`, { method: "PATCH", body: JSON.stringify(body) });
      models.reload();
    } catch (e) {
      setRowError((e as Error).message);
    }
  }

  async function remove(m: Model) {
    if (!window.confirm(`Delete "${m.name}" v${m.version} and its files? This cannot be undone.`)) return;
    setRowError(null);
    try {
      await api(`/models/${m.id}`, { method: "DELETE" });
      models.reload();
    } catch (e) {
      setRowError((e as Error).message);
    }
  }

  return (
    <>
      <PageHeader title="Model Manager" subtitle="Import, verify, activate and remove model packages." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <Card>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); send(e.dataTransfer.files[0]); }}
            className={cx("flex flex-col items-center rounded-3xl border-2 border-dashed px-6 py-12 text-center transition", dragging ? "border-ink bg-lime-soft" : "border-line bg-soft")}
          >
            <span className="grid h-14 w-14 place-items-center rounded-full bg-ink text-lime"><UploadCloud size={24} /></span>
            <p className="mt-4 text-lg font-semibold">Drop a model ZIP here</p>
            <p className="mt-1 text-sm text-muted">
              .pth, .index, .onnx, JSON and YAML inside are inspected, never executed.
              {config.data && ` Up to ${config.data.max_upload_mb >= 1024 ? `${config.data.max_upload_mb / 1024} GB` : `${config.data.max_upload_mb} MB`}.`}
            </p>
            <input ref={input} type="file" accept=".zip" className="hidden" onChange={(e) => { send(e.target.files?.[0]); e.target.value = ""; }} />
            <Button className="mt-5" onClick={() => input.current?.click()} disabled={upload.state === "uploading"}>Choose file</Button>
          </div>

          {upload.state === "uploading" && (
            <div className="mt-4">
              <div className="flex justify-between text-[13px] font-medium">
                <span className="truncate">{upload.name}</span>
                <span>{upload.progress < 1 ? `${Math.round(upload.progress * 100)}%` : "Inspecting…"}</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-soft"><div className="h-2 rounded-full bg-ink transition-all" style={{ width: `${upload.progress * 100}%` }} /></div>
            </div>
          )}
          {upload.state === "error" && <div className="mt-4"><Notice tone="red"><strong className="font-semibold">Import rejected.</strong> {upload.message}</Notice></div>}
          {upload.state === "done" && (
            <div className="mt-4 rounded-2xl bg-lime-soft p-4">
              <div className="flex flex-wrap items-center gap-2">
                <FileArchive size={16} />
                <span className="font-semibold">{upload.model.name}</span>
                <KindBadge model={upload.model} />
                <span className="text-[13px] text-lime-deep">{upload.model.architecture_label}</span>
                <button className="ml-auto text-[13px] font-semibold underline underline-offset-4" onClick={() => setOpen(upload.model.id)}>Full report</button>
              </div>
              {upload.model.warnings.map((w) => <p key={w} className="mt-2 text-[13px] text-ink/70">{w}</p>)}
            </div>
          )}
        </Card>

        <Card>
          <CardTitle>How imports are checked</CardTitle>
          <ul className="space-y-3 text-[13px] leading-relaxed text-ink/75">
            {[
              "Archive paths are validated, so nothing can be written outside the model folder.",
              "Scripts and executables in the ZIP are skipped, not extracted.",
              "Checkpoints are scanned opcode by opcode. They are never unpickled, so embedded code cannot run.",
              "Architecture comes from tensor names and configs, not from the file extension.",
              "RVC is labelled voice conversion, never text-to-speech.",
            ].map((t) => (
              <li key={t} className="flex gap-2.5"><ShieldCheck size={16} className="mt-0.5 shrink-0 text-lime-deep" />{t}</li>
            ))}
          </ul>
        </Card>
      </div>

      <Card className="mt-4">
        <CardTitle>Installed models</CardTitle>
        <LoadState error={models.error} loading={!models.data && !models.error} />
        {rowError && <div className="mb-3"><Notice tone="red">{rowError}</Notice></div>}
        {models.data?.length === 0 && <p className="py-4 text-sm text-muted">No models installed.</p>}
        <ul className="divide-y divide-line">
          {models.data?.map((m) => (
            <li key={m.id} className="flex flex-wrap items-center gap-x-4 gap-y-3 py-4">
              <button className="min-w-0 flex-1 basis-56 text-left" onClick={() => setOpen(m.id)}>
                <span className="block truncate text-sm font-semibold hover:underline">{m.name} <span className="font-medium text-muted">v{m.version}</span></span>
                <span className="block text-xs text-muted">{m.architecture_label} · {formatBytes(m.size_bytes)}</span>
              </button>
              <KindBadge model={m} />
              <label className="flex items-center gap-2 text-xs text-muted">
                License
                <select
                  value={m.license_status}
                  onChange={(e) => patch(m.id, { license_status: e.target.value })}
                  className="h-8 rounded-full bg-soft px-3 text-xs font-semibold text-ink outline-none"
                >
                  {LICENSES.map((l) => <option key={l} value={l}>{l}</option>)}
                </select>
              </label>
              <span className="hidden sm:block"><LicenseBadge model={m} /></span>
              <button
                role="switch"
                aria-checked={m.active}
                aria-label={`${m.active ? "Deactivate" : "Activate"} ${m.name}`}
                onClick={() => patch(m.id, { active: !m.active })}
                className={cx("relative h-7 w-12 rounded-full transition", m.active ? "bg-ink" : "bg-line")}
              >
                <span className={cx("absolute top-1 h-5 w-5 rounded-full transition-all", m.active ? "left-6 bg-lime" : "left-1 bg-white")} />
              </button>
              <button aria-label={`Delete ${m.name}`} onClick={() => remove(m)} className="grid h-9 w-9 place-items-center rounded-full bg-red-50 text-red-700 hover:bg-red-100">
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>
      </Card>
      {open && <ModelDrawer id={open} onClose={() => setOpen(null)} />}
    </>
  );
}
