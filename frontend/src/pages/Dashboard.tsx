import { ArrowUpRight, Cpu, MemoryStick, MonitorCog } from "lucide-react";
import type { Engine, Hardware, Model } from "../api";
import { api, formatBytes } from "../api";
import { KindBadge } from "../components/ModelBits";
import { Card, CardTitle, LoadState, Notice, PageHeader, Waveform, cx, useLoad } from "../components/ui";

function Stat({ label, value, hint }: { label: string; value: number | string; hint: string }) {
  return (
    <Card className="!p-5">
      <p className="text-[13px] font-medium text-muted">{label}</p>
      <p className="mt-2 text-[40px] font-semibold leading-none tracking-tight">{value}</p>
      <p className="mt-2 text-xs text-muted">{hint}</p>
    </Card>
  );
}

export default function Dashboard() {
  const models = useLoad(() => api<Model[]>("/models"));
  const hw = useLoad(() => api<Hardware>("/system/hardware"));
  const engines = useLoad(() => api<Engine[]>("/engines"));
  const list = models.data ?? [];
  const tts = list.filter((m) => m.kind === "tts" && m.completeness === "complete").length;
  const vc = list.filter((m) => m.kind === "voice_conversion").length;
  const flagged = list.filter((m) => m.license_status === "unknown" || m.license_status === "noncommercial").length;
  const ready = engines.data?.some((e) => e.ready && e.kind === "tts") ?? false;
  const ramUsed = hw.data ? 1 - hw.data.ram_available_mb / hw.data.ram_total_mb : 0;

  return (
    <>
      <PageHeader title="Dashboard" subtitle="What is installed, what it really is, and what this machine can run." />
      <LoadState error={models.error} loading={false} />

      <div className="mt-4 grid grid-cols-2 gap-4 xl:grid-cols-4">
        <Stat label="Models installed" value={list.length} hint="Imported and inspected" />
        <Stat label="Text-to-speech" value={tts} hint="Complete TTS packages" />
        <Stat label="Voice conversion" value={vc} hint="Need source speech (RVC)" />
        <Stat label="License flags" value={flagged} hint="Unknown or non-commercial" />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <section className="relative overflow-hidden rounded-[28px] bg-ink p-6 text-white shadow-card xl:col-span-2">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-white/50">Voice engine</p>
              <p className="mt-2 text-2xl font-semibold">{ready ? "Ready to speak" : "No TTS engine installed"}</p>
              <p className="mt-1 max-w-md text-sm text-white/60">
                {ready
                  ? "Piper is built in and runs on this CPU. Add a voice in Voice Library, then generate in Voice Studio."
                  : "The Piper runtime is not installed, so nothing can be generated."}
              </p>
            </div>
            <span className={cx("mt-1 h-2.5 w-2.5 shrink-0 rounded-full", ready ? "bg-lime" : "bg-amber-400")} />
          </div>
          <Waveform active={ready} className="mt-6" />
          <div className="mt-6 flex flex-wrap gap-2">
            {engines.data?.map((e) => (
              <span key={e.architecture} className="rounded-full bg-white/10 px-3 py-1.5 text-[11px] font-semibold text-white/80">
                {e.label} · {e.ready ? "ready" : e.planned}
              </span>
            ))}
          </div>
        </section>

        <a href="#/manager" className="group flex flex-col justify-between rounded-[28px] bg-lime p-6 shadow-card transition hover:brightness-95">
          <div className="flex items-start justify-between">
            <span className="rounded-full bg-ink px-3 py-1 text-[11px] font-semibold text-white">Next step</span>
            <span className="grid h-11 w-11 place-items-center rounded-full bg-ink text-white transition group-hover:rotate-45">
              <ArrowUpRight size={18} />
            </span>
          </div>
          <div className="mt-10">
            <p className="text-2xl font-semibold leading-tight">Import a voice model ZIP</p>
            <p className="mt-2 text-sm text-ink/70">VOIX opens it safely, tells you what architecture it is, and whether it can do text-to-speech at all.</p>
          </div>
        </a>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardTitle action={<a href="#/models" className="text-xs font-semibold text-muted hover:text-ink">View all</a>}>Recent models</CardTitle>
          {list.length === 0 ? (
            <p className="py-6 text-sm text-muted">Nothing imported yet.</p>
          ) : (
            <ul className="divide-y divide-line">
              {list.slice(0, 5).map((m) => (
                <li key={m.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3 text-[13px]">
                  <span className="min-w-0 flex-1 basis-full truncate font-semibold sm:basis-0">{m.name}</span>
                  <span className="text-muted">{m.architecture_label}</span>
                  <KindBadge model={m} />
                  <span className="w-16 text-right text-muted">{formatBytes(m.size_bytes)}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          <CardTitle>This machine</CardTitle>
          <LoadState error={hw.error} loading={!hw.data} />
          {hw.data && (
            <div className="space-y-4 text-[13px]">
              <div className="flex items-center gap-3">
                <Cpu size={16} className="text-muted" />
                <span className="flex-1">CPU</span>
                <span className="font-semibold">{hw.data.cpu_threads} threads</span>
              </div>
              <div>
                <div className="flex items-center gap-3">
                  <MemoryStick size={16} className="text-muted" />
                  <span className="flex-1">RAM</span>
                  <span className="font-semibold">
                    {((hw.data.ram_total_mb - hw.data.ram_available_mb) / 1024).toFixed(1)} / {(hw.data.ram_total_mb / 1024).toFixed(1)} GB
                  </span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-soft">
                  <div className="h-2 rounded-full bg-ink" style={{ width: `${Math.round(ramUsed * 100)}%` }} />
                </div>
              </div>
              <div className="flex items-center gap-3">
                <MonitorCog size={16} className="text-muted" />
                <span className="flex-1">GPU</span>
                <span className="font-semibold">{hw.data.gpus[0]?.name ?? "None (CPU only)"}</span>
              </div>
              {hw.data.warnings.slice(0, 1).map((w) => (
                <Notice key={w}>{w}</Notice>
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
