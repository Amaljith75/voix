import type { ReactNode } from "react";
import { useState } from "react";
import type { Config, Engine, Hardware, RuntimeBundle } from "../api";
import { api, getApiKey, setApiKey } from "../api";
import { BundleInstall } from "../components/VoiceChanger";
import { Badge, Button, Card, CardTitle, LoadState, Notice, PageHeader, useLoad } from "../components/ui";

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-line py-2.5 text-[13px] last:border-0">
      <span className="text-muted">{label}</span>
      <span className="text-right font-semibold">{value}</span>
    </div>
  );
}

export default function Settings() {
  const hw = useLoad(() => api<Hardware>("/system/hardware"));
  const config = useLoad(() => api<Config>("/system/config"));
  const engines = useLoad(() => api<Engine[]>("/engines"));
  const bundles = useLoad(() => api<RuntimeBundle[]>("/runtime"));
  const [key, setKey] = useState(getApiKey);
  const [saved, setSaved] = useState(false);

  return (
    <>
      <PageHeader title="Settings" subtitle="Server limits, access and the hardware VOIX detected." />
      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardTitle>API access</CardTitle>
          <p className="mb-3 text-[13px] text-muted">
            {config.data?.auth_required
              ? "This server requires an API key for every change. The key is kept in this browser only."
              : "This server is running without VOIX_API_KEY, so writes are open. Set it before exposing the API beyond localhost."}
          </p>
          <div className="flex gap-2">
            <input type="password" value={key} onChange={(e) => { setKey(e.target.value); setSaved(false); }} placeholder="X-API-Key" aria-label="API key" autoComplete="off" className="h-10 min-w-0 flex-1 rounded-full bg-soft px-4 text-sm outline-none focus:ring-2 focus:ring-ink/10" />
            <Button onClick={() => { setApiKey(key.trim()); setSaved(true); }}>Save</Button>
          </div>
          {saved && <p className="mt-2 text-xs font-semibold text-lime-deep">Saved in this browser.</p>}
          <div className="mt-4">
            <LoadState error={config.error} loading={!config.data && !config.error} />
            {config.data && (
              <>
                <Row label="Max upload" value={`${config.data.max_upload_mb} MB`} />
                <Row label="Max extracted size" value={`${config.data.max_extracted_mb} MB`} />
                <Row label="Max files per ZIP" value={config.data.max_zip_entries} />
                <Row label="Database" value={config.data.database} />
              </>
            )}
          </div>
        </Card>

        <Card>
          <CardTitle>Hardware</CardTitle>
          <LoadState error={hw.error} loading={!hw.data && !hw.error} />
          {hw.data && (
            <>
              <Row label="Platform" value={`${hw.data.platform} · ${hw.data.machine}`} />
              <Row label="Python" value={hw.data.python} />
              <Row label="CPU" value={`${hw.data.cpu_cores} cores / ${hw.data.cpu_threads} threads`} />
              <Row label="RAM" value={`${(hw.data.ram_total_mb / 1024).toFixed(1)} GB`} />
              <Row label="GPU" value={hw.data.gpus.map((g) => `${g.name} (${Math.round(g.vram_total_mb / 1024)} GB)`).join(", ") || "None detected"} />
              <Row label="FFmpeg" value={hw.data.ffmpeg ? "Installed" : "Missing"} />
              <Row label="Inference device" value={hw.data.recommended_device.toUpperCase()} />
              <div className="mt-3 space-y-2">{hw.data.warnings.map((w) => <Notice key={w}>{w}</Notice>)}</div>
            </>
          )}
        </Card>

        <Card className="xl:col-span-2">
          <CardTitle>Optional downloads</CardTitle>
          <LoadState error={bundles.error} loading={!bundles.data && !bundles.error} />
          <ul className="divide-y divide-line">
            {bundles.data?.map((b) => (
              <li key={b.id} className="grid gap-3 py-4 md:grid-cols-[1fr_300px] md:items-center">
                <div>
                  <p className="text-sm font-semibold">{b.label} {b.installed && <Badge tone="lime">Installed</Badge>}</p>
                  <p className="mt-1 text-[13px] text-muted">{b.note}</p>
                </div>
                <BundleInstall id={b.id} compact onReady={() => { bundles.reload(); engines.reload(); }} />
              </li>
            ))}
          </ul>
        </Card>

        <Card className="xl:col-span-2">
          <CardTitle>Engine adapters</CardTitle>
          <LoadState error={engines.error} loading={!engines.data && !engines.error} />
          <ul className="divide-y divide-line">
            {engines.data?.map((e) => (
              <li key={e.architecture} className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3 text-[13px]">
                <span className="w-32 font-semibold">{e.label}</span>
                <Badge tone={e.kind === "tts" ? "lime" : "sky"}>{e.kind === "tts" ? "Text-to-speech" : "Voice conversion"}</Badge>
                <span className="flex-1 text-muted">{e.ready ? "Ready" : e.reason}</span>
                <Badge tone={e.adapter_installed ? "dark" : "neutral"}>{e.adapter_installed ? "Installed" : "Not installed"}</Badge>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  );
}
