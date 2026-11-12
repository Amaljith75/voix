import { Check, Download, KeyRound, PackageSearch, Trash2 } from "lucide-react";
import type { FormEvent } from "react";
import { useState } from "react";
import type { HubCandidate, HubInspect, HubTask, SupportModel } from "../api";
import { api, formatBytes, waitTask } from "../api";
import { Badge, Button, Card, CardTitle, Empty, LoadState, Notice, PageHeader, useLoad } from "../components/ui";

const field = "h-11 w-full rounded-2xl bg-soft px-4 text-sm outline-none placeholder:text-muted focus:ring-2 focus:ring-ink/10";
const KIND: Record<string, { label: string; tone: "lime" | "sky" | "amber" | "neutral" }> = {
  rvc: { label: "RVC character voice", tone: "lime" },
  qwen3_tts: { label: "Qwen3-TTS", tone: "sky" },
  piper: { label: "Piper voice", tone: "neutral" },
  zip: { label: "ZIP package", tone: "amber" },
};

function Progress({ task }: { task: HubTask }) {
  const pct = task.total_bytes ? Math.min(100, (task.done_bytes / task.total_bytes) * 100) : 0;
  return (
    <div className="mt-2">
      <div className="h-2 overflow-hidden rounded-full bg-soft">
        <div className="h-full rounded-full bg-ink transition-all" style={{ width: `${task.status === "completed" ? 100 : pct}%` }} />
      </div>
      <p className="mt-1 text-[11px] text-muted">
        {task.status === "running" ? `${formatBytes(task.done_bytes)} of ${formatBytes(task.total_bytes)}` : task.status === "completed" ? "Installed" : task.error}
      </p>
    </div>
  );
}

function SupportModels() {
  const list = useLoad(() => api<SupportModel[]>("/support"));
  const [tasks, setTasks] = useState<Record<string, HubTask>>({});
  const [error, setError] = useState<string | null>(null);

  async function install(s: SupportModel) {
    setError(null);
    try {
      const task = await api<HubTask>(`/support/${s.id}/install`, { method: "POST" });
      const done = await waitTask(task, (t) => setTasks((prev) => ({ ...prev, [s.id]: t })));
      if (done.status === "failed") setError(`${s.label}: ${done.error}`);
      list.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(s: SupportModel) {
    await api(`/support/${s.id}`, { method: "DELETE" });
    list.reload();
  }

  return (
    <Card>
      <CardTitle>Support models</CardTitle>
      <p className="-mt-2 mb-4 text-[13px] text-muted">
        Shared helpers. RVC voices need the first two and fetch them on first use if they are missing. Getting them here first avoids a long first generation.
      </p>
      {error && <div className="mb-3"><Notice tone="red">{error}</Notice></div>}
      <LoadState error={list.error} loading={!list.data && !list.error} />
      <ul className="divide-y divide-line">
        {list.data?.map((s) => (
          <li key={s.id} className="py-4">
            <div className="flex flex-wrap items-center gap-3">
              <div className="min-w-0 flex-1 basis-60">
                <p className="text-sm font-semibold">{s.label}</p>
                <p className="mt-0.5 text-xs text-muted">{s.purpose}</p>
                <p className="mt-0.5 font-mono text-[11px] text-muted">{s.repo} · {s.license}</p>
              </div>
              {s.installed ? (
                <>
                  <Badge tone="lime"><Check size={12} /> Installed</Badge>
                  <button aria-label={`Remove ${s.label}`} onClick={() => remove(s)} className="grid h-8 w-8 place-items-center rounded-full bg-soft text-red-700 hover:bg-red-50"><Trash2 size={14} /></button>
                </>
              ) : (
                <Button className="!h-9" disabled={tasks[s.id]?.status === "running"} onClick={() => install(s)}>
                  <Download size={14} /> {tasks[s.id]?.status === "running" ? "Downloading…" : `Get · ${s.size_mb} MB`}
                </Button>
              )}
            </div>
            {tasks[s.id] && !s.installed && <Progress task={tasks[s.id]} />}
          </li>
        ))}
      </ul>
    </Card>
  );
}

function TokenCard() {
  const status = useLoad(() => api<{ set: boolean; source: string | null }>("/hub/token"));
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function save(value: string | null) {
    setError(null);
    try {
      await api("/hub/token", { method: "PUT", body: JSON.stringify({ token: value }) });
      setToken("");
      status.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <Card>
      <CardTitle action={status.data && <Badge tone={status.data.set ? "lime" : "neutral"}>{status.data.set ? `Set (${status.data.source})` : "Not set"}</Badge>}>
        Hugging Face token
      </CardTitle>
      <p className="-mt-2 mb-3 text-[13px] text-muted">
        Needed for your private repositories and gated models. Create a <b>read</b> token at huggingface.co → Settings → Access Tokens. It is stored on this machine only, sent to huggingface.co only, and never shown again.
      </p>
      {status.data?.source === "env" ? (
        <Notice tone="lime">Using the token from the VOIX_HF_TOKEN / HF_TOKEN environment variable.</Notice>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); save(token); }} className="flex gap-2">
          <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} placeholder="hf_…" aria-label="Hugging Face token" className={field} />
          <Button type="submit" disabled={!token.trim()}><KeyRound size={14} /> Save</Button>
          {status.data?.set && <Button type="button" variant="danger" onClick={() => save(null)}>Remove</Button>}
        </form>
      )}
      {error && <div className="mt-3"><Notice tone="red">{error}</Notice></div>}
    </Card>
  );
}

export default function HuggingFace() {
  const [repo, setRepo] = useState("");
  const [result, setResult] = useState<HubInspect | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const [tasks, setTasks] = useState<Record<string, HubTask>>({});

  async function inspect(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api<HubInspect>(`/hub/inspect?repo=${encodeURIComponent(repo.trim())}`));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const key = (c: HubCandidate) => c.files.join("|");

  async function install(c: HubCandidate) {
    if (!result) return;
    try {
      const task = await api<HubTask>("/hub/install", {
        method: "POST",
        body: JSON.stringify({ repo_id: result.repo_id, revision: result.revision, files: c.files, name: names[key(c)] || undefined, license: result.license }),
      });
      await waitTask(task, (t) => setTasks((prev) => ({ ...prev, [key(c)]: t })));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  return (
    <>
      <PageHeader
        title="Hugging Face"
        subtitle="Install RVC character voices (.pth + .index), Qwen3-TTS models and Piper voices straight from a Hugging Face repository. Every download is inspected like a ZIP import; code is never downloaded or run."
      />
      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <div className="space-y-4">
          <Card>
            <CardTitle>Repository</CardTitle>
            <form onSubmit={inspect} className="flex flex-col gap-2 sm:flex-row">
              <input value={repo} onChange={(e) => setRepo(e.target.value)} placeholder="owner/name, or a huggingface.co link to a folder" aria-label="Repository" className={field} />
              <Button type="submit" disabled={!repo.trim() || busy} className="!h-11 shrink-0"><PackageSearch size={15} /> {busy ? "Reading…" : "Look inside"}</Button>
            </form>
            <p className="mt-3 text-xs text-muted">
              Examples: <button className="underline" onClick={() => setRepo("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")}>Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice</button> ·{" "}
              <button className="underline" onClick={() => setRepo("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")}>Qwen3-TTS 0.6B (lighter, no emotion)</button> · or your own RVC voice repository.
            </p>
            {error && <div className="mt-3"><Notice tone="red">{error}</Notice></div>}
          </Card>

          {result && (
            <Card>
              <CardTitle action={<div className="flex gap-1.5"><Badge>{result.license}</Badge>{result.gated && <Badge tone="amber">Gated</Badge>}</div>}>
                {result.repo_id}{result.subfolder && <span className="text-muted"> / {result.subfolder}</span>}
              </CardTitle>
              {result.candidates.length === 0 && (
                <Empty icon={<PackageSearch size={20} />} title="Nothing VOIX can install here">
                  No RVC .pth, Qwen3-TTS folder, Piper voice or ZIP was found among {result.files.length} files.
                </Empty>
              )}
              <ul className="divide-y divide-line">
                {result.candidates.map((c) => {
                  const k = key(c);
                  const task = tasks[k];
                  return (
                    <li key={k} className="py-4">
                      <div className="flex flex-wrap items-center gap-3">
                        <div className="min-w-0 flex-1 basis-64">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge tone={KIND[c.kind]?.tone ?? "neutral"}>{KIND[c.kind]?.label ?? c.kind}</Badge>
                            <p className="truncate text-sm font-semibold">{c.label}</p>
                          </div>
                          <p className="mt-1 truncate font-mono text-[11px] text-muted">{c.files.length > 3 ? `${c.files.length} files` : c.files.join(" + ")}</p>
                          {c.kind === "rvc" && !c.has_index && <p className="mt-1 text-[11px] text-amber-800">No .index next to it. It works, with a little less timbre detail.</p>}
                        </div>
                        <input value={names[k] ?? ""} onChange={(e) => setNames((p) => ({ ...p, [k]: e.target.value }))} placeholder="Name (optional)" aria-label="Name" className="h-9 w-40 rounded-full bg-soft px-3 text-xs outline-none" />
                        {task?.status === "completed" ? (
                          <Badge tone="lime"><Check size={12} /> Installed</Badge>
                        ) : (
                          <Button className="!h-9" disabled={task?.status === "running"} onClick={() => install(c)}>
                            <Download size={14} /> {task?.status === "running" ? "Installing…" : `Install · ${formatBytes(c.size)}`}
                          </Button>
                        )}
                      </div>
                      {task && <Progress task={task} />}
                    </li>
                  );
                })}
              </ul>
            </Card>
          )}
        </div>
        <div className="space-y-4">
          <TokenCard />
          <SupportModels />
        </div>
      </div>
    </>
  );
}
