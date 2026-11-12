import { FileAudio, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import type { Job, RvcOptions } from "../api";
import { api, convertFile } from "../api";
import { Player } from "../components/Player";
import { VoiceChangerControls } from "../components/VoiceChanger";
import { Badge, Button, Card, CardTitle, Notice, PageHeader, Waveform, cx, useLoad } from "../components/ui";

export default function VoiceChanger() {
  const [file, setFile] = useState<File | null>(null);
  const [rvc, setRvc] = useState<RvcOptions | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const recent = useLoad(() => api<Job[]>("/history?kind=conversion&limit=10"));

  function choose(f: File | undefined) {
    if (!f) return;
    if (!/\.(wav|mp3|flac|ogg)$/i.test(f.name)) return setError("Use a WAV, MP3, FLAC or OGG file.");
    setError(null);
    setFile(f);
  }

  async function run() {
    if (!file || !rvc) return;
    setBusy(true);
    setError(null);
    try {
      const result = await convertFile(file, rvc, setJob);
      if (result.status === "failed") setError(result.error ?? "Conversion failed.");
      recent.reload();
    } catch (e) {
      setJob(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader title="Voice Changer" subtitle="Give any recording a character's voice. The words and timing stay; the voice changes." />
      <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
        <div className="space-y-4">
          <Card>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); choose(e.dataTransfer.files[0]); }}
              className={cx("flex flex-col items-center rounded-3xl border-2 border-dashed px-6 py-10 text-center transition", dragging ? "border-ink bg-lime-soft" : "border-line bg-soft")}
            >
              <span className="grid h-14 w-14 place-items-center rounded-full bg-ink text-lime">{file ? <FileAudio size={24} /> : <UploadCloud size={24} />}</span>
              <p className="mt-4 text-lg font-semibold">{file ? file.name : "Drop an audio file here"}</p>
              <p className="mt-1 text-sm text-muted">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP3, WAV, FLAC or OGG · speech works best · up to 10 minutes"}</p>
              <input ref={input} type="file" accept=".mp3,.wav,.flac,.ogg,audio/*" className="hidden" onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ""; }} />
              <Button variant="ghost" className="mt-5 !bg-white" onClick={() => input.current?.click()}>{file ? "Choose another file" : "Choose file"}</Button>
            </div>
          </Card>

          <section className="rounded-[28px] bg-ink p-6 text-white shadow-card">
            <p className="mb-5 text-xs font-semibold uppercase tracking-wider text-white/50">Converted audio</p>
            {job?.status === "completed" ? <Player job={job} /> : <Waveform active={busy} className="my-2" />}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <Button variant="lime" disabled={!file || !rvc || busy} onClick={run} className="!h-12 !px-6 !text-sm">
                {busy ? (job?.status === "queued" ? "Queued…" : "Converting…") : "Convert voice"}
              </Button>
              {job?.status === "completed" && <span className="text-xs text-white/50">{job.audio_seconds?.toFixed(1)} s converted in {job.generation_seconds?.toFixed(1)} s</span>}
            </div>
            {!file && <p className="mt-4 text-[13px] text-white/60">Choose an audio file.</p>}
            {file && !rvc && <p className="mt-4 text-[13px] text-white/60">Choose a character voice.</p>}
            {error && <p className="mt-4 rounded-2xl bg-red-500/15 px-4 py-3 text-[13px] text-red-200">{error}</p>}
            {job?.warnings.map((w) => <p key={w} className="mt-3 rounded-2xl bg-amber-400/15 px-4 py-3 text-[13px] text-amber-100">{w}</p>)}
          </section>
        </div>

        <div className="space-y-4">
          <Card>
            <CardTitle action={<Badge tone="sky">RVC</Badge>}>Character voice</CardTitle>
            <VoiceChangerControls value={rvc} onChange={setRvc} allowNone={false} />
          </Card>
          <Card>
            <CardTitle>Before you use a voice</CardTitle>
            <Notice>Only convert into voices you have the right to use. Community RVC models are often clones of real people or studio characters and carry no licence.</Notice>
          </Card>
          {!!recent.data?.length && (
            <Card>
              <CardTitle action={<a href="#/history" className="text-xs font-semibold text-muted hover:text-ink">History</a>}>Recent conversions</CardTitle>
              <ul className="space-y-2">
                {recent.data.map((j) => (
                  <li key={j.job_id}>
                    <button onClick={() => j.status === "completed" && setJob(j)} className="w-full rounded-2xl bg-soft px-3 py-2.5 text-left text-[13px] hover:bg-line">
                      <span className="block truncate font-semibold">{j.text.replace("Audio file: ", "")}</span>
                      <span className="text-xs text-muted">{j.status === "completed" ? `${j.audio_seconds?.toFixed(1)} s` : j.status}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
