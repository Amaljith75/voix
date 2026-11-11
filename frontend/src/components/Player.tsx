import { Download, Pause, Play } from "lucide-react";
import type { MouseEvent } from "react";
import { useEffect, useRef, useState } from "react";
import type { Job } from "../api";
import { cx } from "./ui";

const BARS = 96;

/** Peak per bar, read from the real audio samples. */
async function loadPeaks(url: string): Promise<number[]> {
  const data = await (await fetch(url)).arrayBuffer();
  const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new Ctx();
  try {
    const samples = (await ctx.decodeAudioData(data)).getChannelData(0);
    const step = Math.max(1, Math.floor(samples.length / BARS));
    const peaks = Array.from({ length: BARS }, (_, i) => {
      let top = 0;
      for (let j = i * step; j < Math.min((i + 1) * step, samples.length); j += 8) top = Math.max(top, Math.abs(samples[j]));
      return top;
    });
    const max = Math.max(...peaks, 0.001);
    return peaks.map((p) => p / max);
  } finally {
    ctx.close();
  }
}

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function Player({ job, dark = true }: { job: Job; dark?: boolean }) {
  const audio = useRef<HTMLAudioElement>(null);
  const [peaks, setPeaks] = useState<number[] | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const url = job.audio_url!;
  const total = job.audio_seconds ?? 0;

  useEffect(() => {
    let live = true;
    setPeaks(null);
    setTime(0);
    setPlaying(false);
    loadPeaks(`${url}?format=wav`).then((p) => live && setPeaks(p)).catch(() => live && setPeaks([]));
    return () => {
      live = false;
    };
  }, [url]);

  const toggle = () => {
    const el = audio.current;
    if (!el) return;
    el.paused ? el.play() : el.pause();
  };
  const seek = (e: MouseEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    if (audio.current && total) audio.current.currentTime = ((e.clientX - box.left) / box.width) * total;
  };
  const played = total ? time / total : 0;
  const idle = dark ? "bg-white/25" : "bg-ink/15";
  const done = dark ? "bg-lime" : "bg-ink";

  return (
    <div>
      <audio
        ref={audio}
        src={`${url}?format=wav`}
        preload="auto"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
      />
      <div className="flex items-center gap-4">
        <button
          onClick={toggle}
          aria-label={playing ? "Pause" : "Play"}
          className={cx("grid h-12 w-12 shrink-0 place-items-center rounded-full transition", dark ? "bg-lime text-ink hover:brightness-95" : "bg-ink text-lime")}
        >
          {playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" className="translate-x-px" />}
        </button>
        <div onClick={seek} className="flex h-16 min-w-0 flex-1 cursor-pointer items-center gap-[2px]" role="slider" aria-label="Seek" aria-valuemin={0} aria-valuemax={Math.round(total)} aria-valuenow={Math.round(time)}>
          {(peaks?.length ? peaks : Array.from({ length: BARS }, () => 0.08)).map((p, i) => (
            <span key={i} className={cx("flex-1 rounded-full", i / BARS < played ? done : idle)} style={{ height: `${Math.max(6, p * 100)}%` }} />
          ))}
        </div>
        <span className={cx("w-20 shrink-0 text-right text-xs tabular-nums", dark ? "text-white/60" : "text-muted")}>
          {clock(time)} / {clock(total)}
        </span>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {(["wav", "mp3"] as const).map((fmt) => (
          <a
            key={fmt}
            href={`${url}?format=${fmt}&download=true`}
            className={cx("inline-flex h-9 items-center gap-2 rounded-full px-4 text-xs font-semibold transition", dark ? "bg-white/10 text-white hover:bg-white/20" : "bg-soft text-ink hover:bg-line")}
          >
            <Download size={14} /> {fmt.toUpperCase()}
          </a>
        ))}
      </div>
    </div>
  );
}
