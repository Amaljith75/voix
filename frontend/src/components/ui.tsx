import type { ButtonHTMLAttributes, ReactNode } from "react";
import { useEffect, useState } from "react";

export const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(" ");

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <section className={cx("rounded-[28px] bg-white p-6 shadow-card", className)}>{children}</section>;
}

export function CardTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3">
      <h2 className="text-[15px] font-semibold">{children}</h2>
      {action}
    </div>
  );
}

type Tone = "neutral" | "lime" | "amber" | "red" | "sky" | "dark";
const tones: Record<Tone, string> = {
  neutral: "bg-soft text-ink/70",
  lime: "bg-lime-soft text-lime-deep",
  amber: "bg-amber-100 text-amber-800",
  red: "bg-red-100 text-red-700",
  sky: "bg-sky-100 text-sky-800",
  dark: "bg-ink text-white",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-1 text-[11px] font-semibold", tones[tone])}>
      {children}
    </span>
  );
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "lime" | "ghost" | "danger" };
export function Button({ variant = "primary", className, ...props }: ButtonProps) {
  const styles = {
    primary: "bg-ink text-white hover:bg-ink/85",
    lime: "bg-lime text-ink hover:brightness-95",
    ghost: "bg-soft text-ink hover:bg-line",
    danger: "bg-red-50 text-red-700 hover:bg-red-100",
  }[variant];
  return (
    <button
      {...props}
      className={cx(
        "inline-flex h-10 items-center justify-center gap-2 rounded-full px-4 text-[13px] font-semibold transition disabled:opacity-40",
        styles,
        className,
      )}
    />
  );
}

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-[30px] font-semibold leading-tight tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-muted">{subtitle}</p>}
      </div>
      {action}
    </header>
  );
}

export function Empty({ icon, title, children, action }: { icon: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-3xl border border-dashed border-line px-6 py-12 text-center">
      <div className="mb-4 grid h-12 w-12 place-items-center rounded-full bg-soft text-ink/60">{icon}</div>
      <p className="font-semibold">{title}</p>
      {children && <p className="mt-1 max-w-md text-sm text-muted">{children}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function Notice({ tone = "amber", children }: { tone?: "amber" | "red" | "lime"; children: ReactNode }) {
  const styles = { amber: "bg-amber-50 text-amber-900", red: "bg-red-50 text-red-800", lime: "bg-lime-soft text-lime-deep" }[tone];
  return <div className={cx("rounded-2xl px-4 py-3 text-[13px] leading-relaxed", styles)}>{children}</div>;
}

const BARS = [0.3, 0.55, 0.9, 0.6, 1, 0.7, 0.45, 0.8, 0.5, 0.95, 0.65, 0.4, 0.75, 0.55, 0.3];
/** Decorative level meter. Animates only while real audio work is happening. */
export function Waveform({ active = false, className }: { active?: boolean; className?: string }) {
  return (
    <div className={cx("flex h-20 items-center justify-center gap-1.5", className)} aria-hidden>
      {BARS.map((h, i) => (
        <span
          key={i}
          className={cx("w-2 origin-center rounded-full bg-white", active ? "animate-bar" : "opacity-30")}
          style={{ height: `${(active ? h : h * 0.35) * 100}%`, animationDelay: `${i * 70}ms` }}
        />
      ))}
    </div>
  );
}

/** Load data once, with explicit loading and error states. */
export function useLoad<T>(loader: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    setError(null);
    loader()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e instanceof TypeError ? "Cannot reach the VOIX API. Is the backend running on port 8000?" : e.message));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);
  return { data, error, reload: () => setTick((t) => t + 1) };
}

export function LoadState({ error, loading }: { error: string | null; loading: boolean }) {
  if (error) return <Notice tone="red">{error}</Notice>;
  if (loading) return <p className="text-sm text-muted">Loading…</p>;
  return null;
}
