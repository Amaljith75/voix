import { Send } from "lucide-react";
import { useState } from "react";
import { getApiKey } from "../api";
import { Badge, Button, Card, CardTitle, PageHeader, cx } from "../components/ui";

const ENDPOINTS = [
  { method: "GET", path: "/api/v1/health" },
  { method: "GET", path: "/api/v1/models" },
  { method: "GET", path: "/api/v1/voices" },
  { method: "GET", path: "/api/v1/characters" },
  { method: "GET", path: "/api/v1/engines" },
  { method: "GET", path: "/api/v1/system/hardware" },
  { method: "GET", path: "/api/v1/system/audit" },
  {
    method: "POST",
    path: "/api/v1/tts/generate",
    body: { text: "നമസ്കാരം കുട്ടികളേ! ഇന്ന് നമുക്ക് ശാസ്ത്രം പഠിക്കാം.", voice_id: "scol_teacher_01", language: "ml", output_format: "mp3", speed: 1.0 },
  },
] as const;

export default function ApiPlayground() {
  const [index, setIndex] = useState(0);
  const [body, setBody] = useState("");
  const [result, setResult] = useState<{ status: number; ms: number; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const endpoint = ENDPOINTS[index];

  function pick(i: number) {
    setIndex(i);
    setResult(null);
    const e = ENDPOINTS[i];
    setBody("body" in e ? JSON.stringify(e.body, null, 2) : "");
  }

  async function send() {
    setBusy(true);
    const started = performance.now();
    try {
      const headers: Record<string, string> = {};
      if (getApiKey()) headers["X-API-Key"] = getApiKey();
      if (endpoint.method === "POST") headers["Content-Type"] = "application/json";
      const res = await fetch(endpoint.path, { method: endpoint.method, headers, body: endpoint.method === "POST" ? body : undefined });
      const raw = await res.text();
      let text = raw;
      try {
        text = JSON.stringify(JSON.parse(raw), null, 2);
      } catch {
        /* not JSON: show as-is */
      }
      setResult({ status: res.status, ms: Math.round(performance.now() - started), text });
    } catch {
      setResult({ status: 0, ms: Math.round(performance.now() - started), text: "Could not reach the VOIX API." });
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="API Playground"
        subtitle="The same REST API your app calls. It takes text from any source; it never talks to an LLM."
        action={<a href="/api/docs" target="_blank" rel="noreferrer"><Button variant="ghost">OpenAPI docs</Button></a>}
      />
      <div className="grid gap-4 xl:grid-cols-[320px_1fr]">
        <Card>
          <CardTitle>Endpoints</CardTitle>
          <ul className="space-y-1">
            {ENDPOINTS.map((e, i) => (
              <li key={e.path}>
                <button onClick={() => pick(i)} className={cx("flex w-full items-center gap-2 rounded-2xl px-3 py-2.5 text-left text-xs transition", i === index ? "bg-ink text-white" : "hover:bg-soft")}>
                  <span className={cx("w-10 font-bold", i === index ? "text-lime" : "text-muted")}>{e.method}</span>
                  <span className="truncate font-mono">{e.path.replace("/api/v1", "")}</span>
                </button>
              </li>
            ))}
          </ul>
        </Card>
        <div className="space-y-4">
          <Card>
            <div className="flex flex-wrap items-center gap-3">
              <Badge tone="dark">{endpoint.method}</Badge>
              <code className="min-w-0 flex-1 truncate font-mono text-[13px]">{endpoint.path}</code>
              <Button onClick={send} disabled={busy}><Send size={14} /> {busy ? "Sending…" : "Send"}</Button>
            </div>
            {endpoint.method === "POST" && (
              <textarea value={body} onChange={(e) => setBody(e.target.value)} aria-label="Request body" spellCheck={false} className="mt-4 h-44 w-full rounded-2xl bg-soft p-4 font-mono text-xs outline-none" />
            )}
          </Card>
          <section className="rounded-[28px] bg-ink p-6 text-white shadow-card">
            <div className="mb-3 flex items-center gap-3 text-xs font-semibold">
              <span className="uppercase tracking-wider text-white/50">Response</span>
              {result && (
                <>
                  <span className={cx("rounded-full px-2.5 py-1", result.status >= 200 && result.status < 300 ? "bg-lime text-ink" : "bg-amber-300 text-ink")}>{result.status || "ERR"}</span>
                  <span className="text-white/50">{result.ms} ms</span>
                </>
              )}
            </div>
            <pre className="max-h-[480px] overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-white/85">
              {result ? result.text : "Send a request to see the live response."}
            </pre>
          </section>
        </div>
      </div>
    </>
  );
}
