export type Inference = { ready: boolean; reason: string };

export type Model = {
  id: string;
  name: string;
  version: number;
  architecture: string;
  architecture_label: string;
  kind: "tts" | "voice_conversion" | "unknown";
  standalone_tts: boolean;
  completeness: "complete" | "incomplete" | "unknown";
  safety: "safe" | "caution";
  active: boolean;
  size_bytes: number;
  languages: string[];
  language_note: string;
  dependencies: string[];
  required_config: string[];
  missing: string[];
  device_support: Record<string, string | boolean>;
  license: string;
  license_status: "permissive" | "conditional" | "noncommercial" | "unknown";
  source_url: string;
  tags: string[];
  inference: Inference;
  warnings: string[];
  created_at: string;
  files?: { path: string; size: number }[];
  sha256?: string;
  report?: {
    detection: { evidence: string[]; confidence: string; details: Record<string, unknown> };
    checkpoints: { path: string; format: string; verdict: string; globals: string[]; storage_count: number; string_count: number }[];
    skipped_files: { path: string; reason: string }[];
  };
};

export type Engine = {
  architecture: string;
  label: string;
  kind: string;
  planned: string;
  adapter_installed: boolean;
  true_streaming: boolean;
  ready: boolean;
  reason: string;
};

export type Hardware = {
  platform: string;
  machine: string;
  python: string;
  cpu_cores: number;
  cpu_threads: number;
  ram_total_mb: number;
  ram_available_mb: number;
  gpus: { name: string; vram_total_mb: number; vram_used_mb: number }[];
  cuda_available: boolean;
  ffmpeg: boolean;
  recommended_device: string;
  warnings: string[];
};

export type Character = {
  id: string;
  name: string;
  description: string;
  image_url: string;
  voice_ml: string | null;
  voice_en: string | null;
  default_speed: number;
  rvc: RvcOptions | null;
};

export type Parameter = { name: string; label: string; type: string; default: number | string; min: number | null; max: number | null };
export type RvcOptions = { model_id: string; pitch: number; index_rate: number; protect?: number };
export type RuntimeBundle = {
  id: string; label: string; size_mb: number; available: boolean; installed: boolean; note: string; reason: string;
  state: "idle" | "installing" | "error"; error: string;
};
export type Voice = {
  id: string; name: string; description: string; model_id: string; languages: string[];
  license: string; license_status: Model["license_status"]; ready: boolean; reason: string; parameters: Parameter[];
};
export type Job = {
  job_id: string; status: "queued" | "processing" | "completed" | "failed"; cached: boolean; language: string;
  kind: "tts" | "conversion"; rvc: RvcOptions | null;
  voice_id: string | null; character_id: string | null; text: string; params: Record<string, number | string>;
  warnings: string[]; created_at: string; audio_url?: string; audio_seconds?: number; generation_seconds?: number;
  chunks?: number; error?: string;
};
export type CatalogVoice = {
  id: string; name: string; language: string; gender: string; size_mb: number; license: string;
  license_status: Model["license_status"]; license_note: string; source_url: string; installed: boolean;
};

/** Poll a queued job until it has completed or failed. */
export async function follow(job: Job, onUpdate?: (job: Job) => void): Promise<Job> {
  onUpdate?.(job);
  while (job.status === "queued" || job.status === "processing") {
    await new Promise((r) => setTimeout(r, 500));
    job = await api<Job>(`/tts/jobs/${job.job_id}`);
    onUpdate?.(job);
  }
  return job;
}

/** Re-voice an audio file with an RVC model. */
export async function convertFile(file: File, rvc: RvcOptions, onUpdate?: (job: Job) => void): Promise<Job> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_id", rvc.model_id);
  form.append("pitch", String(rvc.pitch));
  form.append("index_rate", String(rvc.index_rate));
  return follow(await api<Job>("/vc/convert", { method: "POST", body: form }), onUpdate);
}

/** Start a generation and resolve when it has completed or failed. */
export async function generate(body: object, onUpdate?: (job: Job) => void): Promise<Job> {
  let job = await api<Job>("/tts/generate", { method: "POST", body: JSON.stringify(body) });
  onUpdate?.(job);
  while (job.status === "queued" || job.status === "processing") {
    await new Promise((r) => setTimeout(r, 400));
    job = await api<Job>(`/tts/jobs/${job.job_id}`);
    onUpdate?.(job);
  }
  return job;
}
export type Health = { status: string; version: string; models: number; tts_ready: boolean };
export type Config = { auth_required: boolean; max_upload_mb: number; max_extracted_mb: number; max_zip_entries: number; database: string };

const KEY = "voix.apiKey";
export const getApiKey = () => {
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
};
export const setApiKey = (value: string) => {
  try {
    value ? localStorage.setItem(KEY, value) : localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: key lives for this page only */
  }
};

export class ApiError extends Error {
  constructor(public status: number, message: string, public report?: unknown) {
    super(message);
  }
}

async function parse(res: Response) {
  const text = await res.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : res.statusText;
    throw new ApiError(res.status, message, body?.report);
  }
  return body;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  return parse(await fetch(`/api/v1${path}`, { ...init, headers }));
}

/** Upload with progress. fetch() cannot report upload progress, so this uses XHR. */
export function importModel(file: File, onProgress: (fraction: number) => void): Promise<Model> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append("file", file);
    xhr.open("POST", "/api/v1/models/import");
    const key = getApiKey();
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError(0, "Could not reach the VOIX API"));
    xhr.onload = () => {
      let body: any = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* non-JSON error body */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new ApiError(xhr.status, body?.detail ?? xhr.statusText, body?.report));
    };
    xhr.send(form);
  });
}

export const formatBytes = (n: number) => {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
};

export const LANGUAGE_NAMES: Record<string, string> = {
  ml: "Malayalam", en: "English", hi: "Hindi", ta: "Tamil", zh: "Chinese", ja: "Japanese", ko: "Korean", yue: "Cantonese",
};
export const languageName = (code: string) => LANGUAGE_NAMES[code] ?? code.toUpperCase();
