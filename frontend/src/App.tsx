import {
  AudioLines, CloudDownload, History as HistoryIcon, LayoutGrid, Library, Mic2, PackagePlus, Settings as SettingsIcon, TerminalSquare, Users, Wand2,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { Health } from "./api";
import { api } from "./api";
import { cx, useLoad } from "./components/ui";
import ApiPlayground from "./pages/ApiPlayground";
import Characters from "./pages/Characters";
import Dashboard from "./pages/Dashboard";
import HuggingFace from "./pages/HuggingFace";
import VoiceChanger from "./pages/VoiceChanger";
import ModelLibrary from "./pages/ModelLibrary";
import ModelManager from "./pages/ModelManager";
import Settings from "./pages/Settings";
import { HistoryPage as History, VoiceLibraryPage as VoiceLibrary } from "./pages/Misc";
import VoiceStudio from "./pages/VoiceStudio";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutGrid, page: Dashboard },
  { id: "studio", label: "Voice Studio", icon: AudioLines, page: VoiceStudio },
  { id: "changer", label: "Voice Changer", icon: Wand2, page: VoiceChanger },
  { id: "voices", label: "Voice Library", icon: Mic2, page: VoiceLibrary },
  { id: "huggingface", label: "Hugging Face", icon: CloudDownload, page: HuggingFace },
  { id: "models", label: "Model Library", icon: Library, page: ModelLibrary },
  { id: "manager", label: "Model Manager", icon: PackagePlus, page: ModelManager },
  { id: "history", label: "Generation History", icon: HistoryIcon, page: History },
  { id: "characters", label: "Characters", icon: Users, page: Characters },
  { id: "playground", label: "API Playground", icon: TerminalSquare, page: ApiPlayground },
  { id: "settings", label: "Settings", icon: SettingsIcon, page: Settings },
] as const;

const currentRoute = () => {
  const id = window.location.hash.replace(/^#\/?/, "");
  return NAV.some((n) => n.id === id) ? id : "dashboard";
};

function Brand() {
  return (
    <a href="#/dashboard" className="flex items-center gap-3">
      <span className="grid h-11 w-11 place-items-center rounded-2xl bg-ink text-lime">
        <AudioLines size={22} strokeWidth={2.4} />
      </span>
      <span>
        <span className="block text-[22px] font-extrabold leading-none tracking-[0.18em]">VOIX</span>
        <span className="mt-1 block text-[11px] font-medium text-muted">Text in. Voice out.</span>
      </span>
    </a>
  );
}

function EngineStatus() {
  const { data, error } = useLoad(() => api<Health>("/health"));
  const dot = error ? "bg-red-500" : data?.tts_ready ? "bg-lime" : "bg-amber-400";
  const text = error ? "API offline" : !data ? "Connecting…" : data.tts_ready ? "TTS engine ready" : "No TTS engine yet";
  return (
    <div className="flex items-center gap-2.5 rounded-2xl bg-soft px-3.5 py-3 text-xs font-semibold">
      <span className={cx("h-2 w-2 rounded-full", dot)} />
      <span>{text}</span>
      {data && <span className="ml-auto font-medium text-muted">v{data.version}</span>}
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState(currentRoute);
  useEffect(() => {
    const onHash = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const Page = NAV.find((n) => n.id === route)!.page;

  return (
    <div className="mx-auto flex min-h-full max-w-[1500px] flex-col gap-4 p-3 lg:flex-row lg:gap-6 lg:p-6">
      <aside className="flex shrink-0 flex-col gap-4 rounded-[28px] bg-white p-4 shadow-card lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)] lg:w-[250px] lg:p-5">
        <div className="px-1 pt-1">
          <Brand />
        </div>
        <nav aria-label="Main" className="-mx-1 flex gap-1 overflow-x-auto px-1 lg:mt-3 lg:flex-col lg:overflow-visible">
          {NAV.map(({ id, label, icon: Icon }) => (
            <a
              key={id}
              href={`#/${id}`}
              aria-current={route === id ? "page" : undefined}
              className={cx(
                "flex shrink-0 items-center gap-3 rounded-full px-4 py-2.5 text-[13px] font-semibold transition",
                route === id ? "bg-ink text-white" : "text-ink/60 hover:bg-soft hover:text-ink",
              )}
            >
              <Icon size={17} className={route === id ? "text-lime" : ""} />
              {label}
            </a>
          ))}
        </nav>
        <div className="mt-auto hidden lg:block">
          <EngineStatus />
        </div>
      </aside>
      <main className="min-w-0 flex-1 pb-10 lg:py-2">
        <Page />
      </main>
    </div>
  );
}
