import { History as HistoryIcon, Mic2 } from "lucide-react";
import type { Voice } from "../api";
import { api, languageName } from "../api";
import { Badge, Card, Empty, LoadState, PageHeader, useLoad } from "../components/ui";

export function VoiceLibraryPage() {
  const { data, error } = useLoad(() => api<Voice[]>("/voices"));
  return (
    <>
      <PageHeader title="Voice Library" subtitle="Voices you can generate with. Each one maps to a TTS-capable model." />
      <LoadState error={error} loading={!data && !error} />
      {data?.length === 0 && (
        <Card>
          <Empty icon={<Mic2 size={20} />} title="No voices yet" action={<a href="#/models" className="text-[13px] font-semibold underline underline-offset-4">See installed models</a>}>
            Voices are registered when a TTS engine adapter can load a model. That starts in Phase 2. Voice-conversion models such as RVC never appear here on their own.
          </Empty>
        </Card>
      )}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {data?.map((v) => (
          <Card key={v.id}>
            <span className="grid h-12 w-12 place-items-center rounded-full bg-lime text-lg font-bold">{v.name[0]}</span>
            <p className="mt-4 font-semibold">{v.name}</p>
            <p className="text-[13px] text-muted">{v.description}</p>
            <div className="mt-3 flex flex-wrap gap-1.5">{v.languages.map((l) => <Badge key={l}>{languageName(l)}</Badge>)}</div>
          </Card>
        ))}
      </div>
    </>
  );
}

export function HistoryPage() {
  return (
    <>
      <PageHeader title="Generation History" subtitle="Replay, download, regenerate and search past audio." />
      <Card>
        <Empty icon={<HistoryIcon size={20} />} title="Nothing generated yet">
          History is recorded from the first real generation. No audio has been produced, so there is nothing to list.
        </Empty>
      </Card>
    </>
  );
}
