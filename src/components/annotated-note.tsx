import { segmentText, tintFor, type Entity } from "@/lib/extract";

export function AnnotatedNote({ text, entities }: { text: string; entities: Entity[] }) {
  const segs = segmentText(text, entities);
  return (
    <p className="note-paper whitespace-pre-wrap rounded-lg border border-border bg-bg-elevated p-5 font-sans text-[15px] leading-7 text-fg sm:p-6">
      {segs.map((seg) =>
        seg.entity ? (
          <mark
            key={seg.key}
            className={`${tintFor(seg.entity.feature_num)} mx-0.5 rounded-xs px-0.5 ring-1 ring-border`}
            title={`${seg.entity.type} · ${(seg.entity.confidence * 100).toFixed(0)}%`}
          >
            {seg.text}
          </mark>
        ) : (
          <span key={seg.key}>{seg.text}</span>
        ),
      )}
    </p>
  );
}
