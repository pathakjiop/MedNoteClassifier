import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { loadMlBundle, type ErrorsFile, type MetricsFile } from "@/lib/ml-data";

export const Route = createFileRoute("/models")({ component: ModelsPage });

const FALLBACK = [
  { model: "Empty (majority)", precision: 0, recall: 0, f1: 0, exact_match: 0 },
  { model: "Lexicon", precision: 0, recall: 0, f1: 0, exact_match: 0 },
  { model: "Hybrid (lexicon + logreg)", precision: 0, recall: 0, f1: 0, exact_match: 0 },
  { model: "Tiny transformer", precision: 0, recall: 0, f1: 0, exact_match: 0 },
];

export function ModelsPage() {
  const [metrics, setMetrics] = useState<MetricsFile | null>(null);
  const [errors, setErrors] = useState<ErrorsFile | null>(null);

  useEffect(() => {
    loadMlBundle().then((b) => {
      setMetrics(b.metrics);
      setErrors(b.errors);
    });
  }, []);

  const testRows = (metrics?.comparison ?? []).filter((r) => r.split === "test");
  const rows =
    testRows.length > 0
      ? testRows.map((r) => ({
          model: labelOf(r.model),
          precision: r.precision,
          recall: r.recall,
          f1: r.f1,
          exact_match: r.exact_match,
        }))
      : FALLBACK;

  return (
    <div className="flex flex-col gap-10">
      <header className="max-w-2xl">
        <p className="text-xs uppercase tracking-[0.16em] text-subtle">Evaluation</p>
        <h1 className="mt-2 font-display text-4xl font-semibold tracking-tight">
          Character-level micro-F1 on a note-grouped test split.
        </h1>
        <p className="mt-3 text-muted leading-relaxed">
          Metric matches the Kaggle NBME scorer: predicted vs gold character
          sets per (note, feature), including empty annotations. Numbers below
          are measured, not hand-waved.
        </p>
      </header>

      <div className="overflow-x-auto rounded-xl border border-border bg-bg-elevated">
        <table className="w-full min-w-[32rem] text-left text-sm">
          <thead className="border-b border-border text-xs uppercase tracking-[0.12em] text-subtle">
            <tr>
              <th className="px-4 py-3 font-medium">Model</th>
              <th className="px-4 py-3 font-medium">Precision</th>
              <th className="px-4 py-3 font-medium">Recall</th>
              <th className="px-4 py-3 font-medium">F1</th>
              <th className="px-4 py-3 font-medium">Exact</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model} className="border-b border-border last:border-0">
                <td className="px-4 py-3 font-medium">{r.model}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{fmt(r.precision)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{fmt(r.recall)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{fmt(r.f1)}</td>
                <td className="px-4 py-3 font-mono tabular-nums">{fmt(r.exact_match)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="grid gap-4 md:grid-cols-2">
        <article className="rounded-xl border border-border bg-surface p-5">
          <h2 className="font-display text-xl font-medium">Lexicon baseline</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Phrase bank built only from the training split. Case-insensitive
            exact match of annotated surface forms. Strong when examinees reuse
            the same wording; fails on unseen paraphrases and spelling drift.
          </p>
        </article>
        <article className="rounded-xl border border-border bg-surface p-5">
          <h2 className="font-display text-xl font-medium">Hybrid logreg</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Character n-gram logistic regression over keyword-gated token
            n-grams. Fills lexicon misses. HashingVectorizer, class-balanced,
            trained on 80% of annotated notes.
          </p>
        </article>
        <article className="rounded-xl border border-border bg-surface p-5">
          <h2 className="font-display text-xl font-medium">Tiny transformer</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            2-layer Transformer encoder (d=96, 4 heads) trained from scratch as
            a BIO tagger: [CLS] feature [SEP] note [SEP]. CPU-friendly stand-in
            for DistilBERT; a GPU run of DistilBERT is the obvious next lift.
          </p>
        </article>
        <article className="rounded-xl border border-border bg-surface p-5">
          <h2 className="font-display text-xl font-medium">What still fails</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Negation features, rare paraphrases, truncated long notes, and
            annotation inconsistency (offset/text mismatches in the source
            file). See the error panel.
          </p>
        </article>
      </section>

      {errors && (
        <section className="flex flex-col gap-4">
          <h2 className="font-display text-2xl font-medium">Error analysis</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(errors.buckets).map(([name, b]) => (
              <div key={name} className="rounded-md border border-border bg-bg-elevated p-3">
                <p className="text-xs uppercase tracking-[0.12em] text-subtle">{name.replaceAll("_", " ")}</p>
                <p className="mt-1 font-mono tabular-nums text-lg">{fmt(b.f1)} F1</p>
                <p className="text-xs text-muted">n={b.n}</p>
              </div>
            ))}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <ErrorList title="False positives" items={errors.false_positives} />
            <ErrorList title="False negatives" items={errors.false_negatives} />
          </div>
        </section>
      )}
    </div>
  );
}

function ErrorList({
  title,
  items,
}: {
  title: string;
  items: { id: string; feature: string; note_excerpt: string; gold: { text: string }[]; pred: { text: string }[] }[];
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="font-medium">{title}</h3>
      <ul className="mt-3 flex flex-col gap-3">
        {items.slice(0, 6).map((e) => (
          <li key={e.id + e.feature} className="text-sm">
            <p className="font-medium text-fg">{e.feature}</p>
            <p className="text-xs text-muted">
              gold: {e.gold.map((g) => g.text).join(" | ") || "∅"} · pred:{" "}
              {e.pred.map((g) => g.text).join(" | ") || "∅"}
            </p>
            <p className="mt-1 line-clamp-3 font-mono text-xs text-subtle">{e.note_excerpt}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function labelOf(model: string): string {
  if (model === "empty") return "Empty (majority)";
  if (model === "lexicon") return "Lexicon";
  if (model === "hybrid_logreg") return "Hybrid (lexicon + logreg)";
  if (model === "tiny_transformer") return "Tiny transformer";
  return model;
}

function fmt(n: number): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(3);
}
