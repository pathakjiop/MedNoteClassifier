import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { loadMlBundle, type CaseProfile, type EdaFile } from "@/lib/ml-data";

export const Route = createFileRoute("/dataset")({ component: DatasetPage });

function DatasetPage() {
  const [eda, setEda] = useState<EdaFile | null>(null);
  const [cases, setCases] = useState<CaseProfile[]>([]);

  useEffect(() => {
    loadMlBundle().then((b) => {
      setEda(b.eda);
      setCases(b.cases);
    });
  }, []);

  const caseBars = Object.entries(eda?.features_per_case ?? {}).map(([k, v]) => ({
    case: `C${k}`,
    features: v,
  }));
  const phrases = (eda?.top_phrases ?? []).slice(0, 12);

  return (
    <div className="flex flex-col gap-10">
      <header className="max-w-2xl">
        <p className="text-xs uppercase tracking-[0.16em] text-subtle">NBME corpus</p>
        <h1 className="mt-2 font-display text-4xl font-semibold tracking-tight">
          The labels are rubric features, not SOAP headings.
        </h1>
        <p className="mt-3 text-muted leading-relaxed">
          Inspected from the official Kaggle files. Patient notes are exam artifacts
          from USMLE Step 2 CS. The modeling target is character-span extraction
          for each (note, feature) pair.
        </p>
      </header>

      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat k={eda?.n_unique_notes_all ?? 42146} l="Notes" />
        <Stat k={eda?.n_annotated_notes ?? 1000} l="Annotated notes" />
        <Stat k={eda?.n_train_rows ?? 14300} l="Train rows" />
        <Stat k={eda?.n_features ?? 143} l="Features" />
        <Stat k={eda?.n_nonempty_annotations ?? 0} l="Non-empty spans" />
        <Stat k={eda?.n_empty_annotations ?? 0} l="Empty (true negatives)" />
        <Stat k={Number((eda?.span_len_mean ?? 0).toFixed(1))} l="Mean span chars" />
        <Stat k={eda?.invalid_spans ?? 0} l="Invalid spans flagged" />
      </dl>

      <section className="grid gap-6 lg:grid-cols-2">
        <figure className="rounded-xl border border-border bg-surface p-4">
          <figcaption className="mb-3 text-sm font-medium">Features per case</figcaption>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={caseBars}>
                <CartesianGrid stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="case" stroke="var(--color-muted)" fontSize={12} />
                <YAxis stroke="var(--color-muted)" fontSize={12} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--color-bg-elevated)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                  }}
                />
                <Bar dataKey="features" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </figure>
        <figure className="rounded-xl border border-border bg-surface p-4">
          <figcaption className="mb-3 text-sm font-medium">Most common annotated phrases</figcaption>
          <ol className="flex flex-col gap-2">
            {phrases.map(([p, n]) => (
              <li key={p} className="flex items-baseline justify-between gap-3 text-sm">
                <span className="truncate text-fg">“{p}”</span>
                <span className="font-mono tabular-nums text-subtle">{n}</span>
              </li>
            ))}
          </ol>
        </figure>
      </section>

      <section>
        <h2 className="font-display text-2xl font-medium">Ten cases, 143 features</h2>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          Case titles below are derived from the rubric (age, sex, leading
          symptom). They are not official NBME case names.
        </p>
        <div className="mt-5 grid gap-3 md:grid-cols-2">
          {cases.map((c) => (
            <article key={c.case_num} className="rounded-lg border border-border bg-bg-elevated p-4">
              <h3 className="font-medium">{c.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                {c.features.slice(0, 8).join(" · ")}
                {c.features.length > 8 ? " …" : ""}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section>
        <h2 className="font-display text-2xl font-medium">Figures from the training run</h2>
        <p className="mt-1 text-sm text-muted">Generated from the Kaggle CSVs, not synthetic data.</p>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {[
            ["label_distribution.png", "Top features by annotated mentions"],
            ["annotation_length.png", "Span length in characters"],
            ["annotations_per_note.png", "Spans per annotated note"],
            ["confusion_matrix.png", "Presence detection on the test split"],
          ].map(([file, caption]) => (
            <figure key={file} className="rounded-xl border border-border bg-bg-elevated p-3">
              <img
                src={`/ml/figures/${file}`}
                alt={caption}
                className="w-full rounded-md"
              />
              <figcaption className="mt-2 text-xs text-subtle">{caption}</figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-surface p-5 text-sm leading-relaxed text-muted">
        <h2 className="font-display text-xl font-medium text-fg">Leakage control</h2>
        <p className="mt-2">
          Splits are 80 / 10 / 10 by <code className="font-mono text-fg">pn_num</code>,
          stratified by case, seed 42. A note never appears in two splits. The
          public corpus does not include a stable patient identifier beyond the
          note id — grouping is at the note level, which is the right unit for
          this exam dataset.
        </p>
      </section>
    </div>
  );
}

function Stat({ k, l }: { k: number; l: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-4">
      <dt className="text-xs uppercase tracking-[0.12em] text-subtle">{l}</dt>
      <dd className="mt-1 font-display text-2xl font-medium tabular-nums text-fg">
        {k.toLocaleString()}
      </dd>
    </div>
  );
}
