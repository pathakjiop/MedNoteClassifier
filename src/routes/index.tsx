import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { ChevronDown, LoaderCircle, Stethoscope } from "lucide-react";
import { AnnotatedNote } from "@/components/annotated-note";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { detectCase, extractNote, type Entity, type Feature } from "@/lib/extract";
import { loadMlBundle, type CaseProfile, type ExampleNote } from "@/lib/ml-data";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [cases, setCases] = useState<CaseProfile[]>([]);
  const [examples, setExamples] = useState<ExampleNote[]>([]);
  const [ready, setReady] = useState(false);
  const [text, setText] = useState("");
  const [caseMode, setCaseMode] = useState<"auto" | number>("auto");
  const [entities, setEntities] = useState<Entity[]>([]);
  const [resolvedCase, setResolvedCase] = useState<number>(0);
  const [ran, setRan] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadMlBundle().then((bundle) => {
      if (cancelled) return;
      setFeatures(bundle.features);
      setCases(bundle.cases);
      setExamples(bundle.examples);
      setReady(true);
      if (bundle.examples[0]) {
        setText(bundle.examples[0].text);
        setCaseMode(bundle.examples[0].case_num);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const caseOptions = useMemo(() => cases, [cases]);

  function run(nextText = text, mode: "auto" | number = caseMode) {
    if (!nextText.trim() || features.length === 0) {
      setEntities([]);
      setRan(true);
      return;
    }
    const caseNum = mode === "auto" ? detectCase(nextText, features, cases) : mode;
    const feats = features.filter((f) => f.case_num === caseNum);
    setResolvedCase(caseNum);
    setEntities(extractNote(nextText, feats));
    setRan(true);
  }

  const profile = cases.find((c) => c.case_num === resolvedCase);

  return (
    <div className="flex flex-col gap-10">
      <section className="max-w-3xl">
        <p className="text-xs uppercase tracking-[0.16em] text-subtle">Feature span extraction</p>
        <h1 className="mt-2 font-display text-4xl font-semibold leading-tight tracking-tight text-fg sm:text-5xl">
          Highlight the clinical phrases a scoring rubric actually cares about.
        </h1>
        <p className="mt-4 max-w-2xl text-base leading-relaxed text-muted">
          MedNoteClassifier is a research NLP model trained on the Kaggle NBME
          patient-note corpus. It does not classify SOAP sections. It extracts
          character spans for 143 rubric features across 10 USMLE-style cases.
        </p>
      </section>

      <section className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <div className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 shadow-soft sm:p-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <label htmlFor="note" className="text-sm font-medium">
              Patient note
            </label>
            <div className="relative">
              <label className="sr-only" htmlFor="case">
                Case
              </label>
              <select
                id="case"
                className="h-11 min-h-11 appearance-none rounded-sm border border-border bg-bg-elevated py-2 pl-3 pr-9 text-sm"
                value={caseMode === "auto" ? "auto" : String(caseMode)}
                onChange={(e) => {
                  const v = e.target.value;
                  setCaseMode(v === "auto" ? "auto" : Number(v));
                }}
              >
                <option value="auto">Auto-detect case</option>
                {caseOptions.map((c) => (
                  <option key={c.case_num} value={c.case_num}>
                    {c.title}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-3 size-4 text-subtle" />
            </div>
          </div>
          <textarea
            id="note"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={14}
            className="min-h-56 w-full resize-y rounded-md border border-border bg-bg-elevated p-4 font-mono text-sm leading-6 text-fg outline-none ring-ring focus:ring-2"
            placeholder="Paste a de-identified history of present illness…"
          />
          <div className="flex flex-wrap gap-2">
            {examples.map((ex) => (
              <button
                key={ex.pn_num}
                type="button"
                className="min-h-11 rounded-full border border-border bg-bg-elevated px-3 py-1.5 text-xs text-muted hover:text-fg"
                onClick={() => {
                  setText(ex.text);
                  setCaseMode(ex.case_num);
                  run(ex.text, ex.case_num);
                }}
              >
                Case {ex.case_num} example
              </button>
            ))}
          </div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <Button type="button" onClick={() => run()} disabled={!ready} className="min-h-11">
              {ready ? (
                "Extract features"
              ) : (
                <>
                  <LoaderCircle className="size-4 animate-spin" />
                  Loading model
                </>
              )}
            </Button>
            <Button
              type="button"
              variant="secondary"
              className="min-h-11"
              onClick={() => {
                setText("");
                setEntities([]);
                setRan(false);
              }}
            >
              Clear
            </Button>
            <p className="text-xs text-subtle sm:ml-auto">
              Demo inference uses the train-split lexicon. Python CLI serves logistic and
              transformer checkpoints.
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          {ran ? (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Badge className="border-primary/20 bg-highlight text-primary">Case {resolvedCase}</Badge>
                <span className="text-sm text-muted">{profile?.title}</span>
                <span className="ml-auto font-mono text-xs tabular-nums text-subtle">
                  {entities.length} spans
                </span>
              </div>
              <AnnotatedNote text={text} entities={entities} />
              <ul className="flex flex-col gap-2">
                {entities.length === 0 ? (
                  <li className="rounded-md border border-border bg-bg-elevated p-4 text-sm text-muted">
                    No rubric spans found for this case. Try another example or a fuller history.
                  </li>
                ) : (
                  entities.map((e) => (
                    <li
                      key={`${e.feature_num}-${e.start}`}
                      className="flex flex-col gap-1 rounded-md border border-border bg-bg-elevated px-3 py-2 sm:flex-row sm:items-baseline sm:justify-between"
                    >
                      <div>
                        <p className="text-sm font-medium text-fg">{e.type}</p>
                        <p className="font-mono text-xs text-muted">
                          “{e.text}” · {e.start}–{e.end}
                        </p>
                      </div>
                      <span className="font-mono text-xs tabular-nums text-subtle">
                        {(e.confidence * 100).toFixed(0)}% · {e.source}
                      </span>
                    </li>
                  ))
                )}
              </ul>
            </>
          ) : (
            <div className="flex min-h-72 flex-col justify-between rounded-xl border border-dashed border-border-strong bg-bg-elevated p-6">
              <div>
                <Stethoscope className="size-5 text-primary" />
                <h2 className="mt-3 font-display text-2xl font-medium">Run extraction on a note</h2>
                <p className="mt-2 text-sm leading-relaxed text-muted">
                  Spans are scored with phrases learned from the training split only — no
                  test-note leakage. Each hit is a rubric feature, not a diagnosis.
                </p>
              </div>
              <p className="text-xs text-subtle">
                Model family: lexicon baseline + char-n-gram logistic regression + tiny
                Transformer encoder (BIO tags).
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-3">
        {[
          { k: "42,146", v: "patient notes in the public corpus" },
          { k: "1,000", v: "notes with physician span annotations" },
          { k: "143", v: "case-specific clinical features" },
        ].map((s) => (
          <div key={s.v} className="rounded-lg border border-border bg-surface px-4 py-5">
            <p className="font-display text-3xl font-medium tabular-nums text-primary">{s.k}</p>
            <p className="mt-1 text-sm text-muted">{s.v}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
