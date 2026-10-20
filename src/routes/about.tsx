import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/about")({ component: AboutPage });

export function AboutPage() {
  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-8">
      <header>
        <p className="text-xs uppercase tracking-[0.16em] text-subtle">Model card</p>
        <h1 className="mt-2 font-display text-4xl font-semibold tracking-tight">
          Intended as a portfolio NLP system, not a clinician.
        </h1>
      </header>

      <section className="flex flex-col gap-2">
        <h2 className="font-display text-2xl font-medium">Purpose</h2>
        <p className="text-muted leading-relaxed">
          Extract mentions of exam-rubric clinical features from de-identified
          patient notes. Designed to show clinical NLP practice: span
          validation, note-level splits, a linear baseline, a transformer
          tagger, and honest error analysis.
        </p>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-display text-2xl font-medium">Intended use</h2>
        <ul className="list-disc space-y-1 pl-5 text-muted">
          <li>Education, research, and hiring-portfolio demonstration</li>
          <li>Offline scoring experiments on the public NBME corpus</li>
        </ul>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-display text-2xl font-medium">Out of scope</h2>
        <ul className="list-disc space-y-1 pl-5 text-muted">
          <li>Diagnosis, triage, treatment, or charting assistance</li>
          <li>Live EHR deployment or HIPAA/GDPR compliance claims</li>
          <li>SOAP section classification (this dataset does not label sections)</li>
        </ul>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-display text-2xl font-medium">Training data</h2>
        <p className="text-muted leading-relaxed">
          Kaggle competition “NBME — Score Clinical Patient Notes”: 1,000
          annotated notes (100 per case) and 42,146 unlabeled notes. Features
          are 143 rubric concepts. Empty annotations are true negatives.
        </p>
      </section>

      <section className="flex flex-col gap-2">
        <h2 className="font-display text-2xl font-medium">Limitations and bias</h2>
        <p className="text-muted leading-relaxed">
          Notes are written by exam candidates in a simulated setting, not by
          practicing clinicians in hospitals. Language is USMLE-English, with
          frequent misspellings. The ten cases do not represent the full
          distribution of clinical documentation. The model will overfit exam
          phrasing and underperform on real EHR text, other languages, and
          rare findings.
        </p>
      </section>

      <section className="rounded-xl border border-danger/30 bg-bg-elevated p-5">
        <h2 className="font-display text-xl font-medium text-danger">Clinical safety</h2>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          Do not paste identifiable patient data into this demo. Do not use
          predictions as medical advice. A missed negation or a hallucinated
          span can invert clinical meaning. This software has not been
          clinically validated.
        </p>
      </section>
    </div>
  );
}
