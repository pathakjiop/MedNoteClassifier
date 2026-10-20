import type { Feature } from "./extract";

export type CaseProfile = {
  case_num: number;
  n_features: number;
  age_cue: string | null;
  sex_cue: string | null;
  features: string[];
  title: string;
};

export type ExampleNote = {
  pn_num: number;
  case_num: number;
  text: string;
  entities: {
    start: number;
    end: number;
    text: string;
    type: string;
    feature_num: number;
  }[];
};

export type MetricsFile = {
  comparison: {
    model: string;
    split: string;
    precision: number;
    recall: number;
    f1: number;
    exact_match: number;
    support: number;
  }[];
  transformer?: Record<string, unknown>;
};

export type EdaFile = {
  n_train_rows: number;
  n_unique_notes_all: number;
  n_annotated_notes: number;
  n_features: number;
  n_cases: number;
  n_empty_annotations: number;
  n_nonempty_annotations: number;
  note_char_len_mean: number;
  span_len_mean: number;
  invalid_spans: number;
  mean_spans_per_annotated_note: number;
  top_phrases: [string, number][];
  features_per_case: Record<string, number>;
};

export type ErrorsFile = {
  false_positives: ErrorExample[];
  false_negatives: ErrorExample[];
  buckets: Record<string, { precision: number; recall: number; f1: number; n: number }>;
  hardest_features: { feature_num: number; display: string; f1: number; precision: number; recall: number }[];
};

export type ErrorExample = {
  id: string;
  feature: string;
  gold: { start: number; end: number; text: string }[];
  pred: { start: number; end: number; text: string }[];
  note_excerpt: string;
};

async function loadJson<T>(url: string): Promise<T | null> {
  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function loadMlBundle() {
  const [features, cases, examples, metrics, eda, errors] = await Promise.all([
    loadJson<Feature[]>("/ml/features.json"),
    loadJson<CaseProfile[]>("/ml/case_profiles.json"),
    loadJson<ExampleNote[]>("/ml/examples.json"),
    loadJson<MetricsFile>("/ml/metrics.json"),
    loadJson<EdaFile>("/ml/eda.json"),
    loadJson<ErrorsFile>("/ml/errors.json"),
  ]);
  return {
    features: features ?? [],
    cases: cases ?? [],
    examples: examples ?? [],
    metrics,
    eda,
    errors,
  };
}
