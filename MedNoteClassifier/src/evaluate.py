"""NBME character-level micro-F1 and supporting metrics.

The competition scores predicted vs. gold *character sets* for every
(note, feature) instance, including empty annotations. Two empty sets
count as a true positive of 1 so that correct "feature not mentioned"
predictions are rewarded.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.data_loader import Span


def span_char_set(spans: Sequence[Span] | Sequence[tuple[int, int]]) -> set[int]:
    chars: set[int] = set()
    for s in spans:
        if isinstance(s, Span):
            start, end = s.start, s.end
        else:
            start, end = int(s[0]), int(s[1])
        chars.update(range(start, end))
    return chars


def instance_pr(pred: set[int], gold: set[int]) -> tuple[int, int, int]:
    """Return (tp, fp, fn) using the Kaggle empty-empty convention."""
    if not pred and not gold:
        return 1, 0, 0
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return tp, fp, fn


def micro_f1(pairs: Iterable[tuple[set[int], set[int]]]) -> dict[str, float]:
    tp = fp = fn = 0
    n = 0
    exact = 0
    for pred, gold in pairs:
        n += 1
        t, f_p, f_n = instance_pr(pred, gold)
        tp += t
        fp += f_p
        fn += f_n
        if pred == gold:
            exact += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact / n if n else 0.0,
        "support": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def per_feature_f1(
    ids: Sequence[int],
    pairs: Sequence[tuple[set[int], set[int]]],
) -> dict[int, dict[str, float]]:
    buckets: dict[int, list[tuple[set[int], set[int]]]] = {}
    for feat_id, pair in zip(ids, pairs):
        buckets.setdefault(int(feat_id), []).append(pair)
    return {k: micro_f1(v) for k, v in buckets.items()}


def format_metrics_table(rows: list[dict[str, object]]) -> str:
    header = "| Model | Precision | Recall | F1 | Exact match |"
    sep = "| --- | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['exact_match']:.4f} |"
        )
    return "\n".join(lines)
