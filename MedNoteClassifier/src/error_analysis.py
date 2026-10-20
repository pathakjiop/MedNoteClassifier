"""Error analysis over a held-out split: FP / FN, rare labels, negation, length."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Callable

from src.data_loader import AnnotationRow, Span
from src.evaluate import instance_pr, span_char_set
from src.utils import feature_display_name

logger = logging.getLogger("mednote.errors")


def _span_dump(spans: list[Span], text: str) -> list[dict]:
    out = []
    for s in spans:
        out.append(
            {
                "start": s.start,
                "end": s.end,
                "text": text[s.start : s.end],
            }
        )
    return out


def analyze_errors(
    rows: list[AnnotationRow],
    predict_fn: Callable[[AnnotationRow], list[Span]],
    max_examples: int = 40,
) -> dict:
    fp_examples: list[dict] = []
    fn_examples: list[dict] = []
    by_feature = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    by_bucket = {
        "empty_gold": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
        "short_span": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
        "long_span": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
        "negation_feature": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
        "rare_feature": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
        "long_note": {"tp": 0, "fp": 0, "fn": 0, "n": 0},
    }
    feat_support = Counter(r.feature_num for r in rows if r.spans)
    rare = {f for f, c in feat_support.items() if c <= 8}

    for row in rows:
        pred_spans = predict_fn(row)
        pred = span_char_set(pred_spans)
        gold = span_char_set(row.spans)
        tp, fp, fn = instance_pr(pred, gold)
        rec = by_feature[row.feature_num]
        rec["tp"] += tp
        rec["fp"] += fp
        rec["fn"] += fn

        gold_len = sum(s.end - s.start for s in row.spans)
        buckets = []
        if not row.spans:
            buckets.append("empty_gold")
        if gold_len and gold_len <= 8:
            buckets.append("short_span")
        if gold_len >= 40:
            buckets.append("long_span")
        if row.feature_text.lower().startswith("no-") or "no-" in row.feature_text.lower():
            buckets.append("negation_feature")
        if row.feature_num in rare:
            buckets.append("rare_feature")
        if len(row.pn_history) >= 900:
            buckets.append("long_note")
        for b in buckets:
            by_bucket[b]["tp"] += tp
            by_bucket[b]["fp"] += fp
            by_bucket[b]["fn"] += fn
            by_bucket[b]["n"] += 1

        if fp and len(fp_examples) < max_examples:
            fp_examples.append(
                {
                    "id": row.id,
                    "feature": feature_display_name(row.feature_text),
                    "gold": _span_dump(row.spans, row.pn_history),
                    "pred": _span_dump(pred_spans, row.pn_history),
                    "note_excerpt": _excerpt(row.pn_history, pred_spans or row.spans),
                }
            )
        if fn and not pred and row.spans and len(fn_examples) < max_examples:
            fn_examples.append(
                {
                    "id": row.id,
                    "feature": feature_display_name(row.feature_text),
                    "gold": _span_dump(row.spans, row.pn_history),
                    "pred": [],
                    "note_excerpt": _excerpt(row.pn_history, row.spans),
                }
            )

    def _f1(d: dict) -> dict:
        p = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        r = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {**d, "precision": p, "recall": r, "f1": f1}

    hardest = []
    for fid, rec in by_feature.items():
        scored = _f1(rec)
        scored["feature_num"] = fid
        hardest.append(scored)
    hardest.sort(key=lambda x: x["f1"])

    return {
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
        "buckets": {k: _f1(v) for k, v in by_bucket.items()},
        "hardest_features": hardest[:15],
    }


def _excerpt(text: str, spans: list[Span], window: int = 80) -> str:
    if not spans:
        return text[:160]
    start = max(0, min(s.start for s in spans) - window)
    end = min(len(text), max(s.end for s in spans) + window)
    return text[start:end]
