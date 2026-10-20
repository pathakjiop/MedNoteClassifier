"""Baselines for NBME span extraction.

1. Majority / empty baseline — always predict no span.
2. Lexicon matcher — exact + whitespace-normalized lookup of training phrases.
3. Character n-gram logistic regression over candidate token n-grams.

The lexicon is surprisingly strong on this dataset because physician annotators
reuse a small set of surface forms per rubric feature. Logistic regression
adds generalization to unseen paraphrases.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    BASELINE_C,
    BASELINE_NGRAM_RANGE,
    MAX_CANDIDATE_TOKENS,
    MODELS_BASELINE,
    NEGATIVE_PER_POSITIVE,
    SEED,
)
from src.data_loader import AnnotationRow, Span
from src.evaluate import micro_f1, span_char_set
from src.preprocessing import tokenize_with_offsets
from src.utils import ensure_dir, normalize_for_match

logger = logging.getLogger("mednote.baseline")


STOP = {"or", "of", "no", "to", "a", "the", "and", "with", "in", "for", "on", "at"}
BANNED_PHRASES = STOP | {"old", "st", "ing", "er", "ed", "ly", "al", "an", "as", "is", "be", "by"}


def _keep_phrase(phrase: str, feature_text: str) -> bool:
    p = phrase.strip()
    if not p:
        return False
    if all(ch in ",./;:-–—()[]{}\"'`" for ch in p):
        return False
    low = p.lower()
    if low in BANNED_PHRASES:
        return False
    feat = feature_text.lower()
    if len(p) == 1:
        return p.upper() in {"M", "F"} and feat in {"male", "female"}
    if len(p) == 2:
        return low in {"yo", "hx", "mi", "ed"} or p.upper() in {"IUD"}
    return len(p) >= 3


def feature_keywords(feature_text: str) -> set[str]:
    tokens = feature_text.lower().replace("-", " ").replace("/", " ").split()
    return {t for t in tokens if t not in STOP and len(t) > 2}


def _candidate_ngrams(
    text: str,
    max_tokens: int = MAX_CANDIDATE_TOKENS,
    keywords: set[str] | None = None,
) -> list[Span]:
    tokens = tokenize_with_offsets(text)
    spans: list[Span] = []
    n = len(tokens)
    for i in range(n):
        has_kw = False
        for j in range(i, min(i + max_tokens, n)):
            if keywords and tokens[j].text.lower() in keywords:
                has_kw = True
            if keywords and not has_kw:
                continue
            start = tokens[i].start
            end = tokens[j].end
            spans.append(Span(start, end, text[start:end]))
    return spans


def _overlap(a: Span, b: Span) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union else 0.0


@dataclass
class LexiconModel:
    """Per-feature phrase bank built only from the training split."""

    phrases: dict[int, list[tuple[str, str, int]]] = field(default_factory=dict)
    feature_text: dict[int, str] = field(default_factory=dict)

    def fit(self, rows: Iterable[AnnotationRow]) -> "LexiconModel":
        bags: dict[int, Counter[str]] = defaultdict(Counter)
        for row in rows:
            self.feature_text[row.feature_num] = row.feature_text
            for span in row.spans:
                phrase = span.text.strip()
                if phrase and _keep_phrase(phrase, row.feature_text):
                    bags[row.feature_num][phrase] += 1
        self.phrases = {
            fid: [(p, normalize_for_match(p), c) for p, c in counts.most_common()]
            for fid, counts in bags.items()
        }
        logger.info("Lexicon features with phrases: %s", len(self.phrases))
        return self

    def predict_row(self, row: AnnotationRow) -> list[Span]:
        return self.predict_text(row.pn_history, row.feature_num)

    def predict_text(self, text: str, feature_num: int) -> list[Span]:
        bank = self.phrases.get(int(feature_num), [])
        if not bank or not text:
            return []
        lowered = text.lower()
        found: list[Span] = []
        occupied: list[tuple[int, int]] = []
        ranked = sorted(bank, key=lambda t: (-len(t[0]), -t[2]))
        for phrase, _norm, _count in ranked:
            needle = phrase.lower().strip()
            if not _keep_phrase(phrase, self.feature_text.get(int(feature_num), "")):
                continue
            start = 0
            while True:
                idx = lowered.find(needle, start)
                if idx < 0:
                    break
                end = idx + len(needle)
                start = idx + max(1, len(needle))
                if len(needle) <= 4:
                    left_ok = idx == 0 or not (text[idx - 1].isalnum())
                    right_ok = end >= len(text) or not (text[end].isalnum())
                    if not (left_ok and right_ok):
                        continue
                if any(not (end <= a or idx >= b) for a, b in occupied):
                    continue
                occupied.append((idx, end))
                found.append(Span(idx, end, text[idx:end]))
        return found


def _locate_substring(
    text: str, lowered: str, phrase: str, _norm_idx: int, _norm_len: int
) -> tuple[int, int] | None:
    needle = phrase.lower()
    pos = lowered.find(needle)
    if pos >= 0:
        return pos, pos + len(phrase)
    collapsed = " ".join(needle.split())
    window = " ".join(lowered.split())
    # Fallback: walk original characters skipping extra whitespace.
    target = collapsed
    n = len(text)
    i = 0
    while i < n:
        j = i
        k = 0
        while j < n and k < len(target):
            cj = text[j].lower()
            if cj.isspace():
                if k < len(target) and target[k] == " ":
                    k += 1
                    while j < n and text[j].isspace():
                        j += 1
                    continue
                j += 1
                continue
            if cj == target[k]:
                k += 1
                j += 1
            else:
                break
        if k == len(target):
            return i, j
        i += 1
    return None


def _featurize(span_text: str, feature_text: str, note_len: int, start: int, end: int) -> str:
    rel = start / max(note_len, 1)
    return (
        f"feat={feature_text.lower()} "
        f"span={span_text.lower()} "
        f"len={end - start} rel={rel:.2f}"
    )


class SpanLogReg:
    """Classify candidate n-grams as feature spans vs. not."""

    def __init__(self) -> None:
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "vec",
                    HashingVectorizer(
                        analyzer="char_wb",
                        ngram_range=BASELINE_NGRAM_RANGE,
                        n_features=2**16,
                        alternate_sign=False,
                        lowercase=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=BASELINE_C,
                        max_iter=200,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        self.threshold: float = 0.62

    def _samples(self, rows: list[AnnotationRow], rng: np.random.RandomState) -> tuple[list[str], np.ndarray]:
        X: list[str] = []
        y: list[int] = []
        for row in rows:
            gold = row.spans
            if gold:
                for g in gold:
                    X.append(_featurize(g.text, row.feature_text, len(row.pn_history), g.start, g.end))
                    y.append(1)
            cands = _candidate_ngrams(row.pn_history, keywords=feature_keywords(row.feature_text) or None)
            negatives: list[Span] = []
            for c in cands:
                if gold and max(_overlap(c, g) for g in gold) >= 0.6:
                    continue
                negatives.append(c)
            n_pos = max(len(gold), 1)
            k = min(len(negatives), n_pos * NEGATIVE_PER_POSITIVE)
            if k and negatives:
                pick = rng.choice(len(negatives), size=k, replace=False)
                for i in pick:
                    c = negatives[i]
                    X.append(_featurize(c.text, row.feature_text, len(row.pn_history), c.start, c.end))
                    y.append(0)
            elif not gold:
                # True-negative instance: a handful of random candidates.
                if negatives:
                    pick = rng.choice(len(negatives), size=min(2, len(negatives)), replace=False)
                    for i in pick:
                        c = negatives[i]
                        X.append(_featurize(c.text, row.feature_text, len(row.pn_history), c.start, c.end))
                        y.append(0)
        return X, np.asarray(y, dtype=np.int32)

    def fit(self, rows: list[AnnotationRow]) -> "SpanLogReg":
        rng = np.random.RandomState(SEED)
        X, y = self._samples(rows, rng)
        logger.info("LogReg samples: %s (positives=%s)", len(y), int(y.sum()))
        self.pipeline.fit(X, y)
        return self

    def predict_text(
        self, text: str, feature_text: str, threshold: float | None = None
    ) -> list[Span]:
        thr = self.threshold if threshold is None else threshold
        kws = feature_keywords(feature_text)
        hay = text.lower()
        if kws and not any(k in hay for k in kws):
            return []
        cands = _candidate_ngrams(text, keywords=kws or None)
        if not cands:
            return []
        # Cap candidates so inference stays interactive.
        if len(cands) > 400:
            cands = sorted(cands, key=lambda s: abs((s.end - s.start) - 12))[:400]
        feats = [_featurize(c.text, feature_text, len(text), c.start, c.end) for c in cands]
        proba = self.pipeline.predict_proba(feats)[:, 1]
        ranked = sorted(zip(cands, proba), key=lambda t: t[1], reverse=True)
        chosen: list[Span] = []
        for cand, p in ranked:
            if p < thr:
                continue
            if any(_overlap(cand, s) > 0.4 for s in chosen):
                continue
            chosen.append(cand)
            if len(chosen) >= 4:
                break
        return chosen

    def predict_row(self, row: AnnotationRow) -> list[Span]:
        return self.predict_text(row.pn_history, row.feature_text)


class HybridBaseline:
    """Lexicon first; logistic regression fills gaps (no lexicon hit)."""

    def __init__(self) -> None:
        self.lexicon = LexiconModel()
        self.logreg = SpanLogReg()

    def fit(self, rows: list[AnnotationRow]) -> "HybridBaseline":
        t0 = time.time()
        self.lexicon.fit(rows)
        self.logreg.fit(rows)
        logger.info("Hybrid baseline trained in %.1fs", time.time() - t0)
        return self

    def predict_row(self, row: AnnotationRow) -> list[Span]:
        lex = self.lexicon.predict_row(row)
        if lex:
            return lex
        return self.logreg.predict_row(row)

    def save(self, directory=MODELS_BASELINE) -> None:
        directory = ensure_dir(directory)
        joblib.dump(self, directory / "hybrid_baseline.joblib")
        logger.info("Saved baseline to %s", directory)

    @classmethod
    def load(cls, directory=MODELS_BASELINE) -> "HybridBaseline":
        return joblib.load(directory / "hybrid_baseline.joblib")


def empty_predict(_row: AnnotationRow) -> list[Span]:
    return []


def evaluate_predictor(rows: list[AnnotationRow], predict_fn) -> dict[str, float]:
    pairs = []
    for row in rows:
        pred = span_char_set(predict_fn(row))
        gold = span_char_set(row.spans)
        pairs.append((pred, gold))
    return micro_f1(pairs)
