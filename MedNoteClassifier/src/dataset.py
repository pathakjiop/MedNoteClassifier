"""Patient-note-level splits and token/span example builders.

Leakage rule: every `pn_num` (one clinical note) lives in exactly one of
train / val / test. Splits are stratified by `case_num` so each of the ten
USMLE cases is represented in every partition.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from src.data_loader import AnnotationRow, iter_annotations, merge_annotated
from src.preprocessing import clean_annotation

logger = logging.getLogger("mednote.dataset")


def assign_note_splits(
    train: pd.DataFrame,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> pd.DataFrame:
    """Return a pn_num → split map. Ratios must sum to 1.0."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    notes = train[["pn_num", "case_num"]].drop_duplicates()
    rng = np.random.RandomState(seed)
    parts: list[pd.DataFrame] = []
    for case_num, group in notes.groupby("case_num"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Keep test as the remainder so counts sum to n.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        split = np.array(["train"] * n, dtype=object)
        split[n_train : n_train + n_val] = "val"
        split[n_train + n_val :] = "test"
        chunk = group.loc[idx].copy()
        chunk["split"] = split
        chunk["case_num"] = case_num
        parts.append(chunk)
    mapping = pd.concat(parts, ignore_index=True)
    logger.info(
        "Note splits: %s",
        mapping["split"].value_counts().to_dict(),
    )
    return mapping


def attach_splits(merged: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = merged.merge(mapping[["pn_num", "split"]], on="pn_num", how="left")
    missing = int(out["split"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} annotation rows did not receive a split")
    return out


def cleaned_rows(frame: pd.DataFrame) -> list[AnnotationRow]:
    rows: list[AnnotationRow] = []
    n_issues = 0
    for raw in iter_annotations(frame):
        cleaned, issues = clean_annotation(raw)
        n_issues += len(issues)
        rows.append(cleaned)
    if n_issues:
        logger.info("Span validation flags: %s", n_issues)
    return rows


def by_split(rows: list[AnnotationRow], split: str) -> list[AnnotationRow]:
    return [r for r in rows if r.split == split]


def build_processed_frame(
    train: pd.DataFrame,
    notes: pd.DataFrame,
    features: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, list[AnnotationRow]]:
    merged = merge_annotated(train, notes, features)
    mapping = assign_note_splits(merged, seed=seed)
    merged = attach_splits(merged, mapping)
    rows = cleaned_rows(merged)
    return merged, rows


def iter_split(rows: list[AnnotationRow], split: str) -> Iterator[AnnotationRow]:
    for row in rows:
        if row.split == split:
            yield row
