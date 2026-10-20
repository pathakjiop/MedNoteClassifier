"""Load NBME CSVs and parse annotation / location strings.

The Kaggle train file stores Python-literal lists as strings:

    annotation = "['chest pressure']"
    location   = "['203 217']"

Locations may contain multiple spans, and a single item may encode
discontinuous ranges with a semicolon (`'10 15;20 25'`).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from src.config import FEATURES_CSV, NOTES_CSV, TEST_CSV, TRAIN_CSV

logger = logging.getLogger("mednote.data")


@dataclass
class Span:
    start: int
    end: int
    text: str = ""

    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end

    def char_set(self) -> set[int]:
        return set(range(self.start, self.end))


@dataclass
class AnnotationRow:
    id: str
    case_num: int
    pn_num: int
    feature_num: int
    feature_text: str
    pn_history: str
    spans: list[Span] = field(default_factory=list)
    split: str = ""


def parse_span_list(location_str: str) -> list[tuple[int, int]]:
    """Parse a Kaggle `location` cell into (start, end) character offsets."""
    if location_str is None or (isinstance(location_str, float) and pd.isna(location_str)):
        return []
    raw = str(location_str).strip()
    if raw in {"", "[]", "nan"}:
        return []
    try:
        items = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        items = [raw]
    if not isinstance(items, list):
        items = [items]
    spans: list[tuple[int, int]] = []
    for item in items:
        if item is None:
            continue
        for part in str(item).split(";"):
            part = part.strip()
            if not part:
                continue
            bits = part.replace(",", " ").split()
            if len(bits) < 2:
                continue
            try:
                start, end = int(bits[0]), int(bits[1])
            except ValueError:
                continue
            if end > start >= 0:
                spans.append((start, end))
    return spans


def parse_annotation_list(annotation_str: str) -> list[str]:
    if annotation_str is None or (isinstance(annotation_str, float) and pd.isna(annotation_str)):
        return []
    raw = str(annotation_str).strip()
    if raw in {"", "[]", "nan"}:
        return []
    try:
        items = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return [raw]
    if isinstance(items, list):
        return [str(x) for x in items if str(x).strip()]
    return [str(items)]


def load_raw_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four Kaggle CSVs. Raises FileNotFoundError with a download hint."""
    for path, name in (
        (TRAIN_CSV, "train.csv"),
        (NOTES_CSV, "patient_notes.csv"),
        (FEATURES_CSV, "features.csv"),
        (TEST_CSV, "test.csv"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} at {path}. Download the NBME competition data "
                "from https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/data "
                "and place the CSVs in data/raw/."
            )
    train = pd.read_csv(TRAIN_CSV)
    notes = pd.read_csv(NOTES_CSV)
    features = pd.read_csv(FEATURES_CSV)
    test = pd.read_csv(TEST_CSV)
    logger.info(
        "Loaded train=%s notes=%s features=%s kaggle_test=%s",
        train.shape,
        notes.shape,
        features.shape,
        test.shape,
    )
    return train, notes, features, test


def merge_annotated(train: pd.DataFrame, notes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join annotations to note text and feature rubric strings."""
    merged = train.merge(notes, on=["pn_num", "case_num"], how="left", validate="many_to_one")
    merged = merged.merge(features, on=["feature_num", "case_num"], how="left", validate="many_to_one")
    missing_notes = int(merged["pn_history"].isna().sum())
    missing_feat = int(merged["feature_text"].isna().sum())
    if missing_notes:
        logger.warning("%s annotation rows have no matching patient note", missing_notes)
    if missing_feat:
        logger.warning("%s annotation rows have no matching feature rubric", missing_feat)
    return merged


def row_to_annotation(row: pd.Series) -> AnnotationRow:
    text = "" if pd.isna(row.get("pn_history")) else str(row["pn_history"])
    locations = parse_span_list(row.get("location", "[]"))
    phrases = parse_annotation_list(row.get("annotation", "[]"))
    spans: list[Span] = []
    for i, (start, end) in enumerate(locations):
        snippet = text[start:end] if 0 <= start < end <= len(text) else ""
        if i < len(phrases) and not snippet:
            snippet = phrases[i]
        spans.append(Span(start=start, end=end, text=snippet))
    return AnnotationRow(
        id=str(row["id"]),
        case_num=int(row["case_num"]),
        pn_num=int(row["pn_num"]),
        feature_num=int(row["feature_num"]),
        feature_text="" if pd.isna(row.get("feature_text")) else str(row["feature_text"]),
        pn_history=text,
        spans=spans,
        split=str(row.get("split", "")),
    )


def iter_annotations(frame: pd.DataFrame) -> Iterable[AnnotationRow]:
    for _, row in frame.iterrows():
        yield row_to_annotation(row)
