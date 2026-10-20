"""Clinical-text preprocessing that preserves annotation spans.

We deliberately do **not** stem, lemmatize, or drop stopwords: those steps
destroy clinical meaning and invalidate character offsets. Normalization is
limited to Unicode NFKC and trailing/leading whitespace on copies used for
matching — original `pn_history` is always kept for inference.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from src.data_loader import AnnotationRow, Span

logger = logging.getLogger("mednote.preprocess")

_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Token:
    text: str
    start: int
    end: int


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text) if text else ""


def normalize_whitespace_copy(text: str) -> str:
    """Return a whitespace-collapsed copy. Do not use this for span alignment."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize_with_offsets(text: str) -> list[Token]:
    """Whitespace / punctuation tokenizer that keeps character offsets."""
    return [Token(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


def validate_span(text: str, span: Span) -> tuple[bool, str]:
    """Check that a span is in-bounds and (approximately) matches its stored text."""
    if span.start < 0 or span.end > len(text) or span.end <= span.start:
        return False, "out_of_bounds"
    snippet = text[span.start : span.end]
    if not snippet.strip():
        return False, "empty_snippet"
    if span.text and nfkc(snippet).lower() != nfkc(span.text).lower():
        # NBME has known offset/text mismatches; keep the span but flag it.
        return True, "text_mismatch"
    return True, "ok"


def clean_annotation(row: AnnotationRow) -> tuple[AnnotationRow, list[str]]:
    """Drop invalid spans; keep the original note text untouched."""
    issues: list[str] = []
    kept: list[Span] = []
    for span in row.spans:
        ok, reason = validate_span(row.pn_history, span)
        if not ok:
            issues.append(f"{row.id}:{span.start}-{span.end}:{reason}")
            continue
        if reason != "ok":
            issues.append(f"{row.id}:{span.start}-{span.end}:{reason}")
        kept.append(span)
    row.spans = kept
    return row, issues


def bio_labels(tokens: list[Token], spans: list[Span]) -> list[str]:
    """Assign BIO tags to tokens from character spans."""
    labels = ["O"] * len(tokens)
    for span in spans:
        first = True
        for i, tok in enumerate(tokens):
            if tok.end <= span.start or tok.start >= span.end:
                continue
            labels[i] = "B" if first else "I"
            first = False
    return labels


def spans_from_bio(tokens: list[Token], labels: list[str]) -> list[Span]:
    """Decode BIO labels back to character spans (inclusive of token text)."""
    spans: list[Span] = []
    start: int | None = None
    end: int | None = None
    for tok, lab in zip(tokens, labels):
        if lab == "B":
            if start is not None and end is not None:
                spans.append(Span(start, end))
            start, end = tok.start, tok.end
        elif lab == "I" and start is not None:
            end = tok.end
        else:
            if start is not None and end is not None:
                spans.append(Span(start, end))
            start, end = None, None
    if start is not None and end is not None:
        spans.append(Span(start, end))
    return spans
