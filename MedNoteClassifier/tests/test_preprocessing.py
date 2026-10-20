from src.data_loader import Span, parse_annotation_list, parse_span_list
from src.preprocessing import bio_labels, spans_from_bio, tokenize_with_offsets, validate_span


def test_parse_location_single() -> None:
    assert parse_span_list("['696 724']") == [(696, 724)]


def test_parse_location_multi() -> None:
    assert parse_span_list("['70 91', '176 183']") == [(70, 91), (176, 183)]


def test_parse_location_empty() -> None:
    assert parse_span_list("[]") == []


def test_parse_annotation_list() -> None:
    assert parse_annotation_list("['chest pressure']") == ["chest pressure"]


def test_tokenize_offsets_roundtrip() -> None:
    text = "Patient reports headache for two days."
    toks = tokenize_with_offsets(text)
    assert "".join(text[t.start : t.end] for t in toks if t.text.isalnum() or True)
    rebuilt = "".join(t.text for t in toks)
    assert "headache" in rebuilt
    h = next(t for t in toks if t.text.lower() == "headache")
    assert text[h.start : h.end].lower() == "headache"


def test_span_validation() -> None:
    text = "hello world"
    ok, reason = validate_span(text, Span(0, 5, "hello"))
    assert ok and reason == "ok"
    ok, reason = validate_span(text, Span(-1, 2, "x"))
    assert not ok


def test_bio_roundtrip() -> None:
    text = "Patient reports headache today"
    toks = tokenize_with_offsets(text)
    h = next(t for t in toks if t.text == "headache")
    labels = bio_labels(toks, [Span(h.start, h.end, "headache")])
    assert "B" in labels
    decoded = spans_from_bio(toks, labels)
    assert decoded[0].start == h.start
    assert decoded[0].end == h.end
