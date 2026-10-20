import pytest

from src.data_loader import AnnotationRow
from src.predict import predict_text
from src.train_baseline import LexiconModel, _keep_phrase


def test_lexicon_extracts_known_phrase() -> None:
    model = LexiconModel()
    model.phrases = {9: [("heart pounding", "heart pounding", 4)]}
    model.feature_text = {9: "heart-pounding-OR-heart-racing"}
    text = "17-year-old male with heart pounding for two months."
    spans = model.predict_text(text, 9)
    assert spans
    assert "heart pounding" in spans[0].text.lower()


def test_keep_phrase_drops_junk() -> None:
    assert not _keep_phrase("s", "Family-history-of-MI-OR-Family-history-of-myocardial-infarction")
    assert not _keep_phrase("and", "Shortness-of-breath")
    assert not _keep_phrase(",", "No-vaginal-discharge")
    assert _keep_phrase("M", "Male")
    assert _keep_phrase("chest pressure", "Chest-pressure")


def test_lexicon_ignores_fragment_inside_word() -> None:
    model = LexiconModel()
    model.phrases = {0: [("st", "st", 1)]}
    model.feature_text = {0: "Family-history-of-MI-OR-Family-history-of-myocardial-infarction"}
    assert model.predict_text("chest pressure", 0) == []



def test_lexicon_extracts_known_phrase() -> None:
    model = LexiconModel()
    model.phrases = {9: [("heart pounding", "heart pounding", 4)]}
    model.feature_text = {9: "heart-pounding-OR-heart-racing"}
    text = "17-year-old male with heart pounding for two months."
    spans = model.predict_text(text, 9)
    assert spans
    assert "heart pounding" in spans[0].text.lower()


def test_predict_empty_input() -> None:
    result = predict_text("   ")
    assert result["error"] == "empty_input"
    assert result["entities"] == []
    assert "Research model" in result["warning"]


def test_predict_rejects_blank_none() -> None:
    result = predict_text("")
    assert result["n_entities"] if "n_entities" in result else result["entities"] == []


@pytest.mark.skipif(
    True, reason="full model load is covered in pipeline; keep unit tests offline-friendly"
)
def test_predict_long_text_truncates() -> None:
    result = predict_text("headache " * 5000, case_num=9, model_name="hybrid")
    assert len(result["input_text"]) <= 20000
