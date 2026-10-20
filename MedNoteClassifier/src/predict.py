"""CLI inference for MedNoteClassifier.

    python -m src.predict --text "Patient reports headache for two days."
    python -m src.predict --text-file note.txt --case 9 --model hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import MODELS_BASELINE, MODELS_TRANSFORMER, PROJECT_ROOT
from src.data_loader import AnnotationRow, load_raw_frames
from src.utils import feature_display_name, setup_logging


RESEARCH_WARNING = (
    "WARNING: Research model only. Not a medical device. Not for diagnosis, "
    "treatment, or clinical decision support."
)


def _load_hybrid():
    from src.train_baseline import HybridBaseline

    path = MODELS_BASELINE / "hybrid_baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Baseline checkpoint missing at {path}. Run python -m src.run_pipeline first.")
    return HybridBaseline.load()


def _load_transformer():
    from src.train_transformer import TransformerTagger

    path = MODELS_TRANSFORMER / "span_transformer.pt"
    if not path.exists():
        raise FileNotFoundError(f"Transformer checkpoint missing at {path}.")
    return TransformerTagger.load()


def _features_for_case(case_num: int | None) -> list[tuple[int, str]]:
    _train, _notes, features, _test = load_raw_frames()
    if case_num is None:
        rows = features
    else:
        rows = features[features["case_num"] == case_num]
    return [(int(r.feature_num), str(r.feature_text)) for r in rows.itertuples()]


def predict_text(
    text: str,
    case_num: int | None = None,
    model_name: str = "hybrid",
) -> dict:
    if not text or not str(text).strip():
        return {
            "input_text": text,
            "entities": [],
            "model": model_name,
            "warning": RESEARCH_WARNING,
            "error": "empty_input",
        }
    text = str(text)
    if len(text) > 20000:
        text = text[:20000]
    if model_name == "transformer":
        model = _load_transformer()
        predict_fn = model.predict_row
        loaded = "tiny-transformer-encoder"
    else:
        model = _load_hybrid()
        predict_fn = model.predict_row
        loaded = "hybrid-lexicon-logreg"

    entities = []
    for feature_num, feature_text in _features_for_case(case_num):
        row = AnnotationRow(
            id="cli",
            case_num=case_num if case_num is not None else -1,
            pn_num=-1,
            feature_num=feature_num,
            feature_text=feature_text,
            pn_history=text,
            spans=[],
        )
        spans = predict_fn(row)
        for span in spans:
            entities.append(
                {
                    "text": text[span.start : span.end],
                    "type": feature_display_name(feature_text),
                    "feature_num": feature_num,
                    "feature_text": feature_text,
                    "start": span.start,
                    "end": span.end,
                }
            )
    entities.sort(key=lambda e: (e["start"], e["end"]))
    return {
        "input_text": text,
        "entities": entities,
        "model": loaded,
        "case_num": case_num,
        "n_entities": len(entities),
        "warning": RESEARCH_WARNING,
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Extract clinical features from a patient note.")
    parser.add_argument("--text", type=str, default=None, help="Patient note text")
    parser.add_argument("--text-file", type=Path, default=None, help="Path to a UTF-8 note")
    parser.add_argument("--case", type=int, default=None, help="USMLE case number 0-9 (all features if omitted)")
    parser.add_argument("--model", choices=["hybrid", "transformer"], default="hybrid")
    args = parser.parse_args(argv)

    if args.text is None and args.text_file is None:
        parser.error("provide --text or --text-file")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    result = predict_text(text, case_num=args.case, model_name=args.model)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
