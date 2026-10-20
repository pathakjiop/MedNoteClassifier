"""Central configuration for MedNoteClassifier.

Paths resolve relative to the MedNoteClassifier project root so the package
can be imported from the CLI, notebooks, or tests.
"""

from __future__ import annotations

from pathlib import Path

SEED: int = 42

# Splits are grouped by patient-note id (pn_num) and stratified by case_num.
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

# --- Baseline (span candidate classifier) ---
BASELINE_NGRAM_RANGE: tuple[int, int] = (3, 5)
BASELINE_MAX_FEATURES: int = 16384
BASELINE_C: float = 1.0
MAX_CANDIDATE_TOKENS: int = 12
NEGATIVE_PER_POSITIVE: int = 4

# --- Tiny transformer (trained in this repo on CPU) ---
TRANSFORMER_D_MODEL: int = 96
TRANSFORMER_NHEAD: int = 4
TRANSFORMER_LAYERS: int = 2
TRANSFORMER_FF: int = 192
TRANSFORMER_MAX_LEN: int = 160
TRANSFORMER_BATCH_SIZE: int = 32
TRANSFORMER_EPOCHS: int = 2
TRANSFORMER_LR: float = 3e-4
TRANSFORMER_WEIGHT_DECAY: float = 0.01
TRANSFORMER_DROPOUT: float = 0.1
TRANSFORMER_MIN_FREQ: int = 2
TRANSFORMER_MAX_VOCAB: int = 8000
TRANSFORMER_PATIENCE: int = 1

# Hugging Face DistilBERT settings (optional GPU / larger-machine run).
MODEL_NAME: str = "distilbert-base-uncased"
MAX_LENGTH: int = 256
BATCH_SIZE: int = 8
LEARNING_RATE: float = 2e-5
EPOCHS: int = 3
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.1

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_BASELINE: Path = PROJECT_ROOT / "models" / "baseline"
MODELS_TRANSFORMER: Path = PROJECT_ROOT / "models" / "transformer"
REPORTS: Path = PROJECT_ROOT / "reports"
FIGURES: Path = REPORTS / "figures"
METRICS: Path = REPORTS / "metrics"
ERROR_DIR: Path = REPORTS / "error_analysis"
WEB_EXPORT: Path = Path("/workspace/public/ml")

TRAIN_CSV: Path = DATA_RAW / "train.csv"
NOTES_CSV: Path = DATA_RAW / "patient_notes.csv"
FEATURES_CSV: Path = DATA_RAW / "features.csv"
TEST_CSV: Path = DATA_RAW / "test.csv"

BIO_LABELS: tuple[str, ...] = ("O", "B", "I")
LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL: dict[int, str] = {i: l for l, i in LABEL2ID.items()}

# Clinical paper palette used for figures (matches the product UI).
PALETTE: dict[str, str] = {
    "ink": "#1a211c",
    "teal": "#0f5c4d",
    "sage": "#8fa89c",
    "paper": "#f4f0e6",
    "muted": "#5c675f",
    "warn": "#8a6a32",
    "danger": "#8f3d3d",
}
