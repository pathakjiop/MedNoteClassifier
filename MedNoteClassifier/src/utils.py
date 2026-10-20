"""Shared helpers: seeding, logging, JSON I/O, display names."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import SEED


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mednote")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_display_name(feature_text: str) -> str:
    """Turn NBME `feature_text` (hyphenated rubric) into a readable label."""
    text = feature_text.replace("-OR-", " / ").replace("-", " ")
    text = " ".join(text.split())
    return text[0].upper() + text[1:] if text else text


def normalize_for_match(text: str) -> str:
    """Light normalization that does not destroy clinical meaning."""
    return " ".join(text.lower().split())
