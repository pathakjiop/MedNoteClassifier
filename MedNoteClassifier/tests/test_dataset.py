import pandas as pd

from src.data_loader import load_raw_frames
from src.dataset import assign_note_splits, build_processed_frame
from src.evaluate import micro_f1


def test_load_raw_columns() -> None:
    train, notes, features, test = load_raw_frames()
    assert list(train.columns) == ["id", "case_num", "pn_num", "feature_num", "annotation", "location"]
    assert list(notes.columns) == ["pn_num", "case_num", "pn_history"]
    assert list(features.columns) == ["feature_num", "case_num", "feature_text"]
    assert len(features) == 143
    assert train["pn_num"].nunique() == 1000


def test_note_level_split_no_leakage() -> None:
    train, notes, features, _ = load_raw_frames()
    merged, rows = build_processed_frame(train, notes, features, seed=42)
    mapping = merged[["pn_num", "split"]].drop_duplicates()
    # Each note in exactly one split
    assert mapping["pn_num"].duplicated().sum() == 0
    counts = mapping["split"].value_counts().to_dict()
    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100
    # Stratification: every case in every split
    for split in ("train", "val", "test"):
        cases = set(merged.loc[merged["split"] == split, "case_num"].unique())
        assert cases == set(range(10))


def test_assign_note_splits_reproducible() -> None:
    train = pd.DataFrame(
        {
            "pn_num": list(range(100)),
            "case_num": [i // 10 for i in range(100)],
        }
    )
    a = assign_note_splits(train, seed=42)
    b = assign_note_splits(train, seed=42)
    pd.testing.assert_frame_equal(
        a.sort_values("pn_num").reset_index(drop=True),
        b.sort_values("pn_num").reset_index(drop=True),
    )


def test_micro_f1_empty_empty() -> None:
    stats = micro_f1([(set(), set())])
    assert stats["f1"] == 1.0


def test_micro_f1_partial_overlap() -> None:
    pred = {0, 1, 2}
    gold = {1, 2, 3}
    stats = micro_f1([(pred, gold)])
    assert 0.6 < stats["f1"] < 0.7
