"""End-to-end reproducible pipeline: EDA → split → train → evaluate → export.

    python -m src.run_pipeline
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FIGURES,
    METRICS,
    ERROR_DIR,
    DATA_PROCESSED,
    MODELS_BASELINE,
    PALETTE,
    SEED,
    WEB_EXPORT,
    PROJECT_ROOT,
)
from src.data_loader import load_raw_frames, parse_annotation_list, parse_span_list
from src.dataset import build_processed_frame, by_split
from src.error_analysis import analyze_errors
from src.evaluate import format_metrics_table, micro_f1, per_feature_f1, span_char_set
from src.train_baseline import HybridBaseline, LexiconModel, empty_predict, evaluate_predictor
from src.utils import (
    ensure_dir,
    feature_display_name,
    set_seed,
    setup_logging,
    write_json,
)

logger = logging.getLogger("mednote.pipeline")


def _style_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, color=PALETTE["ink"], loc="left", fontsize=12, pad=10)
    ax.set_facecolor(PALETTE["paper"])
    ax.tick_params(colors=PALETTE["muted"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["sage"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.xaxis.label.set_color(PALETTE["muted"])


def run_eda(train: pd.DataFrame, notes: pd.DataFrame, features: pd.DataFrame, merged: pd.DataFrame) -> dict:
    ensure_dir(FIGURES)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["sage"],
            "text.color": PALETTE["ink"],
        }
    )

    merged = merged.copy()
    merged["n_spans"] = merged["location"].map(lambda x: len(parse_span_list(x)))
    merged["empty"] = merged["n_spans"] == 0
    span_lens: list[int] = []
    phrases: list[str] = []
    invalid = 0
    overlap_notes = 0
    for _, row in merged.iterrows():
        text = str(row.get("pn_history") or "")
        spans = parse_span_list(row.get("location", "[]"))
        anns = parse_annotation_list(row.get("annotation", "[]"))
        phrases.extend(anns)
        for start, end in spans:
            span_lens.append(end - start)
            if start < 0 or end > len(text) or end <= start:
                invalid += 1
        ordered = sorted(spans)
        for i in range(1, len(ordered)):
            if ordered[i][0] < ordered[i - 1][1]:
                overlap_notes += 1
                break

    label_counts = (
        merged.groupby(["feature_num", "feature_text", "case_num"])
        .agg(n=("id", "count"), n_nonempty=("empty", lambda s: int((~s).sum())))
        .reset_index()
        .sort_values("n_nonempty", ascending=False)
    )

    # 1. nonempty annotations per feature (top 20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top = label_counts.head(20)
    ax.barh(
        [feature_display_name(t)[:42] for t in top["feature_text"][::-1]],
        top["n_nonempty"][::-1],
        color=PALETTE["teal"],
    )
    _style_ax(ax, "Top 20 clinical features by annotated mentions")
    ax.set_xlabel("Non-empty annotations")
    fig.tight_layout()
    fig.savefig(FIGURES / "label_distribution.png", dpi=140)
    plt.close(fig)

    # 2. annotation length
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(span_lens, bins=40, color=PALETTE["teal"], edgecolor=PALETTE["paper"])
    _style_ax(ax, "Annotation span length (characters)")
    ax.set_xlabel("Span length")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES / "annotation_length.png", dpi=140)
    plt.close(fig)

    # 3. annotations per note
    per_note = merged.groupby("pn_num")["n_spans"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(per_note, bins=20, color=PALETTE["sage"], edgecolor=PALETTE["paper"])
    _style_ax(ax, "Annotated spans per patient note")
    ax.set_xlabel("Spans in note")
    ax.set_ylabel("Notes")
    fig.tight_layout()
    fig.savefig(FIGURES / "annotations_per_note.png", dpi=140)
    plt.close(fig)

    # 4. features per case
    fig, ax = plt.subplots(figsize=(8, 4.5))
    per_case = features.groupby("case_num").size()
    ax.bar(per_case.index.astype(str), per_case.values, color=PALETTE["teal"])
    _style_ax(ax, "Rubric features per USMLE case")
    ax.set_xlabel("Case")
    ax.set_ylabel("Features")
    fig.tight_layout()
    fig.savefig(FIGURES / "features_per_case.png", dpi=140)
    plt.close(fig)

    # 5. dataset size summary
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["Patient notes", "Annotated notes", "Train rows", "Features", "Cases"]
    values = [
        int(notes["pn_num"].nunique()),
        int(train["pn_num"].nunique()),
        int(len(train)),
        int(len(features)),
        int(features["case_num"].nunique()),
    ]
    ax.barh(labels[::-1], values[::-1], color=[PALETTE["teal"], PALETTE["sage"], PALETTE["ink"], PALETTE["teal"], PALETTE["sage"]])
    _style_ax(ax, "Dataset size")
    fig.tight_layout()
    fig.savefig(FIGURES / "dataset_size.png", dpi=140)
    plt.close(fig)

    # confusion-style empty vs nonempty per case
    fig, ax = plt.subplots(figsize=(8, 4.5))
    empty_by_case = merged.groupby("case_num")["empty"].mean()
    ax.bar(empty_by_case.index.astype(str), empty_by_case.values, color=PALETTE["warn"])
    _style_ax(ax, "Fraction of empty annotations by case")
    ax.set_xlabel("Case")
    ax.set_ylabel("Empty rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "empty_rate_by_case.png", dpi=140)
    plt.close(fig)

    phrase_counts = Counter(phrases)
    summary = {
        "n_train_rows": int(len(train)),
        "n_unique_notes_all": int(notes["pn_num"].nunique()),
        "n_annotated_notes": int(train["pn_num"].nunique()),
        "n_features": int(len(features)),
        "n_cases": int(features["case_num"].nunique()),
        "n_empty_annotations": int(merged["empty"].sum()),
        "n_nonempty_annotations": int((~merged["empty"]).sum()),
        "missing_values_train": {k: int(v) for k, v in train.isna().sum().items()},
        "duplicate_train_ids": int(train["id"].duplicated().sum()),
        "note_char_len_mean": float(notes["pn_history"].str.len().mean()),
        "note_char_len_max": int(notes["pn_history"].str.len().max()),
        "span_len_mean": float(np.mean(span_lens)) if span_lens else 0,
        "span_len_median": float(np.median(span_lens)) if span_lens else 0,
        "invalid_spans": invalid,
        "notes_with_overlapping_spans": overlap_notes,
        "notes_with_zero_spans": int((per_note == 0).sum()),
        "mean_spans_per_annotated_note": float(per_note.mean()),
        "top_phrases": phrase_counts.most_common(30),
        "features_per_case": {int(k): int(v) for k, v in per_case.items()},
        "columns_train": list(train.columns),
        "columns_notes": list(notes.columns),
        "columns_features": list(features.columns),
    }
    write_json(METRICS / "eda_summary.json", summary)
    logger.info("EDA complete: %s annotated notes, %s features", summary["n_annotated_notes"], summary["n_features"])
    return summary


def _metrics_payload(name: str, split: str, stats: dict, extra: dict | None = None) -> dict:
    payload = {"model": name, "split": split, **stats}
    if extra:
        payload.update(extra)
    return payload


def export_web_artifacts(
    features: pd.DataFrame,
    notes: pd.DataFrame,
    merged: pd.DataFrame,
    rows,
    lexicon: LexiconModel,
    comparison: list[dict],
    eda: dict,
    errors: dict,
    transformer_meta: dict,
) -> None:
    ensure_dir(WEB_EXPORT)
    feat_payload = []
    for r in features.itertuples():
        fid = int(r.feature_num)
        phrases = [
            {"text": p, "norm": n, "count": c} for p, n, c in lexicon.phrases.get(fid, [])
        ]
        feat_payload.append(
            {
                "feature_num": fid,
                "case_num": int(r.case_num),
                "feature_text": r.feature_text,
                "display": feature_display_name(r.feature_text),
                "phrases": phrases[:80],
            }
        )

    case_profiles = []
    for case_num, grp in features.groupby("case_num"):
        names = [feature_display_name(t) for t in grp["feature_text"]]
        age = next((n for n in names if "year" in n.lower()), None)
        sex = next((n for n in names if n.lower() in {"male", "female"}), None)
        case_profiles.append(
            {
                "case_num": int(case_num),
                "n_features": int(len(grp)),
                "age_cue": age,
                "sex_cue": sex,
                "features": names,
                "title": _case_title(int(case_num), names),
            }
        )

    examples = []
    test_rows = [r for r in rows if r.split == "test"]
    seen_cases: set[int] = set()
    for row in test_rows:
        if row.case_num in seen_cases:
            continue
        if not row.pn_history:
            continue
        # collect all gold spans for this note
        note_rows = [r for r in test_rows if r.pn_num == row.pn_num]
        entities = []
        for r in note_rows:
            for s in r.spans:
                entities.append(
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": row.pn_history[s.start : s.end],
                        "type": feature_display_name(r.feature_text),
                        "feature_num": r.feature_num,
                    }
                )
        examples.append(
            {
                "pn_num": row.pn_num,
                "case_num": row.case_num,
                "text": row.pn_history,
                "entities": sorted(entities, key=lambda e: e["start"]),
            }
        )
        seen_cases.add(row.case_num)
        if len(examples) >= 10:
            break

    write_json(WEB_EXPORT / "features.json", feat_payload)
    write_json(WEB_EXPORT / "case_profiles.json", case_profiles)
    write_json(WEB_EXPORT / "examples.json", examples)
    write_json(WEB_EXPORT / "metrics.json", {"comparison": comparison, "transformer": transformer_meta})
    write_json(WEB_EXPORT / "eda.json", {k: v for k, v in eda.items() if k != "top_phrases"} | {"top_phrases": eda["top_phrases"][:20]})
    write_json(
        WEB_EXPORT / "errors.json",
        {
            "false_positives": errors.get("false_positives", [])[:12],
            "false_negatives": errors.get("false_negatives", [])[:12],
            "buckets": errors.get("buckets", {}),
            "hardest_features": _named_hardest(errors.get("hardest_features", []), features),
        },
    )
    # copy figures for the web
    import shutil

    web_fig = WEB_EXPORT / "figures"
    ensure_dir(web_fig)
    for png in FIGURES.glob("*.png"):
        shutil.copy2(png, web_fig / png.name)


def _named_hardest(hardest: list[dict], features: pd.DataFrame) -> list[dict]:
    names = {int(r.feature_num): feature_display_name(r.feature_text) for r in features.itertuples()}
    out = []
    for h in hardest[:10]:
        item = dict(h)
        item["display"] = names.get(int(h["feature_num"]), str(h["feature_num"]))
        out.append(item)
    return out


def _case_title(case_num: int, names: list[str]) -> str:
    age = next((n for n in names if "year" in n.lower()), "adult")
    sex = next((n.lower() for n in names if n.lower() in {"male", "female"}), "patient")
    # pick a salient symptom that is not demo
    skip = {"male", "female"} | {n.lower() for n in names if "year" in n.lower()}
    cue = next((n for n in names if n.lower() not in skip), "clinical encounter")
    return f"Case {case_num} · {age} {sex} · {cue}"


def main() -> None:
    setup_logging()
    set_seed(SEED)
    t_all = time.time()
    for path in (FIGURES, METRICS, ERROR_DIR, DATA_PROCESSED, MODELS_BASELINE, WEB_EXPORT):
        ensure_dir(path)

    train, notes, features, kaggle_test = load_raw_frames()
    merged, rows = build_processed_frame(train, notes, features, seed=SEED)
    merged.to_parquet(DATA_PROCESSED / "merged.parquet", index=False)
    split_map = merged[["pn_num", "case_num", "split"]].drop_duplicates()
    split_map.to_csv(DATA_PROCESSED / "splits.csv", index=False)

    eda = run_eda(train, notes, features, merged)

    train_rows = by_split(rows, "train")
    val_rows = by_split(rows, "val")
    test_rows = by_split(rows, "test")
    logger.info("Rows train/val/test = %s/%s/%s", len(train_rows), len(val_rows), len(test_rows))

    # --- Baselines ---
    empty_val = evaluate_predictor(val_rows, empty_predict)
    empty_test = evaluate_predictor(test_rows, empty_predict)

    lexicon = LexiconModel().fit(train_rows)
    lex_val = evaluate_predictor(val_rows, lexicon.predict_row)
    lex_test = evaluate_predictor(test_rows, lexicon.predict_row)
    logger.info("Lexicon val F1=%.4f test F1=%.4f", lex_val["f1"], lex_test["f1"])

    hybrid = HybridBaseline()
    hybrid.lexicon = lexicon
    logger.info("Fitting logistic regression span classifier…")
    hybrid.logreg.fit(train_rows)
    hybrid.save()
    hyb_val = evaluate_predictor(val_rows, hybrid.predict_row)
    hyb_test = evaluate_predictor(test_rows, hybrid.predict_row)
    logger.info("Hybrid val F1=%.4f test F1=%.4f", hyb_val["f1"], hyb_test["f1"])

    # --- Transformer ---
    from src.train_transformer import train_transformer

    tagger, tr_meta = train_transformer(train_rows, val_rows)
    tr_val = evaluate_predictor(val_rows, tagger.predict_row)
    tr_test = evaluate_predictor(test_rows, tagger.predict_row)
    logger.info("Transformer val F1=%.4f test F1=%.4f", tr_val["f1"], tr_test["f1"])

    comparison = [
        _metrics_payload("empty", "test", empty_test),
        _metrics_payload("lexicon", "val", lex_val),
        _metrics_payload("lexicon", "test", lex_test),
        _metrics_payload("hybrid_logreg", "val", hyb_val),
        _metrics_payload("hybrid_logreg", "test", hyb_test),
        _metrics_payload("tiny_transformer", "val", tr_val, {"train_seconds": tr_meta.get("seconds")}),
        _metrics_payload("tiny_transformer", "test", tr_test, {"architecture": tr_meta.get("architecture")}),
    ]
    write_json(METRICS / "baseline_metrics.json", {
        "empty_test": empty_test,
        "lexicon_val": lex_val,
        "lexicon_test": lex_test,
        "hybrid_val": hyb_val,
        "hybrid_test": hyb_test,
        "seed": SEED,
        "split": "pn_num grouped 80/10/10 stratified by case_num",
    })
    write_json(
        METRICS / "transformer_metrics.json",
        {"val": tr_val, "test": tr_test, "train": tr_meta, "seed": SEED},
    )
    write_json(METRICS / "comparison.json", comparison)

    table = format_metrics_table(
        [
            {"model": "Empty (majority)", **empty_test},
            {"model": "Lexicon", **lex_test},
            {"model": "Hybrid (lexicon + logreg)", **hyb_test},
            {"model": "Tiny transformer", **tr_test},
        ]
    )
    (METRICS / "comparison.md").write_text(table + "\n", encoding="utf-8")
    logger.info("\n%s", table)

    # Per-feature for best of lexicon vs transformer on test
    pairs = [(span_char_set(lexicon.predict_row(r)), span_char_set(r.spans)) for r in test_rows]
    per_feat = per_feature_f1([r.feature_num for r in test_rows], pairs)
    named = []
    fmap = {int(r.feature_num): r.feature_text for r in features.itertuples()}
    for fid, stats in per_feat.items():
        named.append({"feature_num": fid, "feature_text": fmap.get(fid, ""), "display": feature_display_name(fmap.get(fid, "")), **stats})
    named.sort(key=lambda x: x["f1"])
    write_json(METRICS / "per_feature_lexicon_test.json", named)

    errors = analyze_errors(test_rows, lexicon.predict_row)
    write_json(ERROR_DIR / "lexicon_test.json", errors)

    # Confusion-style figure: F1 by case
    fig, ax = plt.subplots(figsize=(8, 4.5))
    case_f1 = []
    for case in range(10):
        sub = [r for r in test_rows if r.case_num == case]
        m = evaluate_predictor(sub, lexicon.predict_row)
        case_f1.append(m["f1"])
    ax.bar([str(i) for i in range(10)], case_f1, color=PALETTE["teal"])
    _style_ax(ax, "Lexicon character-F1 by case (test)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Case")
    fig.tight_layout()
    fig.savefig(FIGURES / "f1_by_case.png", dpi=140)
    plt.close(fig)

    # "confusion" analogue: empty-gold predicted nonempty vs not
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels_cm = np.zeros((2, 2), dtype=int)
    for r in test_rows:
        pred = bool(lexicon.predict_row(r))
        gold = bool(r.spans)
        labels_cm[int(gold)][int(pred)] += 1
    im = ax.imshow(labels_cm, cmap="Greens")
    ax.set_xticks([0, 1], ["Pred empty", "Pred span"])
    ax.set_yticks([0, 1], ["Gold empty", "Gold span"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(labels_cm[i, j]), ha="center", va="center", color=PALETTE["ink"], fontsize=14)
    _style_ax(ax, "Presence detection (lexicon, test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    export_web_artifacts(features, notes, merged, rows, lexicon, comparison, eda, errors, tr_meta)

    elapsed = time.time() - t_all
    write_json(
        METRICS / "run.json",
        {
            "seconds": elapsed,
            "seed": SEED,
            "project_root": str(PROJECT_ROOT),
            "n_train_rows": len(train_rows),
            "n_val_rows": len(val_rows),
            "n_test_rows": len(test_rows),
        },
    )
    logger.info("Pipeline finished in %.1fs", elapsed)


if __name__ == "__main__":
    main()
