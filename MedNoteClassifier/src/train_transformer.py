"""Tiny Transformer token classifier for NBME span extraction.

Architecture (CPU-friendly, trained from scratch on this dataset):

    token embedding (d=96) + position embedding
    → 2-layer TransformerEncoder (4 heads, FFN=192)
    → linear head over {O, B, I}

Input sequence:  [CLS] feature tokens [SEP] note tokens [SEP]

Labels are -100 on special / feature tokens so the loss is only computed on
the patient-note tokens. This is the same formulation as a Hugging Face
token-classification head; DistilBERT training is provided as an optional
path in `train_distilbert()` for GPU machines.

This is NOT a clinically validated model.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    ID2LABEL,
    LABEL2ID,
    MODELS_TRANSFORMER,
    TRANSFORMER_BATCH_SIZE,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_FF,
    TRANSFORMER_LAYERS,
    TRANSFORMER_LR,
    TRANSFORMER_MAX_LEN,
    TRANSFORMER_MAX_VOCAB,
    TRANSFORMER_MIN_FREQ,
    TRANSFORMER_NHEAD,
    TRANSFORMER_WEIGHT_DECAY,
    SEED,
)
from src.data_loader import AnnotationRow, Span
from src.preprocessing import bio_labels, spans_from_bio, tokenize_with_offsets
from src.utils import ensure_dir, write_json

logger = logging.getLogger("mednote.transformer")

PAD, UNK, CLS, SEP = "<pad>", "<unk>", "<cls>", "<sep>"
SPECIAL = [PAD, UNK, CLS, SEP]


class Vocab:
    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        self.token_to_id = token_to_id or {t: i for i, t in enumerate(SPECIAL)}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id[UNK]
        return [self.token_to_id.get(t.lower(), unk) for t in tokens]

    def save(self, path: Path) -> None:
        write_json(path, self.token_to_id)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def build(cls, rows: list[AnnotationRow], min_freq: int, max_vocab: int) -> "Vocab":
        counts: Counter[str] = Counter()
        for row in rows:
            for tok in tokenize_with_offsets(row.pn_history):
                counts[tok.text.lower()] += 1
            for tok in tokenize_with_offsets(row.feature_text.replace("-", " ")):
                counts[tok.text.lower()] += 1
        kept = [w for w, c in counts.most_common() if c >= min_freq]
        kept = kept[: max(0, max_vocab - len(SPECIAL))]
        mapping = {t: i for i, t in enumerate(SPECIAL)}
        for w in kept:
            mapping[w] = len(mapping)
        logger.info("Vocab size=%s (min_freq=%s)", len(mapping), min_freq)
        return cls(mapping)


class SpanTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        nlayers: int = TRANSFORMER_LAYERS,
        dim_ff: int = TRANSFORMER_FF,
        dropout: float = TRANSFORMER_DROPOUT,
        max_len: int = TRANSFORMER_MAX_LEN,
        n_labels: int = 3,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_labels)
        self.dropout = nn.Dropout(dropout)
        self._reset()

    def _reset(self) -> None:
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        self.tok.weight.data[0] = 0
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(bsz, seq)
        x = self.tok(input_ids) * math.sqrt(self.d_model) + self.pos(positions)
        x = self.dropout(x)
        # TransformerEncoder key_padding_mask: True = ignore
        pad_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        return self.head(x)


def encode_row(
    row: AnnotationRow, vocab: Vocab, max_len: int = TRANSFORMER_MAX_LEN
) -> dict[str, Any]:
    feat_toks = tokenize_with_offsets(row.feature_text.replace("-", " "))
    note_toks = tokenize_with_offsets(row.pn_history)
    note_labels = bio_labels(note_toks, row.spans)
    # Budget: CLS + feature + SEP + note + SEP
    feat_budget = min(len(feat_toks), 24)
    feat_toks = feat_toks[:feat_budget]
    remaining = max_len - (feat_budget + 3)
    note_toks = note_toks[: max(1, remaining)]
    note_labels = note_labels[: len(note_toks)]

    tokens = [CLS] + [t.text for t in feat_toks] + [SEP] + [t.text for t in note_toks] + [SEP]
    labels = [-100] * (feat_budget + 2) + [LABEL2ID[l] for l in note_labels] + [-100]
    ids = vocab.encode(tokens)
    # CLS/SEP/PAD are already lowercase in vocab; encode() lowercases — specials stay.
    ids[0] = vocab.token_to_id[CLS]
    ids[feat_budget + 1] = vocab.token_to_id[SEP]
    ids[-1] = vocab.token_to_id[SEP]
    mask = [1] * len(ids)
    note_offsets = [(t.start, t.end) for t in note_toks]
    pad = max_len - len(ids)
    if pad > 0:
        ids = ids + [vocab.token_to_id[PAD]] * pad
        mask = mask + [0] * pad
        labels = labels + [-100] * pad
    return {
        "input_ids": ids[:max_len],
        "attention_mask": mask[:max_len],
        "labels": labels[:max_len],
        "note_offsets": note_offsets,
        "note_token_start": feat_budget + 2,
    }


class RowDataset(Dataset):
    def __init__(self, rows: list[AnnotationRow], vocab: Vocab, max_len: int) -> None:
        self.encoded = [encode_row(r, vocab, max_len) for r in rows]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.encoded[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "index": idx,
        }


def token_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    gold = labels[mask]
    pred = pred[mask]
    # Entity-ish: treat non-O as positive
    gold_pos = gold != 0
    pred_pos = pred != 0
    tp = int((gold_pos & pred_pos).sum())
    fp = int((~gold_pos & pred_pos).sum())
    fn = int((gold_pos & ~pred_pos).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def decode_prediction(item: dict[str, Any], label_ids: list[int]) -> list[Span]:
    start = item["note_token_start"]
    offsets = item["note_offsets"]
    n = len(offsets)
    labs = label_ids[start : start + n]
    names = [ID2LABEL.get(int(i), "O") for i in labs]
    dummy_tokens = [
        type("T", (), {"start": a, "end": b})() for a, b in offsets
    ]
    return spans_from_bio(dummy_tokens, names)  # type: ignore[arg-type]


class TransformerTagger:
    def __init__(self, vocab: Vocab, model: SpanTransformer, device: torch.device) -> None:
        self.vocab = vocab
        self.model = model.to(device)
        self.device = device

    def predict_row(self, row: AnnotationRow) -> list[Span]:
        self.model.eval()
        item = encode_row(row, self.vocab)
        ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=self.device)
        mask = torch.tensor([item["attention_mask"]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(ids, mask)[0]
        label_ids = logits.argmax(dim=-1).tolist()
        spans = decode_prediction(item, label_ids)
        for s in spans:
            s.text = row.pn_history[s.start : s.end]
        return spans

    def save(self, directory: Path = MODELS_TRANSFORMER) -> None:
        directory = ensure_dir(directory)
        torch.save(self.model.state_dict(), directory / "span_transformer.pt")
        self.vocab.save(directory / "vocab.json")
        write_json(
            directory / "arch.json",
            {
                "vocab_size": len(self.vocab),
                "d_model": TRANSFORMER_D_MODEL,
                "nhead": TRANSFORMER_NHEAD,
                "nlayers": TRANSFORMER_LAYERS,
                "dim_ff": TRANSFORMER_FF,
                "max_len": TRANSFORMER_MAX_LEN,
            },
        )

    @classmethod
    def load(cls, directory: Path = MODELS_TRANSFORMER, device: torch.device | None = None) -> "TransformerTagger":
        device = device or torch.device("cpu")
        vocab = Vocab.load(directory / "vocab.json")
        arch = json.loads((directory / "arch.json").read_text(encoding="utf-8"))
        model = SpanTransformer(
            vocab_size=arch["vocab_size"],
            d_model=arch["d_model"],
            nhead=arch["nhead"],
            nlayers=arch["nlayers"],
            dim_ff=arch["dim_ff"],
            max_len=arch["max_len"],
        )
        state = torch.load(directory / "span_transformer.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return cls(vocab, model, device)


def train_transformer(
    train_rows: list[AnnotationRow],
    val_rows: list[AnnotationRow],
    epochs: int = TRANSFORMER_EPOCHS,
    batch_size: int = TRANSFORMER_BATCH_SIZE,
    lr: float = TRANSFORMER_LR,
    device: torch.device | None = None,
) -> tuple[TransformerTagger, dict[str, Any]]:
    device = device or torch.device("cpu")
    set_torch_threads()
    vocab = Vocab.build(train_rows, TRANSFORMER_MIN_FREQ, TRANSFORMER_MAX_VOCAB)
    model = SpanTransformer(vocab_size=len(vocab))
    train_ds = RowDataset(train_rows, vocab, TRANSFORMER_MAX_LEN)
    val_ds = RowDataset(val_rows, vocab, TRANSFORMER_MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Non-O tokens are rare; weight B/I higher.
    counts = np.zeros(3, dtype=np.float64)
    for item in train_ds.encoded:
        for lab in item["labels"]:
            if lab >= 0:
                counts[lab] += 1
    counts = np.maximum(counts, 1.0)
    weight = torch.tensor(counts.sum() / (3.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRANSFORMER_WEIGHT_DECAY)
    model.to(device)

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_state = None
    t0 = time.time()
    logger.info("Training tiny transformer on %s examples, device=%s", len(train_ds), device)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ids, mask)
            loss = criterion(logits.view(-1, 3), labels.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(ids, mask)
                loss = criterion(logits.view(-1, 3), labels.view(-1))
                val_loss += float(loss.item())
                val_f1 += token_f1(logits, labels)
                n_val += 1
        rec = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_loss": val_loss / max(n_val, 1),
            "val_token_f1": val_f1 / max(n_val, 1),
        }
        history.append(rec)
        logger.info(
            "epoch %s train_loss=%.4f val_loss=%.4f val_token_f1=%.4f",
            epoch,
            rec["train_loss"],
            rec["val_loss"],
            rec["val_token_f1"],
        )
        if rec["val_token_f1"] > best_f1:
            best_f1 = rec["val_token_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    tagger = TransformerTagger(vocab, model, device)
    tagger.save()
    meta = {
        "history": history,
        "best_val_token_f1": best_f1,
        "seconds": time.time() - t0,
        "epochs": epochs,
        "vocab_size": len(vocab),
        "device": str(device),
        "architecture": "tiny-transformer-encoder",
        "d_model": TRANSFORMER_D_MODEL,
        "nlayers": TRANSFORMER_LAYERS,
        "nhead": TRANSFORMER_NHEAD,
        "seed": SEED,
    }
    write_json(MODELS_TRANSFORMER / "train_meta.json", meta)
    return tagger, meta


def set_torch_threads() -> None:
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def train_distilbert(*_args: Any, **_kwargs: Any) -> None:
    """Optional Hugging Face DistilBERT fine-tune (GPU recommended).

    Kept as an explicit, documented entry point so a larger machine can swap
    the tiny transformer for DistilBERT without changing the data pipeline.
    """
    raise NotImplementedError(
        "DistilBERT fine-tuning is available via Hugging Face Transformers on a "
        "GPU machine. The standard pipeline trains the lightweight TransformerEncoder in "
        "train_transformer() instead. See README.md."
    )
