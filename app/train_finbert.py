from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from app.model_utils import LABEL_ORDER, PROJECT_ROOT
from app.train_model import load_dataset


DEFAULT_BASE_MODEL = "yiyanghkust/finbert-pretrain"
DEFAULT_DATA_PATH = PROJECT_ROOT / "outputs" / "data" / "financial_phrasebank_75agree.csv"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "finbert-sentiment"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "finbert_comparison.json"
LABEL2ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


@dataclass
class TrainingConfig:
    base_model: str
    data_path: str
    baseline_path: str
    output_dir: str
    test_size: float
    random_state: int
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    weight_decay: float
    warmup_ratio: float


class FinancialSentenceDataset(Dataset):
    def __init__(
        self,
        sentences: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.sentences[index],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.labels[index]
        return encoded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "precision_macro": precision_score(
            y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0
        ),
        "per_class": {},
    }
    for label in LABEL_ORDER:
        result["per_class"][label] = {
            "precision": precision_score(
                y_true, y_pred, labels=[label], average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, labels=[label], average="macro", zero_division=0
            ),
        }
    return result


def class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (len(LABEL_ORDER) * counts[index]) for index in range(len(LABEL_ORDER))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_model(config: TrainingConfig) -> dict[str, Any]:
    seed_everything(config.random_state)
    data = load_dataset(Path(config.data_path))
    train_df, test_df = train_test_split(
        data,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(LABEL_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    device = choose_device()
    model.to(device)

    train_labels = [LABEL2ID[label] for label in train_df["label"].tolist()]
    test_labels = [LABEL2ID[label] for label in test_df["label"].tolist()]
    train_dataset = FinancialSentenceDataset(
        train_df["sentence"].tolist(), train_labels, tokenizer, config.max_length
    )
    test_dataset = FinancialSentenceDataset(
        test_df["sentence"].tolist(), test_labels, tokenizer, config.max_length
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_generator = torch.Generator().manual_seed(config.random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(16, config.batch_size),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = int(np.ceil(len(train_loader) / config.gradient_accumulation_steps))
    total_updates = updates_per_epoch * config.epochs
    warmup_steps = int(total_updates * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train_labels, device))

    print(f"Device: {device}", flush=True)
    print(f"Train rows: {len(train_dataset)} | Test rows: {len(test_dataset)}", flush=True)
    print(
        f"Batch size: {config.batch_size} | Gradient accumulation: "
        f"{config.gradient_accumulation_steps} | Updates: {total_updates}",
        flush=True,
    )

    training_started = time.perf_counter()
    epoch_losses: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        running_loss = 0.0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels) / config.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.item()) * config.gradient_accumulation_steps

            should_update = (
                step % config.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 50 == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                print(
                    f"Epoch {epoch + 1}/{config.epochs} | step {step}/{len(train_loader)} "
                    f"| loss {running_loss / step:.4f} | {elapsed:.1f}s",
                    flush=True,
                )

            # MPS uses unified memory and may retain large temporary buffers during
            # fine-tuning. Releasing them periodically prevents severe swap slowdown.
            del logits, loss, labels, inputs, batch
            if device.type == "mps" and step % 25 == 0:
                gc.collect()
                torch.mps.empty_cache()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
    training_seconds = time.perf_counter() - training_started

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    model.eval()
    predicted_ids: list[int] = []
    evaluation_started = time.perf_counter()
    with torch.inference_mode():
        for batch in test_loader:
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            predicted_ids.extend(logits.argmax(dim=-1).cpu().tolist())
    finbert_inference_seconds = time.perf_counter() - evaluation_started
    y_true = test_df["label"].tolist()
    finbert_predictions = [ID2LABEL[index] for index in predicted_ids]
    finbert_metrics = classification_metrics(y_true, finbert_predictions)

    baseline_artifact = joblib.load(config.baseline_path)
    baseline_pipeline = baseline_artifact["pipeline"]
    baseline_started = time.perf_counter()
    baseline_predictions = baseline_pipeline.predict(test_df["sentence"]).tolist()
    baseline_inference_seconds = time.perf_counter() - baseline_started
    baseline_metrics = classification_metrics(y_true, baseline_predictions)

    report = {
        "protocol": {
            "dataset_rows": len(data),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "test_size": config.test_size,
            "random_state": config.random_state,
            "same_test_split": True,
            "labels": LABEL_ORDER,
        },
        "training": {
            "config": asdict(config),
            "device": str(device),
            "epoch_losses": epoch_losses,
            "training_seconds": training_seconds,
        },
        "baseline": {
            "model": baseline_artifact.get("model_type", "TF-IDF + Logistic Regression"),
            "metrics": baseline_metrics,
            "inference_seconds": baseline_inference_seconds,
            "milliseconds_per_item": 1000 * baseline_inference_seconds / len(test_df),
        },
        "finbert": {
            "model": config.base_model,
            "metrics": finbert_metrics,
            "inference_seconds": finbert_inference_seconds,
            "milliseconds_per_item": 1000 * finbert_inference_seconds / len(test_df),
        },
    }
    return report


def write_report(report: dict[str, Any], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline = report["baseline"]
    finbert = report["finbert"]
    baseline_metrics = baseline["metrics"]
    finbert_metrics = finbert["metrics"]
    precision_gain = 100 * (
        finbert_metrics["precision_macro"] - baseline_metrics["precision_macro"]
    )
    recall_gain = 100 * (
        finbert_metrics["recall_macro"] - baseline_metrics["recall_macro"]
    )
    speed_ratio = finbert["milliseconds_per_item"] / baseline["milliseconds_per_item"]
    markdown_path = report_path.with_suffix(".md")
    markdown = f"""# FinBERT vs Baseline

Both models are evaluated on the same stratified 20% test split with random state 42.

| Model | Macro precision | Macro recall | Inference ms/item |
|---|---:|---:|---:|
| {baseline['model']} | {baseline['metrics']['precision_macro']:.4f} | {baseline['metrics']['recall_macro']:.4f} | {baseline['milliseconds_per_item']:.3f} |
| FinBERT | {finbert['metrics']['precision_macro']:.4f} | {finbert['metrics']['recall_macro']:.4f} | {finbert['milliseconds_per_item']:.3f} |

| Class | Baseline precision | FinBERT precision | Baseline recall | FinBERT recall |
|---|---:|---:|---:|---:|
| Negative | {baseline_metrics['per_class']['negative']['precision']:.4f} | {finbert_metrics['per_class']['negative']['precision']:.4f} | {baseline_metrics['per_class']['negative']['recall']:.4f} | {finbert_metrics['per_class']['negative']['recall']:.4f} |
| Neutral | {baseline_metrics['per_class']['neutral']['precision']:.4f} | {finbert_metrics['per_class']['neutral']['precision']:.4f} | {baseline_metrics['per_class']['neutral']['recall']:.4f} | {finbert_metrics['per_class']['neutral']['recall']:.4f} |
| Positive | {baseline_metrics['per_class']['positive']['precision']:.4f} | {finbert_metrics['per_class']['positive']['precision']:.4f} | {baseline_metrics['per_class']['positive']['recall']:.4f} | {finbert_metrics['per_class']['positive']['recall']:.4f} |

FinBERT improves macro precision by {precision_gain:.1f} percentage points and macro recall by {recall_gain:.1f} percentage points. It is about {speed_ratio:.0f}x slower per item on the training machine, but its {finbert['milliseconds_per_item']:.1f} ms latency remains practical for batch news screening.

FinBERT base checkpoint: `{finbert['model']}`.
"""
    markdown_path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT and compare it with baseline.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        base_model=args.base_model,
        data_path=str(args.data.resolve()),
        baseline_path=str(args.baseline.resolve()),
        output_dir=str(args.out.resolve()),
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )
    report = train_one_model(config)
    write_report(report, args.report.resolve())
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved FinBERT model to {args.out.resolve()}", flush=True)
    print(f"Saved comparison to {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
