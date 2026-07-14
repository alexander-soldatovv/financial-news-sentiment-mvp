from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.model_utils import LABEL_ORDER, PROJECT_ROOT


DEFAULT_DATA_PATH = PROJECT_ROOT / "outputs" / "data" / "financial_phrasebank_75agree.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"


def load_dataset(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    expected_columns = {"sentence", "label"}
    missing = expected_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    data = data[["sentence", "label"]].dropna()
    data["sentence"] = data["sentence"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data = data[data["sentence"] != ""]
    data = data[data["label"].isin(LABEL_ORDER)]
    conflicting_labels = data.groupby("sentence")["label"].nunique()
    if (conflicting_labels > 1).any():
        raise ValueError("Dataset contains identical sentences with conflicting labels.")
    data = data.drop_duplicates(subset=["sentence"], keep="first").reset_index(drop=True)
    if data.empty:
        raise ValueError("Dataset has no valid rows after cleaning.")
    return data


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def collect_metrics(y_true: pd.Series, y_pred: list[str], label_counts: dict[str, int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
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
        "label_counts": label_counts,
    }
    for label in LABEL_ORDER:
        metrics["per_class"][label] = {
            "precision": precision_score(
                y_true, y_pred, labels=[label], average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_true, y_pred, labels=[label], average="macro", zero_division=0
            ),
        }
    return metrics


def train(data_path: Path, model_path: Path, test_size: float, random_state: int) -> dict[str, Any]:
    data = load_dataset(data_path)
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(train_df["sentence"], train_df["label"])
    predictions = pipeline.predict(test_df["sentence"])
    label_counts = data["label"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()
    metrics = collect_metrics(test_df["label"], predictions, label_counts)

    artifact = {
        "pipeline": pipeline,
        "model_type": "TF-IDF + Logistic Regression",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "labels": LABEL_ORDER,
        "metrics": metrics,
        "test_size": test_size,
        "random_state": random_state,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train financial news sentiment model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = train(
        data_path=args.data.resolve(),
        model_path=args.out.resolve(),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = artifact["metrics"]
    print(f"Saved model to {args.out.resolve()}")
    print(f"Model type: {artifact['model_type']}")
    print(f"Precision macro: {metrics['precision_macro']:.3f}")
    print(f"Recall macro: {metrics['recall_macro']:.3f}")


if __name__ == "__main__":
    main()
