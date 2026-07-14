from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.joblib"
DEFAULT_FINBERT_PATH = PROJECT_ROOT / "models" / "finbert-sentiment"
DEFAULT_FINBERT_REPORT_PATH = PROJECT_ROOT / "outputs" / "finbert_comparison.json"
LABEL_ORDER = ["negative", "neutral", "positive"]
MIN_SENTIMENT_CONFIDENCE = float(os.getenv("MIN_SENTIMENT_CONFIDENCE", "0.55"))
MIN_SENTIMENT_MARGIN = float(os.getenv("MIN_SENTIMENT_MARGIN", "0.10"))
EVENT_TYPE_WEIGHTS = {
    "earnings": 0.9,
    "guidance": 0.95,
    "merger_acquisition": 1.0,
    "legal_regulatory": 0.9,
    "management": 0.7,
    "macro": 0.65,
    "product_partnership": 0.55,
    "other": 0.35,
}


def get_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser().resolve()


def get_finbert_model_path() -> Path:
    return Path(os.getenv("FINBERT_MODEL_PATH", DEFAULT_FINBERT_PATH)).expanduser().resolve()


@lru_cache(maxsize=1)
def load_model_artifact(model_path: str | None = None) -> dict[str, Any]:
    path = Path(model_path).expanduser().resolve() if model_path else get_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m app.train_model` first."
        )
    artifact = joblib.load(path)
    if "pipeline" not in artifact:
        raise ValueError(f"Invalid model artifact at {path}: missing `pipeline`.")
    return artifact


class BaselineSentimentEngine:
    backend = "baseline"

    def __init__(self, artifact: dict[str, Any]) -> None:
        self.artifact = artifact
        self.pipeline = artifact["pipeline"]

    def predict_probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        classes = list(self.pipeline.classes_)
        rows = self.pipeline.predict_proba(texts)
        return [
            {
                label: float(row[classes.index(label)]) if label in classes else 0.0
                for label in LABEL_ORDER
            }
            for row in rows
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_backend": self.backend,
            "model_type": self.artifact.get("model_type", "TF-IDF + Logistic Regression"),
            "trained_at": self.artifact.get("trained_at"),
            "dataset": self.artifact.get("dataset"),
            "labels": self.artifact.get("labels", LABEL_ORDER),
            "device": "cpu",
            "metrics": self.artifact.get("metrics", {}),
        }


class FinBertSentimentEngine:
    backend = "finbert"

    def __init__(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT dependencies are unavailable. Install `torch` and `transformers`."
            ) from exc

        self.torch = torch
        self.model_path = model_path
        self.device = self._select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self._inference_lock = threading.Lock()

    def _select_device(self) -> Any:
        requested = os.getenv("MODEL_DEVICE", "auto").strip().lower()
        if requested == "auto":
            if self.torch.cuda.is_available():
                return self.torch.device("cuda")
            if self.torch.backends.mps.is_available():
                return self.torch.device("mps")
            return self.torch.device("cpu")
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda was requested, but CUDA is unavailable.")
        if requested == "mps" and not self.torch.backends.mps.is_available():
            raise RuntimeError("MODEL_DEVICE=mps was requested, but Apple MPS is unavailable.")
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError("MODEL_DEVICE must be one of: auto, cpu, cuda, mps.")
        return self.torch.device(requested)

    def predict_probabilities(self, texts: list[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        with self._inference_lock, self.torch.inference_mode():
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu()
                for row in probabilities.tolist():
                    results.append(
                        {
                            self.model.config.id2label[index].lower(): float(value)
                            for index, value in enumerate(row)
                        }
                    )
        return [
            {label: row.get(label, 0.0) for label in LABEL_ORDER}
            for row in results
        ]

    def metadata(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        if DEFAULT_FINBERT_REPORT_PATH.exists():
            try:
                report = json.loads(DEFAULT_FINBERT_REPORT_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = {}
        training_config = report.get("training", {}).get("config", {})
        trained_at = datetime.fromtimestamp(
            self.model_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
        return {
            "model_backend": self.backend,
            "model_type": "Fine-tuned FinBERT",
            "base_model": training_config.get("base_model", "yiyanghkust/finbert-pretrain"),
            "trained_at": trained_at,
            "dataset": training_config.get("data_path"),
            "labels": LABEL_ORDER,
            "device": str(self.device),
            "metrics": report.get("finbert", {}).get("metrics", {}),
        }


def resolve_model_backend(requested: str | None = None) -> str:
    backend = (requested or os.getenv("MODEL_BACKEND", "auto")).strip().lower()
    if backend == "auto":
        return "finbert" if (get_finbert_model_path() / "config.json").exists() else "baseline"
    if backend not in {"baseline", "finbert"}:
        raise ValueError("MODEL_BACKEND must be one of: auto, baseline, finbert.")
    return backend


@lru_cache(maxsize=2)
def _load_sentiment_engine(backend: str) -> BaselineSentimentEngine | FinBertSentimentEngine:
    if backend == "baseline":
        return BaselineSentimentEngine(load_model_artifact())
    model_path = get_finbert_model_path()
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"FinBERT model not found at {model_path}. Run `python -m app.train_finbert` first."
        )
    return FinBertSentimentEngine(model_path)


def load_sentiment_engine(
    backend: str | None = None,
) -> BaselineSentimentEngine | FinBertSentimentEngine:
    return _load_sentiment_engine(resolve_model_backend(backend))


def clamp_company_relevance(company_relevance: int | float) -> float:
    value = float(company_relevance)
    return max(1.0, min(5.0, value))


def event_weight(event_type: str) -> float:
    normalized = normalize_event_type(event_type)
    return EVENT_TYPE_WEIGHTS.get(normalized, EVENT_TYPE_WEIGHTS["other"])


def normalize_event_type(event_type: str) -> str:
    normalized = str(event_type or "other").strip().lower()
    return normalized if normalized in EVENT_TYPE_WEIGHTS else "other"


def interpret_sentiment(probabilities: dict[str, float]) -> dict[str, Any]:
    ranked = sorted(
        ((label, float(probabilities[label])) for label in LABEL_ORDER),
        key=lambda item: item[1],
        reverse=True,
    )
    model_label, confidence = ranked[0]
    confidence_margin = confidence - ranked[1][1]
    uncertainty_reasons: list[str] = []
    if confidence < MIN_SENTIMENT_CONFIDENCE:
        uncertainty_reasons.append("confidence_below_55_percent")
    if confidence_margin < MIN_SENTIMENT_MARGIN:
        uncertainty_reasons.append("top_two_classes_within_10_percent")
    return {
        "label": "uncertain" if uncertainty_reasons else model_label,
        "model_label": model_label,
        "confidence": confidence,
        "confidence_margin": confidence_margin,
        "is_uncertain": bool(uncertainty_reasons),
        "uncertainty_reasons": uncertainty_reasons,
    }


def calculate_priority(
    probabilities: dict[str, float],
    label: str,
    confidence: float,
    company_relevance: int | float = 3,
    event_type: str = "other",
) -> dict[str, Any]:
    relevance = clamp_company_relevance(company_relevance)
    relevance_norm = (relevance - 1.0) / 4.0
    impact_weight = event_weight(event_type)

    # Sentiment controls the review action, not the materiality score. Model
    # confidence measures classification certainty and is not event importance.
    relevance_contribution = 50.0 * relevance_norm
    event_contribution = 30.0 * impact_weight
    interaction_contribution = 20.0 * relevance_norm * impact_weight
    score = relevance_contribution + event_contribution + interaction_contribution
    if label == "uncertain":
        priority_label = "high" if score >= 75 else "medium" if score >= 45 else "low"
        action = "Manual sentiment review"
    elif score >= 75:
        priority_label = "high"
        if label == "negative":
            action = "Escalate for risk review"
        elif label == "positive":
            action = "Review as a potential catalyst"
        else:
            action = "Review event context"
    elif score >= 45:
        priority_label = "medium"
        action = "Review in the analyst queue"
    else:
        priority_label = "low"
        action = "Include in the monitoring digest"

    return {
        "priority_score": round(score, 2),
        "priority_label": priority_label,
        "recommended_action": action,
        "priority_drivers": {
            "company_relevance": relevance,
            "normalized_company_relevance": round(relevance_norm, 4),
            "event_weight": impact_weight,
            "relevance_contribution": round(relevance_contribution, 2),
            "event_contribution": round(event_contribution, 2),
            "interaction_contribution": round(interaction_contribution, 2),
        },
    }


def predict_sentiments(
    items: list[dict[str, Any]],
    engine: BaselineSentimentEngine | FinBertSentimentEngine | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        cleaned = str(item.get("text", "")).strip()
        if not cleaned:
            raise ValueError(f"Text at position {index} must not be empty.")
        normalized_items.append(
            {
                "text": cleaned,
                "company_relevance": clamp_company_relevance(item.get("company_relevance", 3)),
                "event_type": normalize_event_type(item.get("event_type", "other")),
            }
        )

    engine = engine or load_sentiment_engine()
    probability_rows = engine.predict_probabilities(
        [item["text"] for item in normalized_items]
    )
    results: list[dict[str, Any]] = []
    for item, probabilities in zip(normalized_items, probability_rows, strict=True):
        sentiment = interpret_sentiment(probabilities)
        priority = calculate_priority(
            probabilities=probabilities,
            label=sentiment["label"],
            confidence=sentiment["confidence"],
            company_relevance=item["company_relevance"],
            event_type=item["event_type"],
        )
        results.append(
            {
                **item,
                **sentiment,
                "probabilities": probabilities,
                **priority,
            }
        )
    return results


def predict_sentiment(
    text: str,
    company_relevance: int | float = 3,
    event_type: str = "other",
    engine: BaselineSentimentEngine | FinBertSentimentEngine | None = None,
) -> dict[str, Any]:
    return predict_sentiments(
        [
            {
                "text": text,
                "company_relevance": company_relevance,
                "event_type": event_type,
            }
        ],
        engine=engine,
    )[0]


def model_metadata(
    engine: BaselineSentimentEngine | FinBertSentimentEngine | None = None,
) -> dict[str, Any]:
    engine = engine or load_sentiment_engine()
    return {
        **engine.metadata(),
        "event_types": list(EVENT_TYPE_WEIGHTS.keys()),
        "uncertainty_thresholds": {
            "minimum_confidence": MIN_SENTIMENT_CONFIDENCE,
            "minimum_top_class_margin": MIN_SENTIMENT_MARGIN,
        },
    }
