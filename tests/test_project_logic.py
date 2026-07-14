from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.model_utils import BaselineSentimentEngine, calculate_priority, predict_sentiments
from app.news_provider import infer_event_type, normalize_tickers
from app.train_model import load_dataset


class DummyPipeline:
    classes_ = ["negative", "neutral", "positive"]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.05, 0.10, 0.85] for _ in texts]


class UncertainPipeline:
    classes_ = ["negative", "neutral", "positive"]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.34, 0.32, 0.34] for _ in texts]


class ProjectLogicTests(unittest.TestCase):
    def test_event_keywords_match_whole_words(self) -> None:
        self.assertEqual(infer_event_type("The company refined its strategy"), "other")
        self.assertEqual(infer_event_type("The SEC opened an investigation"), "legal_regulatory")
        self.assertEqual(infer_event_type("Quarterly profit increased"), "earnings")

    def test_tickers_are_normalized_and_deduplicated(self) -> None:
        self.assertEqual(normalize_tickers([" aapl ", "MSFT", "AAPL"]), ["AAPL", "MSFT"])

    def test_priority_uses_business_context_and_sentiment_selects_action(self) -> None:
        negative = calculate_priority(
            {"negative": 0.9, "neutral": 0.05, "positive": 0.05},
            label="negative",
            confidence=0.9,
            company_relevance=5,
            event_type="earnings",
        )
        positive = calculate_priority(
            {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
            label="positive",
            confidence=0.9,
            company_relevance=5,
            event_type="earnings",
        )
        self.assertEqual(negative["recommended_action"], "Escalate for risk review")
        self.assertEqual(positive["recommended_action"], "Review as a potential catalyst")
        self.assertEqual(negative["priority_score"], 100.0)
        self.assertEqual(negative["priority_score"], positive["priority_score"])
        self.assertNotIn("sentiment_contribution", negative["priority_drivers"])

    def test_model_confidence_does_not_change_priority_score(self) -> None:
        uncertain = calculate_priority(
            {"negative": 0.34, "neutral": 0.32, "positive": 0.34},
            label="positive",
            confidence=0.34,
            company_relevance=3,
            event_type="legal_regulatory",
        )
        confident = calculate_priority(
            {"negative": 0.01, "neutral": 0.01, "positive": 0.98},
            label="positive",
            confidence=0.98,
            company_relevance=3,
            event_type="legal_regulatory",
        )
        self.assertEqual(uncertain["priority_score"], confident["priority_score"])
        self.assertEqual(uncertain["priority_score"], 64.0)

    def test_close_probabilities_require_manual_sentiment_review(self) -> None:
        engine = BaselineSentimentEngine({"pipeline": UncertainPipeline()})
        result = predict_sentiments(
            [{"text": "Mixed financial outlook", "company_relevance": 3}],
            engine=engine,
        )[0]
        self.assertEqual(result["label"], "uncertain")
        self.assertEqual(result["model_label"], "negative")
        self.assertTrue(result["is_uncertain"])
        self.assertEqual(result["confidence_margin"], 0.0)
        self.assertEqual(result["recommended_action"], "Manual sentiment review")
        self.assertEqual(
            result["uncertainty_reasons"],
            ["confidence_below_55_percent", "top_two_classes_within_10_percent"],
        )

    def test_batch_prediction_uses_one_engine_call(self) -> None:
        engine = BaselineSentimentEngine({"pipeline": DummyPipeline()})
        results = predict_sentiments(
            [
                {"text": "Profit increased", "event_type": "earnings"},
                {"text": "Sales rose", "event_type": "unknown"},
            ],
            engine=engine,
        )
        self.assertEqual([result["label"] for result in results], ["positive", "positive"])
        self.assertEqual(results[1]["event_type"], "other")

    def test_dataset_loader_removes_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame(
                {
                    "sentence": ["Profit rose", "Profit rose", "Loss increased"],
                    "label": ["positive", "positive", "negative"],
                }
            ).to_csv(path, index=False)
            data = load_dataset(path)
        self.assertEqual(len(data), 2)

    def test_dataset_loader_rejects_conflicting_duplicate_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame(
                {
                    "sentence": ["Profit rose", "Profit rose"],
                    "label": ["positive", "negative"],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
