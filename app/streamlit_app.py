from __future__ import annotations

import os
from io import StringIO
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8001").rstrip("/")
EVENT_TYPES = [
    "earnings",
    "guidance",
    "merger_acquisition",
    "legal_regulatory",
    "management",
    "macro",
    "product_partnership",
    "other",
]


st.set_page_config(
    page_title="Financial News Sentiment",
    page_icon="",
    layout="wide",
)


def call_api(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{API_URL}{path}"
    if payload is None:
        response = requests.get(url, timeout=10)
    else:
        response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def render_prediction(result: dict[str, Any]) -> None:
    left, middle, right = st.columns(3)
    left.metric("Predicted sentiment", result["label"], f"{result['confidence']:.1%} confidence")
    middle.metric("Priority", result["priority_label"], f"{result['priority_score']:.0f}/100")
    right.metric("Recommended action", result["recommended_action"])
    probabilities = pd.DataFrame(
        {
            "label": list(result["probabilities"].keys()),
            "probability": list(result["probabilities"].values()),
        }
    )
    st.bar_chart(probabilities, x="label", y="probability", height=260)


st.title("Financial News Sentiment")
st.caption("First-level screening assistant for investor-oriented financial news.")

with st.sidebar:
    st.subheader("API")
    st.write(API_URL)
    if st.button("Check API health"):
        try:
            health = call_api("/health")
            st.success(health["status"])
            st.json(health.get("metrics", {}))
        except Exception as exc:
            st.error(f"API is not available: {exc}")

single_tab, batch_tab = st.tabs(["Single news item", "CSV batch"])

with single_tab:
    text = st.text_area(
        "Financial sentence or news item",
        height=150,
        placeholder="Example: Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.",
    )
    relevance = st.slider("Company relevance", min_value=1, max_value=5, value=3)
    event_type = st.selectbox("Event type", options=EVENT_TYPES, index=EVENT_TYPES.index("other"))
    if st.button("Predict sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter a sentence first.")
        else:
            try:
                result = call_api(
                    "/predict",
                    {
                        "text": text,
                        "company_relevance": relevance,
                        "event_type": event_type,
                    },
                )
                render_prediction(result)
                with st.expander("Priority drivers"):
                    st.json(result["priority_drivers"])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with batch_tab:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("CSV is empty.")
        else:
            text_column = st.selectbox("Text column", options=list(df.columns))
            default_relevance = st.slider(
                "Default company relevance",
                min_value=1,
                max_value=5,
                value=3,
                key="batch_relevance",
            )
            default_event_type = st.selectbox(
                "Default event type",
                options=EVENT_TYPES,
                index=EVENT_TYPES.index("other"),
                key="batch_event_type",
            )
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("Predict CSV", type="primary"):
                texts = df[text_column].fillna("").astype(str).str.strip()
                valid_mask = texts != ""
                if not valid_mask.any():
                    st.warning("Selected column has no non-empty text.")
                else:
                    try:
                        items = [
                            {
                                "text": text,
                                "company_relevance": default_relevance,
                                "event_type": default_event_type,
                            }
                            for text in texts[valid_mask].tolist()
                        ]
                        response = call_api(
                            "/predict_batch",
                            {"items": items},
                        )
                        result_df = df.loc[valid_mask].copy()
                        predictions = pd.DataFrame(response["results"])
                        result_df["sentiment"] = predictions["label"].values
                        result_df["confidence"] = predictions["confidence"].values
                        result_df["priority_score"] = predictions["priority_score"].values
                        result_df["priority_label"] = predictions["priority_label"].values
                        result_df["recommended_action"] = predictions["recommended_action"].values
                        for label in ["negative", "neutral", "positive"]:
                            result_df[f"p_{label}"] = predictions["probabilities"].apply(
                                lambda values: values[label]
                            ).values

                        st.success(f"Predicted {len(result_df)} rows.")
                        result_df = result_df.sort_values(
                            ["priority_score", "confidence"],
                            ascending=False,
                        )
                        st.dataframe(result_df, use_container_width=True)
                        csv_buffer = StringIO()
                        result_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            "Download predictions as CSV",
                            data=csv_buffer.getvalue(),
                            file_name="financial_news_sentiment_predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Batch prediction failed: {exc}")
