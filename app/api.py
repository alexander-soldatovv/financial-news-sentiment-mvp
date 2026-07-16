from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model_utils import (
    EVENT_TYPE_WEIGHTS,
    load_sentiment_engine,
    model_metadata,
    predict_sentiment,
    predict_sentiments,
)
from app.news_provider import (
    NewsProviderError,
    fetch_company_news,
    finnhub_is_configured,
    normalize_tickers,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


app = FastAPI(
    title="Financial News Sentiment API",
    description="Investor-oriented financial news classification and review prioritization.",
    version="0.2.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8001,http://localhost:8001",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EventType = Literal[
    "earnings",
    "guidance",
    "merger_acquisition",
    "legal_regulatory",
    "management",
    "macro",
    "product_partnership",
    "other",
]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Financial sentence or news item.")
    company_relevance: int = Field(
        3,
        ge=1,
        le=5,
        description="How relevant this company/news source is for the analyst universe.",
    )
    event_type: EventType = Field(
        "other",
        description=f"Event category. Supported values: {', '.join(EVENT_TYPE_WEIGHTS.keys())}.",
    )


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] | None = Field(default=None, min_length=1, max_length=1000)
    texts: list[str] | None = Field(default=None, min_length=1, max_length=1000)


class NewsFetchRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, max_length=10)
    days: int = Field(3, ge=1, le=30)
    limit_per_ticker: int = Field(25, ge=1, le=50)
    company_relevance: dict[str, Annotated[int, Field(ge=1, le=5)]] = Field(
        default_factory=dict,
        description="Persistent watchlist relevance for each ticker; unspecified tickers default to 3.",
    )


class PredictionResponse(BaseModel):
    text: str
    label: str
    model_label: str
    confidence: float
    confidence_margin: float
    is_uncertain: bool
    uncertainty_reasons: list[str]
    probabilities: dict[str, float]
    company_relevance: float
    event_type: str
    priority_score: float
    priority_label: str
    recommended_action: str
    priority_drivers: dict[str, float]


@app.get("/api-info")
def root() -> dict[str, str]:
    return {"service": "financial-news-sentiment", "docs": "/docs"}


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    try:
        engine = load_sentiment_engine()
        return {
            "status": "ok",
            "model_loaded": True,
            "news_provider": {
                "name": "finnhub",
                "configured": finnhub_is_configured(),
            },
            **model_metadata(engine),
        }
    except Exception as exc:
        response.status_code = 503
        return {"status": "error", "model_loaded": False, "detail": str(exc)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        return predict_sentiment(
            request.text,
            company_relevance=request.company_relevance,
            event_type=request.event_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict_batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    if request.items is not None and request.texts is not None:
        raise HTTPException(status_code=400, detail="Provide `items` or `texts`, not both.")
    if request.items is not None:
        items = request.items
    elif request.texts is not None:
        try:
            items = [PredictRequest(text=text) for text in request.texts]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid text: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="Provide either `items` or `texts`.")

    try:
        results = predict_sentiments(
            [
                {
                    "text": item.text,
                    "company_relevance": item.company_relevance,
                    "event_type": item.event_type,
                }
                for item in items
            ]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"count": len(results), "results": results}


@app.post("/news/fetch")
def fetch_news(request: NewsFetchRequest) -> dict[str, Any]:
    try:
        normalized_tickers = normalize_tickers(request.tickers)
        articles = fetch_company_news(
            normalized_tickers,
            days=request.days,
            limit_per_ticker=request.limit_per_ticker,
            company_relevance=request.company_relevance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NewsProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        predictions = predict_sentiments(
            [
                {
                    "text": article["text"],
                    "company_relevance": article["company_relevance"],
                    "event_type": article["event_type"],
                    "event_status_multiplier": article["event_status_multiplier"],
                    "company_directness_multiplier": article["company_directness_multiplier"],
                }
                for article in articles
            ]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    results = [
        {**article, **prediction}
        for article, prediction in zip(articles, predictions, strict=True)
    ]
    return {
        "provider": "finnhub",
        "count": len(results),
        "tickers": normalized_tickers,
        "results": results,
    }


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
