from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]{1,12}$")
DEFAULT_COMPANY_RELEVANCE = 3


class NewsProviderError(RuntimeError):
    pass


def finnhub_is_configured() -> bool:
    return bool(os.getenv("FINNHUB_API_KEY", "").strip())


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_ticker in tickers:
        ticker = str(raw_ticker).strip().upper()
        if not ticker:
            continue
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError(f"Invalid ticker: {raw_ticker!r}.")
        if ticker not in normalized:
            normalized.append(ticker)
    if not normalized:
        raise ValueError("Provide at least one ticker.")
    return normalized


def normalize_company_relevance(
    tickers: list[str],
    values: dict[str, int] | None = None,
) -> dict[str, int]:
    normalized_tickers = normalize_tickers(tickers)
    supplied = {
        str(raw_ticker).strip().upper(): value
        for raw_ticker, value in (values or {}).items()
    }
    normalized: dict[str, int] = {}
    for ticker in normalized_tickers:
        value = supplied.get(ticker, DEFAULT_COMPANY_RELEVANCE)
        if not 1 <= value <= 5:
            raise ValueError(f"Company relevance for {ticker} must be between 1 and 5.")
        normalized[ticker] = value
    return normalized


def relevance_for_article(
    article_tickers: list[str],
    company_relevance: dict[str, int],
) -> int:
    return max(
        (company_relevance.get(ticker, DEFAULT_COMPANY_RELEVANCE) for ticker in article_tickers),
        default=DEFAULT_COMPANY_RELEVANCE,
    )


def infer_event_type(headline: str, summary: str = "") -> str:
    text = f"{headline} {summary}".lower()
    keyword_groups = [
        ("guidance", ("guidance", "outlook", "forecast", "profit warning", "expects")),
        (
            "merger_acquisition",
            ("acquisition", "acquire", "merger", "takeover", "buyout", "bid for"),
        ),
        (
            "legal_regulatory",
            (
                "regulator",
                "regulatory",
                "lawsuit",
                "court",
                "investigation",
                "antitrust",
                "approval",
                "fine",
                "sec",
            ),
        ),
        (
            "management",
            ("ceo", "cfo", "chief executive", "chief financial", "board", "resign", "appoint"),
        ),
        (
            "macro",
            (
                "inflation",
                "interest rate",
                "federal reserve",
                "central bank",
                "gdp",
                "unemployment",
            ),
        ),
        (
            "product_partnership",
            ("launch", "product", "partnership", "contract", "collaboration", "supply agreement"),
        ),
        (
            "earnings",
            (
                "earnings",
                "revenue",
                "profit",
                "sales",
                "eps",
                "quarter",
                "quarterly",
                "operating income",
            ),
        ),
    ]
    for event_type, keywords in keyword_groups:
        if any(_contains_phrase(text, keyword) for keyword in keywords):
            return event_type
    return "other"


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


def _published_at(value: Any) -> str:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).isoformat()
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def fetch_company_news(
    tickers: list[str],
    days: int = 3,
    limit_per_ticker: int = 25,
    company_relevance: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise NewsProviderError(
            "FINNHUB_API_KEY is not configured. Add it to the backend environment and restart the service."
        )

    normalized_tickers = normalize_tickers(tickers)
    relevance_by_ticker = normalize_company_relevance(normalized_tickers, company_relevance)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=max(0, days - 1))
    deduplicated: dict[str, dict[str, Any]] = {}

    with requests.Session() as session:
        for ticker in normalized_tickers:
            try:
                response = session.get(
                    FINNHUB_COMPANY_NEWS_URL,
                    params={
                        "symbol": ticker,
                        "from": start_date.isoformat(),
                        "to": end_date.isoformat(),
                        "token": api_key,
                    },
                    timeout=15,
                )
            except requests.RequestException as exc:
                raise NewsProviderError(f"Finnhub request failed: {exc}") from exc

            if response.status_code == 429:
                raise NewsProviderError("Finnhub rate limit reached. Try again later.")
            if response.status_code in {401, 403}:
                raise NewsProviderError("Finnhub rejected the API key.")
            try:
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:
                raise NewsProviderError("Finnhub returned an invalid response.") from exc

            if isinstance(payload, dict):
                raise NewsProviderError(str(payload.get("error") or "Finnhub returned an error."))

            sorted_payload = sorted(
                payload,
                key=lambda item: int(item.get("datetime") or 0),
                reverse=True,
            )
            for article in sorted_payload[:limit_per_ticker]:
                headline = str(article.get("headline") or "").strip()
                if not headline:
                    continue
                summary = str(article.get("summary") or "").strip()
                url = str(article.get("url") or "").strip()
                provider_id = str(article.get("id") or "").strip()
                dedupe_key = url or provider_id or headline.lower()

                if dedupe_key in deduplicated:
                    existing_tickers = deduplicated[dedupe_key]["tickers"]
                    if ticker not in existing_tickers:
                        existing_tickers.append(ticker)
                    continue

                deduplicated[dedupe_key] = {
                    "provider_id": provider_id,
                    "tickers": [ticker],
                    "ticker": ticker,
                    "text": headline,
                    "headline": headline,
                    "summary": summary,
                    "source": str(article.get("source") or "Finnhub").strip(),
                    "published_at": _published_at(article.get("datetime")),
                    "url": url,
                    "image_url": str(article.get("image") or "").strip(),
                    "category": str(article.get("category") or "company").strip(),
                    "related": str(article.get("related") or "").strip(),
                    "company_relevance": relevance_by_ticker[ticker],
                    "event_type": infer_event_type(headline, summary),
                }

    articles = list(deduplicated.values())
    for article in articles:
        article["ticker"] = ", ".join(article["tickers"][:3])
        article["company_relevance"] = relevance_for_article(
            article["tickers"],
            relevance_by_ticker,
        )
    return sorted(articles, key=lambda article: article["published_at"], reverse=True)
