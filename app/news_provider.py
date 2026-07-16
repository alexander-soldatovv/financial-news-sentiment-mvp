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
FINNHUB_COMPANY_PROFILE_URL = "https://finnhub.io/api/v1/stock/profile2"
TICKER_PATTERN = re.compile(r"^[A-Z0-9.-]{1,12}$")
DEFAULT_COMPANY_RELEVANCE = 3
DEFAULT_TICKER_ALIAS_OVERRIDES: dict[str, set[str]] = {
    # Private-company/demo symbol used in the analyst workflow. Finnhub can tag
    # SpaceX-related articles with SPCX, but company profiles do not reliably
    # provide "SpaceX" as an alias for this pseudo-symbol.
    "SPCX": {"spacex", "space x"},
}
PROFILE_ALIAS_CACHE: dict[str, set[str]] = {}


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


def infer_event_status(headline: str) -> tuple[str, float]:
    text = headline.lower().strip()
    speculative_phrases = (
        "rumor",
        "reportedly",
        "might",
        "could",
        "considering",
        "possible",
        "likely",
        "prediction",
        "should you",
        "in focus",
        "what it means",
        "what this means",
        "here's why",
    )
    speculative_patterns = (
        r"\bmay\s+(?:be|have|face|seek|consider|acquire|merge|buy|sell|raise|cut|"
        r"report|announce|benefit|hurt|fall|rise|miss|beat|lose|gain|need|take|move)\b",
    )
    forecast_phrases = (
        "analyst",
        "analysts",
        "betting market",
        "wall street expects",
        "expected",
        "expects",
        "forecast",
        "odds",
        "polymarket",
        "prediction market",
        "price target",
        "estimated",
        "seen",
        "projects",
    )
    confirmed_phrases = (
        "completed",
        "completes",
        "approved",
        "signed",
        "reports",
        "reported",
        "results",
        "posted",
        "awarded",
        "acquired",
        "rose",
        "fell",
        "increased",
        "decreased",
        "raised guidance",
        "cut guidance",
    )
    announced_phrases = (
        "announces",
        "announced",
        "plans",
        "launches",
        "launch",
        "appoints",
        "appointed",
        "resigns",
        "files",
        "sues",
        "enters agreement",
    )
    if any(_contains_phrase(text, phrase) for phrase in forecast_phrases):
        return "forecast", 0.65
    if (
        "?" in headline
        or any(_contains_phrase(text, phrase) for phrase in speculative_phrases)
        or any(re.search(pattern, text) for pattern in speculative_patterns)
    ):
        return "speculative", 0.4
    if any(_contains_phrase(text, phrase) for phrase in confirmed_phrases):
        return "confirmed", 1.0
    if any(_contains_phrase(text, phrase) for phrase in announced_phrases):
        return "announced", 0.85
    return "announced", 0.85


def infer_company_directness(
    headline: str,
    article_tickers: list[str],
    aliases_by_ticker: dict[str, set[str]],
) -> tuple[str, float]:
    text = headline.lower()
    matched_tickers: set[str] = set()
    first_match: int | None = None
    for ticker in article_tickers:
        aliases = aliases_by_ticker.get(ticker, {ticker.lower()}) | _ticker_alias_overrides(ticker)
        for alias in aliases:
            normalized_alias = alias.strip().lower()
            if not normalized_alias:
                continue
            if _is_short_symbol_alias(normalized_alias, ticker):
                match = _ticker_symbol_match(headline, ticker)
            else:
                match = _phrase_match(text, normalized_alias)
            if match is None:
                continue
            matched_tickers.add(ticker)
            first_match = match.start() if first_match is None else min(first_match, match.start())

    if not matched_tickers:
        return "incidental", 0.3

    broad_patterns = (
        r"global\s+.+\s+market",
        r"market\s+(?:size|trends|outlook)",
        r"industry\s+trends",
        r"stocks?\s+to\s+watch",
        r"\bin\s+focus\b",
        r"\bfutures\b",
        r"what(?:'s|\s+is)\s+moving\s+markets",
        r"featuring\b",
        r"\band\s+\d+\s+other\b",
    )
    is_broad = any(re.search(pattern, text) for pattern in broad_patterns)
    if is_broad or len(matched_tickers) > 1 or (first_match is not None and first_match > 80):
        return "broad", 0.65
    return "direct", 1.0


def _contains_phrase(text: str, phrase: str) -> bool:
    return _phrase_match(text, phrase) is not None


def _phrase_match(text: str, phrase: str) -> re.Match[str] | None:
    escaped = re.escape(phrase.strip()).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text)


def _is_short_symbol_alias(alias: str, ticker: str) -> bool:
    compact_ticker = ticker.replace(".", "").replace("-", "")
    return alias == ticker.lower() and len(compact_ticker) <= 2 and compact_ticker.isalpha()


def _ticker_symbol_match(headline: str, ticker: str) -> re.Match[str] | None:
    escaped = re.escape(ticker)
    return re.search(rf"(?<![A-Z0-9.-]){escaped}(?![A-Z0-9.-])", headline)


def _ticker_alias_overrides(ticker: str) -> set[str]:
    return set(DEFAULT_TICKER_ALIAS_OVERRIDES.get(ticker.upper(), set()))


def _company_aliases(
    session: requests.Session,
    ticker: str,
    api_key: str,
) -> set[str]:
    if ticker in PROFILE_ALIAS_CACHE:
        return PROFILE_ALIAS_CACHE[ticker]

    aliases = {ticker.lower()} | _ticker_alias_overrides(ticker)
    try:
        response = session.get(
            FINNHUB_COMPANY_PROFILE_URL,
            params={"symbol": ticker, "token": api_key},
            timeout=10,
        )
        response.raise_for_status()
        profile = response.json()
        company_name = str(profile.get("name") or "").strip()
        if company_name:
            aliases.add(company_name.lower())
            short_name = re.sub(
                r"\s+(?:incorporated|inc\.?|corporation|corp\.?|company|co\.?|limited|ltd\.?)$",
                "",
                company_name,
                flags=re.IGNORECASE,
            ).strip()
            if len(short_name) >= 3:
                aliases.add(short_name.lower())
    except (requests.RequestException, ValueError, AttributeError):
        pass

    PROFILE_ALIAS_CACHE[ticker] = aliases
    return aliases


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
        aliases_by_ticker = {
            ticker: _company_aliases(session, ticker, api_key)
            for ticker in normalized_tickers
        }
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
        event_status, status_multiplier = infer_event_status(article["headline"])
        company_directness, directness_multiplier = infer_company_directness(
            article["headline"],
            article["tickers"],
            aliases_by_ticker,
        )
        article["event_status"] = event_status
        article["event_status_multiplier"] = status_multiplier
        article["company_directness"] = company_directness
        article["company_directness_multiplier"] = directness_multiplier
    return sorted(articles, key=lambda article: article["published_at"], reverse=True)
