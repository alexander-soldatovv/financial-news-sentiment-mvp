import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Download,
  ExternalLink,
  Filter,
  Inbox,
  LoaderCircle,
  Menu,
  Newspaper,
  Radio,
  RefreshCw,
  ScanSearch,
  Search,
  Settings2,
  SlidersHorizontal,
  X,
} from "lucide-react";
import Papa from "papaparse";
import {
  itemsForQueueView,
  itemTickers,
  mergeQueueItems,
  normalizeRelevanceMap,
  normalizeStoredItems,
  parseTickers,
  recalculatePriorityForItem,
} from "./queueUtils.js";

const API_URL = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.DEV ? "http://127.0.0.1:8001" : "")
).replace(/\/$/, "");
const QUEUE_STORAGE_KEY = "sentiment-desk-queue-v9-company-match-audit";
const WATCHLIST_STORAGE_KEY = "sentiment-desk-watchlist-v1";
const RELEVANCE_STORAGE_KEY = "sentiment-desk-watchlist-relevance-v1";
const DEFAULT_WATCHLIST_TICKERS = ["AAPL", "MSFT", "NVDA"];
const MAX_WATCHLIST_TICKERS = 10;
const TICKER_PATTERN = /^[A-Z0-9.-]{1,12}$/;

const EVENT_TYPES = [
  ["earnings", "Earnings"],
  ["guidance", "Guidance"],
  ["merger_acquisition", "M&A"],
  ["legal_regulatory", "Legal & regulatory"],
  ["management", "Management"],
  ["macro", "Macro"],
  ["product_partnership", "Product & partnership"],
  ["other", "Other"],
];

const sentimentMeta = {
  negative: { label: "Negative", icon: ArrowDownRight },
  neutral: { label: "Neutral", icon: Activity },
  positive: { label: "Positive", icon: ArrowUpRight },
  uncertain: { label: "Uncertain", icon: CircleHelp },
};

const priorityMeta = {
  high: { label: "High" },
  medium: { label: "Medium" },
  low: { label: "Low" },
};

const reviewStatusMeta = {
  new: "New",
  reviewed: "Reviewed",
  dismissed: "Review later",
};

function eventLabel(value) {
  return EVENT_TYPES.find(([key]) => key === value)?.[1] || "Other";
}

function contextLabel(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/^./, (character) => character.toUpperCase());
}

function formatPublishedAt(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  const now = new Date();
  const sameDay = parsed.toDateString() === now.toDateString();
  if (sameDay) {
    return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return parsed.toLocaleDateString([], { month: "short", day: "numeric" });
}

function storedJson(key, fallback) {
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function storeJson(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage is an enhancement; the live queue remains usable if it is unavailable.
  }
}

async function api(path, options = {}) {
  let response;
  try {
    const { headers, ...fetchOptions } = options;
    response = await fetch(`${API_URL}${path}`, {
      ...fetchOptions,
      headers: { "Content-Type": "application/json", ...(headers || {}) },
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Cannot reach the backend. Check that the server is running and try again.");
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((item) => item.msg || String(item)).join("; ")
      : payload.detail;
    const error = new Error(detail || `Request failed with status ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return response.json();
}

function SentimentBadge({ value }) {
  const meta = sentimentMeta[value] || sentimentMeta.neutral;
  const Icon = meta.icon;
  return (
    <span className={`badge sentiment ${value}`}>
      <Icon size={14} aria-hidden="true" />
      {meta.label}
    </span>
  );
}

function PriorityBadge({ value }) {
  const label = priorityMeta[value]?.label || value;
  return (
    <span className={`badge priority ${value}`}>
      <span className="priority-dot" />
      {label}
    </span>
  );
}

function PriorityMeter({ value, compact = false }) {
  const score = Math.max(0, Math.min(100, Number(value) || 0));
  return (
    <span className={`priority-meter ${compact ? "compact" : ""}`} aria-hidden="true">
      <span style={{ width: `${score}%` }} />
    </span>
  );
}

function ProbabilityBar({ label, value }) {
  return (
    <div className="probability-row">
      <div className="probability-label">
        <span>{sentimentMeta[label]?.label || label}</span>
        <strong>{Math.round(value * 100)}%</strong>
      </div>
      <div className="probability-track">
        <div className={`probability-fill ${label}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function EmptyState({ title = "Your review queue is empty", description = "Enter company tickers above and fetch news to build the ranked analyst queue." }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Inbox size={25} /></div>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [items, setItems] = useState(() => normalizeStoredItems(storedJson(QUEUE_STORAGE_KEY, [])));
  const [selectedId, setSelectedId] = useState(null);
  const [query, setQuery] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [companyFilter, setCompanyFilter] = useState("all");
  const [queueView, setQueueView] = useState("active");
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [trackedTickers, setTrackedTickers] = useState(() => {
    const storedTickers = storedJson(WATCHLIST_STORAGE_KEY, null);
    const tickers = storedTickers === null ? DEFAULT_WATCHLIST_TICKERS : storedTickers;
    const parsedTickers = parseTickers(Array.isArray(tickers) ? tickers.join(" ") : tickers)
      .filter((ticker) => TICKER_PATTERN.test(ticker))
      .slice(0, MAX_WATCHLIST_TICKERS);
    return parsedTickers.length || storedTickers !== null ? parsedTickers : DEFAULT_WATCHLIST_TICKERS;
  });
  const [tickerDraft, setTickerDraft] = useState("");
  const [tickerRelevance, setTickerRelevance] = useState(() => (
    normalizeRelevanceMap(storedJson(RELEVANCE_STORAGE_KEY, {}))
  ));
  const [newsDays, setNewsDays] = useState(3);
  const [fetchingNews, setFetchingNews] = useState(false);
  const [toast, setToast] = useState(null);
  const tickerInput = useRef(null);
  const fetchButton = useRef(null);
  const activeFetchController = useRef(null);
  const fetchRequestId = useRef(0);
  const trackedTickersRef = useRef(trackedTickers);
  const tickerRelevanceRef = useRef(tickerRelevance);

  const selectedItem = items.find((item) => item.id === selectedId) || null;

  const refreshHealth = async () => {
    try {
      const data = await api("/health");
      if (data.status !== "ok") throw new Error(data.detail || "Model unavailable");
      setHealth(data);
      setHealthError("");
    } catch (error) {
      setHealth(null);
      setHealthError(error.message);
    }
  };

  useEffect(() => {
    refreshHealth();
  }, []);

  useEffect(() => {
    storeJson(QUEUE_STORAGE_KEY, items);
  }, [items]);

  useEffect(() => {
    trackedTickersRef.current = trackedTickers;
    storeJson(WATCHLIST_STORAGE_KEY, trackedTickers);
  }, [trackedTickers]);

  useEffect(() => {
    const activeTickers = new Set(trackedTickers);
    setItems((current) => current.filter((item) => (
      itemTickers(item).some((ticker) => activeTickers.has(ticker))
    )));
  }, [trackedTickers]);

  useEffect(() => {
    tickerRelevanceRef.current = tickerRelevance;
    storeJson(RELEVANCE_STORAGE_KEY, tickerRelevance);
  }, [tickerRelevance]);

  useEffect(() => () => activeFetchController.current?.abort(), []);

  useEffect(() => {
    setItems((current) => current.map((item) => recalculatePriorityForItem(item, tickerRelevance)));
  }, [tickerRelevance]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 3500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const scopedItems = useMemo(
    () => itemsForQueueView(items, queueView),
    [items, queueView],
  );

  const filteredItems = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return scopedItems
      .filter((item) => companyFilter === "all" || itemTickers(item).includes(companyFilter))
      .filter((item) => sentimentFilter === "all" || item.label === sentimentFilter)
      .filter((item) => priorityFilter === "all" || item.priority_label === priorityFilter)
      .filter((item) => {
        if (!needle) return true;
        return [item.text, item.summary, item.ticker, item.source, item.event_type]
          .some((value) => String(value || "").toLowerCase().includes(needle));
      })
      .sort((a, b) => {
        if (queueView === "completed") {
          const completedOrder = new Date(b.reviewed_at || 0).getTime()
            - new Date(a.reviewed_at || 0).getTime();
          if (completedOrder) return completedOrder;
        }
        return b.priority_score - a.priority_score
          || new Date(b.published_at || 0).getTime() - new Date(a.published_at || 0).getTime()
          || b.confidence - a.confidence;
      });
  }, [companyFilter, priorityFilter, query, queueView, scopedItems, sentimentFilter]);

  const companyStats = useMemo(() => trackedTickers.map((ticker) => {
    const companyItems = scopedItems.filter((item) => itemTickers(item).includes(ticker));
    return {
      ticker,
      total: companyItems.length,
      high: companyItems.filter((item) => item.priority_label === "high").length,
      relevance: Number(tickerRelevance[ticker] || 3),
    };
  }), [scopedItems, tickerRelevance, trackedTickers]);

  const stats = useMemo(() => {
    const active = items.filter((item) => !["reviewed", "dismissed"].includes(item.review_status));
    const directional = active.filter((item) => ["negative", "positive"].includes(item.label)).length;
    const high = active.filter((item) => item.priority_label === "high").length;
    const completed = items.filter((item) => ["reviewed", "dismissed"].includes(item.review_status)).length;
    return { total: active.length, directional, high, completed };
  }, [items]);

  const addPredictions = (
    predictions,
    metadata,
    { selectFirst = true, replaceTickers = [] } = {},
  ) => {
    const timestamp = Date.now();
    const normalized = predictions
      .map((prediction, index) => {
        return {
          ...metadata[index],
          ...prediction,
          id: `${timestamp}-${index}-${Math.random().toString(36).slice(2, 7)}`,
          review_status: "new",
        };
      });
    setItems((current) => mergeQueueItems(current, normalized, { replaceTickers }));
    if (selectFirst && normalized[0]) setSelectedId(normalized[0].id);
    return normalized;
  };

  const updateReviewStatus = (id, reviewStatus) => {
    setItems((current) => current.map((item) => (
      item.id === id
        ? {
          ...item,
          review_status: reviewStatus,
          reviewed_at: reviewStatus === "new" ? null : new Date().toISOString(),
        }
        : item
    )));
    setSelectedId(null);
    const label = reviewStatusMeta[reviewStatus] || reviewStatus;
    setToast({
      type: "success",
      message: reviewStatus === "new"
        ? "Item moved back to the review queue."
        : (reviewStatus === "dismissed"
          ? "Item saved for later and moved to Completed."
          : `Item marked as ${label.toLowerCase()} and moved to Completed.`),
    });
  };

  const removeTrackedTicker = (ticker) => {
    const relatedItems = items.filter((item) => itemTickers(item).includes(ticker));
    const relatedIds = new Set(relatedItems.map((item) => item.id));
    fetchRequestId.current += 1;
    activeFetchController.current?.abort();
    activeFetchController.current = null;
    setFetchingNews(false);
    const nextTickers = trackedTickersRef.current.filter((value) => value !== ticker);
    trackedTickersRef.current = nextTickers;
    setTrackedTickers(nextTickers);
    setTickerRelevance((current) => {
      const next = { ...current };
      delete next[ticker];
      tickerRelevanceRef.current = next;
      return next;
    });
    setItems((current) => current.filter((item) => !itemTickers(item).includes(ticker)));
    if (relatedIds.has(selectedId)) setSelectedId(null);
    if (companyFilter === ticker) setCompanyFilter("all");
    setToast({
      type: "success",
      message: `${ticker} removed. ${relatedItems.length} related article${relatedItems.length === 1 ? "" : "s"} deleted from the queue.`,
    });
  };

  const addWatchlistTickers = (value, { autoFetch = true } = {}) => {
    if (activeFetchController.current) {
      setToast({
        type: "error",
        message: "Wait for the current news refresh to finish before adding another ticker.",
      });
      return false;
    }
    const candidates = parseTickers(value);
    if (!candidates.length) return false;

    const invalidTicker = candidates.find((ticker) => !TICKER_PATTERN.test(ticker));
    if (invalidTicker) {
      setToast({
        type: "error",
        message: `${invalidTicker} is not a valid ticker. Use letters, numbers, dot or hyphen.`,
      });
      return false;
    }

    const currentTickers = trackedTickersRef.current;
    const existing = new Set(currentTickers);
    const additions = candidates.filter((ticker) => !existing.has(ticker));
    if (!additions.length) {
      setTickerDraft("");
      return true;
    }

    const openSlots = MAX_WATCHLIST_TICKERS - currentTickers.length;
    if (openSlots <= 0) {
      setToast({ type: "error", message: `Watchlist is limited to ${MAX_WATCHLIST_TICKERS} tickers.` });
      return false;
    }

    const accepted = additions.slice(0, openSlots);
    const nextTickers = [...currentTickers, ...accepted];
    trackedTickersRef.current = nextTickers;
    setTrackedTickers(nextTickers);
    setTickerDraft("");

    if (additions.length > openSlots) {
      setToast({
        type: "error",
        message: `Added ${accepted.length}; skipped ${additions.length - openSlots} because the watchlist is full.`,
      });
    }

    if (autoFetch) {
      fetchWatchlistNews({
        tickers: nextTickers,
        draft: "",
        rollbackTickers: currentTickers,
        rollbackRelevance: tickerRelevanceRef.current,
      });
    }

    return true;
  };

  const handleTickerDraftKeyDown = (event) => {
    const shouldCommitTicker = ["Enter", "Tab", ",", ";", " "].includes(event.key);
    if (shouldCommitTicker && tickerDraft.trim()) {
      event.preventDefault();
      addWatchlistTickers(tickerDraft);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      fetchWatchlistNews();
      return;
    }
    if (event.key === "Backspace" && !tickerDraft && trackedTickers.length) {
      removeTrackedTicker(trackedTickers[trackedTickers.length - 1]);
    }
  };

  const handleTickerPaste = (event) => {
    const pastedText = event.clipboardData.getData("text");
    if (!parseTickers(pastedText).length) return;
    event.preventDefault();
    addWatchlistTickers(pastedText);
  };

  const updateTickerRelevance = (ticker, value) => {
    setTickerRelevance((current) => {
      const next = { ...current, [ticker]: Number(value) };
      tickerRelevanceRef.current = next;
      return next;
    });
    setToast({
      type: "success",
      message: `${ticker} relevance set to ${value}/5. Existing rankings updated.`,
    });
  };

  const fetchWatchlistNews = async ({
    tickers: requestedTickers = null,
    draft = tickerDraft,
    rollbackTickers = null,
    rollbackRelevance = null,
  } = {}) => {
    const pendingTickers = parseTickers(draft);
    const invalidTicker = pendingTickers.find((ticker) => !TICKER_PATTERN.test(ticker));
    if (invalidTicker) {
      setToast({
        type: "error",
        message: `${invalidTicker} is not a valid ticker. Use letters, numbers, dot or hyphen.`,
      });
      return;
    }

    const tickers = [...new Set([...(requestedTickers || trackedTickersRef.current), ...pendingTickers])]
      .slice(0, MAX_WATCHLIST_TICKERS);
    if (!tickers.length) {
      setToast({ type: "error", message: "Enter at least one company ticker." });
      return;
    }

    const previousTickers = rollbackTickers || trackedTickersRef.current;
    const previousRelevance = rollbackRelevance || tickerRelevanceRef.current;
    activeFetchController.current?.abort();
    const controller = new AbortController();
    const requestId = fetchRequestId.current + 1;
    fetchRequestId.current = requestId;
    activeFetchController.current = controller;
    setFetchingNews(true);
    setSelectedId(null);
    const relevanceByTicker = Object.fromEntries(
      tickers.map((ticker) => [ticker, Number(tickerRelevanceRef.current[ticker] || 3)]),
    );
    trackedTickersRef.current = tickers;
    tickerRelevanceRef.current = relevanceByTicker;
    setTrackedTickers(tickers);
    setTickerDraft("");
    setTickerRelevance(relevanceByTicker);
    try {
      const response = await api("/news/fetch", {
        method: "POST",
        body: JSON.stringify({
          tickers,
          days: Number(newsDays),
          limit_per_ticker: 50,
          company_relevance: relevanceByTicker,
        }),
        signal: controller.signal,
      });
      if (requestId !== fetchRequestId.current) return;
      const currentRelevance = tickerRelevanceRef.current;
      const refreshedResults = response.results.map((item) => (
        recalculatePriorityForItem(item, currentRelevance)
      ));
      addPredictions(refreshedResults, refreshedResults, {
        selectFirst: false,
        replaceTickers: tickers,
      });
      if (!response.results.length) {
        setToast({ type: "success", message: "No company news found for this period." });
        return;
      }
      const counts = tickers.map((ticker) => {
        const count = refreshedResults.filter((item) => itemTickers(item).includes(ticker)).length;
        return `${ticker}: ${count}`;
      });
      setToast({
        type: "success",
        message: `${response.count} articles loaded · ${counts.join(" · ")}`,
      });
    } catch (error) {
      if (error.name === "AbortError" || requestId !== fetchRequestId.current) return;
      if (error.status === 400 && /^No compan(?:y|ies) found/.test(error.message)) {
        trackedTickersRef.current = previousTickers;
        tickerRelevanceRef.current = previousRelevance;
        setTrackedTickers(previousTickers);
        setTickerRelevance(previousRelevance);
      }
      setToast({ type: "error", message: error.message });
    } finally {
      if (requestId === fetchRequestId.current) {
        activeFetchController.current = null;
        setFetchingNews(false);
      }
    }
  };

  const exportCsv = () => {
    if (!filteredItems.length) return;
    const csv = Papa.unparse(filteredItems.map((item) => ({
      ticker: item.ticker,
      text: item.text,
      summary: item.summary || "",
      source: item.source,
      sentiment: item.label,
      confidence: item.confidence,
      priority_score: item.priority_score,
      priority: item.priority_label,
      event_type: item.event_type,
      recommended_action: item.recommended_action,
      review_status: item.review_status,
    })));
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sentiment-desk-queue.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ScanSearch size={21} /></div>
          <div>
            <div className="brand-name">Sentiment Desk</div>
            <div className="brand-subtitle">Financial news intelligence</div>
          </div>
        </div>
        <div className="topbar-actions">
          <button className="icon-button mobile-menu" title="Filters" onClick={() => setMobileFiltersOpen(true)}>
            <Menu size={19} />
          </button>
          <div
            className={`service-status ${health ? "online" : healthError ? "offline" : "pending"}`}
            title={health ? "Model API is available" : healthError || "Checking model API"}
          >
            <span className="status-dot" />
            <span>{health ? "Model online" : healthError ? "Model offline" : "Checking model"}</span>
          </div>
        </div>
      </header>

      <div className="workspace">
        <aside className={`sidebar ${mobileFiltersOpen ? "mobile-open" : ""}`}>
          <div className="mobile-sidebar-head">
            <strong>Queue filters</strong>
            <button className="icon-button" title="Close filters" onClick={() => setMobileFiltersOpen(false)}><X size={19} /></button>
          </div>
          <nav className="nav-section" aria-label="Workspace">
            <div className="nav-label">Workspace</div>
            <button
              className={`nav-item ${queueView === "active" ? "active" : ""}`}
              aria-current={queueView === "active" ? "page" : undefined}
              onClick={() => { setQueueView("active"); setSelectedId(null); setMobileFiltersOpen(false); }}
            >
              <Newspaper size={18} /> News queue <span>{stats.total}</span>
            </button>
            <button
              className={`nav-item ${queueView === "completed" ? "active" : ""}`}
              aria-current={queueView === "completed" ? "page" : undefined}
              onClick={() => { setQueueView("completed"); setSelectedId(null); setMobileFiltersOpen(false); }}
            >
              <Check size={18} /> Completed <span>{stats.completed}</span>
            </button>
          </nav>

          <div className="sidebar-section">
            <div className="nav-label">Sentiment</div>
            {["all", "negative", "neutral", "positive", "uncertain"].map((value) => (
              <button
                key={value}
                className={`filter-item ${sentimentFilter === value ? "selected" : ""}`}
                onClick={() => { setSentimentFilter(value); setMobileFiltersOpen(false); }}
              >
                <span className={`filter-swatch ${value}`} />
                {value === "all" ? "All sentiment" : sentimentMeta[value].label}
                {sentimentFilter === value && <Check size={16} />}
              </button>
            ))}
          </div>

          <div className="sidebar-section">
            <div className="nav-label">Priority</div>
            {["all", "high", "medium", "low"].map((value) => (
              <button
                key={value}
                className={`filter-item ${priorityFilter === value ? "selected" : ""}`}
                onClick={() => { setPriorityFilter(value); setMobileFiltersOpen(false); }}
              >
                <span className={`filter-swatch priority-${value}`} />
                {value === "all" ? "All priorities" : priorityMeta[value].label}
                {priorityFilter === value && <Check size={16} />}
              </button>
            ))}
          </div>

          <div className="model-card">
            <div className="model-card-head">
              <BarChart3 size={17} />
              <strong>Active model</strong>
            </div>
            <p>{health?.model_type || "Waiting for model metadata"}{health?.device ? ` · ${health.device.toUpperCase()}` : ""}</p>
            <div className="model-metrics">
              <span>Precision <strong>{health?.metrics?.precision_macro ? `${(health.metrics.precision_macro * 100).toFixed(1)}%` : "—"}</strong></span>
              <span>Recall <strong>{health?.metrics?.recall_macro ? `${(health.metrics.recall_macro * 100).toFixed(1)}%` : "—"}</strong></span>
            </div>
          </div>
        </aside>

        {mobileFiltersOpen && <button className="sidebar-backdrop" aria-label="Close filters" onClick={() => setMobileFiltersOpen(false)} />}

        <main className="main-content">
          <div className="page-heading">
            <div>
              <p className="eyebrow">Analysis workspace</p>
              <h1>News review queue</h1>
              <p>Prioritize financial news by sentiment, relevance and event impact.</p>
            </div>
            <button className="icon-button" title="Refresh model status" onClick={refreshHealth}>
              <RefreshCw size={18} />
            </button>
          </div>

          <section className="watchlist-panel" aria-label="Watchlist news feed">
            <div className="watchlist-copy">
              <div className="watchlist-icon"><Radio size={18} /></div>
              <div>
                <div className="watchlist-title-row">
                  <h2>Watchlist news</h2>
                  <span className={`provider-state ${health ? (health.news_provider?.configured ? "connected" : "setup") : "pending"}`}>
                    <span />
                    {health
                      ? (health.news_provider?.configured ? "Finnhub connected" : "API key required")
                      : (healthError ? "Provider unavailable" : "Checking provider")}
                  </span>
                </div>
                <p>Add companies to monitoring and load every available headline into one ranked queue.</p>
              </div>
            </div>
            <div className="watchlist-controls">
              <div className="ticker-input">
                <div className="ticker-input-head">
                  <span id="watchlist-tickers-label">Tickers</span>
                  <small>{trackedTickers.length}/{MAX_WATCHLIST_TICKERS}</small>
                </div>
                <div
                  className="ticker-combobox"
                  role="group"
                  aria-labelledby="watchlist-tickers-label"
                  onClick={() => tickerInput.current?.focus()}
                >
                  {trackedTickers.map((ticker) => (
                    <span className="ticker-chip" key={ticker}>
                      <span>{ticker}</span>
                      <button
                        type="button"
                        aria-label={`Remove ${ticker}`}
                        title={`Remove ${ticker}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          removeTrackedTicker(ticker);
                        }}
                      >
                        <X size={11} aria-hidden="true" />
                      </button>
                    </span>
                  ))}
                  <input
                    ref={tickerInput}
                    className="ticker-entry-input"
                    disabled={fetchingNews}
                    value={tickerDraft}
                    onBlur={(event) => {
                      if (fetchingNews) return;
                      if (event.relatedTarget === fetchButton.current) return;
                      addWatchlistTickers(tickerDraft);
                    }}
                    onChange={(event) => setTickerDraft(event.target.value)}
                    onKeyDown={handleTickerDraftKeyDown}
                    onPaste={handleTickerPaste}
                    placeholder={trackedTickers.length ? "Add ticker" : "AAPL"}
                    aria-label="Add company ticker"
                  />
                </div>
              </div>
              <label className="period-select">
                <span>Period</span>
                <select
                  value={newsDays}
                  onChange={(event) => setNewsDays(event.target.value)}
                  aria-label="News period"
                  disabled={fetchingNews}
                >
                  <option value="1">Today</option>
                  <option value="3">3 days</option>
                  <option value="7">7 days</option>
                  <option value="14">14 days</option>
                </select>
                <ChevronDown size={15} />
              </label>
              <button
                ref={fetchButton}
                className="button primary fetch-button"
                onClick={() => fetchWatchlistNews()}
                disabled={fetchingNews || (!trackedTickers.length && !tickerDraft.trim())}
              >
                {fetchingNews ? <LoaderCircle size={17} className="spin" /> : <RefreshCw size={17} />}
                {fetchingNews ? "Loading" : "Track & fetch"}
              </button>
            </div>
          </section>

          {trackedTickers.length > 0 && (
            <section className="tracking-bar" aria-label="Tracked companies">
              <div className="tracking-heading">
                <div>
                  <span>Tracked companies</span>
                  <strong>News ranking</strong>
                </div>
                <small><BarChart3 size={14} /> Sorted by priority score</small>
              </div>
              <div className="company-tabs">
                <button
                  className={`company-filter all-companies ${companyFilter === "all" ? "active" : ""}`}
                  onClick={() => setCompanyFilter("all")}
                >
                  <strong>All</strong>
                  <span>{scopedItems.length} articles</span>
                </button>
                {companyStats.map(({ ticker, total, high, relevance }) => (
                  <div className={`company-filter-group ${companyFilter === ticker ? "active" : ""}`} key={ticker}>
                    <button className="company-filter" onClick={() => setCompanyFilter(ticker)}>
                      <strong>{ticker}</strong>
                      <span>{total} articles · {high} high</span>
                    </button>
                    <label
                      className="ticker-relevance"
                      title="Company relevance: 1 is peripheral, 5 is a core holding"
                    >
                      <span>Rel.</span>
                      <span className="ticker-relevance-control">
                        <select
                          value={relevance}
                          onChange={(event) => updateTickerRelevance(ticker, event.target.value)}
                          aria-label={`${ticker} company relevance`}
                        >
                          {[1, 2, 3, 4, 5].map((value) => <option value={value} key={value}>{value}/5</option>)}
                        </select>
                        <ChevronDown size={11} aria-hidden="true" />
                      </span>
                    </label>
                    <button className="remove-company" title={`Stop tracking ${ticker}`} onClick={() => removeTrackedTicker(ticker)}>
                      <X size={13} />
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="stats-grid" aria-label="Queue summary">
            <div className="stat-block">
              <span className="stat-label">Queue</span>
              <strong>{stats.total}</strong>
              <span className="stat-detail">items to review</span>
            </div>
            <div className="stat-block">
              <span className="stat-label">High priority</span>
              <strong>{stats.high}</strong>
              <span className="stat-detail">review immediately</span>
            </div>
            <div className="stat-block">
              <span className="stat-label">Directional</span>
              <strong>{stats.directional}</strong>
              <span className="stat-detail">positive or negative</span>
            </div>
            <div className="stat-block">
              <span className="stat-label">Completed</span>
              <strong>{stats.completed}</strong>
              <span className="stat-detail">reviewed or saved for later</span>
            </div>
          </section>

          <section className="queue-panel">
            <div className="queue-toolbar">
              <div className="search-box">
                <Search size={17} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search headline, ticker or source" />
                {query && <button className="clear-search" title="Clear search" onClick={() => setQuery("")}><X size={15} /></button>}
              </div>
              <div className="toolbar-summary">
                <Filter size={15} />
                {filteredItems.length} of {scopedItems.length}
              </div>
              <button className="icon-button" title="Export visible queue as CSV" onClick={exportCsv} disabled={!filteredItems.length}>
                <Download size={18} />
              </button>
            </div>

            {!scopedItems.length ? (
              <EmptyState
                title={queueView === "completed" ? "No completed articles yet" : (items.length ? "You're all caught up" : "Your review queue is empty")}
                description={queueView === "completed"
                  ? "Articles marked Reviewed or Review later will appear here."
                  : (items.length
                    ? "Open Completed to revisit processed articles."
                    : "Enter company tickers above and fetch news to build the ranked analyst queue.")}
              />
            ) : (
              <div className="table-wrap">
                <table className="news-table">
                  <thead>
                    <tr>
                      <th>Company</th>
                      <th>Headline</th>
                      <th>Event</th>
                      <th>Sentiment</th>
                      <th>Priority</th>
                      <th aria-label="Open" />
                    </tr>
                  </thead>
                  <tbody>
                    {filteredItems.map((item, index) => (
                      <tr
                        key={item.id}
                        className={`${selectedId === item.id ? "selected-row" : ""} priority-row-${item.priority_label}`}
                        onClick={() => setSelectedId(item.id)}
                      >
                        <td data-label="Company">
                          <div className="ticker-cell">
                            <div className="ticker-rank"><span>#{index + 1}</span><strong>{item.ticker}</strong></div>
                            <span>{formatPublishedAt(item.published_at)}</span>
                          </div>
                        </td>
                        <td data-label="Headline">
                          <div className="headline-cell">
                            <strong title={item.text}>{item.text}</strong>
                            <span>{item.source}</span>
                          </div>
                        </td>
                        <td data-label="Event">
                          <div className="event-cell">
                            <span className="event-label">{eventLabel(item.event_type)}</span>
                            {queueView === "completed" && (
                              <span className={`review-status-chip ${item.review_status}`}>
                                {reviewStatusMeta[item.review_status]}
                              </span>
                            )}
                          </div>
                        </td>
                        <td data-label="Sentiment">
                          <SentimentBadge value={item.label} />
                          <span className="confidence-text">{Math.round(item.confidence * 100)}%</span>
                        </td>
                        <td data-label="Priority">
                          <div className="priority-cell">
                            <div className="priority-cell-head">
                              <PriorityBadge value={item.priority_label} />
                              <strong>{Math.round(item.priority_score)}</strong>
                            </div>
                            <PriorityMeter value={item.priority_score} compact />
                          </div>
                        </td>
                        <td><ChevronRight size={18} className="row-chevron" /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!filteredItems.length && (
                  <div className="no-results"><Search size={22} /><strong>No matching news</strong><span>Try a different search or filter.</span></div>
                )}
              </div>
            )}
          </section>
        </main>
      </div>

      {selectedItem && (
        <DetailPanel
          item={selectedItem}
          onClose={() => setSelectedId(null)}
          onSetStatus={(status) => updateReviewStatus(selectedItem.id, status)}
        />
      )}

      {toast && (
        <div className={`toast ${toast.type}`}>
          {toast.type === "success" ? <Check size={18} /> : <AlertCircle size={18} />}
          <span>{toast.message}</span>
          <button title="Dismiss" onClick={() => setToast(null)}><X size={15} /></button>
        </div>
      )}
    </div>
  );
}

function DetailPanel({ item, onClose, onSetStatus }) {
  const drivers = item.priority_drivers || {};
  const relevanceContribution = Number(drivers.relevance_contribution || 0);
  const eventContribution = Number(drivers.event_contribution || 0);
  const interactionContribution = Number(drivers.interaction_contribution || 0);
  const scoreParts = [relevanceContribution, eventContribution];
  if (interactionContribution > 0) {
    scoreParts.push(interactionContribution);
  }
  const scoreBreakdown = `${scoreParts.map((value) => value.toFixed(0)).join(" + ")} = ${Math.round(item.priority_score)}`;
  const eventStatusMultiplier = Number(drivers.event_status_multiplier ?? 1);
  const companyDirectnessMultiplier = Number(drivers.company_directness_multiplier ?? 1);
  const eventImpact = Number(drivers.event_impact || 0);

  return (
    <>
      <button className="drawer-backdrop" aria-label="Close details" onClick={onClose} />
      <aside className="detail-drawer">
        <div className="drawer-header">
          <div>
            <span className="drawer-kicker">Analysis detail</span>
            <h2>{item.ticker}</h2>
          </div>
          <button className="icon-button" title="Close details" onClick={onClose}><X size={19} /></button>
        </div>

        <div className="drawer-body">
          <div className="detail-meta">
            <span><Clock3 size={15} /> {formatPublishedAt(item.published_at)}</span>
            <span><Newspaper size={15} /> {item.source}</span>
          </div>
          <h3 className="detail-headline">{item.text}</h3>
          {item.summary && item.summary.trim() !== item.text.trim() && (
            <div className="detail-summary">
              <span>Article summary</span>
              <p>{item.summary}</p>
            </div>
          )}

          <div className={`decision-strip priority-brief-${item.priority_label}`}>
            <div>
              <span>Sentiment</span>
              <SentimentBadge value={item.label} />
            </div>
            <div>
              <span>Priority</span>
              <PriorityBadge value={item.priority_label} />
            </div>
            <div>
              <span>Priority score</span>
              <strong>{Math.round(item.priority_score)}<small>/100</small></strong>
              <PriorityMeter value={item.priority_score} />
            </div>
          </div>

          <section className="detail-section">
            <div className="section-title"><h4>Model probabilities</h4><CircleHelp size={15} /></div>
            {Object.entries(item.probabilities).map(([label, value]) => (
              <ProbabilityBar key={label} label={label} value={value} />
            ))}
            {item.is_uncertain && (
              <div className="uncertainty-note">
                <CircleHelp size={16} />
                <div>
                  <strong>Manual sentiment review required</strong>
                  <span>
                    The model leaned {item.model_label}, but confidence was {Math.round(item.confidence * 100)}%
                    {` and the top-class margin was ${Math.round(item.confidence_margin * 100)}%`}.
                  </span>
                </div>
              </div>
            )}
          </section>

          <section className="detail-section">
            <h4>Priority context</h4>
            <div className="context-grid">
              <div><span>Company relevance</span><strong>{item.company_relevance}/5</strong></div>
              <div><span>Event type</span><strong>{eventLabel(item.event_type)}</strong></div>
              <div><span>Event status</span><strong>{contextLabel(item.event_status || "confirmed")} ×{eventStatusMultiplier.toFixed(2)}</strong></div>
              <div><span>Company match</span><strong>{contextLabel(item.company_directness || "direct")} ×{companyDirectnessMultiplier.toFixed(2)}</strong></div>
              <div><span>Final event impact</span><strong>{Math.round(eventImpact * 100)}%</strong></div>
              <div><span>Score breakdown</span><strong>{scoreBreakdown}</strong></div>
            </div>
          </section>

          <section className="recommendation">
            <div className="recommendation-icon"><SlidersHorizontal size={18} /></div>
            <div><span>Recommended workflow</span><strong>{item.recommended_action}</strong></div>
          </section>

          <section className="review-workflow">
            <div className="review-state">
              <span>Analyst status</span>
              <strong>{reviewStatusMeta[item.review_status] || "New"}</strong>
            </div>
            <div className="review-actions">
              <button className="button secondary" onClick={() => onSetStatus("reviewed")} disabled={item.review_status === "reviewed"}>
                <Check size={16} /> Mark reviewed
              </button>
              <button className="button secondary" onClick={() => onSetStatus("dismissed")} disabled={item.review_status === "dismissed"}>
                <Clock3 size={16} /> Review later
              </button>
              {["reviewed", "dismissed"].includes(item.review_status) && (
                <button className="button secondary restore-review" onClick={() => onSetStatus("new")}>
                  <RefreshCw size={16} /> Move back to queue
                </button>
              )}
            </div>
          </section>

          {item.url && (
            <a className="button secondary source-link" href={item.url} target="_blank" rel="noreferrer">
              <ExternalLink size={16} /> Open original article
            </a>
          )}
        </div>
      </aside>
    </>
  );
}

export default App;
