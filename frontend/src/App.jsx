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
  FileUp,
  Filter,
  Inbox,
  LoaderCircle,
  Menu,
  Newspaper,
  Plus,
  Radio,
  RefreshCw,
  ScanSearch,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import Papa from "papaparse";

const API_URL = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8001").replace(/\/$/, "");
const QUEUE_STORAGE_KEY = "sentiment-desk-queue-v6-event-impact";
const WATCHLIST_STORAGE_KEY = "sentiment-desk-watchlist-v1";
const RELEVANCE_STORAGE_KEY = "sentiment-desk-watchlist-relevance-v1";

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

const SAMPLE_NEWS = [
  {
    ticker: "NOVA",
    source: "Sample feed",
    published_at: "09:42",
    text: "Operating profit rose to EUR 13.1 million from EUR 8.7 million.",
    company_relevance: 5,
    event_type: "earnings",
  },
  {
    ticker: "ALTO",
    source: "Sample feed",
    published_at: "09:18",
    text: "Operating loss totalled EUR 0.9 million, down from a profit of EUR 2.7 million.",
    company_relevance: 5,
    event_type: "earnings",
  },
  {
    ticker: "KITE",
    source: "Sample feed",
    published_at: "08:56",
    text: "The board appointed a new chief operating officer effective next month.",
    company_relevance: 3,
    event_type: "management",
  },
  {
    ticker: "MERA",
    source: "Sample feed",
    published_at: "08:31",
    text: "The company received regulatory approval for its new manufacturing facility.",
    company_relevance: 4,
    event_type: "legal_regulatory",
  },
  {
    ticker: "VOLT",
    source: "Sample feed",
    published_at: "08:07",
    text: "Merger negotiations are continuing and no final agreement has been signed.",
    company_relevance: 4,
    event_type: "merger_acquisition",
  },
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
  escalated: "Escalated",
  reviewed: "Reviewed",
  dismissed: "Deferred",
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

function itemFingerprint(item) {
  return item.url
    || (item.provider_id ? `provider:${item.provider_id}` : "")
    || `${String(item.ticker || "").toUpperCase()}|${String(item.text || "").trim().toLowerCase()}`;
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

function itemTickers(item) {
  if (Array.isArray(item.tickers) && item.tickers.length) return item.tickers;
  return String(item.ticker || "")
    .split(",")
    .map((ticker) => ticker.trim().toUpperCase())
    .filter(Boolean);
}

async function api(path, options = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed with status ${response.status}`);
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

function EmptyState({ onAnalyze, onSamples }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Inbox size={25} /></div>
      <h2>Your review queue is empty</h2>
      <p>Analyze a headline or import a CSV to build the queue.</p>
      <div className="empty-actions">
        <button className="button primary" onClick={onAnalyze}><Plus size={17} /> Analyze news</button>
        <button className="button secondary" onClick={onSamples}><Sparkles size={17} /> Load samples</button>
      </div>
    </div>
  );
}

function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [items, setItems] = useState(() => storedJson(QUEUE_STORAGE_KEY, []));
  const [selectedId, setSelectedId] = useState(null);
  const [query, setQuery] = useState("");
  const [sentimentFilter, setSentimentFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [companyFilter, setCompanyFilter] = useState("all");
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [importing, setImporting] = useState(false);
  const [watchlistInput, setWatchlistInput] = useState("AAPL, MSFT, NVDA");
  const [trackedTickers, setTrackedTickers] = useState(() => storedJson(WATCHLIST_STORAGE_KEY, []));
  const [tickerRelevance, setTickerRelevance] = useState(() => storedJson(RELEVANCE_STORAGE_KEY, {}));
  const [newsDays, setNewsDays] = useState(3);
  const [fetchingNews, setFetchingNews] = useState(false);
  const [toast, setToast] = useState(null);
  const fileInput = useRef(null);

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
    storeJson(WATCHLIST_STORAGE_KEY, trackedTickers);
  }, [trackedTickers]);

  useEffect(() => {
    storeJson(RELEVANCE_STORAGE_KEY, tickerRelevance);
  }, [tickerRelevance]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 3500);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const filteredItems = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const statusRank = { escalated: 3, new: 2, reviewed: 1, dismissed: 0 };
    return items
      .filter((item) => companyFilter === "all" || itemTickers(item).includes(companyFilter))
      .filter((item) => sentimentFilter === "all" || item.label === sentimentFilter)
      .filter((item) => priorityFilter === "all" || item.priority_label === priorityFilter)
      .filter((item) => {
        if (!needle) return true;
        return [item.text, item.summary, item.ticker, item.source, item.event_type]
          .some((value) => String(value || "").toLowerCase().includes(needle));
      })
      .sort((a, b) => (
        b.priority_score - a.priority_score
        || (statusRank[b.review_status] || 0) - (statusRank[a.review_status] || 0)
        || new Date(b.published_at || 0).getTime() - new Date(a.published_at || 0).getTime()
        || b.confidence - a.confidence
      ));
  }, [companyFilter, items, priorityFilter, query, sentimentFilter]);

  const companyStats = useMemo(() => trackedTickers.map((ticker) => {
    const companyItems = items.filter((item) => itemTickers(item).includes(ticker));
    return {
      ticker,
      total: companyItems.length,
      high: companyItems.filter((item) => item.priority_label === "high").length,
      relevance: Number(tickerRelevance[ticker] || 3),
    };
  }), [items, tickerRelevance, trackedTickers]);

  const stats = useMemo(() => {
    const active = items.filter((item) => !["reviewed", "dismissed"].includes(item.review_status));
    const directional = active.filter((item) => ["negative", "positive"].includes(item.label)).length;
    const high = active.filter((item) => item.priority_label === "high").length;
    const completed = items.filter((item) => ["reviewed", "dismissed"].includes(item.review_status)).length;
    return { total: active.length, directional, high, completed };
  }, [items]);

  const addPredictions = (predictions, metadata, { selectFirst = true } = {}) => {
    const timestamp = Date.now();
    const seen = new Set();
    const normalized = predictions
      .map((prediction, index) => {
        const item = {
          ...metadata[index],
          ...prediction,
          id: `${timestamp}-${index}-${Math.random().toString(36).slice(2, 7)}`,
          review_status: "new",
        };
        return { ...item, fingerprint: itemFingerprint(item) };
      })
      .filter((item) => {
        if (seen.has(item.fingerprint)) return false;
        seen.add(item.fingerprint);
        return true;
      });
    setItems((current) => {
      const existingByFingerprint = new Map(current.map((item) => [item.fingerprint, item]));
      const merged = normalized.map((item) => ({
        ...item,
        review_status: existingByFingerprint.get(item.fingerprint)?.review_status || "new",
      }));
      return [
        ...merged,
        ...current.filter((item) => !seen.has(item.fingerprint)),
      ];
    });
    if (selectFirst && normalized[0]) setSelectedId(normalized[0].id);
    return normalized;
  };

  const updateReviewStatus = (id, reviewStatus) => {
    setItems((current) => current.map((item) => (
      item.id === id ? { ...item, review_status: reviewStatus } : item
    )));
    const label = reviewStatusMeta[reviewStatus] || reviewStatus;
    setToast({ type: "success", message: `Item marked as ${label.toLowerCase()}.` });
  };

  const removeTrackedTicker = (ticker) => {
    setTrackedTickers((current) => current.filter((value) => value !== ticker));
    setTickerRelevance((current) => {
      const next = { ...current };
      delete next[ticker];
      return next;
    });
    if (companyFilter === ticker) setCompanyFilter("all");
    setToast({ type: "success", message: `${ticker} removed from the watchlist. Existing articles remain in the queue.` });
  };

  const updateTickerRelevance = (ticker, value) => {
    setTickerRelevance((current) => ({ ...current, [ticker]: Number(value) }));
    setToast({
      type: "success",
      message: `${ticker} relevance set to ${value}/5. Fetch news again to update its ranking.`,
    });
  };

  const loadSamples = async () => {
    setLoadingSamples(true);
    try {
      const response = await api("/predict_batch", {
        method: "POST",
        body: JSON.stringify({
          items: SAMPLE_NEWS.map(({ text, company_relevance, event_type }) => ({
            text,
            company_relevance,
            event_type,
          })),
        }),
      });
      addPredictions(response.results, SAMPLE_NEWS);
      setToast({ type: "success", message: "Sample queue analyzed by the model." });
    } catch (error) {
      setToast({ type: "error", message: error.message });
    } finally {
      setLoadingSamples(false);
    }
  };

  const fetchWatchlistNews = async () => {
    const tickers = [...new Set(
      watchlistInput
        .split(/[\s,;]+/)
        .map((ticker) => ticker.trim().toUpperCase())
        .filter(Boolean),
    )].slice(0, 10);
    if (!tickers.length) {
      setToast({ type: "error", message: "Enter at least one company ticker." });
      return;
    }

    setFetchingNews(true);
    setSelectedId(null);
    setTrackedTickers((current) => [...new Set([...current, ...tickers])]);
    const relevanceByTicker = Object.fromEntries(
      tickers.map((ticker) => [ticker, Number(tickerRelevance[ticker] || 3)]),
    );
    setTickerRelevance((current) => ({ ...relevanceByTicker, ...current }));
    try {
      const response = await api("/news/fetch", {
        method: "POST",
        body: JSON.stringify({
          tickers,
          days: Number(newsDays),
          limit_per_ticker: 50,
          company_relevance: relevanceByTicker,
        }),
      });
      if (!response.results.length) {
        setToast({ type: "success", message: "No company news found for this period." });
        return;
      }
      addPredictions(response.results, response.results, { selectFirst: false });
      const counts = tickers.map((ticker) => {
        const count = response.results.filter((item) => itemTickers(item).includes(ticker)).length;
        return `${ticker}: ${count}`;
      });
      setToast({
        type: "success",
        message: `${response.count} articles loaded · ${counts.join(" · ")}`,
      });
    } catch (error) {
      setToast({ type: "error", message: error.message });
    } finally {
      setFetchingNews(false);
    }
  };

  const analyzeOne = async (form) => {
    const result = await api("/predict", {
      method: "POST",
      body: JSON.stringify({
        text: form.text,
        company_relevance: Number(form.company_relevance),
        event_type: form.event_type,
      }),
    });
    const [added] = addPredictions([result], [{
      ticker: form.ticker.trim().toUpperCase() || "—",
      source: "Manual input",
      published_at: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }]);
    setAnalysisOpen(false);
    setToast({ type: "success", message: `${added.ticker} added to the review queue.` });
  };

  const importCsv = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setImporting(true);
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: async ({ data, meta, errors }) => {
        try {
          if (errors.length) throw new Error(errors[0].message);
          const columns = meta.fields || [];
          const textColumn = ["text", "sentence", "headline", "title", "news"]
            .find((name) => columns.includes(name)) || columns[0];
          if (!textColumn) throw new Error("The CSV has no columns.");
          const rows = data
            .map((row) => ({ ...row, __text: String(row[textColumn] || "").trim() }))
            .filter((row) => row.__text)
            .slice(0, 1000);
          if (!rows.length) throw new Error(`Column '${textColumn}' has no text.`);

          const response = await api("/predict_batch", {
            method: "POST",
            body: JSON.stringify({
              items: rows.map((row) => ({
                text: row.__text,
                company_relevance: Number(row.company_relevance || 3),
                event_type: row.event_type || "other",
              })),
            }),
          });
          addPredictions(response.results, rows.map((row) => ({
            ticker: String(row.ticker || row.symbol || "—").toUpperCase(),
            source: row.source || file.name,
            published_at: row.published_at || row.time || "Imported",
          })));
          setToast({ type: "success", message: `${rows.length} CSV rows analyzed using '${textColumn}'.` });
        } catch (error) {
          setToast({ type: "error", message: error.message });
        } finally {
          setImporting(false);
        }
      },
      error: (error) => {
        setImporting(false);
        setToast({ type: "error", message: error.message });
      },
    });
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
          <button className="button secondary top-import" onClick={() => fileInput.current?.click()} disabled={importing}>
            {importing ? <LoaderCircle size={17} className="spin" /> : <FileUp size={17} />}
            Import CSV
          </button>
          <button className="button primary" onClick={() => setAnalysisOpen(true)}>
            <Plus size={17} /> Analyze news
          </button>
        </div>
      </header>

      <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,text/csv" onChange={importCsv} />

      <div className="workspace">
        <aside className={`sidebar ${mobileFiltersOpen ? "mobile-open" : ""}`}>
          <div className="mobile-sidebar-head">
            <strong>Queue filters</strong>
            <button className="icon-button" title="Close filters" onClick={() => setMobileFiltersOpen(false)}><X size={19} /></button>
          </div>
          <nav className="nav-section" aria-label="Workspace">
            <div className="nav-label">Workspace</div>
            <button className="nav-item active"><Newspaper size={18} /> News queue <span>{items.length}</span></button>
            <button className="nav-item" onClick={() => setAnalysisOpen(true)}><ScanSearch size={18} /> Single analysis</button>
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
              <label className="ticker-input">
                <span>Tickers</span>
                <input
                  value={watchlistInput}
                  onChange={(event) => setWatchlistInput(event.target.value)}
                  onKeyDown={(event) => event.key === "Enter" && fetchWatchlistNews()}
                  placeholder="AAPL, MSFT, NVDA"
                  aria-label="Company tickers"
                />
              </label>
              <label className="period-select">
                <span>Period</span>
                <select value={newsDays} onChange={(event) => setNewsDays(event.target.value)} aria-label="News period">
                  <option value="1">Today</option>
                  <option value="3">3 days</option>
                  <option value="7">7 days</option>
                  <option value="14">14 days</option>
                </select>
                <ChevronDown size={15} />
              </label>
              <button className="button primary fetch-button" onClick={fetchWatchlistNews} disabled={fetchingNews}>
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
                  <span>{items.length} articles</span>
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
                      <select
                        value={relevance}
                        onChange={(event) => updateTickerRelevance(ticker, event.target.value)}
                        aria-label={`${ticker} company relevance`}
                      >
                        {[1, 2, 3, 4, 5].map((value) => <option value={value} key={value}>{value}</option>)}
                      </select>
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
              <span className="stat-detail">reviewed or deferred</span>
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
                {filteredItems.length} of {items.length}
              </div>
              <button className="icon-button" title="Export visible queue as CSV" onClick={exportCsv} disabled={!filteredItems.length}>
                <Download size={18} />
              </button>
            </div>

            {!items.length ? (
              <EmptyState onAnalyze={() => setAnalysisOpen(true)} onSamples={loadSamples} />
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
                      <tr key={item.id} className={selectedId === item.id ? "selected-row" : ""} onClick={() => setSelectedId(item.id)}>
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
                        <td data-label="Event"><span className="event-label">{eventLabel(item.event_type)}</span></td>
                        <td data-label="Sentiment">
                          <SentimentBadge value={item.label} />
                          <span className="confidence-text">{Math.round(item.confidence * 100)}%</span>
                        </td>
                        <td data-label="Priority">
                          <PriorityBadge value={item.priority_label} />
                          <span className="score-text">{Math.round(item.priority_score)}/100</span>
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

      {analysisOpen && (
        <AnalysisModal onClose={() => setAnalysisOpen(false)} onSubmit={analyzeOne} />
      )}

      {loadingSamples && (
        <div className="loading-overlay"><LoaderCircle className="spin" size={24} /><span>Analyzing sample queue…</span></div>
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

          <div className="decision-strip">
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
              <div><span>Event status</span><strong>{contextLabel(item.event_status || "confirmed")} ×{item.priority_drivers.event_status_multiplier.toFixed(2)}</strong></div>
              <div><span>Company match</span><strong>{contextLabel(item.company_directness || "direct")} ×{item.priority_drivers.company_directness_multiplier.toFixed(2)}</strong></div>
              <div><span>Final event impact</span><strong>{Math.round(item.priority_drivers.event_impact * 100)}%</strong></div>
              <div><span>Score breakdown</span><strong>{item.priority_drivers.relevance_contribution.toFixed(0)} + {item.priority_drivers.event_contribution.toFixed(0)} = {Math.round(item.priority_score)}</strong></div>
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
              <button className="button primary" onClick={() => onSetStatus("escalated")} disabled={item.review_status === "escalated"}>
                <AlertCircle size={16} /> Escalate to PM
              </button>
              <button className="button secondary" onClick={() => onSetStatus("reviewed")} disabled={item.review_status === "reviewed"}>
                <Check size={16} /> Mark reviewed
              </button>
              <button className="button secondary" onClick={() => onSetStatus("dismissed")} disabled={item.review_status === "dismissed"}>
                <X size={16} /> Defer
              </button>
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

function AnalysisModal({ onClose, onSubmit }) {
  const [form, setForm] = useState({
    ticker: "",
    text: "",
    company_relevance: 3,
    event_type: "other",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const update = (field, value) => setForm((current) => ({ ...current, [field]: value }));

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!form.text.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      await onSubmit(form);
    } catch (requestError) {
      setError(requestError.message);
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <form className="analysis-modal" onSubmit={handleSubmit}>
        <div className="modal-header">
          <div>
            <span className="drawer-kicker">New analysis</span>
            <h2>Analyze financial news</h2>
          </div>
          <button type="button" className="icon-button" title="Close" onClick={onClose}><X size={19} /></button>
        </div>

        <div className="modal-body">
          <label className="field full-width">
            <span>Headline or financial sentence</span>
            <textarea
              value={form.text}
              onChange={(event) => update("text", event.target.value)}
              placeholder="Operating profit rose to EUR 13.1 million from EUR 8.7 million."
              autoFocus
              required
            />
          </label>
          <div className="form-grid">
            <label className="field">
              <span>Company ticker</span>
              <input value={form.ticker} onChange={(event) => update("ticker", event.target.value)} placeholder="AAPL" maxLength={12} />
            </label>
            <label className="field select-field">
              <span>Event type</span>
              <select value={form.event_type} onChange={(event) => update("event_type", event.target.value)}>
                {EVENT_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
              <ChevronDown size={16} />
            </label>
          </div>
          <label className="field range-field">
            <div className="range-label"><span>Company relevance</span><strong>{form.company_relevance}/5</strong></div>
            <input type="range" min="1" max="5" step="1" value={form.company_relevance} onChange={(event) => update("company_relevance", event.target.value)} />
            <div className="range-scale"><span>Low</span><span>Core holding</span></div>
          </label>
          {error && <div className="form-error"><AlertCircle size={16} /> {error}</div>}
        </div>

        <div className="modal-footer">
          <button type="button" className="button secondary" onClick={onClose}>Cancel</button>
          <button className="button primary" disabled={submitting || !form.text.trim()}>
            {submitting ? <LoaderCircle size={17} className="spin" /> : <ScanSearch size={17} />}
            {submitting ? "Analyzing" : "Run analysis"}
          </button>
        </div>
      </form>
    </div>
  );
}

export default App;
