const DEFAULT_PRIORITY_FORMULA_WEIGHTS = {
  relevance: 0.35,
  event: 0.5,
  interaction: 0.15,
};

const REVIEW_STATUSES = new Set(["new", "reviewed", "dismissed"]);

export function isCompletedItem(item) {
  return ["reviewed", "dismissed"].includes(item.review_status);
}

export function itemsForQueueView(items, queueView) {
  return items.filter((item) => (
    queueView === "completed" ? isCompletedItem(item) : !isCompletedItem(item)
  ));
}

export function itemFingerprint(item) {
  return item.url
    || (item.provider_id ? `provider:${item.provider_id}` : "")
    || `${String(item.ticker || "").toUpperCase()}|${String(item.text || "").trim().toLowerCase()}`;
}

export function itemTickers(item) {
  if (Array.isArray(item.tickers) && item.tickers.length) {
    return [...new Set(item.tickers.map((ticker) => String(ticker).trim().toUpperCase()).filter(Boolean))];
  }
  return [...new Set(
    String(item.ticker || "")
      .split(",")
      .map((ticker) => ticker.trim().toUpperCase())
      .filter(Boolean),
  )];
}

export function parseTickers(value) {
  return [...new Set(
    String(value || "")
      .split(/[\s,;]+/)
      .map((ticker) => ticker.trim().replace(/^\$/, "").toUpperCase())
      .filter(Boolean),
  )];
}

export function clampRelevance(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 3;
  return Math.max(1, Math.min(5, numeric));
}

export function normalizeRelevanceMap(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return Object.fromEntries(
    Object.entries(value)
      .map(([ticker, relevance]) => [String(ticker).trim().toUpperCase(), clampRelevance(relevance)])
      .filter(([ticker]) => ticker),
  );
}

export function normalizeStoredItems(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  return value
    .filter((item) => item && typeof item === "object" && item.company_directness !== "incidental")
    .map((item, index) => {
      const fingerprint = itemFingerprint(item);
      const reviewStatus = item.review_status === "escalated" ? "new" : item.review_status;
      return {
        ...item,
        id: item.id || `stored-${index}-${fingerprint}`,
        fingerprint,
        review_status: REVIEW_STATUSES.has(reviewStatus) ? reviewStatus : "new",
      };
    })
    .filter((item) => {
      if (seen.has(item.fingerprint)) return false;
      seen.add(item.fingerprint);
      return true;
    });
}

function roundedPriorityScore(value) {
  return Math.round(value * 100) / 100;
}

function numericOr(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function priorityLabelFromScore(score) {
  if (score >= 75) return "high";
  if (score >= 45) return "medium";
  return "low";
}

function recommendedActionFor(label, score) {
  if (label === "uncertain") return "Manual sentiment review";
  if (score >= 75) {
    if (label === "negative") return "Escalate for risk review";
    if (label === "positive") return "Review as a potential catalyst";
    return "Review event context";
  }
  if (score >= 45) return "Review in the analyst queue";
  return "Include in the monitoring digest";
}

function relevanceForItem(item, relevanceByTicker) {
  const values = itemTickers(item)
    .filter((ticker) => Object.prototype.hasOwnProperty.call(relevanceByTicker, ticker))
    .map((ticker) => relevanceByTicker[ticker])
    .map(Number)
    .filter(Number.isFinite);
  if (values.length) return clampRelevance(Math.max(...values));
  return clampRelevance(item.company_relevance || 3);
}

export function recalculatePriorityForItem(item, relevanceByTicker) {
  const drivers = item.priority_drivers || {};
  const companyRelevance = relevanceForItem(item, relevanceByTicker);
  const normalizedRelevance = (companyRelevance - 1) / 4;
  const eventImpact = numericOr(
    drivers.event_impact
      ?? drivers.event_weight
      ?? (
        numericOr(drivers.base_event_weight, 0)
        * numericOr(drivers.event_status_multiplier, 1)
        * numericOr(drivers.company_directness_multiplier, 1)
      ),
    0,
  );
  const relevanceEventInteraction = normalizedRelevance * eventImpact;
  const relevanceWeight = numericOr(
    drivers.formula_weight_company_relevance ?? DEFAULT_PRIORITY_FORMULA_WEIGHTS.relevance,
    DEFAULT_PRIORITY_FORMULA_WEIGHTS.relevance,
  );
  const eventWeight = numericOr(
    drivers.formula_weight_event_impact ?? DEFAULT_PRIORITY_FORMULA_WEIGHTS.event,
    DEFAULT_PRIORITY_FORMULA_WEIGHTS.event,
  );
  const interactionWeight = numericOr(
    drivers.formula_weight_relevance_event_interaction ?? DEFAULT_PRIORITY_FORMULA_WEIGHTS.interaction,
    DEFAULT_PRIORITY_FORMULA_WEIGHTS.interaction,
  );
  const relevanceContribution = 100 * relevanceWeight * normalizedRelevance;
  const eventContribution = 100 * eventWeight * eventImpact;
  const interactionContribution = 100 * interactionWeight * relevanceEventInteraction;
  const priorityScore = roundedPriorityScore(
    relevanceContribution + eventContribution + interactionContribution,
  );

  return {
    ...item,
    company_relevance: companyRelevance,
    priority_score: priorityScore,
    priority_label: priorityLabelFromScore(priorityScore),
    recommended_action: recommendedActionFor(item.label, priorityScore),
    priority_drivers: {
      ...drivers,
      company_relevance: companyRelevance,
      normalized_company_relevance: Math.round(normalizedRelevance * 10000) / 10000,
      event_impact: Math.round(eventImpact * 10000) / 10000,
      event_weight: eventImpact,
      relevance_event_interaction: Math.round(relevanceEventInteraction * 10000) / 10000,
      formula_weight_company_relevance: relevanceWeight,
      formula_weight_event_impact: eventWeight,
      formula_weight_relevance_event_interaction: interactionWeight,
      relevance_contribution: roundedPriorityScore(relevanceContribution),
      event_contribution: roundedPriorityScore(eventContribution),
      interaction_contribution: roundedPriorityScore(interactionContribution),
    },
  };
}

export function mergeQueueItems(currentItems, incomingItems, { replaceTickers = [] } = {}) {
  const current = normalizeStoredItems(currentItems);
  const existingByFingerprint = new Map(current.map((item) => [item.fingerprint, item]));
  const refreshedTickers = new Set(replaceTickers.map((ticker) => String(ticker).toUpperCase()));
  const seen = new Set();
  const incoming = normalizeStoredItems(incomingItems).map((item) => {
    const existing = existingByFingerprint.get(item.fingerprint);
    seen.add(item.fingerprint);
    return {
      ...item,
      id: existing?.id || item.id,
      review_status: existing?.review_status || item.review_status,
      reviewed_at: existing?.reviewed_at || item.reviewed_at,
    };
  });
  const retained = current.filter((item) => (
    !seen.has(item.fingerprint)
    && (
      isCompletedItem(item)
      || !itemTickers(item).some((ticker) => refreshedTickers.has(ticker))
    )
  ));
  return [...incoming, ...retained];
}
