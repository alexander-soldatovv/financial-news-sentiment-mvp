import assert from "node:assert/strict";
import test from "node:test";

import {
  itemsForQueueView,
  mergeQueueItems,
  normalizeStoredItems,
  parseTickers,
  recalculatePriorityForItem,
} from "./queueUtils.js";

function article(overrides = {}) {
  return {
    id: "article-1",
    provider_id: "1",
    ticker: "AAPL",
    tickers: ["AAPL"],
    text: "Apple reports quarterly results",
    label: "negative",
    review_status: "new",
    company_relevance: 3,
    priority_drivers: {
      event_impact: 0.85,
      formula_weight_company_relevance: 0.35,
      formula_weight_event_impact: 0.5,
      formula_weight_relevance_event_interaction: 0.15,
    },
    ...overrides,
  };
}

test("ticker input accepts common separators and removes duplicates", () => {
  assert.deepEqual(parseTickers("$aapl, msft;AAPL\nGOOG"), ["AAPL", "MSFT", "GOOG"]);
});

test("stored queue migration restores fingerprints, ids and valid review states", () => {
  const stored = normalizeStoredItems([
    article({ id: undefined, review_status: "escalated" }),
    article({ id: undefined, review_status: "broken" }),
    article({ provider_id: "2", company_directness: "incidental" }),
  ]);
  assert.equal(stored.length, 1);
  assert.ok(stored[0].id);
  assert.equal(stored[0].fingerprint, "provider:1");
  assert.equal(stored[0].review_status, "new");
});

test("refresh replaces stale articles but preserves review state for matching articles", () => {
  const current = [
    article({ review_status: "reviewed", reviewed_at: "2026-07-17T10:00:00Z" }),
    article({ id: "stale", provider_id: "stale", text: "Old AAPL story" }),
    article({ id: "goog", provider_id: "goog", ticker: "GOOG", tickers: ["GOOG"] }),
  ];
  const incoming = [article({ id: "new-id", priority_score: 90 })];
  const merged = mergeQueueItems(current, incoming, { replaceTickers: ["AAPL"] });
  assert.deepEqual(merged.map((item) => item.provider_id), ["1", "goog"]);
  assert.equal(merged[0].id, "article-1");
  assert.equal(merged[0].review_status, "reviewed");
  assert.equal(merged[0].reviewed_at, "2026-07-17T10:00:00Z");
  assert.equal(merged[0].priority_score, 90);
});

test("empty refresh clears only articles for refreshed companies", () => {
  const current = [
    article(),
    article({ id: "reviewed", provider_id: "reviewed", review_status: "reviewed" }),
    article({ id: "goog", provider_id: "goog", ticker: "GOOG", tickers: ["GOOG"] }),
  ];
  const merged = mergeQueueItems(current, [], { replaceTickers: ["AAPL"] });
  assert.deepEqual(merged.map((item) => item.provider_id), ["reviewed", "goog"]);
});

test("active and completed queue views are separated by analyst status", () => {
  const items = [
    article(),
    article({ id: "reviewed", provider_id: "reviewed", review_status: "reviewed" }),
    article({ id: "deferred", provider_id: "deferred", review_status: "dismissed" }),
  ];
  assert.deepEqual(itemsForQueueView(items, "active").map((item) => item.id), ["article-1"]);
  assert.deepEqual(
    itemsForQueueView(items, "completed").map((item) => item.id),
    ["reviewed", "deferred"],
  );
});

test("relevance changes recalculate the complete priority formula", () => {
  const updated = recalculatePriorityForItem(article(), { AAPL: 5 });
  assert.equal(updated.company_relevance, 5);
  assert.equal(updated.priority_score, 90.25);
  assert.equal(updated.priority_label, "high");
  assert.equal(updated.priority_drivers.interaction_contribution, 12.75);
});
