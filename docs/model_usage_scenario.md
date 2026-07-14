# Model Usage Scenario

## Purpose and user

The primary user is an equity analyst or portfolio analyst who monitors a defined
universe of companies. The service is a decision-support and triage tool. It does
not recommend trades and does not automatically change portfolio positions.

The supported decision is: **which newly published company news should the
analyst inspect first, and what type of follow-up does it require?** Sentiment is
one input to this decision, not the final decision itself.

## Current MVP workflow

1. **Define the monitoring universe.** The analyst enters up to ten stock tickers
   and selects a lookback period. Requested tickers are saved as a persistent
   browser watchlist. Each ticker has a persistent company relevance score from
   1 (peripheral) to 5 (core holding), with 3 as the default.
2. **Collect news.** FastAPI requests company news from Finnhub. The backend
   normalizes ticker symbols, rejects invalid input, removes duplicate articles,
   and keeps the newest items.
3. **Add business context.** A transparent keyword rule assigns an event type,
   such as earnings, guidance, M&A, legal/regulatory, management, or product news.
   Finnhub company news receives its ticker's watchlist relevance. If one article
   concerns several tracked companies, the highest of their relevance values is used.
4. **Classify investor-oriented sentiment.** Fine-tuned FinBERT receives the
   headline text and returns probabilities for negative, neutral, and positive.
   If the transformer artifact is unavailable, the service can use the TF-IDF and
   Logistic Regression baseline.
5. **Calculate priority.** The service combines company relevance and event
   weight into a transparent score. Sentiment determines the type of review but
   does not increase materiality.
6. **Build the review queue.** The service loads up to 50 articles per ticker,
   removes cross-company duplicates, and sorts the combined queue by priority
   score. Company tabs show article and high-priority counts. Repeated Finnhub
   refreshes update existing queue entries instead of creating duplicates.
7. **Make a human decision.** The analyst opens the source article, checks the
   event against the investment thesis and current exposure, then marks the item
   as escalated, reviewed, or deferred.
8. **Export the audit trail.** The visible queue can be exported with model output,
   priority, recommendation, and analyst status.

## Model input and output

The ML model receives one English financial headline or sentence. It does not
receive the ticker, price history, portfolio position, source reputation, event
type, or publication time.

The model returns:

- `label`: negative, neutral, positive, or uncertain from an investor perspective;
- `model_label`: the leading model class before the uncertainty rule;
- `probabilities`: one probability for each sentiment class;
- `confidence`: the largest of those probabilities.

The service returns `uncertain` and requires manual sentiment review when the
largest probability is below `55%` or the difference between the two leading
classes is below `10%`. This prevents a marginal argmax from being presented as a
decisive positive or negative classification.

Ticker relevance and event type are deliberately handled outside the ML model.
This keeps the sentiment task aligned with Financial PhraseBank and makes the
business prioritization rule inspectable.

## Priority and actions

The current heuristic is:

```text
priority = (
    60 * normalized_company_relevance
  + 40 * event_impact
)
```

The weights and thresholds are product assumptions, not learned facts. They must
be validated during a pilot with analysts.

The two terms are deliberately additive: company relevance and event impact are
different business inputs, and neither is counted twice. M&A is the only event
category with impact `1.0`; therefore `100` is reserved for M&A concerning a
`5/5` core holding. Earnings has impact `0.90`, so core-holding earnings scores
`96`. High priority starts at `75`, and medium priority starts at `45`.

Model confidence is deliberately excluded from this formula. It estimates how
certain the classifier is about its label, not how economically important the
underlying event is. The sentiment label instead selects the review perspective
shown in the action table below.

| Queue result | Analyst action | Decision supported |
|---|---|---|
| High + negative | Escalate for risk review | Check exposure, thesis risk, guidance revisions, or need for PM notification |
| High + positive | Review as a potential catalyst | Check whether the event changes forecasts, valuation, or the upside case |
| High + neutral | Review event context | Inspect a material event whose tone alone is not directional |
| Any priority + uncertain | Manual sentiment review | Read the source before assigning a directional interpretation |
| Medium | Review in the normal queue | Read after immediate items and record whether it affects the thesis |
| Low | Include in the monitoring digest | Defer full reading unless selected in a quality-control sample |

The service therefore does not assume that every positive or negative item has
the same importance. A product mention for a peripheral company can rank below a
neutral regulatory event for a core holding, regardless of model confidence.

## Human-in-the-loop controls

- No trade or portfolio action is automated.
- The original source remains available for verification.
- High-priority items are recommendations, not mandatory conclusions.
- Analysts can override the queue by escalating, completing, or deferring an item.
- A sample of low-priority items should still be reviewed to estimate missed
  relevant news and detect model drift.

## Validation and monitoring

### Offline model metrics

- macro precision and macro recall on a held-out test set;
- precision and recall for each of the three classes;
- comparison with the TF-IDF and Logistic Regression baseline.

### Workflow metrics

- median time from publication to first review for high-priority items;
- share of incoming items deferred without a full read;
- share of escalated items confirmed as relevant by an analyst;
- analyst override rate by sentiment and event type;
- low-priority quality-control miss rate.

### Business metrics

The defensible time-saving estimate is:

```text
monthly_value = working_days * deferred_items_per_day
                * manual_hours_per_item * analyst_hourly_cost
                - monthly_model_cost
```

`deferred_items_per_day` must be measured from actual usage. It is not equal to
all news processed by the model because analysts still read high- and
medium-priority items.

Business value beyond time savings should be evaluated through faster review of
material events, broader watchlist coverage, and fewer relevant events discovered
late. These are pilot metrics; the current dataset cannot prove portfolio return.

## Current limitations

- The MVP fetch is triggered manually; there is no scheduled ingestion worker.
- Watchlist and queue state are persisted in browser local storage, not in a
  shared database.
- FinBERT classifies the headline only; it does not estimate price impact.
- Event type is a keyword heuristic and can be wrong.
- Company relevance is a manual watchlist assumption and can become stale as the
  portfolio or analyst coverage changes.
- Model probabilities have not been calibrated, so the priority thresholds need
  pilot validation.
- Financial PhraseBank has no ticker, event, source, or price-reaction fields.

## Production extension

The next production iteration should add scheduled ingestion, server-side watchlist
weights, a database-backed queue, analyst feedback capture, probability
calibration, model/version logging, and monitoring dashboards. Feedback can then
be used to train a separate relevance or materiality model; sentiment alone should
not be stretched into that role.
