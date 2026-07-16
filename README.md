# Financial News Sentiment MVP

This project is a small Industrial ML service for first-level financial news screening.

## How It Works

1. `app/train_model.py` trains a TF-IDF and Logistic Regression baseline.
2. `app/train_finbert.py` fine-tunes a finance-domain transformer and compares it with the baseline.
3. `app/api.py` exposes the best available model through a shared inference layer.
4. `frontend/` provides the primary React analyst dashboard.
5. `app/streamlit_app.py` remains available as a lightweight fallback interface.
6. A prioritization layer combines company relevance with event importance.

The service returns one of four investor-oriented sentiment states:

- `negative`
- `neutral`
- `positive`
- `uncertain` when the highest class probability is below `55%` or the margin
  between the two leading classes is below `10%`

For uncertain items, the original argmax class remains available as `model_label`
and the workflow recommendation is `Manual sentiment review`.

It does not directly predict stock price movement.

The detailed analyst workflow, decision policy, monitoring plan, and limitations
are documented in `docs/model_usage_scenario.md`.

## FinBERT Experiment

`app/train_finbert.py` fine-tunes `yiyanghkust/finbert-pretrain` on the same
Financial PhraseBank 75Agree training split and compares it with the baseline
on the same stratified test split.

```bash
python3 -m app.train_finbert
```

Current test results:

| Model | Macro precision | Macro recall |
|---|---:|---:|
| TF-IDF + Logistic Regression | 84.0% | 78.6% |
| Fine-tuned FinBERT | 91.5% | 92.4% |

The full comparison is in `outputs/finbert_comparison.md`, and the saved
transformer is in `models/finbert-sentiment/`. The API uses FinBERT automatically
when that artifact exists and falls back to the baseline otherwise. Set
`MODEL_BACKEND=baseline` to force the lightweight model.

The trained FinBERT weights are not committed to GitHub because the model file is
larger than GitHub's regular file limit. Recreate the local artifact with:

```bash
python3 -m app.train_finbert
```

The smaller trained baseline artifact is included so a cloned project can run
without retraining.

## Business Workflow

The model is used as a decision-support tool inside a broader analyst workflow:

1. An analyst enters company tickers in a watchlist and sets each ticker's relevance
   from `1` (peripheral) to `5` (core holding); new tickers default to `3`.
2. The backend downloads recent company news from Finnhub and removes duplicate articles.
3. The model predicts investor-oriented sentiment and confidence for each headline.
4. The service calculates a priority score using:
   - company relevance from `1` to `5`;
   - event impact based on event type, event status, and whether the headline is
     directly about the tracked company.
5. Sentiment determines the review perspective: downside risk, potential catalyst,
   or neutral event context. It does not increase the priority score.
6. Analysts review high-priority items first and open the original source when needed.

The browser keeps the watchlist relevance settings between sessions. A shared
article uses the highest relevance of the tracked companies it concerns.

Manual text analysis and CSV upload remain available as fallback workflows.

This separates materiality from interpretation. A news item about a core holding
and a high-impact event ranks above a peripheral product mention, while sentiment
determines whether the analyst checks downside risk, a catalyst, or neutral context.

## Priority Score

The MVP priority score is a simple transparent business rule:

```text
R = (company_relevance - 1) / 4

E = base_event_weight * status_multiplier * company_directness

priority = 100 * (
    0.35 * R
  + 0.50 * E
  + 0.15 * R * E
)
```

Base weights range from `1.00` for M&A to `0.25` for other news. Earnings and
legal/regulatory events use `0.85`, guidance uses `0.90`, and management,
product/partnership, and macro events use `0.65`, `0.55`, and `0.45`.

The score is not a trading signal. It is a triage score that helps analysts decide what to read first. Its weights and thresholds are assumptions that require validation in an analyst pilot.
Model confidence is displayed separately because classification certainty is not
the same as the economic importance of a news item.

Company relevance and event impact are separate, directly inspectable inputs.
The event type supplies the base weight. Confirmed, announced, forecast, and
speculative wording then applies a multiplier of `1.00`, `0.85`, `0.65`, or
`0.40`. The interaction term `R * E` boosts news only when an important company
and a material event coincide. Company Profile aliases from Finnhub classify the
headline as direct (`1.00`), broad (`0.65`), or incidental (`0.30`). Therefore
`100` requires a confirmed direct M&A item concerning a `5/5` core holding. High
priority starts at `75`; medium priority starts at `45`.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m app.train_model
```

## Run Backend

Create a free Finnhub API key and provide it to the backend process:

```bash
export FINNHUB_API_KEY="your_key_here"
```

The key is read only by FastAPI and is never sent to the browser. Without the
key, model prediction, sample data and CSV import continue to work, while the
watchlist feed shows a setup error.

```bash
uvicorn app.api:app --reload --port 8001
```

The production frontend build is served by FastAPI at:

```text
http://127.0.0.1:8001
```

FastAPI docs:

```text
http://127.0.0.1:8001/docs
```

## Develop React Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Vite development URL:

```text
http://127.0.0.1:5173
```

The frontend reads `VITE_API_URL` when it is set and otherwise connects to
`http://127.0.0.1:8001`.

The dashboard supports:

- persistent tracked-company watchlists;
- up to 50 recent Finnhub articles per requested ticker in one deduplicated queue;
- company filters, per-ticker article counts, and priority ranking;
- single-headline analysis;
- CSV import and batch prediction;
- sentiment and priority filters;
- probability and priority-driver inspection;
- CSV export of the filtered review queue.

The watchlist and review queue are saved in browser local storage. Repeated
fetches update existing articles instead of adding duplicates. A database remains
the recommended persistence layer for a multi-user deployment.

## Optional Streamlit Frontend

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

## API Example

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Operating profit rose to EUR 13.1 mn from EUR 8.7 mn.","company_relevance":4,"event_type":"earnings"}'
```
