# E-commerce Demand Forecasting — Full Project Code (Real Dataset)

Companion code to Day 10 of the 100-day LinkedIn series. Uses the real
UCI "Online Retail" dataset — raw UK e-commerce invoice line items, run
end to end. Every number in `PROJECT_DOCUMENTATION.md` came from
executing this code, not from an illustrative example.

## Data source

**UCI Online Retail** — 541,909 raw invoice line items from a UK-based
online gift retailer, 01/12/2010 to 09/12/2011.
[Dataset page](https://archive.ics.uci.edu/dataset/352/online+retail).

The CSV is already included at `data/online_retail.csv`.

## What's genuinely different about this project vs. Day 6

Day 6 (Walmart sales forecasting) started from data that was already
aggregated to weekly store-department sales. This project starts from
**raw invoice line items** — building a clean daily demand series is
real work here, and it surfaced genuine data-quality problems worth
reading before you trust any number below:

1. **A naive "exclude cancellation invoices" filter isn't enough.**
   `StockCode 23843` shows an 80,995-unit order placed and cancelled 12
   minutes later — under a **different invoice number**, so filtering
   out only rows starting with "C" leaves the original 80,995-unit row
   completely untouched. Fixed by capping per-line-item quantity at the
   99.5th percentile — see `clean_and_engineer.py` for why this is a
   more robust fix than trying to pair up every order with a matching
   cancellation.
2. **The business is closed on Saturdays — literally zero transactions
   occur on any Saturday across all 541,909 rows.** Saturdays are
   excluded from the daily date grid entirely, not filled with 0 (a 0
   for "we weren't open" is a different concept than a 0 for "no one
   bought this today").
3. **Only ~1 year of history exists**, so unlike Day 6's `Lag_52`
   (year-over-year), this project uses `Lag_7` (same day last week) as
   the closest available seasonality signal.
4. **A genuine model-selection reversal.** Random Forest actually beat
   XGBoost on the primary business metric (WMAE) on both validation AND
   test — the code in `train_model.py` picks whichever model wins
   programmatically rather than assuming XGBoost is always the right
   choice, and this time Random Forest won.

## A limitation worth knowing before you demo the API

Tree-based models don't extrapolate well outside a product's typical
range. Feeding the API an unrealistically low `Lag_1` for a genuine
best-seller (e.g. `Lag_1=5` for a SKU that never actually sells fewer
than ~20-30 units/day) can produce a HIGHER forecast than a more typical
low value would — the model has no training examples anywhere near that
input for that SKU/month combination, so its output there isn't
reliable. Use realistic values (see the `/` test form's defaults) — this
is documented, not hidden, in `PROJECT_DOCUMENTATION.md` Section 8.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & inspects the raw invoice line items
python eda.py                  # Phase 6 — cancellation/outlier findings, top products, seasonality + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — cleaning, daily aggregation, lag/rolling features
python train_model.py          # Phase 8-9 — naive baseline + Random Forest + XGBoost, data-driven selection, business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
        "StockCode": "85099B", "Date": "2011-11-24",
        "Lag_1": 200, "Lag_7": 180, "Rolling_7day_mean": 190
      }'
```

Expected: a forecast around **194 units** — this profile (a top seller,
peak-season date, consistently high recent demand) matches the pattern
EDA identified for the November ramp-up.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads raw invoice line items, documents source |
| `eda.py` | 6 | Cancellation/outlier investigation, top products, day-of-week and monthly seasonality |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, daily aggregation, lag/rolling features — **fully commented with reasoning** |
| `split.py` | 7 | Time-based (chronological) train/val/test split |
| `train_model.py` | 8-9 | Naive baseline, Random Forest, XGBoost — **programmatic, data-driven model selection** |
| `app.py` | 10 | FastAPI forecasting service — every form field wired |
| `monitor.py` | 13 | PSI-based drift check (on genuinely future data) + retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`, `eda_summary.png`, `experiment_log.csv`
- `business_validation.json` — measured WMAE improvement over naive baseline
- `feature_importance.csv`
- `demand_model.joblib`, `feature_columns.joblib`, `model_card.json`

## Known limitations (stated honestly, not hidden)

- **Limited to the top 20 products** by volume — the long tail of
  4,050+ other SKUs isn't modeled.
- **Only ~1 year of history** — no genuine year-over-year seasonality
  signal was possible to build, unlike Day 6's multi-year dataset.
- **Tree models don't extrapolate reliably outside a product's typical
  range** — see the warning above.
- **No per-unit cost/margin data exists** in this dataset, so (like
  Day 6) no dollar-value inventory-cost business case is calculated —
  only the measured WMAE improvement (19.2%) is reported.
- **PSI flags Lag_1, Rolling_7day_mean, and Month as "significantly
  drifted"** between train and test — expected, since the test window
  covers only the Nov-Dec peak season, not a real-world problem (see
  `monitor.py`'s comments).
