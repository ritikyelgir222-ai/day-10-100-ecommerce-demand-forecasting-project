# Project Documentation: E-commerce Demand Forecasting
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. Every number below came from actually running the
code in this repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Forecast next operating day's demand per top-selling product for inventory/reorder planning |
| **Client context** | E-commerce (online gift retailer) inventory/ops team |
| **Data source** | UCI Online Retail — raw invoice line items, 541,909 rows |
| **Dataset size** | Top 20 products x ~321 operating days = 6,420 daily demand rows |
| **Final model** | **Random Forest** Regressor, 26 features (chosen programmatically — see Section 5) |
| **Test WMAE** | 79.376 (vs. 98.22 for the naive baseline — a 19.2% improvement) |
| **Business framing** | Forecast accurately enough to reduce stockouts/overstock for the products that drive the most volume, with extra weight on the Nov-Dec demand ramp-up |

---

## 2. Architecture

```
data/online_retail.csv (raw invoice line items)
        │
        ▼
data_loader.py ──────► loads raw CSV, parses dates
        │
        ▼
clean_and_engineer.py ─► excludes cancellations & stock-adjustment rows,
        │                caps outlier line-item quantities, aggregates
        │                to daily demand for the top 20 products,
        │                builds lag/rolling/date features
        ▼
   split.py ───────────► CHRONOLOGICAL train/val/test split
        │
        ▼
train_model.py ───────► naive-persistence baseline, Random Forest,
        │                XGBoost; PROGRAMMATIC selection by val WMAE
        ▼
outputs/demand_model.joblib, feature_columns.joblib
        │
        ├──────────────► app.py ─── FastAPI service, /forecast endpoint
        │
        └──────────────► monitor.py ─ PSI drift check on genuinely
                                       future data + WMAE-increase
                                       retrain trigger
```

---

## 3. Data Dictionary & Cleaning Decisions

| Raw column | Description |
|---|---|
| `InvoiceNo` | Transaction ID; starts with "C" for cancellations |
| `StockCode` | Product ID |
| `Description` | Product name (1,454 rows missing) |
| `Quantity` | Units per line item (can be negative — see below) |
| `InvoiceDate` | Transaction timestamp |
| `UnitPrice` | Price per unit (2 rows negative, 2,515 rows exactly 0) |
| `CustomerID` | 135,080 rows missing (guest/wholesale checkouts) |
| `Country` | 38 countries represented |

**Cleaning decisions, each verified against the data rather than assumed:**

1. **Excluded cancellation invoices** (1.71% of rows) — verified 100% of
   these have negative quantity, confirming they represent fully
   reversed orders.
2. **Excluded 1,336 non-cancellation negative-quantity rows** — verified
   these have `UnitPrice == 0` for all of them and a missing
   `Description` for 862 of 1,336 — manual stock write-off/adjustment
   entries, not customer returns.
3. **Capped per-line-item quantity at the 99.5th percentile (160
   units)** — motivated by a specific, verified case: `StockCode 23843`
   shows an 80,995-unit order placed at 09:15 and cancelled at 09:27
   under a *different* invoice number (`C581484`, not a "C"-prefixed
   version of the original `581483`). A plain cancellation-invoice
   filter does not catch this. Capping handles the general pattern
   (one-off bulk order/cancel events) more robustly than trying to
   pair up every order with a matching cancellation by SKU/timestamp.

**Engineered features:**

| Feature | Formula | Rationale |
|---|---|---|
| `DayOfWeek`, `Month`, `IsDecember` | From `Date` | Within-week and holiday-ramp-up seasonality |
| `Lag_1` | Same product's quantity, 1 operating day prior | Strongest per-series signal |
| `Lag_7` | Same product's quantity, 7 operating days prior | Within-week seasonality — the closest equivalent this ~1-year dataset can support to Day 6's year-over-year lag |
| `Rolling_7day_mean` | Trailing 7-operating-day average, shifted to avoid leakage | Smooths single-day noise |

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| **Zero Saturday activity** | Confirmed across all 541,909 raw rows — this business does not operate on Saturdays at all |
| **The outlier order** | `StockCode 23843`: 80,995 units ordered and cancelled 12 minutes later under a different invoice number |
| **Top products shift after outlier capping** | `StockCode 23843` and `23166` (both outlier-driven) drop out of the top 10 entirely once quantities are capped — the "real" top sellers are `85099B`, `22197`, `84879`, and others |
| **Day-of-week pattern** | Thursday is the highest-volume day (1,059,107 units); Sunday the lowest among operating days (446,400) |
| **Monthly ramp-up** | Volume climbs from ~250-350K/month (Jan-Aug) to 515K (Sep), 556K (Oct), and 696K (Nov) — the holiday ramp-up. December 2011 shows only 211K, but this is because the dataset **ends December 9** — an incomplete month, not a real December drop. Worth checking before concluding anything from that number. |

---

## 5. Modeling Results

### Experiment log (validation set — Oct 16 to Nov 11, 2011)

| Model | Val WMAE | Val % within tolerance |
|---|---|---|
| Naive persistence (baseline) | 78.992 | 20.83% |
| **Random Forest** | **66.580** | 17.29% |
| XGBoost | 68.419 | 18.54% |

**Note on model selection — a genuine reversal from the rest of this
series:** in every other project, XGBoost won the final selection. Here,
`train_model.py` programmatically compares both candidates on
`val_wmae` and selects whichever is lower — **Random Forest won**, and
this held up on the held-out test set too (Random Forest: 79.376 vs.
XGBoost: 83.86, checked directly). A plausible explanation: daily
per-product demand with only ~1 year of history per series is noisier
and has less signal than the aggregated weekly store-department data in
the sales-forecasting project, where XGBoost's error-correcting boosting
approach had a clearer edge. This is a single run, not proof that Random
Forest is generally better for this problem class — but the code
reflects what actually happened rather than a scripted assumption.

### Held-out test set (Nov 13 to Dec 9, 2011 — genuinely covers the November peak)

| Metric | Value |
|---|---|
| Test WMAE | 79.376 |
| Test MAE (unweighted) | 80.429 |
| Test % within tolerance | 19.38% |
| Naive baseline test WMAE | 98.22 |
| **Improvement over naive baseline** | **19.19%** |
| N test rows | 480 |

The `% within tolerance` figures (17-21%) are noticeably lower than the
equivalent metric in other regression projects in this series (e.g. Day
6's 60-67% within ±15%). This is an honest reflection of the problem's
difficulty, not a modeling failure: **daily, per-SKU demand is
intrinsically noisier than weekly, store-level aggregates** — less data
to average over per observation, more idiosyncratic day-to-day
variation. Worth stating directly rather than picking a looser tolerance
band just to make the number look better.

### What drives the model (feature importance)

Top 5 by importance:

1. `Rolling_7day_mean` (0.3776)
2. `Lag_1` (0.1654)
3. `DayOfWeek` (0.1176)
4. `Lag_7` (0.1124)
5. `Month` (0.0729)

The three sales-history features together account for **65.5%** of the
model's decision weight — the same pattern found in Day 6 (a product's
own recent history dominates), even though the specific lag structure
had to be adapted for this dataset's shorter history.

### Business validation

The model beat the naive "no change" baseline by **19.19%** on WMAE on
480 held-out test rows. Like Day 6, **no dollar-value business case is
calculated** — this dataset has no per-unit cost or margin data, so
fabricating a dollar figure would be misleading rather than illustrative.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Test form, dropdown limited to the 20 modeled products |
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health check |
| `/forecast` | POST | Forecast a single product-day |

**A documented model limitation, found during testing, not hidden:**
feeding `Lag_1=5` for `StockCode 85099B` (a top seller that, per its own
training data, averages 93-177 units/month and rarely drops below ~20)
produced a HIGHER forecast (98 units) than a much higher, in-range
`Lag_1=45` did (88 units) — a counter-intuitive, wrong-seeming result.
This is a known failure mode of tree-based models: they don't
extrapolate reliably outside the range of values they saw during
training for a given segment. Retesting with realistic in-range values
(`Lag_1=200` vs. `Lag_1=30` for the same SKU/date) produced the expected
monotonic result (194 vs. 88 units). **Lesson for anyone re-using this
API**: only feed lag values that are plausible for the specific product
being forecast.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

- **Feature drift (PSI)**: `Lag_1`, `Rolling_7day_mean`, and `Month` all
  showed significant drift between train and test. **This is expected,
  not a problem** — the test window covers only the Nov-Dec peak season,
  during which recent-history features are naturally much higher than
  the yearly average (the same principle as Day 6's Temperature
  finding, extended here to the lag features too, since this dataset's
  "seasonal window" effect is stronger given only 1 year of data to
  average over).
- **Performance decay check**: no retrain triggered on this run (same
  test set as training evaluation).

**In production**, this should run weekly against genuinely new sales
data, and the December-drift finding above should be re-examined once a
second full holiday season of data exists to compare against.

---

## 8. Known Limitations (stated for the handover record)

1. **Limited to the top 20 products** — no attempt to forecast the long
   tail of 4,050+ other SKUs.
2. **Only ~1 year of history** — no genuine year-over-year seasonality
   signal was possible, unlike Day 6.
3. **Tree-model extrapolation limitation**, documented directly above
   with a real example that was caught during testing.
4. **No dollar-value business case** — this dataset lacks per-unit
   cost/margin data.
5. **The quantity cap (99.5th percentile) is a general-pattern fix, not
   a targeted one** — it would not catch a similarly-sized bulk
   order/cancel event that happened to fall just under the cap.
6. **Random Forest was selected over XGBoost based on a single run** —
   worth re-validating across more data or additional random seeds
   before treating this as a durable preference for this problem class.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw invoice line items, document source |
| `eda.py` | 6 | Cancellation/outlier findings, seasonality |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Cleaning, daily aggregation, feature engineering |
| `split.py` | 7 | Chronological train/val/test split |
| `train_model.py` | 8-9 | Model training, data-driven selection, business validation |
| `app.py` | 10 | FastAPI forecasting service |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
