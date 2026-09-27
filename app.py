"""
Phase 10: MLOps & Deployment
--------------------------------
A minimal FastAPI service that forecasts next operating day's demand for
one of the top 20 products. Same important caveat as Day 6's
sales-forecasting API: this model's most important features (Lag_1,
Lag_7, Rolling_7day_mean) come from a specific product's own recent
sales HISTORY, which a real production system would look up from a
sales database at request time rather than require the caller to supply
-- this demo API accepts them as direct inputs instead, a stated
simplification (see PROJECT_DOCUMENTATION.md Section 8).

Run with:  uvicorn app:app --reload
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="E-commerce Demand Forecast API", version="1.0")

MODEL = joblib.load("outputs/demand_model.joblib")
FEATURE_COLUMNS = joblib.load("outputs/feature_columns.joblib")
TOP_SKUS = sorted([c.replace("SKU_", "") for c in FEATURE_COLUMNS if c.startswith("SKU_")])


class ForecastRequest(BaseModel):
    StockCode: str
    Date: str  # "YYYY-MM-DD" -- must not be a Saturday (the business doesn't operate that day)
    Lag_1: float  # actual quantity sold on this product's last operating day
    Lag_7: float = None  # actual quantity sold 7 days prior (falls back to Lag_1 if unknown)
    Rolling_7day_mean: float = None  # trailing 7-operating-day average (falls back to Lag_1 if unknown)


class ForecastResponse(BaseModel):
    forecast_quantity: float
    top_factors: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    options = "".join(f"<option>{sku}</option>" for sku in TOP_SKUS)
    return f"""
    <html>
    <head><title>E-commerce Demand Forecast</title></head>
    <body style="font-family: sans-serif; max-width: 600px; margin: 40px auto;">
        <h2>E-commerce Demand Forecast — Test Form</h2>
        <p>Only the top 20 products by volume are modeled — pick one below.
           <code>Lag_1</code> is required (last operating day's actual
           quantity for this product). Full API docs at <a href="/docs">/docs</a>.</p>
        <form id="forecastForm" style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px 20px;">
            <label>Product (StockCode)<br><select name="StockCode" required>{options}</select></label>
            <label>Date to forecast<br><input name="Date" type="date" value="2011-11-24" required></label>
            <label>Last operating day's actual qty — Lag_1<br><input name="Lag_1" type="number" step="0.01" value="20" required></label>
            <label>Same weekday last week's qty — Lag_7<br><input name="Lag_7" type="number" step="0.01" value="18"></label>
            <label>Trailing 7-day avg qty<br><input name="Rolling_7day_mean" type="number" step="0.01" value="19"></label>
            <button type="submit" style="grid-column: 1 / -1; margin-top: 10px;">Forecast</button>
        </form>
        <h3 id="result"></h3>
        <script>
        document.getElementById("forecastForm").addEventListener("submit", async function(e) {{
            e.preventDefault();
            const form = new FormData(e.target);
            const asFloat = (name) => parseFloat(form.get(name));
            const payload = {{
                StockCode: form.get("StockCode"),
                Date: form.get("Date"),
                Lag_1: asFloat("Lag_1"),
                Lag_7: form.get("Lag_7") ? asFloat("Lag_7") : null,
                Rolling_7day_mean: form.get("Rolling_7day_mean") ? asFloat("Rolling_7day_mean") : null
            }};
            const res = await fetch("/forecast", {{
                method: "POST",
                headers: {{"Content-Type": "application/json"}},
                body: JSON.stringify(payload)
            }});
            if (!res.ok) {{
                document.getElementById("result").innerText =
                    "Error " + res.status + ": " + await res.text();
                return;
            }}
            const data = await res.json();
            document.getElementById("result").innerText =
                "Forecast: " + data.forecast_quantity.toFixed(1) + " units" +
                " | Top factors: " + data.top_factors.join(", ");
        }});
        </script>
    </body>
    </html>
    """


@app.post("/forecast", response_model=ForecastResponse)
def forecast_demand(record: ForecastRequest):
    raw = record.model_dump()
    date = pd.to_datetime(raw["Date"])

    if date.day_name() == "Saturday":
        return ForecastResponse(forecast_quantity=0.0, top_factors=["not_an_operating_day"])

    lag_1 = raw["Lag_1"]
    lag_7 = raw["Lag_7"] if raw["Lag_7"] is not None else lag_1
    rolling_7day = raw["Rolling_7day_mean"] if raw["Rolling_7day_mean"] is not None else lag_1

    row = {
        "DayOfWeek": date.dayofweek,
        "Month": date.month,
        "IsDecember": 1 if date.month == 12 else 0,
        "Lag_1": lag_1, "Lag_7": lag_7, "Rolling_7day_mean": rolling_7day,
    }
    for sku in TOP_SKUS:
        row[f"SKU_{sku}"] = 1 if sku == raw["StockCode"] else 0

    X = pd.DataFrame([row]).reindex(columns=FEATURE_COLUMNS, fill_value=0)

    pred = float(MODEL.predict(X)[0])
    pred = max(pred, 0.0)  # demand can't be negative

    if hasattr(MODEL, "feature_importances_"):
        importances = pd.Series(MODEL.feature_importances_, index=FEATURE_COLUMNS)
        top_factors = importances.sort_values(ascending=False).head(3).index.tolist()
    else:
        top_factors = ["Lag_1", "Rolling_7day_mean", "DayOfWeek"]

    return ForecastResponse(forecast_quantity=round(pred, 2), top_factors=top_factors)
