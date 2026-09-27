"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it. This project's Phase 5 work is unusually substantial
compared to the rest of this series, because the raw data is invoice
LINE ITEMS, not a pre-aggregated daily sales table -- building a clean
daily demand series is real work here, not a given.

FINAL FEATURE LIST is built at the bottom as FEATURE_COLUMNS.
"""

import numpy as np
import pandas as pd

TOP_N_PRODUCTS = 20
QUANTITY_CAP_PERCENTILE = 0.995


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: exclude cancellation invoices (InvoiceNo starts with "C").
    # WHY: verified in eda.py that 100% of cancellation rows have
    # negative quantity and represent a fully reversed order -- these
    # were never fulfilled demand and shouldn't count as sales.
    # -----------------------------------------------------------------
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    df = df[~is_cancellation]

    # -----------------------------------------------------------------
    # Fix 2: exclude the remaining negative-quantity, zero-price,
    # missing-description rows (manual stock adjustments/write-offs).
    # WHY: verified in eda.py these 1,336 rows have UnitPrice == 0 for
    # ALL of them and a missing Description for the large majority --
    # not a customer purchase or return, an internal inventory
    # adjustment entry. Netting these against real sales would distort
    # the demand signal for reasons that have nothing to do with
    # customer demand.
    # -----------------------------------------------------------------
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df["Description"].notna()]

    # -----------------------------------------------------------------
    # Fix 3: cap per-line-item Quantity at the 99.5th percentile (160
    # units).
    # WHY THIS SPECIFIC FIX, NOT A DIFFERENT ONE: EDA found a concrete,
    # verified case (StockCode 23843) where an 80,995-unit order was
    # placed and cancelled 12 minutes later under a DIFFERENT invoice
    # number -- which Fix 1 above does NOT catch, since the original
    # order row has its own, non-"C" invoice number and survives that
    # filter untouched. Rather than write bespoke matching logic to pair
    # up every order with a same-SKU, same-customer, near-simultaneous
    # cancellation (fragile and likely to miss other cases), capping
    # extreme per-line-item quantities is a simpler, more robust fix for
    # the general PATTERN this case represents: one-off bulk
    # orders/cancellations that don't reflect steady product demand,
    # whatever their eventual invoice history looks like.
    # -----------------------------------------------------------------
    cap = df["Quantity"].quantile(QUANTITY_CAP_PERCENTILE)
    df["Quantity"] = df["Quantity"].clip(upper=cap)

    return df


def build_daily_demand(clean_df: pd.DataFrame, top_n: int = TOP_N_PRODUCTS) -> pd.DataFrame:
    """
    Aggregates cleaned invoice line items into a daily demand series for
    the top N products by total (post-cleaning) quantity sold.
    WHY TOP N PRODUCTS, NOT ALL 4,070: the long tail of rarely-sold SKUs
    doesn't have enough daily volume to forecast meaningfully (most days
    would be 0), and a real ops team would prioritize forecasting effort
    on the products that actually drive stockout/overstock risk. This
    mirrors the same "focus where it matters" scoping judgment used
    throughout this series.
    """
    df = clean_df.copy()
    df["Date"] = df["InvoiceDate"].dt.normalize()

    top_products = df.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(top_n).index
    df = df[df["StockCode"].isin(top_products)]

    daily = df.groupby(["StockCode", "Date"])["Quantity"].sum().reset_index()

    # -----------------------------------------------------------------
    # WHY WE EXCLUDE SATURDAYS FROM THE DATE GRID ENTIRELY (rather than
    # filling them with 0 the way an ordinary "no sales that day" gap
    # would be handled): EDA confirmed ZERO transactions occurred on any
    # Saturday across the entire 541,909-row raw dataset -- this
    # business does not operate that day at all. A 0 for a genuine
    # non-operating day is a different concept than a 0 for a slow
    # business day, and including it would teach the model a fake
    # "Saturday is always zero demand" pattern that's actually "Saturday
    # doesn't exist as a sales day" -- the date grid below reflects that
    # directly rather than papering over it with a misleading zero.
    # -----------------------------------------------------------------
    full_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    full_dates = full_dates[full_dates.day_name() != "Saturday"]

    grid = pd.MultiIndex.from_product([top_products, full_dates], names=["StockCode", "Date"]).to_frame(index=False)
    daily = grid.merge(daily, on=["StockCode", "Date"], how="left")
    # WHY FILL REMAINING GAPS WITH 0: for a genuine operating day with no
    # matching aggregated row, the correct interpretation IS zero units
    # sold that day for that product -- unlike the Saturday case above,
    # this reflects real (if slow) demand, not a non-operating day.
    daily["Quantity"] = daily["Quantity"].fillna(0)

    return daily


def engineer_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df = df.sort_values(["StockCode", "Date"])

    # -----------------------------------------------------------------
    # Date features
    # BUSINESS LOGIC: DayOfWeek and Month let the model learn the
    # within-week and holiday-ramp-up seasonality EDA found directly.
    # WHY NOT A YEAR-OVER-YEAR LAG (unlike Day 6's Lag_52): this dataset
    # spans just over one year, so there is no genuine "same week last
    # year" history available for most of the series -- a meaningful
    # difference in what's possible here versus the multi-year Walmart
    # dataset in the sales-forecasting project.
    # -----------------------------------------------------------------
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsDecember"] = (df["Month"] == 12).astype(int)

    # -----------------------------------------------------------------
    # Lag & rolling features
    # WHY Lag_1 and Lag_7 (not Lag_52): Lag_7 (same day of week, one
    # week prior) is the closest equivalent this dataset can support to
    # Day 6's year-over-year lag -- it captures within-week seasonality
    # (e.g. "Thursdays are consistently busier") at a timescale this
    # ~1-year dataset can actually estimate reliably, unlike a 52-week
    # lag which would have almost no repeated history to learn from.
    # -----------------------------------------------------------------
    grp = df.groupby("StockCode")["Quantity"]
    df["Lag_1"] = grp.shift(1)
    df["Lag_7"] = grp.shift(7)
    df["Rolling_7day_mean"] = df.groupby("StockCode")["Quantity"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )

    # WHY FILL MISSING LAGS WITH THE SERIES' OWN MEAN (not 0): the first
    # few days of any product's history have no prior day/week to look
    # back on -- filling with 0 would look like "predict no demand,"
    # which is a misleading signal for a product that's clearly an
    # active top seller. The series' own mean is a more reasonable
    # "no history yet" default, the same reasoning used in the
    # sales-forecasting project.
    series_mean = df.groupby("StockCode")["Quantity"].transform("mean")
    df["Lag_1"] = df["Lag_1"].fillna(series_mean)
    df["Lag_7"] = df["Lag_7"].fillna(df["Lag_1"])
    df["Rolling_7day_mean"] = df["Rolling_7day_mean"].fillna(df["Lag_1"])

    df = pd.get_dummies(df, columns=["StockCode"], prefix="SKU")
    dummy_cols = [c for c in df.columns if c.startswith("SKU_")]
    df[dummy_cols] = df[dummy_cols].astype(int)

    return df


NUMERIC_MODEL_FEATURES = ["DayOfWeek", "Month", "IsDecember", "Lag_1", "Lag_7", "Rolling_7day_mean"]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if c.startswith("SKU_")]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    print(f"Raw rows: {len(raw)}  ->  Cleaned (real sales) rows: {len(cleaned)}")

    daily = build_daily_demand(cleaned)
    print(f"Daily demand rows (top {TOP_N_PRODUCTS} products x operating days): {len(daily)}")

    engineered = engineer_features(daily)
    feature_cols = get_feature_columns(engineered)
    print(f"\nFinal feature columns ({len(feature_cols)}):")
    for c in feature_cols[:10]:
        print(f"  - {c}")
    print(f"  ... and {len(feature_cols) - 10} more (one SKU-dummy each)")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
