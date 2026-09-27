"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: before any aggregation or modeling, an
e-commerce ops stakeholder needs to know whether the raw transaction log
can be trusted at face value -- which it can't, entirely (see the
cancellation/outlier findings below) -- and which products actually
matter enough to build a per-SKU forecast for.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()
    df["is_cancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    print("=== Cancellation check ===")
    print(f"Cancellation line items: {df['is_cancellation'].sum()} ({df['is_cancellation'].mean():.2%})")
    print(f"Of those, rows with negative quantity: {(df['is_cancellation'] & (df['Quantity'] < 0)).sum()} (should equal the total -- cancellations are always full-order reversals)")
    # WHY: confirms cancellation invoices are a clean, self-consistent
    # signal (always negative quantity) before relying on them to filter
    # anything.

    non_cancel_negative = df[(~df["is_cancellation"]) & (df["Quantity"] < 0)]
    print(f"\n=== Non-cancellation rows with negative quantity: {len(non_cancel_negative)} ===")
    print(f"Of those, missing Description: {non_cancel_negative['Description'].isnull().sum()}")
    print(f"Of those, UnitPrice == 0: {(non_cancel_negative['UnitPrice'] == 0).sum()}")
    print("-> These look like manual stock write-off/adjustment entries (no product")
    print("   description, zero price), not genuine customer returns -- handled")
    print("   explicitly in clean_and_engineer.py, not silently netted against sales.")

    print("\n=== THE OUTLIER FINDING: StockCode 23843 ===")
    sub = df[df["StockCode"] == "23843"][["InvoiceNo", "Quantity", "InvoiceDate", "Description"]]
    print(sub.to_string(index=False))
    print("-> An 80,995-unit order placed and then cancelled 12 minutes later,")
    print("   under a DIFFERENT invoice number. Filtering out only rows where")
    print("   InvoiceNo starts with 'C' does NOT catch this -- the original order")
    print("   row survives that filter untouched, inflating this SKU's demand by")
    print("   80,995 units for a sale that never actually happened. See")
    print("   clean_and_engineer.py for the quantity-cap fix and why it works here.")

    sold = df[(~df["is_cancellation"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0) & (df["Description"].notna())]
    print(f"\n=== Quantity distribution among genuine sold line items (n={len(sold)}) ===")
    print(sold["Quantity"].describe(percentiles=[.5, .9, .95, .99, .995, .999]).round(1))

    print("\n=== Top 10 products by total quantity sold, BEFORE outlier capping ===")
    print(sold.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(10))

    cap = sold["Quantity"].quantile(0.995)
    sold_capped = sold.assign(Quantity=sold["Quantity"].clip(upper=cap))
    print(f"\n=== Top 10 products by total quantity sold, AFTER capping at the 99.5th percentile ({cap:.0f} units) ===")
    top10_after = sold_capped.groupby("StockCode")["Quantity"].sum().sort_values(ascending=False).head(10)
    print(top10_after)
    print("-> Notice StockCode 23843 (the one-off outlier above) drops out of the")
    print("   top 10 entirely once capped -- confirming it was outlier-driven, not")
    print("   a genuinely popular product.")

    sold_capped["Weekday"] = sold_capped["InvoiceDate"].dt.day_name()
    print("\n=== Total quantity sold by day of week ===")
    print(sold_capped.groupby("Weekday")["Quantity"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ))
    # WHY: e-commerce demand often follows a different weekly rhythm than
    # physical retail (no Saturday shopping-trip spike) -- worth checking
    # directly rather than assuming Day 6's retail pattern carries over.

    sold_capped["Month"] = sold_capped["InvoiceDate"].dt.to_period("M")
    monthly = sold_capped.groupby("Month")["Quantity"].sum()
    print("\n=== Total quantity sold by month (the holiday ramp-up) ===")
    print(monthly)

    # ---- Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    top10_after.plot(kind="barh", ax=axes[0], color="#4C72B0", title="Top 10 products (after outlier capping)")
    axes[0].invert_yaxis()

    weekday_totals = sold_capped.groupby("Weekday")["Quantity"].sum().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    weekday_totals.plot(kind="bar", ax=axes[1], color="#55A868", title="Quantity sold by day of week")
    axes[1].tick_params(axis="x", rotation=45)

    monthly.plot(ax=axes[2], color="#C44E52", marker="o", title="Monthly quantity sold (holiday ramp-up)")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
