"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the UCI "Online Retail" dataset — real, raw transaction
line items from a UK-based online gift retailer, covering every
transaction between 01/12/2010 and 09/12/2011:
    https://archive.ics.uci.edu/dataset/352/online+retail

It contains 541,909 raw invoice line items (InvoiceNo, StockCode,
Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country) —
NOT pre-aggregated into daily/weekly sales the way the Walmart dataset in
this series' sales-forecasting project (Day 6) was. Every aggregation,
cleaning decision, and outlier judgment in this project is made from
scratch against raw invoice data, which is the more realistic starting
point for e-commerce demand forecasting in practice.

If you're following along from the UCI source directly: download
`Online Retail.xlsx` from the link above, convert to CSV, and place it
at `data/online_retail.csv` — the schema is identical to the file
already included in this project.

WHY THIS DATASET (AND WHY IT'S GENUINELY DIFFERENT FROM DAY 6)
-----------------------------------------------------------------
- It's RAW transaction-level data, not pre-aggregated — the "aggregate
  raw invoices into a clean daily demand series" step is itself real
  data engineering work this project has to do (see
  clean_and_engineer.py), unlike Day 6's Walmart dataset which arrived
  already aggregated to weekly store-department sales.
- It contains genuine messy real-world data quality issues: order
  cancellations, a large share of missing CustomerID (guest/wholesale
  checkouts), and at least one dramatic one-off bulk order that was
  cancelled 12 minutes later under a DIFFERENT invoice number — which a
  naive "just exclude cancellation invoices" filter completely misses
  (see clean_and_engineer.py for how this was caught and handled).
- It's a genuine e-commerce business (not a chain of physical stores),
  matching this specific day's "e-commerce demand forecasting" framing
  more directly than Day 6's Walmart brick-and-mortar data did.
"""

import pandas as pd


def load_raw_data(path: str = "data/online_retail.csv") -> pd.DataFrame:
    # WHY encoding="latin1": the source file contains non-UTF-8 characters
    # in some product descriptions (special characters in gift/craft item
    # names) that raise a UnicodeDecodeError under the default utf-8
    # encoding -- latin1 (the encoding the original UCI file was saved in)
    # reads it correctly.
    df = pd.read_csv(path, encoding="latin1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%y %H:%M")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} raw invoice line items, {len(df.columns)} columns")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Unique products (StockCode): {df['StockCode'].nunique()}  |  Unique customers: {df['CustomerID'].nunique()}  |  Countries: {df['Country'].nunique()}")
    is_cancellation = df["InvoiceNo"].astype(str).str.startswith("C")
    print(f"\nCancellation line items: {is_cancellation.sum()} ({is_cancellation.mean():.2%})")
    print(f"Negative-quantity, NON-cancellation rows (stock adjustments -- see clean_and_engineer.py): {((~is_cancellation) & (df['Quantity'] < 0)).sum()}")
    print(f"\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
