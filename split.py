"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY A TIME-BASED SPLIT (same reasoning as Day 6's sales-forecasting
project): this is genuinely longitudinal data -- the same 20 products
observed daily over ~370 operating days -- so a random split would let
the model train on, say, October and test on August for the same
product, leaking future information backward. Trained on the past,
tested on weeks it has never seen, all after every training day.

WHY A SHORTER TEST WINDOW THAN DAY 6 (4 weeks here vs. 12 there): this
dataset covers about 1 year total, roughly a quarter of Day 6's ~2.7
years -- a 12-week test window would leave too little data to train on.
4 weeks is proportionally similar and still covers a meaningful
mix of weekdays.
"""

import pandas as pd


def split_data(df: pd.DataFrame, date_col: str = "Date", test_days: int = 28, val_days: int = 28):
    max_date = df[date_col].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_df = df[df[date_col] <= val_cutoff].copy()
    val_df = df[(df[date_col] > val_cutoff) & (df[date_col] <= test_cutoff)].copy()
    test_df = df[df[date_col] > test_cutoff].copy()

    return train_df, val_df, test_df
