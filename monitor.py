"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THIS SCRIPT'S "NEW BATCH" IS GENUINELY FUTURE DATA (same principle as
Day 6, unlike the cross-sectional projects in this series): the test set
is the actual most recent 4 weeks of the dataset (Nov 11 - Dec 9, 2011),
so any drift detected here reflects a real change over time within this
~1-year window, not just an artifact of random sampling.
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, build_daily_demand, engineer_features, get_feature_columns
from split import split_data
from train_model import weighted_mae, pct_within_tolerance

WMAE_INCREASE_THRESHOLD = 0.15
DRIFT_FEATURES = ["Lag_1", "Rolling_7day_mean", "DayOfWeek", "Month"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/demand_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    daily = build_daily_demand(cleaned)
    engineered = engineer_features(daily)
    train_df, val_df, test_df = split_data(engineered)

    print("=== Feature Drift (PSI): train period vs. most-recent-4-weeks batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(train_df[feat], test_df[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY Month SHOWS UP AS DRIFTED (expected, not a problem): the test
    # window (Nov-Dec) covers only 2 of 12 months, while training spans
    # the full year -- similar to Day 6's Temperature finding, this is
    # expected seasonal coverage difference, not a data problem.

    X_test = test_df[feature_cols]
    y_test = test_df["Quantity"]
    preds = model.predict(X_test)
    wmae = weighted_mae(y_test, preds, X_test["IsDecember"])
    pct_within = pct_within_tolerance(y_test, preds)

    print(f"\n=== Performance on held-out batch ===")
    print(f"WMAE: {wmae:.3f}")
    print(f"% within tolerance: {pct_within:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_wmae = json.load(f)["test_wmae"]

    pct_increase = (wmae - baseline_wmae) / baseline_wmae
    if pct_increase > WMAE_INCREASE_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: WMAE increased {pct_increase:.1%} vs. baseline (threshold: {WMAE_INCREASE_THRESHOLD:.0%})")
    else:
        print(f"\n✅ No retrain needed (WMAE change: {pct_increase:+.1%}, threshold: {WMAE_INCREASE_THRESHOLD:.0%})")


if __name__ == "__main__":
    run_monitoring_check()
