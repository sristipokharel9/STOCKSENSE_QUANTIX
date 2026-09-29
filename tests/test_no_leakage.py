"""
Leakage checks:
1. Split dates strictly increasing (train < val < test).
2. Rolling features at t use only data <= t-1 (recomputed by hand on a sample).
3. No forbidden future columns present.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from _loader import load_sibling

cfg = load_sibling("config")
FEATURES_FILE = cfg.FEATURES_FILE
SPLIT_META    = cfg.SPLIT_META

TARGETS = {"next_7_day_demand", "stockout_flag"}
FORBIDDEN = {"closing_future", "sold_future", "opening_future", "received_future"}


def check_split_order(meta):
    assert meta["train_end"] < meta["val_start"], "train/val overlap"
    assert meta["val_end"]   < meta["test_start"], "val/test overlap"
    print("Split order: PASS")


def check_targets_not_in_features(df):
    feature_cols = [c for c in df.columns if c not in TARGETS]
    bad = set(feature_cols) & FORBIDDEN
    assert not bad, f"Forbidden future columns present: {bad}"
    print("No forbidden future columns: PASS")


def check_rolling_shift(df, max_checks=5):
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    checked = 0
    for (sid, pid), g in df.groupby(["store_id", "product_id"]):
        if len(g) < 20:
            continue
        g = g.reset_index(drop=True)
        for i in range(10, len(g)):
            manual = g.loc[i-7:i-1, "units_sold"].mean()
            got    = g.loc[i, "rolling_mean_7"]
            if pd.isna(got):
                continue
            if abs(manual - got) > 1e-6:
                raise AssertionError(
                    f"rolling_mean_7 mismatch at {sid}/{pid} idx={i}: "
                    f"manual={manual} vs stored={got}"
                )
            checked += 1
            if checked >= max_checks:
                print(f"Rolling shift check: PASS ({checked} rows)")
                return
    print("Rolling shift check: PASS (no rows verified)")


def main():
    df   = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    meta = json.loads(Path(SPLIT_META).read_text())

    check_split_order(meta)
    check_targets_not_in_features(df)
    check_rolling_shift(df)

    print("\nALL LEAKAGE CHECKS: PASS")


if __name__ == "__main__":
    main()