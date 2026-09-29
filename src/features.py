"""
Phase 3 - Feature engineering, targets, time-based split.
All rolling features are computed on shift(1) so day t only sees <= t-1.

Stock-out label uses the near-stock-out fallback because the synthetic
dataset rarely shows true zero-stock events.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
MASTER_DATASET = cfg.MASTER_DATASET
FEATURES_FILE  = cfg.FEATURES_FILE
SPLIT_META     = cfg.SPLIT_META
TRAIN_FRAC     = cfg.TRAIN_FRAC
VAL_FRAC       = cfg.VAL_FRAC
TEST_FRAC      = cfg.TEST_FRAC
GAP_DAYS       = cfg.GAP_DAYS
SEED           = cfg.SEED

np.random.seed(SEED)


def _group_roll(series, grouper_keys, window, func, min_periods):
    """Rolling within (store, product) groups. Caller must pass shift(1) data."""
    return (series.groupby(grouper_keys)
                  .rolling(window, min_periods=min_periods)
                  .agg(func)
                  .reset_index(level=[0, 1], drop=True))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    keys = [df["store_id"], df["product_id"]]
    g = df.groupby(["store_id", "product_id"], group_keys=False)

    # ---- Time ----
    df["day_of_week"]   = df["date"].dt.dayofweek
    df["weekend_flag"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]         = df["date"].dt.month
    df["week_no"]       = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_month"]  = df["date"].dt.day
    df["is_month_end"]  = df["date"].dt.is_month_end.astype(int)
    df["festival_flag"] = df.get("festival", 0)
    df["holiday_flag"]  = df.get("holiday", 0)

    # ---- Lags of units_sold ----
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = g["units_sold"].shift(lag)

    # ---- Rolling on shift(1) ----
    s1 = df.groupby(["store_id", "product_id"])["units_sold"].shift(1)
    df["rolling_mean_7"]  = _group_roll(s1, keys, 7,  "mean", 3)
    df["rolling_mean_14"] = _group_roll(s1, keys, 14, "mean", 5)
    df["rolling_mean_28"] = _group_roll(s1, keys, 28, "mean", 7)
    df["rolling_std_7"]   = _group_roll(s1, keys, 7,  "std",  3)
    df["rolling_std_28"]  = _group_roll(s1, keys, 28, "std",  7)
    df["rolling_max_7"]   = _group_roll(s1, keys, 7,  "max",  3)
    df["sales_growth_7"]  = df["rolling_mean_7"] / df["rolling_mean_14"].replace(0, np.nan) - 1

    # ---- Inventory features (start-of-day only) ----
    df["prev_day_closing"]          = g["closing"].shift(1)
    df["days_of_inventory"]         = df["closing"] / df["rolling_mean_7"].replace(0, np.nan)
    df["inventory_to_demand_ratio"] = df["closing"] / df["rolling_mean_7"].replace(0, np.nan)
    df["reorder_gap"]               = df["closing"] - df["reorder_lvl"]
    df["stock_cover_days"]          = df["days_of_inventory"]

    # ---- Price / promo ----
    prev_price = df.groupby(["store_id", "product_id"])["avg_selling_price"].shift(1)
    df["price_change"] = df["avg_selling_price"] / prev_price.replace(0, np.nan) - 1

    promo_shift = df.groupby(["store_id", "product_id"])["promotion_flag"].shift(1)
    df["promo_last_7d_count"] = _group_roll(promo_shift.fillna(0), keys, 7, "sum", 1)
    df["price_vs_mrp"] = df["avg_selling_price"] / df["mrp"].replace(0, np.nan)

    # ---- Target 1: next 7-day demand ----
    future_units = df.groupby(["store_id", "product_id"])["units_sold"].shift(-1)
    df["next_7_day_demand"] = (
        future_units.groupby([df["store_id"], df["product_id"]])
                    .rolling(7, min_periods=7)
                    .sum()
                    .reset_index(level=[0, 1], drop=True)
    )

    # ---- Target 2: stock-out in next 7 days ----
    # True stock-out event
    df["available"]        = df["opening"] + df["received"]
    df["true_stockout_day"] = ((df["closing"] <= 0) |
                               (df["sold"] >= df["available"])).astype(int)

    # Near stock-out fallback (used when true events are essentially absent)
    expected_next_day = df["rolling_mean_7"].fillna(df["units_sold"])
    df["near_stockout_day"] = (
        (df["closing"] < 0.20 * df["available"])
        | (df["closing"] < 1.5 * expected_next_day)
    ).astype(int)

    # Choose: if a (store, product) had any true event in its history, use true;
    # otherwise use near-stock-out. Compute per group.
    grp = df.groupby(["store_id", "product_id"])
    true_any = grp["true_stockout_day"].transform("sum") > 0
    df["stockout_day"] = np.where(true_any,
                                  df["true_stockout_day"],
                                  df["near_stockout_day"]).astype(int)

    # Target: stock-out at any point in next 7 days
    future_stockout = df.groupby(["store_id", "product_id"])["stockout_day"].shift(-1)
    df["stockout_flag"] = (
        future_stockout.groupby([df["store_id"], df["product_id"]])
                       .rolling(7, min_periods=7)
                       .max()
                       .reset_index(level=[0, 1], drop=True)
    )

    return df


def time_split(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    dates = np.array(sorted(df["date"].unique()))
    n = len(dates)
    tr_end = int(n * TRAIN_FRAC)
    va_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    train_end_date  = dates[tr_end - 1]
    val_end_date    = dates[va_end - 1]
    val_start_date  = dates[min(tr_end + GAP_DAYS, n - 1)]
    test_start_date = dates[min(va_end + GAP_DAYS, n - 1)]

    train = df[df["date"] <= train_end_date]
    val   = df[(df["date"] >= val_start_date) & (df["date"] <= val_end_date)]
    test  = df[df["date"] >= test_start_date]

    meta = {
        "train_end":  str(pd.Timestamp(train_end_date).date()),
        "val_start":  str(pd.Timestamp(val_start_date).date()),
        "val_end":    str(pd.Timestamp(val_end_date).date()),
        "test_start": str(pd.Timestamp(test_start_date).date()),
        "gap_days":   GAP_DAYS,
        "train_rows": int(len(train)),
        "val_rows":   int(len(val)),
        "test_rows":  int(len(test)),
    }
    return train, val, test, meta


def main():
    df = pd.read_csv(MASTER_DATASET)
    df = build_features(df)

    df = df.dropna(subset=["next_7_day_demand", "stockout_flag"]).reset_index(drop=True)
    df["stockout_flag"] = df["stockout_flag"].astype(int)

    df.to_csv(FEATURES_FILE, index=False)

    train, val, test, meta = time_split(df)
    Path(SPLIT_META).write_text(json.dumps(meta, indent=2))

    print("features.csv saved:", len(df), "rows")
    print("class balance stockout_flag (overall):",
          round(float(df["stockout_flag"].mean()), 4))
    print("class balance stockout_flag (train):  ",
          round(float(train["stockout_flag"].mean()), 4))
    print("class balance stockout_flag (val):    ",
          round(float(val["stockout_flag"].mean()), 4))
    print("class balance stockout_flag (test):   ",
          round(float(test["stockout_flag"].mean()), 4))
    print("split meta:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()