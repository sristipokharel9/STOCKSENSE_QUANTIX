"""
Phase 4 - Model 1: 7-day demand forecast.

Baselines:
  - naive_mean : rolling_mean_7 * 7
  - naive_lag  : lag_7 * 7

Models compared (permitted algorithms only):
  Linear Regression, Decision Tree, Random Forest, XGBoost, KNN, SVR.

Time split loaded from split_meta.json (no random split).
Outputs:
  reports/model_comparison_forecast.csv
  models/demand_model.joblib
  models/demand_model_meta.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
FEATURES_FILE   = cfg.FEATURES_FILE
SPLIT_META      = cfg.SPLIT_META
DEMAND_MODEL    = cfg.DEMAND_MODEL
DEMAND_META     = cfg.DEMAND_META
FORECAST_CMP    = cfg.FORECAST_CMP
SEED            = cfg.SEED

TARGET = "next_7_day_demand"

# Columns to drop from features
DROP_COLS = ["date", "next_7_day_demand", "stockout_flag", "stockout_day",
             "available", "revenue"]

# Explicitly mark these as categorical, even if they look numeric
# (store_id, product_id may be read as strings or ints depending on source)
FORCE_CATEGORICAL = ["store_id", "product_id", "category", "store_type",
                     "region", "city", "sub_category", "brand", "supplier_id"]


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def mape(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = a > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def wape(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - b)) / s * 100) if s else float("nan")


def evaluate(name, y_true, y_pred, t_train=None):
    return {
        "model": name,
        "MAE":   round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "WAPE":  round(wape(y_true, y_pred), 4),
        "R2":    round(r2_score(y_true, y_pred), 4),
        "train_time_sec": round(t_train, 2) if t_train is not None else np.nan,
    }


def split_feature_columns(df):
    """Return (feature_cols, num_cols, cat_cols) using dtype inspection
    plus a forced categorical list. This makes it impossible for a string
    column like 'Snacks' to end up in the numeric pipeline."""
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    num_cols, cat_cols = [], []
    for c in feature_cols:
        if c in FORCE_CATEGORICAL:
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return feature_cols, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def main():
    print("=" * 72)
    print("PHASE 4 - MODEL 1: 7-DAY DEMAND FORECAST")
    print("=" * 72)

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    meta = json.loads(Path(SPLIT_META).read_text())

    train = df[df["date"] <= pd.Timestamp(meta["train_end"])].copy()
    val   = df[(df["date"] >= pd.Timestamp(meta["val_start"])) &
               (df["date"] <= pd.Timestamp(meta["val_end"]))].copy()
    test  = df[df["date"] >= pd.Timestamp(meta["test_start"])].copy()

    print(f"train rows={len(train):,}  val rows={len(val):,}  test rows={len(test):,}")

    feature_cols, num_cols, cat_cols = split_feature_columns(df)
    print(f"feature columns: {len(feature_cols)}  "
          f"(numeric={len(num_cols)}, categorical={len(cat_cols)})")
    print(f"categorical: {cat_cols}")

    X_tr, y_tr = train[feature_cols], train[TARGET].values
    X_va, y_va = val[feature_cols],   val[TARGET].values
    X_te, y_te = test[feature_cols],  test[TARGET].values

    results = []

    # ---- baselines ----
    if "rolling_mean_7" in val.columns:
        base_val  = (val["rolling_mean_7"].fillna(0) * 7).values
        results.append(evaluate("baseline_rolling_mean_7_x7", y_va, base_val))
        print("baseline_rolling_mean_7_x7: val MAE =", results[-1]["MAE"])

    if "lag_7" in val.columns:
        base_val2 = (val["lag_7"].fillna(0) * 7).values
        results.append(evaluate("baseline_lag_7_x7", y_va, base_val2))
        print("baseline_lag_7_x7: val MAE =", results[-1]["MAE"])

    # ---- models ----
    pre = build_preprocessor(num_cols, cat_cols)

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree":     DecisionTreeRegressor(random_state=SEED, max_depth=12),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=14,
                                                  n_jobs=-1, random_state=SEED),
        "XGBoost":          XGBRegressor(n_estimators=300, max_depth=6,
                                         learning_rate=0.08, subsample=0.9,
                                         colsample_bytree=0.9,
                                         random_state=SEED, n_jobs=-1,
                                         tree_method="hist"),
        "KNN":              KNeighborsRegressor(n_neighbors=15, n_jobs=-1),
        "SVR":              SVR(C=1.0, epsilon=1.0),
    }

    best_name, best_mae, best_model = None, np.inf, None

    for name, est in models.items():
        print(f"\n-- {name}")
        if name in {"KNN", "SVR"} and len(X_tr) > 20000:
            idx = np.random.RandomState(SEED).choice(len(X_tr), 20000, replace=False)
            Xt, yt = X_tr.iloc[idx], y_tr[idx]
            print(f"   subsampled train to {len(Xt):,} rows")
        else:
            Xt, yt = X_tr, y_tr

        pipe = Pipeline([("pre", pre), ("model", est)])
        t0 = time.time()
        pipe.fit(Xt, yt)
        t_train = time.time() - t0

        pred_val  = pipe.predict(X_va)
        pred_test = pipe.predict(X_te)

        row = evaluate(name, y_va, pred_val, t_train)
        row["test_MAE"]  = round(mean_absolute_error(y_te, pred_test), 4)
        row["test_RMSE"] = round(rmse(y_te, pred_test), 4)
        row["test_MAPE"] = round(mape(y_te, pred_test), 4)
        row["test_WAPE"] = round(wape(y_te, pred_test), 4)
        row["test_R2"]   = round(r2_score(y_te, pred_test), 4)
        results.append(row)

        print(f"   val  MAE={row['MAE']:.3f}  R2={row['R2']:.3f}")
        print(f"   test MAE={row['test_MAE']:.3f}  R2={row['test_R2']:.3f}  "
              f"time={row['train_time_sec']}s")

        if row["MAE"] < best_mae:
            best_mae   = row["MAE"]
            best_name  = name
            best_model = pipe

    cmp = pd.DataFrame(results)
    cmp.to_csv(FORECAST_CMP, index=False)
    print("\nmodel comparison saved:", FORECAST_CMP)
    print(cmp.to_string(index=False))

    # ---- refit best on train+val, evaluate once on test ----
    print(f"\nBest model on validation: {best_name} (MAE={best_mae:.3f})")
    X_trva = pd.concat([X_tr, X_va])
    y_trva = np.concatenate([y_tr, y_va])

    if best_name in {"KNN", "SVR"} and len(X_trva) > 20000:
        idx = np.random.RandomState(SEED).choice(len(X_trva), 20000, replace=False)
        X_fit, y_fit = X_trva.iloc[idx], y_trva[idx]
    else:
        X_fit, y_fit = X_trva, y_trva

    pipe_final = Pipeline([("pre", pre), ("model", models[best_name])])
    pipe_final.fit(X_fit, y_fit)
    final_pred = pipe_final.predict(X_te)
    final_metrics = evaluate(best_name + "_refit", y_te, final_pred)
    print("Refit test metrics:", final_metrics)

    joblib.dump(pipe_final, DEMAND_MODEL)
    meta_out = {
        "best_model":     best_name,
        "features":       feature_cols,
        "cat_features":   cat_cols,
        "num_features":   num_cols,
        "target":         TARGET,
        "test_metrics":   final_metrics,
        "split":          meta,
        "seed":           SEED,
    }
    Path(DEMAND_META).write_text(json.dumps(meta_out, indent=2, default=str))
    print("saved:", DEMAND_MODEL)
    print("saved:", DEMAND_META)


if __name__ == "__main__":
    main()