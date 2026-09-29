"""
Phase 6 - Explainability.

Provides:
- global_feature_importance(model, X, y)  -> DataFrame of permutation importance
- explain_row(store_id, product_id)       -> list of manager-language reasons

The per-row explanation uses XGBoost's pred_contribs when the model is XGBoost,
otherwise a permutation-based fallback. Everything is translated into
manager language via a fixed dictionary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.inspection import permutation_importance
from xgboost import DMatrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
FEATURES_FILE   = cfg.FEATURES_FILE
SPLIT_META      = cfg.SPLIT_META
DEMAND_MODEL    = cfg.DEMAND_MODEL
STOCKOUT_MODEL  = cfg.STOCKOUT_MODEL
DEMAND_META     = cfg.DEMAND_META
STOCKOUT_META   = cfg.STOCKOUT_META


# ------------------------------------------------------------
# Manager-language dictionary
# ------------------------------------------------------------
PHRASES = {
    "promotion_flag":         "Promotion is running",
    "promo_last_7d_count":    "Frequent promotions in the last 7 days",
    "weekend_flag":           "Weekend approaching",
    "festival_flag":          "Festival effect",
    "holiday_flag":           "Public holiday effect",
    "sales_growth_7":         "Recent sales growth",
    "rolling_mean_7":         "High recent average demand",
    "rolling_mean_14":        "Strong 2-week average",
    "rolling_mean_28":        "Strong 4-week average",
    "rolling_std_7":          "Demand has been unpredictable lately",
    "rolling_std_28":         "Long-term demand volatility",
    "rolling_max_7":          "Recent demand spikes",
    "lag_1":                  "Yesterday's sales",
    "lag_7":                  "Sales one week ago",
    "lag_14":                 "Sales two weeks ago",
    "lag_28":                 "Sales four weeks ago",
    "days_of_inventory":      "Days of stock left",
    "inventory_to_demand_ratio": "Stock vs demand ratio",
    "reorder_gap":            "Stock vs reorder level",
    "stock_cover_days":       "Stock cover days",
    "prev_day_closing":       "Yesterday's closing stock",
    "discount_pct":           "Discount is boosting demand",
    "price_change":           "Price change since yesterday",
    "price_vs_mrp":           "Price vs MRP",
    "lead_days":              "Supplier lead time",
    "temp_c":                 "Temperature",
    "rain_mm":                "Rainfall",
    "day_of_week":            "Day of week effect",
    "week_no":                "Week of year effect",
    "month":                  "Monthly seasonality",
    "is_month_end":           "Month-end shopping pattern",
}


def humanize(feature: str) -> str:
    return PHRASES.get(feature, feature.replace("_", " "))


# ------------------------------------------------------------
# Global importance
# ------------------------------------------------------------
def global_permutation_importance(model, X, y, n_repeats=5, sample=3000):
    """Permutation importance on a sample of rows."""
    if len(X) > sample:
        idx = np.random.RandomState(42).choice(len(X), sample, replace=False)
        Xs, ys = X.iloc[idx], np.asarray(y)[idx]
    else:
        Xs, ys = X, y
    r = permutation_importance(model, Xs, ys, n_repeats=n_repeats,
                               random_state=42, n_jobs=-1)
    imp = pd.DataFrame({
        "feature":    X.columns,
        "importance": r.importances_mean,
        "std":        r.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp


# ------------------------------------------------------------
# Local explanation
# ------------------------------------------------------------
def _xgb_contribs(model, row_df):
    """Return per-feature contributions for a single row using XGBoost."""
    # find the xgb step
    if "model" not in model.named_steps:
        return None
    xgb_est = model.named_steps["model"]
    if not hasattr(xgb_est, "get_booster"):
        return None
    # transform row through the preprocessor
    pre = model.named_steps["pre"]
    Xt = pre.transform(row_df)
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(Xt.shape[1])]
    dm = DMatrix(Xt, feature_names=list(feature_names))
    contribs = xgb_est.get_booster().predict(dm, pred_contribs=True)
    return feature_names, contribs[0][:-1]  # last element is bias


def explain_row(row: pd.Series, model, feature_cols, top_k=5):
    """
    Return top_k (feature, contribution) for a single row.
    Uses XGBoost native contributions when available.
    """
    row_df = pd.DataFrame([row[feature_cols].to_dict()])

    res = _xgb_contribs(model, row_df)
    if res is not None:
        names, values = res
        pairs = list(zip(names, values))
    else:
        # Fallback: feature value x global importance
        imp = global_permutation_importance(model, row_df, np.array([0.0]),
                                            n_repeats=2, sample=1)
        pairs = [(f, float(row[f]) * float(imp.set_index('feature').loc[f, 'importance'])
                     if f in imp['feature'].values else 0.0)
                 for f in feature_cols]

    # keep only positive drivers
    pairs = [(f, v) for f, v in pairs if v is not None and v > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)

    total_pos = sum(v for _, v in pairs) or 1.0
    top = pairs[:top_k]

    out = []
    for feat_name, val in top:
        # translate: strip one-hot prefix like "cat__category_Groceries"
        base = feat_name
        if "__" in base:
            base = base.split("__", 1)[1]
        # cut off the one-hot suffix if any
        for known in PHRASES:
            if base.startswith(known):
                base = known
                break
        phrase = humanize(base)
        share  = 100.0 * val / total_pos
        out.append({"feature": feat_name, "phrase": phrase,
                    "impact_pct": round(share, 1)})
    return out


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    dm = json.loads(Path(DEMAND_META).read_text())
    sm = json.loads(Path(STOCKOUT_META).read_text())

    demand_model   = joblib.load(DEMAND_MODEL)
    stockout_model = joblib.load(STOCKOUT_MODEL)

    feature_cols = dm["features"]
    X = df[feature_cols]
    y = df["next_7_day_demand"].values

    print("Computing global permutation importance for demand model ...")
    imp_d = global_permutation_importance(demand_model, X, y, sample=2000)
    imp_d.to_csv(cfg.REPORTS_DIR / "importance_demand.csv", index=False)
    print(imp_d.head(15).to_string(index=False))

    print("\nComputing global permutation importance for stockout model ...")
    yc = df["stockout_flag"].values
    imp_c = global_permutation_importance(stockout_model, X, yc, sample=2000)
    imp_c.to_csv(cfg.REPORTS_DIR / "importance_stockout.csv", index=False)
    print(imp_c.head(15).to_string(index=False))

    # small demo: explain one row
    print("\nExample local explanation (first row):")
    row = df.iloc[0]
    reasons = explain_row(row, stockout_model, feature_cols, top_k=5)
    for r in reasons:
        print("  -", r)

    Path(cfg.EXPLAIN_META).write_text(json.dumps({
        "top_demand": imp_d.head(15).to_dict(orient="records"),
        "top_stockout": imp_c.head(15).to_dict(orient="records"),
    }, indent=2, default=str))
    print("\nExplainability artefacts saved.")


if __name__ == "__main__":
    main()