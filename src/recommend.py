"""
Phase 7 - Recommendation / Business Action Layer.

For every store x product on the latest prediction date:
  current_stock, incoming_stock, forecast_7d, stockout_prob, risk_level,
  safety_stock, recommended_stock, reorder_qty, top_reasons,
  manager_action, confidence, estimated_revenue_at_risk, overstock_flag.

Reads: features.csv, demand_model.joblib, stockout_model.joblib
Writes: data/processed/recommendations.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling
from explain import explain_row

cfg = load_sibling("config")
FEATURES_FILE   = cfg.FEATURES_FILE
DEMAND_MODEL    = cfg.DEMAND_MODEL
STOCKOUT_MODEL  = cfg.STOCKOUT_MODEL
DEMAND_META     = cfg.DEMAND_META
STOCKOUT_META   = cfg.STOCKOUT_META
RECOMMENDATIONS = cfg.RECOMMENDATIONS
SERVICE_LEVEL_Z = cfg.SERVICE_LEVEL_Z
RISK_HIGH       = cfg.RISK_HIGH
RISK_MEDIUM     = cfg.RISK_MEDIUM


def risk_level(p):
    if p >= RISK_HIGH:
        return "HIGH"
    if p >= RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def manager_action(risk, reorder_qty, reasons, overstock):
    if overstock:
        return ("Pause reorder. Stock covers more than 2x forecast demand; "
                "consider a markdown to avoid expiry.")
    if risk == "HIGH":
        top = ", ".join(r["phrase"] for r in reasons[:3]) if reasons else "risk signals"
        return f"Raise replenishment order today for {int(reorder_qty)} units. Reason: {top}."
    if risk == "MEDIUM":
        return f"Monitor daily and pre-approve a top-up of {int(reorder_qty)} units."
    return "No action needed."


def main():
    print("=" * 72)
    print("PHASE 7 - RECOMMENDATIONS")
    print("=" * 72)

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    dm = json.loads(Path(DEMAND_META).read_text())
    sm = json.loads(Path(STOCKOUT_META).read_text())

    feature_cols = dm["features"]
    demand_model   = joblib.load(DEMAND_MODEL)
    stockout_model = joblib.load(STOCKOUT_MODEL)

    # latest prediction date per (store, product)
    latest = (df.sort_values("date")
                .groupby(["store_id", "product_id"], as_index=False)
                .tail(1)
                .reset_index(drop=True))
    print(f"latest prediction rows: {len(latest):,}")

    X = latest[feature_cols]
    latest["forecast_7d"]   = demand_model.predict(X).clip(min=0).round().astype(int)
    proba = stockout_model.predict_proba(X)
    latest["stockout_prob"] = proba[:, 1] if proba.shape[1] > 1 else 0.0

    latest["risk_level"]    = latest["stockout_prob"].apply(risk_level)
    latest["current_stock"] = latest["closing"].fillna(0).clip(lower=0).astype(int)
    latest["incoming_stock"] = latest["received"].fillna(0).astype(int)

    # safety stock
    sigma_daily = latest["rolling_std_28"].fillna(latest["rolling_std_7"]).fillna(0)
    lead = latest["lead_days"].fillna(2)
    latest["safety_stock"] = (SERVICE_LEVEL_Z * sigma_daily * np.sqrt(lead)).round().astype(int)

    latest["recommended_stock"] = (latest["forecast_7d"] + latest["safety_stock"]).astype(int)
    latest["reorder_qty"] = (latest["recommended_stock"]
                             - latest["current_stock"]
                             - latest["incoming_stock"]).clip(lower=0).round().astype(int)

    # overstock: current + incoming > 2x recommended AND shelf life short
    latest["overstock_flag"] = (
        (latest["current_stock"] + latest["incoming_stock"]) >
        2 * latest["recommended_stock"].replace(0, np.nan)
    ).fillna(False).astype(int)

    # confidence
    latest["confidence"] = np.where(latest["sparse_history"].fillna(0) > 0,
                                    "low_confidence", "normal")

    # revenue at risk
    latest["estimated_revenue_at_risk"] = (
        (latest["forecast_7d"] - latest["current_stock"] - latest["incoming_stock"])
        .clip(lower=0) * latest["avg_selling_price"].fillna(0)
    ).round(2)

    # local explanations (top 3) — only for rows at risk to save time
    print("Computing local explanations for at-risk rows ...")
    at_risk = latest["risk_level"].isin(["HIGH", "MEDIUM"])
    reasons_list = []
    for i, row in latest.iterrows():
        if not at_risk.loc[i]:
            reasons_list.append([])
            continue
        try:
            reasons = explain_row(row, stockout_model, feature_cols, top_k=3)
        except Exception:
            reasons = []
        reasons_list.append(reasons)
    latest["top_reasons"] = reasons_list

    latest["manager_action"] = latest.apply(
        lambda r: manager_action(r["risk_level"], r["reorder_qty"],
                                 r["top_reasons"], r["overstock_flag"]),
        axis=1,
    )

    # final output
    out_cols = ["store_id", "product_id", "category", "city", "store_type",
                "date", "current_stock", "incoming_stock", "forecast_7d",
                "stockout_prob", "risk_level", "safety_stock",
                "recommended_stock", "reorder_qty", "top_reasons",
                "manager_action", "confidence", "estimated_revenue_at_risk",
                "overstock_flag", "sparse_history"]
    out = latest[out_cols].copy()
    out["top_reasons_str"] = out["top_reasons"].apply(
        lambda lst: "; ".join(f"{r['phrase']} (+{r['impact_pct']}%)"
                              for r in lst) if lst else ""
    )
    out = out.drop(columns=["top_reasons"])
    out = out.sort_values(["risk_level", "reorder_qty"],
                          ascending=[True, False])
    out.to_csv(RECOMMENDATIONS, index=False)

    print(f"recommendations saved: {RECOMMENDATIONS}")
    print(f"HIGH:   {(out['risk_level']=='HIGH').sum()}")
    print(f"MEDIUM: {(out['risk_level']=='MEDIUM').sum()}")
    print(f"LOW:    {(out['risk_level']=='LOW').sum()}")
    print(f"Overstock flagged: {out['overstock_flag'].sum()}")
    print("\nSample:")
    print(out[["store_id", "product_id", "forecast_7d",
               "stockout_prob", "risk_level", "reorder_qty",
               "manager_action"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()