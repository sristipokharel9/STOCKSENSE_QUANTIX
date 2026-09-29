"""
KPI computation for StockSense.
Reads : data/processed/master_dataset.csv
Writes: reports/kpi_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
MASTER_DATASET = cfg.MASTER_DATASET
KPI_SUMMARY    = cfg.KPI_SUMMARY


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    total_revenue = float(df["revenue"].sum())
    total_units   = int(df["units_sold"].sum())

    available = df["opening"] + df["received"]
    stockout_mask = (df["closing"] <= 0) | (df["sold"] >= available)
    stockout_rate = float(stockout_mask.mean())

    cogs = float((df["units_sold"] * df["cost_price"]).sum())
    avg_inv_value = float(((df["opening"] + df["closing"]) / 2 * df["cost_price"]).mean())
    inv_turnover = cogs / avg_inv_value if avg_inv_value else np.nan

    avg_daily = df.groupby(["store_id", "product_id"])["units_sold"].transform("mean")
    avg_daily = avg_daily.replace(0, np.nan)
    doi = (df["closing"] / avg_daily).replace([np.inf, -np.inf], np.nan)
    days_of_inventory = float(doi.median())

    promo = df.loc[df["promotion_flag"] == 1, "units_sold"].mean()
    normal = df.loc[df["promotion_flag"] == 0, "units_sold"].mean()
    promo_lift = float((promo - normal) / normal) if normal and not np.isnan(normal) else np.nan

    demand_est = df["sold"]
    unfulfilled = np.maximum(0, demand_est - available)
    est_lost_sales = float((unfulfilled * df["avg_selling_price"]).sum())

    return pd.DataFrame([
        ("Revenue (Rs)",              round(total_revenue, 2)),
        ("Units sold",                total_units),
        ("Stockout rate",             round(stockout_rate, 4)),
        ("Inventory turnover",        round(inv_turnover, 3)),
        ("Median days of inventory",  round(days_of_inventory, 2)),
        ("Promotion lift",            round(promo_lift, 4)),
        ("Estimated lost sales (Rs)", round(est_lost_sales, 2)),
    ], columns=["kpi", "value"])


def main():
    df = pd.read_csv(MASTER_DATASET)
    kpis = compute_kpis(df)
    kpis.to_csv(KPI_SUMMARY, index=False)
    print("KPIs written to:", KPI_SUMMARY)
    print(kpis.to_string(index=False))


if __name__ == "__main__":
    main()