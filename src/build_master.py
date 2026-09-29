"""
Phase 1 - Build master table: ONE ROW = ONE DATE x ONE STORE x ONE PRODUCT.

Steps:
1. Aggregate transactions to daily grain.
2. Merge products, stores, inventory, external factors.
3. Re-index to a complete date x store x product grid.
4. Assert uniqueness and save.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRANSACTIONS, CLEANED_PRODUCTS, CLEANED_STORES,
    CLEANED_INVENTORY, CLEANED_EXTERNAL, MASTER_DATASET, SEED,
)

np.random.seed(SEED)


def aggregate_transactions(txn: pd.DataFrame) -> pd.DataFrame:
    txn["date"] = pd.to_datetime(txn["date"])
    g = txn.groupby(["date", "store_id", "product_id"])
    daily = g.agg(
        units_sold        = ("quantity", "sum"),
        revenue           = ("selling_price", lambda s: float((s * txn.loc[s.index, "quantity"]).sum())),
        avg_selling_price = ("selling_price", "mean"),
        avg_discount_pct  = ("discount_pct", "mean"),
        promotion_flag    = ("promotion_flag", "max"),
        n_transactions    = ("transaction_id", "count"),
        n_customers       = ("customer_id", "nunique"),
    ).reset_index()
    daily["revenue"] = daily["revenue"].round(2)
    return daily


def build_master() -> pd.DataFrame:
    print("=" * 72)
    print("PHASE 1 - MASTER BUILD")
    print("=" * 72)

    txn       = pd.read_csv(CLEANED_TRANSACTIONS)
    products  = pd.read_csv(CLEANED_PRODUCTS)
    stores    = pd.read_csv(CLEANED_STORES)
    inventory = pd.read_csv(CLEANED_INVENTORY)
    external  = pd.read_csv(CLEANED_EXTERNAL)

    daily = aggregate_transactions(txn)
    print(f"daily transactions grid: {len(daily):,} rows")

    # --- build complete date x store x product grid ---
    inv = inventory.copy()
    inv["date"] = pd.to_datetime(inv["date"])
    all_dates = pd.date_range(inv["date"].min(), inv["date"].max(), freq="D")
    combos = inv[["store_id", "product_id"]].drop_duplicates()
    grid = combos.merge(pd.DataFrame({"date": all_dates}), how="cross")

    df = grid.merge(daily, on=["date", "store_id", "product_id"], how="left")

    # fill units_sold=0 for missing days of active products
    df["units_sold"]        = df["units_sold"].fillna(0).astype(int)
    df["revenue"]           = df["revenue"].fillna(0.0)
    df["n_transactions"]    = df["n_transactions"].fillna(0).astype(int)
    df["n_customers"]       = df["n_customers"].fillna(0).astype(int)
    df["promotion_flag"]    = df["promotion_flag"].fillna(0).astype(int)
    df["avg_discount_pct"]  = df["avg_discount_pct"].fillna(0.0)
    df["avg_selling_price"] = df["avg_selling_price"].fillna(
        df.groupby("product_id")["avg_selling_price"].transform("median")
    )

    # --- merge inventory (date, store, product) ---
    inv["date"] = pd.to_datetime(inv["date"])
    df = df.merge(inv, on=["date", "store_id", "product_id"], how="left")

    # --- merge products ---
    df = df.merge(products, on="product_id", how="left")

    # --- merge stores ---
    df = df.merge(stores, on="store_id", how="left")

    # --- merge external on (date, city) ---
    ext = external.copy()
    ext["date"] = pd.to_datetime(ext["date"])
    df = df.merge(ext, on=["date", "city"], how="left")

    # final dtype fix
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # --- assert grain uniqueness ---
    assert not df.duplicated(subset=["date", "store_id", "product_id"]).any(), \
        "Master grain not unique on (date, store_id, product_id)!"
    print("Grain uniqueness: PASS")

    # --- asserts expected columns ---
    required = ["date", "store_id", "product_id", "units_sold",
                "opening", "received", "sold", "closing", "reorder_lvl",
                "category", "city", "store_type", "region", "lead_days"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Master missing required columns: {missing}")

    df.to_csv(MASTER_DATASET, index=False)
    print(f"master_dataset.csv saved: {len(df):,} rows")
    print(f"date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"stores: {df['store_id'].nunique()}  products: {df['product_id'].nunique()}")
    print(f"categories: {sorted(df['category'].dropna().unique().tolist())}")
    return df


if __name__ == "__main__":
    build_master()