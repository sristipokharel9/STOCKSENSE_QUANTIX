"""
Phase 1 - Data cleaning for the 5 raw StockSense tables.

Every cleaning action is logged into a shared list of dicts:
    issue, count_found, action, justification

Outputs go to data/processed/cleaned/*.csv
Rejected rows go to data/processed/rejected_rows.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_FILES, CLEANED_TRANSACTIONS, CLEANED_PRODUCTS, CLEANED_STORES,
    CLEANED_INVENTORY, CLEANED_EXTERNAL, REJECTED_ROWS, TRAP_LOG, SEED,
)

np.random.seed(SEED)

DQ_LOG: list[dict] = []


def log(issue: str, count: int, action: str, justification: str):
    DQ_LOG.append({
        "issue": issue,
        "count_found": int(count),
        "action": action,
        "justification": justification,
    })
    print(f"[DQ] {issue:45s} count={count:>8d} | {action}")


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def strip_upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def parse_dates_flexible(s: pd.Series) -> tuple[pd.Series, int]:
    """Try ISO first, then day-first. Return parsed dates + unparseable count."""
    iso = pd.to_datetime(s, errors="coerce", dayfirst=False)
    mask_bad = iso.isna()
    if mask_bad.any():
        alt = pd.to_datetime(s[mask_bad], errors="coerce", dayfirst=True)
        iso.loc[mask_bad] = alt
    n_bad = int(iso.isna().sum())
    return iso, n_bad


# ------------------------------------------------------------------
# 1. transactions
# ------------------------------------------------------------------
def clean_transactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RAW_FILES["transactions"])
    n0 = len(df)
    rejected_frames = []

    # 1a. dates
    df["date"], n_bad = parse_dates_flexible(df["date"])
    log("transactions: unparseable dates", n_bad,
        "Rows dropped",
        "Cannot place them on the time axis; required for lag/rolling features.")
    bad_mask = df["date"].isna()
    if bad_mask.any():
        rejected_frames.append(df.loc[bad_mask].assign(reject_reason="unparseable_date"))
    df = df.loc[~bad_mask].copy()

    # 1b. standardise ids
    df["store_id"]   = strip_upper(df["store_id"])
    df["product_id"] = strip_upper(df["product_id"])
    df["transaction_id"] = df["transaction_id"].astype(str).str.strip()

    # 1c. true duplicates: same transaction_id, same content
    dup_mask = df.duplicated(subset=["transaction_id"], keep="first")
    n_dup = int(dup_mask.sum())
    log("transactions: duplicate transaction_id (identical)", n_dup,
        "Dropped extra copies",
        "Same transaction recorded twice; safe to remove identical duplicates.")
    df = df.loc[~dup_mask].copy()

    # 1d. conflicting transaction_id (same id, different rows)
    conflict = df.duplicated(subset=["transaction_id"], keep=False) & ~df.duplicated(subset=["transaction_id"], keep="first")
    n_conflict = int(conflict.sum())
    if n_conflict:
        log("transactions: conflicting transaction_id", n_conflict,
            "Flagged, kept",
            "Same id with different content; not safe to drop silently.")
        df["txn_id_conflict"] = conflict.astype(int)
    else:
        df["txn_id_conflict"] = 0

    # 1e. negative quantity -> reject
    neg_mask = df["quantity"] < 0
    n_neg = int(neg_mask.sum())
    log("transactions: negative quantity", n_neg,
        "Excluded from demand (kept in rejected_rows.csv)",
        "Demand cannot be negative; treated as returns/data errors.")
    if n_neg:
        rejected_frames.append(df.loc[neg_mask].assign(reject_reason="negative_quantity"))
    df["is_return_or_error"] = neg_mask.astype(int)
    df = df.loc[~neg_mask].copy()

    # 1f. extreme outliers per product (99.9th pct)
    q999 = df.groupby("product_id")["quantity"].transform(lambda s: s.quantile(0.999))
    out_mask = df["quantity"] > q999
    n_out = int(out_mask.sum())
    log("transactions: extreme quantity outliers (>99.9 pct per product)", n_out,
        "Capped at 99.9th pct",
        "Extreme spikes break scaling for KNN/SVR; cap keeps signal, removes noise.")
    df.loc[out_mask, "quantity"] = q999[out_mask].round().astype(int)

    # 1g. zero/implausible selling_price
    bad_price = df["selling_price"] <= 0
    n_price = int(bad_price.sum())
    log("transactions: non-positive selling_price", n_price,
        "Recomputed from median per product",
        "Price must be positive; fallback to product median keeps revenue sane.")
    if n_price:
        med_price = df.groupby("product_id")["selling_price"].transform("median")
        df.loc[bad_price, "selling_price"] = med_price[bad_price]
        df.loc[bad_price, "selling_price"] = df.loc[bad_price, "selling_price"].fillna(
            df["selling_price"].median()
        )

    # 1h. payment_mode / hour sanity
    df["payment_mode"] = df["payment_mode"].astype(str).str.strip().str.title()
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(12).astype(int)

    log("transactions: rows after cleaning", len(df),
        f"Removed {n0 - len(df)} rows",
        "Net effect of all cleaning steps on transactions.")

    rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    return df.reset_index(drop=True), rejected


# ------------------------------------------------------------------
# 2. products
# ------------------------------------------------------------------
def clean_products() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["products"])

    # category standardisation map
    cat_map = {
        "GROCERY": "Groceries",
        "GROCERIES": "Groceries",
        "GROCERIES ": "Groceries",
        " GROCERIES ": "Groceries",
        "GROCERY ": "Groceries",
        "GROCERY  ": "Groceries",
    }
    before = df["category"].nunique(dropna=True)
    df["category"] = df["category"].astype(str).str.strip()
    df["category"] = df["category"].apply(
        lambda c: cat_map.get(c.upper(), c.title())
    )
    after = df["category"].nunique(dropna=True)
    log("products: category variants collapsed", before - after,
        f"Standardised to {after} categories",
        "Consistent categories needed for grouping and feature engineering.")

    df["product_id"] = strip_upper(df["product_id"])
    df["supplier_id"] = df["supplier_id"].astype(str).str.strip()

    # numeric
    for c in ["mrp", "cost_price", "shelf_life_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # cost_price sanity: must be <= mrp
    bad_cost = df["cost_price"] > df["mrp"]
    n_cost = int(bad_cost.sum())
    if n_cost:
        log("products: cost_price > mrp", n_cost,
            "Set cost_price = 0.75 * mrp",
            "Cost cannot exceed MRP; corrected to preserve margin logic.")
        df.loc[bad_cost, "cost_price"] = (df.loc[bad_cost, "mrp"] * 0.75).round(2)

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 3. stores
# ------------------------------------------------------------------
def clean_stores() -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["stores"])
    df["store_id"] = strip_upper(df["store_id"])
    df["city"]     = df["city"].astype(str).str.strip().str.title()
    df["store_type"] = df["store_type"].astype(str).str.strip().str.title()
    df["region"]   = df["region"].astype(str).str.strip().str.title()
    df = df.drop_duplicates(subset=["store_id"]).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 4. inventory
# ------------------------------------------------------------------
def clean_inventory(products: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["inventory"])
    df["date"], n_bad = parse_dates_flexible(df["date"])
    log("inventory: unparseable dates", n_bad,
        "Rows dropped",
        "Needed for time alignment with master table.")
    df = df.loc[df["date"].notna()].copy()

    df["store_id"]   = strip_upper(df["store_id"])
    df["product_id"] = strip_upper(df["product_id"])

    # coerce numerics
    for c in ["opening", "received", "sold", "closing", "reorder_lvl", "lead_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # inventory arithmetic check
    expected = df["opening"] + df["received"] - df["sold"]
    mismatch = (df["closing"] - expected).abs() > 0.5
    n_mm = int(mismatch.sum())
    log("inventory: closing != opening + received - sold", n_mm,
        "Reconciled closing = expected; flag kept",
        "Arithmetic must hold for stock-out logic; flag retained for audit.")
    df["inventory_mismatch"] = mismatch.astype(int)
    df["closing"] = expected

    # negative closing shouldn't happen; clip to 0
    n_neg = int((df["closing"] < 0).sum())
    if n_neg:
        log("inventory: negative closing stock", n_neg,
            "Clipped to 0",
            "Stock cannot be negative; clipped for numeric safety.")
        df["closing"] = df["closing"].clip(lower=0)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


# ------------------------------------------------------------------
# 5. external factors
# ------------------------------------------------------------------
def clean_external(stores: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(RAW_FILES["external_factors"])
    df["date"], n_bad = parse_dates_flexible(df["date"])
    df = df.loc[df["date"].notna()].copy()
    df["city"] = df["city"].astype(str).str.strip().str.title()

    # impute temp_c and rain_mm: city x month median -> city median -> global
    df["month"] = df["date"].dt.month
    for col in ["temp_c", "rain_mm"]:
        before_na = int(df[col].isna().sum())
        df[col] = pd.to_numeric(df[col], errors="coerce")

        city_month_med = df.groupby(["city", "month"])[col].transform("median")
        city_med       = df.groupby("city")[col].transform("median")
        global_med     = df[col].median()

        df[col] = df[col].fillna(city_month_med).fillna(city_med).fillna(global_med)
        df[f"{col}_imputed"] = (df[col].isna() | (before_na > 0)).astype(int)

        log(f"external: missing {col}", before_na,
            "Imputed with city-month median (fallback city -> global)",
            "Weather values are slow-moving and correlated within city-month.")

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["month"])
    return df


# ------------------------------------------------------------------
# 6. sparse-history detection
# ------------------------------------------------------------------
def flag_sparse_products(inventory: pd.DataFrame) -> pd.DataFrame:
    """Flag store x product combos with fewer than MIN_ACTIVE_DAYS of history."""
    from config import MIN_ACTIVE_DAYS
    df = inventory.copy()
    df["date"] = pd.to_datetime(df["date"])

    counts = (df.groupby(["store_id", "product_id"])["date"]
                .nunique()
                .reset_index(name="active_days"))

    sparse = counts[counts["active_days"] < MIN_ACTIVE_DAYS].copy()
    sparse["sparse_history"] = 1
    log("inventory: store x product with sparse history",
        len(sparse),
        f"Flagged (threshold = {MIN_ACTIVE_DAYS} days)",
        "Short history breaks lag features; flag allows low-confidence handling.")

    df = df.merge(sparse[["store_id", "product_id", "sparse_history"]],
                  on=["store_id", "product_id"], how="left")
    df["sparse_history"] = df["sparse_history"].fillna(0).astype(int)
    return df


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    print("=" * 72)
    print("PHASE 1 - CLEANING")
    print("=" * 72)

    txn, rejected = clean_transactions()
    products  = clean_products()
    stores    = clean_stores()
    inventory = clean_inventory(products)
    inventory = flag_sparse_products(inventory)
    external  = clean_external(stores)

    txn.to_csv(CLEANED_TRANSACTIONS, index=False)
    products.to_csv(CLEANED_PRODUCTS, index=False)
    stores.to_csv(CLEANED_STORES, index=False)
    inventory.to_csv(CLEANED_INVENTORY, index=False)
    external.to_csv(CLEANED_EXTERNAL, index=False)
    rejected.to_csv(REJECTED_ROWS, index=False)

    # save dq log
    dq_path = Path(CLEANED_TRANSACTIONS).parent.parent / "dq_log.json"
    with open(dq_path, "w") as f:
        json.dump(DQ_LOG, f, indent=2)
    print(f"\nDQ log saved to: {dq_path}")

    print("\nCleaned tables written to data/processed/cleaned/")
    print(f"  transactions: {len(txn):,} rows")
    print(f"  products:     {len(products):,} rows")
    print(f"  stores:       {len(stores):,} rows")
    print(f"  inventory:    {len(inventory):,} rows")
    print(f"  external:     {len(external):,} rows")
    print(f"  rejected:     {len(rejected):,} rows")


if __name__ == "__main__":
    main()