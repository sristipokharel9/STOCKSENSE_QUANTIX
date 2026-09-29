"""
Stage A: Simulation bridge from the Kaggle CSV to the 5 official hackathon tables.
Injects 6 data-quality traps with a fixed seed and writes a trap log.

Run:
    python src/simulate_hackathon_tables.py

Outputs:
    data/raw/transactions.csv
    data/raw/products.csv
    data/raw/stores.csv
    data/raw/inventory.csv
    data/raw/external_factors.csv
    data/raw/injected_traps_log.json
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# --- import config from same directory regardless of cwd ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    KAGGLE_FILE, RAW_FILES, TRAP_LOG, SEED,
    TRAP_MISSING_TEMP_RATE, TRAP_DUPLICATE_RATE, TRAP_CATEGORY_INCONSISTENT,
    TRAP_NEGATIVE_QTY_RATE, TRAP_INVENTORY_MISMATCH,
    TRAP_SPARSE_PRODUCTS, TRAP_SPARSE_DAYS_RANGE,
)

rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# ============================================================
# 1. Load and inspect the Kaggle CSV
# ============================================================
def load_kaggle() -> pd.DataFrame:
    if not KAGGLE_FILE.exists():
        raise FileNotFoundError(
            f"\nKaggle CSV not found at: {KAGGLE_FILE}\n"
            f"Please download the 'Retail Store Inventory Forecasting Dataset' "
            f"from Kaggle and place it there, named 'retail_store_inventory.csv'."
        )
    df = pd.read_csv(KAGGLE_FILE)
    print("=" * 72)
    print("KAGGLE CSV INSPECTION")
    print("=" * 72)
    print(f"Shape          : {df.shape}")
    print(f"Columns        : {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nDescribe (all):\n{df.describe(include='all').T}")
    # quick uniques
    print("\nUnique value previews:")
    for c in df.columns:
        try:
            n = df[c].nunique(dropna=True)
            if n <= 30:
                print(f"  - {c}: {sorted(df[c].dropna().unique().tolist())}")
            else:
                print(f"  - {c}: {n} unique values")
        except Exception:
            pass
    print("=" * 72)
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the actual column name that best matches any candidate (fuzzy)."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        if n in norm_map:
            return norm_map[n]
    # partial match fallback
    for cand in candidates:
        n = norm(cand)
        for k, v in norm_map.items():
            if n in k or k in n:
                return v
    return None


def detect_columns(df: pd.DataFrame) -> dict:
    candidates = {
        "date":            ["Date", "TransactionDate", "date"],
        "store":           ["Store ID", "StoreID", "Store", "store_id"],
        "product":         ["Product ID", "ProductID", "Product", "SKU", "product_id"],
        "category":        ["Category", "category"],
        "region":          ["Region", "region"],
        "inventory_level": ["Inventory Level", "InventoryLevel", "Inventory", "inventory"],
        "units_sold":      ["Units Sold", "UnitsSold", "Sales", "units_sold"],
        "units_ordered":   ["Units Ordered", "UnitsOrdered", "Ordered", "units_ordered"],
        "demand_forecast": ["Demand Forecast", "DemandForecast", "Demand", "demand"],
        "price":           ["Price", "UnitPrice", "price"],
        "discount":        ["Discount", "DiscountPct", "discount"],
        "weather":         ["Weather Condition", "WeatherCondition", "Weather", "weather"],
        "promotion":       ["Holiday/Promotion", "HolidayPromotion", "Promotion", "Holiday", "promotion"],
        "competitor":      ["Competitor Pricing", "CompetitorPricing", "competitor"],
        "seasonality":     ["Seasonality", "Season", "season"],
    }
    colmap = {}
    for k, cands in candidates.items():
        colmap[k] = find_col(df, cands)
    print("\nDetected column mapping:")
    for k, v in colmap.items():
        print(f"  {k:20s} -> {v}")
    missing = [k for k, v in colmap.items() if v is None and k in (
        "date", "store", "product", "units_sold", "price", "inventory_level",
    )]
    if missing:
        raise ValueError(f"Required columns not found in Kaggle CSV: {missing}")
    return colmap


# ============================================================
# 2. Static lookup tables (documented assumptions)
# ============================================================
CITY_MAP = {}          # filled at runtime based on unique store IDs
STORE_TYPE_CYCLE = ["Supermarket", "Hypermarket", "Express"]
TN_CITIES = ["Coimbatore", "Chennai", "Madurai", "Salem", "Trichy",
             "Erode", "Tirunelveli", "Vellore", "Thanjavur", "Kanchipuram"]

# per-category cost factor, shelf life, lead time, sub-categories, brands
CATEGORY_META = {
    "Groceries":    dict(cost=0.75, shelf=30,  lead=2, sub=["Staples", "Snacks", "Beverages"],
                         brands=["FreshFarm", "DailyBest", "NaturePick"]),
    "Electronics":  dict(cost=0.80, shelf=730, lead=5, sub=["Accessories", "Audio", "Small Appliances"],
                         brands=["VoltX", "PixelPro", "NovaTech"]),
    "Clothing":     dict(cost=0.45, shelf=540, lead=4, sub=["Men", "Women", "Kids"],
                         brands=["UrbanThreads", "StyleHub", "EverydayWear"]),
    "Furniture":    dict(cost=0.60, shelf=1095, lead=5, sub=["Home", "Office", "Outdoor"],
                         brands=["WoodCraft", "Homely", "ComfortPlus"]),
    "Toys":         dict(cost=0.55, shelf=900, lead=4, sub=["Educational", "Outdoor", "Plush"],
                         brands=["FunBox", "PlaySmart", "TinyWorld"]),
    "Groceries ":   dict(cost=0.75, shelf=30,  lead=2, sub=["Staples", "Snacks", "Beverages"],
                         brands=["FreshFarm", "DailyBest", "NaturePick"]),
}
DEFAULT_CAT = dict(cost=0.65, shelf=180, lead=3, sub=["General"], brands=["Generic"])


def cat_meta(cat: str) -> dict:
    return CATEGORY_META.get(str(cat).strip(), DEFAULT_CAT)


# ============================================================
# 3. Build the 5 tables
# ============================================================
def standard_date(s: str) -> str:
    """ISO date string for the clean version."""
    return pd.to_datetime(s, errors="coerce").strftime("%Y-%m-%d")


def build_transactions(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    print("\nBuilding transactions.csv ...")
    keep_cols = [colmap[k] for k in
                 ["date", "store", "product", "units_sold", "price",
                  "discount", "promotion"] if colmap.get(k)]
    sub = df[keep_cols].copy()
    sub.columns = [k for k in ["date", "store", "product", "units_sold",
                                "price", "discount", "promotion"]
                   if colmap.get(k)]

    if "discount" not in sub.columns: sub["discount"] = 0.0
    if "promotion" not in sub.columns: sub["promotion"] = 0
    sub["date"]        = pd.to_datetime(sub["date"], errors="coerce")
    sub["units_sold"]  = pd.to_numeric(sub["units_sold"], errors="coerce").fillna(0).astype(int)
    sub["price"]       = pd.to_numeric(sub["price"], errors="coerce").fillna(0.0)
    sub["discount"]    = pd.to_numeric(sub["discount"], errors="coerce").fillna(0.0)
    sub["promotion"]   = pd.to_numeric(sub["promotion"], errors="coerce").fillna(0).astype(int)
    sub = sub.dropna(subset=["date"])

    records = []
    next_id = 0
    for _, r in sub.iterrows():
        qty = int(r["units_sold"])
        if qty <= 0:
            continue
        # split into 1-3 transactions
        p = rng.random()
        n_txn = 1 if p < 0.70 else (2 if p < 0.92 else 3)
        if n_txn == 1:
            splits = [qty]
        else:
            # random multinomial split
            splits = list(rng.multinomial(qty, [1/n_txn]*n_txn))
        for q in splits:
            if q <= 0:
                continue
            records.append({
                "transaction_id": f"TXN{next_id:08d}",
                "date":           r["date"].strftime("%Y-%m-%d"),
                "store_id":       str(r["store"]),
                "product_id":     str(r["product"]),
                "quantity":       int(q),
                "selling_price":  round(float(r["price"]) * (1 - float(r["discount"])/100.0), 2),
                "discount_pct":   float(r["discount"]),
                "promotion_flag": int(r["promotion"]),
                "customer_id":    f"C{rng.integers(1, 100000):06d}",
                "payment_mode":   str(rng.choice(["UPI", "Card", "Cash"], p=[0.5, 0.3, 0.2])),
                "hour":           int(rng.integers(7, 23)),
            })
            next_id += 1
    txn = pd.DataFrame(records)
    print(f"  transactions rows: {len(txn):,}")
    return txn


def build_products(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    print("\nBuilding products.csv ...")
    g = df.groupby(colmap["product"])
    rows = []
    for pid, sub in g:
        cat = sub[colmap["category"]].mode().iloc[0] if colmap.get("category") else "Groceries"
        meta = cat_meta(cat)
        mrp = float(pd.to_numeric(sub[colmap["price"]], errors="coerce").median())
        rows.append({
            "product_id":     str(pid),
            "category":       str(cat).strip(),
            "sub_category":   str(rng.choice(meta["sub"])),
            "brand":          str(rng.choice(meta["brands"])),
            "mrp":            round(mrp, 2),
            "cost_price":     round(mrp * meta["cost"], 2),
            "shelf_life_days": int(meta["shelf"]),
            "supplier_id":    f"SUP{int(rng.integers(1, 20)):03d}",
        })
    p = pd.DataFrame(rows)
    print(f"  products rows: {len(p):,}")
    return p


def build_stores(df: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    print("\nBuilding stores.csv ...")
    stores = sorted(df[colmap["store"]].astype(str).unique())
    rows = []
    for i, s in enumerate(stores):
        if colmap.get("region"):
            region_vals = df.loc[df[colmap["store"]].astype(str) == s, colmap["region"]]
            region = str(region_vals.mode().iloc[0]) if len(region_vals) else "South"
        else:
            region = "South"
        rows.append({
            "store_id":          s,
            "city":              TN_CITIES[i % len(TN_CITIES)],
            "store_type":        STORE_TYPE_CYCLE[i % len(STORE_TYPE_CYCLE)],
            "floor_area_sqft":   int(rng.integers(2500, 18000)),
            "avg_daily_customers": int(rng.integers(400, 3500)),
            "region":            region,
        })
    st = pd.DataFrame(rows)
    print(f"  stores rows: {len(st):,}")
    return st


def build_inventory(df: pd.DataFrame, colmap: dict,
                    products: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding inventory.csv ...")
    sub = df[[colmap["date"], colmap["store"], colmap["product"],
              colmap["inventory_level"], colmap["units_sold"]]
              + ([colmap["units_ordered"]] if colmap.get("units_ordered") else [])].copy()
    sub.columns = (["date", "store_id", "product_id", "opening", "sold"]
                   + (["received"] if colmap.get("units_ordered") else []))
    sub["date"]       = pd.to_datetime(sub["date"], errors="coerce")
    sub["opening"]    = pd.to_numeric(sub["opening"], errors="coerce").fillna(0).astype(int)
    sub["sold"]       = pd.to_numeric(sub["sold"], errors="coerce").fillna(0).astype(int)
    if "received" not in sub.columns:
        sub["received"] = 0
    sub["received"]   = pd.to_numeric(sub["received"], errors="coerce").fillna(0).astype(int)
    sub["store_id"]   = sub["store_id"].astype(str)
    sub["product_id"] = sub["product_id"].astype(str)
    sub["closing"]    = sub["opening"] + sub["received"] - sub["sold"]

    # lead_days + reorder_lvl from product category
    prod_meta = products.set_index("product_id")[["category"]]
    sub = sub.merge(prod_meta, left_on="product_id", right_index=True, how="left")
    sub["lead_days"] = sub["category"].apply(lambda c: cat_meta(c)["lead"])
    # reorder level = lead_days * avg daily demand (approx) * safety factor
    avg_daily = sub.groupby(["store_id", "product_id"])["sold"].transform("mean").fillna(1)
    sub["reorder_lvl"] = (sub["lead_days"] * avg_daily * 1.5).round().astype(int)

    inv = sub[["date", "store_id", "product_id", "opening",
               "received", "sold", "closing", "reorder_lvl", "lead_days"]].copy()
    inv["date"] = inv["date"].dt.strftime("%Y-%m-%d")
    print(f"  inventory rows: {len(inv):,}")
    return inv


def build_external(df: pd.DataFrame, colmap: dict,
                   stores: pd.DataFrame) -> pd.DataFrame:
    print("\nBuilding external_factors.csv ...")
    dts = pd.to_datetime(df[colmap["date"]], errors="coerce").dropna().unique()
    dts = pd.to_datetime(sorted(dts))
    rows = []
    # festival list within the observed date range (approx. India)
    fest_dates = {}
    for d in pd.date_range(dts.min(), dts.max(), freq="D"):
        fest_dates[d] = 0
    for m, day, name in [(1, 1, "New Year"), (1, 14, "Pongal"),
                         (8, 15, "Independence Day"), (10, 2, "Gandhi Jayanti"),
                         (11, 1, "Diwali"), (12, 25, "Christmas")]:
        for y in range(dts.min().year, dts.max().year + 1):
            try:
                d = pd.Timestamp(year=y, month=m, day=day)
                if d in fest_dates:
                    fest_dates[d] = 1
            except Exception:
                pass

    weather_map = {"Sunny": (33, 0), "Cloudy": (28, 2),
                   "Rainy": (26, 12), "Snowy": (5, 5), "Windy": (27, 1)}

    for _, st in stores.iterrows():
        for d in dts:
            if colmap.get("weather"):
                w_series = df.loc[pd.to_datetime(df[colmap["date"]]) == d, colmap["weather"]]
                w = str(w_series.mode().iloc[0]) if len(w_series) else "Sunny"
            else:
                w = "Sunny"
            base_t, base_r = weather_map.get(w, (30, 0))
            rows.append({
                "date":        d.strftime("%Y-%m-%d"),
                "city":        st["city"],
                "temp_c":      round(base_t + rng.normal(0, 2), 1),
                "rain_mm":     round(max(0.0, base_r + rng.normal(0, 2)), 1),
                "holiday":     int(d.weekday() >= 5),
                "festival":    int(fest_dates.get(d, 0)),
                "weekend":     int(d.weekday() >= 5),
                "local_event": int(rng.random() < 0.05),
            })
    ef = pd.DataFrame(rows)
    print(f"  external_factors rows: {len(ef):,}")
    return ef


# ============================================================
# 4. Inject the 6 traps (fixed seed, log counts)
# ============================================================
def inject_traps(txn, products, stores, inventory, external):
    trap_log = {}

    # --- Trap 1: missing values in temp_c / rain_mm ---
    for col, rate in [("temp_c", TRAP_MISSING_TEMP_RATE), ("rain_mm", TRAP_MISSING_TEMP_RATE/2)]:
        idx = rng.choice(external.index, size=int(len(external) * rate), replace=False)
        external.loc[idx, col] = np.nan
        trap_log[f"missing_{col}"] = int(len(idx))

    # --- Trap 2: duplicate transaction rows (~1%) with same transaction_id ---
    n_dup = int(len(txn) * TRAP_DUPLICATE_RATE)
    dup_idx = rng.choice(txn.index, size=n_dup, replace=False)
    dup_rows = txn.loc[dup_idx].copy()
    txn = pd.concat([txn, dup_rows], ignore_index=True)
    txn = txn.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    trap_log["duplicate_transactions"] = int(n_dup)

    # --- Trap 3: category inconsistency on products (~8%) ---
    variants = {"Groceries": ["groceries", "GROCERIES", " Groceries ", "Grocery"]}
    n_cat = int(len(products) * TRAP_CATEGORY_INCONSISTENT)
    cat_idx = rng.choice(products.index, size=n_cat, replace=False)
    for i in cat_idx:
        c = products.at[i, "category"]
        if c in variants:
            products.at[i, "category"] = str(rng.choice(variants[c]))
        else:
            products.at[i, "category"] = " " + str(c).lower() + " "
    trap_log["category_inconsistent"] = int(n_cat)

    # --- Trap 4: negative + extreme quantity in transactions ---
    n_neg = max(1, int(len(txn) * TRAP_NEGATIVE_QTY_RATE))
    neg_idx = rng.choice(txn.index, size=n_neg, replace=False)
    txn.loc[neg_idx, "quantity"] = -rng.integers(1, 6, size=n_neg)
    n_out = max(1, int(len(txn) * 0.0005))
    out_idx = rng.choice(txn.index, size=n_out, replace=False)
    txn.loc[out_idx, "quantity"] = rng.integers(400, 900, size=n_out)
    trap_log["negative_quantity"] = int(n_neg)
    trap_log["extreme_quantity"]  = int(n_out)

    # --- Trap 5: inventory arithmetic mismatch (~2%) ---
    n_mm = int(len(inventory) * TRAP_INVENTORY_MISMATCH)
    mm_idx = rng.choice(inventory.index, size=n_mm, replace=False)
    noise = rng.integers(-25, 26, size=n_mm)
    inventory.loc[mm_idx, "closing"] = inventory.loc[mm_idx, "closing"] + noise
    trap_log["inventory_mismatch"] = int(n_mm)

    # --- Trap 6: sparse history — 3 products only 5-10 days ---
    all_dates = sorted(inventory["date"].unique())
    sparse_products = rng.choice(products["product_id"].unique(),
                                 size=TRAP_SPARSE_PRODUCTS, replace=False)
    n_removed = 0
    for p in sparse_products:
        keep_days = int(rng.integers(*TRAP_SPARSE_DAYS_RANGE))
        keep_dates = set(rng.choice(all_dates, size=keep_days, replace=False))
        mask = (inventory["product_id"] == p) & (~inventory["date"].isin(keep_dates))
        n_removed += int(mask.sum())
        inventory = inventory.loc[~mask].reset_index(drop=True)
    trap_log["sparse_products"]      = [str(x) for x in sparse_products]
    trap_log["sparse_rows_removed"]  = int(n_removed)

    # --- Bonus: messy date formats + inconsistent ID formatting ---
    n_messy_date = max(1, int(len(txn) * 0.01))
    messy_idx = rng.choice(txn.index, size=n_messy_date, replace=False)
    for i in messy_idx:
        d = pd.to_datetime(txn.at[i, "date"], errors="coerce")
        if pd.notna(d):
            txn.at[i, "date"] = d.strftime("%d/%m/%Y")
    trap_log["messy_date_rows"] = int(n_messy_date)

    n_id = max(1, int(len(txn) * 0.01))
    id_idx = rng.choice(txn.index, size=n_id, replace=False)
    for i in id_idx:
        sid = str(txn.at[i, "store_id"])
        choice = rng.integers(0, 3)
        if choice == 0:
            txn.at[i, "store_id"] = sid.lower()
        elif choice == 1:
            txn.at[i, "store_id"] = sid + " "
        else:
            txn.at[i, "store_id"] = " " + sid
    trap_log["messy_store_id_rows"] = int(n_id)

    # seed information
    trap_log["seed"] = SEED
    return txn, products, stores, inventory, external, trap_log


# ============================================================
# 5. Main
# ============================================================
def main():
    df = load_kaggle()
    colmap = detect_columns(df)

    print("\nBuilding the 5 raw tables ...")
    txn       = build_transactions(df, colmap)
    products  = build_products(df, colmap)
    stores    = build_stores(df, colmap)
    inventory = build_inventory(df, colmap, products)
    external  = build_external(df, colmap, stores)

    print("\nInjecting 6 traps ...")
    txn, products, stores, inventory, external, trap_log = inject_traps(
        txn, products, stores, inventory, external
    )

    print("\nSaving files ...")
    txn.to_csv(RAW_FILES["transactions"], index=False)
    products.to_csv(RAW_FILES["products"], index=False)
    stores.to_csv(RAW_FILES["stores"], index=False)
    inventory.to_csv(RAW_FILES["inventory"], index=False)
    external.to_csv(RAW_FILES["external_factors"], index=False)

    with open(TRAP_LOG, "w") as f:
        json.dump(trap_log, f, indent=2)

    print("\n" + "=" * 72)
    print("TRAP LOG")
    print("=" * 72)
    for k, v in trap_log.items():
        print(f"  {k:28s} = {v}")

    print("\nDone. 5 raw CSVs + injected_traps_log.json written to data/raw/.")


if __name__ == "__main__":
    main()