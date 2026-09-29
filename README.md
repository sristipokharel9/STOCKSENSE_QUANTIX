# StockSense — Decision-Support for NovaMart Retail

Predicts 7-day demand per store-product, classifies stock-out risk,
explains the reason in plain language, and recommends a reorder quantity
plus a manager action. All in one interactive Streamlit dashboard.

Built for the IntelliData 2026 Data Science Hackathon — StockSense challenge.

---

## What the system does

1. **Data cleaning & quality** — the pipeline ingests 5 messy raw CSVs, logs every fix with issue / count / action / justification, and produces a Data Quality Report.
2. **EDA & statistics** — 6 business-question charts, a KPI table, and 4 hypothesis tests.
3. **Feature engineering** — time, lag, rolling, inventory, price/promo, store/product, external. All rolling features use `shift(1)`, checked by an automated leakage test.
4. **Model 1 — demand forecast** — predicts next 7-day demand per store-product. Baselines plus Linear Regression, Decision Tree, Random Forest, XGBoost, KNN, SVR. Time-based split with a 7-day gap.
5. **Model 2 — stock-out risk** — classifies risk as HIGH / MEDIUM / LOW for the next 7 days. Decision Tree, Random Forest, XGBoost, KNN, SVC, GaussianNB. Class imbalance handled with class weights / `scale_pos_weight`.
6. **Explainability** — global permutation importance and per-product local reasons translated into manager language.
7. **Recommendations** — reorder quantity, safety stock, revenue at risk, overstock flag, and a plain-sentence manager action.
8. **Dashboard** — 7 tabs including a What-If simulator.

---

## Data

Primary dataset: Kaggle — *Retail Store Inventory Forecasting Dataset* by Anirudh Chauhan.
`https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset`

The hackathon expects 5 messy tables (`transactions.csv`, `products.csv`,
`stores.csv`, `inventory.csv`, `external_factors.csv`). Since the Kaggle file
is one clean table, `src/simulate_hackathon_tables.py` is a **simulation bridge**:
it splits the Kaggle CSV into the 5 official tables and injects 6 data-quality
traps with a fixed seed. If the organisers supply the official files later,
drop them into `data/raw/` and re-run — the rest of the pipeline is unchanged.

### Traps injected (logged in `data/raw/injected_traps_log.json`)

1. Missing values in `temp_c` and `rain_mm`
2. Duplicate transaction rows (~1%)
3. Category inconsistencies (`Groceries`, `groceries`, `GROCERY`, ` Groceries `)
4. Negative and extreme `quantity`
5. Inventory arithmetic mismatch (~2%)
6. Sparse history for 3 products
7. Bonus: messy date formats and inconsistent store-ID formatting

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt