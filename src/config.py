"""
Central configuration for the StockSense project.
All paths, seeds, thresholds, and split fractions live here.
Nothing else in the codebase should hard-code these.
"""
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT             = Path(__file__).resolve().parent.parent
DATA_RAW         = ROOT / "data" / "raw"
DATA_PROCESSED   = ROOT / "data" / "processed"
MODELS_DIR       = ROOT / "models"
REPORTS_DIR      = ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"
DASHBOARD_DIR    = ROOT / "dashboard"
NOTEBOOKS_DIR    = ROOT / "notebooks"
TESTS_DIR        = ROOT / "tests"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR,
          FIGURES_DIR, DASHBOARD_DIR, NOTEBOOKS_DIR, TESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------
# Kaggle source + simulated official files
# ------------------------------------------------------------
KAGGLE_FILE = DATA_RAW / "retail_store_inventory.csv"

RAW_FILES = {
    "transactions":     DATA_RAW / "transactions.csv",
    "products":         DATA_RAW / "products.csv",
    "stores":           DATA_RAW / "stores.csv",
    "inventory":        DATA_RAW / "inventory.csv",
    "external_factors": DATA_RAW / "external_factors.csv",
}
TRAP_LOG = DATA_RAW / "injected_traps_log.json"

# ------------------------------------------------------------
# Processed outputs
# ------------------------------------------------------------
MASTER_DATASET  = DATA_PROCESSED / "master_dataset.csv"
FEATURES_FILE   = DATA_PROCESSED / "features.csv"
RECOMMENDATIONS = DATA_PROCESSED / "recommendations.csv"
REJECTED_ROWS   = DATA_PROCESSED / "rejected_rows.csv"

# ------------------------------------------------------------
# Saved models + metadata
# ------------------------------------------------------------
DEMAND_MODEL   = MODELS_DIR / "demand_model.joblib"
DEMAND_META    = MODELS_DIR / "demand_model_meta.json"
STOCKOUT_MODEL = MODELS_DIR / "stockout_model.joblib"
STOCKOUT_META  = MODELS_DIR / "stockout_model_meta.json"

# ------------------------------------------------------------
# Report outputs
# ------------------------------------------------------------
DQ_REPORT_MD  = REPORTS_DIR / "data_quality_report.md"
DQ_REPORT_CSV = REPORTS_DIR / "data_quality_report.csv"
FORECAST_CMP  = REPORTS_DIR / "model_comparison_forecast.csv"
STOCKOUT_CMP  = REPORTS_DIR / "model_comparison_stockout.csv"
STAT_TESTS    = REPORTS_DIR / "statistical_tests.md"
KPI_SUMMARY   = REPORTS_DIR / "kpi_summary.csv"
PITCH_OUTLINE = REPORTS_DIR / "PITCH_OUTLINE.md"

# ------------------------------------------------------------
# Target horizon and split
# ------------------------------------------------------------
TARGET_HORIZON = 7        # predict next 7 days
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
TEST_FRAC      = 0.15
GAP_DAYS       = 7        # gap between train/val and val/test targets

# ------------------------------------------------------------
# Business thresholds
# ------------------------------------------------------------
RISK_HIGH        = 0.70
RISK_MEDIUM      = 0.40
SERVICE_LEVEL_Z  = 1.65   # 95% service level
MIN_ACTIVE_DAYS  = 30     # sparse-history flag threshold
LOW_CONFIDENCE   = "low_confidence"

# ------------------------------------------------------------
# Traps injected into the 5 raw CSVs (for the trap log)
# ------------------------------------------------------------
TRAP_MISSING_TEMP_RATE     = 0.04
TRAP_DUPLICATE_RATE        = 0.01
TRAP_CATEGORY_INCONSISTENT = 0.08
TRAP_NEGATIVE_QTY_RATE     = 0.003
TRAP_INVENTORY_MISMATCH    = 0.02
TRAP_SPARSE_PRODUCTS       = 3
TRAP_SPARSE_DAYS_RANGE     = (5, 10)

# ------------------------------------------------------------
# Phase 1 outputs
# ------------------------------------------------------------
CLEANED_DIR       = DATA_PROCESSED / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_TRANSACTIONS     = CLEANED_DIR / "transactions_clean.csv"
CLEANED_PRODUCTS         = CLEANED_DIR / "products_clean.csv"
CLEANED_STORES           = CLEANED_DIR / "stores_clean.csv"
CLEANED_INVENTORY        = CLEANED_DIR / "inventory_clean.csv"
CLEANED_EXTERNAL         = CLEANED_DIR / "external_factors_clean.csv"
DQ_LOG_JSON              = DATA_PROCESSED / "dq_log.json"

# ------------------------------------------------------------
# Phase 2 / 3 outputs
# ------------------------------------------------------------
KPI_SUMMARY      = REPORTS_DIR / "kpi_summary.csv"
STAT_TESTS       = REPORTS_DIR / "statistical_tests.md"
FEATURES_FILE    = DATA_PROCESSED / "features.csv"
SPLIT_META       = DATA_PROCESSED / "split_meta.json"

# ------------------------------------------------------------
# Phase 6 / 7 outputs
# ------------------------------------------------------------
EXPLAIN_META = MODELS_DIR / "explain_meta.json"