"""
Dashboard data loader. Reads only from saved artefacts.
"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    p = DATA_PROCESSED / "recommendations.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    p = DATA_PROCESSED / "features.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    p = DATA_PROCESSED / "master_dataset.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame:
    p = REPORTS_DIR / "kpi_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_forecast_cmp() -> pd.DataFrame:
    p = REPORTS_DIR / "model_comparison_forecast.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_stockout_cmp() -> pd.DataFrame:
    p = REPORTS_DIR / "model_comparison_stockout.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_importance(kind: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"importance_{kind}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_meta(kind: str) -> dict:
    p = MODELS_DIR / f"{kind}_model_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_stats_tests() -> str:
    p = REPORTS_DIR / "statistical_tests.md"
    return p.read_text() if p.exists() else ""


@st.cache_data(show_spinner=False)
def load_dq_report() -> str:
    p = REPORTS_DIR / "data_quality_report.md"
    return p.read_text() if p.exists() else ""


@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    dm = MODELS_DIR / "demand_model.joblib"
    sm = MODELS_DIR / "stockout_model.joblib"
    demand = joblib.load(dm) if dm.exists() else None
    stock  = joblib.load(sm) if sm.exists() else None
    return demand, stock