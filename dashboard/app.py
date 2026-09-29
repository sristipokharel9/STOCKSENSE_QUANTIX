"""
StockSense - Streamlit dashboard.
Run: streamlit run dashboard/app.py

Six tabs + What-If simulator. Reads only from saved artefacts.
"""
import importlib.util
import json
import sys
from pathlib import Path

# --- load loader.py by absolute path (works no matter how streamlit launches) ---
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("loader", HERE / "loader.py")
_loader = importlib.util.module_from_spec(_spec)
sys.modules["loader"] = _loader
_spec.loader.exec_module(_loader)

load_recommendations = _loader.load_recommendations
load_features        = _loader.load_features
load_master          = _loader.load_master
load_kpis            = _loader.load_kpis
load_forecast_cmp    = _loader.load_forecast_cmp
load_stockout_cmp    = _loader.load_stockout_cmp
load_importance      = _loader.load_importance
load_meta            = _loader.load_meta
load_stats_tests     = _loader.load_stats_tests
load_dq_report       = _loader.load_dq_report
load_models          = _loader.load_models

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="StockSense", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
recs     = load_recommendations()
feats    = load_features()
master   = load_master()
kpis     = load_kpis()
fcmp     = load_forecast_cmp()
scmp     = load_stockout_cmp()
imp_d    = load_importance("demand")
imp_s    = load_importance("stockout")
meta_d   = load_meta("demand")
meta_s   = load_meta("stockout")
stats_md = load_stats_tests()
dq_md    = load_dq_report()
d_model, s_model = load_models()

# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.title("StockSense")
st.sidebar.caption("NovaMart decision-support")

if not recs.empty:
    stores = ["All"] + sorted(recs["store_id"].astype(str).unique().tolist())
    cats   = ["All"] + sorted(recs["category"].astype(str).unique().tolist())
    risks  = ["All", "HIGH", "MEDIUM", "LOW"]

    sel_store = st.sidebar.selectbox("Store", stores)
    sel_cat   = st.sidebar.selectbox("Category", cats)
    sel_risk  = st.sidebar.selectbox("Risk level", risks)

    view = recs.copy()
    if sel_store != "All":
        view = view[view["store_id"].astype(str) == sel_store]
    if sel_cat != "All":
        view = view[view["category"] == sel_cat]
    if sel_risk != "All":
        view = view[view["risk_level"] == sel_risk]
else:
    view = recs

st.sidebar.markdown("---")
st.sidebar.caption(f"Recommendations: {len(recs):,}")
st.sidebar.caption(f"Filtered rows: {len(view):,}")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tabs = st.tabs([
    "Executive Summary",
    "Demand Intelligence",
    "Inventory Risk",
    "Manager Action Centre",
    "Model Performance",
    "Explainability",
    "What-If Simulator",
])

# ============================================================
# 1. Executive Summary
# ============================================================
with tabs[0]:
    st.header("Executive Summary")
    if kpis.empty:
        st.warning("No KPI file. Run Phase 2.")
    else:
        kpi_dict = dict(zip(kpis["kpi"], kpis["value"]))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"Rs {float(kpi_dict.get('Revenue (Rs)', 0)):,.0f}")
        c2.metric("Units sold", f"{int(kpi_dict.get('Units sold', 0)):,}")
        c3.metric("Stock-out rate",
                  f"{float(kpi_dict.get('Stockout rate', 0))*100:.2f}%")
        c4.metric("Est. lost sales",
                  f"Rs {float(kpi_dict.get('Estimated lost sales (Rs)', 0)):,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Inventory turnover",
                  f"{float(kpi_dict.get('Inventory turnover', 0)):.2f}")
        c6.metric("Days of inventory",
                  f"{float(kpi_dict.get('Median days of inventory', 0)):.1f}")
        c7.metric("Promotion lift",
                  f"{float(kpi_dict.get('Promotion lift', 0))*100:.1f}%")
        if not recs.empty:
            c8.metric("Products at HIGH risk",
                      f"{(recs['risk_level']=='HIGH').sum():,}")

    st.subheader("Top 5 high-risk items")
    if not recs.empty:
        top5 = recs[recs["risk_level"] == "HIGH"].nlargest(5, "stockout_prob")
        if top5.empty:
            top5 = recs.nlargest(5, "stockout_prob")
        st.dataframe(
            top5[["store_id", "product_id", "forecast_7d", "current_stock",
                  "stockout_prob", "risk_level", "reorder_qty",
                  "manager_action"]],
            use_container_width=True, hide_index=True)

    st.subheader("Revenue trend")
    if not master.empty:
        rev = (master.groupby(master["date"].dt.to_period("W").astype(str))
                     ["revenue"].sum().reset_index())
        fig = px.line(rev, x="date", y="revenue", title="Weekly revenue")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2. Demand Intelligence
# ============================================================
with tabs[1]:
    st.header("Demand Intelligence")
    if feats.empty:
        st.warning("features.csv missing. Run Phase 3.")
    else:
        c1, c2 = st.columns(2)
        store_opt = c1.selectbox("Store",
                                 sorted(feats["store_id"].astype(str).unique()))
        prod_opt  = c2.selectbox("Product",
                                 sorted(feats["product_id"].astype(str).unique()))
        sub = feats[(feats["store_id"].astype(str) == store_opt) &
                    (feats["product_id"].astype(str) == prod_opt)].sort_values("date")
        if not sub.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["units_sold"],
                                     name="Actual", mode="lines"))
            fig.add_trace(go.Scatter(x=sub["date"], y=sub["rolling_mean_7"],
                                     name="Rolling mean 7", mode="lines"))
            fig.update_layout(title=f"Actual vs rolling mean: {store_opt}/{prod_opt}")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Category trend")
        cat_trend = (master.assign(week=master["date"].dt.to_period("W").astype(str))
                          .groupby(["week", "category"])["units_sold"].sum()
                          .reset_index())
        fig2 = px.line(cat_trend, x="week", y="units_sold", color="category",
                       title="Weekly units by category")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Store trend")
        store_trend = (master.assign(week=master["date"].dt.to_period("W").astype(str))
                             .groupby(["week", "store_id"])["units_sold"].sum()
                             .reset_index())
        fig3 = px.line(store_trend, x="week", y="units_sold", color="store_id",
                       title="Weekly units by store")
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# 3. Inventory Risk
# ============================================================
with tabs[2]:
    st.header("Inventory Risk")
    if recs.empty:
        st.warning("recommendations.csv missing. Run Phase 7.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("HIGH",   int((recs['risk_level'] == 'HIGH').sum()))
        c2.metric("MEDIUM", int((recs['risk_level'] == 'MEDIUM').sum()))
        c3.metric("LOW",    int((recs['risk_level'] == 'LOW').sum()))

        st.subheader("Risk heatmap (store x category)")
        if not view.empty:
            pivot = view.pivot_table(index="store_id", columns="category",
                                     values="stockout_prob", aggfunc="mean")
            fig = px.imshow(pivot, aspect="auto",
                            color_continuous_scale="RdYlGn_r",
                            title="Mean stock-out probability")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Filterable risk table")
        st.dataframe(
            view[["store_id", "product_id", "category", "forecast_7d",
                  "current_stock", "incoming_stock", "stockout_prob",
                  "risk_level", "reorder_qty"]]
                .sort_values("stockout_prob", ascending=False),
            use_container_width=True, hide_index=True)

# ============================================================
# 4. Manager Action Centre
# ============================================================
with tabs[3]:
    st.header("Manager Action Centre")
    if recs.empty:
        st.warning("recommendations.csv missing.")
    else:
        st.caption("Sorted by risk level then reorder quantity")
        table = view[["store_id", "product_id", "category", "risk_level",
                      "stockout_prob", "forecast_7d", "current_stock",
                      "incoming_stock", "reorder_qty", "manager_action",
                      "top_reasons_str"]]
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.download_button("Download action list (CSV)",
                           data=view.to_csv(index=False).encode("utf-8"),
                           file_name="stocksense_actions.csv",
                           mime="text/csv")

        st.subheader("Recommendation card")
        if not view.empty:
            options = view.apply(
                lambda r: f"{r['store_id']} | {r['product_id']}", axis=1).tolist()
            sel = st.selectbox("Pick a store / product", options,
                               key="rec_card_sel")
            row = view.iloc[options.index(sel)]
            st.markdown(f"### {row['store_id']} - {row['product_id']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted 7-day demand", f"{int(row['forecast_7d'])}")
            c2.metric("Current stock", f"{int(row['current_stock'])}")
            c3.metric("Incoming stock", f"{int(row['incoming_stock'])}")
            c4, c5 = st.columns(2)
            c4.metric("Stock-out probability", f"{row['stockout_prob']*100:.1f}%")
            c5.metric("Risk", row["risk_level"])
            st.markdown(f"**Recommended reorder:** {int(row['reorder_qty'])} units")
            if row.get("top_reasons_str"):
                st.markdown(f"**Why?** {row['top_reasons_str']}")
            st.success(row["manager_action"])

# ============================================================
# 5. Model Performance
# ============================================================
with tabs[4]:
    st.header("Model Performance")
    st.subheader("Demand forecast model comparison")
    if not fcmp.empty:
        st.dataframe(fcmp, use_container_width=True, hide_index=True)
    if meta_d:
        st.caption(f"Best demand model: {meta_d.get('best_model')}")
        st.json(meta_d.get("test_metrics", {}))

    st.subheader("Stock-out classification comparison")
    if not scmp.empty:
        st.dataframe(scmp, use_container_width=True, hide_index=True)
    if meta_s:
        st.caption(f"Best stock-out model: {meta_s.get('best_model')} "
                   f"| threshold = {meta_s.get('threshold')}")
        st.json(meta_s.get("test_metrics", {}))

    if not scmp.empty and "ROC_AUC" in scmp.columns:
        st.subheader("Precision vs Recall trade-off")
        fig = px.scatter(scmp, x="Recall", y="Precision", text="model",
                         size="F1", color="model",
                         title="Precision vs Recall on validation")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 6. Explainability
# ============================================================
with tabs[5]:
    st.header("Explainability")
    st.subheader("Global feature importance - demand model")
    if not imp_d.empty:
        fig = px.bar(imp_d.head(15).iloc[::-1], x="importance", y="feature",
                     orientation="h", title="Top 15 drivers (permutation)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global feature importance - stock-out model")
    if not imp_s.empty:
        fig = px.bar(imp_s.head(15).iloc[::-1], x="importance", y="feature",
                     orientation="h", title="Top 15 drivers (permutation)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why is this product at risk?")
    if not recs.empty:
        opts = recs[recs["risk_level"].isin(["HIGH", "MEDIUM"])]
        if opts.empty:
            opts = recs
        options = opts.apply(
            lambda r: f"{r['store_id']} | {r['product_id']}", axis=1).tolist()
        sel = st.selectbox("Pick a store / product ", options, key="explain_sel")
        row = opts.iloc[options.index(sel)]
        st.markdown(f"**Risk: {row['risk_level']}** "
                    f"(prob={row['stockout_prob']*100:.1f}%)")
        if row.get("top_reasons_str"):
            reasons = [x.strip() for x in str(row["top_reasons_str"]).split(";")
                       if x.strip()]
            parsed = []
            for r in reasons:
                if "(+" in r:
                    phrase, pct = r.rsplit("(+", 1)
                    pct = pct.replace("%)", "")
                    parsed.append((phrase.strip(), float(pct)))
                else:
                    parsed.append((r, 0.0))
            if parsed:
                df_r = pd.DataFrame(parsed, columns=["reason", "impact_pct"])
                fig = px.bar(df_r.iloc[::-1], x="impact_pct", y="reason",
                             orientation="h",
                             title="Positive drivers (% of contribution)")
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Statistical tests report"):
        st.markdown(stats_md or "Not available.")
    with st.expander("Data quality report"):
        st.markdown(dq_md or "Not available.")

# ============================================================
# 7. What-If Simulator
# ============================================================
with tabs[6]:
    st.header("What-If Simulator")
    st.caption("Modify inputs and see how forecast, risk and reorder change.")

    if d_model is None or s_model is None or feats.empty:
        st.warning("Models not available. Run Phase 4 and Phase 5.")
    else:
        f = feats.copy()
        f["store_id"]   = f["store_id"].astype(str)
        f["product_id"] = f["product_id"].astype(str)

        c1, c2 = st.columns(2)
        store_sel = c1.selectbox("Store", sorted(f["store_id"].unique()),
                                 key="whatif_store")
        prod_sel  = c2.selectbox("Product",
                                 sorted(f.loc[f["store_id"] == store_sel,
                                              "product_id"].unique()),
                                 key="whatif_prod")

        base = f[(f["store_id"] == store_sel) &
                 (f["product_id"] == prod_sel)].sort_values("date").tail(1)
        if base.empty:
            st.warning("No row for that combination.")
        else:
            row = base.iloc[0].copy()
            feature_cols = meta_d.get("features", list(f.columns))

            discount_delta  = st.slider("Discount change (pct points)",
                                        -20, 30, 0, 5, key="w_disc")
            delay_days      = st.slider("Supplier delay (extra days)",
                                        0, 7, 0, 1, key="w_delay")
            festival_uplift = st.slider("Festival demand uplift (%)",
                                        0, 100, 0, 5, key="w_fest")

            row_mod = row.copy()
            if "discount_pct" in row_mod:
                row_mod["discount_pct"] = float(row_mod["discount_pct"]) + discount_delta
            if "lead_days" in row_mod:
                row_mod["lead_days"] = float(row_mod["lead_days"]) + delay_days
            for c in ["lag_1", "lag_7", "lag_14", "lag_28",
                      "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]:
                if c in row_mod and pd.notna(row_mod[c]):
                    row_mod[c] = float(row_mod[c]) * (1 + festival_uplift / 100.0)

            X_row = pd.DataFrame([row_mod[feature_cols].to_dict()])
            fc = float(d_model.predict(X_row)[0])
            pr = s_model.predict_proba(X_row)
            prob = float(pr[0, 1]) if pr.shape[1] > 1 else 0.0

            risk = ("HIGH" if prob >= meta_s.get("risk_high", 0.7)
                    else "MEDIUM" if prob >= meta_s.get("risk_medium", 0.4)
                    else "LOW")

            sigma  = float(row_mod.get("rolling_std_28", 0) or 0)
            lead   = float(row_mod.get("lead_days", 2) or 2)
            safety = 1.65 * sigma * np.sqrt(max(lead, 1))
            rec_stock = max(fc, 0) + safety
            reorder = max(0, rec_stock
                          - float(row_mod.get("closing", 0) or 0)
                          - float(row_mod.get("received", 0) or 0))

            k1, k2, k3 = st.columns(3)
            k1.metric("Forecast 7d", f"{fc:.0f}")
            k2.metric("Stock-out prob", f"{prob*100:.1f}%")
            k3.metric("Risk", risk)

            k4, k5, k6 = st.columns(3)
            k4.metric("Safety stock", f"{safety:.0f}")
            k5.metric("Recommended stock", f"{rec_stock:.0f}")
            k6.metric("Reorder qty", f"{reorder:.0f}")

            st.info("Compare with baseline (no changes) by moving sliders to zero.")