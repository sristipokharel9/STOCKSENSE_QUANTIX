"""
EDA charts for StockSense. Each chart answers a business question.
Saves to reports/figures/*.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
MASTER_DATASET = cfg.MASTER_DATASET
FIGURES_DIR    = cfg.FIGURES_DIR


def save(fig, name):
    p = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("saved:", p)


def main():
    df = pd.read_csv(MASTER_DATASET)
    df["date"] = pd.to_datetime(df["date"])
    df["weekend_flag"] = df["date"].dt.weekday >= 5
    df["available"] = df["opening"] + df["received"]
    df["stockout"]  = ((df["closing"] <= 0) | (df["sold"] >= df["available"])).astype(int)

    # Q1
    cat_rev = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    cum = cat_rev.cumsum() / cat_rev.sum()
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(cat_rev.index, cat_rev.values, color="steelblue")
    ax1.set_ylabel("Revenue")
    ax2 = ax1.twinx()
    ax2.plot(cat_rev.index, cum.values * 100, color="darkred", marker="o")
    ax2.set_ylabel("Cumulative %")
    ax1.set_title("Q1: Which categories drive revenue? (Pareto)")
    ax1.tick_params(axis="x", rotation=30)
    save(fig, "q1_category_pareto.png")

    # Q2
    store_week = (df.assign(week=df["date"].dt.to_period("W").astype(str))
                    .groupby(["week", "store_id"])["revenue"].sum().reset_index())
    fig, ax = plt.subplots(figsize=(10, 5))
    for s, g in store_week.groupby("store_id"):
        ax.plot(g["week"], g["revenue"], label=str(s))
    ax.set_title("Q2: Weekly revenue trend per store")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(title="store", fontsize=7)
    save(fig, "q2_store_weekly_trend.png")

    # Q3
    promo = df.groupby(["category", "promotion_flag"])["units_sold"].mean().unstack()
    promo["lift"] = (promo[1] - promo[0]) / promo[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    promo["lift"].sort_values().plot.bar(ax=ax, color="seagreen")
    ax.set_ylabel("Promotion lift (relative)")
    ax.set_title("Q3: Promotion lift by category")
    save(fig, "q3_promo_lift_category.png")

    # Q4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x="category", y="units_sold", hue="weekend_flag",
                ax=ax, errorbar=None)
    ax.set_title("Q4: Weekend vs weekday mean demand by category")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "q4_weekend_vs_weekday.png")

    # Q5
    cov = df.groupby("product_id")["units_sold"].agg(["mean", "std"]).dropna()
    cov["cv"] = cov["std"] / cov["mean"].replace(0, np.nan)
    cov = cov.sort_values("cv", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    cov["cv"].iloc[::-1].plot.barh(ax=ax, color="indianred")
    ax.set_title("Q5: Top 20 most volatile products (CoV)")
    save(fig, "q5_volatility_products.png")

    # Q6
    hm = df.pivot_table(values="stockout", index="store_id",
                        columns="category", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(hm, annot=True, fmt=".2f", cmap="Reds", ax=ax)
    ax.set_title("Q6: Stock-out rate - store x category")
    save(fig, "q6_stockout_heatmap.png")


if __name__ == "__main__":
    main()