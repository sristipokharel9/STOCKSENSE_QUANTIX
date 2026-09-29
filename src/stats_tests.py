"""
Statistical tests for StockSense.
Writes: reports/statistical_tests.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
MASTER_DATASET = cfg.MASTER_DATASET
STAT_TESTS     = cfg.STAT_TESTS


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    pooled = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled else np.nan


def rank_biserial(u, n1, n2):
    return float(1 - (2 * u) / (n1 * n2))


def cramers_v(confusion):
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.sum()
    r, k = confusion.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def main():
    df = pd.read_csv(MASTER_DATASET)
    df["weekend_flag"] = pd.to_datetime(df["date"]).dt.weekday >= 5
    df["available"] = df["opening"] + df["received"]
    df["stockout"]  = ((df["closing"] <= 0) | (df["sold"] >= df["available"])).astype(int)

    lines = ["# StockSense - Statistical Tests\n",
             "Alpha = 0.05 for all tests.\n"]

    promo  = df.loc[df["promotion_flag"] == 1, "units_sold"].dropna()
    normal = df.loc[df["promotion_flag"] == 0, "units_sold"].dropna()
    u, p   = stats.mannwhitneyu(promo, normal, alternative="two-sided")
    d      = cohens_d(promo, normal)
    rb     = rank_biserial(u, len(promo), len(normal))
    lines += [
        "\n## Test 1 - Do promotions significantly change units sold?",
        "- H0: distribution of units_sold is the same with and without promotion.",
        "- H1: distributions differ.",
        "- Test: Mann-Whitney U.",
        f"- n(promo) = {len(promo):,}, n(normal) = {len(normal):,}",
        f"- U = {u:,.1f}, p = {p:.4g}",
        f"- Effect size: Cohen's d = {d:.3f}, rank-biserial = {rb:.3f}",
        f"- Decision: {'REJECT H0' if p < 0.05 else 'FAIL TO REJECT H0'}",
        "- Business reading: promotions raise per-day units sold significantly."
        if p < 0.05 else
        "- Business reading: no significant promotion effect detected in this data.",
    ]

    groups = [g["units_sold"].values for _, g in df.groupby("store_type") if len(g) > 30]
    labels = [k for k, g in df.groupby("store_type") if len(g) > 30]
    if len(groups) >= 2:
        h, p2 = stats.kruskal(*groups)
        lines += [
            "\n## Test 2 - Does mean demand differ across store types?",
            "- H0: demand distribution is identical across store types.",
            "- H1: at least one store type differs.",
            "- Test: Kruskal-Wallis.",
            f"- groups = {labels}",
            f"- H = {h:.3f}, p = {p2:.4g}",
            f"- Decision: {'REJECT H0' if p2 < 0.05 else 'FAIL TO REJECT H0'}",
            "- Business reading: store type is a meaningful segmentation for demand."
            if p2 < 0.05 else
            "- Business reading: store type alone does not explain demand differences.",
        ]

    cont = pd.crosstab(df["promotion_flag"], df["stockout"])
    chi2, p3, _, _ = stats.chi2_contingency(cont)
    v = cramers_v(cont.values)
    lines += [
        "\n## Test 3 - Is stock-out associated with promotion status?",
        "- H0: promotion and stock-out are independent.",
        "- H1: they are associated.",
        "- Test: Chi-square test of independence.",
        f"- chi2 = {chi2:.3f}, p = {p3:.4g}, Cramer's V = {v:.3f}",
        f"- Decision: {'REJECT H0' if p3 < 0.05 else 'FAIL TO REJECT H0'}",
        "- Business reading: promotions raise stock-out risk; plan replenishment."
        if p3 < 0.05 else
        "- Business reading: promotions do not measurably shift stock-out risk here.",
    ]

    wknd = df.loc[df["weekend_flag"], "units_sold"].dropna()
    wkdy = df.loc[~df["weekend_flag"], "units_sold"].dropna()
    u4, p4 = stats.mannwhitneyu(wknd, wkdy, alternative="two-sided")
    d4 = cohens_d(wknd, wkdy)
    lines += [
        "\n## Test 4 - Does weekend demand differ from weekday demand?",
        "- H0: same distribution weekday vs weekend.",
        "- H1: distributions differ.",
        "- Test: Mann-Whitney U.",
        f"- U = {u4:,.1f}, p = {p4:.4g}, Cohen's d = {d4:.3f}",
        f"- Decision: {'REJECT H0' if p4 < 0.05 else 'FAIL TO REJECT H0'}",
        "- Business reading: weekend-specific replenishment planning is warranted."
        if p4 < 0.05 else
        "- Business reading: weekend demand is not statistically different here.",
    ]

    Path(STAT_TESTS).write_text("\n".join(lines))
    print("Statistical tests written to:", STAT_TESTS)


if __name__ == "__main__":
    main()