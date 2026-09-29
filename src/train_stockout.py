"""
Phase 5 - Model 2: Stock-out risk classification (next 7 days).

Models (permitted only): Decision Tree, Random Forest, XGBoost, KNN, SVC, GaussianNB.
Class imbalance handled via class_weight / scale_pos_weight.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.
Risk levels: HIGH >= 0.70, MEDIUM >= 0.40, LOW < 0.40.
Outputs:
  reports/model_comparison_stockout.csv
  models/stockout_model.joblib
  models/stockout_model_meta.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _loader import load_sibling

cfg = load_sibling("config")
FEATURES_FILE   = cfg.FEATURES_FILE
SPLIT_META      = cfg.SPLIT_META
STOCKOUT_MODEL  = cfg.STOCKOUT_MODEL
STOCKOUT_META   = cfg.STOCKOUT_META
STOCKOUT_CMP    = cfg.STOCKOUT_CMP
SEED            = cfg.SEED
RISK_HIGH       = cfg.RISK_HIGH
RISK_MEDIUM     = cfg.RISK_MEDIUM

TARGET = "stockout_flag"

DROP_COLS = ["date", "next_7_day_demand", "stockout_flag", "stockout_day",
             "true_stockout_day", "near_stockout_day",
             "available", "revenue"]

FORCE_CATEGORICAL = ["store_id", "product_id", "category", "store_type",
                     "region", "city", "sub_category", "brand", "supplier_id"]


def evaluate_clf(name, y_true, y_prob, threshold=0.5, t_train=None):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": threshold,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else np.nan,
        "PR_AUC":    round(average_precision_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "train_time_sec": round(t_train, 2) if t_train is not None else np.nan,
    }


def best_threshold_f2(y_true, y_prob):
    best_t, best_f2 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.05):
        pred = (y_prob >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        if tp == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        if prec + rec == 0:
            continue
        f2 = 5 * prec * rec / (4 * prec + rec)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t, best_f2


def split_feature_columns(df):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    num_cols, cat_cols = [], []
    for c in feature_cols:
        if c in FORCE_CATEGORICAL:
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return feature_cols, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def get_proba(pipe, X):
    """Return probability of positive class; return zeros if model saw only one class."""
    proba = pipe.predict_proba(X)
    if proba.shape[1] < 2:
        return np.zeros(len(X))
    return proba[:, 1]


def main():
    print("=" * 72)
    print("PHASE 5 - MODEL 2: STOCK-OUT RISK CLASSIFICATION")
    print("=" * 72)

    df = pd.read_csv(FEATURES_FILE, parse_dates=["date"])
    meta = json.loads(Path(SPLIT_META).read_text())

    train = df[df["date"] <= pd.Timestamp(meta["train_end"])].copy()
    val   = df[(df["date"] >= pd.Timestamp(meta["val_start"])) &
               (df["date"] <= pd.Timestamp(meta["val_end"]))].copy()
    test  = df[df["date"] >= pd.Timestamp(meta["test_start"])].copy()

    print(f"train rows={len(train):,}  val rows={len(val):,}  test rows={len(test):,}")
    print(f"stockout_flag balance (train): {train[TARGET].mean():.4f}")
    print(f"stockout_flag balance (val):   {val[TARGET].mean():.4f}")
    print(f"stockout_flag balance (test):  {test[TARGET].mean():.4f}")

    if train[TARGET].sum() == 0 or train[TARGET].mean() > 0.95:
        raise RuntimeError(
            "stockout_flag has degenerate class balance after fallback. "
            "Relax the near-stock-out threshold in features.py and re-run Phase 3."
        )

    feature_cols, num_cols, cat_cols = split_feature_columns(df)
    print(f"feature columns: {len(feature_cols)}  "
          f"(numeric={len(num_cols)}, categorical={len(cat_cols)})")

    X_tr, y_tr = train[feature_cols], train[TARGET].values.astype(int)
    X_va, y_va = val[feature_cols],   val[TARGET].values.astype(int)
    X_te, y_te = test[feature_cols],  test[TARGET].values.astype(int)

    pre = build_preprocessor(num_cols, cat_cols)

    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    spw = neg / pos

    models = {
        "DecisionTree": DecisionTreeClassifier(
            random_state=SEED, max_depth=12, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=14, n_jobs=-1,
            random_state=SEED, class_weight="balanced"),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=spw, random_state=SEED, n_jobs=-1,
            tree_method="hist", eval_metric="logloss"),
        "KNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "SVC": SVC(C=1.0, kernel="rbf", probability=True,
                   class_weight="balanced", random_state=SEED),
        "GaussianNB": GaussianNB(),
    }

    results = []
    best_name, best_pr_auc, best_model = None, -np.inf, None

    for name, est in models.items():
        print(f"\n-- {name}")
        if name in {"KNN", "SVC"} and len(X_tr) > 15000:
            idx = np.random.RandomState(SEED).choice(len(X_tr), 15000, replace=False)
            Xt, yt = X_tr.iloc[idx], y_tr[idx]
            print(f"   subsampled train to {len(Xt):,} rows")
        else:
            Xt, yt = X_tr, y_tr

        pipe = Pipeline([("pre", pre), ("model", est)])
        t0 = time.time()
        try:
            pipe.fit(Xt, yt)
        except Exception as e:
            print(f"   SKIP {name}: {e}")
            continue
        t_train = time.time() - t0

        # guard against degenerate classes
        classes = getattr(pipe.named_steps["model"], "classes_", None)
        if classes is None or len(classes) < 2:
            print(f"   SKIP {name}: model saw only one class")
            continue

        prob_val = get_proba(pipe, X_va)
        row = evaluate_clf(name, y_va, prob_val, 0.5, t_train)
        results.append(row)
        print(f"   val F1={row['F1']:.3f}  Recall={row['Recall']:.3f}  "
              f"PR_AUC={row['PR_AUC']}  ROC_AUC={row['ROC_AUC']}  "
              f"time={t_train:.1f}s")

        if row["PR_AUC"] > best_pr_auc:
            best_pr_auc = row["PR_AUC"]
            best_name   = name
            best_model  = pipe

    if best_model is None:
        raise RuntimeError("No model trained successfully with both classes.")

    cmp = pd.DataFrame(results)
    cmp.to_csv(STOCKOUT_CMP, index=False)
    print("\nmodel comparison saved:", STOCKOUT_CMP)
    print(cmp.to_string(index=False))

    print(f"\nBest model on validation: {best_name} (PR_AUC={best_pr_auc})")

    prob_val_best = get_proba(best_model, X_va)
    thr, f2 = best_threshold_f2(y_va, prob_val_best)
    print(f"tuned threshold: {thr:.2f}  (val F2={f2:.4f})")

    X_trva = pd.concat([X_tr, X_va])
    y_trva = np.concatenate([y_tr, y_va])

    if best_name in {"KNN", "SVC"} and len(X_trva) > 15000:
        idx = np.random.RandomState(SEED).choice(len(X_trva), 15000, replace=False)
        X_fit, y_fit = X_trva.iloc[idx], y_trva[idx]
    else:
        X_fit, y_fit = X_trva, y_trva

    pipe_final = Pipeline([("pre", pre), ("model", models[best_name])])
    pipe_final.fit(X_fit, y_fit)

    prob_test = get_proba(pipe_final, X_te)
    final_metrics = evaluate_clf(best_name + "_refit", y_te, prob_test, thr)
    print("Refit test metrics:", final_metrics)

    joblib.dump(pipe_final, STOCKOUT_MODEL)
    meta_out = {
        "best_model":     best_name,
        "features":       feature_cols,
        "cat_features":   cat_cols,
        "num_features":   num_cols,
        "target":         TARGET,
        "threshold":      float(thr),
        "risk_high":      RISK_HIGH,
        "risk_medium":    RISK_MEDIUM,
        "test_metrics":   final_metrics,
        "split":          meta,
        "seed":           SEED,
    }
    Path(STOCKOUT_META).write_text(json.dumps(meta_out, indent=2, default=str))
    print("saved:", STOCKOUT_MODEL)
    print("saved:", STOCKOUT_META)


if __name__ == "__main__":
    main()