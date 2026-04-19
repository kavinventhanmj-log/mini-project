"""
train_model.py
--------------
Offline training pipeline for the LightGBM fraud detection model.

Run once before starting the stream engine:
    python train_model.py

Produces:
    models/model.pkl        ← trained LightGBM pipeline
    models/thresholds.json  ← optimized ALLOW/BLOCK thresholds
    models/scaler.pkl       ← feature scaler
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")

from config import (
    DATASET_PATH, MODEL_PATH, THRESHOLD_PATH, SCALER_PATH,
    TARGET_COL, TEST_SIZE, RANDOM_STATE, LGBM_PARAMS,
)
from feature_engineering import engineer_features, get_feature_columns
from decision_engine import optimize_thresholds, save_thresholds
from explainability import init_explainer

# ── Colour helpers (terminal) ──────────────────────────────────────────────
def _c(text, code): return f"\033[{code}m{text}\033[0m"
OK  = lambda t: _c(t, "32")
ERR = lambda t: _c(t, "31")
HDR = lambda t: _c(t, "36;1")
BLD = lambda t: _c(t, "1")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(HDR("\n══════════════════════════════════════════════"))
    print(HDR("  FRAUD PREVENTION SYSTEM — TRAINING PIPELINE"))
    print(HDR("══════════════════════════════════════════════\n"))

    # ── 1. Load data ──────────────────────────────────────────────────
    print(BLD("Step 1/7 — Loading dataset …"))
    if not os.path.exists(DATASET_PATH):
        print(ERR(f"  ✗ Dataset not found at '{DATASET_PATH}'"))
        print(ERR("  → Place creditcard.csv in the data/ folder and retry."))
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"  ✓ Loaded {len(df):,} rows  |  fraud rate: "
          f"{df[TARGET_COL].mean()*100:.2f}%")

    # ── 2. Feature engineering ────────────────────────────────────────
    print(BLD("\nStep 2/7 — Engineering features …"))
    df_eng = engineer_features(df)
    feature_cols = get_feature_columns(df_eng)
    print(f"  ✓ {len(feature_cols)} features ready: {feature_cols[:6]} …")

    X = df_eng[feature_cols]
    y = df_eng[TARGET_COL].astype(int)

    # ── 3. Train / test split ─────────────────────────────────────────
    print(BLD("\nStep 3/7 — Splitting data …"))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✓ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 4. Scale features ─────────────────────────────────────────────
    print(BLD("\nStep 4/7 — Scaling features …"))
    scaler = RobustScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols
    )
    with open(SCALER_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"  ✓ Scaler saved → {SCALER_PATH}")

    # ── 5. Train LightGBM ─────────────────────────────────────────────
    print(BLD("\nStep 5/7 — Training LightGBM …"))

    # Compute class weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"  Class ratio → 1:{scale_pos_weight:.0f}  "
          f"(neg={neg:,}  pos={pos:,})")

    params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )
    print(f"  ✓ Best iteration: {model.best_iteration_}")

    # ── 6. Evaluate ───────────────────────────────────────────────────
    print(BLD("\nStep 6/7 — Evaluating …"))
    y_prob  = model.predict_proba(X_test_s)[:, 1]
    y_pred  = (y_prob >= 0.5).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    cm  = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-9)

    print(f"\n  {'Metric':<25} {'Value':>8}")
    print(f"  {'─'*34}")
    print(f"  {'ROC-AUC':<25} {auc:>8.4f}")
    print(f"  {'Avg Precision (PR-AUC)':<25} {ap:>8.4f}")
    print(f"  {'Precision':<25} {prec:>8.4f}")
    print(f"  {'Recall':<25} {rec:>8.4f}")
    print(f"  {'F1-Score':<25} {f1:>8.4f}")
    print(f"  {'False Positive Rate':<25} {fpr:>8.4f}  ← primary KPI")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TP={tp:,}")

    # Optimize thresholds
    allow_t, block_t = optimize_thresholds(y_test.values, y_prob)
    print(f"\n  {'Optimized ALLOW threshold':<25} {allow_t:>8.4f}")
    print(f"  {'Optimized BLOCK threshold':<25} {block_t:>8.4f}")
    save_thresholds(allow_t, block_t)

    # ── SHAP background ───────────────────────────────────────────────
    background = X_train_s.sample(min(200, len(X_train_s)),
                                   random_state=RANDOM_STATE)
    try:
        init_explainer(model, background)
    except Exception as e:
        print(f"  ⚠ SHAP init skipped: {e}")

    # Save SHAP background for inference-time explainer reload
    bg_path = os.path.join(os.path.dirname(MODEL_PATH), "shap_background.pkl")
    with open(bg_path, "wb") as f:
        pickle.dump({"background": background, "feature_cols": feature_cols}, f)

    # ── 7. Save model ─────────────────────────────────────────────────
    print(BLD("\nStep 7/7 — Saving model …"))
    artifact = {
        "model":        model,
        "feature_cols": feature_cols,
        "thresholds":   {"allow": allow_t, "block": block_t},
        "metrics":      {"auc": auc, "ap": ap, "fpr": fpr, "f1": f1},
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  ✓ Model saved → {MODEL_PATH}")

    print(OK("\n  ✓ TRAINING COMPLETE — ready to run stream_engine.py\n"))


if __name__ == "__main__":
    main()
