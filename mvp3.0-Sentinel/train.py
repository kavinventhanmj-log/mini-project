"""
train.py
========
Trains a LightGBM classifier with focus on HIGH PRECISION
(minimising false positives). Saves model + encoders.
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from preprocess import preprocess_training_data

CSV_PATH   = "transactions.csv"
MODEL_PATH = "model.pkl"


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return round(n_neg / n_pos, 4)


import os
import glob

FEEDBACK_DIR = "feedback_logs"

def train(incremental_data=None):
    print("=" * 60)
    print("  TRANSACTION MONITORING — MODEL TRAINING")
    print("=" * 60)

    # ── Load & preprocess ──────────────────────────────────────
    print("\n📂 Loading base dataset...")
    df = pd.read_csv(CSV_PATH)
    
    # Check for consolidated feedback file
    FEEDBACK_FILE = "consolidated_feedback.csv"
    if os.path.exists(FEEDBACK_FILE):
        print(f"   📥 Incorporating human feedback from {FEEDBACK_FILE}...")
        try:
            all_feedback = pd.read_csv(FEEDBACK_FILE)
            
            # Map decisions to labels
            # ALLOW -> Legit (0), BLOCK -> Fraud (1)
            decision_map = {"ALLOW": 0, "BLOCK": 1}
            
            # Use vectorized update if possible
            for _, row in all_feedback.iterrows():
                if row['human_decision'] in decision_map:
                    txn_id = row['transaction_id']
                    label = decision_map[row['human_decision']]
                    df.loc[df['transaction_id'] == txn_id, 'is_fraud'] = label
            
            print(f"   ✅ Base dataset updated with {len(all_feedback)} human labels.")
        except Exception as e:
            print(f"   ⚠️ Error processing feedback: {e}")
    
    # Also keep check for feedback_logs for backward compatibility
    elif os.path.exists(FEEDBACK_DIR):
        feedback_files = glob.glob(os.path.join(FEEDBACK_DIR, "*.csv"))
        if feedback_files:
            # ... (rest of the old logic for fallback)
            feedback_data = []
            for f in feedback_files:
                try:
                    feedback_data.append(pd.read_csv(f))
                except: continue
            if feedback_data:
                all_feedback = pd.concat(feedback_data, ignore_index=True)
                decision_map = {"ALLOW": 0, "BLOCK": 1}
                for _, row in all_feedback.iterrows():
                    if row['human_decision'] in decision_map:
                        df.loc[df['transaction_id'] == row['transaction_id'], 'is_fraud'] = decision_map[row['human_decision']]
                print(f"   ✅ Updated with {len(all_feedback)} human labels from logs.")


    print(f"   Rows: {len(df):,}  |  Fraud: {df['is_fraud'].sum():,}")

    X_train, X_test, y_train, y_test, _ = preprocess_training_data(df)

    # ── Class imbalance ────────────────────────────────────────
    spw = compute_scale_pos_weight(y_train)
    print(f"\n⚖️  scale_pos_weight = {spw}")

    # ── LightGBM ───────────────────────────────────────────────
    print("\n🤖 Training LightGBM...")
    model = lgb.LGBMClassifier(
        objective         = "binary",
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 31,
        max_depth         = -1,
        min_child_samples = 20,
        scale_pos_weight  = spw,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(100)],
    )

    # ── Evaluation ─────────────────────────────────────────────
    print("\n📈 Evaluation on Test Set:")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_pred_proba)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n   Precision  : {precision:.4f}  ← PRIMARY (false positive control)")
    print(f"   Recall     : {recall:.4f}")
    print(f"   F1-Score   : {f1:.4f}")
    print(f"   ROC-AUC    : {auc:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"   FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")

    fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100
    print(f"\n   False Positive Rate : {fp_rate:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    # ── Save model ─────────────────────────────────────────────
    temp_path = MODEL_PATH + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(model, f)
    os.replace(temp_path, MODEL_PATH)
    print(f"💾 Model saved atomically → {MODEL_PATH}")
    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    train()