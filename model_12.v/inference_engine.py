"""
inference_engine.py
-------------------
Real-time inference pipeline.

Loaded once at stream startup.
Called per-transaction with < 5ms overhead.

Output format (per transaction):
{
  "transaction_id":         str,
  "fraud_probability":      float,
  "risk_level":             "low" | "medium" | "high",
  "decision":               "allow" | "review" | "block",
  "confidence_score":       float,
  "explanation_summary":    str,
  "top_features_contributing": [...],
  "rule_warnings":          [...],
  "fp_memory_match":        bool,
}
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from config import MODEL_PATH, SCALER_PATH, MODEL_DIR
from feature_engineering import engineer_features, transaction_to_dataframe
from decision_engine import make_decision, confidence_score, load_thresholds
from rule_engine import run_pre_model_rules, run_post_model_rules
from fp_memory import fp_memory
from explainability import init_explainer, explain_transaction


# ── Model Registry ─────────────────────────────────────────────────────────

_model        = None
_scaler       = None
_feature_cols = None
_ready        = False


def load_model() -> bool:
    """
    Load the trained model, scaler, thresholds, and SHAP explainer.
    Returns True on success.
    """
    global _model, _scaler, _feature_cols, _ready

    if not os.path.exists(MODEL_PATH):
        print(f"[INFERENCE] ⚠  Model not found at '{MODEL_PATH}'. "
              "Run train_model.py first.")
        return False

    # Model
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    _model        = artifact["model"]
    _feature_cols = artifact["feature_cols"]
    load_thresholds()

    # Scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        _scaler = sc["scaler"]

    # SHAP explainer
    bg_path = os.path.join(MODEL_DIR, "shap_background.pkl")
    if os.path.exists(bg_path):
        try:
            with open(bg_path, "rb") as f:
                bg = pickle.load(f)
            init_explainer(_model, bg["background"])
        except Exception as e:
            print(f"[INFERENCE] SHAP init failed: {e}")

    _ready = True
    thresholds = artifact.get("thresholds", {})
    metrics    = artifact.get("metrics", {})
    print(f"[INFERENCE] Model loaded ✓  "
          f"| AUC={metrics.get('auc', '?'):.4f}  "
          f"| FPR={metrics.get('fpr', '?'):.4f}  "
          f"| thresholds: allow<{thresholds.get('allow')}  "
          f"block>{thresholds.get('block')}")
    return True


def is_ready() -> bool:
    return _ready


# ── Core Inference ─────────────────────────────────────────────────────────

def predict_transaction(txn: dict) -> Dict[str, Any]:
    """
    Full inference pipeline for a single transaction.

    Parameters
    ----------
    txn : structured transaction event from the stream engine

    Returns
    -------
    Enriched result dict with all output fields.
    """
    txn_id = txn.get("transaction_id", "UNKNOWN")
    result = _base_result(txn_id)

    # ── Pre-model rules ───────────────────────────────────────────────
    rule_out = run_pre_model_rules(txn)
    result["rule_warnings"] = rule_out.warnings

    if rule_out.hard_decision:
        # Rule overrides model entirely
        score = 1.0 if rule_out.hard_decision == "block" else 0.0
        result["fraud_probability"]   = score
        result["risk_level"]          = "high" if score == 1.0 else "low"
        result["decision"]            = rule_out.hard_decision
        result["confidence_score"]    = 1.0
        result["explanation_summary"] = (
            f"Rule override: {rule_out.warnings[0] if rule_out.warnings else 'rule triggered'}."
        )
        return result

    # ── Feature engineering ───────────────────────────────────────────
    try:
        df_raw = transaction_to_dataframe(txn)
        df_eng = engineer_features(df_raw)
    except Exception as e:
        result["explanation_summary"] = f"Feature engineering failed: {e}"
        return result

    # Align columns with training
    for col in _feature_cols:
        if col not in df_eng.columns:
            df_eng[col] = 0.0
    X = df_eng[_feature_cols].fillna(0.0)

    # Scale
    if _scaler is not None:
        X_scaled = pd.DataFrame(
            _scaler.transform(X), columns=_feature_cols
        )
    else:
        X_scaled = X

    # ── False Positive memory check ───────────────────────────────────
    feature_vec = X_scaled.values.flatten().tolist()
    fp_match = fp_memory.is_known_legitimate(
        feature_vec, txn.get("amount", 0.0)
    )
    if fp_match:
        result["fp_memory_match"] = True
        result["fraud_probability"]   = 0.05
        result["risk_level"]          = "low"
        result["decision"]            = "allow"
        result["confidence_score"]    = 0.90
        result["explanation_summary"] = (
            f"Transaction matches known-legitimate pattern "
            f"(similarity={fp_match['similarity']:.2f}) — auto-allowed."
        )
        return result

    # ── Model inference ───────────────────────────────────────────────
    raw_score = float(_model.predict_proba(X_scaled)[:, 1][0])

    # Apply rule score boost (post-model)
    adjusted_score = run_post_model_rules(txn, raw_score, rule_out)

    # ── Decision ──────────────────────────────────────────────────────
    risk_level, decision = make_decision(adjusted_score)
    conf = confidence_score(adjusted_score)

    # ── Explanation ───────────────────────────────────────────────────
    explanation = explain_transaction(
        row=X_scaled,
        feature_names=_feature_cols,
        risk_level=risk_level,
        decision=decision,
        score=adjusted_score,
    )

    result.update({
        "fraud_probability":         round(adjusted_score, 4),
        "raw_model_score":           round(raw_score, 4),
        "risk_level":                risk_level,
        "decision":                  decision,
        "confidence_score":          conf,
        "explanation_summary":       explanation["summary"],
        "top_features_contributing": explanation["top_features"],
        "visualization_ready":       explanation.get("visualization_ready", []),
    })

    return result


# ── Helpers ────────────────────────────────────────────────────────────────

def _base_result(txn_id: str) -> Dict[str, Any]:
    return {
        "transaction_id":           txn_id,
        "fraud_probability":        0.0,
        "raw_model_score":          0.0,
        "risk_level":               "low",
        "decision":                 "allow",
        "confidence_score":         0.5,
        "explanation_summary":      "Not scored.",
        "top_features_contributing":[],
        "rule_warnings":            [],
        "fp_memory_match":          False,
        "visualization_ready":      [],
    }
