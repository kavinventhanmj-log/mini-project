"""
explain.py
==========
Wraps predict.py with:
  1. Decision engine   → ALLOW / REVIEW / BLOCK
  2. SHAP explainability → top-4 human-readable reasons
  3. Final output dict
"""

import pickle
import numpy as np
import shap
import predict
from predict import predict_transaction, _load_artefacts, MODEL_PATH

# ─────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────
ALLOW_THRESHOLD  = 0.3
BLOCK_THRESHOLD  = 0.7


# ─────────────────────────────────────────────────────
# DECISION ENGINE
# ─────────────────────────────────────────────────────
def make_decision(risk_score: float) -> str:
    if risk_score < ALLOW_THRESHOLD:
        return "ALLOW"
    elif risk_score < BLOCK_THRESHOLD:
        return "REVIEW"
    else:
        return "BLOCK"


# ─────────────────────────────────────────────────────
# FEATURE → HUMAN REASON MAPPING
# ─────────────────────────────────────────────────────
REASON_MAP = {
    "txn_velocity"           : "High transaction velocity",
    "is_known_device"        : "New device detected",
    "amount"                 : "Unusual transaction amount",
    "amount_to_balance_ratio": "Unusual transaction amount",
    "merchant"               : "Unusual merchant activity",
    "location"               : "Unusual location detected",
    "login_attempts"         : "Multiple login attempts",
    "time_since_last_txn"    : "Rapid successive transactions",
    "balance"                : "Account balance anomaly",
    "device"                 : "Suspicious device pattern",
}

DEFAULT_REASON = "Anomalous transaction pattern"


def _feature_to_reason(feature_name: str) -> str:
    return REASON_MAP.get(feature_name, DEFAULT_REASON)


# ─────────────────────────────────────────────────────
# SHAP EXPLAINER (cached at module level)
# ─────────────────────────────────────────────────────
_explainer = None


def _get_explainer():
    global _explainer
    if _explainer is None:
        _load_artefacts()
        _explainer = shap.TreeExplainer(predict._model)
    return _explainer


# ─────────────────────────────────────────────────────
# CORE PUBLIC FUNCTION
# ─────────────────────────────────────────────────────
def explain_transaction(
    transaction_dict: dict,
    transaction_id: str | None = None,
    top_n: int = 4,
) -> dict:
    """
    Full pipeline: raw transaction → final output with decision + reasons.

    Args:
        transaction_dict : raw transaction dict
        transaction_id   : optional ID string
        top_n            : number of top reasons to return (default 4)

    Returns:
        {
            "transaction_id": ...,
            "risk_score"    : float,
            "decision"      : "ALLOW" | "REVIEW" | "BLOCK",
            "reasons"       : [str, str, str, str]
        }
    """
    # 1. Predict
    risk_score, feature_df = predict_transaction(transaction_dict)

    # 2. Decision
    decision = make_decision(risk_score)

    # 3. SHAP explanations
    explainer  = _get_explainer()
    shap_vals  = explainer.shap_values(feature_df)

    # For binary LightGBM, shap_values returns list [neg_class, pos_class]
    if isinstance(shap_vals, list):
        fraud_shap = shap_vals[1][0]        # positive class, first (only) row
    else:
        fraud_shap = shap_vals[0]           # some versions return 2-D array

    feature_names = feature_df.columns.tolist()

    # 4. Rank by absolute SHAP value, take top_n
    shap_pairs = sorted(
        zip(feature_names, fraud_shap),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    top_features = [feat for feat, _ in shap_pairs[:top_n]]
    reasons      = [_feature_to_reason(f) for f in top_features]

    # Deduplicate while preserving order
    seen, unique_reasons = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique_reasons.append(r)
    # Pad back to top_n if deduplication reduced count
    while len(unique_reasons) < top_n:
        unique_reasons.append(DEFAULT_REASON)

    return {
        "transaction_id": transaction_id,
        "risk_score"    : round(risk_score, 4),
        "decision"      : decision,
        "reasons"       : unique_reasons[:top_n],
    }


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    sample = {
        "user_id"            : "U001",
        "amount"             : 75000.00,
        "device"             : "tablet",
        "location"           : "Mumbai",
        "merchant"           : "Unknown_Merchant",
        "timestamp"          : "2024-05-01 02:15:00",
        "login_attempts"     : 4,
        "balance"            : 12000.00,
        "time_since_last_txn": 0.02,
        "txn_velocity"       : 8,
        "is_known_device"    : 0,
    }

    print("🔍 Running explainability demo...\n")
    result = explain_transaction(sample, transaction_id="TXN_DEMO_001")
    print(json.dumps(result, indent=2))