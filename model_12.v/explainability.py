"""
explainability.py
-----------------
SHAP-based local explanation engine.

Provides:
  • Per-transaction top-N feature contributions
  • Analyst-friendly natural language summary
  • Structured output for UI integration

Designed to be imported by the inference engine only.
SHAP is NOT imported at stream startup — only when explain() is called.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from config import SHAP_TOP_N_FEATURES


# ── SHAP Explainer (lazy-loaded) ───────────────────────────────────────────

_explainer = None


def init_explainer(model, X_background: pd.DataFrame):
    """
    Initialize SHAP TreeExplainer with a background dataset.
    Call once after training.

    Parameters
    ----------
    model        : trained LightGBM model
    X_background : sample of training data (100–200 rows is enough)
    """
    global _explainer
    import shap
    _explainer = shap.TreeExplainer(model)
    print(f"[EXPLAINABILITY] SHAP TreeExplainer initialized "
          f"(background: {len(X_background)} rows)")


def explain_transaction(
    row: pd.DataFrame,
    feature_names: List[str],
    risk_level: str,
    decision: str,
    score: float,
    top_n: int = SHAP_TOP_N_FEATURES,
) -> Dict[str, Any]:
    """
    Generate a local SHAP explanation for a single transaction.

    Parameters
    ----------
    row           : single-row DataFrame (engineered features)
    feature_names : list of feature column names
    risk_level    : "low" | "medium" | "high"
    decision      : "allow" | "review" | "block"
    score         : fraud probability (0-1)
    top_n         : number of top features to return

    Returns
    -------
    dict with keys: summary, top_features, shap_values, visualization_ready
    """
    if _explainer is None:
        return _fallback_explanation(score, risk_level, decision)

    import shap

    X = row[feature_names].values
    shap_vals = _explainer.shap_values(X)

    # For binary classifiers, shap_values may be list [neg_class, pos_class]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_vals = shap_vals.flatten()

    # Build feature contribution table
    contributions = [
        {"feature": f, "shap_value": round(float(v), 4),
         "direction": "increases_risk" if v > 0 else "decreases_risk"}
        for f, v in zip(feature_names, shap_vals)
    ]
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_features = contributions[:top_n]

    summary = _build_summary(top_features, score, risk_level, decision)

    return {
        "summary":            summary,
        "top_features":       top_features,
        "all_contributions":  contributions,
        "visualization_ready": _viz_payload(top_features),
    }


# ── Natural Language Summary ───────────────────────────────────────────────

def _build_summary(top_features: List[dict], score: float,
                   risk_level: str, decision: str) -> str:
    risk_words = {
        "low":    "low fraud risk",
        "medium": "moderate fraud risk",
        "high":   "high fraud risk",
    }
    decision_words = {
        "allow":  "Transaction approved.",
        "review": "Transaction flagged for analyst review.",
        "block":  "Transaction blocked.",
    }

    # Pick top 2 positive drivers
    drivers = [f["feature"] for f in top_features
               if f["direction"] == "increases_risk"][:2]
    reducers = [f["feature"] for f in top_features
                if f["direction"] == "decreases_risk"][:1]

    parts = [decision_words.get(decision, "")]

    if drivers:
        readable = [_readable_feature(f) for f in drivers]
        parts.append(f"Risk driven by: {', '.join(readable)}.")

    if reducers:
        readable = [_readable_feature(f) for f in reducers]
        parts.append(f"Mitigated by: {', '.join(readable)}.")

    parts.append(
        f"Fraud probability {score:.1%} → {risk_words.get(risk_level, '')}."
    )

    return " ".join(parts)


def _readable_feature(feat: str) -> str:
    """Convert feature names to analyst-friendly labels."""
    mapping = {
        "log_amount":       "transaction amount (log-scaled)",
        "Amount":           "transaction amount",
        "pca_anomaly_score":"PCA anomaly deviation",
        "v_std":            "feature variability",
        "v_max_abs":        "feature magnitude",
        "is_night":         "night-time transaction",
        "hour_of_day":      "transaction hour",
        "amount_zscore":    "amount deviation from norm",
        "is_large_amount":  "large transaction flag",
        "is_micro_amount":  "micro transaction flag",
    }
    if feat in mapping:
        return mapping[feat]
    if feat.startswith("V"):
        return f"PCA component {feat}"
    return feat.replace("_", " ")


def _viz_payload(top_features: List[dict]) -> List[dict]:
    """Structured data for bar chart rendering in a UI."""
    return [
        {
            "label":  _readable_feature(f["feature"]),
            "value":  f["shap_value"],
            "color":  "#e74c3c" if f["direction"] == "increases_risk" else "#2ecc71",
        }
        for f in top_features
    ]


# ── Fallback (no SHAP available) ──────────────────────────────────────────

def _fallback_explanation(score: float, risk_level: str,
                          decision: str) -> Dict[str, Any]:
    return {
        "summary": (
            f"Transaction scored {score:.1%} fraud probability → "
            f"{risk_level} risk → {decision}. "
            "SHAP explainer not initialized; feature-level detail unavailable."
        ),
        "top_features":       [],
        "all_contributions":  [],
        "visualization_ready": [],
    }
