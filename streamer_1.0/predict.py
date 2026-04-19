"""
models/predict.py
-----------------
ML MODEL INTEGRATION POINT
===========================

INSTRUCTIONS
------------
Replace the body of `predict_transaction` with your real model.

Example (scikit-learn):
    import joblib, numpy as np
    _model = joblib.load("models/fraud_rf.pkl")

    def predict_transaction(txn: dict) -> float:
        features = list(txn["features"].values())   # V1–V28
        X = np.array(features).reshape(1, -1)
        return float(_model.predict_proba(X)[0, 1])

Example (XGBoost):
    import xgboost as xgb, numpy as np
    _model = xgb.Booster(); _model.load_model("models/fraud_xgb.json")

    def predict_transaction(txn: dict) -> float:
        features = list(txn["features"].values())
        dmat = xgb.DMatrix([features])
        return float(_model.predict(dmat)[0])

The processor auto-discovers this file.  
Once this file exists and predict_transaction is callable, 
the mock scorer is bypassed automatically — no other changes needed.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Load your model here (once, at import time)
# ---------------------------------------------------------------------------

# _model = joblib.load("models/your_model.pkl")   # ← uncomment when ready


# ---------------------------------------------------------------------------
# Prediction function  (REQUIRED SIGNATURE — do not rename)
# ---------------------------------------------------------------------------

def predict_transaction(txn: dict) -> float:
    """
    Score a transaction for fraud probability.

    Parameters
    ----------
    txn : dict
        Structured transaction event with keys:
            transaction_id, timestamp, amount, features (dict V1–V28), ...

    Returns
    -------
    float in [0.0, 1.0]
        Probability that the transaction is fraudulent.
        0.0 = definitely legitimate, 1.0 = definitely fraud.
    """
    # ── STUB: Replace below with real model inference ─────────────────
    features = list(txn.get("features", {}).values())   # V1–V28 as list
    amount   = txn.get("amount", 0.0)

    # Placeholder heuristic (DELETE this when using a real model)
    score = float(np.clip(amount / 5000.0, 0.0, 1.0))
    return round(score, 4)
    # ─────────────────────────────────────────────────────────────────
