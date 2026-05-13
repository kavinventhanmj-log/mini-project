"""
predict.py
==========
Loads the trained model + encoders and runs inference on a
single transaction dict. Returns risk_score (0–1).
Used by both the decision engine and explainability layer.
"""

import pickle
import numpy as np
import pandas as pd
from feature_engine import engineer_features, features_to_dataframe, load_encoders

MODEL_PATH    = "model.pkl"
ENCODERS_PATH = "encoders.pkl"

# ─────────────────────────────────────────────────────
# LOAD ARTEFACTS (once at module import — fast inference)
# ─────────────────────────────────────────────────────
_model    = None
_encoders = None


def _load_artefacts():
    global _model, _encoders
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if _encoders is None:
        _encoders = load_encoders(ENCODERS_PATH)


# ─────────────────────────────────────────────────────
# CORE PREDICTION FUNCTION
# ─────────────────────────────────────────────────────
def predict_transaction(transaction_dict: dict) -> tuple[float, pd.DataFrame]:
    """
    Args:
        transaction_dict : raw transaction (from streamer or API)

    Returns:
        risk_score  (float, 0–1)
        feature_df  (DataFrame, single row — used by explain.py)
    """
    _load_artefacts()

    # Feature engineering (with encoding)
    features   = engineer_features(transaction_dict, encoders=_encoders)
    feature_df = features_to_dataframe(features)

    # Inference
    risk_score = float(_model.predict_proba(feature_df)[0][1])
    return risk_score, feature_df