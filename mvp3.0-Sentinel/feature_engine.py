"""
feature_engine.py
=================
Converts a raw streamed transaction dict into a clean feature dict
ready for the ML model. Mirrors the preprocessing done during training.
"""

from __future__ import annotations
import pickle
import numpy as np
import pandas as pd

ENCODERS_PATH = "encoders.pkl"
CATEGORICAL_COLS = ["device", "location", "merchant"]

# Features fed to the model (order matters for SHAP)
MODEL_FEATURE_COLS = [
    "amount",
    "device",
    "location",
    "merchant",
    "login_attempts",
    "balance",
    "time_since_last_txn",
    "txn_velocity",
    "is_known_device",
    "amount_to_balance_ratio",
]


def load_encoders(path: str = ENCODERS_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def engineer_features(txn: dict, encoders: dict | None = None) -> dict:
    """
    Takes a raw transaction dict, returns a processed feature dict.
    If encoders are provided, label-encodes categoricals.
    """
    features = {}

    # ── Numerical passthrough ──────────────────────────────────
    features["amount"]              = float(txn.get("amount", 0))
    features["login_attempts"]      = int(txn.get("login_attempts", 0))
    features["balance"]             = float(txn.get("balance", 0))
    features["time_since_last_txn"] = float(txn.get("time_since_last_txn", 0))
    features["txn_velocity"]        = int(txn.get("txn_velocity", 0))
    features["is_known_device"]     = int(txn.get("is_known_device", 1))

    # ── Derived feature ────────────────────────────────────────
    features["amount_to_balance_ratio"] = features["amount"] / (features["balance"] + 1)

    # ── Categorical encoding ───────────────────────────────────
    for col in CATEGORICAL_COLS:
        raw_val = txn.get(col, "unknown")
        if encoders:
            le = encoders.get(col)
            if le:
                try:
                    features[col] = int(le.transform([raw_val])[0])
                except ValueError:
                    # Unseen label → encode as -1 (safe fallback)
                    features[col] = -1
            else:
                features[col] = -1
        else:
            features[col] = raw_val      # raw string (used during training fit)

    return features


def features_to_dataframe(features: dict) -> pd.DataFrame:
    """Convert a single feature dict to a model-ready DataFrame row."""
    return pd.DataFrame([features])[MODEL_FEATURE_COLS]


# ─────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_txn = {
        "user_id": "U001",
        "amount": 3563.71,
        "device": "mobile_ios",
        "location": "Hyderabad",
        "merchant": "IRCTC",
        "timestamp": "2024-03-15 14:22:00",
        "login_attempts": 0,
        "balance": 38535.42,
        "time_since_last_txn": 0.0,
        "txn_velocity": 0,
        "is_known_device": 1,
    }
    feats = engineer_features(sample_txn)
    print("✅ Engineered features:")
    for k, v in feats.items():
        print(f"   {k:<30} = {v}")