"""
preprocess.py
=============
Handles all data preprocessing for training:
  - Drop irrelevant columns
  - Handle missing values
  - Create derived features
  - Label-encode categoricals
  - Save encoders
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ENCODERS_PATH    = "encoders.pkl"
CATEGORICAL_COLS = ["device", "location", "merchant"]
DROP_COLS        = ["user_id", "timestamp"]
TARGET_COL       = "is_fraud"

MODEL_FEATURE_COLS = [
    "amount", "device", "location", "merchant",
    "login_attempts", "balance", "time_since_last_txn",
    "txn_velocity", "is_known_device", "amount_to_balance_ratio",
]


def preprocess_training_data(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline for training.

    Returns:
        X_train, X_test, y_train, y_test, encoders (dict)
    """
    df = df.copy()

    # 1. Drop unused columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. Handle missing values
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna("unknown", inplace=True)

    # 3. Derived feature
    df["amount_to_balance_ratio"] = df["amount"] / (df["balance"] + 1)

    # 4. Label encode categoricals — fit & save encoders
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # 5. Save encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"💾 Encoders saved → {ENCODERS_PATH}")

    # 6. Split
    X = df[MODEL_FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Fraud rate — Train: {y_train.mean()*100:.2f}%  Test: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test, encoders