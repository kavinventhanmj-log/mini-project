"""
feature_engineering.py
-----------------------
All feature transformations applied IDENTICALLY during training and inference.
Never import training-only libraries here — this runs in the stream too.
"""

import numpy as np
import pandas as pd
from config import (
    AMOUNT_LOG, TIME_FEATURES,
    AMOUNT_BINS, AMOUNT_BIN_LABELS,
)


# ── Public API ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a DataFrame.
    Works on a single row (inference) or the full dataset (training).

    Parameters
    ----------
    df : DataFrame with at minimum columns: Time, Amount, V1–V28

    Returns
    -------
    Enriched DataFrame (original columns preserved + new ones appended).
    """
    df = df.copy()

    # ── Amount features ───────────────────────────────────────────────
    if AMOUNT_LOG:
        df["log_amount"] = np.log1p(df["Amount"])

    df["amount_bin"] = pd.cut(
        df["Amount"],
        bins=AMOUNT_BINS,
        labels=AMOUNT_BIN_LABELS,
        right=False,
    ).astype(str)
    df["amount_bin"] = pd.Categorical(
        df["amount_bin"], categories=AMOUNT_BIN_LABELS
    ).codes                                     # integer encoding

    df["amount_zscore"] = _zscore_clip(df["Amount"])

    # ── Time features ─────────────────────────────────────────────────
    if TIME_FEATURES:
        # Time col = seconds since first txn in dataset
        seconds_in_day = 86_400
        df["hour_of_day"] = (df["Time"] % seconds_in_day) // 3600
        df["is_night"]    = df["hour_of_day"].apply(
            lambda h: 1 if (h >= 22 or h <= 5) else 0
        )
        df["time_bin"]    = df["hour_of_day"] // 6     # 0-3  (6-hr buckets)

    # ── Anomaly indicators ────────────────────────────────────────────
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        df["v_mean"]   = df[v_cols].mean(axis=1)
        df["v_std"]    = df[v_cols].std(axis=1)
        df["v_max_abs"]= df[v_cols].abs().max(axis=1)

        # PCA deviation score: squared sum of top anomaly-linked features
        # V14, V12, V10, V16, V3 are most fraud-correlated in this dataset
        top_v = [c for c in ["V14","V12","V10","V16","V3"] if c in df.columns]
        if top_v:
            df["pca_anomaly_score"] = (df[top_v] ** 2).sum(axis=1)

    # ── Behavioral deviation signals ──────────────────────────────────
    # Within available data: flag transactions far from "normal" amounts
    df["is_round_amount"]  = (df["Amount"] % 1 == 0).astype(int)
    df["is_micro_amount"]  = (df["Amount"] < 1.0).astype(int)
    df["is_large_amount"]  = (df["Amount"] > 1000).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the list of feature columns to feed into the model.
    Excludes target, raw identifiers, and non-numeric columns.
    """
    exclude = {"Class", "Time", "transaction_id", "timestamp",
               "status", "label", "features"}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype != object]


# ── Helpers ────────────────────────────────────────────────────────────────

def _zscore_clip(series: pd.Series, clip: float = 5.0) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - mu) / sigma).clip(-clip, clip)


def transaction_to_dataframe(txn: dict) -> pd.DataFrame:
    """
    Convert a streaming transaction event dict into a single-row DataFrame
    ready for feature engineering and model inference.

    Handles both:
      • flat dict  (keys: Time, Amount, V1..V28)
      • nested dict (keys: features={V1..V28}, amount=..., timestamp=...)
    """
    if "features" in txn and isinstance(txn["features"], dict):
        # Structured event from stream engine
        row = dict(txn["features"])                  # V1–V28
        row["Amount"] = txn.get("amount", 0.0)
        # Derive Time from timestamp offset — default 0 for streaming
        row["Time"]   = txn.get("time_offset", 0.0)
    else:
        row = {k: v for k, v in txn.items()
               if k not in {"transaction_id", "timestamp",
                            "status", "label", "Class"}}

    df = pd.DataFrame([row])

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
