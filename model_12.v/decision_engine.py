"""
decision_engine.py
------------------
Converts a fraud probability score into a 3-way business decision:
    ALLOW  | REVIEW | BLOCK

Thresholds are:
  • Computed automatically after training (saved to thresholds.json)
  • Loaded at inference time
  • Fallback to config defaults if file not found
"""

import json
import numpy as np
from typing import Tuple
from config import (
    THRESHOLD_PATH,
    DEFAULT_ALLOW_THRESHOLD,
    DEFAULT_BLOCK_THRESHOLD,
)


# ── Threshold Store ────────────────────────────────────────────────────────

_thresholds = {
    "allow": DEFAULT_ALLOW_THRESHOLD,
    "block": DEFAULT_BLOCK_THRESHOLD,
}


def load_thresholds():
    """Load thresholds from file (called at inference startup)."""
    global _thresholds
    try:
        with open(THRESHOLD_PATH) as f:
            _thresholds = json.load(f)
    except FileNotFoundError:
        pass    # use defaults


def save_thresholds(allow: float, block: float):
    """Persist computed thresholds after training."""
    data = {"allow": round(allow, 4), "block": round(block, 4)}
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(data, f, indent=2)
    _thresholds.update(data)
    print(f"[DECISION ENGINE] Thresholds saved → allow<{allow:.4f}  block>{block:.4f}")


# ── Threshold Optimizer ────────────────────────────────────────────────────

def optimize_thresholds(y_true: np.ndarray,
                        y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Automatically determine ALLOW and BLOCK thresholds.

    Strategy:
      • ALLOW threshold : highest score where FPR ≤ 1%  (minimize false blocks)
      • BLOCK threshold : lowest score where Precision ≥ 85% on fraud class

    Parameters
    ----------
    y_true : ground truth labels (0/1)
    y_prob : model fraud probabilities

    Returns
    -------
    (allow_threshold, block_threshold)
    """
    from sklearn.metrics import precision_score

    candidates = np.linspace(0.01, 0.99, 200)

    # ── ALLOW threshold ───────────────────────────────────────────────
    # Highest cutoff where FP rate ≤ 1 %
    allow_t = DEFAULT_ALLOW_THRESHOLD
    for t in candidates:
        preds  = (y_prob >= t).astype(int)
        fp     = ((preds == 1) & (y_true == 0)).sum()
        tn     = ((preds == 0) & (y_true == 0)).sum()
        fpr    = fp / (fp + tn + 1e-9)
        if fpr > 0.01:
            break
        allow_t = t

    # ── BLOCK threshold ───────────────────────────────────────────────
    # Lowest cutoff where fraud precision ≥ 85 %
    block_t = DEFAULT_BLOCK_THRESHOLD
    for t in reversed(candidates):
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        prec = precision_score(y_true, preds, zero_division=0)
        if prec >= 0.85:
            block_t = t
            break

    # Sanity: allow must be strictly less than block
    if allow_t >= block_t:
        allow_t = block_t * 0.5

    return round(float(allow_t), 4), round(float(block_t), 4)


# ── Decision Maker ─────────────────────────────────────────────────────────

def make_decision(score: float) -> Tuple[str, str]:
    """
    Convert a fraud probability into (risk_level, decision).

    Returns
    -------
    risk_level : "low" | "medium" | "high"
    decision   : "allow" | "review" | "block"
    """
    allow_t = _thresholds["allow"]
    block_t = _thresholds["block"]

    if score < allow_t:
        return "low",    "allow"
    elif score < block_t:
        return "medium", "review"
    else:
        return "high",   "block"


def confidence_score(score: float) -> float:
    """
    Heuristic confidence: how far the score is from the nearest boundary.
    Returns a value in [0.5, 1.0] — higher = more confident.
    """
    allow_t = _thresholds["allow"]
    block_t = _thresholds["block"]
    mid     = (allow_t + block_t) / 2.0

    dist_from_mid = abs(score - mid)
    max_dist      = max(mid, 1.0 - mid)
    confidence    = 0.5 + 0.5 * (dist_from_mid / max_dist)
    return round(min(1.0, confidence), 4)


# Load thresholds on module import
load_thresholds()
