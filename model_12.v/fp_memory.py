"""
fp_memory.py
------------
False Positive Memory Module.

Stores past transactions that were blocked/reviewed but turned out to be
legitimate (false positives). Uses cosine similarity to identify when a
new transaction matches a known-legitimate pattern, reducing repeated
blocking of the same customer behaviour.
"""

import json
import numpy as np
from typing import Optional, List, Dict
from config import FP_MEMORY_PATH, FP_MEMORY_MAX_RECORDS, FP_SIMILARITY_THRESHOLD


# ── FP Memory Store ───────────────────────────────────────────────────────

class FalsePositiveMemory:
    """
    Lightweight in-memory + file-persisted store of known-legitimate patterns.
    """

    def __init__(self):
        self._records: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_false_positive(self, txn_id: str, feature_vector: List[float],
                               amount: float, original_decision: str):
        """
        Record a transaction as a confirmed false positive.

        Parameters
        ----------
        txn_id           : transaction identifier
        feature_vector   : numeric features as a flat list
        amount           : transaction amount
        original_decision: "review" or "block"
        """
        if len(self._records) >= FP_MEMORY_MAX_RECORDS:
            self._records.pop(0)        # FIFO eviction

        self._records.append({
            "txn_id":    txn_id,
            "vector":    feature_vector,
            "amount":    amount,
            "decision":  original_decision,
        })
        self._save()

    def is_known_legitimate(
        self,
        feature_vector: List[float],
        amount: float,
        amount_tolerance: float = 0.3,
    ) -> Optional[Dict]:
        """
        Check if this transaction closely resembles a known false positive.

        Returns the most similar FP record if similarity > threshold, else None.
        """
        if not self._records:
            return None

        vec = np.array(feature_vector, dtype=float)
        best_sim  = -1.0
        best_rec  = None

        for rec in self._records:
            stored = np.array(rec["vector"], dtype=float)
            if len(stored) != len(vec):
                continue

            sim = _cosine_similarity(vec, stored)
            amt_diff = abs(rec["amount"] - amount) / (rec["amount"] + 1e-9)

            # Both feature pattern AND amount must be similar
            if sim > best_sim and amt_diff <= amount_tolerance:
                best_sim = sim
                best_rec = rec

        if best_sim >= FP_SIMILARITY_THRESHOLD:
            return {"record": best_rec, "similarity": round(best_sim, 4)}

        return None

    def size(self) -> int:
        return len(self._records)

    def clear(self):
        self._records = []
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        try:
            with open(FP_MEMORY_PATH, "w") as f:
                json.dump(self._records, f)
        except Exception:
            pass

    def _load(self):
        try:
            with open(FP_MEMORY_PATH) as f:
                self._records = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._records = []


# ── Helpers ────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Singleton instance
fp_memory = FalsePositiveMemory()
