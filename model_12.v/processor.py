"""
processor.py
------------
Processing pipeline worker thread.

Pops transactions from the queue and runs them through:
  1. Rule engine  (pre-model)
  2. ML inference  (LightGBM via inference_engine)
  3. Decision engine  (allow / review / block)
  4. Explainability  (SHAP top features)
  5. Status tracker  (lifecycle update)

Falls back to mock scoring if model is not loaded.
"""

import time
import random
import threading
import logging
from typing import Optional, List

from queue_manager import TransactionQueue, StatusTracker, TxnStatus

logger = logging.getLogger("processor")

# ── ML Integration ─────────────────────────────────────────────────────────
# Import inference engine. If model not loaded, fall back to mock.
try:
    import inference_engine as _ie
    _MODEL_AVAILABLE = _ie.load_model()
except Exception as _e:
    logger.warning(f"[PROCESSOR] Inference engine unavailable: {_e} — using mock scorer.")
    _MODEL_AVAILABLE = False
    _ie = None


# ── Mock Scorer (fallback) ─────────────────────────────────────────────────

def _mock_score(txn: dict) -> dict:
    score    = round(random.betavariate(1.5, 8), 4)
    amount   = txn.get("amount", 0)
    if amount > 1000:
        score = min(1.0, score + random.uniform(0.1, 0.3))

    if score < 0.3:
        risk, decision = "low",    "allow"
    elif score < 0.6:
        risk, decision = "medium", "review"
    else:
        risk, decision = "high",   "block"

    return {
        "fraud_probability":          score,
        "risk_level":                 risk,
        "decision":                   decision,
        "confidence_score":           round(0.5 + abs(score - 0.45), 4),
        "explanation_summary":        f"[MOCK] Score={score:.4f} → {decision}",
        "top_features_contributing":  [],
        "rule_warnings":              [],
        "fp_memory_match":            False,
    }


# ── Core Pipeline ──────────────────────────────────────────────────────────

def process_transaction(
    txn: dict,
    tracker: StatusTracker,
) -> dict:
    """
    Full processing pipeline for one transaction.
    Updates tracker at each stage.
    Returns enriched transaction dict.
    """
    txn_id = txn["transaction_id"]

    # Stage: PROCESSING
    tracker.update(txn_id, TxnStatus.PROCESSING)
    logger.debug(f"[PROCESSING] {txn_id}")

    # Run inference or mock
    if _MODEL_AVAILABLE and _ie and _ie.is_ready():
        inference_result = _ie.predict_transaction(txn)
    else:
        inference_result = _mock_score(txn)

    score    = inference_result["fraud_probability"]
    decision = inference_result["decision"]
    risk     = inference_result["risk_level"]
    conf     = inference_result["confidence_score"]

    # Stage: SCORED
    tracker.update(txn_id, TxnStatus.SCORED, score=score)

    # Stage: COMPLETED
    tracker.update(
        txn_id, TxnStatus.COMPLETED,
        score=score,
        decision=decision,
        risk_level=risk,
        confidence=conf,
    )

    # Merge result back into txn dict
    txn.update(inference_result)
    txn["status"] = TxnStatus.COMPLETED
    return txn


# ── Processor Worker ───────────────────────────────────────────────────────

class TransactionProcessor:
    def __init__(
        self,
        queue: TransactionQueue,
        tracker: StatusTracker,
        poll_interval: float = 0.05,
        on_completed=None,
    ):
        self.queue         = queue
        self.tracker       = tracker
        self.poll_interval = poll_interval
        self.on_completed  = on_completed

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.processed: List[dict] = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TxnProcessor"
        )
        self._thread.start()
        mode = "REAL ML MODEL" if _MODEL_AVAILABLE else "MOCK SCORER"
        logger.info(f"[PROCESSOR] Started  │  scoring mode: {mode}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[PROCESSOR] Stopped.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        while not self._stop_event.is_set():
            txn = self.queue.get(timeout=self.poll_interval)
            if txn is None:
                continue
            try:
                result = process_transaction(txn, self.tracker)
                self.processed.append(result)
                if self.on_completed:
                    self.on_completed(result)
            except Exception as exc:
                logger.error(f"[PROCESSOR] Error on {txn.get('transaction_id')}: {exc}")
                self.tracker.update(
                    txn.get("transaction_id", "UNKNOWN"),
                    TxnStatus.FAILED, error=str(exc)
                )
