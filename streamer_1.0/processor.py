"""
processor.py
------------
Transaction Processing Pipeline.

Responsibilities:
  • Pull transactions from the queue.
  • Run each through a processing pipeline:
      1. Validation
      2. Feature extraction / normalization (placeholder)
      3. ML scoring  ← PLUG YOUR MODEL HERE
      4. Decision logic (placeholder)
  • Update status tracker at every stage.
  • Run in its own background thread.

ML INTEGRATION POINT
--------------------
Drop a file  models/predict.py  that exports:

    def predict_transaction(transaction: dict) -> float:
        ...

The processor imports it automatically if found.
If not found, a mock scorer is used instead.
"""

import time
import random
import threading
import importlib
import logging
from typing import Optional, List

from queue_manager import TransactionQueue, StatusTracker, TxnStatus

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("processor")


# ---------------------------------------------------------------------------
# ML Model Loader  (plug-and-play)
# ---------------------------------------------------------------------------

def _load_model_predict():
    """
    Attempt to import  models.predict.predict_transaction.
    Returns the function if available, else None.
    """
    try:
        module = importlib.import_module("models.predict")
        fn = getattr(module, "predict_transaction", None)
        if callable(fn):
            logger.info("[MODEL] Real ML model loaded from models/predict.py")
            return fn
    except ModuleNotFoundError:
        pass
    logger.info("[MODEL] No ML model found — using mock scorer.")
    return None


_predict_fn = _load_model_predict()


# ---------------------------------------------------------------------------
# Scoring Layer
# ---------------------------------------------------------------------------

def _score_transaction(txn: dict) -> float:
    """
    Score a transaction.

    If a real model is available, call it.
    Otherwise return a random mock fraud probability.
    """
    if _predict_fn is not None:
        return float(_predict_fn(txn))

    # Mock scorer: returns a low score usually, occasionally high
    base = random.betavariate(1.5, 8)          # skewed towards 0
    if txn.get("amount", 0) > 1500:
        base = min(1.0, base + random.uniform(0.2, 0.4))
    return round(base, 4)


# ---------------------------------------------------------------------------
# Decision Layer  (placeholder — extend as needed)
# ---------------------------------------------------------------------------

def _make_decision(score: float, threshold: float = 0.5) -> str:
    """
    Translate a fraud score into a business decision.

    Returns
    -------
    "BLOCK"  if score >= threshold
    "REVIEW" if score >= threshold * 0.6
    "PASS"   otherwise
    """
    if score >= threshold:
        return "BLOCK"
    if score >= threshold * 0.6:
        return "REVIEW"
    return "PASS"


# ---------------------------------------------------------------------------
# Validation Layer  (placeholder)
# ---------------------------------------------------------------------------

def _validate_transaction(txn: dict) -> tuple[bool, Optional[str]]:
    """
    Basic sanity checks on the transaction dict.

    Returns (is_valid, error_message).
    Extend this with real business rules.
    """
    if not txn.get("transaction_id"):
        return False, "Missing transaction_id"
    if txn.get("amount") is None or txn["amount"] < 0:
        return False, f"Invalid amount: {txn.get('amount')}"
    if not isinstance(txn.get("features"), dict):
        return False, "Missing or malformed features"
    return True, None


# ---------------------------------------------------------------------------
# Core Pipeline Function  (the integration seam)
# ---------------------------------------------------------------------------

def process_transaction(
    txn: dict,
    tracker: StatusTracker,
    decision_threshold: float = 0.5,
    simulate_latency: bool = True,
) -> dict:
    """
    Run a single transaction through the full processing pipeline.

    Parameters
    ----------
    txn                : structured transaction event dict
    tracker            : StatusTracker instance for lifecycle updates
    decision_threshold : fraud probability above which we block
    simulate_latency   : add a short sleep to mimic real model inference

    Returns
    -------
    Enriched transaction dict with 'score' and 'decision' fields.
    """
    txn_id = txn["transaction_id"]

    # ── Stage 1: PROCESSING ───────────────────────────────────────────
    tracker.update(txn_id, TxnStatus.PROCESSING)
    logger.info(f"[PROCESSING] {txn_id}")

    # ── Stage 2: Validation ───────────────────────────────────────────
    valid, err = _validate_transaction(txn)
    if not valid:
        tracker.update(txn_id, TxnStatus.FAILED, error=err)
        logger.warning(f"[FAILED]     {txn_id}  ✗  {err}")
        txn["status"] = TxnStatus.FAILED
        txn["error"] = err
        return txn

    # ── Stage 3: Feature extraction (placeholder) ─────────────────────
    # Future: normalise features, add derived columns, etc.
    if simulate_latency:
        time.sleep(random.uniform(0.01, 0.05))      # mimic pre-processing

    # ── Stage 4: ML Scoring ───────────────────────────────────────────
    score = _score_transaction(txn)
    tracker.update(txn_id, TxnStatus.SCORED, score=score)
    logger.info(f"[SCORED]     {txn_id}  →  score: {score:.4f}")

    # Simulate model inference time
    if simulate_latency:
        time.sleep(random.uniform(0.01, 0.08))

    # ── Stage 5: Decision ─────────────────────────────────────────────
    decision = _make_decision(score, threshold=decision_threshold)
    tracker.update(txn_id, TxnStatus.COMPLETED, score=score, decision=decision)
    logger.info(f"[COMPLETED]  {txn_id}  →  decision: {decision}")

    txn["score"] = score
    txn["decision"] = decision
    txn["status"] = TxnStatus.COMPLETED
    return txn


# ---------------------------------------------------------------------------
# Processor Worker  (runs in a background thread)
# ---------------------------------------------------------------------------

class TransactionProcessor:
    """
    Continuously pops transactions from a queue and processes them.

    Parameters
    ----------
    queue              : TransactionQueue to consume from
    tracker            : StatusTracker for lifecycle updates
    decision_threshold : fraud score cut-off
    poll_interval      : seconds to wait when queue is empty
    on_completed       : optional callback(processed_txn) after each txn
    """

    def __init__(
        self,
        queue: TransactionQueue,
        tracker: StatusTracker,
        decision_threshold: float = 0.5,
        poll_interval: float = 0.1,
        on_completed=None,
    ):
        self.queue = queue
        self.tracker = tracker
        self.decision_threshold = decision_threshold
        self.poll_interval = poll_interval
        self.on_completed = on_completed

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.processed: List[dict] = []          # completed transaction log

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("[PROCESSOR] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TxnProcessor"
        )
        self._thread.start()
        logger.info("[PROCESSOR] Started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[PROCESSOR] Stopped.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            txn = self.queue.get(timeout=self.poll_interval)
            if txn is None:
                continue        # queue was empty — try again

            try:
                result = process_transaction(
                    txn,
                    self.tracker,
                    decision_threshold=self.decision_threshold,
                )
                self.processed.append(result)

                if self.on_completed:
                    self.on_completed(result)

            except Exception as exc:
                logger.error(f"[PROCESSOR] Unhandled error on {txn.get('transaction_id')}: {exc}")
                self.tracker.update(
                    txn.get("transaction_id", "UNKNOWN"),
                    TxnStatus.FAILED,
                    error=str(exc),
                )
