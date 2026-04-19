"""
queue_manager.py
----------------
FIFO Transaction Queue + Status Tracker.

Responsibilities:
  • Buffer incoming transactions before the processor picks them up.
  • Track each transaction through its lifecycle:
      RECEIVED → QUEUED → PROCESSING → SCORED → COMPLETED
  • Thread-safe: can be written to by the generator thread and
    read from by the processor thread simultaneously.
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Transaction Lifecycle States
# ---------------------------------------------------------------------------

class TxnStatus:
    RECEIVED   = "RECEIVED"
    QUEUED     = "QUEUED"
    PROCESSING = "PROCESSING"
    SCORED     = "SCORED"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"


# ---------------------------------------------------------------------------
# Status Tracker
# ---------------------------------------------------------------------------

class StatusTracker:
    """
    Maintains an in-memory ledger of every transaction and its current status.
    Thread-safe via a single RLock.
    """

    def __init__(self):
        self._records: Dict[str, dict] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def register(self, txn: dict):
        """Register a brand-new transaction (status = RECEIVED)."""
        with self._lock:
            self._records[txn["transaction_id"]] = {
                "transaction_id": txn["transaction_id"],
                "amount":         txn.get("amount"),
                "status":         TxnStatus.RECEIVED,
                "score":          None,
                "decision":       None,
                "timestamps":     {TxnStatus.RECEIVED: _now()},
                "error":          None,
            }

    def update(self, txn_id: str, status: str,
               score: Optional[float] = None,
               decision: Optional[str] = None,
               error: Optional[str] = None):
        """Move a transaction to a new status and record the timestamp."""
        with self._lock:
            if txn_id not in self._records:
                return
            rec = self._records[txn_id]
            rec["status"] = status
            rec["timestamps"][status] = _now()
            if score is not None:
                rec["score"] = round(score, 4)
            if decision is not None:
                rec["decision"] = decision
            if error is not None:
                rec["error"] = error

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, txn_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._records.get(txn_id, {}))

    def get_all(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._records.values()]

    def filter_by_status(self, status: str) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._records.values()
                    if r["status"] == status]

    def summary(self) -> dict:
        """Return count of transactions per status."""
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self._records.values():
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            counts["TOTAL"] = len(self._records)
            return counts

    def clear(self):
        with self._lock:
            self._records.clear()


# ---------------------------------------------------------------------------
# Transaction Queue
# ---------------------------------------------------------------------------

class TransactionQueue:
    """
    Thread-safe FIFO queue backed by collections.deque.

    The generator thread pushes transactions in;
    the processor thread pops them out.
    """

    def __init__(self, maxsize: int = 0):
        """
        Parameters
        ----------
        maxsize : int
            Maximum items allowed in the queue (0 = unlimited).
        """
        self._queue: deque = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(self, txn: dict, timeout: float = 5.0) -> bool:
        """
        Add a transaction to the queue.

        Returns True on success, False if queue is full after timeout.
        """
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while self._maxsize > 0 and len(self._queue) >= self._maxsize:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._not_empty.wait(timeout=remaining)
            self._queue.append(txn)
            self._not_empty.notify_all()
            return True

    def put_many(self, txns: List[dict]) -> int:
        """Enqueue a list of transactions. Returns number successfully queued."""
        count = 0
        for txn in txns:
            if self.put(txn):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def get(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Pop the oldest transaction from the queue.

        Returns None if empty after timeout.
        """
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)
            return self._queue.popleft()

    def get_batch(self, max_items: int = 1) -> List[dict]:
        """
        Pop up to max_items transactions without blocking.
        Returns an empty list if queue is empty.
        """
        with self._not_empty:
            items = []
            while self._queue and len(items) < max_items:
                items.append(self._queue.popleft())
            return items

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def clear(self):
        with self._lock:
            self._queue.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
