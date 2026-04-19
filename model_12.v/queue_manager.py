import threading
import time
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime


class TxnStatus:
    RECEIVED   = "RECEIVED"
    QUEUED     = "QUEUED"
    PROCESSING = "PROCESSING"
    SCORED     = "SCORED"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"


class StatusTracker:
    def __init__(self):
        self._records: Dict[str, dict] = {}
        self._lock = threading.RLock()

    def register(self, txn: dict):
        with self._lock:
            self._records[txn["transaction_id"]] = {
                "transaction_id": txn["transaction_id"],
                "amount":   txn.get("amount"),
                "label":    txn.get("label"),
                "status":   TxnStatus.RECEIVED,
                "score":    None,
                "decision": None,
                "risk_level": None,
                "confidence": None,
                "timestamps": {TxnStatus.RECEIVED: _now()},
                "error":    None,
            }

    def update(self, txn_id: str, status: str, **kwargs):
        with self._lock:
            if txn_id not in self._records:
                return
            rec = self._records[txn_id]
            rec["status"] = status
            rec["timestamps"][status] = _now()
            for k, v in kwargs.items():
                rec[k] = v

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

    def filter_by_decision(self, decision: str) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._records.values()
                    if r.get("decision") == decision]

    def summary(self) -> dict:
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self._records.values():
                s = r["status"]
                counts[s] = counts.get(s, 0) + 1
            counts["TOTAL"] = len(self._records)
            return counts

    def decision_summary(self) -> dict:
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self._records.values():
                d = r.get("decision") or "pending"
                counts[d] = counts.get(d, 0) + 1
            return counts

    def fraud_stats(self) -> dict:
        with self._lock:
            completed = [r for r in self._records.values()
                         if r["status"] == TxnStatus.COMPLETED
                         and r.get("label") is not None]
            if not completed:
                return {}
            tp = sum(1 for r in completed if r["label"] == 1 and r.get("decision") == "block")
            fp = sum(1 for r in completed if r["label"] == 0 and r.get("decision") == "block")
            tn = sum(1 for r in completed if r["label"] == 0 and r.get("decision") == "allow")
            fn = sum(1 for r in completed if r["label"] == 1 and r.get("decision") != "block")
            return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                    "total_evaluated": len(completed)}

    def clear(self):
        with self._lock:
            self._records.clear()


class TransactionQueue:
    def __init__(self, maxsize: int = 0):
        self._queue: deque = deque()
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, txn: dict, timeout: float = 5.0) -> bool:
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
        return sum(1 for t in txns if self.put(t))

    def get(self, timeout: float = 1.0) -> Optional[dict]:
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)
            return self._queue.popleft()

    def get_batch(self, max_items: int = 1) -> List[dict]:
        with self._not_empty:
            items = []
            while self._queue and len(items) < max_items:
                items.append(self._queue.popleft())
            return items

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


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
