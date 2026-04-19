"""
generator.py
------------
Pseudo Real-Time Transaction Stream Generator.
Emits transactions from a CSV file or synthetic data,
one-by-one or in mini-batches, with configurable delay.

Supports: start / stop / pause / resume controls.
"""

import time
import uuid
import threading
import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Iterator, List, Optional, Callable


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

def generate_synthetic_transactions(n: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mimics the Kaggle Credit Card
    Fraud Detection dataset (V1-V28 + Amount + Class).
    """
    rng = np.random.default_rng(seed=42)
    n_fraud = max(1, int(n * 0.02))          # ~2 % fraud rate
    n_legit = n - n_fraud

    legit = pd.DataFrame(rng.standard_normal((n_legit, 28)),
                         columns=[f"V{i}" for i in range(1, 29)])
    legit["Amount"] = rng.exponential(scale=88, size=n_legit).round(2)
    legit["Class"] = 0

    fraud = pd.DataFrame(rng.standard_normal((n_fraud, 28)),
                         columns=[f"V{i}" for i in range(1, 29)])
    fraud["Amount"] = rng.exponential(scale=500, size=n_fraud).round(2)
    fraud["Class"] = 1

    df = pd.concat([legit, fraud], ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# CSV / DataFrame Loader
# ---------------------------------------------------------------------------

def load_transactions(csv_path: Optional[str] = None,
                      max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load transactions from a CSV file, or fall back to synthetic data.

    Parameters
    ----------
    csv_path : str | None
        Path to a CSV file (e.g. Kaggle creditcard.csv).
        If None or file not found, synthetic data is generated.
    max_rows : int | None
        Limit the number of rows loaded.
    """
    if csv_path:
        try:
            df = pd.read_csv(csv_path, nrows=max_rows)
            print(f"[GENERATOR] Loaded {len(df):,} transactions from '{csv_path}'.")
            return df
        except FileNotFoundError:
            print(f"[GENERATOR] '{csv_path}' not found — using synthetic data.")

    n = max_rows or 500
    df = generate_synthetic_transactions(n=n)
    print(f"[GENERATOR] Generated {len(df):,} synthetic transactions.")
    return df


# ---------------------------------------------------------------------------
# Transaction Event Builder
# ---------------------------------------------------------------------------

def build_transaction_event(row: pd.Series, index: int) -> dict:
    """
    Convert a DataFrame row into a structured transaction event object.

    Returns
    -------
    dict with keys:
        transaction_id, timestamp, amount, features (dict V1-V28),
        label (ground-truth class if available), status
    """
    feature_cols = [c for c in row.index if c.startswith("V")]
    features = {col: round(float(row[col]), 6) for col in feature_cols}

    return {
        "transaction_id": f"TXN_{index:06d}",
        "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "amount": round(float(row.get("Amount", random.uniform(1, 2000))), 2),
        "features": features,
        "label": int(row["Class"]) if "Class" in row.index else None,
        "status": "RECEIVED",
    }


# ---------------------------------------------------------------------------
# Stream Generator Class
# ---------------------------------------------------------------------------

class TransactionStreamGenerator:
    """
    Emits transaction events from a DataFrame in pseudo real-time.

    Parameters
    ----------
    df            : source DataFrame
    delay         : seconds between emissions (float)
    batch_size    : number of transactions per emission (1 = single)
    on_emit       : callback(List[dict]) invoked for each batch
    loop          : whether to restart from the beginning when exhausted
    """

    def __init__(
        self,
        df: pd.DataFrame,
        delay: float = 0.5,
        batch_size: int = 1,
        on_emit: Optional[Callable[[List[dict]], None]] = None,
        loop: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.delay = delay
        self.batch_size = batch_size
        self.on_emit = on_emit
        self.loop = loop

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()          # not paused initially
        self._thread: Optional[threading.Thread] = None
        self._index = 0

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def start(self):
        """Start streaming in a background thread."""
        if self._thread and self._thread.is_alive():
            print("[GENERATOR] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="StreamGen")
        self._thread.start()
        print(f"[GENERATOR] Stream started  │  delay={self.delay}s  │  batch={self.batch_size}")

    def stop(self):
        """Stop streaming permanently."""
        self._stop_event.set()
        self._pause_event.set()          # unblock if paused
        if self._thread:
            self._thread.join(timeout=5)
        print("[GENERATOR] Stream stopped.")

    def pause(self):
        """Pause emission (resumes from where it left off)."""
        self._pause_event.clear()
        print("[GENERATOR] Stream paused.")

    def resume(self):
        """Resume a paused stream."""
        self._pause_event.set()
        print("[GENERATOR] Stream resumed.")

    def set_delay(self, delay: float):
        """Adjust emission delay on the fly (seconds)."""
        self.delay = delay
        print(f"[GENERATOR] Delay updated → {delay}s")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal runner
    # ------------------------------------------------------------------

    def _run(self):
        total = len(self.df)

        while not self._stop_event.is_set():
            if self._index >= total:
                if self.loop:
                    self._index = 0
                    print("[GENERATOR] Dataset exhausted — restarting (loop=True).")
                else:
                    print("[GENERATOR] Dataset exhausted — stream ending.")
                    break

            # Pause checkpoint
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            # Slice a mini-batch
            end = min(self._index + self.batch_size, total)
            batch_rows = self.df.iloc[self._index:end]
            self._index = end

            events = [
                build_transaction_event(row, self._index - len(batch_rows) + i)
                for i, (_, row) in enumerate(batch_rows.iterrows())
            ]

            if self.on_emit:
                self.on_emit(events)

            time.sleep(self.delay)

    # ------------------------------------------------------------------
    # Generator (iterator) API — for direct iteration without threads
    # ------------------------------------------------------------------

    def iter_transactions(self) -> Iterator[List[dict]]:
        """
        Yield batches of transaction events synchronously.
        Useful for testing without background threads.
        """
        for i in range(0, len(self.df), self.batch_size):
            batch_rows = self.df.iloc[i:i + self.batch_size]
            events = [
                build_transaction_event(row, i + j)
                for j, (_, row) in enumerate(batch_rows.iterrows())
            ]
            yield events
            time.sleep(self.delay)
