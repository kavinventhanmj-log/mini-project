"""
generator.py
------------
Pseudo Real-Time Transaction Stream Generator.

Supports the exact Kaggle Credit Card Fraud dataset schema:
    Time, V1–V28, Amount, Class

New in this version:
  • Reads real CSV (Time, V1-V28, Amount, Class)
  • Uses Time column to compute realistic inter-transaction delays
  • Preserves Class label as ground-truth for evaluation
  • Falls back to synthetic data if CSV not found
  • Supports pause / resume / speed control / batch mode
"""

import time
import threading
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Iterator, List, Optional, Callable

from config import DATASET_PATH, STREAM_DELAY, STREAM_BATCH_SIZE


# ── Dataset Loader ─────────────────────────────────────────────────────────

def load_transactions(csv_path: Optional[str] = None,
                      max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the Kaggle credit card dataset OR generate synthetic data.

    Parameters
    ----------
    csv_path : Path to creditcard.csv (falls back to config.DATASET_PATH)
    max_rows : Limit rows loaded
    """
    path = csv_path or DATASET_PATH

    try:
        df = pd.read_csv(path, nrows=max_rows)

        # Validate expected schema
        required = {"Time", "Amount", "Class"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns: {required - set(df.columns)}")

        # Strip quotes from Class if present (some exports wrap in "")
        df["Class"] = pd.to_numeric(
            df["Class"].astype(str).str.strip('"'), errors="coerce"
        ).fillna(0).astype(int)

        print(f"[GENERATOR] Loaded {len(df):,} real transactions "
              f"from '{path}'  |  fraud rate: {df['Class'].mean()*100:.2f}%")
        return df.reset_index(drop=True)

    except FileNotFoundError:
        print(f"[GENERATOR] '{path}' not found — generating synthetic data.")
    except Exception as e:
        print(f"[GENERATOR] CSV load error ({e}) — generating synthetic data.")

    n = max_rows or 500
    return _generate_synthetic(n)


# ── Synthetic Data ─────────────────────────────────────────────────────────

def _generate_synthetic(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n_fraud = max(1, int(n * 0.02))
    n_legit = n - n_fraud

    def _block(size, fraud=False):
        df = pd.DataFrame(
            rng.standard_normal((size, 28)),
            columns=[f"V{i}" for i in range(1, 29)]
        )
        df["Amount"] = (rng.exponential(500 if fraud else 88, size)).round(2)
        df["Class"]  = int(fraud)
        df["Time"]   = np.sort(rng.uniform(0, 172_800, size))
        return df

    df = pd.concat([_block(n_legit), _block(n_fraud, fraud=True)],
                   ignore_index=True).sample(frac=1, random_state=42)
    print(f"[GENERATOR] Generated {len(df):,} synthetic transactions.")
    return df.reset_index(drop=True)


# ── Transaction Event Builder ──────────────────────────────────────────────

def build_transaction_event(row: pd.Series, index: int) -> dict:
    """
    Convert a DataFrame row into a structured transaction event.

    Schema
    ------
    {
        transaction_id : "TXN_000001"
        timestamp      : ISO-8601 UTC wall-clock time
        time_offset    : raw 'Time' value from dataset (seconds)
        amount         : float
        features       : { V1 … V28 }
        label          : int (0/1 ground truth, None if unavailable)
        status         : "RECEIVED"
    }
    """
    v_cols   = [c for c in row.index if c.startswith("V")]
    features = {col: round(float(row[col]), 6) for col in v_cols}

    return {
        "transaction_id": f"TXN_{index:06d}",
        "timestamp":      datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "time_offset":    float(row.get("Time", 0.0)),
        "amount":         round(float(row.get("Amount", 0.0)), 2),
        "features":       features,
        "label":          int(row["Class"]) if "Class" in row.index else None,
        "status":         "RECEIVED",
    }


# ── Stream Generator ───────────────────────────────────────────────────────

class TransactionStreamGenerator:
    """
    Emits transaction events in pseudo real-time.

    Parameters
    ----------
    df           : source DataFrame (Kaggle schema)
    delay        : fixed delay in seconds (overrides time-based if set)
    use_time_col : if True, derive delays from dataset 'Time' column
    time_scale   : compression factor for time-based delays
                   (1.0 = real-time, 10.0 = 10× faster)
    batch_size   : transactions per emission
    on_emit      : callback(List[dict]) for each batch
    loop         : restart when dataset is exhausted
    """

    def __init__(
        self,
        df: pd.DataFrame,
        delay: float = STREAM_DELAY,
        use_time_col: bool = True,
        time_scale: float = 100.0,
        batch_size: int = STREAM_BATCH_SIZE,
        on_emit: Optional[Callable[[List[dict]], None]] = None,
        loop: bool = False,
    ):
        self.df          = df.reset_index(drop=True)
        self.delay       = delay
        self.use_time_col = use_time_col and "Time" in df.columns
        self.time_scale  = time_scale
        self.batch_size  = batch_size
        self.on_emit     = on_emit
        self.loop        = loop

        self._stop_event  = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._thread: Optional[threading.Thread] = None
        self._index  = 0

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            print("[GENERATOR] Already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="StreamGen"
        )
        self._thread.start()
        mode = f"time-col (×{self.time_scale})" if self.use_time_col else f"{self.delay}s fixed"
        print(f"[GENERATOR] Stream started  │  delay={mode}  │  batch={self.batch_size}")

    def stop(self):
        self._stop_event.set()
        self._pause_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("[GENERATOR] Stream stopped.")

    def pause(self):
        self._pause_event.clear()
        print("[GENERATOR] Stream paused.")

    def resume(self):
        self._pause_event.set()
        print("[GENERATOR] Stream resumed.")

    def set_delay(self, delay: float):
        self.delay = delay
        self.use_time_col = False
        print(f"[GENERATOR] Fixed delay set → {delay}s")

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
                    print("[GENERATOR] Dataset exhausted — restarting.")
                else:
                    print("[GENERATOR] Dataset exhausted — stream ending.")
                    break

            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            end  = min(self._index + self.batch_size, total)
            rows = self.df.iloc[self._index:end]
            base = self._index
            self._index = end

            events = [
                build_transaction_event(row, base + i)
                for i, (_, row) in enumerate(rows.iterrows())
            ]

            if self.on_emit:
                self.on_emit(events)

            # Compute sleep duration
            sleep_t = self._compute_delay(rows)
            time.sleep(sleep_t)

    def _compute_delay(self, rows: pd.DataFrame) -> float:
        """
        Use the Time column difference for realistic pacing,
        or fall back to fixed delay.
        """
        if self.use_time_col and len(rows) > 0 and self._index < len(self.df):
            try:
                t_curr = rows.iloc[-1]["Time"]
                t_next = self.df.iloc[self._index]["Time"] \
                    if self._index < len(self.df) else t_curr
                raw_gap = max(0.0, float(t_next - t_curr))
                scaled  = raw_gap / self.time_scale
                return min(scaled, 2.0)          # cap at 2s per step
            except Exception:
                pass
        return self.delay

    # ------------------------------------------------------------------
    # Iterator API (for synchronous testing)
    # ------------------------------------------------------------------

    def iter_transactions(self) -> Iterator[List[dict]]:
        for i in range(0, len(self.df), self.batch_size):
            rows   = self.df.iloc[i:i + self.batch_size]
            events = [
                build_transaction_event(row, i + j)
                for j, (_, row) in enumerate(rows.iterrows())
            ]
            yield events
            time.sleep(self._compute_delay(rows))
