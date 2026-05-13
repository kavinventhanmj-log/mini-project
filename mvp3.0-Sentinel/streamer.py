"""
streamer.py
===========
Simulates a real-time transaction stream by emitting rows one-by-one.
Does NOT make any fraud decisions — just feeds data forward.
"""

import pandas as pd
import time
import json
from typing import Iterator

DEFAULT_CSV   = "transactions.csv"
EMIT_DELAY_MS = 50          # milliseconds between transactions (tunable)


def stream_transactions(
    csv_path: str = DEFAULT_CSV,
    delay_ms: float = EMIT_DELAY_MS,
    max_rows: int | None = None,
) -> Iterator[dict]:
    """
    Yields one transaction dict at a time.

    Args:
        csv_path  : path to generated CSV
        delay_ms  : artificial delay between emissions (ms)
        max_rows  : cap rows for testing (None = full dataset)
    """
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)

    delay_sec = delay_ms / 1000.0

    for idx, row in df.iterrows():
        txn = row.to_dict()
        txn["transaction_id"] = f"TXN{str(idx).zfill(7)}"

        yield txn

        if delay_sec > 0:
            time.sleep(delay_sec)


# ─────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🌊 Starting transaction stream (demo — 5 transactions)...\n")
    for txn in stream_transactions(delay_ms=200, max_rows=5):
        print(json.dumps(txn, indent=2))
        print("─" * 50)