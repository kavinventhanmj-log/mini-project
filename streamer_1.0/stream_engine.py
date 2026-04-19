"""
stream_engine.py
----------------
Main runner for the Pseudo Real-Time Transaction Stream Engine.

Wires together:
  Generator  →  Queue  →  Processor
                  ↕
             StatusTracker

Usage
-----
    # Default run (synthetic data, 0.3 s delay, 500 transactions)
    python stream_engine.py

    # With a real CSV (e.g. Kaggle creditcard.csv)
    python stream_engine.py --csv path/to/creditcard.csv --rows 200

    # Batch mode, 2 transactions at a time, 0.2 s delay
    python stream_engine.py --batch 2 --delay 0.2

    # Run forever (loop dataset)
    python stream_engine.py --loop
"""

import argparse
import logging
import signal
import sys
import time
from typing import List

from generator import TransactionStreamGenerator, load_transactions
from queue_manager import TransactionQueue, StatusTracker, TxnStatus
from processor import TransactionProcessor

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO", log_file: str = "stream_engine.log"):
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ]
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt, handlers=handlers)


logger = logging.getLogger("stream_engine")


# ---------------------------------------------------------------------------
# Engine Orchestrator
# ---------------------------------------------------------------------------

class TransactionStreamEngine:
    """
    Top-level orchestrator that connects:
        Generator → Queue → Processor → StatusTracker
    """

    def __init__(
        self,
        csv_path=None,
        max_rows=500,
        delay=0.3,
        batch_size=1,
        decision_threshold=0.5,
        queue_maxsize=200,
        loop=False,
        print_summary_every=50,
    ):
        # ── Data ─────────────────────────────────────────────────────
        self.df = load_transactions(csv_path=csv_path, max_rows=max_rows)

        # ── Core components ──────────────────────────────────────────
        self.queue   = TransactionQueue(maxsize=queue_maxsize)
        self.tracker = StatusTracker()

        # ── Generator ────────────────────────────────────────────────
        self.generator = TransactionStreamGenerator(
            df=self.df,
            delay=delay,
            batch_size=batch_size,
            on_emit=self._on_emit,
            loop=loop,
        )

        # ── Processor ────────────────────────────────────────────────
        self.processor = TransactionProcessor(
            queue=self.queue,
            tracker=self.tracker,
            decision_threshold=decision_threshold,
            on_completed=self._on_completed,
        )

        self._print_summary_every = print_summary_every
        self._completed_count = 0
        self._running = False

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_emit(self, events: List[dict]):
        """Called by the generator thread each time a batch is emitted."""
        for txn in events:
            self.tracker.register(txn)
            logger.info(f"[RECEIVED]   {txn['transaction_id']}  "
                        f"amount=${txn['amount']:.2f}")

            # Move to QUEUED
            ok = self.queue.put(txn)
            if ok:
                self.tracker.update(txn["transaction_id"], TxnStatus.QUEUED)
                logger.info(f"[QUEUED]     {txn['transaction_id']}  "
                            f"(queue size: {self.queue.size})")
            else:
                logger.warning(f"[DROP]       {txn['transaction_id']}  — queue full!")

    def _on_completed(self, txn: dict):
        """Called by the processor thread after each transaction completes."""
        self._completed_count += 1

        score    = txn.get("score", "N/A")
        decision = txn.get("decision", "N/A")

        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        flag = "🚨" if decision == "BLOCK" else ("⚠️ " if decision == "REVIEW" else "✅")

        logger.info(
            f"[COMPLETED]  {txn['transaction_id']}  "
            f"score={score_str}  decision={decision} {flag}"
        )

        # Periodic summary
        if self._completed_count % self._print_summary_every == 0:
            self._print_summary()

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self):
        logger.info("=" * 60)
        logger.info("  TRANSACTION STREAM ENGINE  —  STARTING")
        logger.info("=" * 60)
        self._running = True
        self.processor.start()
        self.generator.start()

    def stop(self):
        logger.info("[ENGINE] Shutting down …")
        self.generator.stop()
        # Let processor drain the remaining queue
        timeout = 10
        start   = time.monotonic()
        while not self.queue.is_empty() and (time.monotonic() - start) < timeout:
            time.sleep(0.2)
        self.processor.stop()
        self._running = False
        self._print_summary()
        logger.info("[ENGINE] Shutdown complete.")

    def pause(self):
        self.generator.pause()

    def resume(self):
        self.generator.resume()

    def set_delay(self, delay: float):
        self.generator.set_delay(delay)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        summary = self.tracker.summary()
        lines = ["", "─" * 40, "  PIPELINE SUMMARY"]
        for k, v in summary.items():
            lines.append(f"    {k:<12} : {v:>6}")
        completed = self.tracker.filter_by_status(TxnStatus.COMPLETED)
        if completed:
            scores = [r["score"] for r in completed if r["score"] is not None]
            if scores:
                avg = sum(scores) / len(scores)
                high = sum(1 for s in scores if s >= 0.5)
                lines.append(f"    Avg score    : {avg:.4f}")
                lines.append(f"    High-risk    : {high:>6}")
        lines.append("─" * 40)
        for l in lines:
            logger.info(l)

    def wait(self):
        """Block main thread until generator finishes (non-loop mode)."""
        while self.generator.is_alive() or not self.queue.is_empty():
            time.sleep(0.5)
        # Give processor a moment to finish last items
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pseudo Real-Time Transaction Stream Engine"
    )
    parser.add_argument("--csv",     default=None,  help="Path to CSV dataset")
    parser.add_argument("--rows",    type=int, default=500, help="Max transactions to stream")
    parser.add_argument("--delay",   type=float, default=0.3, help="Delay between emissions (s)")
    parser.add_argument("--batch",   type=int, default=1,   help="Transactions per emission")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fraud score threshold")
    parser.add_argument("--loop",    action="store_true",   help="Loop dataset forever")
    parser.add_argument("--log-level", default="INFO",      help="Logging level")
    parser.add_argument("--log-file",  default="stream_engine.log")
    args = parser.parse_args()

    _setup_logging(level=args.log_level, log_file=args.log_file)

    engine = TransactionStreamEngine(
        csv_path=args.csv,
        max_rows=args.rows,
        delay=args.delay,
        batch_size=args.batch,
        decision_threshold=args.threshold,
        loop=args.loop,
    )

    # Graceful shutdown on Ctrl+C / SIGTERM
    def _shutdown(sig, frame):
        logger.info("\n[ENGINE] Signal received — stopping …")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    engine.start()

    if not args.loop:
        engine.wait()
        engine.stop()
    else:
        # Loop mode — run until Ctrl+C
        logger.info("[ENGINE] Running in loop mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop()


if __name__ == "__main__":
    main()
