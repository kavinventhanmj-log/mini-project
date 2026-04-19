"""
stream_engine.py
----------------
Main runner — wires Generator → Queue → Processor → Tracker.

Usage
-----
    # Default (synthetic or data/creditcard.csv, mock scorer if no model)
    python stream_engine.py

    # With real CSV
    python stream_engine.py --csv data/creditcard.csv --rows 500

    # After training model
    python train_model.py
    python stream_engine.py --csv data/creditcard.csv

    # Batch mode, faster
    python stream_engine.py --batch 2 --delay 0.2

    # Loop forever
    python stream_engine.py --loop --delay 0.1
"""

import argparse
import logging
import signal
import sys
import time
from typing import List

from config import (
    LOG_FILE, LOG_LEVEL,
    STREAM_DELAY, STREAM_BATCH_SIZE, STREAM_QUEUE_MAXSIZE,
    DEFAULT_ALLOW_THRESHOLD, DEFAULT_BLOCK_THRESHOLD,
)
from generator import TransactionStreamGenerator, load_transactions
from queue_manager import TransactionQueue, StatusTracker, TxnStatus
from processor import TransactionProcessor


# ── Logging ────────────────────────────────────────────────────────────────

def _setup_logging(level="INFO", log_file=LOG_FILE):
    fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt, datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )

logger = logging.getLogger("stream_engine")

# Decision emoji map
_EMOJI = {"allow": "✅", "review": "⚠️ ", "block": "🚨"}


# ── Engine ─────────────────────────────────────────────────────────────────

class TransactionStreamEngine:
    def __init__(
        self,
        csv_path=None,
        max_rows=500,
        delay=STREAM_DELAY,
        batch_size=STREAM_BATCH_SIZE,
        queue_maxsize=STREAM_QUEUE_MAXSIZE,
        loop=False,
        use_time_col=True,
        time_scale=100.0,
        print_summary_every=50,
    ):
        self.df      = load_transactions(csv_path=csv_path, max_rows=max_rows)
        self.queue   = TransactionQueue(maxsize=queue_maxsize)
        self.tracker = StatusTracker()

        self.generator = TransactionStreamGenerator(
            df=self.df,
            delay=delay,
            use_time_col=use_time_col,
            time_scale=time_scale,
            batch_size=batch_size,
            on_emit=self._on_emit,
            loop=loop,
        )
        self.processor = TransactionProcessor(
            queue=self.queue,
            tracker=self.tracker,
            on_completed=self._on_completed,
        )

        self._summary_every = print_summary_every
        self._completed     = 0

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_emit(self, events: List[dict]):
        for txn in events:
            self.tracker.register(txn)
            label_str = f"  label={'FRAUD' if txn['label'] == 1 else 'legit'}" \
                        if txn.get("label") is not None else ""
            logger.info(
                f"[RECEIVED]   {txn['transaction_id']}  "
                f"amount=${txn['amount']:>10.2f}{label_str}"
            )
            ok = self.queue.put(txn)
            if ok:
                self.tracker.update(txn["transaction_id"], TxnStatus.QUEUED)
                logger.debug(f"[QUEUED]     {txn['transaction_id']}  (q={self.queue.size})")
            else:
                logger.warning(f"[DROP]       {txn['transaction_id']}  — queue full!")

    def _on_completed(self, txn: dict):
        self._completed += 1
        score    = txn.get("fraud_probability", 0)
        decision = txn.get("decision", "?")
        risk     = txn.get("risk_level", "?")
        conf     = txn.get("confidence_score", 0)
        emoji    = _EMOJI.get(decision, "❓")
        summary  = txn.get("explanation_summary", "")[:80]
        warnings = txn.get("rule_warnings", [])

        logger.info(
            f"[COMPLETED]  {txn['transaction_id']}  "
            f"score={score:.4f}  risk={risk:<6}  "
            f"decision={decision:<6} {emoji}  conf={conf:.2f}"
        )
        if summary:
            logger.info(f"             → {summary}")
        for w in warnings:
            logger.warning(f"             ⚡ RULE: {w}")

        if self._completed % self._summary_every == 0:
            self._print_summary()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        logger.info("=" * 65)
        logger.info("  FRAUD PREVENTION STREAM ENGINE  —  STARTING")
        logger.info("=" * 65)
        self.processor.start()
        self.generator.start()

    def stop(self):
        logger.info("[ENGINE] Shutting down …")
        self.generator.stop()
        # Drain remaining queue (max 15s)
        t0 = time.monotonic()
        while not self.queue.is_empty() and (time.monotonic() - t0) < 15:
            time.sleep(0.2)
        self.processor.stop()
        self._print_summary()
        self._print_fraud_stats()
        logger.info("[ENGINE] Shutdown complete.")

    def pause(self):
        self.generator.pause()

    def resume(self):
        self.generator.resume()

    def set_delay(self, delay: float):
        self.generator.set_delay(delay)

    def wait(self):
        while self.generator.is_alive() or not self.queue.is_empty():
            time.sleep(0.5)
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def _print_summary(self):
        s  = self.tracker.summary()
        ds = self.tracker.decision_summary()
        completed = self.tracker.filter_by_status(TxnStatus.COMPLETED)

        lines = ["", "─" * 50, "  PIPELINE SUMMARY"]
        for k, v in s.items():
            lines.append(f"    {k:<15} : {v:>6,}")

        lines.append("  DECISIONS")
        for k, v in ds.items():
            emoji = _EMOJI.get(k, "  ")
            lines.append(f"    {emoji} {k:<12} : {v:>6,}")

        if completed:
            scores = [r["score"] for r in completed if r.get("score") is not None]
            if scores:
                avg = sum(scores) / len(scores)
                lines.append(f"    Avg fraud score  : {avg:.4f}")

        lines.append("─" * 50)
        for l in lines:
            logger.info(l)

    def _print_fraud_stats(self):
        stats = self.tracker.fraud_stats()
        if not stats:
            return
        tp = stats["TP"]; fp = stats["FP"]
        tn = stats["TN"]; fn = stats["FN"]
        total = stats["total_evaluated"]
        fpr  = fp / (fp + tn + 1e-9)
        tpr  = tp / (tp + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)

        lines = [
            "", "─" * 50,
            f"  LIVE EVALUATION  (vs ground-truth labels, n={total:,})",
            f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}",
            f"    False Positive Rate : {fpr:.4f}",
            f"    True Positive Rate  : {tpr:.4f}",
            f"    Precision           : {prec:.4f}",
            "─" * 50,
        ]
        for l in lines:
            logger.info(l)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fraud Prevention — Pseudo Real-Time Stream Engine"
    )
    parser.add_argument("--csv",        default=None,
                        help="Path to creditcard.csv")
    parser.add_argument("--rows",       type=int, default=500,
                        help="Max transactions to stream")
    parser.add_argument("--delay",      type=float, default=STREAM_DELAY,
                        help="Fixed delay between emissions (s)")
    parser.add_argument("--batch",      type=int, default=STREAM_BATCH_SIZE,
                        help="Transactions per emission")
    parser.add_argument("--loop",       action="store_true",
                        help="Loop dataset forever")
    parser.add_argument("--no-time-col", action="store_true",
                        help="Disable Time-column-based delays (use fixed --delay)")
    parser.add_argument("--time-scale", type=float, default=100.0,
                        help="Compression factor for time-col delays (default 100×)")
    parser.add_argument("--summary-every", type=int, default=50,
                        help="Print summary every N completed transactions")
    parser.add_argument("--log-level",  default=LOG_LEVEL)
    parser.add_argument("--log-file",   default=LOG_FILE)
    args = parser.parse_args()

    _setup_logging(level=args.log_level, log_file=args.log_file)

    engine = TransactionStreamEngine(
        csv_path=args.csv,
        max_rows=args.rows,
        delay=args.delay,
        batch_size=args.batch,
        loop=args.loop,
        use_time_col=not args.no_time_col,
        time_scale=args.time_scale,
        print_summary_every=args.summary_every,
    )

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
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            engine.stop()


if __name__ == "__main__":
    main()
