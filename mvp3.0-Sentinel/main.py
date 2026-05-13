"""
main.py (v3) — Real-Time Transaction Monitoring Pipeline
=========================================================
Streams transactions → decision engine → review_panel for REVIEW-flagged txns.

Run order:
  1. python data_generator.py
  2. python train.py
  3. python main.py              ← you are here
"""
import time
import warnings
import pickle
import pandas as pd
import numpy as np
import shap
from streamer import stream_transactions
from feature_engine import engineer_features, features_to_dataframe, load_encoders
from review_panel import run_review_panel   # ← NEW

warnings.filterwarnings("ignore")

MODEL_FILE   = "model.pkl"
ENCODER_FILE = "encoders.pkl"
MAX_ROWS     = 5_000
ALLOW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70
STREAM_DELAY    = 0.0

FEATURE_REASONS = {
    "txn_velocity"            : "High transaction velocity",
    "is_known_device"         : "New / unrecognised device",
    "amount"                  : "Unusual transaction amount",
    "amount_to_balance_ratio" : "Amount unusually high relative to balance",
    "merchant"                : "Unusual merchant activity",
    "login_attempts"          : "Multiple failed login attempts",
    "time_since_last_txn"     : "Suspiciously rapid back-to-back transaction",
    "location"                : "Transaction from unusual location",
    "balance"                 : "Low account balance anomaly",
    "device"                  : "Device type mismatch",
}

MODEL_FEATURES = [
    "amount", "device", "location", "merchant",
    "login_attempts", "balance", "time_since_last_txn",
    "txn_velocity", "is_known_device", "amount_to_balance_ratio"
]


def load_artefacts():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    encoders = load_encoders(ENCODER_FILE)
    return model, encoders


def make_decision(risk_score: float) -> str:
    if risk_score < ALLOW_THRESHOLD:
        return "ALLOW"
    elif risk_score < BLOCK_THRESHOLD:
        return "REVIEW"
    return "BLOCK"


def get_top_reasons(explainer, feature_df: pd.DataFrame, top_n: int = 4) -> list[str]:
    shap_values = explainer.shap_values(feature_df)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    top_indices = np.argsort(np.abs(sv))[::-1][:top_n]
    return [FEATURE_REASONS.get(feature_df.columns[i],
            feature_df.columns[i].replace("_", " ").title()) for i in top_indices]


def print_stream_alert(txn_num: int, txn: dict, risk_score: float,
                       decision: str, reasons: list[str]):
    user     = txn.get("user_id", "N/A")
    amount   = txn.get("amount", 0)
    bucket   = txn.get("bucket", "?")

    if bucket == "FP" and decision == "BLOCK":
        border = "═" * 62
        print(f"\n⚠️  {'FALSE POSITIVE — WRONGLY AUTO-BLOCKED':^58} ⚠️")
        print(border)
        print(f"  Txn #      : {txn_num}")
        print(f"  User       : {user}   Amount: ₹{amount:,.2f}")
        print(f"  Risk Score : {risk_score:.4f}  ⛔ AUTO-BLOCKED (legit txn!)")
        print(f"  Why flagged:")
        for i, r in enumerate(reasons, 1):
            print(f"    {i}. {r}")
        print(border)

    elif bucket == "FP" and decision == "REVIEW":
        print(f"  ⚠️  FP-REVIEW | Txn #{txn_num:>5} | User: {user:<8} | "
              f"₹{amount:>10,.2f} | Score: {risk_score:.3f} → queued for review")

    elif bucket == "TP" and decision == "BLOCK":
        print(f"  🚨 FRAUD    | Txn #{txn_num:>5} | User: {user:<8} | "
              f"₹{amount:>10,.2f} | Score: {risk_score:.3f} | BLOCKED ✅")


def print_pipeline_report(results: list[dict], review_decisions: list[dict]):
    df = pd.DataFrame(results)
    total   = len(df)
    allows  = (df["decision"] == "ALLOW").sum()
    reviews = (df["decision"] == "REVIEW").sum()
    blocks  = (df["decision"] == "BLOCK").sum()

    df["pred_fraud"] = df["decision"].map({"ALLOW": 0, "REVIEW": 1, "BLOCK": 1})
    df["actual"]     = df["is_fraud"].fillna(0).astype(int)

    TP = ((df["pred_fraud"] == 1) & (df["actual"] == 1)).sum()
    FP = ((df["pred_fraud"] == 1) & (df["actual"] == 0)).sum()
    FN = ((df["pred_fraud"] == 0) & (df["actual"] == 1)).sum()
    TN = ((df["pred_fraud"] == 0) & (df["actual"] == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0

    fp_rows              = df[df["bucket"] == "FP"]
    fp_total             = len(fp_rows)
    fp_wrongly_blocked   = (fp_rows["decision"] == "BLOCK").sum()
    fp_wrongly_reviewed  = (fp_rows["decision"] == "REVIEW").sum()
    fp_correctly_allowed = (fp_rows["decision"] == "ALLOW").sum()
    fp_reduction_rate    = fp_correctly_allowed / fp_total if fp_total > 0 else 0

    # ── Account for reviewer overrides ──────────────────────────────────────
    reviewer_allows = sum(1 for r in review_decisions if r["final_decision"] == "ALLOW")
    reviewer_blocks = sum(1 for r in review_decisions if r["final_decision"] == "BLOCK")
    reviewer_pending= sum(1 for r in review_decisions if r["final_decision"] == "PENDING")
    # How many FPs were saved by reviewer (reviewer chose ALLOW on a legit txn)
    fp_saved_by_reviewer = sum(
        1 for r in review_decisions
        if r["final_decision"] == "ALLOW" and r.get("is_fraud", 0) == 0
    )

    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  📊  FULL PIPELINE EVALUATION REPORT")
    print(sep)
    print(f"  Total Processed              : {total:>6,}")
    print(f"  ✅  ALLOW (auto)             : {allows:>6,}  ({allows/total*100:.1f}%)")
    print(f"  ⚠️   REVIEW (queued)          : {reviews:>6,}  ({reviews/total*100:.1f}%)")
    print(f"  🚫  BLOCK (auto)             : {blocks:>6,}  ({blocks/total*100:.1f}%)")

    print(f"\n  {'─'*58}")
    print(f"  CONFUSION MATRIX  (pre-review)")
    print(f"  {'─'*58}")
    print(f"  TP — Real fraud caught       : {TP:>6,}")
    print(f"  FP — Legit wrongly flagged   : {FP:>6,}")
    print(f"  FN — Real fraud missed       : {FN:>6,}")
    print(f"  TN — Legit correctly allowed : {TN:>6,}")

    print(f"\n  {'─'*58}")
    print(f"  MODEL METRICS")
    print(f"  {'─'*58}")
    print(f"  Precision                    : {precision:.4f}")
    print(f"  Recall                       : {recall:.4f}")
    print(f"  F1-Score                     : {f1:.4f}")
    print(f"  False Positive Rate          : {fpr:.4f}  (↓ better)")

    print(f"\n  {'─'*58}")
    print(f"  🎯 FALSE POSITIVE ANALYSIS")
    print(f"  {'─'*58}")
    print(f"  Total FP txns in stream      : {fp_total:>6,}")
    print(f"  Wrongly BLOCKED (auto)       : {fp_wrongly_blocked:>6,}  ← bad")
    print(f"  Sent for REVIEW              : {fp_wrongly_reviewed:>6,}  ← recoverable")
    print(f"  Correctly ALLOWED (auto)     : {fp_correctly_allowed:>6,}  ← good")
    print(f"  FP Reduction Rate (auto)     : {fp_reduction_rate:.2%}")

    if review_decisions:
        print(f"\n  {'─'*58}")
        print(f"  👁️  REVIEWER DECISIONS  (on REVIEW queue)")
        print(f"  {'─'*58}")
        print(f"  Reviewed                     : {len(review_decisions):>6,}")
        print(f"  Reviewer ALLOW               : {reviewer_allows:>6,}")
        print(f"  Reviewer BLOCK               : {reviewer_blocks:>6,}")
        print(f"  Reviewer PENDING (skipped)   : {reviewer_pending:>6,}")
        print(f"  FPs rescued by reviewer      : {fp_saved_by_reviewer:>6,}  ← legit txns saved")

    print(f"\n  {'─'*58}")
    print(f"  BUCKET BREAKDOWN")
    print(f"  {'─'*58}")
    for bucket in ["TP", "FP", "FN", "TN"]:
        sub = df[df["bucket"] == bucket]
        c   = len(sub)
        if c == 0:
            continue
        bl = (sub["decision"] == "BLOCK").sum()
        rv = (sub["decision"] == "REVIEW").sum()
        al = (sub["decision"] == "ALLOW").sum()
        print(f"  {bucket} ({c:>5,}) → BLOCK:{bl:>4} | REVIEW:{rv:>4} | ALLOW:{al:>4}")

    print(sep)
    print(f"  ✅  Report complete.\n")


def main():
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  REAL-TIME TRANSACTION MONITORING PIPELINE  (v3)")
    print(sep)
    print(f"  Streaming {MAX_ROWS:,} transactions | Focus: False Positive Reduction\n")

    model, encoders = load_artefacts()
    explainer = shap.TreeExplainer(model)
    results       = []
    review_queue  = []          # ← REVIEW-flagged transactions accumulate here

    for txn_num, txn in enumerate(
            stream_transactions(delay_ms=0, max_rows=MAX_ROWS), start=1):

        features   = engineer_features(txn, encoders=encoders)
        feature_df = features_to_dataframe(features)
        risk_score = float(model.predict_proba(feature_df)[0][1])
        decision   = make_decision(risk_score)

        reasons = []
        if decision in ("REVIEW", "BLOCK"):
            reasons = get_top_reasons(explainer, feature_df)
            print_stream_alert(txn_num, txn, risk_score, decision, reasons)

        results.append({
            "txn_num"   : txn_num,
            "user_id"   : txn.get("user_id"),
            "amount"    : txn.get("amount"),
            "is_fraud"  : txn.get("is_fraud", 0),
            "bucket"    : txn.get("bucket", "UNKNOWN"),
            "risk_score": risk_score,
            "decision"  : decision,
            "reasons"   : " | ".join(reasons),
        })

        # ── Queue REVIEW transactions for the review panel ────────────────
        if decision == "REVIEW":
            review_queue.append({**txn, "risk_score": risk_score})

        if STREAM_DELAY > 0:
            time.sleep(STREAM_DELAY)

    # ── Automated pipeline report ─────────────────────────────────────────
    review_decisions = []
    print_pipeline_report(results, review_decisions)   # pre-review snapshot

    # ── Hand off to Review Panel ──────────────────────────────────────────
    if review_queue:
        print(f"\n  📥  {len(review_queue)} transactions queued for human review.")
        ans = input("  Launch Review Panel now? (Y / N): ").strip().upper()
        if ans == "Y":
            review_decisions = run_review_panel(review_queue)
            # Reprint final report with reviewer stats folded in
            print_pipeline_report(results, review_decisions)
        else:
            print(f"\n  ℹ️  Review Panel skipped. "
                  f"{len(review_queue)} transactions remain PENDING.\n")
    else:
        print("\n  ✅  No REVIEW transactions — Review Panel not needed.\n")


if __name__ == "__main__":
    main()