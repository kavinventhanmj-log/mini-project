"""
review_panel.py
===============
Terminal-based interactive review panel for REVIEW-flagged transactions.

Wired into main.py's pipeline — after streaming, all REVIEW-bucket transactions
are queued here. The reviewer sees:
  - Full transaction details
  - SHAP-based explanation of WHY it was flagged
  - A prompt to ALLOW or BLOCK the transaction

Usage (standalone): python review_panel.py
Usage (from main):  called automatically after pipeline run if review_queue is non-empty.

Run order:
  1. python data_generator.py
  2. python train.py
  3. python main.py   ← now calls review_panel at the end
"""

import pickle
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from feature_engine import engineer_features, features_to_dataframe, load_encoders

MODEL_FILE   = "model.pkl"
ENCODER_FILE = "encoders.pkl"

# ── Human-readable reason map ──────────────────────────────────────────────────
FEATURE_REASONS = {
    "txn_velocity"            : ("High transaction velocity",
                                 "Too many transactions fired in a short window — a classic account-takeover signal."),
    "is_known_device"         : ("Unrecognised / new device",
                                 "Transaction originated from a device never seen before for this account."),
    "amount"                  : ("Unusual transaction amount",
                                 "The amount is significantly higher than this user's typical spend pattern."),
    "amount_to_balance_ratio" : ("Amount unusually high vs balance",
                                 "The transaction consumes a disproportionately large chunk of available balance."),
    "merchant"                : ("Unusual merchant",
                                 "This merchant category is atypical for this user or is flagged as high-risk."),
    "login_attempts"          : ("Multiple failed login attempts",
                                 "Several failed logins before this transaction — possible credential stuffing."),
    "time_since_last_txn"     : ("Rapid back-to-back transaction",
                                 "Very little time elapsed since the previous transaction — possible automated fraud."),
    "location"                : ("Unusual location",
                                 "Transaction initiated from a location inconsistent with user history."),
    "balance"                 : ("Low balance anomaly",
                                 "Account balance is unusually low relative to the transaction amount."),
    "device"                  : ("Device type mismatch",
                                 "The device category doesn't match the user's typical access pattern."),
}


# ── Loaders (lazy, cached) ─────────────────────────────────────────────────────
_model    = None
_encoders = None
_explainer = None

def _load():
    global _model, _encoders, _explainer
    if _model is None:
        with open(MODEL_FILE, "rb") as f:
            _model = pickle.load(f)
        _encoders  = load_encoders(ENCODER_FILE)
        _explainer = shap.TreeExplainer(_model)


# ── SHAP explanation builder ───────────────────────────────────────────────────
def build_shap_explanation(feature_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """
    Returns a list of dicts:
        { feature, label, short_reason, detail, shap_value, direction }
    Sorted by absolute SHAP value descending.
    """
    shap_values = _explainer.shap_values(feature_df)
    # For binary classifiers shap_values is a list [neg_class, pos_class]
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    explanations = []
    for i, col in enumerate(feature_df.columns):
        val       = feature_df.iloc[0][col]
        shap_val  = sv[i]
        info      = FEATURE_REASONS.get(col, (col.replace("_", " ").title(), ""))
        explanations.append({
            "feature"      : col,
            "label"        : info[0],
            "detail"       : info[1],
            "raw_value"    : val,
            "shap_value"   : shap_val,
            "direction"    : "↑ RISK" if shap_val > 0 else "↓ RISK",
        })

    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanations[:top_n]


# ── Pretty printers ────────────────────────────────────────────────────────────
def _bar(value: float, max_val: float = 0.4, width: int = 20) -> str:
    filled = int(min(abs(value) / max_val, 1.0) * width)
    char   = "█" if value > 0 else "░"
    return char * filled + "·" * (width - filled)


def print_review_card(txn_num: int, total: int, txn: dict,
                      risk_score: float, explanations: list[dict]):
    W  = 68
    sep_heavy = "═" * W
    sep_light = "─" * W

    # ── Header ──
    print(f"\n{sep_heavy}")
    print(f"  🔍  REVIEW PANEL  │  Transaction {txn_num} of {total}")
    print(sep_heavy)

    # ── Transaction Details ──
    print(f"\n  {'TRANSACTION DETAILS':^{W-4}}")
    print(f"  {sep_light}")
    print(f"  {'Transaction ID':<24} {txn.get('transaction_id', 'N/A')}")
    print(f"  {'User ID':<24} {txn.get('user_id', 'N/A')}")
    print(f"  {'Amount':<24} ₹{float(txn.get('amount', 0)):>12,.2f}")
    print(f"  {'Merchant':<24} {txn.get('merchant', 'N/A')}")
    print(f"  {'Device':<24} {txn.get('device', 'N/A')}")
    print(f"  {'Location':<24} {txn.get('location', 'N/A')}")
    print(f"  {'Timestamp':<24} {txn.get('timestamp', 'N/A')}")
    print(f"  {'Balance':<24} ₹{float(txn.get('balance', 0)):>12,.2f}")
    print(f"  {'Login Attempts':<24} {txn.get('login_attempts', 0)}")
    print(f"  {'Txn Velocity':<24} {txn.get('txn_velocity', 0)}")
    print(f"  {'Known Device?':<24} {'Yes' if txn.get('is_known_device') else 'No'}")
    print(f"  {'Amt/Balance Ratio':<24} {float(txn.get('amount_to_balance_ratio', 0)):.4f}")

    # ── Risk Score ──
    score_bar_len = int(risk_score * 40)
    score_bar     = "█" * score_bar_len + "·" * (40 - score_bar_len)
    print(f"\n  {sep_light}")
    risk_label = "HIGH" if risk_score >= 0.70 else "MEDIUM"
    print(f"  ⚠️   RISK SCORE : {risk_score:.4f}  [{risk_label}]")
    print(f"  [{score_bar}]")
    print(f"  Threshold → REVIEW if score ≥ 0.30 | BLOCK if score ≥ 0.70")

    # ── SHAP Explanation ──
    print(f"\n  {sep_light}")
    print(f"  🧠  WHY WAS THIS FLAGGED? (SHAP-based explanation)")
    print(f"  {sep_light}")
    print(f"  {'#':<3} {'Feature':<28} {'Value':>10}  {'SHAP':>8}  {'Bar':<22} Impact")
    print(f"  {'─'*3} {'─'*28} {'─'*10}  {'─'*8}  {'─'*22} {'─'*8}")

    for rank, exp in enumerate(explanations, 1):
        raw  = exp["raw_value"]
        sv   = exp["shap_value"]
        bar  = _bar(sv)
        dirn = exp["direction"]
        val_str = f"{raw:.4f}" if isinstance(raw, float) else str(raw)
        print(f"  {rank:<3} {exp['label']:<28} {val_str:>10}  {sv:>+8.4f}  {bar:<22} {dirn}")

    print(f"\n  Top reason: {explanations[0]['label']}")
    print(f"  ↳ {explanations[0]['detail']}")
    if len(explanations) > 1:
        print(f"\n  2nd reason: {explanations[1]['label']}")
        print(f"  ↳ {explanations[1]['detail']}")

    print(f"\n  {'─'*W}")
    print(f"  Bucket label (from data): {txn.get('bucket', '?')}   "
          f"│  Actual fraud? {'YES ⚠️' if txn.get('is_fraud') else 'NO ✅'}")
    print(f"  {sep_heavy}")


def get_reviewer_decision(txn_num: int) -> str:
    """Prompt reviewer. Returns 'ALLOW' or 'BLOCK'."""
    while True:
        try:
            choice = input(
                f"\n  ➤  Your decision for Txn #{txn_num}:\n"
                f"       [A] ALLOW  — legitimate transaction, release it\n"
                f"       [B] BLOCK  — confirmed suspicious, block it\n"
                f"       [S] SKIP   — decide later (mark as PENDING)\n"
                f"\n  Enter choice (A / B / S): "
            ).strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  ⚠️  Review session interrupted. Remaining transactions marked PENDING.")
            return "PENDING"

        if choice in ("A", "B", "S"):
            return {"A": "ALLOW", "B": "BLOCK", "S": "PENDING"}[choice]
        print("  ❌  Invalid input. Please enter A, B, or S.")


# ── Summary report ─────────────────────────────────────────────────────────────
def print_review_summary(review_decisions: list[dict]):
    W   = 68
    sep = "═" * W
    total   = len(review_decisions)
    allowed = sum(1 for r in review_decisions if r["final_decision"] == "ALLOW")
    blocked = sum(1 for r in review_decisions if r["final_decision"] == "BLOCK")
    pending = sum(1 for r in review_decisions if r["final_decision"] == "PENDING")

    # Accuracy: how often did reviewer agree with ground truth?
    correct = 0
    for r in review_decisions:
        actual = r.get("is_fraud", 0)
        dec    = r["final_decision"]
        if (actual == 0 and dec == "ALLOW") or (actual == 1 and dec == "BLOCK"):
            correct += 1
    accuracy = correct / total if total > 0 else 0

    print(f"\n{sep}")
    print(f"  📋  REVIEW SESSION SUMMARY")
    print(f"  {sep}")
    print(f"  Total Reviewed          : {total:>5}")
    print(f"  ✅  ALLOWed             : {allowed:>5}  ({allowed/total*100:.1f}%)")
    print(f"  🚫  BLOCKed             : {blocked:>5}  ({blocked/total*100:.1f}%)")
    print(f"  ⏳  PENDING (skipped)   : {pending:>5}  ({pending/total*100:.1f}%)")
    print(f"\n  Reviewer Accuracy       : {accuracy:.2%}  (vs ground truth)")
    print(f"  {sep}")

    # Per-decision breakdown
    print(f"\n  {'TXN ID':<14} {'User':<10} {'Amount':>12}  {'Score':>7}  "
          f"{'Actual':^8}  {'Decision'}")
    print(f"  {'─'*14} {'─'*10} {'─'*12}  {'─'*7}  {'─'*8}  {'─'*10}")
    for r in review_decisions:
        fraud_label = "FRAUD ⚠️" if r.get("is_fraud") else "LEGIT  "
        dec_icon = {"ALLOW": "✅ ALLOW", "BLOCK": "🚫 BLOCK", "PENDING": "⏳ SKIP"}
        print(f"  {r['transaction_id']:<14} {r['user_id']:<10} "
              f"₹{float(r['amount']):>11,.2f}  {r['risk_score']:>7.4f}  "
              f"{fraud_label:<8}  {dec_icon.get(r['final_decision'], r['final_decision'])}")

    print(f"\n{sep}\n")


# ── Main entry point (also callable from main.py) ─────────────────────────────
def run_review_panel(review_queue: list[dict]) -> list[dict]:
    """
    review_queue : list of txn dicts that were flagged as REVIEW in main.py.
                   Each dict must contain at least the standard transaction fields
                   plus 'transaction_id', 'risk_score'.

    Returns list of dicts with added key 'final_decision'.
    """
    if not review_queue:
        print("\n  ✅  No transactions queued for review. Review panel idle.\n")
        return []

    _load()

    total           = len(review_queue)
    review_decisions = []

    print(f"\n{'═'*68}")
    print(f"  🔍  INTERACTIVE REVIEW PANEL  —  {total} transaction(s) to review")
    print(f"{'═'*68}")
    print(f"  Instructions:")
    print(f"  • Read each transaction carefully + study the SHAP breakdown.")
    print(f"  • Decide: ALLOW (safe), BLOCK (fraud), or SKIP (decide later).")
    print(f"  • Press Ctrl+C at any time to abort — remaining will be PENDING.")
    print(f"{'═'*68}\n")
    input("  Press ENTER to begin review session...")

    for idx, txn in enumerate(review_queue, start=1):
        # Re-run feature engineering + SHAP for this txn
        features   = engineer_features(txn, encoders=_encoders)
        feature_df = features_to_dataframe(features)
        risk_score = txn.get("risk_score", float(_model.predict_proba(feature_df)[0][1]))
        explanations = build_shap_explanation(feature_df, top_n=5)

        print_review_card(idx, total, txn, risk_score, explanations)
        decision = get_reviewer_decision(txn.get("transaction_id", f"TXN#{idx}"))

        review_decisions.append({
            "transaction_id" : txn.get("transaction_id", f"TXN{idx:07d}"),
            "user_id"        : txn.get("user_id", "N/A"),
            "amount"         : txn.get("amount", 0),
            "is_fraud"       : txn.get("is_fraud", 0),
            "bucket"         : txn.get("bucket", "?"),
            "risk_score"     : risk_score,
            "final_decision" : decision,
        })

        icon = {"ALLOW": "✅", "BLOCK": "🚫", "PENDING": "⏳"}[decision]
        print(f"\n  {icon}  Decision recorded: {decision}  "
              f"({idx}/{total} reviewed)\n")

        if idx < total:
            try:
                input("  Press ENTER for next transaction...")
            except (EOFError, KeyboardInterrupt):
                # Mark remaining as PENDING
                for remaining in review_queue[idx:]:
                    review_decisions.append({
                        "transaction_id" : remaining.get("transaction_id", "N/A"),
                        "user_id"        : remaining.get("user_id", "N/A"),
                        "amount"         : remaining.get("amount", 0),
                        "is_fraud"       : remaining.get("is_fraud", 0),
                        "bucket"         : remaining.get("bucket", "?"),
                        "risk_score"     : remaining.get("risk_score", 0),
                        "final_decision" : "PENDING",
                    })
                break

    print_review_summary(review_decisions)
    return review_decisions


# ── Standalone mode (test with dummy data) ────────────────────────────────────
if __name__ == "__main__":
    dummy_queue = [
        {
            "transaction_id"       : "TXN0000042",
            "user_id"              : "U0123",
            "amount"               : 45000.00,
            "device"               : "tablet",
            "location"             : "Mumbai",
            "merchant"             : "IRCTC",
            "timestamp"            : "2024-03-15 02:47:00",
            "login_attempts"       : 3,
            "balance"              : 52000.00,
            "time_since_last_txn"  : 0.04,
            "txn_velocity"         : 6,
            "is_known_device"      : 0,
            "amount_to_balance_ratio": 0.8653,
            "is_fraud"             : 0,
            "bucket"               : "FP",
            "risk_score"           : 0.61,
        },
        {
            "transaction_id"       : "TXN0000117",
            "user_id"              : "U0089",
            "amount"               : 32500.50,
            "device"               : "desktop_firefox",
            "location"             : "Ahmedabad",
            "merchant"             : "Amazon",
            "timestamp"            : "2024-04-02 14:22:00",
            "login_attempts"       : 2,
            "balance"              : 41000.00,
            "time_since_last_txn"  : 0.09,
            "txn_velocity"         : 5,
            "is_known_device"      : 0,
            "amount_to_balance_ratio": 0.7926,
            "is_fraud"             : 0,
            "bucket"               : "FP",
            "risk_score"           : 0.54,
        },
    ]

    run_review_panel(dummy_queue)