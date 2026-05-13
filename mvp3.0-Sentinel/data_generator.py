"""
data_generator.py
=================
Generates synthetic banking transaction dataset.

Overall bucket distribution (of 25,000 rows):
  FP          30 %  <- dominant: legit txns the model wrongly flags
    FP_ALLOW  12 %    legit + clean-ish  -> model ALLOW  (escaped FP)
    FP_REVIEW  9 %    legit + borderline -> model REVIEW (wrongly held)
    FP_BLOCK   9 %    legit + very shady -> model BLOCK  (wrongly blocked)
  TP          35 %  real fraud, detectable
  FN          15 %  real fraud, looks clean (model misses)
  TN          20 %  legit + clearly safe

Output: transactions.csv
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

SEED        = 42
NUM_ROWS    = 25_000
OUTPUT_FILE = "transactions.csv"

TP_RATIO  = 0.25
FP_RATIO  = 0.50
FN_RATIO  = 0.10
TN_RATIO  = 0.15

FP_ALLOW_RATIO  = 0.30
FP_REVIEW_RATIO = 0.35
FP_BLOCK_RATIO  = 0.35

np.random.seed(SEED)
random.seed(SEED)

DEVICES   = ["mobile_ios", "mobile_android", "desktop_chrome", "desktop_firefox", "tablet"]
LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
MERCHANTS = ["Amazon", "Flipkart", "IRCTC", "Zomato", "Swiggy", "BigBasket",
             "Myntra", "PhonePe", "ATM_Withdrawal", "Unknown_Merchant"]
USER_IDS  = [f"U{str(i).zfill(4)}" for i in range(1, 501)]


def make_TP():
    return dict(amount=round(np.random.uniform(25_000, 80_000), 2),
                device=random.choice(["mobile_android", "tablet"]),
                merchant="Unknown_Merchant",
                login_attempts=random.randint(3, 5),
                txn_velocity=random.randint(6, 10),
                is_known_device=0,
                time_since_last_txn=round(random.uniform(0.01, 0.08), 4),
                is_fraud=1, bucket="TP")


def make_FN():
    return dict(amount=round(np.random.uniform(500, 8_000), 2),
                device=random.choice(["mobile_ios", "desktop_chrome"]),
                merchant=random.choice(["Amazon", "Flipkart", "Zomato"]),
                login_attempts=random.randint(0, 1),
                txn_velocity=random.randint(0, 2),
                is_known_device=1,
                time_since_last_txn=round(random.uniform(2.0, 48.0), 4),
                is_fraud=1, bucket="FN")


def make_TN():
    return dict(amount=round(np.random.uniform(100, 8_000), 2),
                device=random.choice(["mobile_ios", "desktop_chrome"]),
                merchant=random.choice(["Amazon", "Zomato", "Swiggy", "PhonePe", "BigBasket"]),
                login_attempts=0,
                txn_velocity=random.randint(0, 2),
                is_known_device=1,
                time_since_last_txn=round(random.uniform(2.0, 72.0), 4),
                is_fraud=0, bucket="TN")


def make_FP_ALLOW():
    """Legit txn, mild signals. Model correctly lets through. ESCAPED FP."""
    return dict(amount=round(np.random.uniform(8_000, 18_000), 2),
                device=random.choice(["mobile_ios", "desktop_chrome", "desktop_firefox"]),
                merchant=random.choice(["IRCTC", "Amazon", "Myntra", "Flipkart"]),
                login_attempts=random.randint(0, 1),
                txn_velocity=random.randint(1, 3),
                is_known_device=random.choice([0, 1]),
                time_since_last_txn=round(random.uniform(0.5, 3.0), 4),
                is_fraud=0, bucket="FP_ALLOW")


def make_FP_REVIEW():
    """Legit txn, borderline signals. Model wrongly holds for REVIEW."""
    return dict(amount=round(np.random.uniform(18_000, 35_000), 2),
                device=random.choice(["tablet", "desktop_firefox", "mobile_android"]),
                merchant=random.choice(["IRCTC", "ATM_Withdrawal", "Flipkart", "Amazon"]),
                login_attempts=random.randint(1, 2),
                txn_velocity=random.randint(3, 5),
                is_known_device=0,
                time_since_last_txn=round(random.uniform(0.1, 0.5), 4),
                is_fraud=0, bucket="FP_REVIEW")


def make_FP_BLOCK():
    """Legit txn, very shady signals. Model wrongly BLOCKs. Most damaging FP."""
    return dict(amount=round(np.random.uniform(30_000, 70_000), 2),
                device=random.choice(["tablet", "mobile_android"]),
                merchant=random.choice(["ATM_Withdrawal", "Unknown_Merchant"]),
                login_attempts=random.randint(2, 3),
                txn_velocity=random.randint(5, 8),
                is_known_device=0,
                time_since_last_txn=round(random.uniform(0.02, 0.15), 4),
                is_fraud=0, bucket="FP_BLOCK")  # LEGIT — wrongly blocked


MAKERS = {
    "TP": make_TP, "FP_ALLOW": make_FP_ALLOW,
    "FP_REVIEW": make_FP_REVIEW, "FP_BLOCK": make_FP_BLOCK,
    "FN": make_FN, "TN": make_TN,
}

FP_SUB_NAMES   = ["FP_ALLOW", "FP_REVIEW", "FP_BLOCK"]
FP_SUB_WEIGHTS = [FP_ALLOW_RATIO, FP_REVIEW_RATIO, FP_BLOCK_RATIO]
TOP_NAMES      = ["TP", "FP", "FN", "TN"]
TOP_WEIGHTS    = [TP_RATIO, FP_RATIO, FN_RATIO, TN_RATIO]


def generate_dataset():
    base_time = datetime(2024, 1, 1)
    records   = []
    user_last_txn      = {uid: None for uid in USER_IDS}
    user_known_devices = {uid: {random.choice(DEVICES)} for uid in USER_IDS}
    top_buckets    = np.random.choice(TOP_NAMES,    size=NUM_ROWS, p=TOP_WEIGHTS)
    fp_sub_buckets = np.random.choice(FP_SUB_NAMES, size=NUM_ROWS, p=FP_SUB_WEIGHTS)

    for i in range(NUM_ROWS):
        uid      = random.choice(USER_IDS)
        ts       = base_time + timedelta(seconds=random.randint(0, 90 * 24 * 3600))
        location = random.choice(LOCATIONS)
        balance  = round(np.random.lognormal(10.5, 0.8), 2)
        top        = top_buckets[i]
        bucket_key = fp_sub_buckets[i] if top == "FP" else top
        overrides  = MAKERS[bucket_key]()
        row = dict(
            transaction_id=f"TXN{str(i).zfill(7)}",
            user_id=uid, 
            timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
            location=location, 
            balance=balance
        )
        row.update(overrides)
        row["amount_to_balance_ratio"] = round(row["amount"] / (balance + 1), 6)
        records.append(row)
        user_last_txn[uid] = ts
        user_known_devices[uid].add(row["device"])

    df = pd.DataFrame(records)
    col_order = ["transaction_id", "user_id", "amount", "device", "location", "merchant",
                 "timestamp", "login_attempts", "balance",
                 "time_since_last_txn", "txn_velocity", "is_known_device",
                 "amount_to_balance_ratio", "is_fraud", "bucket"]
    return df[col_order]


if __name__ == "__main__":
    print("Generating FP-dominant dataset...")
    print(f"  Rows   : {NUM_ROWS:,}")
    print(f"  Buckets: TP:{TP_RATIO:.0%}  FP:{FP_RATIO:.0%}  FN:{FN_RATIO:.0%}  TN:{TN_RATIO:.0%}")
    print(f"  FP split: ALLOW:{FP_ALLOW_RATIO:.0%}  REVIEW:{FP_REVIEW_RATIO:.0%}  BLOCK:{FP_BLOCK_RATIO:.0%}\n")
    df = generate_dataset()
    total = len(df)
    fraud = df["is_fraud"].sum()
    legit = total - fraud
    bc    = df["bucket"].value_counts()
    print(f"Done: {total:,} rows")
    print(f"  Fraud (TP+FN) : {fraud:,}  ({fraud/total*100:.1f}%)")
    print(f"  Legit (FP+TN) : {legit:,}  ({legit/total*100:.1f}%)")
    print(f"\nBucket breakdown:")
    for b in ["TP", "FP_ALLOW", "FP_REVIEW", "FP_BLOCK", "FN", "TN"]:
        n = bc.get(b, 0)
        print(f"  {b:<12} : {n:>6,}  ({n/total*100:.1f}%)")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")