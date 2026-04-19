# pseudo_stream_engine.py

import pandas as pd
import time


class PseudoTransactionEngine:
    def __init__(self, csv_file, delay_before_next=1):
        self.csv_file = csv_file
        self.delay_before_next = delay_before_next
        self.data = None

    def load_dataset(self):
        print("\n[ENGINE] Loading dataset...")
        self.data = pd.read_csv(self.csv_file)

        # Add Transaction ID if not present
        if "TransactionID" not in self.data.columns:
            self.data.insert(0, "TransactionID", [f"TXN{i+1:04d}" for i in range(len(self.data))])

        print(f"[ENGINE] Dataset loaded successfully.")
        print(f"[ENGINE] Total transactions available: {len(self.data)}\n")

    def display_transaction(self, txn):
        print("=" * 70)
        print(f"[STREAM] New Transaction Fetched: {txn['TransactionID']}")
        print("-" * 70)

        # Show only important columns first
        important_cols = ["TransactionID", "Time", "Amount"]
        for col in important_cols:
            if col in txn:
                print(f"{col:<15}: {txn[col]}")

        # Show remaining columns if needed
        extra_cols = [col for col in txn.index if col not in important_cols]
        if extra_cols:
            print("\n[FEATURE SNAPSHOT]")
            for col in extra_cols[:8]:   # only first few for clean terminal demo
                print(f"{col:<15}: {txn[col]}")

        print("-" * 70)
        print("[STATUS] Waiting for validation...")
        print("=" * 70)

    def validate_transaction(self, txn):
        """
        Theoretical validation stage for demo.
        Here we simulate validation instead of real validation logic.
        """

        print(f"\n[VALIDATION] Validating {txn['TransactionID']} ...")
        time.sleep(2)

        # Demo-only theoretical validation
        validation_result = "VALIDATED"

        print(f"[VALIDATION] {txn['TransactionID']} -> {validation_result}")
        return validation_result

    def decision_stage(self, txn):
        """
        This is just for prototype flow display.
        Since you said validation is only in theory stage,
        decision here is also a demo placeholder.
        """

        print(f"[DECISION] Processing decision for {txn['TransactionID']} ...")
        time.sleep(1)

        amount = txn.get("Amount", 0)

        # Simple demo-only placeholder decision
        if amount > 2000:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        print(f"[DECISION] {txn['TransactionID']} -> {decision}")
        return decision

    def run_stream(self):
        if self.data is None:
            print("[ERROR] Dataset not loaded.")
            return

        print("\n================ PSEUDO TRANSACTION STREAM STARTED ================\n")

        for _, txn in self.data.iterrows():
            self.display_transaction(txn)

            # Pause until user says continue validation
            input("Press ENTER to start validation for this transaction...")

            validation_status = self.validate_transaction(txn)

            if validation_status == "VALIDATED":
                decision = self.decision_stage(txn)
            else:
                decision = "REJECTED"

            print(f"[FINAL STATUS] {txn['TransactionID']} completed with decision: {decision}\n")

            input("Press ENTER to fetch the next transaction...")
            time.sleep(self.delay_before_next)

        print("\n================ ALL TRANSACTIONS PROCESSED =================\n")


if __name__ == "__main__":
    # Change this to your dataset file name
    csv_path = "creditcard.csv"

    engine = PseudoTransactionEngine(csv_path, delay_before_next=1)
    engine.load_dataset()
    engine.run_stream()