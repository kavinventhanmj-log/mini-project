# Sentinel (MVP 3.0) - Implementation Documentation

This document outlines the technical architecture, implementation details, and core logic of the Sentinel Transaction Monitoring System.

---

## 🏗️ Technical Architecture

Sentinel is built as a modular, full-stack application designed for high-throughput transaction monitoring with a focus on **Precision** and **False Positive Reduction**.

### 1. Data Generation & Simulation
- **File:** `data_generator.py`
- **Purpose:** Generates a synthetic dataset of 25,000 transactions designed to mimic real-world banking patterns.
- **Bucket Logic:** Unlike standard datasets, Sentinel uses a "bucketed" approach to simulate specific model failure modes:
    - **TP (True Positive):** Obvious fraud.
    - **FP (False Positive):** Legit transactions with "shady" signals (unusual amount, new device).
    - **FN (False Negative):** Sophisticated fraud that mimics legit behavior.
    - **TN (True Negative):** Clearly safe transactions.
- **Focus:** The generator is weighted heavily towards FP (50%) to stress-test the system's ability to recover legit transactions via human review.

### 2. Machine Learning Pipeline
- **Algorithm:** LightGBM Classifier.
- **Training (`train.py`):**
    - Uses `scale_pos_weight` to handle class imbalance.
    - Optimized for **High Precision** to minimize automatic blocking of legit customers.
    - **Incremental Retraining:** The training script is designed to incorporate human feedback from `feedback_logs/`, allowing the model to learn from manual corrections.
- **Feature Engineering (`feature_engine.py`):**
    - **Transaction Velocity:** Count of transactions in a short window.
    - **Amount-to-Balance Ratio:** Identifies disproportionately large spends.
    - **Known Device Check:** Flags transactions from unrecognized hardware.
    - **Encoding:** Robust handling of categorical features (Location, Merchant, Device).

### 3. Decision Engine
- **Thresholds:**
    - `risk_score < 0.30`: **ALLOW** (Auto-cleared).
    - `0.30 <= risk_score < 0.70`: **REVIEW** (Queued for human inspection).
    - `risk_score >= 0.70`: **BLOCK** (Auto-blocked, but may be sent to review if score is below 0.82 to avoid FP).
- **Verification Logic:** The system identifies "high-risk FP areas" (e.g., scores between 0.70 and 0.82) and flags them for human verification even if they meet the auto-block threshold.

### 4. AI Explainability (SHAP)
- **Engine:** `shap.TreeExplainer`
- **Integration:** For every flagged transaction (REVIEW/BLOCK), the system generates a SHAP-based explanation.
- **Reason Mapping:** Raw feature importance values are mapped to high-signal human reasons (e.g., `txn_velocity` → "High transaction velocity").
- **Post-Processing:**
    - **Ranking:** Features are ranked by their absolute SHAP value.
    - **Deduplication:** Similar reasons are deduplicated while preserving order.
    - **Padding:** If fewer than 4 distinct reasons are found, the system pads the output with a default "Anomalous transaction pattern" reason to ensure a consistent UI layout.
- **Output:** A ranked list of the Top 4 features contributing to the risk score.

### 5. Backend (FastAPI)
- **Core:** `api.py`
- **Concurrency:** Uses a background thread to run the transaction pipeline without blocking the API.
- **Real-Time Streaming:** Implements WebSockets (`/ws/stream`) to push transaction results to the UI as they are processed.
- **State Management:** Maintains an in-memory state of the current pipeline run, review queue, and reviewer decisions.
- **Feedback Loop:** Human decisions are persisted as individual CSV files in `feedback_logs/` for auditability and retraining.

### 6. Frontend (Next.js)
- **Dashboard:** Real-time visualization of the transaction stream.
- **Review Queue:** A dedicated interface for human analysts to inspect flagged transactions, view AI explanations, and make Final Decisions (Allow/Block).
- **Analytics:** Tracks real-time metrics like Precision, Recall, and the "FP Reduction Rate"—the percentage of legit transactions saved from auto-blocking.

---

## 🔄 Human-in-the-Loop Workflow

1.  **Detection:** The model scores a transaction.
2.  **Flagging:** If the score is borderline, the transaction is marked as `REVIEW`.
3.  **Human Analysis:** The analyst views the transaction detail and the **AI Explanation** (SHAP).
4.  **Final Decision:** The analyst clicks **Allow** or **Block**.
5.  **Learning:** The decision is saved and the model is automatically queued for retraining with the new "ground truth" provided by the human.

---

## 📈 Key Metrics Tracked
- **FP Reduction Rate:** Measures how many legitimate customers were *not* blocked thanks to the Review Queue.
- **Precision (Model):** Accuracy of fraud flags.
- **Recall (System):** Percentage of total fraud caught (Auto-Block + Human Block).
