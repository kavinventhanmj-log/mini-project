"""
config.py
---------
Central configuration for the entire fraud prevention system.
Edit values here — nothing else needs to change.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

DATASET_PATH    = os.path.join(DATA_DIR, "creditcard.csv")
MODEL_PATH      = os.path.join(MODEL_DIR, "model.pkl")
THRESHOLD_PATH  = os.path.join(MODEL_DIR, "thresholds.json")
FP_MEMORY_PATH  = os.path.join(MODEL_DIR, "fp_memory.json")
SCALER_PATH     = os.path.join(MODEL_DIR, "scaler.pkl")

# ── Dataset ────────────────────────────────────────────────────────────────
TARGET_COL      = "Class"
DROP_COLS       = []                        # columns to drop before training
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ── Feature Engineering ────────────────────────────────────────────────────
AMOUNT_LOG      = True                      # add log(Amount+1) feature
TIME_FEATURES   = True                      # add hour_bucket, is_night
AMOUNT_BINS     = [0, 10, 50, 200, 500, 1000, 5000, 1e9]
AMOUNT_BIN_LABELS = ["micro","small","medium","large","xlarge","huge","extreme"]

# ── LightGBM ───────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           ["binary_logloss", "auc"],
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "max_depth":        -1,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "verbose":          -1,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
}

# ── Decision Thresholds (auto-computed during training, defaults here) ──────
DEFAULT_ALLOW_THRESHOLD  = 0.3     # below → ALLOW
DEFAULT_BLOCK_THRESHOLD  = 0.6     # above → BLOCK  (between → REVIEW)

# ── Rule Engine ────────────────────────────────────────────────────────────
RULE_EXTREME_AMOUNT      = 5000.0  # flag amounts above this
RULE_RAPID_TXN_WINDOW    = 60      # seconds (for future use)
RULE_RAPID_TXN_MAX       = 5       # max txns in window (for future use)

# ── False Positive Memory ──────────────────────────────────────────────────
FP_MEMORY_MAX_RECORDS    = 10_000
FP_SIMILARITY_THRESHOLD  = 0.85    # cosine similarity to flag as known-legit

# ── SHAP ───────────────────────────────────────────────────────────────────
SHAP_TOP_N_FEATURES      = 5

# ── Stream Engine ──────────────────────────────────────────────────────────
STREAM_DELAY             = 0.3     # seconds between transactions
STREAM_BATCH_SIZE        = 1
STREAM_QUEUE_MAXSIZE     = 500

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL                = "INFO"
LOG_FILE                 = os.path.join(LOG_DIR, "stream_engine.log")

# ── Ensure dirs exist ──────────────────────────────────────────────────────
for _d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)
