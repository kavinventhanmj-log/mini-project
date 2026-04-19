# Pseudo Real-Time Transaction Stream Engine

> Phase 1 of the **Explainable Hybrid False-Positive Prevention System**  
> Pure backend simulation — no UI, no SHAP, no full decision engine yet.

---

## Project Structure

```
transaction_stream_engine/
├── stream_engine.py      ← Main runner / orchestrator (start here)
├── generator.py          ← Synthetic/CSV stream generator
├── queue_manager.py      ← FIFO queue + status tracker
├── processor.py          ← Processing pipeline (ML integration seam)
├── models/
│   ├── __init__.py
│   └── predict.py        ← DROP YOUR REAL MODEL HERE later
├── requirements.txt
└── stream_engine.log     ← Created at runtime
```

---

## Setup on Pop!_OS with VSCode

### 1. Install Python (if not already installed)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
python3 --version     # confirm 3.10+
```

### 2. Open the project in VSCode

```bash
code /path/to/transaction_stream_engine
```

Or: File → Open Folder → select `transaction_stream_engine/`

### 3. Create and activate a virtual environment

Open the integrated terminal in VSCode (**Ctrl + `**):

```bash
# Inside the project folder
python3 -m venv .venv
source .venv/bin/activate
```

VSCode will detect the venv and ask you to select it as the Python interpreter.  
Click **Yes**, or press **Ctrl+Shift+P** → "Python: Select Interpreter" → choose `.venv`.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Engine

### Default run (synthetic data, 500 transactions, 0.3 s delay)

```bash
python stream_engine.py
```

### With a real CSV (Kaggle creditcard.csv)

```bash
python stream_engine.py --csv /path/to/creditcard.csv --rows 300
```

### Batch mode (2 transactions at a time, faster)

```bash
python stream_engine.py --batch 2 --delay 0.2
```

### Loop forever (until Ctrl+C)

```bash
python stream_engine.py --loop --delay 0.1
```

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | None (synthetic) | Path to CSV dataset |
| `--rows` | 500 | Max transactions to stream |
| `--delay` | 0.3 | Seconds between emissions |
| `--batch` | 1 | Transactions per emission |
| `--threshold` | 0.5 | Fraud score cut-off for BLOCK |
| `--loop` | False | Restart dataset when exhausted |
| `--log-level` | INFO | DEBUG / INFO / WARNING |
| `--log-file` | stream_engine.log | Path to log file |

---

## Sample Log Output

```
10:42:01  INFO      [GENERATOR] Generated 500 synthetic transactions.
10:42:01  INFO      [GENERATOR] Stream started  │  delay=0.3s  │  batch=1
10:42:01  INFO      [RECEIVED]   TXN_000000  amount=$142.50
10:42:01  INFO      [QUEUED]     TXN_000000  (queue size: 1)
10:42:01  INFO      [PROCESSING] TXN_000000
10:42:01  INFO      [SCORED]     TXN_000000  →  score: 0.0285
10:42:01  INFO      [COMPLETED]  TXN_000000  →  decision: PASS ✅
10:42:01  INFO      [RECEIVED]   TXN_000001  amount=$1842.30
10:42:01  INFO      [QUEUED]     TXN_000001  (queue size: 1)
10:42:01  INFO      [PROCESSING] TXN_000001
10:42:01  INFO      [SCORED]     TXN_000001  →  score: 0.7213
10:42:01  INFO      [COMPLETED]  TXN_000001  →  decision: BLOCK 🚨
```

---

## Plugging in Your Real ML Model

1. Train your model and save it (e.g. `models/fraud_rf.pkl`)
2. Open `models/predict.py`
3. Replace the stub body with your real inference code:

```python
import joblib, numpy as np

_model = joblib.load("models/fraud_rf.pkl")

def predict_transaction(txn: dict) -> float:
    features = list(txn["features"].values())   # V1–V28
    X = np.array(features).reshape(1, -1)
    return float(_model.predict_proba(X)[0, 1])
```

**That's all.** The processor auto-discovers `predict_transaction` at startup.  
No changes to `stream_engine.py`, `processor.py`, or `queue_manager.py` needed.

---

## Transaction Lifecycle

```
RECEIVED → QUEUED → PROCESSING → SCORED → COMPLETED
                                        ↘ FAILED (on error)
```

---

## Next Phases (future integration points)

| Phase | File to add / modify |
|-------|----------------------|
| Real ML model | `models/predict.py` |
| SHAP explainability | `processor.py` → scoring stage |
| Decision engine | `processor.py` → `_make_decision()` |
| UI dashboard | Wire `StatusTracker` to a WebSocket / REST API |
| Persistent storage | Swap `tracker._records` dict for a database |
