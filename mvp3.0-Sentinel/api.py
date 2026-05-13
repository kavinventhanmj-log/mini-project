"""
api.py — FastAPI Backend for Transaction Monitoring UI
=======================================================
Wraps all existing pipeline files and exposes them as REST + WebSocket endpoints.

Endpoints:
  GET  /api/status                  → health check
  POST /api/run-pipeline            → run full pipeline, store results in memory
  WS   /ws/stream                   → stream transactions live via WebSocket
  GET  /api/review-queue            → get all REVIEW-flagged transactions
  POST /api/review/{txn_id}         → submit reviewer decision
  GET  /api/report                  → full evaluation report
  GET  /api/report/reviewer         → reviewer-only stats

Install deps:
  pip install fastapi uvicorn websockets shap lightgbm pandas numpy scikit-learn

Run:
  uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import pickle
import threading
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import shap
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Import your existing pipeline modules ──────────────────────────────────────
from streamer import stream_transactions
from feature_engine import engineer_features, features_to_dataframe, load_encoders
from train import train as train_model, MODEL_PATH

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FILE      = "model.pkl"
ENCODER_FILE    = "encoders.pkl"
FEEDBACK_DIR    = "feedback_logs"
ALLOW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.70
MAX_ROWS        = 5_000

FEATURE_REASONS = {
    "txn_velocity"            : ("High transaction velocity",
                                 "Too many transactions in a short window — classic account-takeover signal."),
    "is_known_device"         : ("Unrecognised device",
                                 "Transaction from a device never seen before for this account."),
    "amount"                  : ("Unusual transaction amount",
                                 "Amount significantly higher than this user's typical spend."),
    "amount_to_balance_ratio" : ("Amount high vs balance",
                                 "Transaction consumes a disproportionately large chunk of available balance."),
    "merchant"                : ("Unusual merchant",
                                 "Merchant is atypical for this user or flagged as high-risk."),
    "login_attempts"          : ("Multiple failed login attempts",
                                 "Several failed logins before this transaction — possible credential stuffing."),
    "time_since_last_txn"     : ("Rapid back-to-back transaction",
                                 "Very little time since previous transaction — possible automated fraud."),
    "location"                : ("Unusual location",
                                 "Transaction from a location inconsistent with user history."),
    "balance"                 : ("Low balance anomaly",
                                 "Account balance unusually low relative to transaction amount."),
    "device"                  : ("Device type mismatch",
                                 "Device category doesn't match the user's typical access pattern."),
}

# ── In-memory state (replaces a database for now) ─────────────────────────────
_state = {
    "model"           : None,
    "encoders"        : None,
    "explainer"       : None,
    "pipeline_results": [],      # all processed transactions
    "review_queue"    : [],      # REVIEW-flagged transactions
    "review_decisions": {},      # txn_id → decision dict
    "pipeline_running": False,
    "pipeline_done"   : False,
    "stream_clients"  : [],      # active WebSocket connections
    "retraining_lock" : threading.Lock(),
    "report_cache"    : None,
}


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Transaction Monitoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Simplified for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────
class ReviewDecision(BaseModel):
    decision: str   # "ALLOW" | "BLOCK" | "SKIP"

class PipelineConfig(BaseModel):
    max_rows: Optional[int] = MAX_ROWS


# ── Helpers ────────────────────────────────────────────────────────────────────
def _load_artefacts():
    if _state["model"] is None:
        with open(MODEL_FILE, "rb") as f:
            _state["model"] = pickle.load(f)
        _state["encoders"]  = load_encoders(ENCODER_FILE)
        _state["explainer"] = shap.TreeExplainer(_state["model"])


def _make_decision(risk_score: float) -> tuple[str, bool]:
    """
    Returns (decision, verification_needed).
    Focuses on False Positive (FP) detection.
    """
    # 1. Automatic Allow
    if risk_score < ALLOW_THRESHOLD:
        return "ALLOW", False
    
    # 2. Review range
    if risk_score < BLOCK_THRESHOLD:
        # If score is low (0.3 - 0.5), the model is only 'slightly' suspicious.
        # This is a high-risk area for False Positives.
        needs_verify = risk_score < 0.50 
        return "REVIEW", needs_verify

    # 3. Block range
    # If score is between 0.7 and 0.82, the model 'doubts' its block decision.
    # We want a human to verify so we don't block a legit customer (FP).
    needs_verify = risk_score < 0.82
    return "BLOCK", needs_verify


def _save_feedback_to_csv(decision_data: dict):
    """Saves human-in-the-loop feedback to a consolidated CSV file."""
    import os
    import csv
    
    filepath = "consolidated_feedback.csv"
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "transaction_id", "user_id", "amount", "risk_score", 
            "model_decision", "human_decision", "is_fp_correction", "timestamp"
        ])
        if not file_exists:
            writer.writeheader()
        
        is_fp = (decision_data["model_decision"] in ("BLOCK", "REVIEW") and 
                 decision_data["human_decision"] == "ALLOW")
        
        writer.writerow({
            **decision_data,
            "is_fp_correction": 1 if is_fp else 0,
            "timestamp"       : time.strftime("%Y-%m-%d %H:%M:%S")
        })
    print(f"📝 Feedback appended to {filepath}")


def _trigger_retraining():
    """Background task to retrain the model with human feedback."""
    if _state["pipeline_running"]:
        print("⚠️ [Auto-Train] Pipeline is running. Skipping retraining.")
        return

    if not _state["retraining_lock"].acquire(blocking=False):
        print("⚠️ [Auto-Train] Retraining already in progress. Skipping.")
        return

    print("🔄 [Auto-Train] Starting automated model retraining...")
    try:
        train_model()
        # Reload the model and update explainer
        with open(MODEL_FILE, "rb") as f:
            _state["model"] = pickle.load(f)
        _state["explainer"] = shap.TreeExplainer(_state["model"])
        _state["report_cache"] = None # Invalidate cache
        print(f"✅ [Auto-Train] Model updated and reloaded successfully.")
    except Exception as e:
        print(f"❌ [Auto-Train] Retraining failed: {e}")
    finally:
        _state["retraining_lock"].release()


def _get_shap_explanation(feature_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    shap_values = _state["explainer"].shap_values(feature_df)
    sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    explanations = []
    for i, col in enumerate(feature_df.columns):
        val      = float(feature_df.iloc[0][col])
        shap_val = float(sv[i])
        info     = FEATURE_REASONS.get(col, (col.replace("_", " ").title(), ""))
        explanations.append({
            "feature"    : col,
            "label"      : info[0],
            "detail"     : info[1],
            "raw_value"  : val,
            "shap_value" : shap_val,
            "direction"  : "increase" if shap_val > 0 else "decrease",
        })

    explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return explanations[:top_n]


async def _broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    clients_count = len(_state["stream_clients"])
    if clients_count > 0:
        print(f"📡 Broadcasting {message.get('type')} to {clients_count} clients")
    
    dead = []
    for ws in _state["stream_clients"]:
        try:
            await ws.send_json(message)
        except Exception as e:
            print(f"❌ Failed to send to client: {e}")
            dead.append(ws)
    for ws in dead:
        if ws in _state["stream_clients"]:
            _state["stream_clients"].remove(ws)


def _run_pipeline_sync(max_rows: int, loop: asyncio.AbstractEventLoop):
    """
    Runs the full pipeline in a background thread.
    Pushes each transaction result to WebSocket clients via the event loop.
    """
    _load_artefacts()
    _state["pipeline_results"] = []
    _state["review_queue"]     = []
    _state["pipeline_running"] = True
    _state["pipeline_done"]    = False
    
    print(f"🚀 Pipeline started for {max_rows} rows")

    for txn_num, txn in enumerate(
            stream_transactions(delay_ms=0, max_rows=max_rows), start=1):

        features   = engineer_features(txn, encoders=_state["encoders"])
        feature_df = features_to_dataframe(features)
        risk_score = float(_state["model"].predict_proba(feature_df)[0][1])
        decision, needs_verify = _make_decision(risk_score)

        shap_data = []
        if decision in ("REVIEW", "BLOCK"):
            shap_data = _get_shap_explanation(feature_df)

        result = {
            "txn_num"       : txn_num,
            "transaction_id": txn.get("transaction_id", f"TXN{txn_num:07d}"),
            "user_id"       : txn.get("user_id"),
            "amount"        : float(txn.get("amount", 0)),
            "merchant"      : txn.get("merchant"),
            "device"        : txn.get("device"),
            "location"      : txn.get("location"),
            "timestamp"     : txn.get("timestamp"),
            "balance"       : float(txn.get("balance", 0)),
            "login_attempts": int(txn.get("login_attempts", 0)),
            "txn_velocity"  : int(txn.get("txn_velocity", 0)),
            "is_known_device": int(txn.get("is_known_device", 1)),
            "amount_to_balance_ratio": float(txn.get("amount_to_balance_ratio", 0)),
            "is_fraud"      : int(txn.get("is_fraud", 0)),
            "bucket"        : txn.get("bucket", "UNKNOWN"),
            "risk_score"    : round(risk_score, 4),
            "decision"      : decision,
            "needs_verify"  : needs_verify,
            "shap"          : shap_data,
        }

        _state["pipeline_results"].append(result)

        if decision == "REVIEW" or needs_verify:
            _state["review_queue"].append({**txn, **result})

        # Push to WebSocket clients (thread-safe via event loop)
        asyncio.run_coroutine_threadsafe(
            _broadcast({"type": "transaction", "data": result}), loop
        )

        # Faster pacing for better responsiveness
        time.sleep(0.05)

    _state["pipeline_running"] = False
    _state["pipeline_done"]    = True
    print("🏁 Pipeline complete")

    # Notify clients pipeline is complete
    asyncio.run_coroutine_threadsafe(
        _broadcast({"type": "pipeline_complete", "total": len(_state["pipeline_results"])}),
        loop
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    pending_count = len([
        txn for txn in _state["review_queue"]
        if txn.get("transaction_id") not in _state["review_decisions"]
    ])
    return {
        "status"          : "ok",
        "pipeline_running": _state["pipeline_running"],
        "pipeline_done"   : _state["pipeline_done"],
        "total_processed" : len(_state["pipeline_results"]),
        "review_queue"    : pending_count,
    }



@app.post("/api/run-pipeline")
async def run_pipeline(config: PipelineConfig):
    """Trigger the full pipeline. Streams results via /ws/stream."""
    if _state["pipeline_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running.")

    _load_artefacts()
    loop = asyncio.get_event_loop()

    thread = threading.Thread(
        target=_run_pipeline_sync,
        args=(config.max_rows, loop),
        daemon=True,
    )
    thread.start()

    return {"message": f"Pipeline started. Streaming {config.max_rows} transactions.", "status": "started"}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint — pushes real-time transaction decisions to the frontend."""
    await websocket.accept()
    _state["stream_clients"].append(websocket)
    print(f"🔌 New WebSocket connection. Total clients: {len(_state['stream_clients'])}")
    try:
        while True:
            # Keep connection alive; data is pushed from _broadcast()
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        if websocket in _state["stream_clients"]:
            _state["stream_clients"].remove(websocket)
    except Exception as e:
        print(f"🔌 WebSocket error: {e}")
        if websocket in _state["stream_clients"]:
            _state["stream_clients"].remove(websocket)


@app.get("/api/review-queue")
def get_review_queue():
    """Returns all REVIEW-flagged transactions with their SHAP explanations."""
    queue = []
    for txn in _state["review_queue"]:
        txn_id   = txn.get("transaction_id")
        existing = _state["review_decisions"].get(txn_id)

        # Ensure SHAP is computed for each review item
        shap_data = txn.get("shap", [])
        if not shap_data:
            features   = engineer_features(txn, encoders=_state["encoders"])
            feature_df = features_to_dataframe(features)
            shap_data  = _get_shap_explanation(feature_df)

        queue.append({
            "transaction_id"         : txn_id,
            "user_id"                : txn.get("user_id"),
            "amount"                 : float(txn.get("amount", 0)),
            "merchant"               : txn.get("merchant"),
            "device"                 : txn.get("device"),
            "location"               : txn.get("location"),
            "timestamp"              : txn.get("timestamp"),
            "balance"                : float(txn.get("balance", 0)),
            "login_attempts"         : int(txn.get("login_attempts", 0)),
            "txn_velocity"           : int(txn.get("txn_velocity", 0)),
            "is_known_device"        : int(txn.get("is_known_device", 1)),
            "amount_to_balance_ratio": float(txn.get("amount_to_balance_ratio", 0)),
            "is_fraud"               : int(txn.get("is_fraud", 0)),
            "bucket"                 : txn.get("bucket", "?"),
            "risk_score"             : float(txn.get("risk_score", 0)),
            "needs_verify"           : txn.get("needs_verify", False),
            "shap"                   : shap_data,
            "review_decision"        : existing["decision"] if existing else "PENDING",
        })

    return {"count": len(queue), "transactions": queue}


@app.post("/api/review/{txn_id}")
def submit_review(txn_id: str, body: ReviewDecision):
    """Submit a reviewer decision for a specific transaction."""
    if body.decision not in ("ALLOW", "BLOCK", "SKIP"):
        raise HTTPException(status_code=400, detail="Decision must be ALLOW, BLOCK, or SKIP.")

    # Find the transaction in the review queue
    txn = next((t for t in _state["review_queue"]
                if t.get("transaction_id") == txn_id), None)
    if not txn:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not in review queue.")

    _state["review_decisions"][txn_id] = {
        "transaction_id": txn_id,
        "decision"      : body.decision,
        "is_fraud"      : int(txn.get("is_fraud", 0)),
        "amount"        : float(txn.get("amount", 0)),
        "risk_score"    : float(txn.get("risk_score", 0)),
        "user_id"       : txn.get("user_id"),
        "bucket"        : txn.get("bucket"),
    }

    # Persist to unique CSV for retraining
    _save_feedback_to_csv({
        "transaction_id"  : txn_id,
        "user_id"         : txn.get("user_id"),
        "amount"          : float(txn.get("amount", 0)),
        "risk_score"      : float(txn.get("risk_score", 0)),
        "model_decision"  : txn.get("decision"),
        "human_decision"  : body.decision
    })

    # Trigger automated retraining in a background thread
    threading.Thread(target=_trigger_retraining, daemon=True).start()

    return {
        "message"       : f"Decision recorded: {body.decision}",
        "transaction_id": txn_id,
        "decision" : body.decision,
    }


@app.get("/api/report")
def get_report():
    """Full pipeline evaluation report — mirrors your terminal report."""
    # Return cached report if pipeline is not running and cache exists
    if not _state["pipeline_running"] and _state["report_cache"]:
        # Update reviewer stats in cache
        _state["report_cache"]["reviewer"]["total_reviewed"] = len(_state["review_decisions"])
        return _state["report_cache"]

    results = _state["pipeline_results"]
    if not results:
        empty_report = {
            "summary": {"total": 0, "allows": 0, "reviews": 0, "blocks": 0},
            "confusion_matrix": {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
            "metrics": {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "fpr": 0,
            },
            "fp_analysis": {
                "total": 0,
                "wrongly_blocked": 0,
                "wrongly_reviewed": 0,
                "correctly_allowed": 0,
                "reduction_rate": 0,
            },
            "bucket_breakdown": {
                "TP": {"total": 0, "block": 0, "review": 0, "allow": 0},
                "FP": {"total": 0, "block": 0, "review": 0, "allow": 0},
                "FN": {"total": 0, "block": 0, "review": 0, "allow": 0},
                "TN": {"total": 0, "block": 0, "review": 0, "allow": 0},
            },
            "risk_histogram": [{"bin": i/10, "count": 0} for i in range(11)],
            "reviewer": {
                "total_reviewed": 0,
            },
        }
        return empty_report

    df = pd.DataFrame(results)
    total   = len(df)
    allows  = int((df["decision"] == "ALLOW").sum())
    reviews = int((df["decision"] == "REVIEW").sum())
    blocks  = int((df["decision"] == "BLOCK").sum())

    df["pred_fraud"] = df["decision"].map({"ALLOW": 0, "REVIEW": 1, "BLOCK": 1})
    df["actual"]     = df["is_fraud"].fillna(0).astype(int)

    TP = int(((df["pred_fraud"] == 1) & (df["actual"] == 1)).sum())
    FP = int(((df["pred_fraud"] == 1) & (df["actual"] == 0)).sum())
    FN = int(((df["pred_fraud"] == 0) & (df["actual"] == 1)).sum())
    TN = int(((df["pred_fraud"] == 0) & (df["actual"] == 0)).sum())

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0

    # FP bucket analysis
    fp_rows              = df[df["bucket"].isin(["FP", "FP_ALLOW", "FP_REVIEW", "FP_BLOCK"])]
    fp_total             = len(fp_rows)
    fp_wrongly_blocked   = int((fp_rows["decision"] == "BLOCK").sum())
    fp_wrongly_reviewed  = int((fp_rows["decision"] == "REVIEW").sum())
    fp_correctly_allowed = int((fp_rows["decision"] == "ALLOW").sum())

    # Bucket breakdown for charts
    bucket_breakdown = {}
    for b in ["TP", "FP", "FN", "TN"]:
        # Handle sub-buckets if necessary
        if b == "FP":
            sub = df[df["bucket"].str.startswith("FP")]
        else:
            sub = df[df["bucket"] == b]
            
        bucket_breakdown[b] = {
            "total": len(sub),
            "block": int((sub["decision"] == "BLOCK").sum()),
            "review": int((sub["decision"] == "REVIEW").sum()),
            "allow": int((sub["decision"] == "ALLOW").sum()),
        }

    # Risk histogram
    counts, bins = np.histogram(df["risk_score"], bins=10, range=(0, 1))
    risk_histogram = [
        {"bin": float(bins[i]), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    report = {
        "summary": {
            "total"  : total,
            "allows" : allows,
            "reviews": reviews,
            "blocks" : blocks,
        },
        "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
        "metrics": {
            "precision": round(precision, 4),
            "recall"   : round(recall, 4),
            "f1"       : round(f1, 4),
            "fpr"      : round(fpr, 4),
        },
        "fp_analysis": {
            "total"            : fp_total,
            "wrongly_blocked"  : fp_wrongly_blocked,
            "wrongly_reviewed" : fp_wrongly_reviewed,
            "correctly_allowed": fp_correctly_allowed,
            "reduction_rate"   : round(fp_correctly_allowed / fp_total, 4) if fp_total > 0 else 0,
        },
        "bucket_breakdown": bucket_breakdown,
        "risk_histogram": risk_histogram,
        "reviewer": {
            "total_reviewed"  : len(_state["review_decisions"]),
        },
    }
    
    # Only cache if pipeline is done
    if _state["pipeline_done"]:
        _state["report_cache"] = report
        
    return report


@app.get("/api/reviewer/decisions")
def get_reviewer_decisions():
    """Returns the history of decisions made by human reviewers."""
    return {
        "count": len(_state["review_decisions"]),
        "decisions": list(_state["review_decisions"].values())
    }


@app.get("/api/transactions")
def get_transactions(
    limit: int = 100,
    offset: int = 0,
    decision: Optional[str] = None,
    bucket: Optional[str] = None,
):
    """Paginated transaction list with optional filters."""
    results = _state["pipeline_results"]

    if decision:
        results = [r for r in results if r["decision"] == decision.upper()]
    if bucket:
        results = [r for r in results if r["bucket"] == bucket.upper()]

    total   = len(results)
    paged   = results[offset: offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "transactions": paged}

from fastapi.responses import FileResponse
import csv
import os

@app.get("/api/feedback/download")
def download_feedback():
    """Aggregates all individual feedback CSVs into one and provides it as a download."""
    if not os.path.exists(FEEDBACK_DIR) or not os.listdir(FEEDBACK_DIR):
        raise HTTPException(status_code=404, detail="No feedback logs found.")
    
    consolidated_file = "consolidated_feedback.csv"
    first = True
    
    with open(consolidated_file, "w", newline="") as outfile:
        writer = None
        for filename in sorted(os.listdir(FEEDBACK_DIR)):
            if not filename.endswith(".csv"):
                continue
            filepath = os.path.join(FEEDBACK_DIR, filename)
            with open(filepath, "r") as infile:
                reader = csv.DictReader(infile)
                if first:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    first = False
                for row in reader:
                    writer.writerow(row)
    
    return FileResponse(
        path=consolidated_file, 
        filename=f"feedback_export_{int(time.time())}.csv",
        media_type="text/csv"
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
