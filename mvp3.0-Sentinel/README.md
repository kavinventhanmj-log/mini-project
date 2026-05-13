# Sentinel: Transaction Monitoring System (MVP 3.0)

A professional, full-stack fraud detection pipeline featuring a FastAPI backend, a LightGBM machine learning model, and a Next.js real-time dashboard with SHAP-based AI explanations.

---

## 🚀 Quick Start Guide

To run the full system manually, follow these steps in two separate terminal windows.

### 1. Backend Setup (FastAPI)
**Terminal 1:**
```bash
# Navigate to the project root
cd sentinel-txn-monitor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api:app --reload --port 8000
```
*   **API URL:** `http://localhost:8000`
*   **API Docs (Swagger):** `http://localhost:8000/docs`

---

### 2. Frontend Setup (Next.js)
**Terminal 2:**
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies (only required the first time)
npm install

# Start the Next.js development server
npm run dev
```
*   **Dashboard URL:** `http://localhost:3000`

---

## 🛠 Features

*   **Live Monitor:** Real-time transaction streaming via WebSockets.
*   **AI Explanations:** SHAP-based horizontal bar charts showing exactly *why* a transaction was flagged.
*   **Human-in-the-Loop:** A Review Queue for manual overrides (Allow/Block/Skip).
*   **Performance Metrics:** Real-time tracking of Precision, Recall, and False Positive reduction.
*   **Industry Standard Tech:** FastAPI, Next.js 14, Tailwind CSS, Recharts, and Zustand.

## 📁 Project Structure

*   `api.py`: FastAPI backend and WebSocket management.
*   `/frontend`: Next.js application source code.
*   `model.pkl`: Pre-trained LightGBM fraud detection model.
*   `streamer.py`: Transaction simulation logic.
*   `feature_engine.py`: ML preprocessing and encoding.

---

## 💡 How to Use
1.  Ensure both servers are running.
2.  Open `http://localhost:3000`.
3.  Click **"Start Pipeline"** on the dashboard.
4.  Watch transactions stream in real-time.
5.  Navigate to **"Review Queue"** to handle items flagged for manual review.
