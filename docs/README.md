**Mini-project: the journey till mvp_1**

# 🚀 1. Problem Identification (Where it all started)

* In fintech systems, fraud detection models **flag transactions**.
* But a big issue: **False Positives (FP)** → legitimate users get blocked ❌
* This affects user trust and banking experience.

👉 Your core realization:

> “The problem is NOT fraud detection… it’s lack of explainability + false positive handling”

📌 From your doc:
False positives disrupt users and current systems lack transparency 

---

# 💡 2. Idea Formation (Your Project Core)

You proposed:

> Build an **Explainable ML-based filtering layer**

Key concept:

* NOT replacing fraud detection ❌
* ADDING a **secondary intelligent layer** ✅

📌 From product overview:
System acts as a **post-transaction analytical module** 

---

# 🧠 3. Concept Finalization (SRS Phase)

You formally defined:

### 🔹 System Type:

* Analytical Decision Support Tool
* Focus → **False Positive Filtering**

### 🔹 Core Technologies:

* ML Model → LightGBM
* Explainability → SHAP
* UI → Streamlit

📌 From SRS:
System provides explainability + analyst interaction 

---

# ⚙️ 4. Development Approach (How you built it)

You followed an **iterative ML workflow**:

### Step-by-step:

1. **EDA (Exploratory Data Analysis)**

   * Understand class imbalance (fraud vs legit)

2. **Model Benchmarking**

   * Compared XGBoost vs LightGBM
   * Selected **LightGBM (better stability)**

3. **Threshold Optimization**

   * Instead of fixed 0.5 → dynamic tuning

4. **Explainability Validation**

   * Using SHAP to verify model decisions

📌 From SRS lifecycle:
EDA → Benchmark → Prototype → Validate 

---

# 🏗️ 5. System Architecture Design

You designed a **clean 3-tier system**:

### 🔹 1. Presentation Layer (UI)

* Streamlit dashboard
* Upload dataset, view results

### 🔹 2. Logic Layer (Core Engine)

* Prediction
* Threshold filtering
* SHAP explainability

### 🔹 3. Data Layer

* model.pkl
* feedback logs (CSV + SQLite)

📌 From design doc:
Clear separation of UI, ML logic, and storage 

---

# 🔄 6. Core Workflow Pipeline (System Flow)

Your system works like this:

1. Upload dataset 📂
2. Validate schema ✔️
3. Run ML model (LightGBM) 🤖
4. Apply threshold 🎯
5. Generate predictions (Fraud / Legit)
6. Compute metrics (Accuracy, F1, etc.)
7. Generate SHAP explanations 📊
8. Analyst reviews + gives feedback 🧑‍💻
9. Store feedback for future use 💾

📌 This full pipeline is defined in your DFD 

---

# 📊 7. Explainability Integration (Your Key Innovation)

This is your **main highlight** 🔥

### 🔹 Global Explanation:

* Feature importance across dataset

### 🔹 Local Explanation:

* Why THIS transaction was flagged

📌 From SRS:
SHAP removes black-box behavior 

---

# 🧑‍💻 8. Analyst-in-the-Loop System

You added **human interaction layer**:

* Analyst reviews predictions
* Marks:

  * ✅ Correct
  * ❌ Wrong (False Positive)
* Feedback stored for future improvement

📌 From design:
Feedback stored in SQLite + CSV logs 

---

# 🖥️ 9. MVP Implementation (What you built now)

Your MVP includes:

### ✅ Functional Features:

* Dataset upload
* Model prediction
* Threshold slider
* Confusion matrix
* SHAP plots (global + local)
* Feedback module
* Export results

### ✅ Tech Stack:

* Python + LightGBM
* SHAP
* Streamlit UI

### ✅ Performance Goal:

* ~50k rows processed < 10 sec 

---

# ⚠️ 10. MVP Limitation (Important for viva)

Right now your system is:

* ❌ Not real-time
* ❌ Not primary fraud system
* ❌ Works on offline dataset

📌 From product overview:
Prototype runs in **offline simulation mode** 

---

# 🔥 11. Transition Thought:

Now THIS is where your thinking evolved:

👉 From:

* Secondary analysis tool

👉 To:

* **Primary intelligent filtering system**
* Real-time simulation
* Continuous learning (feedback loop)

This is exactly what you discussed recently 💯

---



