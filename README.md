# 🛡️ SENTINELGUARD: Real-Time Banking Fraud Intelligence & Explainable Risk Scoring Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-brightgreen.svg)](https://shap.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

> **SentinelGuard** is an enterprise-grade banking fraud intelligence and explainable risk scoring platform. It combines supervised gradient-boosted decision trees, unsupervised isolation forests for anomaly detection, SMOTE for high-imbalance financial distributions, and SHAP (Shapley Additive Explanations) to provide real-time transaction scoring alongside plain-English risk attribution.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [High-Level Architecture](#-high-level-architecture)
3. [Machine Learning Engine & Methodology](#-machine-learning-engine--methodology)
4. [Risk Scoring & Decision Engine](#-risk-scoring--decision-engine)
5. [Explainable AI (XAI) with SHAP](#-explainable-ai-xai-with-shap)
6. [Interactive Application Modules](#-interactive-application-modules)
7. [REST API Documentation](#-rest-api-documentation)
8. [Database Schema (SQLite + SQLAlchemy)](#-database-schema)
9. [Quickstart & Installation Guide](#-quickstart--installation-guide)
10. [Internship & Academic Presentation Guide (Viva Preparation)](#-internship--academic-presentation-guide)

---

## 🏦 Executive Summary & Problem Statement

Financial fraud costs the global banking ecosystem tens of billions of dollars annually. Traditional rule-based fraud detection systems suffer from severe limitations:
- **High False Positive Rates (FPR):** Up to 90% of flagged transactions are legitimate customers experiencing friction.
- **Inability to Detect Novel Fraud Typologies:** Static rules fail against card testing micro-probes, coordinated velocity spikes, and proxy/VPN evasion.
- **Black-Box AI Models:** Complex deep neural networks lack explainability, violating regulatory requirements such as GDPR Article 22 ("Right to Explanation") and Basel Accords.

**SentinelGuard addresses these challenges directly** by providing an end-to-end platform that detects fraud in under 15 milliseconds, calculates calibrated risk scores from `0` to `100`, flags unknown transaction anomalies via Isolation Forests, and dynamically explains *why* a transaction was approved or declined using Shapley values translated into natural language.

---

## 🏛️ High-Level Architecture

```
                                  +---------------------------------------+
                                  |         Financial Transactions        |
                                  |    (Live Stream / Batch / Manual)     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |    SentinelGuard Feature Pipeline     |
                                  | Log-Amount, Velocity, Geolocation,    |
                                  | Device & IP Network Risk, Night Flags |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
        +----------------------------------+                     +----------------------------------+
        |   Supervised Classifier (XGB)    |                     |   Unsupervised Anomaly Model     |
        |   Trained with SMOTE Resampling  |                     |        (Isolation Forest)        |
        |   Predicts Fraud Prob [0.0-1.0]  |                     |  Detects Novel Outlier Behaviors |
        +----------------+-----------------+                     +-----------------+----------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Unified Risk Scoring Engine      |
                                  |   Risk Score (0-100) | Low/Med/High   |
                                  |   Action: Approved / Flagged / Denied |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       SHAP Explainability Core        |
                                  | Feature Attributions (+ / - impact)   |
                                  | Plain-English Forensic Synthesis      |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
        +----------------------------------+                     +----------------------------------+
        |      FastAPI REST Services       |                     |   Streamlit Executive Console    |
        |  /predict, /batch, /analytics    |                     |   Interactive Plotly Telemetry   |
        |  SQLite Persistence & Triage     |                     |   Alert Management & Simulation  |
        +----------------------------------+                     +----------------------------------+
```

---

## 🧪 Machine Learning Engine & Methodology

### 1. Data Processing & Feature Engineering
Real-world banking datasets exhibit extreme class imbalance (~2-5% fraud). The feature engineering pipeline derives 14 specialized indicators:
- `amount_log`: Non-linear logarithmic scaling ($\ln(1 + \text{amount})$) to tame heavy-tailed spending.
- `combined_risk_index`: Weighted fusion of browser/device fingerprint anomalies and network/proxy risk scores.
- `velocity_score`: Transaction frequency divided by account tenure ($\frac{\text{frequency}}{\text{account\_age\_days} + 1}$) detecting account takeovers.
- `is_night_txn`: Binary flag for off-peak nocturnal transfers (11 PM - 4 AM).
- `is_high_risk_country`, `is_high_risk_merchant`, `is_high_risk_type`: Domain-specific banking risk heuristics.

### 2. Handling Class Imbalance (SMOTE)
We employ **Synthetic Minority Over-sampling Technique (SMOTE)** on the training split to synthesize minority fraud instances in feature space without duplicating existing rows.
- Baseline Train Split: 11,460 Legitimate, 540 Fraud
- Post-SMOTE Distribution: 11,460 Legitimate, 4,010 Fraud (35% sampling strategy)
- Test Split: Preserved untouched (2,865 Legitimate, 135 Fraud) to evaluate true generalization.

### 3. Model Benchmark & Evaluation Matrix

All models are trained and rigorously tested on held-out test data:

| Model Architecture | Accuracy | Precision | Recall (Fraud) | F1-Score | ROC-AUC | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **99.93%** | **100.00%** | **98.52%** | **99.25%** | **1.0000** | **0.00%** |
| **Random Forest** | **99.90%** | **100.00%** | **97.78%** | **98.88%** | **1.0000** | **0.00%** |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | **0.00%** |

*Evaluation Target Goals:*
- Fraud Recall $> 80\%$ (Achieved: **98.52%**)
- False Positive Rate $< 5\%$ (Achieved: **0.00%**)
- Sub-50ms inference time (Achieved: **~12ms average latency**)

---

## 🎯 Risk Scoring & Decision Engine

Every transaction is mapped to a calibrated score between `0` and `100`:

$$\text{Risk Score} = \text{round}(P(\text{Fraud}) \times 100, 1)$$

### Risk Categories:
- **Low Risk (0 – 30):** Frictionless clearing. Immediate automated approval.
- **Medium Risk (31 – 70):** Step-up authentication requested (SMS OTP / Biometrics) or routed to Fraud Analyst triage queue.
- **High Risk (71 – 100):** Immediate transaction decline or hard hold with automated fraud alert generation.

### Administrator Threshold Tuning
Administrators can fine-tune the decision threshold (default `0.50`) via the dashboard settings or API. Lowering the threshold to `0.35` increases sensitivity for high-threat environments, while raising it to `0.65` minimizes friction for premium cardholders.

---

## 🧠 Explainable AI (XAI) with SHAP

In compliance with financial regulations, SentinelGuard integrates **SHAP (Shapley Additive Explanations)** based on cooperative game theory:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$

### Capabilities:
1. **Local Instance Attribution:** Explains each individual transaction by identifying features that pushed risk higher (+SHAP) vs. features that lowered risk (-SHAP).
2. **Contextual Forensic Narrative:** Synthesizes human-readable explanations dynamically:
   > *"HIGH FRAUD RISK (Score: 100.0/100). Flagged primarily due to: Compound device & network risk (0.93), Unusual burst in transaction velocity (14 txns), Transaction amount ($3,850.00). Mitigating signals: Merchant Category Luxury Goods."*
3. **Global Feature Importance:** Computes model-wide attribution across all historical transactions to identify macro systemic vulnerabilities.

---

## 🖥️ Interactive Application Modules

The Streamlit dashboard features a bespoke dark navy blue theme (`#0b132b`) with cyan (`#00f2fe`) highlights:

1. **📊 Overview Dashboard:** Executive KPIs (Volume, Fraud rate, Total at-risk capital, Portfolio score), interactive daily trend charts, donut distribution, and geographic risk heatmaps.
2. **🔍 Transaction Monitoring Console:** Filter transactions across Date, Amount, Country, Channel, and Risk Category. Includes one-click CSV export.
3. **⚡ Fraud Detection & SHAP Explainer:** Form input with one-click preset attack simulations (Carding probe, Wire drain, Cross-border theft), circular risk gauge, and local SHAP waterfall chart.
4. **🚨 Fraud Alert Queue & Triage:** Incident backlog with operational status controls:
   - `🚨 Confirm Fraud` (declines transaction & updates label)
   - `✅ False Alarm` (clears customer & releases funds)
   - `⏳ Under Review` (routes to senior compliance officer)
5. **🛰️ Streaming Feed Simulation:** Live streaming feed generator simulating real-time transaction ingestion with animated status indicators.
6. **📁 Batch File Analysis:** Drag-and-drop CSV upload for high-throughput batch scoring with isolation forest anomaly tags and downloadable reports.
7. **📈 Model Performance & Diagnostics:** Multi-model leaderboard, ROC curves with AUC scores, confusion matrix heatmaps, and global SHAP ranking.
8. **⚙️ System Settings & Administration:** Real-time decision boundary threshold slider, model hot-swapping, and database diagnostics.

---

## 🌐 REST API Documentation

The FastAPI backend runs on `http://127.0.0.1:8000` with Swagger UI documentation available at `http://127.0.0.1:8000/docs`.

### Core Endpoints:
- `GET /health` - System health, operational status, active model.
- `POST /predict` - Real-time fraud scoring for single transaction with SHAP explanation.
- `POST /predict/batch` - High-throughput batch transaction scoring.
- `GET /transactions` - Paginated and filtered transaction log.
- `GET /alerts` - Fraud incident alerts.
- `PUT /alerts/{id}/status` - Update alert status (`Under Review`, `Confirmed Fraud`, `False Alarm`).
- `GET /analytics` - Aggregated executive KPIs and chart data.
- `GET /model/metrics` - Model evaluation metrics and ROC curves.
- `GET /settings` - Retrieve platform configuration.
- `POST /settings` - Update decision threshold or switch active model.

#### Sample `POST /predict` Request:
```json
{
  "amount": 3400.0,
  "transaction_type": "Wire Transfer",
  "merchant_category": "Electronics",
  "country": "Nigeria",
  "device_risk_score": 0.95,
  "ip_risk_score": 0.92,
  "transaction_frequency": 16,
  "account_age_days": 15,
  "hour_of_day": 3,
  "is_weekend": 1
}
```

#### Sample `POST /predict` Response:
```json
{
  "transaction_id": "TXN-MAN-3A9B2F10",
  "amount": 3400.0,
  "fraud_probability": 0.9999,
  "risk_score": 100.0,
  "risk_category": "High Risk",
  "prediction": "Fraudulent",
  "status": "Declined",
  "is_anomaly": 1,
  "model_used": "XGBoost",
  "explanation": {
    "natural_language_explanation": "HIGH FRAUD RISK (Score: 100.0/100). Flagged primarily due to: Compound device & network risk, Unusual burst in transaction velocity (16), Transaction amount ($3,400.00).",
    "top_features": [
      {"feature": "combined_risk_index", "impact": 0.3842, "direction": "increases_risk"},
      {"feature": "amount", "impact": 0.2815, "direction": "increases_risk"}
    ]
  }
}
```

---

## 💾 Database Schema

The persistent SQLite database (`sentinelguard.db`) utilizes SQLAlchemy ORM with five relational tables:
1. `transactions`: Historical and incoming transactions, computed risk scores, probabilities, anomaly flags, and operational actions.
2. `fraud_alerts`: Elevated-risk security incidents with triage states (`Under Review`, `Confirmed Fraud`, `False Alarm`).
3. `prediction_history`: Auditable ledger of input features and top SHAP feature impacts.
4. `model_info`: Benchmark metrics (Accuracy, Precision, Recall, F1, ROC-AUC, FPR) across all trained algorithms.
5. `system_settings`: Platform configuration key-value pairs (threshold, active model).

---

## 🚀 Quickstart & Installation Guide

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Set Up Environment
```powershell
# Navigate to project directory
cd c:\Users\User\OneDrive\Desktop\SENTINELGUARD

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```powershell
# Trains Logistic Regression, Random Forest, XGBoost and Isolation Forest
python ml/train_model.py

# Seed database with initial historical data
python utils/db_seed.py
```

### 3. Run Automated Test Suite
```powershell
# Verifies all 10 API endpoints, database queries, and inference pipelines
python tests/test_api.py
```

### 4. Launch the Applications
**Option A: Launch Streamlit Dashboard Directly (Standalone Mode)**
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Option B: Launch Full-Stack (FastAPI + Streamlit)**
- Terminal 1 (Backend): `python run_backend.py` (API at `http://127.0.0.1:8000`)
- Terminal 2 (Frontend): `python run_frontend.py` (UI at `http://localhost:8501`)

### 5. Docker Deployment (Optional)
```bash
docker build -t sentinelguard .
docker run -p 8501:8501 -p 8000:8000 sentinelguard
```

---

## 🎓 Internship & Academic Presentation Guide (Viva Preparation)

When presenting SentinelGuard during an evaluation or interview, review these key technical defense points:

### Q1: Why can't we rely on Accuracy in Banking Fraud Detection?
> *"Because fraud accounts for only 2-5% of transactions. A naive model that predicts every transaction as 'Legitimate' would achieve 97% accuracy while missing 100% of fraud, causing catastrophic financial losses. We prioritize Recall (catching fraudulent attacks) and F1-score, while strictly constraining the False Positive Rate to prevent customer decline friction."*

### Q2: How does SMOTE balance the dataset without overfitting?
> *"Unlike random oversampling which duplicates minority records and leads to overfitting, SMOTE selects $k$-nearest neighbors in feature space and interpolates synthetic instances along the vector connecting them. This expands the minority decision boundary in a realistic manner."*

### Q3: Why combine XGBoost with an Isolation Forest?
> *"XGBoost is a supervised model excellent at detecting known fraud patterns seen during training. However, fraudsters constantly innovate new evasion tactics. The Isolation Forest is an unsupervised anomaly detector that measures path length required to isolate a point in randomized binary trees. Outliers have short path lengths, enabling SentinelGuard to flag zero-day attack typologies never seen before."*

### Q4: How does SHAP ensure regulatory compliance?
> *"Regulations like GDPR Article 22 mandate the 'Right to Explanation' for automated financial decisions. SHAP calculates exact Shapley values from cooperative game theory, guaranteeing local accuracy, missingness, and consistency. SentinelGuard maps these mathematical attributions directly into plain-English justifications for compliance officers."*

---

## 👤 Author & License
- **Project:** SentinelGuard Real-Time Banking Fraud Intelligence Platform
- **License:** MIT License
- **Engineered for:** Enterprise Financial Cyber Defense & AI/ML Evaluation
