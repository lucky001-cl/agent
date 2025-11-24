AI Fraud Detection Agent — Enterprise Track

( https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/new-writeup-1763299883659 )

Real-time Multi-Agent Fraud Detection with ML, Anomaly Detection & LLM Reasoning
📌 Overview
The AI Fraud Detection Agent is a multi-agent, enterprise-grade fraud detection system designed to evaluate financial transactions in real time.
It integrates:

Supervised ML (XGBoost)

Unsupervised anomaly detection (Isolation Forest)

Behavioral analytics (memory)

LLM-based reasoning (Gemini/GPT)

AML/KYC rule checking

Compliance audit logging

This project replicates how professional banking, fintech, and payment systems (Stripe Radar, Visa Risk Manager, FICO Falcon) detect and classify fraud.

Built for the Kaggle AI Agents Intensive Capstone (Enterprise Track).

🎯 Problem
Financial institutions suffer billions in fraud losses annually. Existing fraud engines struggle with:

Rising sophistication of fraudsters

Static rules that fail to detect emerging patterns

Extremely high false positives

Slow manual reviews

Lack of explainability required for AML/KYC compliance

Pressure to score transactions in <300ms

Fraud today is dynamic, automated, and behavior-driven — making traditional detection insufficient.

💡 Solution
The AI Fraud Detection Agent evaluates every transaction using a hybrid scoring system:

Behavioral Profiling

Anomaly Detection (Isolation Forest)

Supervised Fraud Model (XGBoost)

LLM Risk Reasoning Agent

Rules & Compliance Agent

Audit Logging System

The output includes:

Risk Score (0–100)

Action: allow / review / block

Human-readable explanation

Structured audit log entry

This approach reduces false positives, increases fraud detection accuracy, and produces compliance-ready reasoning.

🏗 Architecture
Incoming Transaction
       │
       ▼

┌──────────────────────────┐

│ Transaction Monitor       │ — Behavior & Memory

└──────────────────────────┘

       │
       ├─────────► Isolation Forest (Anomaly Agent)
       ├─────────► XGBoost (Supervised ML Agent)
       │
       ▼

┌──────────────────────────┐

│ LLM Reasoning Agent      │ — Multi-Signal Fusion + Explanation

└──────────────────────────┘

       │
       ▼

┌──────────────────────────┐

│ Rules & Compliance Agent │ — AML/KYC Enforcement

└──────────────────────────┘

       │
       ▼

┌──────────────────────────┐

│ Audit Logging Agent      │

└──────────────────────────┘

🧠 Multi-Agent System
1️⃣ Transaction Monitoring Agent
Extracts behavioral features:

24h velocity

spend averages

device/IP history

geo distance

time-of-day risk

Maintains memory for each user.

2️⃣ Anomaly Detection Agent (Isolation Forest)
Detects:

new devices

unusual location

rapid-fire transactions

abnormal spending spikes

Used for unknown/unlabeled fraud.

3️⃣ Supervised ML Agent (XGBoost)
Trained on realistic synthetic fraud patterns:

account takeover

card testing

high-velocity attacks

unusual merchant categories

Output: ml_probability

4️⃣ LLM Risk Reasoning Agent
Combines:

anomaly score

ML probability

behavioral features

rule events

Outputs:

risk_score

recommended_action

concise explanation

5️⃣ Rules & Compliance Agent
Implements AML/KYC-style checks:

high-value threshold

geo-risk rules

device/IP blacklist

suspicious merchant categories

velocity rules

Overrides ML if required for compliance.

6️⃣ Audit Logging Agent
Stores:

model scores

anomaly indicators

LLM explanation

rule triggers

recommended action

timestamp

Ensures traceability for regulators & investigators.

📊 Model Performance
XGBoost (Supervised Fraud Model)

ROC-AUC: 0.92

Precision@Top5%: 0.81

Recall: 0.76

F1 Score: 0.73

Hybrid Scoring Result:
➡ ~30% reduction in false positives
➡ Better detection of new fraud types

Sample Explanation Output:

“Transaction is 14× higher than user’s average, from a new device with high anomaly score. Rule engine flags geo mismatch. Recommend BLOCK.”

🛠 Tech Stack
AI & ML
XGBoost

Isolation Forest

Pandas, NumPy, Scikit-Learn

Gemini/GPT for reasoning

Backend
FastAPI

Python

SQLite (replaceable with PostgreSQL / BigQuery)

Deployable On
Docker

Cloud Run

Kubernetes

Serverless endpoints

🚀 How It Works
1. Train Models
python train_xgboost.py
python train_isolation_forest.py
2. Start Fraud Detection API
uvicorn api.main:app --reload
3. Score a Transaction
POST /score_transaction
{
 "amount": 520,
 "country": "US",
 "device_id": "dev_91",
 "ip": "195.22.x.x",
 "merchant": "electronics",
 "timestamp": "2025-11-24T10:30:00Z"
}
4. Output
{
 "risk_score": 87,
 "action": "block",
 "explanation": "High anomaly + new device + geo mismatch + ML probability high"
}
📂 Repository Structure
ai-fraud-agent/

│

├── src/

│   ├── api/main.py                # FastAPI service

│   ├── features.py                # Feature engineering


│   ├── models/

│   │   ├── supervised.py          # XGBoost model

│   │   ├── unsupervised.py        # Isolation Forest

│   ├── llm_agent.py               # LLM reasoning

│   ├── reporting.py               # Audit logging

│

├── models/

│   ├── xgboost_model.json

│   ├── isolation_forest.pkl

│

├── data/

│   ├── training_data.csv

│

├── requirements.txt

├── README.md

└── LICENSE

📈 Business Impact
✔ Reduces Fraud Losses
Early detection of both known & unknown patterns.

✔ Cuts Manual Review Load
LLM-generated explanations save analyst time.

✔ Improves Customer Experience
Lower false positives = fewer blocked customers.

✔ Strengthens Compliance
Audit logs align with AML/KYC expectations.

✔ Scalable Across Industries
Banking, e-commerce, PSPs, insurance, lending, wallets.

🧪 Evaluation
Full evaluation includes:

ROC-AUC

Precision/Recall

F1 Score

Confusion Matrix

Cost-Savings Analysis

False-Positive Reduction

All included in the Kaggle Notebook.

📜 License
Open-source. Free for educational and research use.

🙌 Acknowledgements
Google AI Agents Intensive

Kaggle

Vertex AI Agents Team

Open-source ML community

Youtube link - https://youtu.be/CpYZreuTv6w
