# src/api/main.py
import os
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path

from src.features import build_features
from src.models.supervised import SupervisedModel
from src.models.unsupervised import UnsupervisedModel
from src.llm_agent import llm_reason
from src.reporting import init_db, log_decision, query_recent

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Try to load models if available
SUP_MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
UNSUP_MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"

sup_model = SupervisedModel.load(str(SUP_MODEL_PATH)) if SUP_MODEL_PATH.exists() else None
unsup_model = UnsupervisedModel.load(str(UNSUP_MODEL_PATH)) if UNSUP_MODEL_PATH.exists() else None

app = FastAPI(title="AI Fraud Detection Agent - API")

class Txn(BaseModel):
    tx_id: str
    user_id: str
    amount: float
    currency: Optional[str] = "USD"
    merchant: Optional[str] = ""
    ip: Optional[str] = ""
    device_id: Optional[str] = ""
    timestamp: Optional[str] = ""
    context: Optional[Dict[str, Any]] = {}

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def root():
    return {"service": "AI Fraud Detection Agent", "status": "ok"}

@app.post("/score")
def score(txn: Txn):
    try:
        txn_dict = txn.dict()
        # 1. feature engineering
        features = build_features(txn_dict)

        # 2. unsupervised anomaly
        anom_score = None
        if unsup_model is not None:
            anom_score = float(unsup_model.score_dict(features))

        # 3. supervised prediction
        ml_prob = None
        if sup_model is not None:
            ml_prob = float(sup_model.predict_proba_dict(features))

        # 4. LLM reasoning (structured)
        llm_out = llm_reason(txn_dict, features, anom_score, ml_prob)

        # Compose final risk score (example weighted aggregation)
        risk_score = None
        if ml_prob is not None and llm_out.get("score") is not None:
            risk_score = int(round(0.6 * ml_prob * 100 + 0.4 * llm_out["score"]))
        elif ml_prob is not None:
            risk_score = int(round(ml_prob * 100))
        elif llm_out.get("score") is not None:
            risk_score = int(round(llm_out["score"]))
        else:
            risk_score = None

        # Persist decision in audit DB
        log_decision(
            tx_id=txn.tx_id,
            user_id=txn.user_id,
            risk_score=risk_score,
            ml_score=ml_prob,
            anomaly_score=anom_score,
            reason=llm_out.get("reason", "")
        )

        return {
            "tx_id": txn.tx_id,
            "ml_prob": ml_prob,
            "anomaly_score": anom_score,
            "risk_score": risk_score,
            "explain": llm_out
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/unsupervised")
def train_unsupervised(path: str = "data/train_unsup.csv"):
    """
    Trains an IsolationForest on numeric columns found in CSV and saves to models/isolation_forest.pkl
    """
    try:
        model = UnsupervisedModel.train_from_csv(path, save_path=str(UNSUP_MODEL_PATH))
        global unsup_model
        unsup_model = model
        return {"status": "ok", "model_path": str(UNSUP_MODEL_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/supervised")
def train_supervised(path: str = "data/train_sup.csv", label_col: str = "is_fraud"):
    """
    Trains XGBoost on CSV with label_col and saves to models/xgboost_model.pkl
    """
    try:
        model = SupervisedModel.train_from_csv(path, label_col=label_col, save_path=str(SUP_MODEL_PATH))
        global sup_model
        sup_model = model
        return {"status": "ok", "model_path": str(SUP_MODEL_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audit/recent")
def audit_recent(limit: int = 20):
    try:
        rows = query_recent(limit=limit)
        return {"recent": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
