# src/reporting.py
import sqlite3
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "audit.db"

def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_id TEXT,
        user_id TEXT,
        risk_score INTEGER,
        ml_score REAL,
        anomaly_score REAL,
        reason TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_decision(tx_id: str, user_id: str, risk_score: int, ml_score: float, anomaly_score: float, reason: str):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("""
        INSERT INTO audit (tx_id, user_id, risk_score, ml_score, anomaly_score, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (tx_id, user_id, risk_score, ml_score, anomaly_score, str(reason), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def query_recent(limit: int = 20):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT tx_id, user_id, risk_score, ml_score, anomaly_score, reason, created_at FROM audit ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    res = []
    for r in rows:
        res.append({
            "tx_id": r[0],
            "user_id": r[1],
            "risk_score": r[2],
            "ml_score": r[3],
            "anomaly_score": r[4],
            "reason": r[5],
            "created_at": r[6]
        })
    return res
