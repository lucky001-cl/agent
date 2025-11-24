# src/llm_agent.py
import os
import json
from typing import Dict, Any

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change as available

def llm_reason(txn: Dict[str, Any], features: Dict[str, Any], anomaly_score: float, ml_prob: float) -> Dict[str, Any]:
    """
    Returns a structured dictionary:
      {"score": int(0-100), "reason": str, "action": "allow|review|block"}
    If OPENAI_API_KEY is present, attempt to call the OpenAI ChatCompletion endpoint.
    Otherwise, return a deterministic heuristic.
    """
    # Build compact payload
    payload = {
        "transaction": {k: txn.get(k) for k in ["tx_id", "user_id", "amount", "merchant", "ip", "device_id", "timestamp"]},
        "features": features,
        "anomaly_score": anomaly_score,
        "ml_prob": ml_prob
    }

    # If API key is present, attempt to call OpenAI (best-effort; user must install openai & set key)
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            prompt = build_prompt(payload)
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": "You are a concise fraud analyst. Return EXACT JSON."},
                          {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            text = resp.choices[0].message["content"].strip()
            try:
                parsed = json.loads(text)
                # Validate fields
                if "score" in parsed and "reason" in parsed and "action" in parsed:
                    return parsed
            except Exception:
                # fall through to heuristic
                pass
        except Exception:
            # any failure -> fallback
            pass

    # deterministic heuristic fallback
    return heuristic_reason(payload, anomaly_score, ml_prob)

def build_prompt(payload: dict) -> str:
    p = {
        "task": "Given the JSON payload, produce EXACT JSON: {\"score\":<0-100>,\"reason\":\"...\",\"action\":\"allow|review|block\"}",
        "payload": payload
    }
    return json.dumps(p, default=str)

def heuristic_reason(payload: dict, anomaly_score: float, ml_prob: float) -> Dict[str, Any]:
    """
    Simple deterministic scoring fallback:
    - base score = ml_prob * 100 if available else 0
    - bump by anomaly_score*100
    - bump for amount thresholds
    """
    score = 0
    parts = []

    if ml_prob is not None:
        score = max(score, int(round(ml_prob * 100)))
        parts.append(f"ml_prob={ml_prob:.2f}")

    if anomaly_score is not None:
        score = max(score, int(round(anomaly_score * 100)))
        parts.append(f"anomaly={anomaly_score:.2f}")

    txn = payload.get("transaction", {})
    amt = float(txn.get("amount", 0.0) or 0.0)
    if amt >= 10000:
        score = max(score, 90)
        parts.append("high_amount")
    elif amt >= 2000:
        score = max(score, max(score, 70))
        parts.append("elevated_amount")

    # device novelty
    if payload["features"].get("is_new_device", 0.0) >= 1.0:
        score = min(100, score + 10)
        parts.append("new_device")

    # short rule - if both ml_prob and anomaly high
    if (ml_prob or 0) >= 0.8 and (anomaly_score or 0) >= 0.6:
        score = min(100, max(score, 95))
        parts.append("ml+anom_high")

    # action mapping
    if score >= 80:
        action = "block"
    elif score >= 50:
        action = "review"
    else:
        action = "allow"

    reason = "; ".join(parts) if parts else "no strong signals"
    return {"score": int(score), "reason": reason, "action": action}
