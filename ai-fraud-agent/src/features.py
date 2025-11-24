# src/features.py
from datetime import datetime
import math

def build_features(txn: dict) -> dict:
    """
    Convert transaction dict into numeric features consumed by models.
    Keep features simple and deterministic for reproducibility.
    """
    features = {}
    # numeric
    features['amount'] = float(txn.get('amount', 0.0))

    # timestamp -> hour, weekday
    ts = txn.get('timestamp')
    if ts:
        try:
            # Accept ISO8601-like, best-effort
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            features['hour'] = float(dt.hour)
            features['dow'] = float(dt.weekday())
        except Exception:
            features['hour'] = 0.0
            features['dow'] = 0.0
    else:
        features['hour'] = 0.0
        features['dow'] = 0.0

    # simple hashing for categorical-like fields (device_id, ip, merchant)
    features['device_hash'] = simple_hash(txn.get('device_id', ''))
    features['ip_hash'] = simple_hash(txn.get('ip', ''))
    features['merchant_hash'] = simple_hash(txn.get('merchant', ''))

    # context (past activity) - user-supplied or default to 0
    ctx = txn.get('context', {}) or {}
    features['past_24h_txn_count'] = float(ctx.get('past_24h_txn_count', 0.0))
    features['past_24h_amount_mean'] = float(ctx.get('past_24h_amount_mean', 0.0))
    features['is_new_device'] = 1.0 if ctx.get('is_new_device', False) else 0.0

    # derived features
    features['amount_to_mean_ratio'] = 0.0
    if features['past_24h_amount_mean'] > 0:
        features['amount_to_mean_ratio'] = features['amount'] / (features['past_24h_amount_mean'] + 1e-9)

    return features

def simple_hash(s: str, mod: int = 1000) -> float:
    """
    Deterministic simple hash transform to numeric range.
    Not cryptographic. Good enough for feature hashing in demo.
    """
    if not s:
        return 0.0
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % mod
    return float(h)
