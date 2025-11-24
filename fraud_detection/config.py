"""
Configuration settings for the AI Fraud Detection Agent.
"""

# ML Model Configuration
ML_CONFIG = {
    "contamination": 0.1,  # Expected proportion of outliers (0.0 - 0.5)
    "n_estimators": 100,   # Number of trees in Isolation Forest
    "random_state": 42,    # Random seed for reproducibility
}

# Risk Thresholds
RISK_THRESHOLDS = {
    "high_amount": 5000.0,        # Amount above which is considered high risk
    "very_high_amount": 10000.0,  # Amount above which is very high risk
    "unusual_hours_start": 22,     # Start of unusual hours (10 PM)
    "unusual_hours_end": 6,        # End of unusual hours (6 AM)
    "high_velocity_count": 5,      # Number of transactions in window to be high velocity
    "velocity_window_hours": 1,    # Time window for velocity checking
}

# High-Risk Categories
HIGH_RISK_CATEGORIES = [
    "gambling",
    "cryptocurrency",
    "wire_transfer",
    "money_order",
    "atm_withdrawal",
    "cash_advance"
]

# Fraud Detection Scoring
SCORING_CONFIG = {
    "ml_weight": 0.6,              # Weight for ML model score
    "llm_weight": 0.4,             # Weight for LLM analysis
    "fraud_threshold": 0.6,        # Threshold for marking as fraudulent
    "critical_threshold": 0.8,     # Threshold for critical risk
    "high_threshold": 0.6,         # Threshold for high risk
    "medium_threshold": 0.4,       # Threshold for medium risk
}

# LLM Configuration
LLM_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.3,
    "max_tokens": 300,
    "timeout": 30,  # seconds
}

# Stream Processing Configuration
STREAM_CONFIG = {
    "queue_timeout": 1,           # Timeout for queue operations (seconds)
    "processing_sleep": 0.1,      # Sleep time when queue is empty (seconds)
    "batch_size": 100,            # Default batch size for processing
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "fraud_detection.log",
}
