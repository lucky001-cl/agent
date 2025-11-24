"""
Transaction data models and schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class TransactionStatus(Enum):
    """Status of a transaction."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class FraudRiskLevel(Enum):
    """Risk level classification for transactions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Transaction:
    """Represents a financial transaction."""
    transaction_id: str
    user_id: str
    amount: float
    merchant: str
    merchant_category: str
    timestamp: datetime
    location: str
    device_id: str
    ip_address: str
    card_last_four: str
    currency: str = "USD"
    status: TransactionStatus = TransactionStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "amount": self.amount,
            "merchant": self.merchant,
            "merchant_category": self.merchant_category,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "device_id": self.device_id,
            "ip_address": self.ip_address,
            "card_last_four": self.card_last_four,
            "currency": self.currency,
            "status": self.status.value,
            "metadata": self.metadata
        }


@dataclass
class FraudAnalysis:
    """Results of fraud detection analysis."""
    transaction_id: str
    is_fraudulent: bool
    risk_level: FraudRiskLevel
    confidence_score: float  # 0.0 to 1.0
    ml_score: Optional[float] = None
    llm_analysis: Optional[str] = None
    risk_factors: list = field(default_factory=list)
    recommended_action: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "is_fraudulent": self.is_fraudulent,
            "risk_level": self.risk_level.value,
            "confidence_score": self.confidence_score,
            "ml_score": self.ml_score,
            "llm_analysis": self.llm_analysis,
            "risk_factors": self.risk_factors,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat()
        }
