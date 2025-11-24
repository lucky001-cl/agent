"""Tests for fraud detection models."""

import unittest
from datetime import datetime

from fraud_detection.models import (
    Transaction,
    FraudAnalysis,
    TransactionStatus,
    FraudRiskLevel
)


class TestTransaction(unittest.TestCase):
    """Test cases for Transaction model."""
    
    def test_transaction_creation(self):
        """Test creating a transaction."""
        txn = Transaction(
            transaction_id="TEST001",
            user_id="user_123",
            amount=100.50,
            merchant="Test Merchant",
            merchant_category="retail",
            timestamp=datetime.now(),
            location="New York, NY",
            device_id="device_1",
            ip_address="192.168.1.1",
            card_last_four="1234"
        )
        
        self.assertEqual(txn.transaction_id, "TEST001")
        self.assertEqual(txn.amount, 100.50)
        self.assertEqual(txn.status, TransactionStatus.PENDING)
        self.assertEqual(txn.currency, "USD")
    
    def test_transaction_to_dict(self):
        """Test converting transaction to dictionary."""
        timestamp = datetime.now()
        txn = Transaction(
            transaction_id="TEST001",
            user_id="user_123",
            amount=100.50,
            merchant="Test Merchant",
            merchant_category="retail",
            timestamp=timestamp,
            location="New York, NY",
            device_id="device_1",
            ip_address="192.168.1.1",
            card_last_four="1234"
        )
        
        txn_dict = txn.to_dict()
        
        self.assertEqual(txn_dict["transaction_id"], "TEST001")
        self.assertEqual(txn_dict["amount"], 100.50)
        self.assertEqual(txn_dict["status"], "pending")
        self.assertIsInstance(txn_dict["timestamp"], str)


class TestFraudAnalysis(unittest.TestCase):
    """Test cases for FraudAnalysis model."""
    
    def test_fraud_analysis_creation(self):
        """Test creating a fraud analysis."""
        analysis = FraudAnalysis(
            transaction_id="TEST001",
            is_fraudulent=True,
            risk_level=FraudRiskLevel.HIGH,
            confidence_score=0.85,
            ml_score=0.9,
            llm_analysis="High risk transaction",
            risk_factors=["high_amount", "unusual_time"],
            recommended_action="REVIEW"
        )
        
        self.assertEqual(analysis.transaction_id, "TEST001")
        self.assertTrue(analysis.is_fraudulent)
        self.assertEqual(analysis.risk_level, FraudRiskLevel.HIGH)
        self.assertEqual(analysis.confidence_score, 0.85)
    
    def test_fraud_analysis_to_dict(self):
        """Test converting fraud analysis to dictionary."""
        analysis = FraudAnalysis(
            transaction_id="TEST001",
            is_fraudulent=False,
            risk_level=FraudRiskLevel.LOW,
            confidence_score=0.25
        )
        
        analysis_dict = analysis.to_dict()
        
        self.assertEqual(analysis_dict["transaction_id"], "TEST001")
        self.assertFalse(analysis_dict["is_fraudulent"])
        self.assertEqual(analysis_dict["risk_level"], "low")
        self.assertIsInstance(analysis_dict["timestamp"], str)


if __name__ == "__main__":
    unittest.main()
