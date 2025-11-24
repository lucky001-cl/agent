"""Tests for ML agent."""

import unittest
from datetime import datetime

from fraud_detection.agents.ml_agent import MLAnomalyDetectionAgent
from fraud_detection.models import Transaction
from fraud_detection.stream import TransactionGenerator


class TestMLAgent(unittest.TestCase):
    """Test cases for ML Anomaly Detection Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = MLAnomalyDetectionAgent()
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.name, "ML Anomaly Detector")
        self.assertFalse(self.agent.is_trained)
        self.assertEqual(self.agent.contamination, 0.1)
    
    def test_feature_extraction(self):
        """Test feature extraction from transaction."""
        txn = Transaction(
            transaction_id="TEST001",
            user_id="user_123",
            amount=100.50,
            merchant="Test Merchant",
            merchant_category="retail",
            timestamp=datetime(2024, 1, 15, 14, 30),
            location="New York, NY",
            device_id="device_1",
            ip_address="192.168.1.1",
            card_last_four="1234"
        )
        
        features = self.agent.extract_features(txn)
        
        self.assertEqual(features.shape, (1, 5))
        self.assertEqual(features[0, 0], 100.50)  # amount
        self.assertEqual(features[0, 1], 14)  # hour
        self.assertEqual(features[0, 2], 0)  # day of week (Monday)
    
    def test_training(self):
        """Test training the agent."""
        # Generate training data
        transactions = TransactionGenerator.generate_mixed_batch(50, fraud_ratio=0.1)
        
        self.agent.train(transactions)
        
        self.assertTrue(self.agent.is_trained)
        self.assertEqual(len(self.agent.transaction_history), 50)
    
    def test_analysis_with_bootstrap(self):
        """Test analysis with bootstrapped training."""
        txn = TransactionGenerator.generate_normal_transaction("TEST001")
        
        result = self.agent.analyze(txn)
        
        self.assertIn("is_anomaly", result)
        self.assertIn("anomaly_score", result)
        self.assertIn("risk_factors", result)
        self.assertIsInstance(result["is_anomaly"], bool)
        self.assertIsInstance(result["anomaly_score"], float)
        self.assertIsInstance(result["risk_factors"], list)
    
    def test_risk_factor_identification(self):
        """Test identification of risk factors."""
        # High amount transaction
        high_amount_txn = Transaction(
            transaction_id="TEST001",
            user_id="user_123",
            amount=10000.00,
            merchant="Test Merchant",
            merchant_category="retail",
            timestamp=datetime.now(),
            location="New York, NY",
            device_id="device_1",
            ip_address="192.168.1.1",
            card_last_four="1234"
        )
        
        risk_factors = self.agent._identify_risk_factors(high_amount_txn)
        
        self.assertTrue(any("High transaction amount" in factor for factor in risk_factors))
    
    def test_unusual_time_detection(self):
        """Test detection of unusual transaction times."""
        late_night_txn = Transaction(
            transaction_id="TEST001",
            user_id="user_123",
            amount=100.00,
            merchant="Test Merchant",
            merchant_category="retail",
            timestamp=datetime.now().replace(hour=3),
            location="New York, NY",
            device_id="device_1",
            ip_address="192.168.1.1",
            card_last_four="1234"
        )
        
        risk_factors = self.agent._identify_risk_factors(late_night_txn)
        
        self.assertTrue(any("Unusual transaction time" in factor for factor in risk_factors))


if __name__ == "__main__":
    unittest.main()
