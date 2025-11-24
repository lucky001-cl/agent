"""Tests for orchestrator."""

import unittest
from datetime import datetime

from fraud_detection.agents.orchestrator import FraudDetectionOrchestrator
from fraud_detection.models import FraudRiskLevel
from fraud_detection.stream import TransactionGenerator


class TestOrchestrator(unittest.TestCase):
    """Test cases for Fraud Detection Orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = FraudDetectionOrchestrator()
        
        # Train with some data
        training_data = TransactionGenerator.generate_mixed_batch(50, fraud_ratio=0.1)
        self.orchestrator.train_ml_agent(training_data)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator.ml_agent)
        self.assertIsNotNone(self.orchestrator.llm_agent)
        self.assertEqual(len(self.orchestrator.agents), 2)
    
    def test_analyze_normal_transaction(self):
        """Test analyzing a normal transaction."""
        txn = TransactionGenerator.generate_normal_transaction("TEST001")
        
        analysis = self.orchestrator.analyze_transaction(txn)
        
        self.assertEqual(analysis.transaction_id, "TEST001")
        self.assertIsInstance(analysis.is_fraudulent, bool)
        self.assertIsInstance(analysis.risk_level, FraudRiskLevel)
        self.assertGreaterEqual(analysis.confidence_score, 0.0)
        self.assertLessEqual(analysis.confidence_score, 1.0)
    
    def test_analyze_suspicious_transaction(self):
        """Test analyzing a suspicious transaction."""
        txn = TransactionGenerator.generate_suspicious_transaction("TEST002")
        
        analysis = self.orchestrator.analyze_transaction(txn)
        
        self.assertEqual(analysis.transaction_id, "TEST002")
        # Suspicious transactions should typically have higher risk
        self.assertIn(analysis.risk_level, [
            FraudRiskLevel.MEDIUM,
            FraudRiskLevel.HIGH,
            FraudRiskLevel.CRITICAL
        ])
    
    def test_risk_level_determination(self):
        """Test risk level determination."""
        orchestrator = FraudDetectionOrchestrator()
        
        self.assertEqual(
            orchestrator._determine_risk_level(0.9),
            FraudRiskLevel.CRITICAL
        )
        self.assertEqual(
            orchestrator._determine_risk_level(0.7),
            FraudRiskLevel.HIGH
        )
        self.assertEqual(
            orchestrator._determine_risk_level(0.5),
            FraudRiskLevel.MEDIUM
        )
        self.assertEqual(
            orchestrator._determine_risk_level(0.2),
            FraudRiskLevel.LOW
        )
    
    def test_recommended_action(self):
        """Test recommended action generation."""
        orchestrator = FraudDetectionOrchestrator()
        
        action = orchestrator._recommend_action(FraudRiskLevel.CRITICAL, True)
        self.assertIn("BLOCK", action)
        
        action = orchestrator._recommend_action(FraudRiskLevel.HIGH, True)
        self.assertIn("REVIEW", action)
        
        action = orchestrator._recommend_action(FraudRiskLevel.MEDIUM, False)
        self.assertIn("CHALLENGE", action)
        
        action = orchestrator._recommend_action(FraudRiskLevel.LOW, False)
        self.assertIn("APPROVE", action)
    
    def test_statistics(self):
        """Test statistics generation."""
        # Analyze some transactions
        for i in range(10):
            txn = TransactionGenerator.generate_normal_transaction(f"TEST{i:03d}")
            self.orchestrator.analyze_transaction(txn)
        
        stats = self.orchestrator.get_statistics()
        
        self.assertEqual(stats["total_analyzed"], 10)
        self.assertIn("fraudulent", stats)
        self.assertIn("fraud_rate", stats)
        self.assertIn("risk_distribution", stats)
        self.assertIn("average_confidence", stats)
    
    def test_analysis_history(self):
        """Test that analysis history is maintained."""
        initial_count = len(self.orchestrator.analysis_history)
        
        txn = TransactionGenerator.generate_normal_transaction("TEST001")
        self.orchestrator.analyze_transaction(txn)
        
        self.assertEqual(len(self.orchestrator.analysis_history), initial_count + 1)


if __name__ == "__main__":
    unittest.main()
