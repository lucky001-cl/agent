"""
Quick Start Guide for AI Fraud Detection Agent

This script demonstrates basic usage of the fraud detection system.
"""

from fraud_detection.agents import FraudDetectionOrchestrator
from fraud_detection.models import Transaction
from datetime import datetime

# Initialize the fraud detection system
print("Initializing AI Fraud Detection Agent...")
orchestrator = FraudDetectionOrchestrator()

# Example 1: Analyze a normal transaction
print("\n" + "="*60)
print("Example 1: Normal Transaction")
print("="*60)

normal_transaction = Transaction(
    transaction_id="TXN001",
    user_id="user_12345",
    amount=45.99,
    merchant="Amazon",
    merchant_category="online",
    timestamp=datetime.now(),
    location="New York, NY",
    device_id="device_abc123",
    ip_address="192.168.1.100",
    card_last_four="4567"
)

analysis = orchestrator.analyze_transaction(normal_transaction)

print(f"Transaction: ${normal_transaction.amount} at {normal_transaction.merchant}")
print(f"Fraudulent: {'YES ⚠️' if analysis.is_fraudulent else 'NO ✓'}")
print(f"Risk Level: {analysis.risk_level.value.upper()}")
print(f"Confidence: {analysis.confidence_score:.1%}")
print(f"Action: {analysis.recommended_action}")

# Example 2: Analyze a suspicious transaction
print("\n" + "="*60)
print("Example 2: Suspicious Transaction")
print("="*60)

suspicious_transaction = Transaction(
    transaction_id="TXN002",
    user_id="user_12345",
    amount=8500.00,
    merchant="CryptoExchange",
    merchant_category="cryptocurrency",
    timestamp=datetime.now().replace(hour=3),  # 3 AM
    location="International",
    device_id="device_new_xyz",
    ip_address="45.123.45.67",
    card_last_four="4567"
)

analysis = orchestrator.analyze_transaction(suspicious_transaction)

print(f"Transaction: ${suspicious_transaction.amount} at {suspicious_transaction.merchant}")
print(f"Fraudulent: {'YES ⚠️' if analysis.is_fraudulent else 'NO ✓'}")
print(f"Risk Level: {analysis.risk_level.value.upper()}")
print(f"Confidence: {analysis.confidence_score:.1%}")
print(f"Action: {analysis.recommended_action}")

if analysis.risk_factors:
    print("\nRisk Factors Detected:")
    for factor in analysis.risk_factors:
        print(f"  • {factor}")

print("\n" + "="*60)
print("Quick Start Complete!")
print("="*60)
print("\nNext steps:")
print("  1. Run 'python main.py' for a comprehensive demo")
print("  2. See README.md for full documentation")
print("  3. Check tests/ directory for usage examples")
