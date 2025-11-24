"""
Main example demonstrating the AI Fraud Detection Agent system.
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fraud_detection.agents import (
    MLAnomalyDetectionAgent,
    LLMFraudAnalysisAgent,
    FraudDetectionOrchestrator
)
from fraud_detection.stream import TransactionStream, TransactionGenerator
from fraud_detection.models import Transaction

# Load environment variables
load_dotenv()


def process_transaction(orchestrator: FraudDetectionOrchestrator, transaction: Transaction) -> None:
    """Process a single transaction through the fraud detection system.
    
    Args:
        orchestrator: The fraud detection orchestrator
        transaction: Transaction to process
    """
    print(f"\n{'='*80}")
    print(f"Processing Transaction: {transaction.transaction_id}")
    print(f"{'='*80}")
    print(f"User: {transaction.user_id}")
    print(f"Amount: ${transaction.amount:.2f} {transaction.currency}")
    print(f"Merchant: {transaction.merchant} ({transaction.merchant_category})")
    print(f"Location: {transaction.location}")
    print(f"Time: {transaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {transaction.device_id}")
    
    # Analyze the transaction
    analysis = orchestrator.analyze_transaction(transaction)
    
    print(f"\n{'-'*80}")
    print("FRAUD ANALYSIS RESULTS")
    print(f"{'-'*80}")
    print(f"Fraudulent: {'YES' if analysis.is_fraudulent else 'NO'}")
    print(f"Risk Level: {analysis.risk_level.value.upper()}")
    print(f"Confidence Score: {analysis.confidence_score:.2%}")
    print(f"ML Score: {analysis.ml_score:.2f}")
    
    if analysis.risk_factors:
        print(f"\nRisk Factors:")
        for factor in analysis.risk_factors:
            print(f"  • {factor}")
    
    if analysis.llm_analysis:
        print(f"\nLLM Analysis:")
        print(f"  {analysis.llm_analysis}")
    
    print(f"\nRecommended Action:")
    print(f"  {analysis.recommended_action}")
    print(f"{'='*80}\n")


def demo_basic_analysis():
    """Demonstrate basic fraud detection analysis."""
    print("\n" + "="*80)
    print("AI FRAUD DETECTION AGENT - BASIC ANALYSIS DEMO")
    print("="*80)
    
    # Initialize agents
    print("\nInitializing fraud detection system...")
    ml_agent = MLAnomalyDetectionAgent()
    llm_agent = LLMFraudAnalysisAgent()
    orchestrator = FraudDetectionOrchestrator(ml_agent, llm_agent)
    
    # Generate training data
    print("Generating training data...")
    training_data = TransactionGenerator.generate_mixed_batch(100, fraud_ratio=0.1)
    orchestrator.train_ml_agent(training_data)
    
    # Test with normal transaction
    print("\n\n" + "="*80)
    print("TEST 1: NORMAL TRANSACTION")
    print("="*80)
    normal_txn = TransactionGenerator.generate_normal_transaction("TEST001")
    process_transaction(orchestrator, normal_txn)
    
    # Test with suspicious transaction
    print("\n\n" + "="*80)
    print("TEST 2: SUSPICIOUS TRANSACTION")
    print("="*80)
    suspicious_txn = TransactionGenerator.generate_suspicious_transaction("TEST002")
    process_transaction(orchestrator, suspicious_txn)
    
    # Print statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    stats = orchestrator.get_statistics()
    print(f"Total Transactions Analyzed: {stats['total_analyzed']}")
    print(f"Fraudulent Transactions: {stats['fraudulent']}")
    print(f"Fraud Rate: {stats['fraud_rate']:.2%}")
    print(f"Average Confidence Score: {stats['average_confidence']:.2%}")
    print(f"\nRisk Distribution:")
    for level, count in stats['risk_distribution'].items():
        print(f"  {level}: {count}")


def demo_stream_processing():
    """Demonstrate stream-based transaction processing."""
    print("\n\n" + "="*80)
    print("AI FRAUD DETECTION AGENT - STREAM PROCESSING DEMO")
    print("="*80)
    
    # Initialize system
    print("\nInitializing fraud detection system...")
    orchestrator = FraudDetectionOrchestrator()
    
    # Train with historical data
    print("Training ML model with historical data...")
    training_data = TransactionGenerator.generate_mixed_batch(150, fraud_ratio=0.1)
    orchestrator.train_ml_agent(training_data)
    
    # Set up stream processing
    results = []
    
    def stream_callback(transaction: Transaction):
        """Callback for processing streamed transactions."""
        analysis = orchestrator.analyze_transaction(transaction)
        results.append((transaction, analysis))
        
        # Print brief summary
        status = "🚨 FRAUD" if analysis.is_fraudulent else "✓ OK"
        print(f"{status} | {transaction.transaction_id} | ${transaction.amount:.2f} | {analysis.risk_level.value.upper()}")
    
    stream = TransactionStream(callback=stream_callback)
    stream.start()
    
    # Generate and process transactions
    print("\n" + "-"*80)
    print("Processing transaction stream...")
    print("-"*80)
    
    test_transactions = TransactionGenerator.generate_mixed_batch(20, fraud_ratio=0.15)
    stream.ingest_batch(test_transactions)
    
    # Wait for processing
    import time
    while stream.get_queue_size() > 0:
        time.sleep(0.1)
    
    time.sleep(1)  # Extra time to ensure all callbacks complete
    stream.stop()
    
    # Summary
    print("\n" + "="*80)
    print("STREAM PROCESSING SUMMARY")
    print("="*80)
    
    fraudulent_count = sum(1 for _, analysis in results if analysis.is_fraudulent)
    high_risk_count = sum(1 for _, analysis in results 
                         if analysis.risk_level.value in ['high', 'critical'])
    
    print(f"Total Transactions Processed: {len(results)}")
    print(f"Fraudulent Transactions Detected: {fraudulent_count}")
    print(f"High/Critical Risk Transactions: {high_risk_count}")
    print(f"Detection Rate: {fraudulent_count / len(results) * 100:.1f}%")
    
    # Show high-risk transactions
    print(f"\n{'='*80}")
    print("HIGH-RISK TRANSACTIONS DETAIL")
    print(f"{'='*80}")
    
    high_risk = [(txn, analysis) for txn, analysis in results 
                 if analysis.risk_level.value in ['high', 'critical']]
    
    for txn, analysis in high_risk[:5]:  # Show first 5
        print(f"\nTransaction {txn.transaction_id}:")
        print(f"  Amount: ${txn.amount:.2f}")
        print(f"  Merchant: {txn.merchant}")
        print(f"  Risk: {analysis.risk_level.value.upper()}")
        print(f"  Action: {analysis.recommended_action}")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("AI FRAUD DETECTION AGENT SYSTEM")
    print("Hybrid ML + LLM Multi-Agent Fraud Detection")
    print("="*80)
    
    # Run demos
    demo_basic_analysis()
    demo_stream_processing()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nThe AI Fraud Detection Agent system successfully demonstrated:")
    print("  ✓ ML-based anomaly detection")
    print("  ✓ LLM-based contextual analysis")
    print("  ✓ Multi-agent coordination")
    print("  ✓ Transaction stream ingestion")
    print("  ✓ Real-time fraud detection")
    print("\nFor more information, see README.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
