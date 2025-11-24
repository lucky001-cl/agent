# API Reference - AI Fraud Detection Agent

## Core Classes

### Transaction
Represents a financial transaction.

```python
from fraud_detection.models import Transaction
from datetime import datetime

transaction = Transaction(
    transaction_id="TXN001",
    user_id="user_123",
    amount=100.50,
    merchant="Amazon",
    merchant_category="online",
    timestamp=datetime.now(),
    location="New York, NY",
    device_id="device_1",
    ip_address="192.168.1.1",
    card_last_four="1234",
    currency="USD"  # Optional, defaults to "USD"
)
```

**Attributes:**
- `transaction_id` (str): Unique transaction identifier
- `user_id` (str): User/customer identifier
- `amount` (float): Transaction amount
- `merchant` (str): Merchant name
- `merchant_category` (str): Merchant category
- `timestamp` (datetime): Transaction timestamp
- `location` (str): Geographic location
- `device_id` (str): Device identifier
- `ip_address` (str): IP address
- `card_last_four` (str): Last 4 digits of card
- `currency` (str): Currency code (default: "USD")
- `status` (TransactionStatus): Transaction status
- `metadata` (Dict): Additional metadata

### FraudAnalysis
Results of fraud detection analysis.

```python
from fraud_detection.models import FraudAnalysis, FraudRiskLevel

analysis = FraudAnalysis(
    transaction_id="TXN001",
    is_fraudulent=True,
    risk_level=FraudRiskLevel.HIGH,
    confidence_score=0.85,
    ml_score=0.9,
    llm_analysis="High risk transaction",
    risk_factors=["high_amount", "unusual_time"],
    recommended_action="REVIEW"
)
```

**Attributes:**
- `transaction_id` (str): Transaction identifier
- `is_fraudulent` (bool): Fraud determination
- `risk_level` (FraudRiskLevel): Risk classification
- `confidence_score` (float): Confidence (0.0-1.0)
- `ml_score` (float): ML model score
- `llm_analysis` (str): LLM analysis text
- `risk_factors` (list): Identified risk factors
- `recommended_action` (str): Recommended action
- `timestamp` (datetime): Analysis timestamp

## Agent Classes

### MLAnomalyDetectionAgent
Machine learning-based anomaly detection agent.

```python
from fraud_detection.agents import MLAnomalyDetectionAgent

# Initialize agent
ml_agent = MLAnomalyDetectionAgent(
    name="ML Detector",
    contamination=0.1
)

# Train on historical data
ml_agent.train(historical_transactions)

# Analyze a transaction
result = ml_agent.analyze(transaction)
# Returns: {"is_anomaly": bool, "anomaly_score": float, "risk_factors": list}
```

**Methods:**
- `train(transactions: List[Transaction])`: Train the model
- `analyze(transaction: Transaction) -> Dict`: Analyze transaction
- `extract_features(transaction: Transaction) -> np.ndarray`: Extract features
- `save_model(filepath: str)`: Save trained model
- `load_model(filepath: str)`: Load trained model

### LLMFraudAnalysisAgent
LLM-based contextual fraud analysis agent.

```python
from fraud_detection.agents import LLMFraudAnalysisAgent

# Initialize agent
llm_agent = LLMFraudAnalysisAgent(
    name="LLM Analyzer",
    api_key="your-api-key"  # Optional, reads from env
)

# Analyze a transaction
result = llm_agent.analyze(transaction, ml_results)
# Returns: {"analysis": str, "risk_indicators": list, "method": str}
```

**Methods:**
- `analyze(transaction: Transaction, ml_results: Optional[Dict]) -> Dict`: Analyze transaction

### FraudDetectionOrchestrator
Orchestrates multiple agents for comprehensive analysis.

```python
from fraud_detection.agents import FraudDetectionOrchestrator

# Initialize orchestrator
orchestrator = FraudDetectionOrchestrator()

# Or with custom agents
orchestrator = FraudDetectionOrchestrator(
    ml_agent=ml_agent,
    llm_agent=llm_agent
)

# Train ML agent
orchestrator.train_ml_agent(historical_transactions)

# Analyze transaction
analysis = orchestrator.analyze_transaction(transaction)
# Returns: FraudAnalysis object

# Get statistics
stats = orchestrator.get_statistics()
```

**Methods:**
- `analyze_transaction(transaction: Transaction) -> FraudAnalysis`: Comprehensive analysis
- `train_ml_agent(transactions: List[Transaction])`: Train ML component
- `get_statistics() -> dict`: Get analysis statistics

## Stream Processing

### TransactionStream
Handles streaming transaction ingestion.

```python
from fraud_detection.stream import TransactionStream

# Define callback
def process_transaction(txn):
    analysis = orchestrator.analyze_transaction(txn)
    print(f"Processed: {txn.transaction_id}")

# Initialize and start stream
stream = TransactionStream(callback=process_transaction)
stream.start()

# Ingest transactions
stream.ingest_transaction(single_transaction)
stream.ingest_batch(transaction_list)

# Stop stream
stream.stop()
```

**Methods:**
- `start()`: Start processing stream
- `stop()`: Stop processing stream
- `ingest_transaction(transaction: Transaction)`: Add single transaction
- `ingest_batch(transactions: List[Transaction])`: Add multiple transactions
- `get_queue_size() -> int`: Get queue size

### TransactionGenerator
Generates synthetic transaction data.

```python
from fraud_detection.stream import TransactionGenerator

# Generate normal transaction
normal = TransactionGenerator.generate_normal_transaction("TXN001")

# Generate suspicious transaction
suspicious = TransactionGenerator.generate_suspicious_transaction("TXN002")

# Generate mixed batch
batch = TransactionGenerator.generate_mixed_batch(
    count=100,
    fraud_ratio=0.1  # 10% fraudulent
)
```

## Enums

### TransactionStatus
```python
from fraud_detection.models import TransactionStatus

# Values: PENDING, APPROVED, REJECTED, FLAGGED
status = TransactionStatus.PENDING
```

### FraudRiskLevel
```python
from fraud_detection.models import FraudRiskLevel

# Values: LOW, MEDIUM, HIGH, CRITICAL
risk = FraudRiskLevel.HIGH
```

## Configuration

Configuration settings can be customized in `fraud_detection/config.py`:

```python
from fraud_detection.config import (
    ML_CONFIG,
    RISK_THRESHOLDS,
    SCORING_CONFIG,
    HIGH_RISK_CATEGORIES
)

# Customize thresholds
RISK_THRESHOLDS["high_amount"] = 10000.0

# Adjust scoring weights
SCORING_CONFIG["ml_weight"] = 0.7
SCORING_CONFIG["llm_weight"] = 0.3
```

## Examples

### Basic Usage
```python
from fraud_detection.agents import FraudDetectionOrchestrator
from fraud_detection.models import Transaction
from datetime import datetime

orchestrator = FraudDetectionOrchestrator()

transaction = Transaction(
    transaction_id="TXN001",
    user_id="user_123",
    amount=5000.00,
    merchant="Test Merchant",
    merchant_category="retail",
    timestamp=datetime.now(),
    location="New York, NY",
    device_id="device_1",
    ip_address="192.168.1.1",
    card_last_four="1234"
)

analysis = orchestrator.analyze_transaction(transaction)
print(f"Fraudulent: {analysis.is_fraudulent}")
print(f"Risk: {analysis.risk_level.value}")
print(f"Action: {analysis.recommended_action}")
```

### Stream Processing
```python
from fraud_detection.agents import FraudDetectionOrchestrator
from fraud_detection.stream import TransactionStream, TransactionGenerator

orchestrator = FraudDetectionOrchestrator()

def callback(txn):
    analysis = orchestrator.analyze_transaction(txn)
    if analysis.is_fraudulent:
        print(f"⚠️ FRAUD: {txn.transaction_id}")

stream = TransactionStream(callback=callback)
stream.start()

transactions = TransactionGenerator.generate_mixed_batch(100)
stream.ingest_batch(transactions)

stream.stop()
```

### Model Training
```python
from fraud_detection.agents import MLAnomalyDetectionAgent
from fraud_detection.stream import TransactionGenerator

# Generate training data
training_data = TransactionGenerator.generate_mixed_batch(1000, fraud_ratio=0.1)

# Train model
ml_agent = MLAnomalyDetectionAgent()
ml_agent.train(training_data)

# Save model
ml_agent.save_model("models/fraud_model.pkl")

# Load model later
ml_agent.load_model("models/fraud_model.pkl")
```
