# AI Fraud Detection Agent

A hybrid ML + LLM multi-agent system for detecting fraudulent transactions in real-time.

## Overview

This system combines machine learning-based anomaly detection with large language model (LLM) contextual analysis to provide comprehensive fraud detection capabilities. The multi-agent architecture allows for scalable, real-time processing of transaction streams.

## Features

- **ML-Based Anomaly Detection**: Uses Isolation Forest algorithm to detect anomalous transaction patterns
- **LLM Contextual Analysis**: Leverages GPT models for contextual fraud risk assessment
- **Multi-Agent Architecture**: Orchestrates multiple specialized agents for comprehensive analysis
- **Real-Time Stream Processing**: Processes transaction streams in real-time
- **Risk Classification**: Categorizes transactions into LOW, MEDIUM, HIGH, and CRITICAL risk levels
- **Actionable Recommendations**: Provides clear recommendations (APPROVE, CHALLENGE, REVIEW, BLOCK)

## Architecture

The system consists of four main components:

### 1. ML Anomaly Detection Agent
- Uses scikit-learn's Isolation Forest for unsupervised anomaly detection
- Extracts features from transactions (amount, time, merchant category, location)
- Identifies unusual patterns based on historical data
- Provides anomaly scores and specific risk factors

### 2. LLM Fraud Analysis Agent
- Uses OpenAI's GPT models for contextual analysis
- Analyzes transaction context and patterns
- Falls back to rule-based analysis when LLM is unavailable
- Provides natural language explanations of fraud risks

### 3. Fraud Detection Orchestrator
- Coordinates ML and LLM agents
- Combines results using weighted scoring
- Makes final fraud determination
- Provides risk levels and recommended actions

### 4. Transaction Stream Processor
- Ingests transactions from various sources
- Processes transactions asynchronously
- Supports batch and real-time processing
- Includes synthetic data generation for testing

## Installation

```bash
# Clone the repository
git clone https://github.com/lucky001-cl/agent.git
cd agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional for LLM features)
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### Basic Usage

```python
from fraud_detection.agents import FraudDetectionOrchestrator
from fraud_detection.models import Transaction
from datetime import datetime

# Initialize the system
orchestrator = FraudDetectionOrchestrator()

# Create a transaction
transaction = Transaction(
    transaction_id="TXN001",
    user_id="user_123",
    amount=5000.00,
    merchant="Crypto Exchange",
    merchant_category="cryptocurrency",
    timestamp=datetime.now(),
    location="International",
    device_id="device_456",
    ip_address="45.123.45.67",
    card_last_four="1234"
)

# Analyze for fraud
analysis = orchestrator.analyze_transaction(transaction)

print(f"Fraudulent: {analysis.is_fraudulent}")
print(f"Risk Level: {analysis.risk_level.value}")
print(f"Confidence: {analysis.confidence_score:.2%}")
print(f"Action: {analysis.recommended_action}")
```

### Stream Processing

```python
from fraud_detection.stream import TransactionStream, TransactionGenerator

# Define callback for processing
def process_transaction(transaction):
    analysis = orchestrator.analyze_transaction(transaction)
    print(f"Processed {transaction.transaction_id}: {analysis.risk_level.value}")

# Set up stream
stream = TransactionStream(callback=process_transaction)
stream.start()

# Generate and ingest transactions
transactions = TransactionGenerator.generate_mixed_batch(100)
stream.ingest_batch(transactions)

# Wait for processing, then stop
stream.stop()
```

### Running the Demo

```bash
python main.py
```

This will run comprehensive demonstrations of:
- Basic fraud detection analysis
- Stream-based transaction processing
- ML and LLM agent coordination
- Risk assessment and recommendations

## Data Models

### Transaction
Represents a financial transaction with fields:
- `transaction_id`: Unique identifier
- `user_id`: User/customer identifier
- `amount`: Transaction amount
- `merchant`: Merchant name
- `merchant_category`: Category (e.g., retail, online, cryptocurrency)
- `timestamp`: Transaction timestamp
- `location`: Geographic location
- `device_id`: Device identifier
- `ip_address`: IP address
- `card_last_four`: Last 4 digits of card

### FraudAnalysis
Results of fraud detection analysis:
- `is_fraudulent`: Boolean fraud determination
- `risk_level`: LOW, MEDIUM, HIGH, or CRITICAL
- `confidence_score`: 0-1 confidence score
- `ml_score`: ML model's anomaly score
- `llm_analysis`: LLM's contextual analysis
- `risk_factors`: List of identified risk factors
- `recommended_action`: Suggested action to take

## Risk Factors

The system detects various risk factors:
- High transaction amounts (>$5,000)
- Unusual transaction times (late night/early morning)
- High-risk merchant categories (gambling, cryptocurrency, wire transfers)
- International transactions
- High transaction velocity
- Anomalous patterns detected by ML
- Multiple concurrent risk indicators

## Configuration

### ML Agent Configuration
```python
ml_agent = MLAnomalyDetectionAgent(contamination=0.1)
orchestrator.train_ml_agent(historical_transactions)
```

### LLM Agent Configuration
```python
llm_agent = LLMFraudAnalysisAgent(api_key="your-api-key")
```

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM features (optional)

## Testing

The system includes synthetic data generation for testing:

```python
from fraud_detection.stream import TransactionGenerator

# Generate normal transactions
normal = TransactionGenerator.generate_normal_transaction("TXN001")

# Generate suspicious transactions
suspicious = TransactionGenerator.generate_suspicious_transaction("TXN002")

# Generate mixed batch (90% normal, 10% fraudulent)
batch = TransactionGenerator.generate_mixed_batch(count=100, fraud_ratio=0.1)
```

## Performance

- **Throughput**: Processes hundreds of transactions per second
- **Latency**: < 100ms per transaction (without LLM), < 2s (with LLM)
- **Accuracy**: Depends on training data; typically 85-95% with good historical data
- **False Positive Rate**: Configurable via contamination parameter

## Future Enhancements

- Integration with real transaction streams (Kafka, RabbitMQ)
- Model persistence and versioning
- A/B testing framework for model evaluation
- Dashboard for monitoring and alerting
- Federated learning for privacy-preserving training
- Graph-based fraud detection for network analysis
- Real-time model retraining pipeline

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues, please open a GitHub issue.