# System Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transaction Sources                           │
│  (API, Kafka, Queue, File, etc.)                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              TransactionStream (Async Processing)                │
│  • Queue-based ingestion                                         │
│  • Callback pattern                                              │
│  • Thread-safe operations                                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            FraudDetectionOrchestrator (Coordinator)              │
│  • Combines ML + LLM results                                     │
│  • Weighted scoring (60% ML, 40% LLM)                           │
│  • Risk level determination                                      │
│  • Action recommendation                                         │
└───────────┬──────────────────────────────┬──────────────────────┘
            │                              │
            ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────────────┐
│  MLAnomalyDetectionAgent │  │   LLMFraudAnalysisAgent          │
│  ───────────────────────  │  │  ─────────────────────────────   │
│  • Isolation Forest      │  │  • OpenAI GPT-3.5                │
│  • Feature extraction    │  │  • Contextual analysis           │
│  • Anomaly scoring       │  │  • Rule-based fallback           │
│  • Risk factor ID        │  │  • Natural language output       │
│  • Model persistence     │  │  • Risk indicator extraction     │
└──────────┬───────────────┘  └─────────────┬────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FraudAnalysis                               │
│  • is_fraudulent: bool                                           │
│  • risk_level: LOW/MEDIUM/HIGH/CRITICAL                         │
│  • confidence_score: 0.0-1.0                                    │
│  • risk_factors: list                                            │
│  • recommended_action: APPROVE/CHALLENGE/REVIEW/BLOCK           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Action Systems                                │
│  • Transaction approval/rejection                                │
│  • Alert generation                                              │
│  • Manual review queue                                           │
│  • User notification                                             │
│  • Audit logging                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Transaction Ingestion
```
Transaction → TransactionStream → Queue → Processing Thread
```

### 2. Analysis Pipeline
```
Transaction
    ↓
Orchestrator.analyze_transaction()
    ↓
    ├─→ MLAgent.analyze()
    │   ├─ Extract features (amount, time, category, location)
    │   ├─ Scale features
    │   ├─ Predict with Isolation Forest
    │   ├─ Calculate anomaly score
    │   └─ Identify risk factors
    │
    ├─→ LLMAgent.analyze()
    │   ├─ Build context prompt
    │   ├─ Call OpenAI API (or fallback)
    │   ├─ Parse response
    │   └─ Extract risk indicators
    │
    └─→ Combine Results
        ├─ Weight scores (60% ML, 40% LLM)
        ├─ Determine risk level
        ├─ Decide if fraudulent
        ├─ Recommend action
        └─ Create FraudAnalysis
```

### 3. Risk Assessment
```
Combined Score → Risk Level → Action
    0.0-0.4   →    LOW      → APPROVE
    0.4-0.6   →   MEDIUM    → CHALLENGE
    0.6-0.8   →    HIGH     → REVIEW
    0.8-1.0   →  CRITICAL   → BLOCK
```

## Component Details

### ML Anomaly Detection Agent

**Algorithm:** Isolation Forest
- Unsupervised learning
- Identifies outliers in feature space
- Works well with imbalanced data
- No labels required for training

**Features Used:**
- Transaction amount (normalized)
- Hour of day
- Day of week
- Merchant category (hashed)
- Location (hashed)

**Output:**
- is_anomaly (boolean)
- anomaly_score (0.0-1.0)
- risk_factors (list of strings)

### LLM Fraud Analysis Agent

**Primary Mode:** OpenAI GPT-3.5
- Contextual understanding
- Pattern recognition
- Natural language explanation
- Confidence assessment

**Fallback Mode:** Rule-based
- High amount detection
- Unusual time detection
- High-risk merchant categories
- International transaction flags

**Output:**
- analysis (natural language)
- risk_indicators (list)
- method (LLM or rule-based)

### Orchestrator

**Responsibilities:**
1. Agent coordination
2. Result combination
3. Score weighting
4. Risk classification
5. Action recommendation
6. Statistics tracking

**Scoring Algorithm:**
```python
combined_score = (ml_score * 0.6) + (llm_score * 0.4)
is_fraudulent = combined_score > 0.6 or ml_is_anomaly
```

### Transaction Stream

**Processing Model:**
- Asynchronous queue-based
- Callback pattern for results
- Thread-safe operations
- Configurable batch sizes

**Features:**
- Non-blocking ingestion
- Parallel processing capable
- Graceful error handling
- Queue monitoring

## Scalability Considerations

### Horizontal Scaling
```
Load Balancer
    ↓
┌────────┬────────┬────────┐
│ Agent1 │ Agent2 │ Agent3 │
└────────┴────────┴────────┘
    ↓        ↓        ↓
┌──────────────────────────┐
│    Shared Model Store    │
└──────────────────────────┘
```

### Distributed Processing
```
Transaction Queue (Kafka/RabbitMQ)
         ↓
    ┌────┴────┐
    │         │
Worker1   Worker2 ... WorkerN
    │         │
    └────┬────┘
         ↓
   Results Queue
```

## Integration Points

### Input Integration
- REST API endpoints
- Message queues (Kafka, RabbitMQ)
- Batch file processing
- Database triggers
- Webhooks

### Output Integration
- Decision APIs
- Alerting systems
- Dashboard/monitoring
- Audit logs
- Reporting systems

## Security Architecture

### Authentication & Authorization
- API key management
- Role-based access control
- Request validation
- Rate limiting

### Data Protection
- Encryption in transit (TLS)
- Encryption at rest
- PII handling
- Audit logging

### Model Security
- Model versioning
- Access control
- Validation before deployment
- Rollback capability

## Monitoring & Observability

### Key Metrics
- Transactions processed/second
- Average latency
- Fraud detection rate
- False positive rate
- Agent response times
- Queue depth

### Logging
- Transaction IDs
- Analysis results
- Agent decisions
- Errors and exceptions
- Performance metrics

### Alerting
- High fraud rate
- System errors
- Performance degradation
- Queue overflow
- Model drift

## Deployment Options

### Development
```
Single machine
Python environment
Local models
File-based storage
```

### Production
```
Containerized (Docker)
Kubernetes orchestration
Load balanced
Distributed cache
Cloud storage
Monitoring stack
```

## Technology Stack

### Core
- Python 3.8+
- NumPy, Pandas
- scikit-learn
- OpenAI API

### Optional Enhancements
- Redis (caching)
- PostgreSQL (persistence)
- Kafka (streaming)
- Prometheus (metrics)
- Grafana (visualization)
- Docker/Kubernetes (deployment)
