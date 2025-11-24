"""
ML-based Anomaly Detection Agent.

Uses machine learning models to detect anomalous transaction patterns.
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

from fraud_detection.agents.base_agent import BaseAgent
from fraud_detection.models import Transaction


class MLAnomalyDetectionAgent(BaseAgent):
    """ML-based agent for detecting anomalous transactions using Isolation Forest."""
    
    def __init__(self, name: str = "ML Anomaly Detector", contamination: float = 0.1):
        """Initialize the ML agent.
        
        Args:
            name: Agent name
            contamination: Expected proportion of outliers in the dataset
        """
        super().__init__(name)
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.transaction_history: List[Transaction] = []
        
    def extract_features(self, transaction: Transaction) -> np.ndarray:
        """Extract numerical features from a transaction.
        
        Args:
            transaction: Transaction to extract features from
            
        Returns:
            Numpy array of features
        """
        # Time-based features
        hour = transaction.timestamp.hour
        day_of_week = transaction.timestamp.weekday()
        
        # Amount-based features
        amount = transaction.amount
        
        # Categorical features encoded as numbers (simple approach)
        merchant_category_hash = hash(transaction.merchant_category) % 1000
        location_hash = hash(transaction.location) % 1000
        
        # Combine features
        features = np.array([
            amount,
            hour,
            day_of_week,
            merchant_category_hash,
            location_hash
        ])
        
        return features.reshape(1, -1)
    
    def train(self, transactions: List[Transaction]) -> None:
        """Train the anomaly detection model.
        
        Args:
            transactions: List of historical transactions
        """
        if len(transactions) < 10:
            print(f"Warning: Only {len(transactions)} transactions provided. Need more for reliable training.")
            return
        
        # Extract features from all transactions
        features_list = [self.extract_features(txn).flatten() for txn in transactions]
        X = np.vstack(features_list)
        
        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled)
        self.is_trained = True
        self.transaction_history = transactions
        print(f"{self.name} trained on {len(transactions)} transactions")
    
    def analyze(self, transaction: Transaction) -> Dict:
        """Analyze a transaction for anomalies.
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Dictionary with anomaly score and prediction
        """
        if not self.is_trained:
            # If not trained, create synthetic training data
            self._bootstrap_training()
        
        # Extract and scale features
        features = self.extract_features(transaction)
        features_scaled = self.scaler.transform(features)
        
        # Predict anomaly (-1 for anomaly, 1 for normal)
        prediction = self.model.predict(features_scaled)[0]
        
        # Get anomaly score (lower = more anomalous)
        anomaly_score = self.model.score_samples(features_scaled)[0]
        
        # Normalize score to 0-1 range (higher = more anomalous)
        # Isolation forest scores are typically negative, with more negative = more anomalous
        normalized_score = 1 / (1 + np.exp(anomaly_score))
        
        # Analyze specific risk factors
        risk_factors = self._identify_risk_factors(transaction)
        
        return {
            "is_anomaly": bool(prediction == -1),
            "anomaly_score": float(normalized_score),
            "risk_factors": risk_factors,
            "model": "Isolation Forest"
        }
    
    def _identify_risk_factors(self, transaction: Transaction) -> List[str]:
        """Identify specific risk factors in the transaction.
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        
        # High amount transactions
        if transaction.amount > 5000:
            risk_factors.append("High transaction amount (>$5000)")
        
        # Unusual hours (late night/early morning)
        if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 22:
            risk_factors.append("Unusual transaction time (late night/early morning)")
        
        # High-risk merchant categories
        high_risk_categories = ["gambling", "cryptocurrency", "wire_transfer", "money_order"]
        if transaction.merchant_category.lower() in high_risk_categories:
            risk_factors.append(f"High-risk merchant category: {transaction.merchant_category}")
        
        # Check for velocity (multiple transactions in short time)
        if self.transaction_history:
            recent_txns = [
                t for t in self.transaction_history[-50:]  # Check last 50 transactions
                if t.user_id == transaction.user_id and
                (transaction.timestamp - t.timestamp) < timedelta(hours=1)
            ]
            if len(recent_txns) > 5:
                risk_factors.append(f"High transaction velocity: {len(recent_txns)} transactions in last hour")
        
        return risk_factors
    
    def _bootstrap_training(self) -> None:
        """Create synthetic training data for initial model."""
        # Generate synthetic normal transactions
        synthetic_transactions = []
        base_time = datetime.now()
        
        for i in range(100):
            txn = Transaction(
                transaction_id=f"SYN{i:05d}",
                user_id=f"user_{i % 20}",
                amount=np.random.lognormal(4, 1),  # Log-normal distribution for amounts
                merchant=f"merchant_{i % 30}",
                merchant_category=np.random.choice(["retail", "grocery", "restaurant", "online", "gas"]),
                timestamp=base_time - timedelta(hours=np.random.randint(0, 720)),
                location=np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
                device_id=f"device_{i % 15}",
                ip_address=f"192.168.{i % 256}.{i % 256}",
                card_last_four=f"{i % 10000:04d}"
            )
            synthetic_transactions.append(txn)
        
        self.train(synthetic_transactions)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "contamination": self.contamination
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.contamination = model_data["contamination"]
        self.is_trained = True
