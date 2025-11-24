"""
Fraud Detection Orchestrator.

Coordinates multiple agents to provide comprehensive fraud analysis.
"""

from typing import List, Optional
from datetime import datetime

from fraud_detection.agents.base_agent import BaseAgent
from fraud_detection.agents.ml_agent import MLAnomalyDetectionAgent
from fraud_detection.agents.llm_agent import LLMFraudAnalysisAgent
from fraud_detection.models import Transaction, FraudAnalysis, FraudRiskLevel


class FraudDetectionOrchestrator:
    """Orchestrates multiple agents for comprehensive fraud detection."""
    
    def __init__(
        self,
        ml_agent: Optional[MLAnomalyDetectionAgent] = None,
        llm_agent: Optional[LLMFraudAnalysisAgent] = None
    ):
        """Initialize the orchestrator.
        
        Args:
            ml_agent: ML-based anomaly detection agent
            llm_agent: LLM-based fraud analysis agent
        """
        self.ml_agent = ml_agent or MLAnomalyDetectionAgent()
        self.llm_agent = llm_agent or LLMFraudAnalysisAgent()
        self.agents: List[BaseAgent] = [self.ml_agent, self.llm_agent]
        self.analysis_history: List[FraudAnalysis] = []
    
    def analyze_transaction(self, transaction: Transaction) -> FraudAnalysis:
        """Perform comprehensive fraud analysis on a transaction.
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Comprehensive fraud analysis
        """
        # Step 1: ML-based anomaly detection
        ml_results = self.ml_agent.analyze(transaction)
        
        # Step 2: LLM-based contextual analysis
        llm_results = self.llm_agent.analyze(transaction, ml_results)
        
        # Step 3: Combine results and make decision
        fraud_analysis = self._combine_analyses(
            transaction,
            ml_results,
            llm_results
        )
        
        # Store in history
        self.analysis_history.append(fraud_analysis)
        
        return fraud_analysis
    
    def _combine_analyses(
        self,
        transaction: Transaction,
        ml_results: dict,
        llm_results: dict
    ) -> FraudAnalysis:
        """Combine ML and LLM analyses into final decision.
        
        Args:
            transaction: Original transaction
            ml_results: Results from ML agent
            llm_results: Results from LLM agent
            
        Returns:
            Combined fraud analysis
        """
        # Calculate combined confidence score
        ml_score = ml_results.get("anomaly_score", 0)
        ml_weight = 0.6  # ML gets 60% weight
        llm_weight = 0.4  # LLM gets 40% weight
        
        # Estimate LLM risk score based on indicators
        llm_risk_score = len(llm_results.get("risk_indicators", [])) / 5.0
        llm_risk_score = min(llm_risk_score, 1.0)
        
        # Combined score
        combined_score = (ml_score * ml_weight) + (llm_risk_score * llm_weight)
        
        # Determine risk level
        risk_level = self._determine_risk_level(combined_score)
        
        # Determine if fraudulent
        is_fraudulent = combined_score > 0.6 or ml_results.get("is_anomaly", False)
        
        # Collect all risk factors
        risk_factors = ml_results.get("risk_factors", [])
        risk_factors.extend([
            f"LLM: {indicator}" for indicator in llm_results.get("risk_indicators", [])
        ])
        
        # Determine recommended action
        recommended_action = self._recommend_action(risk_level, is_fraudulent)
        
        return FraudAnalysis(
            transaction_id=transaction.transaction_id,
            is_fraudulent=is_fraudulent,
            risk_level=risk_level,
            confidence_score=combined_score,
            ml_score=ml_score,
            llm_analysis=llm_results.get("analysis", ""),
            risk_factors=risk_factors,
            recommended_action=recommended_action,
            timestamp=datetime.now()
        )
    
    def _determine_risk_level(self, score: float) -> FraudRiskLevel:
        """Determine risk level from combined score.
        
        Args:
            score: Combined risk score (0-1)
            
        Returns:
            Risk level classification
        """
        if score >= 0.8:
            return FraudRiskLevel.CRITICAL
        elif score >= 0.6:
            return FraudRiskLevel.HIGH
        elif score >= 0.4:
            return FraudRiskLevel.MEDIUM
        else:
            return FraudRiskLevel.LOW
    
    def _recommend_action(self, risk_level: FraudRiskLevel, is_fraudulent: bool) -> str:
        """Recommend action based on risk assessment.
        
        Args:
            risk_level: Assessed risk level
            is_fraudulent: Whether transaction is deemed fraudulent
            
        Returns:
            Recommended action string
        """
        if risk_level == FraudRiskLevel.CRITICAL:
            return "BLOCK: Decline transaction immediately and notify user"
        elif risk_level == FraudRiskLevel.HIGH:
            return "REVIEW: Hold transaction for manual review"
        elif risk_level == FraudRiskLevel.MEDIUM:
            return "CHALLENGE: Request additional authentication"
        else:
            return "APPROVE: Transaction appears legitimate"
    
    def train_ml_agent(self, transactions: List[Transaction]) -> None:
        """Train the ML agent with historical data.
        
        Args:
            transactions: Historical transactions for training
        """
        self.ml_agent.train(transactions)
    
    def get_statistics(self) -> dict:
        """Get statistics about analyzed transactions.
        
        Returns:
            Dictionary with statistics
        """
        if not self.analysis_history:
            return {
                "total_analyzed": 0,
                "fraudulent": 0,
                "fraud_rate": 0.0,
                "risk_distribution": {}
            }
        
        total = len(self.analysis_history)
        fraudulent = sum(1 for a in self.analysis_history if a.is_fraudulent)
        
        risk_dist = {
            "LOW": sum(1 for a in self.analysis_history if a.risk_level == FraudRiskLevel.LOW),
            "MEDIUM": sum(1 for a in self.analysis_history if a.risk_level == FraudRiskLevel.MEDIUM),
            "HIGH": sum(1 for a in self.analysis_history if a.risk_level == FraudRiskLevel.HIGH),
            "CRITICAL": sum(1 for a in self.analysis_history if a.risk_level == FraudRiskLevel.CRITICAL),
        }
        
        return {
            "total_analyzed": total,
            "fraudulent": fraudulent,
            "fraud_rate": fraudulent / total if total > 0 else 0.0,
            "risk_distribution": risk_dist,
            "average_confidence": sum(a.confidence_score for a in self.analysis_history) / total
        }
