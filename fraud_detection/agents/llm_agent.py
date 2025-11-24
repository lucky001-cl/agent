"""
LLM-based Fraud Analysis Agent.

Uses Large Language Models to provide contextual fraud analysis.
"""

import os
from typing import Dict, Optional
from datetime import datetime

from fraud_detection.agents.base_agent import BaseAgent
from fraud_detection.models import Transaction


class LLMFraudAnalysisAgent(BaseAgent):
    """LLM-based agent for contextual fraud analysis using OpenAI."""
    
    def __init__(self, name: str = "LLM Fraud Analyzer", api_key: Optional[str] = None):
        """Initialize the LLM agent.
        
        Args:
            name: Agent name
            api_key: OpenAI API key (if None, reads from environment)
        """
        super().__init__(name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print(f"Warning: OpenAI library not available. {self.name} will use rule-based analysis.")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        else:
            print(f"Note: No valid OpenAI API key provided. {self.name} will use rule-based analysis.")
    
    def analyze(self, transaction: Transaction, ml_results: Optional[Dict] = None) -> Dict:
        """Analyze a transaction using LLM for contextual understanding.
        
        Args:
            transaction: Transaction to analyze
            ml_results: Optional results from ML analysis
            
        Returns:
            Dictionary with LLM analysis results
        """
        if self.client:
            return self._llm_analysis(transaction, ml_results)
        else:
            return self._rule_based_analysis(transaction, ml_results)
    
    def _llm_analysis(self, transaction: Transaction, ml_results: Optional[Dict]) -> Dict:
        """Perform LLM-based analysis.
        
        Args:
            transaction: Transaction to analyze
            ml_results: Optional ML analysis results
            
        Returns:
            Analysis results
        """
        # Construct prompt for LLM
        prompt = self._build_analysis_prompt(transaction, ml_results)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fraud detection expert analyzing financial transactions. Provide concise, actionable insights about potential fraud risks."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            analysis = response.choices[0].message.content
            
            # Parse the response for risk indicators
            risk_indicators = self._extract_risk_indicators(analysis)
            
            return {
                "analysis": analysis,
                "risk_indicators": risk_indicators,
                "method": "LLM (GPT-3.5)"
            }
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return self._rule_based_analysis(transaction, ml_results)
    
    def _rule_based_analysis(self, transaction: Transaction, ml_results: Optional[Dict]) -> Dict:
        """Fallback rule-based analysis when LLM is not available.
        
        Args:
            transaction: Transaction to analyze
            ml_results: Optional ML analysis results
            
        Returns:
            Analysis results
        """
        risk_indicators = []
        analysis_parts = []
        
        # Analyze transaction patterns
        if transaction.amount > 10000:
            risk_indicators.append("very_high_amount")
            analysis_parts.append(f"Very high transaction amount of ${transaction.amount:.2f} detected.")
        elif transaction.amount > 5000:
            risk_indicators.append("high_amount")
            analysis_parts.append(f"High transaction amount of ${transaction.amount:.2f}.")
        
        # Time-based analysis
        hour = transaction.timestamp.hour
        if hour < 6 or hour > 22:
            risk_indicators.append("unusual_time")
            analysis_parts.append(f"Transaction occurred at unusual hour ({hour}:00).")
        
        # Merchant category analysis
        high_risk_merchants = ["gambling", "cryptocurrency", "wire_transfer", "money_order", "atm_withdrawal"]
        if transaction.merchant_category.lower() in high_risk_merchants:
            risk_indicators.append("high_risk_merchant")
            analysis_parts.append(f"Transaction with high-risk merchant category: {transaction.merchant_category}.")
        
        # Location analysis
        international_indicators = ["international", "foreign", "overseas"]
        if any(indicator in transaction.location.lower() for indicator in international_indicators):
            risk_indicators.append("international_transaction")
            analysis_parts.append("International transaction detected.")
        
        # Incorporate ML results if available
        if ml_results and ml_results.get("is_anomaly"):
            risk_indicators.append("ml_anomaly")
            analysis_parts.append(f"ML model detected anomalous pattern (score: {ml_results.get('anomaly_score', 0):.2f}).")
        
        # Construct final analysis
        if not analysis_parts:
            analysis = "Transaction appears normal with no significant risk indicators."
        else:
            analysis = " ".join(analysis_parts)
            if len(risk_indicators) >= 3:
                analysis += " Multiple risk factors present - recommend manual review."
            elif len(risk_indicators) >= 2:
                analysis += " Moderate risk - suggest additional verification."
        
        return {
            "analysis": analysis,
            "risk_indicators": risk_indicators,
            "method": "Rule-based"
        }
    
    def _build_analysis_prompt(self, transaction: Transaction, ml_results: Optional[Dict]) -> str:
        """Build prompt for LLM analysis.
        
        Args:
            transaction: Transaction to analyze
            ml_results: Optional ML results
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this transaction for fraud risk:

Transaction Details:
- ID: {transaction.transaction_id}
- Amount: ${transaction.amount:.2f} {transaction.currency}
- Merchant: {transaction.merchant} ({transaction.merchant_category})
- Location: {transaction.location}
- Time: {transaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Device: {transaction.device_id}
- IP: {transaction.ip_address}
"""
        
        if ml_results:
            prompt += f"""
ML Analysis Results:
- Anomaly Detected: {ml_results.get('is_anomaly', False)}
- Anomaly Score: {ml_results.get('anomaly_score', 0):.2f}
- Risk Factors: {', '.join(ml_results.get('risk_factors', []))}
"""
        
        prompt += """
Provide a brief analysis covering:
1. Key fraud risk indicators
2. Overall risk assessment
3. Recommended action (approve/review/decline)
"""
        
        return prompt
    
    def _extract_risk_indicators(self, analysis: str) -> list:
        """Extract risk indicators from LLM analysis text.
        
        Args:
            analysis: Analysis text from LLM
            
        Returns:
            List of risk indicators
        """
        indicators = []
        analysis_lower = analysis.lower()
        
        # Check for common risk keywords
        risk_keywords = {
            "high amount": "high_amount",
            "unusual time": "unusual_time",
            "suspicious": "suspicious_pattern",
            "high risk": "high_risk",
            "anomaly": "anomaly",
            "velocity": "high_velocity",
            "international": "international",
            "multiple": "multiple_factors"
        }
        
        for keyword, indicator in risk_keywords.items():
            if keyword in analysis_lower:
                indicators.append(indicator)
        
        return indicators
