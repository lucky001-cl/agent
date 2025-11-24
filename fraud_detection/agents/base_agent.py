"""Base agent interface for fraud detection."""

from abc import ABC, abstractmethod
from typing import Any
from fraud_detection.models import Transaction, FraudAnalysis


class BaseAgent(ABC):
    """Base class for all fraud detection agents."""
    
    def __init__(self, name: str):
        """Initialize the agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name
    
    @abstractmethod
    def analyze(self, transaction: Transaction) -> Any:
        """Analyze a transaction.
        
        Args:
            transaction: Transaction to analyze
            
        Returns:
            Analysis results
        """
        pass
    
    def get_name(self) -> str:
        """Get the agent name."""
        return self.name
