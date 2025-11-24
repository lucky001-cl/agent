"""
Transaction stream ingestion and processing.
"""

import json
from typing import List, Callable, Optional
from datetime import datetime
from queue import Queue
from threading import Thread
import time

from fraud_detection.models import Transaction, TransactionStatus


class TransactionStream:
    """Handles streaming transaction data ingestion."""
    
    def __init__(self, callback: Optional[Callable[[Transaction], None]] = None):
        """Initialize transaction stream.
        
        Args:
            callback: Function to call for each transaction
        """
        self.callback = callback
        self.queue: Queue = Queue()
        self.is_running = False
        self.processing_thread: Optional[Thread] = None
    
    def start(self) -> None:
        """Start processing the transaction stream."""
        if self.is_running:
            print("Stream already running")
            return
        
        self.is_running = True
        self.processing_thread = Thread(target=self._process_stream, daemon=True)
        self.processing_thread.start()
        print("Transaction stream started")
    
    def stop(self) -> None:
        """Stop processing the transaction stream."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        print("Transaction stream stopped")
    
    def ingest_transaction(self, transaction: Transaction) -> None:
        """Add a transaction to the stream.
        
        Args:
            transaction: Transaction to ingest
        """
        self.queue.put(transaction)
    
    def ingest_batch(self, transactions: List[Transaction]) -> None:
        """Add multiple transactions to the stream.
        
        Args:
            transactions: List of transactions to ingest
        """
        for txn in transactions:
            self.queue.put(txn)
    
    def _process_stream(self) -> None:
        """Process transactions from the queue."""
        while self.is_running:
            try:
                if not self.queue.empty():
                    transaction = self.queue.get(timeout=1)
                    if self.callback:
                        self.callback(transaction)
                    self.queue.task_done()
                else:
                    time.sleep(0.1)  # Brief sleep when queue is empty
            except Exception as e:
                print(f"Error processing transaction: {e}")
    
    def get_queue_size(self) -> int:
        """Get the current size of the processing queue.
        
        Returns:
            Number of transactions in queue
        """
        return self.queue.qsize()


class TransactionGenerator:
    """Generates synthetic transaction data for testing."""
    
    @staticmethod
    def generate_normal_transaction(transaction_id: str) -> Transaction:
        """Generate a normal transaction.
        
        Args:
            transaction_id: ID for the transaction
            
        Returns:
            Normal transaction
        """
        import random
        
        merchants = [
            ("Amazon", "online"),
            ("Walmart", "retail"),
            ("Starbucks", "restaurant"),
            ("Shell", "gas"),
            ("Target", "retail"),
            ("McDonald's", "restaurant"),
            ("Whole Foods", "grocery")
        ]
        
        locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX"]
        
        merchant, category = random.choice(merchants)
        
        return Transaction(
            transaction_id=transaction_id,
            user_id=f"user_{random.randint(1000, 9999)}",
            amount=round(random.uniform(5.0, 500.0), 2),
            merchant=merchant,
            merchant_category=category,
            timestamp=datetime.now(),
            location=random.choice(locations),
            device_id=f"device_{random.randint(1, 100)}",
            ip_address=f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            card_last_four=f"{random.randint(0, 9999):04d}",
            currency="USD",
            status=TransactionStatus.PENDING
        )
    
    @staticmethod
    def generate_suspicious_transaction(transaction_id: str) -> Transaction:
        """Generate a suspicious/fraudulent transaction.
        
        Args:
            transaction_id: ID for the transaction
            
        Returns:
            Suspicious transaction
        """
        import random
        
        suspicious_merchants = [
            ("CryptoExchange", "cryptocurrency"),
            ("OffshoreGaming", "gambling"),
            ("QuickWire", "wire_transfer"),
            ("CashAdvance", "atm_withdrawal")
        ]
        
        merchant, category = random.choice(suspicious_merchants)
        
        # Suspicious characteristics
        hour = random.choice([2, 3, 4, 23, 1])  # Late night
        timestamp = datetime.now().replace(hour=hour)
        
        return Transaction(
            transaction_id=transaction_id,
            user_id=f"user_{random.randint(1000, 9999)}",
            amount=round(random.uniform(2000.0, 15000.0), 2),  # High amount
            merchant=merchant,
            merchant_category=category,
            timestamp=timestamp,
            location="International Location",
            device_id=f"device_new_{random.randint(1, 100)}",
            ip_address=f"45.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            card_last_four=f"{random.randint(0, 9999):04d}",
            currency="USD",
            status=TransactionStatus.PENDING
        )
    
    @staticmethod
    def generate_mixed_batch(count: int, fraud_ratio: float = 0.1) -> List[Transaction]:
        """Generate a batch of mixed transactions.
        
        Args:
            count: Number of transactions to generate
            fraud_ratio: Proportion of fraudulent transactions (0-1)
            
        Returns:
            List of transactions
        """
        import random
        
        transactions = []
        fraud_count = int(count * fraud_ratio)
        
        for i in range(count):
            txn_id = f"TXN{i:06d}"
            if i < fraud_count:
                txn = TransactionGenerator.generate_suspicious_transaction(txn_id)
            else:
                txn = TransactionGenerator.generate_normal_transaction(txn_id)
            transactions.append(txn)
        
        random.shuffle(transactions)
        return transactions
