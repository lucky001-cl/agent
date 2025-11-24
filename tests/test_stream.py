"""Tests for transaction stream processing."""

import unittest
import time

from fraud_detection.stream import TransactionStream, TransactionGenerator
from fraud_detection.models import Transaction


class TestTransactionStream(unittest.TestCase):
    """Test cases for Transaction Stream."""
    
    def test_stream_initialization(self):
        """Test stream initialization."""
        stream = TransactionStream()
        
        self.assertFalse(stream.is_running)
        self.assertEqual(stream.get_queue_size(), 0)
    
    def test_stream_start_stop(self):
        """Test starting and stopping the stream."""
        stream = TransactionStream()
        
        stream.start()
        self.assertTrue(stream.is_running)
        
        stream.stop()
        self.assertFalse(stream.is_running)
    
    def test_ingest_transaction(self):
        """Test ingesting a single transaction."""
        stream = TransactionStream()
        txn = TransactionGenerator.generate_normal_transaction("TEST001")
        
        stream.ingest_transaction(txn)
        
        self.assertEqual(stream.get_queue_size(), 1)
    
    def test_ingest_batch(self):
        """Test ingesting a batch of transactions."""
        stream = TransactionStream()
        transactions = TransactionGenerator.generate_mixed_batch(10)
        
        stream.ingest_batch(transactions)
        
        self.assertEqual(stream.get_queue_size(), 10)
    
    def test_stream_processing(self):
        """Test stream processing with callback."""
        processed = []
        
        def callback(txn):
            processed.append(txn)
        
        stream = TransactionStream(callback=callback)
        stream.start()
        
        # Ingest transactions
        transactions = TransactionGenerator.generate_mixed_batch(5)
        stream.ingest_batch(transactions)
        
        # Wait for processing
        time.sleep(2)
        stream.stop()
        
        self.assertEqual(len(processed), 5)


class TestTransactionGenerator(unittest.TestCase):
    """Test cases for Transaction Generator."""
    
    def test_generate_normal_transaction(self):
        """Test generating a normal transaction."""
        txn = TransactionGenerator.generate_normal_transaction("TEST001")
        
        self.assertEqual(txn.transaction_id, "TEST001")
        self.assertGreater(txn.amount, 0)
        self.assertLess(txn.amount, 1000)
        self.assertIsInstance(txn, Transaction)
    
    def test_generate_suspicious_transaction(self):
        """Test generating a suspicious transaction."""
        txn = TransactionGenerator.generate_suspicious_transaction("TEST002")
        
        self.assertEqual(txn.transaction_id, "TEST002")
        self.assertGreater(txn.amount, 2000)  # Suspicious amounts are high
        self.assertIsInstance(txn, Transaction)
    
    def test_generate_mixed_batch(self):
        """Test generating a mixed batch of transactions."""
        batch = TransactionGenerator.generate_mixed_batch(100, fraud_ratio=0.2)
        
        self.assertEqual(len(batch), 100)
        
        # Check that transactions have varied characteristics
        amounts = [txn.amount for txn in batch]
        self.assertGreater(max(amounts), min(amounts))


if __name__ == "__main__":
    unittest.main()
