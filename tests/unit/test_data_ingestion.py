"""
Unit Tests for Data Ingestion Module
Tests for data loading, streaming, Kafka integration, and database operations
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call, mock_open
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import tempfile
import shutil
import os
from pathlib import Path
import h5py
from typing import List, Dict, Any

# Import modules to test
from src.data_ingestion.data_loader import DataLoader
from src.data_ingestion.stream_simulator import StreamSimulator
from src.data_ingestion.kafka_producer import KafkaProducer
from src.data_ingestion.kafka_consumer import KafkaConsumer
from src.data_ingestion.database_manager import DatabaseManager


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader()
        
        # Create sample data
        self.sample_data = np.random.randn(1000, 10)
        self.sample_labels = np.random.randint(0, 2, 1000)
        
        # Create test files
        self.npy_file = os.path.join(self.temp_dir, 'test_data.npy')
        np.save(self.npy_file, self.sample_data)
        
        self.h5_file = os.path.join(self.temp_dir, 'test_data.h5')
        with h5py.File(self.h5_file, 'w') as f:
            f.create_dataset('data', data=self.sample_data)
            f.create_dataset('labels', data=self.sample_labels)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_npy_file(self):
        """Test loading NPY file"""
        data = self.data_loader.load_npy(self.npy_file)
        
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, self.sample_data.shape)
        np.testing.assert_array_almost_equal(data, self.sample_data)
    
    def test_load_h5_file(self):
        """Test loading H5 file"""
        data = self.data_loader.load_h5(self.h5_file, dataset_name='data')
        
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, self.sample_data.shape)
        np.testing.assert_array_almost_equal(data, self.sample_data)
    
    def test_load_h5_with_labels(self):
        """Test loading H5 file with labels"""
        data, labels = self.data_loader.load_h5_with_labels(
            self.h5_file, 
            data_key='data', 
            label_key='labels'
        )
        
        self.assertEqual(data.shape, self.sample_data.shape)
        self.assertEqual(labels.shape, self.sample_labels.shape)
        np.testing.assert_array_equal(labels, self.sample_labels)
    
    def test_load_csv_file(self):
        """Test loading CSV file"""
        csv_file = os.path.join(self.temp_dir, 'test_data.csv')
        df = pd.DataFrame(self.sample_data)
        df.to_csv(csv_file, index=False)
        
        data = self.data_loader.load_csv(csv_file)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, self.sample_data.shape)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_npy('nonexistent.npy')
    
    def test_batch_loading(self):
        """Test batch loading of multiple files"""
        # Create multiple files
        files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f'batch_{i}.npy')
            np.save(file_path, self.sample_data[i*100:(i+1)*100])
            files.append(file_path)
        
        batch_data = self.data_loader.load_batch(files)
        
        self.assertIsInstance(batch_data, list)
        self.assertEqual(len(batch_data), 3)
        for data in batch_data:
            self.assertEqual(data.shape[1], 10)
    
    def test_data_validation(self):
        """Test data validation"""
        # Test with valid data
        is_valid = self.data_loader.validate_data(self.sample_data)
        self.assertTrue(is_valid)
        
        # Test with invalid data (contains NaN)
        invalid_data = self.sample_data.copy()
        invalid_data[0, 0] = np.nan
        is_valid = self.data_loader.validate_data(invalid_data)
        self.assertFalse(is_valid)
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        processed_data = self.data_loader.preprocess(
            self.sample_data,
            normalize=True,
            remove_outliers=True
        )
        
        self.assertIsInstance(processed_data, np.ndarray)
        # Check normalization (mean should be close to 0, std close to 1)
        self.assertAlmostEqual(processed_data.mean(), 0, places=1)
        self.assertAlmostEqual(processed_data.std(), 1, places=1)
    
    @patch('src.data_ingestion.data_loader.requests.get')
    def test_download_data(self, mock_get):
        """Test downloading data from URL"""
        mock_response = Mock()
        mock_response.content = b'test data'
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        output_file = os.path.join(self.temp_dir, 'downloaded.dat')
        success = self.data_loader.download_data(
            'http://example.com/data.dat',
            output_file
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_file))


class TestStreamSimulator(unittest.TestCase):
    """Test cases for StreamSimulator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data = np.random.randn(100, 5)
        self.simulator = StreamSimulator(
            data=self.data,
            stream_rate=10,  # 10 records per second
            buffer_size=50
        )
    
    def test_initialization(self):
        """Test simulator initialization"""
        self.assertEqual(self.simulator.data.shape, self.data.shape)
        self.assertEqual(self.simulator.stream_rate, 10)
        self.assertEqual(self.simulator.buffer_size, 50)
        self.assertEqual(self.simulator.current_index, 0)
    
    def test_get_next_batch(self):
        """Test getting next batch of data"""
        batch = self.simulator.get_next_batch(batch_size=5)
        
        self.assertEqual(batch.shape[0], 5)
        self.assertEqual(batch.shape[1], 5)
        self.assertEqual(self.simulator.current_index, 5)
    
    def test_stream_continuity(self):
        """Test stream continuity and wraparound"""
        # Stream through all data
        total_streamed = 0
        for _ in range(25):  # Stream 25 batches of 5
            batch = self.simulator.get_next_batch(batch_size=5)
            total_streamed += len(batch)
        
        # Should have wrapped around
        self.assertEqual(total_streamed, 125)
        self.assertEqual(self.simulator.current_index, 25)
    
    def test_stream_with_noise(self):
        """Test streaming with added noise"""
        self.simulator.add_noise = True
        self.simulator.noise_level = 0.1
        
        batch = self.simulator.get_next_batch(batch_size=10)
        
        # Check that data has been modified (noise added)
        original_batch = self.data[:10]
        self.assertFalse(np.array_equal(batch, original_batch))
    
    def test_stream_with_anomalies(self):
        """Test injecting anomalies into stream"""
        self.simulator.inject_anomalies = True
        self.simulator.anomaly_rate = 0.2  # 20% anomalies
        
        batches_with_anomalies = 0
        for _ in range(10):
            batch, labels = self.simulator.get_next_batch_with_labels(batch_size=10)
            if np.any(labels == 1):
                batches_with_anomalies += 1
        
        # Should have some batches with anomalies
        self.assertGreater(batches_with_anomalies, 0)
    
    def test_buffer_management(self):
        """Test buffer management"""
        # Fill buffer
        for _ in range(10):
            self.simulator.add_to_buffer(np.random.randn(5))
        
        # Check buffer size
        self.assertLessEqual(len(self.simulator.buffer), self.simulator.buffer_size)
        
        # Get from buffer
        buffer_data = self.simulator.get_from_buffer(5)
        self.assertEqual(len(buffer_data), 5)
    
    @patch('time.sleep')
    def test_stream_rate_limiting(self, mock_sleep):
        """Test stream rate limiting"""
        self.simulator.start_streaming()
        
        # Let it stream for a bit
        for _ in range(5):
            self.simulator.get_next_batch(batch_size=1)
        
        # Check that sleep was called for rate limiting
        self.assertTrue(mock_sleep.called)
    
    def test_reset_stream(self):
        """Test resetting stream"""
        # Stream some data
        self.simulator.get_next_batch(batch_size=20)
        self.assertEqual(self.simulator.current_index, 20)
        
        # Reset
        self.simulator.reset()
        self.assertEqual(self.simulator.current_index, 0)
        self.assertEqual(len(self.simulator.buffer), 0)


class TestKafkaProducer(unittest.TestCase):
    """Test cases for KafkaProducer class"""
    
    @patch('src.data_ingestion.kafka_producer.KafkaProducer')
    def setUp(self, mock_kafka):
        """Set up test fixtures"""
        self.mock_producer = Mock()
        mock_kafka.return_value = self.mock_producer
        
        self.producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            topic='test-topic'
        )
    
    def test_initialization(self):
        """Test Kafka producer initialization"""
        self.assertEqual(self.producer.topic, 'test-topic')
        self.assertIsNotNone(self.producer.producer)
    
    def test_send_message(self):
        """Test sending single message"""
        message = {'sensor_id': 1, 'value': 23.5, 'timestamp': '2024-01-01T00:00:00'}
        
        self.producer.send_message(message)
        
        self.mock_producer.send.assert_called_once()
        call_args = self.mock_producer.send.call_args
        self.assertEqual(call_args[0][0], 'test-topic')
    
    def test_send_batch(self):
        """Test sending batch of messages"""
        messages = [
            {'sensor_id': 1, 'value': 23.5},
            {'sensor_id': 2, 'value': 24.1},
            {'sensor_id': 3, 'value': 22.9}
        ]
        
        self.producer.send_batch(messages)
        
        self.assertEqual(self.mock_producer.send.call_count, 3)
    
    def test_send_with_key(self):
        """Test sending message with key"""
        message = {'sensor_id': 1, 'value': 23.5}
        key = 'sensor_1'
        
        self.producer.send_message(message, key=key)
        
        call_args = self.mock_producer.send.call_args
        self.assertIn('key', call_args[1])
    
    def test_error_handling(self):
        """Test error handling in producer"""
        self.mock_producer.send.side_effect = Exception("Kafka error")
        
        message = {'sensor_id': 1, 'value': 23.5}
        
        with self.assertRaises(Exception):
            self.producer.send_message(message)
    
    def test_flush(self):
        """Test flushing producer"""
        self.producer.flush()
        self.mock_producer.flush.assert_called_once()
    
    def test_close(self):
        """Test closing producer"""
        self.producer.close()
        self.mock_producer.close.assert_called_once()
    
    def test_message_serialization(self):
        """Test message serialization"""
        message = {
            'sensor_id': 1,
            'values': [23.5, 24.1, 22.9],
            'timestamp': datetime.now()
        }
        
        serialized = self.producer.serialize_message(message)
        
        self.assertIsInstance(serialized, bytes)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized['sensor_id'], 1)


class TestKafkaConsumer(unittest.TestCase):
    """Test cases for KafkaConsumer class"""
    
    @patch('src.data_ingestion.kafka_consumer.KafkaConsumer')
    def setUp(self, mock_kafka):
        """Set up test fixtures"""
        self.mock_consumer = Mock()
        mock_kafka.return_value = self.mock_consumer
        
        self.consumer = KafkaConsumer(
            bootstrap_servers='localhost:9092',
            topics=['test-topic'],
            group_id='test-group'
        )
    
    def test_initialization(self):
        """Test Kafka consumer initialization"""
        self.assertEqual(self.consumer.topics, ['test-topic'])
        self.assertEqual(self.consumer.group_id, 'test-group')
        self.assertIsNotNone(self.consumer.consumer)
    
    def test_consume_messages(self):
        """Test consuming messages"""
        mock_message = Mock()
        mock_message.value = b'{"sensor_id": 1, "value": 23.5}'
        mock_message.topic = 'test-topic'
        mock_message.partition = 0
        mock_message.offset = 100
        
        self.mock_consumer.poll.return_value = {
            'test-topic': [mock_message]
        }
        
        messages = self.consumer.consume_messages(max_messages=1)
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['sensor_id'], 1)
    
    def test_consume_with_timeout(self):
        """Test consuming with timeout"""
        self.mock_consumer.poll.return_value = {}
        
        messages = self.consumer.consume_messages(timeout=1.0)
        
        self.assertEqual(len(messages), 0)
        self.mock_consumer.poll.assert_called_with(timeout_ms=1000)
    
    def test_commit_offsets(self):
        """Test committing offsets"""
        self.consumer.commit()
        self.mock_consumer.commit.assert_called_once()
    
    def test_seek_to_beginning(self):
        """Test seeking to beginning"""
        self.consumer.seek_to_beginning()
        self.mock_consumer.seek_to_beginning.assert_called_once()
    
    def test_seek_to_end(self):
        """Test seeking to end"""
        self.consumer.seek_to_end()
        self.mock_consumer.seek_to_end.assert_called_once()
    
    def test_get_current_offsets(self):
        """Test getting current offsets"""
        self.mock_consumer.position.return_value = 100
        
        position = self.consumer.get_position('test-topic', 0)
        
        self.assertEqual(position, 100)
    
    def test_close(self):
        """Test closing consumer"""
        self.consumer.close()
        self.mock_consumer.close.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in consumer"""
        self.mock_consumer.poll.side_effect = Exception("Kafka error")
        
        with self.assertRaises(Exception):
            self.consumer.consume_messages()


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class"""
    
    @patch('src.data_ingestion.database_manager.psycopg2.connect')
    def setUp(self, mock_connect):
        """Set up test fixtures"""
        self.mock_connection = Mock()
        self.mock_cursor = Mock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        mock_connect.return_value = self.mock_connection
        
        self.db_manager = DatabaseManager(
            host='localhost',
            port=5432,
            database='test_db',
            user='test_user',
            password='test_pass'
        )
    
    def test_initialization(self):
        """Test database manager initialization"""
        self.assertIsNotNone(self.db_manager.connection)
        self.assertEqual(self.db_manager.database, 'test_db')
    
    def test_execute_query(self):
        """Test executing SELECT query"""
        self.mock_cursor.fetchall.return_value = [
            (1, 'sensor_1', 23.5),
            (2, 'sensor_2', 24.1)
        ]
        self.mock_cursor.description = [
            ('id',), ('name',), ('value',)
        ]
        
        result = self.db_manager.execute_query(
            "SELECT * FROM sensors WHERE value > %s",
            [20.0]
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.mock_cursor.execute.assert_called_once()
    
    def test_execute_insert(self):
        """Test executing INSERT query"""
        query = "INSERT INTO sensors (name, value) VALUES (%s, %s)"
        params = ['sensor_1', 23.5]
        
        self.db_manager.execute_insert(query, params)
        
        self.mock_cursor.execute.assert_called_with(query, params)
        self.mock_connection.commit.assert_called_once()
    
    def test_execute_batch_insert(self):
        """Test batch insert"""
        query = "INSERT INTO sensors (name, value) VALUES (%s, %s)"
        data = [
            ('sensor_1', 23.5),
            ('sensor_2', 24.1),
            ('sensor_3', 22.9)
        ]
        
        self.db_manager.execute_batch_insert(query, data)
        
        self.mock_cursor.executemany.assert_called_with(query, data)
        self.mock_connection.commit.assert_called_once()
    
    def test_create_table(self):
        """Test creating table"""
        self.db_manager.create_telemetry_table()
        
        self.mock_cursor.execute.assert_called()
        create_table_query = self.mock_cursor.execute.call_args[0][0]
        self.assertIn('CREATE TABLE', create_table_query)
    
    def test_insert_telemetry_data(self):
        """Test inserting telemetry data"""
        data = pd.DataFrame({
            'equipment_id': ['PUMP_001', 'PUMP_002'],
            'timestamp': [datetime.now(), datetime.now()],
            'temperature': [75.5, 76.2],
            'pressure': [150.0, 152.3]
        })
        
        self.db_manager.insert_telemetry_data(data)
        
        self.mock_cursor.executemany.assert_called_once()
    
    def test_query_time_range(self):
        """Test querying data in time range"""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        self.mock_cursor.fetchall.return_value = []
        self.mock_cursor.description = [('id',), ('timestamp',), ('value',)]
        
        result = self.db_manager.query_time_range(
            'telemetry',
            start_time,
            end_time
        )
        
        query = self.mock_cursor.execute.call_args[0][0]
        self.assertIn('BETWEEN', query)
    
    def test_connection_retry(self):
        """Test connection retry on failure"""
        self.mock_connection.cursor.side_effect = [
            Exception("Connection lost"),
            self.mock_cursor
        ]
        
        # Should retry and succeed
        result = self.db_manager.execute_query("SELECT 1")
        
        self.assertEqual(self.mock_connection.cursor.call_count, 2)
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error"""
        self.mock_cursor.execute.side_effect = Exception("Insert failed")
        
        with self.assertRaises(Exception):
            self.db_manager.execute_insert("INSERT INTO test VALUES (1)")
        
        self.mock_connection.rollback.assert_called_once()
    
    def test_close_connection(self):
        """Test closing database connection"""
        self.db_manager.close()
        
        self.mock_cursor.close.assert_called_once()
        self.mock_connection.close.assert_called_once()
    
    def test_create_hypertable(self):
        """Test creating TimescaleDB hypertable"""
        self.db_manager.create_hypertable('telemetry', 'timestamp')
        
        query = self.mock_cursor.execute.call_args[0][0]
        self.assertIn('create_hypertable', query)
    
    def test_connection_pool(self):
        """Test connection pooling"""
        # Test that connection pool is properly managed
        pool_size = 5
        self.db_manager.initialize_pool(pool_size)
        
        connections = []
        for _ in range(pool_size):
            conn = self.db_manager.get_connection_from_pool()
            connections.append(conn)
        
        # All connections should be unique
        self.assertEqual(len(set(connections)), pool_size)


class TestIntegration(unittest.TestCase):
    """Integration tests for data ingestion pipeline"""
    
    @patch('src.data_ingestion.kafka_producer.KafkaProducer')
    @patch('src.data_ingestion.kafka_consumer.KafkaConsumer')
    @patch('src.data_ingestion.database_manager.psycopg2.connect')
    def test_end_to_end_pipeline(self, mock_db, mock_consumer, mock_producer):
        """Test end-to-end data ingestion pipeline"""
        # Setup mocks
        mock_db_conn = Mock()
        mock_db_cursor = Mock()
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db.return_value = mock_db_conn
        
        # Create pipeline components
        data = np.random.randn(100, 5)
        simulator = StreamSimulator(data, stream_rate=10)
        producer = KafkaProducer('localhost:9092', 'telemetry')
        consumer = KafkaConsumer('localhost:9092', ['telemetry'], 'test-group')
        db_manager = DatabaseManager('localhost', 5432, 'test_db', 'user', 'pass')
        
        # Simulate streaming
        batch = simulator.get_next_batch(10)
        
        # Send to Kafka
        for row in batch:
            message = {
                'timestamp': datetime.now().isoformat(),
                'values': row.tolist()
            }
            producer.send_message(message)
        
        # Consume from Kafka
        mock_consumer_instance = mock_consumer.return_value
        mock_message = Mock()
        mock_message.value = json.dumps(message).encode()
        mock_consumer_instance.poll.return_value = {'telemetry': [mock_message]}
        
        messages = consumer.consume_messages(max_messages=10)
        
        # Store in database
        for msg in messages:
            db_manager.insert_telemetry_data(pd.DataFrame([msg]))
        
        # Verify pipeline execution
        self.assertEqual(len(messages), 1)
        mock_db_cursor.executemany.assert_called()


class TestDataQuality(unittest.TestCase):
    """Test cases for data quality checks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
    
    def test_check_missing_values(self):
        """Test checking for missing values"""
        # Data with no missing values
        good_data = np.random.randn(100, 5)
        has_missing = self.data_loader.check_missing_values(good_data)
        self.assertFalse(has_missing)
        
        # Data with missing values
        bad_data = good_data.copy()
        bad_data[0, 0] = np.nan
        has_missing = self.data_loader.check_missing_values(bad_data)
        self.assertTrue(has_missing)
    
    def test_check_data_range(self):
        """Test checking data range"""
        data = np.random.randn(100, 5) * 10
        
        # Check if data is within expected range
        in_range = self.data_loader.check_data_range(data, min_val=-50, max_val=50)
        self.assertTrue(in_range)
        
        # Check with data outside range
        data[0, 0] = 100
        in_range = self.data_loader.check_data_range(data, min_val=-50, max_val=50)
        self.assertFalse(in_range)
    
    def test_check_data_consistency(self):
        """Test checking data consistency"""
        # Consistent data shape
        data1 = np.random.randn(100, 5)
        data2 = np.random.randn(100, 5)
        is_consistent = self.data_loader.check_consistency([data1, data2])
        self.assertTrue(is_consistent)
        
        # Inconsistent data shape
        data3 = np.random.randn(100, 6)
        is_consistent = self.data_loader.check_consistency([data1, data3])
        self.assertFalse(is_consistent)
    
    def test_data_statistics(self):
        """Test calculating data statistics"""
        data = np.random.randn(1000, 5)
        stats = self.data_loader.calculate_statistics(data)
        
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('percentiles', stats)
        
        # Check statistics are reasonable
        self.assertAlmostEqual(stats['mean'], 0, places=1)
        self.assertAlmostEqual(stats['std'], 1, places=1)


class TestPerformance(unittest.TestCase):
    """Performance tests for data ingestion"""
    
    def test_large_file_loading(self):
        """Test loading large files"""
        # Create large test file
        large_data = np.random.randn(100000, 50)
        temp_file = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(temp_file.name, large_data)
        
        try:
            data_loader = DataLoader()
            
            # Measure loading time
            import time
            start_time = time.time()
            loaded_data = data_loader.load_npy(temp_file.name)
            load_time = time.time() - start_time
            
            # Should load in reasonable time (< 5 seconds)
            self.assertLess(load_time, 5.0)
            self.assertEqual(loaded_data.shape, large_data.shape)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_streaming_throughput(self):
        """Test streaming throughput"""
        data = np.random.randn(10000, 10)
        simulator = StreamSimulator(data, stream_rate=1000)
        
        # Measure throughput
        import time
        start_time = time.time()
        total_records = 0
        
        while time.time() - start_time < 1.0:  # Stream for 1 second
            batch = simulator.get_next_batch(100)
            total_records += len(batch)
        
        # Should achieve close to target rate
        self.assertGreater(total_records, 800)  # At least 80% of target
    
    @patch('src.data_ingestion.database_manager.psycopg2.connect')
    def test_batch_insert_performance(self, mock_connect):
        """Test batch insert performance"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        db_manager = DatabaseManager('localhost', 5432, 'test_db', 'user', 'pass')
        
        # Create large batch
        large_batch = [
            (f'sensor_{i}', float(i), datetime.now())
            for i in range(10000)
        ]
        
        # Measure insert time
        import time
        start_time = time.time()
        db_manager.execute_batch_insert(
            "INSERT INTO test VALUES (%s, %s, %s)",
            large_batch
        )
        insert_time = time.time() - start_time
        
        # Should complete quickly (< 1 second for mock)
        self.assertLess(insert_time, 1.0)


if __name__ == '__main__':
    unittest.main()