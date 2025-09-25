"""
Data Flow Integration Testing Suite

Tests data flow integrity across the entire IoT Predictive Maintenance System:
- Data ingestion consistency
- Cross-phase data transformations
- Data quality maintenance
- Real-time streaming validation
- Database integration testing
"""

import unittest
import sys
import os
import asyncio
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import concurrent.futures
import threading
import queue
import tempfile
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import core system components
try:
    from src.data_ingestion.nasa_data_service import NASADataService
    from src.preprocessing.data_preprocessor import DataPreprocessor
    from src.anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
    from src.forecasting.transformer_forecaster import TransformerForecaster
    from src.utils.database_manager import DatabaseManager
    from src.utils.config_manager import ConfigManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Warning: Could not import core system components: {e}")


@dataclass
class DataFlowTestResult:
    """Container for data flow test results"""
    test_name: str
    stage_name: str
    data_integrity_score: float
    transformation_accuracy: float
    data_loss_percentage: float
    processing_latency: float
    throughput_rate: float
    error_rate: float
    timestamp: datetime


class TestDataFlowIntegration(unittest.TestCase):
    """Data flow integration testing across all system phases"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for data flow testing"""
        cls.logger = get_logger(__name__)
        cls.config_manager = ConfigManager()
        cls.test_results = []

        # Test configuration for data flow
        cls.test_config = {
            'data_volume': {
                'sensors_per_test': 40,
                'data_points_per_sensor': 1000,
                'streaming_duration': 60,  # seconds
                'batch_size': 100
            },
            'data_quality': {
                'min_integrity_score': 0.95,
                'max_data_loss': 0.05,  # 5%
                'max_transformation_error': 0.02  # 2%
            },
            'performance': {
                'max_processing_latency': 100,  # ms
                'min_throughput_rate': 500,  # data points per second
                'max_error_rate': 0.01  # 1%
            }
        }

        # Initialize components
        cls._initialize_data_flow_components()

    @classmethod
    def _initialize_data_flow_components(cls):
        """Initialize components for data flow testing"""
        try:
            cls.nasa_data_service = NASADataService()
            cls.data_preprocessor = DataPreprocessor()
            cls.ensemble_detector = EnsembleAnomalyDetector()
            cls.transformer_forecaster = TransformerForecaster()
            cls.database_manager = DatabaseManager()

            cls.logger.info("Data flow components initialized successfully")

        except Exception as e:
            cls.logger.error(f"Failed to initialize data flow components: {e}")
            raise

    def setUp(self):
        """Set up individual test"""
        self.test_start_time = time.time()
        self.test_data = self._generate_test_dataset()

    def tearDown(self):
        """Clean up individual test"""
        execution_time = time.time() - self.test_start_time
        self.logger.info(f"Data flow test {self._testMethodName} completed in {execution_time:.2f}s")

    def _generate_test_dataset(self) -> Dict[str, Any]:
        """Generate comprehensive test dataset for data flow testing"""
        dataset = {
            'raw_sensor_data': {},
            'metadata': {},
            'quality_markers': {}
        }

        sensors_count = self.test_config['data_volume']['sensors_per_test']
        points_per_sensor = self.test_config['data_volume']['data_points_per_sensor']

        for i in range(sensors_count):
            sensor_id = f"test_sensor_{i+1:03d}"

            # Generate realistic time series data
            base_trend = np.linspace(0.4, 0.6, points_per_sensor)
            seasonal_component = 0.1 * np.sin(2 * np.pi * np.arange(points_per_sensor) / 100)
            noise = np.random.normal(0, 0.05, points_per_sensor)
            anomaly_injection = np.zeros(points_per_sensor)

            # Inject controlled anomalies (5% of data points)
            anomaly_indices = np.random.choice(
                points_per_sensor, size=int(points_per_sensor * 0.05), replace=False
            )
            anomaly_injection[anomaly_indices] = np.random.uniform(0.5, 1.0, len(anomaly_indices))

            sensor_values = base_trend + seasonal_component + noise + anomaly_injection

            # Ensure values are within valid range [0, 1]
            sensor_values = np.clip(sensor_values, 0, 1)

            dataset['raw_sensor_data'][sensor_id] = {
                'values': sensor_values,
                'timestamps': pd.date_range(
                    start=datetime.now() - timedelta(hours=24),
                    periods=points_per_sensor,
                    freq='1min'
                ),
                'sensor_type': 'temperature' if i % 2 == 0 else 'pressure',
                'location': f"location_{(i // 5) + 1}",
                'equipment_id': f"equipment_{(i // 10) + 1}"
            }

            # Create quality markers for validation
            dataset['quality_markers'][sensor_id] = {
                'data_checksum': hashlib.md5(sensor_values.tobytes()).hexdigest(),
                'anomaly_indices': anomaly_indices.tolist(),
                'data_statistics': {
                    'mean': float(np.mean(sensor_values)),
                    'std': float(np.std(sensor_values)),
                    'min': float(np.min(sensor_values)),
                    'max': float(np.max(sensor_values))
                }
            }

            dataset['metadata'][sensor_id] = {
                'creation_timestamp': datetime.now(),
                'data_source': 'test_generator',
                'quality_level': 'high',
                'expected_anomalies': len(anomaly_indices)
            }

        return dataset

    def _calculate_data_integrity(self, original_data: np.ndarray, processed_data: np.ndarray) -> float:
        """Calculate data integrity score between original and processed data"""
        if len(original_data) != len(processed_data):
            return 0.0

        # Calculate correlation coefficient
        correlation = np.corrcoef(original_data, processed_data)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Calculate mean squared error (normalized)
        mse = np.mean((original_data - processed_data) ** 2)
        normalized_mse = mse / (np.var(original_data) + 1e-8)

        # Combine metrics for integrity score
        integrity_score = correlation * (1 - normalized_mse)
        return max(0.0, min(1.0, integrity_score))

    def test_data_ingestion_integrity(self):
        """Test data integrity during ingestion process"""
        test_start = time.time()

        try:
            integrity_scores = []
            processing_times = []

            for sensor_id, sensor_data in self.test_data['raw_sensor_data'].items():
                # Test data ingestion
                ingestion_start = time.time()

                ingested_data = self.nasa_data_service.process_sensor_stream(
                    sensor_id, sensor_data['values']
                )

                ingestion_time = time.time() - ingestion_start
                processing_times.append(ingestion_time)

                # Calculate integrity
                if ingested_data is not None and len(ingested_data) > 0:
                    integrity_score = self._calculate_data_integrity(
                        sensor_data['values'], ingested_data
                    )
                    integrity_scores.append(integrity_score)

                    # Validate against quality markers
                    original_checksum = self.test_data['quality_markers'][sensor_id]['data_checksum']
                    new_checksum = hashlib.md5(ingested_data.tobytes()).hexdigest()

                    # Data should be preserved or predictably transformed
                    self.assertIsNotNone(ingested_data, f"Ingested data should not be None for {sensor_id}")
                    self.assertEqual(
                        len(ingested_data), len(sensor_data['values']),
                        f"Data length should be preserved for {sensor_id}"
                    )

            # Validate overall integrity
            avg_integrity = np.mean(integrity_scores) if integrity_scores else 0.0
            avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms

            self.assertGreaterEqual(
                avg_integrity, self.test_config['data_quality']['min_integrity_score'],
                f"Average integrity score {avg_integrity:.3f} below minimum {self.test_config['data_quality']['min_integrity_score']}"
            )

            self.assertLessEqual(
                avg_processing_time, self.test_config['performance']['max_processing_latency'],
                f"Average processing time {avg_processing_time:.1f}ms exceeds limit"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="data_ingestion_integrity",
                stage_name="ingestion",
                data_integrity_score=avg_integrity,
                transformation_accuracy=avg_integrity,
                data_loss_percentage=0.0,
                processing_latency=avg_processing_time,
                throughput_rate=len(self.test_data['raw_sensor_data']) / sum(processing_times),
                error_rate=0.0,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Data ingestion integrity: {avg_integrity:.3f}, "
                           f"avg processing time: {avg_processing_time:.1f}ms")

        except Exception as e:
            self.fail(f"Data ingestion integrity test failed: {e}")

    def test_preprocessing_transformation_accuracy(self):
        """Test preprocessing transformation accuracy and data preservation"""
        test_start = time.time()

        try:
            transformation_accuracies = []
            data_loss_percentages = []
            processing_times = []

            for sensor_id, sensor_data in self.test_data['raw_sensor_data'].items():
                # Test preprocessing
                preprocessing_start = time.time()

                preprocessed_data = self.data_preprocessor.preprocess_sensor_data(
                    sensor_data['values'], sensor_id
                )

                preprocessing_time = time.time() - preprocessing_start
                processing_times.append(preprocessing_time)

                if preprocessed_data is not None:
                    # Calculate transformation accuracy
                    original_stats = self.test_data['quality_markers'][sensor_id]['data_statistics']

                    if len(preprocessed_data) > 0:
                        processed_stats = {
                            'mean': float(np.mean(preprocessed_data)),
                            'std': float(np.std(preprocessed_data)),
                            'min': float(np.min(preprocessed_data)),
                            'max': float(np.max(preprocessed_data))
                        }

                        # Calculate statistical preservation
                        mean_preservation = 1 - abs(original_stats['mean'] - processed_stats['mean']) / (original_stats['mean'] + 1e-8)
                        std_preservation = 1 - abs(original_stats['std'] - processed_stats['std']) / (original_stats['std'] + 1e-8)

                        transformation_accuracy = (mean_preservation + std_preservation) / 2
                        transformation_accuracies.append(max(0.0, transformation_accuracy))

                        # Calculate data loss
                        data_loss = max(0, len(sensor_data['values']) - len(preprocessed_data)) / len(sensor_data['values'])
                        data_loss_percentages.append(data_loss)

                        # Validate preprocessing quality
                        self.assertGreater(
                            len(preprocessed_data), 0,
                            f"Preprocessed data should not be empty for {sensor_id}"
                        )

                        self.assertLessEqual(
                            data_loss, self.test_config['data_quality']['max_data_loss'],
                            f"Data loss {data_loss:.3f} exceeds maximum for {sensor_id}"
                        )

            # Validate overall transformation accuracy
            avg_transformation_accuracy = np.mean(transformation_accuracies) if transformation_accuracies else 0.0
            avg_data_loss = np.mean(data_loss_percentages) if data_loss_percentages else 1.0
            avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms

            self.assertGreaterEqual(
                avg_transformation_accuracy, 1 - self.test_config['data_quality']['max_transformation_error'],
                f"Transformation accuracy {avg_transformation_accuracy:.3f} below threshold"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="preprocessing_transformation_accuracy",
                stage_name="preprocessing",
                data_integrity_score=avg_transformation_accuracy,
                transformation_accuracy=avg_transformation_accuracy,
                data_loss_percentage=avg_data_loss * 100,
                processing_latency=avg_processing_time,
                throughput_rate=len(self.test_data['raw_sensor_data']) / sum(processing_times),
                error_rate=0.0,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Preprocessing accuracy: {avg_transformation_accuracy:.3f}, "
                           f"data loss: {avg_data_loss:.3%}")

        except Exception as e:
            self.fail(f"Preprocessing transformation accuracy test failed: {e}")

    def test_anomaly_detection_data_flow(self):
        """Test data flow through anomaly detection pipeline"""
        test_start = time.time()

        try:
            detection_accuracies = []
            processing_times = []
            anomaly_preservation_scores = []

            for sensor_id, sensor_data in list(self.test_data['raw_sensor_data'].items())[:20]:  # Test subset
                # First preprocess the data
                preprocessed_data = self.data_preprocessor.preprocess_sensor_data(
                    sensor_data['values'], sensor_id
                )

                if preprocessed_data is not None and len(preprocessed_data) > 0:
                    # Test anomaly detection
                    detection_start = time.time()

                    anomaly_results = self.ensemble_detector.detect_anomalies(
                        preprocessed_data, sensor_id
                    )

                    detection_time = time.time() - detection_start
                    processing_times.append(detection_time)

                    if anomaly_results and 'anomaly_indices' in anomaly_results:
                        # Validate anomaly detection accuracy
                        expected_anomalies = set(self.test_data['quality_markers'][sensor_id]['anomaly_indices'])
                        detected_anomalies = set(anomaly_results['anomaly_indices'])

                        # Calculate precision and recall
                        true_positives = len(expected_anomalies.intersection(detected_anomalies))
                        precision = true_positives / len(detected_anomalies) if detected_anomalies else 0
                        recall = true_positives / len(expected_anomalies) if expected_anomalies else 0

                        # F1 score as detection accuracy measure
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        detection_accuracies.append(f1_score)

                        # Check data preservation through detection
                        integrity_score = self._calculate_data_integrity(
                            preprocessed_data, anomaly_results.get('processed_data', preprocessed_data)
                        )
                        anomaly_preservation_scores.append(integrity_score)

                        # Validate detection results
                        self.assertIn('anomaly_count', anomaly_results, f"Missing anomaly_count for {sensor_id}")
                        self.assertIn('anomaly_score', anomaly_results, f"Missing anomaly_score for {sensor_id}")
                        self.assertGreaterEqual(
                            anomaly_results['anomaly_score'], 0.0,
                            f"Anomaly score should be >= 0 for {sensor_id}"
                        )
                        self.assertLessEqual(
                            anomaly_results['anomaly_score'], 1.0,
                            f"Anomaly score should be <= 1 for {sensor_id}"
                        )

            # Validate overall anomaly detection performance
            avg_detection_accuracy = np.mean(detection_accuracies) if detection_accuracies else 0.0
            avg_preservation_score = np.mean(anomaly_preservation_scores) if anomaly_preservation_scores else 0.0
            avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms

            self.assertGreaterEqual(
                avg_preservation_score, self.test_config['data_quality']['min_integrity_score'],
                f"Data preservation through anomaly detection {avg_preservation_score:.3f} below threshold"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="anomaly_detection_data_flow",
                stage_name="anomaly_detection",
                data_integrity_score=avg_preservation_score,
                transformation_accuracy=avg_detection_accuracy,
                data_loss_percentage=0.0,
                processing_latency=avg_processing_time,
                throughput_rate=len(processing_times) / sum(processing_times) if processing_times else 0,
                error_rate=0.0,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Anomaly detection: accuracy={avg_detection_accuracy:.3f}, "
                           f"preservation={avg_preservation_score:.3f}")

        except Exception as e:
            self.fail(f"Anomaly detection data flow test failed: {e}")

    def test_forecasting_data_continuity(self):
        """Test data continuity through forecasting pipeline"""
        test_start = time.time()

        try:
            forecast_accuracies = []
            processing_times = []
            data_continuity_scores = []

            for sensor_id, sensor_data in list(self.test_data['raw_sensor_data'].items())[:10]:  # Test subset
                # Prepare data for forecasting
                preprocessed_data = self.data_preprocessor.preprocess_sensor_data(
                    sensor_data['values'], sensor_id
                )

                if preprocessed_data is not None and len(preprocessed_data) >= 100:
                    # Split data for training and validation
                    train_data = preprocessed_data[:-50]  # Use all but last 50 points for training
                    validation_data = preprocessed_data[-50:]  # Last 50 points for validation

                    # Test forecasting
                    forecasting_start = time.time()

                    forecast_results = self.transformer_forecaster.forecast_sensor_values(
                        train_data, forecast_horizon=len(validation_data)
                    )

                    forecasting_time = time.time() - forecasting_start
                    processing_times.append(forecasting_time)

                    if forecast_results and 'predictions' in forecast_results:
                        predictions = forecast_results['predictions']

                        # Calculate forecasting accuracy
                        if len(predictions) == len(validation_data):
                            mse = np.mean((predictions - validation_data) ** 2)
                            rmse = np.sqrt(mse)
                            normalized_rmse = rmse / (np.std(validation_data) + 1e-8)
                            forecast_accuracy = max(0.0, 1.0 - normalized_rmse)
                            forecast_accuracies.append(forecast_accuracy)

                            # Check data continuity
                            # Last training point should connect smoothly to first forecast point
                            continuity_gap = abs(train_data[-1] - predictions[0])
                            max_expected_gap = 2 * np.std(train_data[-10:])  # Based on recent volatility
                            continuity_score = max(0.0, 1.0 - continuity_gap / max_expected_gap)
                            data_continuity_scores.append(continuity_score)

                            # Validate forecast results
                            self.assertIn('confidence', forecast_results, f"Missing confidence for {sensor_id}")
                            self.assertGreater(
                                forecast_results['confidence'], 0.0,
                                f"Forecast confidence should be > 0 for {sensor_id}"
                            )
                            self.assertLessEqual(
                                forecast_results['confidence'], 1.0,
                                f"Forecast confidence should be <= 1 for {sensor_id}"
                            )

            # Validate overall forecasting performance
            avg_forecast_accuracy = np.mean(forecast_accuracies) if forecast_accuracies else 0.0
            avg_continuity_score = np.mean(data_continuity_scores) if data_continuity_scores else 0.0
            avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms

            self.assertGreaterEqual(
                avg_continuity_score, 0.7,  # At least 70% continuity
                f"Data continuity through forecasting {avg_continuity_score:.3f} below threshold"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="forecasting_data_continuity",
                stage_name="forecasting",
                data_integrity_score=avg_continuity_score,
                transformation_accuracy=avg_forecast_accuracy,
                data_loss_percentage=0.0,
                processing_latency=avg_processing_time,
                throughput_rate=len(processing_times) / sum(processing_times) if processing_times else 0,
                error_rate=0.0,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Forecasting: accuracy={avg_forecast_accuracy:.3f}, "
                           f"continuity={avg_continuity_score:.3f}")

        except Exception as e:
            self.fail(f"Forecasting data continuity test failed: {e}")

    def test_database_integration_consistency(self):
        """Test database integration and data consistency"""
        test_start = time.time()

        try:
            storage_success_rates = []
            retrieval_accuracy_scores = []
            processing_times = []

            for sensor_id, sensor_data in list(self.test_data['raw_sensor_data'].items())[:15]:  # Test subset
                # Preprocess data first
                preprocessed_data = self.data_preprocessor.preprocess_sensor_data(
                    sensor_data['values'], sensor_id
                )

                if preprocessed_data is not None and len(preprocessed_data) > 0:
                    # Test database storage
                    storage_start = time.time()

                    storage_success = self.database_manager.store_sensor_data(
                        sensor_id, preprocessed_data
                    )

                    storage_time = time.time() - storage_start

                    if storage_success:
                        # Test data retrieval
                        retrieval_start = time.time()

                        retrieved_data = self.database_manager.retrieve_sensor_data(
                            sensor_id, limit=len(preprocessed_data)
                        )

                        retrieval_time = time.time() - retrieval_start
                        total_time = storage_time + retrieval_time
                        processing_times.append(total_time)

                        storage_success_rates.append(1.0)

                        if retrieved_data is not None and len(retrieved_data) > 0:
                            # Calculate retrieval accuracy
                            min_length = min(len(preprocessed_data), len(retrieved_data))
                            if min_length > 0:
                                retrieval_accuracy = self._calculate_data_integrity(
                                    preprocessed_data[:min_length],
                                    retrieved_data[:min_length]
                                )
                                retrieval_accuracy_scores.append(retrieval_accuracy)

                                # Validate database consistency
                                self.assertGreater(
                                    retrieval_accuracy, 0.95,
                                    f"Database retrieval accuracy {retrieval_accuracy:.3f} too low for {sensor_id}"
                                )
                    else:
                        storage_success_rates.append(0.0)

            # Validate overall database performance
            avg_storage_success = np.mean(storage_success_rates) if storage_success_rates else 0.0
            avg_retrieval_accuracy = np.mean(retrieval_accuracy_scores) if retrieval_accuracy_scores else 0.0
            avg_processing_time = np.mean(processing_times) * 1000  # Convert to ms

            self.assertGreaterEqual(
                avg_storage_success, 0.95,
                f"Database storage success rate {avg_storage_success:.3f} below threshold"
            )

            self.assertGreaterEqual(
                avg_retrieval_accuracy, 0.95,
                f"Database retrieval accuracy {avg_retrieval_accuracy:.3f} below threshold"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="database_integration_consistency",
                stage_name="database",
                data_integrity_score=avg_retrieval_accuracy,
                transformation_accuracy=avg_storage_success,
                data_loss_percentage=(1 - avg_storage_success) * 100,
                processing_latency=avg_processing_time,
                throughput_rate=len(processing_times) / sum(processing_times) if processing_times else 0,
                error_rate=1 - avg_storage_success,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Database integration: storage={avg_storage_success:.3f}, "
                           f"retrieval={avg_retrieval_accuracy:.3f}")

        except Exception as e:
            self.fail(f"Database integration consistency test failed: {e}")

    def test_streaming_data_flow_performance(self):
        """Test streaming data flow performance under concurrent load"""
        test_start = time.time()

        try:
            streaming_results = []
            error_count = 0
            total_processed = 0

            # Configure streaming test
            stream_duration = 30  # seconds
            sensors_to_stream = list(self.test_data['raw_sensor_data'].keys())[:10]
            batch_size = self.test_config['data_volume']['batch_size']

            def stream_sensor_data(sensor_id: str, sensor_data: Dict) -> Dict:
                """Simulate streaming data processing for a single sensor"""
                results = {
                    'sensor_id': sensor_id,
                    'batches_processed': 0,
                    'total_points': 0,
                    'processing_times': [],
                    'errors': 0
                }

                values = sensor_data['values']
                timestamps = sensor_data['timestamps']

                # Process data in batches
                for i in range(0, len(values), batch_size):
                    batch_values = values[i:i + batch_size]
                    batch_timestamps = timestamps[i:i + batch_size]

                    try:
                        batch_start = time.time()

                        # Simulate streaming pipeline
                        # 1. Ingest batch
                        ingested = self.nasa_data_service.process_sensor_stream(sensor_id, batch_values)

                        # 2. Preprocess batch
                        if ingested is not None:
                            preprocessed = self.data_preprocessor.preprocess_sensor_data(ingested, sensor_id)

                            # 3. Store batch
                            if preprocessed is not None:
                                stored = self.database_manager.store_sensor_data(sensor_id, preprocessed)

                                if stored:
                                    results['batches_processed'] += 1
                                    results['total_points'] += len(batch_values)

                        batch_time = time.time() - batch_start
                        results['processing_times'].append(batch_time)

                    except Exception as e:
                        results['errors'] += 1
                        self.logger.warning(f"Batch processing error for {sensor_id}: {e}")

                    # Simulate real-time delay
                    time.sleep(0.1)

                return results

            # Execute concurrent streaming
            streaming_start = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for sensor_id in sensors_to_stream:
                    sensor_data = self.test_data['raw_sensor_data'][sensor_id]
                    future = executor.submit(stream_sensor_data, sensor_id, sensor_data)
                    futures.append(future)

                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        streaming_results.append(result)
                        total_processed += result['total_points']
                        error_count += result['errors']
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"Streaming future failed: {e}")

            streaming_time = time.time() - streaming_start

            # Calculate streaming performance metrics
            total_batches = sum(result['batches_processed'] for result in streaming_results)
            avg_batch_time = np.mean([
                np.mean(result['processing_times'])
                for result in streaming_results
                if result['processing_times']
            ]) if streaming_results else 0

            throughput_rate = total_processed / streaming_time if streaming_time > 0 else 0
            error_rate = error_count / max(1, total_processed)

            # Validate streaming performance
            self.assertGreaterEqual(
                throughput_rate, self.test_config['performance']['min_throughput_rate'],
                f"Streaming throughput {throughput_rate:.1f} below minimum {self.test_config['performance']['min_throughput_rate']}"
            )

            self.assertLessEqual(
                error_rate, self.test_config['performance']['max_error_rate'],
                f"Streaming error rate {error_rate:.3f} exceeds maximum {self.test_config['performance']['max_error_rate']}"
            )

            # Record test result
            result = DataFlowTestResult(
                test_name="streaming_data_flow_performance",
                stage_name="streaming",
                data_integrity_score=1.0 - error_rate,
                transformation_accuracy=1.0 - error_rate,
                data_loss_percentage=error_rate * 100,
                processing_latency=avg_batch_time * 1000,  # Convert to ms
                throughput_rate=throughput_rate,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Streaming performance: throughput={throughput_rate:.1f} pts/s, "
                           f"error_rate={error_rate:.3%}")

        except Exception as e:
            self.fail(f"Streaming data flow performance test failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Generate data flow integration report"""
        cls._generate_data_flow_report()

    @classmethod
    def _generate_data_flow_report(cls):
        """Generate comprehensive data flow integration report"""
        try:
            report_data = {
                'test_suite': 'Data Flow Integration Testing',
                'execution_timestamp': datetime.now().isoformat(),
                'total_tests': len(cls.test_results),
                'successful_tests': len([r for r in cls.test_results if r.data_integrity_score >= 0.9]),
                'performance_summary': {
                    'avg_data_integrity': np.mean([r.data_integrity_score for r in cls.test_results]),
                    'avg_transformation_accuracy': np.mean([r.transformation_accuracy for r in cls.test_results]),
                    'avg_data_loss_percentage': np.mean([r.data_loss_percentage for r in cls.test_results]),
                    'avg_processing_latency': np.mean([r.processing_latency for r in cls.test_results]),
                    'avg_throughput_rate': np.mean([r.throughput_rate for r in cls.test_results]),
                    'avg_error_rate': np.mean([r.error_rate for r in cls.test_results])
                },
                'stage_performance': {
                    stage: {
                        'tests_count': len([r for r in cls.test_results if r.stage_name == stage]),
                        'avg_integrity': np.mean([r.data_integrity_score for r in cls.test_results if r.stage_name == stage]),
                        'avg_throughput': np.mean([r.throughput_rate for r in cls.test_results if r.stage_name == stage])
                    }
                    for stage in set(r.stage_name for r in cls.test_results)
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'stage_name': r.stage_name,
                        'data_integrity_score': r.data_integrity_score,
                        'transformation_accuracy': r.transformation_accuracy,
                        'data_loss_percentage': r.data_loss_percentage,
                        'processing_latency': r.processing_latency,
                        'throughput_rate': r.throughput_rate,
                        'error_rate': r.error_rate,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in cls.test_results
                ]
            }

            # Save data flow report
            report_path = Path(__file__).parent.parent / "data_flow_integration_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            cls.logger.info(f"Data flow integration report saved to {report_path}")

        except Exception as e:
            cls.logger.error(f"Failed to generate data flow report: {e}")


if __name__ == '__main__':
    # Configure test runner for data flow integration
    unittest.main(verbosity=2, buffer=True)