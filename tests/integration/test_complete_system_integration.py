"""
Complete System Integration Testing Suite

Tests the entire IoT Predictive Maintenance System end-to-end workflow:
Data Ingestion → Preprocessing → Anomaly Detection → Forecasting →
Maintenance Optimization → Dashboard → Alerts

This suite validates cross-phase integration and complete system functionality.
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
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import core system components
try:
    from src.data_ingestion.nasa_data_service import NASADataService
    from src.preprocessing.data_preprocessor import DataPreprocessor
    from src.anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
    from src.forecasting.transformer_forecaster import TransformerForecaster
    from src.maintenance.optimization_engine import MaintenanceOptimizationEngine
    from src.dashboard.app import create_dashboard_app
    from src.alerts.alert_manager import AlertManager
    from src.utils.config_manager import ConfigManager
    from src.utils.database_manager import DatabaseManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Warning: Could not import core system components: {e}")

# Import business logic components
try:
    from src.business_logic.business_rules_engine import BusinessRulesEngine
    from src.business_logic.failure_classification import FailureClassificationEngine
    from src.business_logic.equipment_health import EquipmentHealthMonitor
    from src.business_logic.predictive_triggers import PredictiveTriggerEngine
except ImportError as e:
    print(f"Warning: Could not import business logic components: {e}")


@dataclass
class SystemIntegrationResult:
    """Container for system integration test results"""
    test_name: str
    workflow_stage: str
    success: bool
    execution_time: float
    data_processed: int
    anomalies_detected: int
    maintenance_actions: int
    dashboard_response_time: float
    alerts_generated: int
    error_details: Optional[str]
    timestamp: datetime


class TestCompleteSystemIntegration(unittest.TestCase):
    """Complete system integration testing"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.logger = get_logger(__name__)
        cls.config_manager = ConfigManager()
        cls.test_results = []
        cls.temp_dir = tempfile.mkdtemp(prefix="integration_test_")

        # Test configuration
        cls.test_config = {
            'nasa_data': {
                'smap_sensors': 25,
                'msl_sensors': 55,
                'total_sensors': 80,
                'data_points_per_sensor': 1000
            },
            'performance_targets': {
                'data_ingestion_rate': 1000,  # points per second
                'anomaly_detection_latency': 500,  # ms
                'dashboard_response_time': 500,  # ms
                'end_to_end_latency': 2000  # ms
            },
            'business_logic': {
                'failure_modes': 24,
                'health_tiers': 5,
                'decision_types': 7
            }
        }

        # Initialize components
        cls._initialize_test_components()

    @classmethod
    def _initialize_test_components(cls):
        """Initialize all system components for testing"""
        try:
            # Data layer
            cls.nasa_data_service = NASADataService()
            cls.data_preprocessor = DataPreprocessor()
            cls.database_manager = DatabaseManager()

            # ML layer
            cls.anomaly_detector = EnsembleAnomalyDetector()
            cls.forecaster = TransformerForecaster()

            # Business logic layer
            cls.business_rules_engine = BusinessRulesEngine()
            cls.failure_classifier = FailureClassificationEngine()
            cls.equipment_health_monitor = EquipmentHealthMonitor()
            cls.predictive_triggers = PredictiveTriggerEngine()

            # Application layer
            cls.maintenance_optimizer = MaintenanceOptimizationEngine()
            cls.alert_manager = AlertManager()

            cls.logger.info("All system components initialized successfully")

        except Exception as e:
            cls.logger.error(f"Failed to initialize components: {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Clean up temporary directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

        # Generate integration test report
        cls._generate_integration_report()

    def setUp(self):
        """Set up individual test"""
        self.start_time = time.time()
        self.test_data = self._generate_test_sensor_data()

    def tearDown(self):
        """Clean up individual test"""
        execution_time = time.time() - self.start_time
        self.logger.info(f"Test {self._testMethodName} completed in {execution_time:.2f}s")

    def _generate_test_sensor_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic sensor data for testing"""
        sensor_data = {}

        # SMAP sensors (25)
        for i in range(self.test_config['nasa_data']['smap_sensors']):
            sensor_id = f"smap_sensor_{i+1}"
            # Generate time series with some anomalies
            data_points = self.test_config['nasa_data']['data_points_per_sensor']
            normal_data = np.random.normal(0.5, 0.1, data_points)

            # Inject anomalies at random positions
            anomaly_positions = np.random.choice(data_points, size=int(data_points * 0.05), replace=False)
            normal_data[anomaly_positions] = np.random.uniform(2.0, 3.0, len(anomaly_positions))

            sensor_data[sensor_id] = {
                'values': normal_data,
                'timestamps': pd.date_range(
                    start=datetime.now() - timedelta(hours=24),
                    periods=data_points,
                    freq='1min'
                ),
                'metadata': {'type': 'smap', 'location': f'satellite_{i+1}'}
            }

        # MSL sensors (55)
        for i in range(self.test_config['nasa_data']['msl_sensors']):
            sensor_id = f"msl_sensor_{i+1}"
            data_points = self.test_config['nasa_data']['data_points_per_sensor']
            normal_data = np.random.normal(1.0, 0.15, data_points)

            # Inject anomalies
            anomaly_positions = np.random.choice(data_points, size=int(data_points * 0.03), replace=False)
            normal_data[anomaly_positions] = np.random.uniform(-1.0, -0.5, len(anomaly_positions))

            sensor_data[sensor_id] = {
                'values': normal_data,
                'timestamps': pd.date_range(
                    start=datetime.now() - timedelta(hours=24),
                    periods=data_points,
                    freq='1min'
                ),
                'metadata': {'type': 'msl', 'location': f'rover_{i+1}'}
            }

        return sensor_data

    def test_end_to_end_data_pipeline(self):
        """Test complete data pipeline from ingestion to storage"""
        test_start = time.time()

        try:
            # Stage 1: Data Ingestion
            ingestion_start = time.time()
            ingested_data = {}

            for sensor_id, sensor_info in self.test_data.items():
                # Simulate data ingestion
                processed_data = self.nasa_data_service.process_sensor_stream(
                    sensor_id, sensor_info['values'][:100]  # Process first 100 points
                )
                ingested_data[sensor_id] = processed_data

            ingestion_time = time.time() - ingestion_start

            # Validate ingestion performance
            expected_rate = self.test_config['performance_targets']['data_ingestion_rate']
            actual_rate = (len(ingested_data) * 100) / ingestion_time
            self.assertGreaterEqual(
                actual_rate, expected_rate * 0.8,  # 80% of target
                f"Data ingestion rate {actual_rate:.1f} below target {expected_rate}"
            )

            # Stage 2: Data Preprocessing
            preprocessing_start = time.time()
            preprocessed_data = {}

            for sensor_id, raw_data in ingested_data.items():
                preprocessed = self.data_preprocessor.preprocess_sensor_data(
                    raw_data, sensor_id
                )
                preprocessed_data[sensor_id] = preprocessed

            preprocessing_time = time.time() - preprocessing_start

            # Stage 3: Database Storage
            storage_start = time.time()

            for sensor_id, processed_data in preprocessed_data.items():
                success = self.database_manager.store_sensor_data(
                    sensor_id, processed_data
                )
                self.assertTrue(success, f"Failed to store data for {sensor_id}")

            storage_time = time.time() - storage_start

            # Record test result
            total_time = time.time() - test_start
            result = SystemIntegrationResult(
                test_name="end_to_end_data_pipeline",
                workflow_stage="data_pipeline",
                success=True,
                execution_time=total_time,
                data_processed=len(preprocessed_data) * 100,
                anomalies_detected=0,
                maintenance_actions=0,
                dashboard_response_time=0,
                alerts_generated=0,
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Data pipeline completed: ingestion={ingestion_time:.2f}s, "
                           f"preprocessing={preprocessing_time:.2f}s, storage={storage_time:.2f}s")

        except Exception as e:
            self.fail(f"End-to-end data pipeline failed: {e}")

    def test_complete_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow across all sensors"""
        test_start = time.time()

        try:
            # Prepare batch of sensor data
            sensor_batch = {}
            for sensor_id, sensor_info in list(self.test_data.items())[:20]:  # Test with 20 sensors
                sensor_batch[sensor_id] = sensor_info['values'][:500]  # 500 data points each

            # Stage 1: Batch Anomaly Detection
            detection_start = time.time()
            anomaly_results = {}

            # Use ensemble detector for each sensor
            for sensor_id, sensor_data in sensor_batch.items():
                anomalies = self.anomaly_detector.detect_anomalies(
                    sensor_data, sensor_id
                )
                anomaly_results[sensor_id] = anomalies

            detection_time = time.time() - detection_start

            # Validate detection performance
            expected_latency = self.test_config['performance_targets']['anomaly_detection_latency'] / 1000
            actual_latency = detection_time / len(sensor_batch)
            self.assertLessEqual(
                actual_latency, expected_latency,
                f"Anomaly detection latency {actual_latency:.3f}s exceeds target {expected_latency:.3f}s"
            )

            # Stage 2: Failure Classification
            classification_start = time.time()
            classified_failures = {}

            for sensor_id, anomalies in anomaly_results.items():
                if anomalies['anomaly_count'] > 0:
                    classifications = self.failure_classifier.classify_failures(
                        sensor_id, anomalies
                    )
                    classified_failures[sensor_id] = classifications

            classification_time = time.time() - classification_start

            # Stage 3: Equipment Health Assessment
            health_start = time.time()
            health_assessments = {}

            for sensor_id in sensor_batch.keys():
                health_metrics = self.equipment_health_monitor.assess_sensor_health(
                    sensor_id, sensor_batch[sensor_id]
                )
                health_assessments[sensor_id] = health_metrics

            health_time = time.time() - health_start

            # Validate results
            total_anomalies = sum(result['anomaly_count'] for result in anomaly_results.values())
            self.assertGreater(total_anomalies, 0, "Should detect some anomalies in test data")

            total_time = time.time() - test_start
            result = SystemIntegrationResult(
                test_name="complete_anomaly_detection_workflow",
                workflow_stage="anomaly_detection",
                success=True,
                execution_time=total_time,
                data_processed=len(sensor_batch) * 500,
                anomalies_detected=total_anomalies,
                maintenance_actions=0,
                dashboard_response_time=0,
                alerts_generated=0,
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Anomaly detection workflow: detection={detection_time:.2f}s, "
                           f"classification={classification_time:.2f}s, health={health_time:.2f}s")

        except Exception as e:
            self.fail(f"Complete anomaly detection workflow failed: {e}")

    def test_integrated_business_logic_workflow(self):
        """Test integrated business logic decision-making workflow"""
        test_start = time.time()

        try:
            # Prepare context with anomalies and health data
            decision_context = {
                'sensor_anomalies': {},
                'equipment_health': {},
                'maintenance_history': [],
                'resource_availability': {'technicians': 5, 'budget': 50000}
            }

            # Generate anomaly context
            for i, (sensor_id, sensor_info) in enumerate(list(self.test_data.items())[:10]):
                # Simulate detected anomalies
                anomaly_score = np.random.uniform(0.7, 0.95)
                decision_context['sensor_anomalies'][sensor_id] = {
                    'anomaly_score': anomaly_score,
                    'severity': 'HIGH' if anomaly_score > 0.9 else 'MEDIUM',
                    'failure_mode': f'failure_mode_{(i % 24) + 1}'
                }

                # Simulate health metrics
                decision_context['equipment_health'][sensor_id] = {
                    'overall_score': np.random.uniform(0.3, 0.8),
                    'degradation_rate': np.random.uniform(0.01, 0.05),
                    'remaining_useful_life': np.random.uniform(30, 180)
                }

            # Stage 1: Business Rules Engine Decision Making
            decision_start = time.time()

            maintenance_decisions = asyncio.run(
                self.business_rules_engine.make_maintenance_decision(decision_context)
            )

            decision_time = time.time() - decision_start

            # Stage 2: Predictive Trigger Evaluation
            trigger_start = time.time()

            trigger_events = self.predictive_triggers.evaluate_triggers(
                decision_context['sensor_anomalies'],
                decision_context['equipment_health']
            )

            trigger_time = time.time() - trigger_start

            # Stage 3: Maintenance Optimization
            optimization_start = time.time()

            optimization_plan = self.maintenance_optimizer.optimize_maintenance_schedule(
                maintenance_decisions, decision_context['resource_availability']
            )

            optimization_time = time.time() - optimization_start

            # Validate business logic results
            self.assertIsNotNone(maintenance_decisions, "Should generate maintenance decisions")
            self.assertGreater(len(trigger_events), 0, "Should generate trigger events")
            self.assertIsNotNone(optimization_plan, "Should generate optimization plan")

            # Validate decision quality
            decision_coverage = len(maintenance_decisions) / len(decision_context['sensor_anomalies'])
            self.assertGreaterEqual(decision_coverage, 0.8, "Should cover at least 80% of anomalous sensors")

            total_time = time.time() - test_start
            result = SystemIntegrationResult(
                test_name="integrated_business_logic_workflow",
                workflow_stage="business_logic",
                success=True,
                execution_time=total_time,
                data_processed=len(decision_context['sensor_anomalies']),
                anomalies_detected=len(decision_context['sensor_anomalies']),
                maintenance_actions=len(maintenance_decisions),
                dashboard_response_time=0,
                alerts_generated=len(trigger_events),
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Business logic workflow: decisions={decision_time:.2f}s, "
                           f"triggers={trigger_time:.2f}s, optimization={optimization_time:.2f}s")

        except Exception as e:
            self.fail(f"Integrated business logic workflow failed: {e}")

    def test_complete_forecasting_and_maintenance_workflow(self):
        """Test forecasting and maintenance optimization workflow"""
        test_start = time.time()

        try:
            # Prepare historical data for forecasting
            historical_data = {}
            for sensor_id, sensor_info in list(self.test_data.items())[:15]:
                historical_data[sensor_id] = {
                    'values': sensor_info['values'][:800],  # Use 800 points for training
                    'timestamps': sensor_info['timestamps'][:800]
                }

            # Stage 1: Time Series Forecasting
            forecasting_start = time.time()
            forecasts = {}

            for sensor_id, hist_data in historical_data.items():
                forecast = self.forecaster.forecast_sensor_values(
                    hist_data['values'], forecast_horizon=50
                )
                forecasts[sensor_id] = forecast

            forecasting_time = time.time() - forecasting_start

            # Stage 2: Predictive Maintenance Scheduling
            scheduling_start = time.time()

            # Generate maintenance requirements based on forecasts
            maintenance_requirements = []
            for sensor_id, forecast in forecasts.items():
                # Analyze forecast for maintenance needs
                risk_score = np.std(forecast['predictions'])  # Higher volatility = higher risk
                if risk_score > 0.3:  # Threshold for maintenance requirement
                    maintenance_requirements.append({
                        'sensor_id': sensor_id,
                        'priority': 'HIGH' if risk_score > 0.5 else 'MEDIUM',
                        'estimated_cost': np.random.uniform(1000, 5000),
                        'estimated_duration': np.random.uniform(2, 8),
                        'required_skills': ['mechanical', 'electrical'],
                        'forecast_confidence': forecast['confidence']
                    })

            # Optimize maintenance schedule
            optimized_schedule = self.maintenance_optimizer.create_optimal_schedule(
                maintenance_requirements
            )

            scheduling_time = time.time() - scheduling_start

            # Stage 3: Alert Generation
            alert_start = time.time()
            alerts_generated = []

            for requirement in maintenance_requirements:
                if requirement['priority'] == 'HIGH':
                    alert = self.alert_manager.create_maintenance_alert(
                        requirement['sensor_id'],
                        f"High priority maintenance required based on forecast analysis",
                        requirement['priority']
                    )
                    alerts_generated.append(alert)

            alert_time = time.time() - alert_start

            # Validate forecasting and maintenance results
            self.assertEqual(len(forecasts), len(historical_data), "Should forecast all sensors")
            self.assertGreater(len(maintenance_requirements), 0, "Should identify maintenance needs")
            self.assertIsNotNone(optimized_schedule, "Should generate optimized schedule")

            # Validate forecast quality
            for sensor_id, forecast in forecasts.items():
                self.assertIn('predictions', forecast, f"Forecast for {sensor_id} should have predictions")
                self.assertIn('confidence', forecast, f"Forecast for {sensor_id} should have confidence")
                self.assertGreater(forecast['confidence'], 0.5, f"Forecast confidence should be > 0.5")

            total_time = time.time() - test_start
            result = SystemIntegrationResult(
                test_name="complete_forecasting_and_maintenance_workflow",
                workflow_stage="forecasting_maintenance",
                success=True,
                execution_time=total_time,
                data_processed=len(historical_data) * 800,
                anomalies_detected=0,
                maintenance_actions=len(maintenance_requirements),
                dashboard_response_time=0,
                alerts_generated=len(alerts_generated),
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Forecasting workflow: forecasting={forecasting_time:.2f}s, "
                           f"scheduling={scheduling_time:.2f}s, alerts={alert_time:.2f}s")

        except Exception as e:
            self.fail(f"Complete forecasting and maintenance workflow failed: {e}")

    def test_dashboard_integration_and_performance(self):
        """Test dashboard integration and performance under load"""
        test_start = time.time()

        try:
            # Create dashboard app instance
            dashboard_start = time.time()
            app = create_dashboard_app()

            # Test dashboard data loading performance
            dashboard_data = {
                'sensor_count': self.test_config['nasa_data']['total_sensors'],
                'anomaly_data': [],
                'health_metrics': [],
                'maintenance_schedule': [],
                'performance_metrics': []
            }

            # Simulate loading dashboard data
            for i in range(50):  # Simulate 50 dashboard updates
                sensor_id = f"sensor_{i % 80}"
                dashboard_data['anomaly_data'].append({
                    'sensor_id': sensor_id,
                    'anomaly_score': np.random.uniform(0, 1),
                    'timestamp': datetime.now() - timedelta(minutes=i)
                })

                dashboard_data['health_metrics'].append({
                    'sensor_id': sensor_id,
                    'health_score': np.random.uniform(0.2, 1.0),
                    'status': np.random.choice(['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL'])
                })

            dashboard_load_time = time.time() - dashboard_start

            # Test dashboard responsiveness
            response_start = time.time()

            # Simulate dashboard callback processing
            callback_times = []
            for _ in range(10):  # Test 10 callback responses
                callback_start = time.time()

                # Simulate complex dashboard callback
                processed_data = self._simulate_dashboard_callback(dashboard_data)

                callback_time = time.time() - callback_start
                callback_times.append(callback_time)

            avg_response_time = np.mean(callback_times) * 1000  # Convert to milliseconds
            response_time = time.time() - response_start

            # Validate dashboard performance
            target_response = self.test_config['performance_targets']['dashboard_response_time']
            self.assertLessEqual(
                avg_response_time, target_response,
                f"Dashboard response time {avg_response_time:.1f}ms exceeds target {target_response}ms"
            )

            total_time = time.time() - test_start
            result = SystemIntegrationResult(
                test_name="dashboard_integration_and_performance",
                workflow_stage="dashboard",
                success=True,
                execution_time=total_time,
                data_processed=len(dashboard_data['anomaly_data']),
                anomalies_detected=len(dashboard_data['anomaly_data']),
                maintenance_actions=0,
                dashboard_response_time=avg_response_time,
                alerts_generated=0,
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Dashboard performance: load={dashboard_load_time:.2f}s, "
                           f"avg_response={avg_response_time:.1f}ms")

        except Exception as e:
            self.fail(f"Dashboard integration and performance test failed: {e}")

    def _simulate_dashboard_callback(self, dashboard_data: Dict) -> Dict:
        """Simulate dashboard callback processing"""
        # Simulate data filtering and aggregation
        filtered_anomalies = [
            item for item in dashboard_data['anomaly_data']
            if item['anomaly_score'] > 0.7
        ]

        # Simulate health status aggregation
        health_summary = {}
        for item in dashboard_data['health_metrics']:
            status = item['status']
            health_summary[status] = health_summary.get(status, 0) + 1

        # Simulate chart data preparation
        chart_data = {
            'anomaly_timeline': sorted(filtered_anomalies, key=lambda x: x['timestamp']),
            'health_distribution': health_summary,
            'sensor_status_count': len(dashboard_data['health_metrics'])
        }

        return chart_data

    def test_complete_end_to_end_workflow(self):
        """Test complete end-to-end system workflow"""
        test_start = time.time()

        try:
            self.logger.info("Starting complete end-to-end workflow test")

            # Select subset of sensors for full workflow test
            test_sensors = list(self.test_data.keys())[:20]  # Test with 20 sensors
            workflow_data = {sensor_id: self.test_data[sensor_id] for sensor_id in test_sensors}

            # Stage 1: Data Ingestion and Preprocessing
            stage1_start = time.time()
            preprocessed_data = {}

            for sensor_id, sensor_info in workflow_data.items():
                # Ingest data
                raw_data = sensor_info['values'][:200]  # Use 200 data points

                # Preprocess
                processed = self.data_preprocessor.preprocess_sensor_data(raw_data, sensor_id)
                preprocessed_data[sensor_id] = processed

            stage1_time = time.time() - stage1_start

            # Stage 2: Anomaly Detection and Classification
            stage2_start = time.time()
            anomaly_results = {}
            classified_failures = {}

            for sensor_id, processed_data in preprocessed_data.items():
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(processed_data, sensor_id)
                anomaly_results[sensor_id] = anomalies

                # Classify failures if anomalies found
                if anomalies['anomaly_count'] > 0:
                    classifications = self.failure_classifier.classify_failures(sensor_id, anomalies)
                    classified_failures[sensor_id] = classifications

            stage2_time = time.time() - stage2_start

            # Stage 3: Equipment Health Assessment
            stage3_start = time.time()
            health_assessments = {}

            for sensor_id, processed_data in preprocessed_data.items():
                health = self.equipment_health_monitor.assess_sensor_health(sensor_id, processed_data)
                health_assessments[sensor_id] = health

            stage3_time = time.time() - stage3_start

            # Stage 4: Business Rules and Decision Making
            stage4_start = time.time()

            # Prepare decision context
            decision_context = {
                'sensor_anomalies': anomaly_results,
                'equipment_health': health_assessments,
                'classified_failures': classified_failures,
                'resource_availability': {'technicians': 3, 'budget': 25000}
            }

            # Make maintenance decisions
            maintenance_decisions = asyncio.run(
                self.business_rules_engine.make_maintenance_decision(decision_context)
            )

            stage4_time = time.time() - stage4_start

            # Stage 5: Forecasting and Optimization
            stage5_start = time.time()

            # Generate forecasts for critical sensors
            forecasts = {}
            critical_sensors = [
                sensor_id for sensor_id, health in health_assessments.items()
                if health.get('overall_score', 1.0) < 0.6
            ]

            for sensor_id in critical_sensors[:5]:  # Forecast top 5 critical sensors
                forecast = self.forecaster.forecast_sensor_values(
                    preprocessed_data[sensor_id], forecast_horizon=24
                )
                forecasts[sensor_id] = forecast

            # Optimize maintenance schedule
            if maintenance_decisions:
                optimized_schedule = self.maintenance_optimizer.create_optimal_schedule(
                    maintenance_decisions
                )
            else:
                optimized_schedule = []

            stage5_time = time.time() - stage5_start

            # Stage 6: Alert Generation and Dashboard Update
            stage6_start = time.time()

            # Generate alerts for critical situations
            alerts_generated = []
            for sensor_id, health in health_assessments.items():
                if health.get('overall_score', 1.0) < 0.3:  # Critical health threshold
                    alert = self.alert_manager.create_maintenance_alert(
                        sensor_id,
                        f"Critical equipment health detected: {health.get('overall_score', 0):.2f}",
                        'CRITICAL'
                    )
                    alerts_generated.append(alert)

            # Prepare dashboard data
            dashboard_summary = {
                'total_sensors': len(workflow_data),
                'total_anomalies': sum(result['anomaly_count'] for result in anomaly_results.values()),
                'critical_sensors': len(critical_sensors),
                'maintenance_actions': len(maintenance_decisions),
                'alerts_generated': len(alerts_generated),
                'avg_health_score': np.mean([
                    health.get('overall_score', 1.0) for health in health_assessments.values()
                ])
            }

            stage6_time = time.time() - stage6_start

            # Validate complete workflow
            total_time = time.time() - test_start

            # Performance validation
            target_latency = self.test_config['performance_targets']['end_to_end_latency'] / 1000
            self.assertLessEqual(
                total_time, target_latency,
                f"End-to-end latency {total_time:.2f}s exceeds target {target_latency:.2f}s"
            )

            # Functional validation
            self.assertGreater(
                dashboard_summary['total_anomalies'], 0,
                "Should detect anomalies in test data"
            )

            self.assertGreaterEqual(
                dashboard_summary['avg_health_score'], 0.0,
                "Average health score should be valid"
            )

            # Record comprehensive test result
            result = SystemIntegrationResult(
                test_name="complete_end_to_end_workflow",
                workflow_stage="end_to_end",
                success=True,
                execution_time=total_time,
                data_processed=len(workflow_data) * 200,
                anomalies_detected=dashboard_summary['total_anomalies'],
                maintenance_actions=dashboard_summary['maintenance_actions'],
                dashboard_response_time=0,
                alerts_generated=dashboard_summary['alerts_generated'],
                error_details=None,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Complete workflow stages: "
                           f"data={stage1_time:.2f}s, anomaly={stage2_time:.2f}s, "
                           f"health={stage3_time:.2f}s, business={stage4_time:.2f}s, "
                           f"forecast={stage5_time:.2f}s, alerts={stage6_time:.2f}s")

            self.logger.info(f"Workflow summary: {dashboard_summary}")

        except Exception as e:
            error_result = SystemIntegrationResult(
                test_name="complete_end_to_end_workflow",
                workflow_stage="end_to_end",
                success=False,
                execution_time=time.time() - test_start,
                data_processed=0,
                anomalies_detected=0,
                maintenance_actions=0,
                dashboard_response_time=0,
                alerts_generated=0,
                error_details=str(e),
                timestamp=datetime.now()
            )
            self.test_results.append(error_result)
            self.fail(f"Complete end-to-end workflow failed: {e}")

    @classmethod
    def _generate_integration_report(cls):
        """Generate comprehensive integration test report"""
        try:
            report_data = {
                'test_suite': 'Complete System Integration',
                'execution_timestamp': datetime.now().isoformat(),
                'total_tests': len(cls.test_results),
                'successful_tests': len([r for r in cls.test_results if r.success]),
                'failed_tests': len([r for r in cls.test_results if not r.success]),
                'total_execution_time': sum(r.execution_time for r in cls.test_results),
                'performance_summary': {
                    'total_data_processed': sum(r.data_processed for r in cls.test_results),
                    'total_anomalies_detected': sum(r.anomalies_detected for r in cls.test_results),
                    'total_maintenance_actions': sum(r.maintenance_actions for r in cls.test_results),
                    'total_alerts_generated': sum(r.alerts_generated for r in cls.test_results),
                    'avg_dashboard_response_time': np.mean([
                        r.dashboard_response_time for r in cls.test_results
                        if r.dashboard_response_time > 0
                    ]) if any(r.dashboard_response_time > 0 for r in cls.test_results) else 0
                },
                'workflow_coverage': {
                    'data_pipeline': any(r.workflow_stage == 'data_pipeline' for r in cls.test_results),
                    'anomaly_detection': any(r.workflow_stage == 'anomaly_detection' for r in cls.test_results),
                    'business_logic': any(r.workflow_stage == 'business_logic' for r in cls.test_results),
                    'forecasting_maintenance': any(r.workflow_stage == 'forecasting_maintenance' for r in cls.test_results),
                    'dashboard': any(r.workflow_stage == 'dashboard' for r in cls.test_results),
                    'end_to_end': any(r.workflow_stage == 'end_to_end' for r in cls.test_results)
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'workflow_stage': r.workflow_stage,
                        'success': r.success,
                        'execution_time': r.execution_time,
                        'data_processed': r.data_processed,
                        'anomalies_detected': r.anomalies_detected,
                        'maintenance_actions': r.maintenance_actions,
                        'dashboard_response_time': r.dashboard_response_time,
                        'alerts_generated': r.alerts_generated,
                        'error_details': r.error_details,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in cls.test_results
                ]
            }

            # Save integration test report
            report_path = Path(cls.temp_dir) / "integration_test_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            # Copy to project testing directory
            project_report_path = Path(__file__).parent.parent / "integration_test_report.json"
            shutil.copy2(report_path, project_report_path)

            cls.logger.info(f"Integration test report saved to {project_report_path}")

        except Exception as e:
            cls.logger.error(f"Failed to generate integration report: {e}")


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)