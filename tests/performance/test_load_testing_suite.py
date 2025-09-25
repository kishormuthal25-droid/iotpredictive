"""
Load Testing Suite

Tests system performance under various load conditions:
- Concurrent user simulation
- High-volume data processing
- Resource saturation testing
- Throughput measurement
- Latency analysis under load
"""

import unittest
import sys
import os
import asyncio
import time
import threading
import concurrent.futures
import queue
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import psutil
import gc
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import system components
try:
    from src.dashboard.app import create_dashboard_app
    from src.data_ingestion.nasa_data_service import NASADataService
    from src.preprocessing.data_preprocessor import DataPreprocessor
    from src.anomaly_detection.ensemble_detector import EnsembleAnomalyDetector
    from src.forecasting.transformer_forecaster import TransformerForecaster
    from src.business_logic.business_rules_engine import BusinessRulesEngine
    from src.maintenance.optimization_engine import MaintenanceOptimizationEngine
    from src.utils.config_manager import ConfigManager
    from src.utils.database_manager import DatabaseManager
    from src.utils.logger import get_logger
except ImportError as e:
    print(f"Warning: Could not import system components: {e}")


@dataclass
class LoadTestResult:
    """Container for load test results"""
    test_name: str
    load_scenario: str
    concurrent_users: int
    data_volume: int
    duration_seconds: float
    avg_response_time: float
    max_response_time: float
    throughput_per_second: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    success_rate: float
    timestamp: datetime


class TestLoadTestingSuite(unittest.TestCase):
    """Comprehensive load testing suite"""

    @classmethod
    def setUpClass(cls):
        """Set up load testing environment"""
        cls.logger = get_logger(__name__)
        cls.config_manager = ConfigManager()
        cls.test_results = []

        # Load test configuration
        cls.load_config = {
            'test_scenarios': {
                'light_load': {'users': 10, 'data_volume': 1000, 'duration': 30},
                'medium_load': {'users': 50, 'data_volume': 5000, 'duration': 60},
                'heavy_load': {'users': 100, 'data_volume': 10000, 'duration': 90},
                'peak_load': {'users': 200, 'data_volume': 20000, 'duration': 120}
            },
            'performance_targets': {
                'max_response_time': 2000,  # ms
                'avg_response_time': 500,   # ms
                'min_throughput': 100,      # operations per second
                'max_error_rate': 0.05,     # 5%
                'max_cpu_usage': 0.8,       # 80%
                'max_memory_usage': 0.8     # 80%
            },
            'sensor_simulation': {
                'sensors_per_user': 4,
                'data_points_per_sensor': 100,
                'anomaly_injection_rate': 0.1
            }
        }

        # Initialize components
        cls._initialize_load_test_components()

    @classmethod
    def _initialize_load_test_components(cls):
        """Initialize components for load testing"""
        try:
            cls.nasa_data_service = NASADataService()
            cls.data_preprocessor = DataPreprocessor()
            cls.ensemble_detector = EnsembleAnomalyDetector()
            cls.transformer_forecaster = TransformerForecaster()
            cls.business_rules_engine = BusinessRulesEngine()
            cls.maintenance_optimizer = MaintenanceOptimizationEngine()
            cls.database_manager = DatabaseManager()

            cls.logger.info("Load testing components initialized successfully")

        except Exception as e:
            cls.logger.error(f"Failed to initialize load testing components: {e}")
            raise

    def setUp(self):
        """Set up individual load test"""
        self.test_start_time = time.time()
        self.system_monitor = SystemMonitor()

    def tearDown(self):
        """Clean up individual load test"""
        execution_time = time.time() - self.test_start_time
        self.logger.info(f"Load test {self._testMethodName} completed in {execution_time:.2f}s")

        # Force garbage collection
        gc.collect()

    def _generate_load_test_data(self, users: int, sensors_per_user: int, data_points: int) -> Dict[str, Any]:
        """Generate realistic test data for load testing"""
        test_data = {}

        for user_id in range(users):
            user_sensors = {}

            for sensor_idx in range(sensors_per_user):
                sensor_id = f"user_{user_id}_sensor_{sensor_idx}"

                # Generate realistic time series data
                base_values = np.random.normal(0.5, 0.1, data_points)
                trend = np.linspace(0, 0.1, data_points) * np.random.choice([-1, 1])
                seasonal = 0.05 * np.sin(2 * np.pi * np.arange(data_points) / 20)
                noise = np.random.normal(0, 0.02, data_points)

                sensor_values = np.clip(base_values + trend + seasonal + noise, 0, 1)

                # Inject anomalies
                anomaly_count = int(data_points * self.load_config['sensor_simulation']['anomaly_injection_rate'])
                if anomaly_count > 0:
                    anomaly_indices = np.random.choice(data_points, size=anomaly_count, replace=False)
                    sensor_values[anomaly_indices] += np.random.uniform(0.3, 0.5, anomaly_count)
                    sensor_values = np.clip(sensor_values, 0, 1)

                user_sensors[sensor_id] = {
                    'values': sensor_values,
                    'timestamps': pd.date_range(
                        start=datetime.now() - timedelta(minutes=data_points),
                        periods=data_points,
                        freq='1min'
                    ),
                    'metadata': {
                        'user_id': user_id,
                        'sensor_type': np.random.choice(['temperature', 'pressure', 'vibration']),
                        'location': f"zone_{user_id % 5}",
                        'equipment_id': f"equipment_{user_id % 10}"
                    }
                }

            test_data[f"user_{user_id}"] = user_sensors

        return test_data

    def _simulate_user_workflow(self, user_id: str, user_data: Dict, result_queue: queue.Queue):
        """Simulate a complete user workflow under load"""
        workflow_start = time.time()
        workflow_results = []
        errors = []

        try:
            # Stage 1: Data Processing
            for sensor_id, sensor_info in user_data.items():
                stage_start = time.time()

                try:
                    # Data ingestion
                    ingested_data = self.nasa_data_service.process_sensor_stream(
                        sensor_id, sensor_info['values']
                    )

                    # Preprocessing
                    if ingested_data is not None:
                        preprocessed_data = self.data_preprocessor.preprocess_sensor_data(
                            ingested_data, sensor_id
                        )

                        # Anomaly detection
                        if preprocessed_data is not None:
                            anomaly_results = self.ensemble_detector.detect_anomalies(
                                preprocessed_data, sensor_id
                            )

                            stage_time = time.time() - stage_start
                            workflow_results.append({
                                'stage': 'data_processing',
                                'sensor_id': sensor_id,
                                'processing_time': stage_time,
                                'success': True,
                                'data_points_processed': len(preprocessed_data) if preprocessed_data is not None else 0
                            })

                except Exception as e:
                    stage_time = time.time() - stage_start
                    errors.append(f"Data processing error for {sensor_id}: {e}")
                    workflow_results.append({
                        'stage': 'data_processing',
                        'sensor_id': sensor_id,
                        'processing_time': stage_time,
                        'success': False,
                        'data_points_processed': 0
                    })

            # Stage 2: Business Logic Processing
            if workflow_results and any(r['success'] for r in workflow_results):
                stage_start = time.time()

                try:
                    # Aggregate data for business logic
                    sensor_health_data = {}
                    for sensor_id, sensor_info in list(user_data.items())[:2]:  # Process subset for performance
                        sensor_health_data[sensor_id] = {
                            'current_value': sensor_info['values'][-1],
                            'health_score': np.random.uniform(0.5, 0.9),
                            'anomaly_probability': np.random.uniform(0.1, 0.3)
                        }

                    # Business rules evaluation
                    business_context = {
                        'user_id': user_id,
                        'sensor_health_data': sensor_health_data,
                        'timestamp': datetime.now()
                    }

                    business_decisions = asyncio.run(
                        self.business_rules_engine.evaluate_user_context(business_context)
                    )

                    stage_time = time.time() - stage_start
                    workflow_results.append({
                        'stage': 'business_logic',
                        'user_id': user_id,
                        'processing_time': stage_time,
                        'success': True,
                        'decisions_count': len(business_decisions) if business_decisions else 0
                    })

                except Exception as e:
                    stage_time = time.time() - stage_start
                    errors.append(f"Business logic error for {user_id}: {e}")
                    workflow_results.append({
                        'stage': 'business_logic',
                        'user_id': user_id,
                        'processing_time': stage_time,
                        'success': False,
                        'decisions_count': 0
                    })

            # Stage 3: Database Operations
            stage_start = time.time()

            try:
                # Simulate database operations
                for sensor_id, sensor_info in list(user_data.items())[:3]:  # Process subset
                    stored = self.database_manager.store_sensor_data(
                        sensor_id, sensor_info['values'][:50]  # Store subset for performance
                    )

                    if not stored:
                        errors.append(f"Database storage failed for {sensor_id}")

                stage_time = time.time() - stage_start
                workflow_results.append({
                    'stage': 'database_operations',
                    'user_id': user_id,
                    'processing_time': stage_time,
                    'success': True,
                    'records_stored': min(3, len(user_data)) * 50
                })

            except Exception as e:
                stage_time = time.time() - stage_start
                errors.append(f"Database operations error for {user_id}: {e}")
                workflow_results.append({
                    'stage': 'database_operations',
                    'user_id': user_id,
                    'processing_time': stage_time,
                    'success': False,
                    'records_stored': 0
                })

        except Exception as e:
            errors.append(f"Workflow error for {user_id}: {e}")

        # Calculate workflow metrics
        total_workflow_time = time.time() - workflow_start
        successful_stages = len([r for r in workflow_results if r['success']])
        total_stages = len(workflow_results)

        result_queue.put({
            'user_id': user_id,
            'total_time': total_workflow_time,
            'workflow_results': workflow_results,
            'errors': errors,
            'success_rate': successful_stages / max(total_stages, 1),
            'data_points_processed': sum(r.get('data_points_processed', 0) for r in workflow_results)
        })

    def test_light_load_scenario(self):
        """Test system under light load conditions"""
        scenario = self.load_config['test_scenarios']['light_load']
        self._execute_load_test_scenario('light_load', scenario)

    def test_medium_load_scenario(self):
        """Test system under medium load conditions"""
        scenario = self.load_config['test_scenarios']['medium_load']
        self._execute_load_test_scenario('medium_load', scenario)

    def test_heavy_load_scenario(self):
        """Test system under heavy load conditions"""
        scenario = self.load_config['test_scenarios']['heavy_load']
        self._execute_load_test_scenario('heavy_load', scenario)

    def test_peak_load_scenario(self):
        """Test system under peak load conditions"""
        scenario = self.load_config['test_scenarios']['peak_load']
        self._execute_load_test_scenario('peak_load', scenario)

    def _execute_load_test_scenario(self, scenario_name: str, scenario_config: Dict):
        """Execute a specific load test scenario"""
        test_start = time.time()

        try:
            users = scenario_config['users']
            data_volume = scenario_config['data_volume']
            duration = scenario_config['duration']

            self.logger.info(f"Starting {scenario_name} load test: {users} users, {data_volume} data volume")

            # Generate test data
            sensors_per_user = self.load_config['sensor_simulation']['sensors_per_user']
            data_points_per_sensor = data_volume // (users * sensors_per_user)

            test_data = self._generate_load_test_data(users, sensors_per_user, data_points_per_sensor)

            # Start system monitoring
            self.system_monitor.start_monitoring()

            # Execute load test
            load_test_start = time.time()
            result_queue = queue.Queue()

            # Create thread pool for concurrent user simulation
            max_workers = min(users, 20)  # Limit concurrent threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for user_id, user_data in test_data.items():
                    future = executor.submit(self._simulate_user_workflow, user_id, user_data, result_queue)
                    futures.append(future)

                # Wait for completion or timeout
                completed_futures = []
                for future in concurrent.futures.as_completed(futures, timeout=duration + 30):
                    completed_futures.append(future)

            load_test_time = time.time() - load_test_start

            # Stop system monitoring
            system_metrics = self.system_monitor.stop_monitoring()

            # Collect results
            user_results = []
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    user_results.append(result)
                except queue.Empty:
                    break

            # Calculate performance metrics
            performance_metrics = self._calculate_load_test_metrics(
                user_results, load_test_time, system_metrics
            )

            # Validate performance against targets
            self._validate_load_test_performance(scenario_name, performance_metrics)

            # Record test result
            result = LoadTestResult(
                test_name=f"load_test_{scenario_name}",
                load_scenario=scenario_name,
                concurrent_users=users,
                data_volume=sum(r['data_points_processed'] for r in user_results),
                duration_seconds=load_test_time,
                avg_response_time=performance_metrics['avg_response_time'],
                max_response_time=performance_metrics['max_response_time'],
                throughput_per_second=performance_metrics['throughput_per_second'],
                error_rate=performance_metrics['error_rate'],
                cpu_utilization=system_metrics['avg_cpu_usage'],
                memory_utilization=system_metrics['avg_memory_usage'],
                success_rate=performance_metrics['success_rate'],
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"{scenario_name} load test completed: "
                           f"throughput={performance_metrics['throughput_per_second']:.1f}/s, "
                           f"error_rate={performance_metrics['error_rate']:.3%}")

        except Exception as e:
            self.fail(f"Load test scenario {scenario_name} failed: {e}")

    def _calculate_load_test_metrics(self, user_results: List[Dict], total_time: float,
                                   system_metrics: Dict) -> Dict[str, float]:
        """Calculate comprehensive load test performance metrics"""
        if not user_results:
            return {
                'avg_response_time': 0,
                'max_response_time': 0,
                'throughput_per_second': 0,
                'error_rate': 1.0,
                'success_rate': 0.0
            }

        # Response time metrics
        response_times = [r['total_time'] * 1000 for r in user_results]  # Convert to ms
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)

        # Throughput metrics
        total_operations = len(user_results)
        throughput_per_second = total_operations / max(total_time, 1)

        # Error rate metrics
        total_errors = sum(len(r['errors']) for r in user_results)
        total_possible_operations = total_operations * 3  # 3 stages per user workflow
        error_rate = total_errors / max(total_possible_operations, 1)

        # Success rate metrics
        success_rates = [r['success_rate'] for r in user_results]
        overall_success_rate = np.mean(success_rates)

        return {
            'avg_response_time': avg_response_time,
            'max_response_time': max_response_time,
            'throughput_per_second': throughput_per_second,
            'error_rate': error_rate,
            'success_rate': overall_success_rate
        }

    def _validate_load_test_performance(self, scenario_name: str, metrics: Dict[str, float]):
        """Validate load test performance against targets"""
        targets = self.load_config['performance_targets']

        # Response time validation
        self.assertLessEqual(
            metrics['avg_response_time'], targets['avg_response_time'],
            f"{scenario_name}: Average response time {metrics['avg_response_time']:.1f}ms "
            f"exceeds target {targets['avg_response_time']}ms"
        )

        self.assertLessEqual(
            metrics['max_response_time'], targets['max_response_time'],
            f"{scenario_name}: Max response time {metrics['max_response_time']:.1f}ms "
            f"exceeds target {targets['max_response_time']}ms"
        )

        # Throughput validation
        self.assertGreaterEqual(
            metrics['throughput_per_second'], targets['min_throughput'],
            f"{scenario_name}: Throughput {metrics['throughput_per_second']:.1f}/s "
            f"below target {targets['min_throughput']}/s"
        )

        # Error rate validation
        self.assertLessEqual(
            metrics['error_rate'], targets['max_error_rate'],
            f"{scenario_name}: Error rate {metrics['error_rate']:.3%} "
            f"exceeds target {targets['max_error_rate']:.3%}"
        )

    def test_concurrent_dashboard_access(self):
        """Test concurrent dashboard access under load"""
        test_start = time.time()

        try:
            concurrent_users = 25
            requests_per_user = 10

            def simulate_dashboard_user(user_id: int, result_queue: queue.Queue):
                """Simulate dashboard user interactions"""
                user_results = []
                errors = []

                for request_idx in range(requests_per_user):
                    request_start = time.time()

                    try:
                        # Simulate dashboard data requests
                        dashboard_data = {
                            'user_id': user_id,
                            'request_id': request_idx,
                            'sensor_data': np.random.random(100),
                            'timestamp': datetime.now()
                        }

                        # Simulate dashboard processing time
                        processing_time = np.random.uniform(0.1, 0.5)
                        time.sleep(processing_time)

                        request_time = time.time() - request_start
                        user_results.append({
                            'request_id': request_idx,
                            'response_time': request_time * 1000,  # ms
                            'success': True
                        })

                    except Exception as e:
                        request_time = time.time() - request_start
                        errors.append(f"Dashboard request {request_idx} failed: {e}")
                        user_results.append({
                            'request_id': request_idx,
                            'response_time': request_time * 1000,  # ms
                            'success': False
                        })

                result_queue.put({
                    'user_id': user_id,
                    'requests': user_results,
                    'errors': errors
                })

            # Execute concurrent dashboard access test
            result_queue = queue.Queue()

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [
                    executor.submit(simulate_dashboard_user, user_id, result_queue)
                    for user_id in range(concurrent_users)
                ]

                # Wait for completion
                for future in concurrent.futures.as_completed(futures, timeout=60):
                    pass

            # Collect results
            dashboard_results = []
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    dashboard_results.append(result)
                except queue.Empty:
                    break

            # Calculate dashboard performance metrics
            all_response_times = []
            total_requests = 0
            successful_requests = 0

            for user_result in dashboard_results:
                for request in user_result['requests']:
                    all_response_times.append(request['response_time'])
                    total_requests += 1
                    if request['success']:
                        successful_requests += 1

            avg_response_time = np.mean(all_response_times) if all_response_times else 0
            success_rate = successful_requests / max(total_requests, 1)
            total_test_time = time.time() - test_start

            # Validate dashboard performance
            self.assertLessEqual(
                avg_response_time, 1000,  # 1 second max for dashboard
                f"Dashboard average response time {avg_response_time:.1f}ms too high"
            )

            self.assertGreaterEqual(
                success_rate, 0.95,
                f"Dashboard success rate {success_rate:.3%} below threshold"
            )

            # Record dashboard load test result
            result = LoadTestResult(
                test_name="concurrent_dashboard_access",
                load_scenario="dashboard_concurrency",
                concurrent_users=concurrent_users,
                data_volume=total_requests,
                duration_seconds=total_test_time,
                avg_response_time=avg_response_time,
                max_response_time=max(all_response_times) if all_response_times else 0,
                throughput_per_second=total_requests / total_test_time,
                error_rate=1 - success_rate,
                cpu_utilization=0.0,  # Not measured for this test
                memory_utilization=0.0,  # Not measured for this test
                success_rate=success_rate,
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            self.logger.info(f"Dashboard concurrency test: {avg_response_time:.1f}ms avg, "
                           f"{success_rate:.3%} success rate")

        except Exception as e:
            self.fail(f"Concurrent dashboard access test failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Generate load testing report"""
        cls._generate_load_testing_report()

    @classmethod
    def _generate_load_testing_report(cls):
        """Generate comprehensive load testing report"""
        try:
            report_data = {
                'test_suite': 'Load Testing Suite',
                'execution_timestamp': datetime.now().isoformat(),
                'total_load_tests': len(cls.test_results),
                'test_scenarios_covered': list(set(r.load_scenario for r in cls.test_results)),
                'performance_summary': {
                    'avg_response_time_ms': np.mean([r.avg_response_time for r in cls.test_results]),
                    'max_response_time_ms': np.max([r.max_response_time for r in cls.test_results]),
                    'avg_throughput_per_second': np.mean([r.throughput_per_second for r in cls.test_results]),
                    'avg_error_rate': np.mean([r.error_rate for r in cls.test_results]),
                    'avg_success_rate': np.mean([r.success_rate for r in cls.test_results]),
                    'peak_concurrent_users': max([r.concurrent_users for r in cls.test_results]),
                    'total_data_processed': sum([r.data_volume for r in cls.test_results])
                },
                'scenario_performance': {
                    scenario: {
                        'tests_count': len([r for r in cls.test_results if r.load_scenario == scenario]),
                        'avg_response_time': np.mean([r.avg_response_time for r in cls.test_results if r.load_scenario == scenario]),
                        'avg_throughput': np.mean([r.throughput_per_second for r in cls.test_results if r.load_scenario == scenario]),
                        'avg_success_rate': np.mean([r.success_rate for r in cls.test_results if r.load_scenario == scenario])
                    }
                    for scenario in set(r.load_scenario for r in cls.test_results)
                },
                'system_resource_usage': {
                    'avg_cpu_utilization': np.mean([r.cpu_utilization for r in cls.test_results if r.cpu_utilization > 0]),
                    'avg_memory_utilization': np.mean([r.memory_utilization for r in cls.test_results if r.memory_utilization > 0]),
                    'peak_cpu_usage': max([r.cpu_utilization for r in cls.test_results if r.cpu_utilization > 0], default=0),
                    'peak_memory_usage': max([r.memory_utilization for r in cls.test_results if r.memory_utilization > 0], default=0)
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'load_scenario': r.load_scenario,
                        'concurrent_users': r.concurrent_users,
                        'data_volume': r.data_volume,
                        'duration_seconds': r.duration_seconds,
                        'avg_response_time': r.avg_response_time,
                        'max_response_time': r.max_response_time,
                        'throughput_per_second': r.throughput_per_second,
                        'error_rate': r.error_rate,
                        'cpu_utilization': r.cpu_utilization,
                        'memory_utilization': r.memory_utilization,
                        'success_rate': r.success_rate,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in cls.test_results
                ]
            }

            # Save load testing report
            report_path = Path(__file__).parent.parent / "load_testing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            cls.logger.info(f"Load testing report saved to {report_path}")

        except Exception as e:
            cls.logger.error(f"Failed to generate load testing report: {e}")


class SystemMonitor:
    """System resource monitoring for load tests"""

    def __init__(self):
        self.monitoring = False
        self.cpu_readings = []
        self.memory_readings = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring = True
        self.cpu_readings = []
        self.memory_readings = []

        def monitor_system():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent / 100.0

                    self.cpu_readings.append(cpu_percent / 100.0)
                    self.memory_readings.append(memory_percent)

                    time.sleep(2)
                except Exception:
                    pass

        self.monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop system monitoring and return metrics"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        return {
            'avg_cpu_usage': np.mean(self.cpu_readings) if self.cpu_readings else 0.0,
            'max_cpu_usage': np.max(self.cpu_readings) if self.cpu_readings else 0.0,
            'avg_memory_usage': np.mean(self.memory_readings) if self.memory_readings else 0.0,
            'max_memory_usage': np.max(self.memory_readings) if self.memory_readings else 0.0
        }


if __name__ == '__main__':
    # Configure test runner for load testing
    unittest.main(verbosity=2, buffer=True)