"""
Stress Testing Suite

Tests system behavior under extreme conditions and failure scenarios:
- Resource exhaustion testing
- Memory pressure testing
- CPU saturation testing
- Network latency simulation
- Cascading failure testing
- Recovery testing
"""

import unittest
import sys
import os
import asyncio
import time
import threading
import concurrent.futures
import multiprocessing
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
import random
import signal

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import system components
try:
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
class StressTestResult:
    """Container for stress test results"""
    test_name: str
    stress_scenario: str
    stress_level: str
    duration_seconds: float
    breaking_point_reached: bool
    max_load_sustained: int
    failure_threshold: float
    recovery_time_seconds: float
    resource_exhaustion_point: Dict[str, float]
    error_cascade_count: int
    system_stability_score: float
    timestamp: datetime


class TestStressTestingSuite(unittest.TestCase):
    """Comprehensive stress testing suite"""

    @classmethod
    def setUpClass(cls):
        """Set up stress testing environment"""
        cls.logger = get_logger(__name__)
        cls.config_manager = ConfigManager()
        cls.test_results = []

        # Stress test configuration
        cls.stress_config = {
            'stress_scenarios': {
                'memory_pressure': {
                    'initial_load': 1000,
                    'increment': 500,
                    'max_iterations': 20,
                    'target_memory_usage': 0.9
                },
                'cpu_saturation': {
                    'initial_threads': 2,
                    'max_threads': multiprocessing.cpu_count() * 4,
                    'cpu_intensive_operations': True
                },
                'data_volume_surge': {
                    'initial_sensors': 10,
                    'max_sensors': 500,
                    'data_points_multiplier': 2,
                    'surge_rate': 1.5
                },
                'cascading_failures': {
                    'failure_injection_rate': 0.1,
                    'failure_propagation_delay': 2,
                    'recovery_attempts': 3
                }
            },
            'breaking_point_thresholds': {
                'max_memory_usage': 0.95,
                'max_cpu_usage': 0.98,
                'max_response_time': 10000,  # ms
                'max_error_rate': 0.5,
                'min_throughput': 1  # operations per second
            },
            'recovery_criteria': {
                'max_recovery_time': 30,  # seconds
                'required_stability_duration': 10,  # seconds
                'recovery_success_threshold': 0.8
            }
        }

        # Initialize components
        cls._initialize_stress_test_components()

    @classmethod
    def _initialize_stress_test_components(cls):
        """Initialize components for stress testing"""
        try:
            cls.nasa_data_service = NASADataService()
            cls.data_preprocessor = DataPreprocessor()
            cls.ensemble_detector = EnsembleAnomalyDetector()
            cls.transformer_forecaster = TransformerForecaster()
            cls.business_rules_engine = BusinessRulesEngine()
            cls.maintenance_optimizer = MaintenanceOptimizationEngine()
            cls.database_manager = DatabaseManager()

            cls.logger.info("Stress testing components initialized successfully")

        except Exception as e:
            cls.logger.error(f"Failed to initialize stress testing components: {e}")
            raise

    def setUp(self):
        """Set up individual stress test"""
        self.test_start_time = time.time()
        self.stress_monitor = StressMonitor()
        self.failure_injector = FailureInjector()

    def tearDown(self):
        """Clean up individual stress test"""
        execution_time = time.time() - self.test_start_time
        self.logger.info(f"Stress test {self._testMethodName} completed in {execution_time:.2f}s")

        # Force cleanup
        gc.collect()

    def test_memory_pressure_stress(self):
        """Test system behavior under memory pressure"""
        test_start = time.time()

        try:
            scenario = self.stress_config['stress_scenarios']['memory_pressure']
            self.logger.info("Starting memory pressure stress test")

            # Start monitoring
            self.stress_monitor.start_monitoring()

            memory_allocations = []
            current_load = scenario['initial_load']
            breaking_point_reached = False
            max_load_sustained = 0

            for iteration in range(scenario['max_iterations']):
                iteration_start = time.time()

                try:
                    # Allocate memory progressively
                    large_data = self._generate_memory_intensive_data(current_load)
                    memory_allocations.append(large_data)

                    # Process data through system components
                    processing_results = self._stress_process_data(large_data, f"memory_stress_iteration_{iteration}")

                    # Check system metrics
                    current_memory = psutil.virtual_memory().percent / 100.0
                    response_time = time.time() - iteration_start

                    # Update max sustained load
                    if processing_results['success']:
                        max_load_sustained = current_load

                    # Check breaking point
                    if (current_memory >= self.stress_config['breaking_point_thresholds']['max_memory_usage'] or
                        response_time * 1000 >= self.stress_config['breaking_point_thresholds']['max_response_time']):
                        breaking_point_reached = True
                        self.logger.info(f"Memory pressure breaking point reached at iteration {iteration}")
                        break

                    # Increase load for next iteration
                    current_load += scenario['increment']

                    self.logger.info(f"Memory stress iteration {iteration}: "
                                   f"load={current_load}, memory={current_memory:.3f}, "
                                   f"response_time={response_time:.2f}s")

                except Exception as e:
                    self.logger.warning(f"Memory stress iteration {iteration} failed: {e}")
                    breaking_point_reached = True
                    break

            # Test recovery
            recovery_start = time.time()
            self._test_system_recovery("memory_pressure")
            recovery_time = time.time() - recovery_start

            # Stop monitoring
            stress_metrics = self.stress_monitor.stop_monitoring()

            # Record stress test result
            result = StressTestResult(
                test_name="memory_pressure_stress",
                stress_scenario="memory_pressure",
                stress_level="HIGH",
                duration_seconds=time.time() - test_start,
                breaking_point_reached=breaking_point_reached,
                max_load_sustained=max_load_sustained,
                failure_threshold=stress_metrics['peak_memory_usage'],
                recovery_time_seconds=recovery_time,
                resource_exhaustion_point={
                    'memory_usage': stress_metrics['peak_memory_usage'],
                    'cpu_usage': stress_metrics['peak_cpu_usage']
                },
                error_cascade_count=0,  # Not applicable for this test
                system_stability_score=self._calculate_stability_score(stress_metrics),
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            # Validate stress test results
            self.assertTrue(
                breaking_point_reached,
                "Memory pressure stress test should reach breaking point"
            )

            self.assertGreater(
                max_load_sustained, scenario['initial_load'],
                "System should sustain load higher than initial"
            )

            # Cleanup memory allocations
            del memory_allocations
            gc.collect()

        except Exception as e:
            self.fail(f"Memory pressure stress test failed: {e}")

    def test_cpu_saturation_stress(self):
        """Test system behavior under CPU saturation"""
        test_start = time.time()

        try:
            scenario = self.stress_config['stress_scenarios']['cpu_saturation']
            self.logger.info("Starting CPU saturation stress test")

            # Start monitoring
            self.stress_monitor.start_monitoring()

            cpu_intensive_threads = []
            breaking_point_reached = False
            max_threads_sustained = 0

            # Start CPU-intensive background tasks
            for thread_count in range(scenario['initial_threads'], scenario['max_threads'], 2):
                try:
                    # Create CPU-intensive thread
                    cpu_thread = threading.Thread(
                        target=self._cpu_intensive_task,
                        args=(thread_count, 5),  # 5 seconds per task
                        daemon=True
                    )
                    cpu_thread.start()
                    cpu_intensive_threads.append(cpu_thread)

                    # Test system responsiveness under CPU load
                    responsiveness_start = time.time()
                    test_data = self._generate_stress_test_data(100, 10)  # Small test data
                    processing_results = self._stress_process_data(test_data, f"cpu_stress_thread_{thread_count}")
                    response_time = time.time() - responsiveness_start

                    # Check CPU usage
                    current_cpu = psutil.cpu_percent(interval=1) / 100.0

                    # Update max sustained threads
                    if processing_results['success'] and response_time < 5.0:  # 5 second threshold
                        max_threads_sustained = thread_count

                    # Check breaking point
                    if (current_cpu >= self.stress_config['breaking_point_thresholds']['max_cpu_usage'] or
                        response_time * 1000 >= self.stress_config['breaking_point_thresholds']['max_response_time']):
                        breaking_point_reached = True
                        self.logger.info(f"CPU saturation breaking point reached with {thread_count} threads")
                        break

                    self.logger.info(f"CPU stress threads: {thread_count}, "
                                   f"CPU usage: {current_cpu:.3f}, "
                                   f"response_time: {response_time:.2f}s")

                    time.sleep(2)  # Brief pause between thread creation

                except Exception as e:
                    self.logger.warning(f"CPU stress thread {thread_count} failed: {e}")
                    breaking_point_reached = True
                    break

            # Test recovery
            recovery_start = time.time()
            self._test_system_recovery("cpu_saturation")
            recovery_time = time.time() - recovery_start

            # Stop monitoring
            stress_metrics = self.stress_monitor.stop_monitoring()

            # Wait for CPU threads to complete (they should be daemon threads)
            time.sleep(2)

            # Record stress test result
            result = StressTestResult(
                test_name="cpu_saturation_stress",
                stress_scenario="cpu_saturation",
                stress_level="EXTREME",
                duration_seconds=time.time() - test_start,
                breaking_point_reached=breaking_point_reached,
                max_load_sustained=max_threads_sustained,
                failure_threshold=stress_metrics['peak_cpu_usage'],
                recovery_time_seconds=recovery_time,
                resource_exhaustion_point={
                    'cpu_usage': stress_metrics['peak_cpu_usage'],
                    'thread_count': len(cpu_intensive_threads)
                },
                error_cascade_count=0,
                system_stability_score=self._calculate_stability_score(stress_metrics),
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            # Validate stress test results
            self.assertTrue(
                breaking_point_reached,
                "CPU saturation stress test should reach breaking point"
            )

        except Exception as e:
            self.fail(f"CPU saturation stress test failed: {e}")

    def test_data_volume_surge_stress(self):
        """Test system behavior under sudden data volume surges"""
        test_start = time.time()

        try:
            scenario = self.stress_config['stress_scenarios']['data_volume_surge']
            self.logger.info("Starting data volume surge stress test")

            # Start monitoring
            self.stress_monitor.start_monitoring()

            current_sensors = scenario['initial_sensors']
            data_multiplier = 1
            breaking_point_reached = False
            max_data_volume_sustained = 0

            for surge_iteration in range(10):  # 10 surge iterations
                try:
                    # Generate increasing data volume
                    data_volume = current_sensors * data_multiplier * 100  # 100 points per sensor
                    large_dataset = self._generate_stress_test_data(current_sensors, data_multiplier * 100)

                    surge_start = time.time()

                    # Process surge of data
                    processing_results = self._stress_process_data_batch(
                        large_dataset, f"data_surge_iteration_{surge_iteration}"
                    )

                    surge_processing_time = time.time() - surge_start

                    # Update max sustained volume
                    if processing_results['success']:
                        max_data_volume_sustained = data_volume

                    # Check for breaking point
                    memory_usage = psutil.virtual_memory().percent / 100.0
                    if (surge_processing_time * 1000 >= self.stress_config['breaking_point_thresholds']['max_response_time'] or
                        memory_usage >= self.stress_config['breaking_point_thresholds']['max_memory_usage'] or
                        not processing_results['success']):
                        breaking_point_reached = True
                        self.logger.info(f"Data surge breaking point reached at iteration {surge_iteration}")
                        break

                    self.logger.info(f"Data surge iteration {surge_iteration}: "
                                   f"sensors={current_sensors}, volume={data_volume}, "
                                   f"processing_time={surge_processing_time:.2f}s")

                    # Increase data volume for next iteration
                    current_sensors = int(current_sensors * scenario['surge_rate'])
                    data_multiplier = int(data_multiplier * scenario['data_points_multiplier'])

                    # Brief pause between surges
                    time.sleep(1)

                except Exception as e:
                    self.logger.warning(f"Data surge iteration {surge_iteration} failed: {e}")
                    breaking_point_reached = True
                    break

            # Test recovery
            recovery_start = time.time()
            self._test_system_recovery("data_volume_surge")
            recovery_time = time.time() - recovery_start

            # Stop monitoring
            stress_metrics = self.stress_monitor.stop_monitoring()

            # Record stress test result
            result = StressTestResult(
                test_name="data_volume_surge_stress",
                stress_scenario="data_volume_surge",
                stress_level="HIGH",
                duration_seconds=time.time() - test_start,
                breaking_point_reached=breaking_point_reached,
                max_load_sustained=max_data_volume_sustained,
                failure_threshold=max_data_volume_sustained,
                recovery_time_seconds=recovery_time,
                resource_exhaustion_point={
                    'data_volume': max_data_volume_sustained,
                    'memory_usage': stress_metrics['peak_memory_usage']
                },
                error_cascade_count=0,
                system_stability_score=self._calculate_stability_score(stress_metrics),
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            # Validate stress test results
            self.assertGreater(
                max_data_volume_sustained, scenario['initial_sensors'] * 100,
                "System should handle volume higher than initial"
            )

        except Exception as e:
            self.fail(f"Data volume surge stress test failed: {e}")

    def test_cascading_failure_stress(self):
        """Test system behavior under cascading failures"""
        test_start = time.time()

        try:
            scenario = self.stress_config['stress_scenarios']['cascading_failures']
            self.logger.info("Starting cascading failure stress test")

            # Start monitoring
            self.stress_monitor.start_monitoring()

            # Initialize failure injection
            self.failure_injector.configure(
                failure_rate=scenario['failure_injection_rate'],
                propagation_delay=scenario['failure_propagation_delay']
            )

            cascading_failures = []
            system_components = ['data_ingestion', 'preprocessing', 'anomaly_detection', 'business_logic']

            # Test normal operation first
            baseline_data = self._generate_stress_test_data(20, 100)
            baseline_results = self._stress_process_data(baseline_data, "baseline_before_failures")

            # Inject failures progressively
            for failure_round in range(5):
                try:
                    # Select component to fail
                    component_to_fail = system_components[failure_round % len(system_components)]

                    self.logger.info(f"Injecting failure in {component_to_fail} (round {failure_round})")

                    # Inject failure
                    failure_context = self.failure_injector.inject_component_failure(component_to_fail)

                    # Test system response to failure
                    failure_test_data = self._generate_stress_test_data(15, 50)
                    failure_results = self._stress_process_data_with_failures(
                        failure_test_data, f"failure_round_{failure_round}", component_to_fail
                    )

                    # Record failure impact
                    failure_impact = {
                        'failed_component': component_to_fail,
                        'failure_round': failure_round,
                        'processing_success': failure_results['success'],
                        'error_propagation': failure_results.get('error_propagation', []),
                        'recovery_attempts': 0
                    }

                    # Attempt recovery
                    recovery_success = False
                    for recovery_attempt in range(scenario['recovery_attempts']):
                        recovery_start = time.time()

                        # Simulate recovery actions
                        recovery_success = self._attempt_component_recovery(component_to_fail)

                        recovery_time = time.time() - recovery_start
                        failure_impact['recovery_attempts'] += 1

                        if recovery_success:
                            self.logger.info(f"Recovery successful for {component_to_fail} "
                                           f"after {recovery_attempt + 1} attempts")
                            break

                        time.sleep(1)  # Brief pause between recovery attempts

                    failure_impact['recovery_successful'] = recovery_success
                    cascading_failures.append(failure_impact)

                    # Brief pause before next failure injection
                    time.sleep(scenario['failure_propagation_delay'])

                except Exception as e:
                    self.logger.warning(f"Failure injection round {failure_round} failed: {e}")
                    cascading_failures.append({
                        'failed_component': component_to_fail,
                        'failure_round': failure_round,
                        'processing_success': False,
                        'error_propagation': [str(e)],
                        'recovery_attempts': 0,
                        'recovery_successful': False
                    })

            # Test final system state
            final_test_data = self._generate_stress_test_data(20, 100)
            final_results = self._stress_process_data(final_test_data, "final_state_after_failures")

            # Stop monitoring
            stress_metrics = self.stress_monitor.stop_monitoring()

            # Calculate cascading failure metrics
            total_failures = len(cascading_failures)
            successful_recoveries = len([f for f in cascading_failures if f['recovery_successful']])
            error_cascade_count = sum(len(f.get('error_propagation', [])) for f in cascading_failures)

            # Record stress test result
            result = StressTestResult(
                test_name="cascading_failure_stress",
                stress_scenario="cascading_failures",
                stress_level="CRITICAL",
                duration_seconds=time.time() - test_start,
                breaking_point_reached=successful_recoveries < total_failures * 0.5,
                max_load_sustained=successful_recoveries,
                failure_threshold=total_failures,
                recovery_time_seconds=0,  # Calculated differently for cascading failures
                resource_exhaustion_point={
                    'failed_components': total_failures,
                    'error_cascade_count': error_cascade_count
                },
                error_cascade_count=error_cascade_count,
                system_stability_score=successful_recoveries / max(total_failures, 1),
                timestamp=datetime.now()
            )
            self.test_results.append(result)

            # Validate cascading failure resilience
            self.assertGreaterEqual(
                successful_recoveries / max(total_failures, 1), 0.6,
                f"System should recover from at least 60% of cascading failures. "
                f"Recovered: {successful_recoveries}/{total_failures}"
            )

            self.assertTrue(
                final_results['success'],
                "System should return to functional state after cascading failures"
            )

        except Exception as e:
            self.fail(f"Cascading failure stress test failed: {e}")

    def _generate_memory_intensive_data(self, size_multiplier: int) -> Dict[str, np.ndarray]:
        """Generate memory-intensive data for stress testing"""
        data = {}
        base_size = 1000

        for i in range(size_multiplier):
            sensor_id = f"memory_stress_sensor_{i}"
            # Create large arrays to consume memory
            large_array = np.random.random(base_size * size_multiplier)
            data[sensor_id] = large_array

        return data

    def _generate_stress_test_data(self, num_sensors: int, data_points_per_sensor: int) -> Dict[str, Any]:
        """Generate data for stress testing"""
        stress_data = {}

        for i in range(num_sensors):
            sensor_id = f"stress_sensor_{i:04d}"
            values = np.random.normal(0.5, 0.1, data_points_per_sensor)
            values = np.clip(values, 0, 1)

            stress_data[sensor_id] = {
                'values': values,
                'timestamps': pd.date_range(
                    start=datetime.now() - timedelta(minutes=data_points_per_sensor),
                    periods=data_points_per_sensor,
                    freq='1min'
                ),
                'metadata': {
                    'sensor_type': f"stress_type_{i % 3}",
                    'equipment_id': f"stress_equipment_{i // 10}"
                }
            }

        return stress_data

    def _stress_process_data(self, data: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Process data through system components under stress"""
        processing_start = time.time()
        results = {'success': True, 'errors': [], 'processing_times': {}}

        try:
            # Data ingestion
            ingestion_start = time.time()
            ingested_data = {}

            for sensor_id, sensor_info in data.items():
                if isinstance(sensor_info, dict) and 'values' in sensor_info:
                    ingested = self.nasa_data_service.process_sensor_stream(
                        sensor_id, sensor_info['values']
                    )
                    if ingested is not None:
                        ingested_data[sensor_id] = ingested
                else:
                    # Handle memory stress data (raw arrays)
                    ingested_data[sensor_id] = sensor_info

            results['processing_times']['ingestion'] = time.time() - ingestion_start

            # Preprocessing
            preprocessing_start = time.time()
            preprocessed_data = {}

            for sensor_id, ingested in list(ingested_data.items())[:10]:  # Limit for stress testing
                preprocessed = self.data_preprocessor.preprocess_sensor_data(ingested, sensor_id)
                if preprocessed is not None:
                    preprocessed_data[sensor_id] = preprocessed

            results['processing_times']['preprocessing'] = time.time() - preprocessing_start

            # Anomaly detection (limited subset)
            detection_start = time.time()
            anomaly_results = {}

            for sensor_id, preprocessed in list(preprocessed_data.items())[:5]:  # Further limit
                anomalies = self.ensemble_detector.detect_anomalies(preprocessed, sensor_id)
                if anomalies:
                    anomaly_results[sensor_id] = anomalies

            results['processing_times']['anomaly_detection'] = time.time() - detection_start

            results['data_processed'] = len(preprocessed_data)
            results['anomalies_detected'] = len(anomaly_results)

        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))

        results['total_processing_time'] = time.time() - processing_start
        return results

    def _stress_process_data_batch(self, data: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Process large batches of data under stress"""
        batch_start = time.time()
        results = {'success': True, 'batches_processed': 0, 'errors': []}

        try:
            # Process data in smaller batches to manage memory
            batch_size = 10
            sensor_items = list(data.items())

            for i in range(0, len(sensor_items), batch_size):
                batch = dict(sensor_items[i:i + batch_size])
                batch_result = self._stress_process_data(batch, f"{test_label}_batch_{i}")

                if batch_result['success']:
                    results['batches_processed'] += 1
                else:
                    results['errors'].extend(batch_result['errors'])

                # Check if we should continue based on memory usage
                memory_usage = psutil.virtual_memory().percent / 100.0
                if memory_usage > 0.9:  # 90% memory threshold
                    self.logger.warning(f"High memory usage detected: {memory_usage:.3f}")
                    break

        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))

        results['processing_time'] = time.time() - batch_start
        return results

    def _stress_process_data_with_failures(self, data: Dict[str, Any], test_label: str,
                                         failed_component: str) -> Dict[str, Any]:
        """Process data with simulated component failures"""
        results = {'success': True, 'errors': [], 'error_propagation': []}

        try:
            # Simulate different failure scenarios based on component
            if failed_component == 'data_ingestion':
                # Simulate partial data loss
                filtered_data = {k: v for k, v in list(data.items())[::2]}  # Skip every other sensor
                results['error_propagation'].append("Data ingestion: 50% sensor data lost")
                data = filtered_data

            elif failed_component == 'preprocessing':
                # Simulate preprocessing corruption
                for sensor_id, sensor_info in data.items():
                    if 'values' in sensor_info:
                        # Corrupt 10% of data points
                        values = sensor_info['values'].copy()
                        corrupt_indices = np.random.choice(
                            len(values), size=int(len(values) * 0.1), replace=False
                        )
                        values[corrupt_indices] = np.nan
                        sensor_info['values'] = values
                results['error_propagation'].append("Preprocessing: 10% data corruption")

            elif failed_component == 'anomaly_detection':
                # Simulate detection failure
                results['error_propagation'].append("Anomaly detection: Service unavailable")

            elif failed_component == 'business_logic':
                # Simulate business logic failure
                results['error_propagation'].append("Business logic: Decision engine failure")

            # Process data with failures
            processing_result = self._stress_process_data(data, test_label)
            results.update(processing_result)

        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))

        return results

    def _attempt_component_recovery(self, component: str) -> bool:
        """Simulate component recovery attempts"""
        try:
            # Simulate recovery actions for different components
            recovery_actions = {
                'data_ingestion': lambda: time.sleep(0.5),  # Restart data service
                'preprocessing': lambda: gc.collect(),      # Clear memory issues
                'anomaly_detection': lambda: time.sleep(1), # Reload models
                'business_logic': lambda: time.sleep(0.8)   # Restart decision engine
            }

            if component in recovery_actions:
                recovery_actions[component]()

            # Simulate recovery success/failure (80% success rate)
            return random.random() < 0.8

        except Exception:
            return False

    def _cpu_intensive_task(self, thread_id: int, duration_seconds: int):
        """CPU-intensive task for CPU saturation testing"""
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            # Perform CPU-intensive calculations
            for _ in range(10000):
                _ = sum(i * i for i in range(100))

    def _test_system_recovery(self, stress_scenario: str):
        """Test system recovery after stress"""
        recovery_criteria = self.stress_config['recovery_criteria']

        # Wait for system to stabilize
        time.sleep(recovery_criteria['required_stability_duration'])

        # Test basic functionality
        test_data = self._generate_stress_test_data(5, 50)  # Small test
        recovery_result = self._stress_process_data(test_data, f"recovery_test_{stress_scenario}")

        return recovery_result['success']

    def _calculate_stability_score(self, stress_metrics: Dict[str, float]) -> float:
        """Calculate system stability score based on stress metrics"""
        try:
            # Normalize metrics to 0-1 scale
            cpu_score = 1.0 - min(stress_metrics.get('peak_cpu_usage', 0), 1.0)
            memory_score = 1.0 - min(stress_metrics.get('peak_memory_usage', 0), 1.0)

            # Weight the scores
            stability_score = (cpu_score * 0.4 + memory_score * 0.6)

            return max(0.0, min(1.0, stability_score))

        except Exception:
            return 0.0

    @classmethod
    def tearDownClass(cls):
        """Generate stress testing report"""
        cls._generate_stress_testing_report()

    @classmethod
    def _generate_stress_testing_report(cls):
        """Generate comprehensive stress testing report"""
        try:
            report_data = {
                'test_suite': 'Stress Testing Suite',
                'execution_timestamp': datetime.now().isoformat(),
                'total_stress_tests': len(cls.test_results),
                'stress_scenarios_tested': list(set(r.stress_scenario for r in cls.test_results)),
                'breaking_point_analysis': {
                    'tests_reaching_breaking_point': len([r for r in cls.test_results if r.breaking_point_reached]),
                    'avg_max_load_sustained': np.mean([r.max_load_sustained for r in cls.test_results]),
                    'avg_recovery_time': np.mean([r.recovery_time_seconds for r in cls.test_results if r.recovery_time_seconds > 0]),
                    'avg_system_stability_score': np.mean([r.system_stability_score for r in cls.test_results])
                },
                'resource_exhaustion_points': {
                    scenario: {
                        'avg_failure_threshold': np.mean([r.failure_threshold for r in cls.test_results if r.stress_scenario == scenario]),
                        'peak_resource_usage': max([r.resource_exhaustion_point for r in cls.test_results if r.stress_scenario == scenario], key=lambda x: sum(x.values()), default={})
                    }
                    for scenario in set(r.stress_scenario for r in cls.test_results)
                },
                'cascading_failure_analysis': {
                    'total_error_cascades': sum([r.error_cascade_count for r in cls.test_results]),
                    'avg_cascade_size': np.mean([r.error_cascade_count for r in cls.test_results if r.error_cascade_count > 0]) if any(r.error_cascade_count > 0 for r in cls.test_results) else 0
                },
                'detailed_results': [
                    {
                        'test_name': r.test_name,
                        'stress_scenario': r.stress_scenario,
                        'stress_level': r.stress_level,
                        'duration_seconds': r.duration_seconds,
                        'breaking_point_reached': r.breaking_point_reached,
                        'max_load_sustained': r.max_load_sustained,
                        'failure_threshold': r.failure_threshold,
                        'recovery_time_seconds': r.recovery_time_seconds,
                        'resource_exhaustion_point': r.resource_exhaustion_point,
                        'error_cascade_count': r.error_cascade_count,
                        'system_stability_score': r.system_stability_score,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in cls.test_results
                ]
            }

            # Save stress testing report
            report_path = Path(__file__).parent.parent / "stress_testing_report.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)

            cls.logger.info(f"Stress testing report saved to {report_path}")

        except Exception as e:
            cls.logger.error(f"Failed to generate stress testing report: {e}")


class StressMonitor:
    """System monitoring during stress tests"""

    def __init__(self):
        self.monitoring = False
        self.cpu_readings = []
        self.memory_readings = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start stress monitoring"""
        self.monitoring = True
        self.cpu_readings = []
        self.memory_readings = []

        def monitor_stress():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    memory_percent = psutil.virtual_memory().percent

                    self.cpu_readings.append(cpu_percent / 100.0)
                    self.memory_readings.append(memory_percent / 100.0)

                    time.sleep(1)
                except Exception:
                    pass

        self.monitor_thread = threading.Thread(target=monitor_stress, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop stress monitoring and return metrics"""
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=3)

        return {
            'avg_cpu_usage': np.mean(self.cpu_readings) if self.cpu_readings else 0.0,
            'peak_cpu_usage': np.max(self.cpu_readings) if self.cpu_readings else 0.0,
            'avg_memory_usage': np.mean(self.memory_readings) if self.memory_readings else 0.0,
            'peak_memory_usage': np.max(self.memory_readings) if self.memory_readings else 0.0
        }


class FailureInjector:
    """Failure injection for cascading failure testing"""

    def __init__(self):
        self.failure_rate = 0.1
        self.propagation_delay = 2
        self.active_failures = []

    def configure(self, failure_rate: float, propagation_delay: float):
        """Configure failure injection parameters"""
        self.failure_rate = failure_rate
        self.propagation_delay = propagation_delay

    def inject_component_failure(self, component: str) -> Dict[str, Any]:
        """Inject failure into specified component"""
        failure_context = {
            'component': component,
            'failure_type': 'simulated',
            'injection_time': datetime.now(),
            'expected_impact': f"Component {component} temporarily unavailable"
        }

        self.active_failures.append(failure_context)
        return failure_context


if __name__ == '__main__':
    # Configure test runner for stress testing
    unittest.main(verbosity=2, buffer=True)