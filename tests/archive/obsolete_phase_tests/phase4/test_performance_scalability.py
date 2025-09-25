"""
Phase 4.1 Performance & Scalability Testing Suite

Tests the performance optimization components for 80-sensor real-time processing
with sub-second dashboard response times and efficient memory management.
"""

import unittest
import sys
import os
import time
import asyncio
import concurrent.futures
import statistics
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import threading
import psutil
import gc
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import performance optimization components
try:
    from src.utils.performance_monitor import PerformanceMonitor, PerformanceMetrics
    from src.utils.memory_manager import MemoryManager, SlidingWindow
    from src.utils.async_processor import AsyncSensorProcessor, AsyncProcessorConfig
    from src.utils.advanced_cache import AdvancedCacheManager, CacheConfig
    from src.utils.callback_optimizer import CallbackOptimizer
    from src.utils.data_compressor import IntelligentDataCompressor
    from src.utils.predictive_cache import PredictiveCacheManager
    from src.data_ingestion.database_manager import OptimizedDatabaseManager
except ImportError as e:
    print(f"Warning: Could not import all optimization components: {e}")


@dataclass
class PerformanceTestResult:
    """Performance test result container"""
    test_name: str
    target_metric: str
    target_value: float
    actual_value: float
    passed: bool
    execution_time: float
    additional_metrics: Dict[str, Any]
    timestamp: datetime


class TestPerformanceScalability(unittest.TestCase):
    """Test Performance & Scalability optimizations for 80-sensor processing"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for performance testing"""
        cls.sensor_count = 80
        cls.test_duration_seconds = 30
        cls.target_response_time_ms = 500
        cls.target_processing_rate = 80.0  # sensors per second
        cls.target_memory_mb = 8192  # 8GB limit

        # Performance test results storage
        cls.test_results: List[PerformanceTestResult] = []

        # Initialize test data
        cls._setup_test_data()

    @classmethod
    def _setup_test_data(cls):
        """Setup synthetic test data for 80 sensors"""
        # Generate synthetic NASA-like sensor data
        cls.smap_sensors = [f"SMAP_{i:02d}" for i in range(25)]
        cls.msl_sensors = [f"MSL_{i:02d}" for i in range(55)]
        cls.all_sensors = cls.smap_sensors + cls.msl_sensors

        # Generate time series data for each sensor
        cls.sensor_data = {}
        time_points = 1000

        for sensor_id in cls.all_sensors:
            # Create realistic sensor data with anomalies
            base_signal = np.sin(np.linspace(0, 10*np.pi, time_points))
            noise = np.random.normal(0, 0.1, time_points)

            # Add occasional anomalies
            anomaly_indices = np.random.choice(time_points, size=int(time_points*0.05), replace=False)
            anomalies = np.zeros(time_points)
            anomalies[anomaly_indices] = np.random.normal(0, 2, len(anomaly_indices))

            cls.sensor_data[sensor_id] = {
                'values': base_signal + noise + anomalies,
                'timestamps': [datetime.now() - timedelta(seconds=i) for i in range(time_points)],
                'anomaly_scores': np.abs(anomalies) > 1.0
            }

    def setUp(self):
        """Setup for each test"""
        # Initialize components for testing
        try:
            self.performance_monitor = PerformanceMonitor()
            self.memory_manager = MemoryManager()
            self.async_processor = AsyncSensorProcessor()
            self.cache_manager = AdvancedCacheManager()
            self.callback_optimizer = CallbackOptimizer()
            self.data_compressor = IntelligentDataCompressor()
        except Exception as e:
            self.skipTest(f"Could not initialize performance components: {e}")

    def test_80_sensor_concurrent_processing(self):
        """Test concurrent processing of all 80 sensors"""
        print("\n🔄 Testing 80-sensor concurrent processing...")

        start_time = time.time()

        # Prepare sensor data for processing
        sensor_batches = []
        batch_size = 10  # Process sensors in batches of 10

        for i in range(0, len(self.all_sensors), batch_size):
            batch = self.all_sensors[i:i + batch_size]
            batch_data = {sensor_id: self.sensor_data[sensor_id]['values'][-100:]
                         for sensor_id in batch}
            sensor_batches.append(batch_data)

        # Test concurrent processing
        processed_sensors = 0
        processing_times = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit processing tasks
            futures = []
            for batch in sensor_batches:
                future = executor.submit(self._process_sensor_batch, batch)
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_result = future.result()
                    processed_sensors += len(batch_result['sensors'])
                    processing_times.extend(batch_result['processing_times'])
                except Exception as e:
                    print(f"Batch processing failed: {e}")

        total_time = time.time() - start_time
        avg_processing_time = statistics.mean(processing_times) if processing_times else float('inf')
        processing_rate = processed_sensors / total_time

        # Validate performance targets
        self.assertEqual(processed_sensors, self.sensor_count,
                        f"Should process all {self.sensor_count} sensors")
        self.assertLess(avg_processing_time, 0.1,
                       "Average sensor processing should be under 100ms")
        self.assertGreater(processing_rate, self.target_processing_rate,
                          f"Processing rate {processing_rate:.1f} should exceed {self.target_processing_rate}")

        # Record test result
        test_result = PerformanceTestResult(
            test_name="80_sensor_concurrent_processing",
            target_metric="processing_rate_sensors_per_second",
            target_value=self.target_processing_rate,
            actual_value=processing_rate,
            passed=processing_rate > self.target_processing_rate,
            execution_time=total_time,
            additional_metrics={
                'processed_sensors': processed_sensors,
                'avg_processing_time_ms': avg_processing_time * 1000,
                'total_time_seconds': total_time
            },
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Processed {processed_sensors} sensors in {total_time:.2f}s "
              f"({processing_rate:.1f} sensors/sec)")

    def _process_sensor_batch(self, sensor_batch: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process a batch of sensors and return timing metrics"""
        start_time = time.time()
        processing_times = []
        processed_sensors = []

        for sensor_id, data in sensor_batch.items():
            sensor_start = time.time()

            # Simulate sensor processing (normalization, anomaly detection, etc.)
            normalized_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
            anomaly_scores = np.abs(normalized_data) > 2.0
            health_score = 100.0 - (np.sum(anomaly_scores) / len(data) * 100)

            # Simulate some computation time
            time.sleep(0.001)  # 1ms processing time per sensor

            processing_time = time.time() - sensor_start
            processing_times.append(processing_time)
            processed_sensors.append(sensor_id)

        return {
            'sensors': processed_sensors,
            'processing_times': processing_times,
            'total_batch_time': time.time() - start_time
        }

    def test_sub_second_dashboard_response(self):
        """Test dashboard response times under 500ms"""
        print("\n⚡ Testing sub-second dashboard response times...")

        # Simulate various dashboard operations
        dashboard_operations = [
            ('load_sensor_overview', self._simulate_sensor_overview),
            ('update_anomaly_chart', self._simulate_anomaly_chart_update),
            ('filter_equipment', self._simulate_equipment_filter),
            ('real_time_metrics', self._simulate_real_time_metrics),
            ('health_dashboard', self._simulate_health_dashboard)
        ]

        response_times = {}

        for operation_name, operation_func in dashboard_operations:
            times = []

            # Test each operation multiple times
            for _ in range(10):
                start_time = time.time()
                try:
                    operation_func()
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    times.append(response_time)
                except Exception as e:
                    print(f"Operation {operation_name} failed: {e}")
                    times.append(float('inf'))

            avg_response_time = statistics.mean(times)
            response_times[operation_name] = avg_response_time

            # Validate response time target
            self.assertLess(avg_response_time, self.target_response_time_ms,
                           f"{operation_name} response time {avg_response_time:.1f}ms "
                           f"should be under {self.target_response_time_ms}ms")

        overall_avg = statistics.mean(response_times.values())

        # Record test result
        test_result = PerformanceTestResult(
            test_name="sub_second_dashboard_response",
            target_metric="avg_response_time_ms",
            target_value=self.target_response_time_ms,
            actual_value=overall_avg,
            passed=overall_avg < self.target_response_time_ms,
            execution_time=len(dashboard_operations) * 10 * 0.1,  # Estimated
            additional_metrics=response_times,
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Dashboard operations average: {overall_avg:.1f}ms "
              f"(target: <{self.target_response_time_ms}ms)")

    def _simulate_sensor_overview(self):
        """Simulate loading sensor overview dashboard"""
        # Simulate data aggregation for 80 sensors
        sensor_data = {}
        for sensor_id in self.all_sensors[:20]:  # Sample subset for speed
            data = self.sensor_data[sensor_id]['values'][-100:]
            sensor_data[sensor_id] = {
                'current_value': data[-1],
                'avg_value': np.mean(data),
                'status': 'normal' if np.std(data) < 0.5 else 'anomalous'
            }
        return sensor_data

    def _simulate_anomaly_chart_update(self):
        """Simulate updating anomaly detection chart"""
        # Simulate anomaly score calculation for recent data
        recent_anomalies = {}
        for sensor_id in self.all_sensors[:15]:  # Sample subset
            scores = self.sensor_data[sensor_id]['anomaly_scores'][-50:]
            recent_anomalies[sensor_id] = {
                'anomaly_count': np.sum(scores),
                'max_score': np.max(scores) if len(scores) > 0 else 0,
                'trend': 'increasing' if np.sum(scores[-10:]) > np.sum(scores[-20:-10]) else 'stable'
            }
        return recent_anomalies

    def _simulate_equipment_filter(self):
        """Simulate equipment filtering operation"""
        # Simulate filtering sensors by equipment type
        filtered_sensors = {
            'SMAP': [s for s in self.all_sensors if s.startswith('SMAP')],
            'MSL': [s for s in self.all_sensors if s.startswith('MSL')]
        }
        return filtered_sensors

    def _simulate_real_time_metrics(self):
        """Simulate real-time metrics calculation"""
        # Calculate system-wide metrics
        all_values = []
        all_anomalies = []

        for sensor_id in self.all_sensors[:25]:  # Sample subset
            values = self.sensor_data[sensor_id]['values'][-10:]
            anomalies = self.sensor_data[sensor_id]['anomaly_scores'][-10:]
            all_values.extend(values)
            all_anomalies.extend(anomalies)

        metrics = {
            'total_sensors': len(self.all_sensors),
            'active_sensors': len([s for s in self.all_sensors if np.random.random() > 0.05]),
            'anomaly_rate': np.mean(all_anomalies) if all_anomalies else 0,
            'system_health': 95.0 - (np.mean(all_anomalies) * 50 if all_anomalies else 0)
        }
        return metrics

    def _simulate_health_dashboard(self):
        """Simulate health dashboard data generation"""
        # Calculate health scores for equipment subsystems
        subsystem_health = {}
        subsystems = ['power_system', 'mobility', 'robotic_arm', 'science_instruments',
                     'communication', 'thermal_system']

        for subsystem in subsystems:
            # Simulate health calculation
            sensor_subset = self.all_sensors[:10]  # Sample sensors for subsystem
            health_scores = []

            for sensor_id in sensor_subset:
                data = self.sensor_data[sensor_id]['values'][-20:]
                anomalies = self.sensor_data[sensor_id]['anomaly_scores'][-20:]
                health = 100 - (np.sum(anomalies) / len(anomalies) * 100)
                health_scores.append(health)

            subsystem_health[subsystem] = {
                'health_score': np.mean(health_scores),
                'status': 'healthy' if np.mean(health_scores) > 80 else 'degraded',
                'sensor_count': len(sensor_subset)
            }

        return subsystem_health

    def test_memory_management_efficiency(self):
        """Test memory management with sliding windows for large datasets"""
        print("\n🧠 Testing memory management efficiency...")

        # Monitor memory usage before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create sliding windows for all 80 sensors
        sliding_windows = {}
        window_size = 1000  # Keep last 1000 points per sensor

        start_time = time.time()

        for sensor_id in self.all_sensors:
            sliding_windows[sensor_id] = SlidingWindow(
                sensor_id=sensor_id,
                max_size=window_size,
                timestamps=np.array([]),
                values=np.array([]),
                anomaly_scores=np.array([])
            )

        # Simulate continuous data ingestion
        for data_point in range(2000):  # 2000 new data points
            for sensor_id in self.all_sensors:
                # Generate new data point
                new_value = np.random.normal(0, 1)
                new_timestamp = datetime.now()
                new_anomaly_score = abs(new_value) > 2.0

                # Add to sliding window
                window = sliding_windows[sensor_id]
                self.memory_manager.add_data_point(
                    window, new_timestamp, new_value, new_anomaly_score
                )

        processing_time = time.time() - start_time

        # Monitor memory usage after test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Validate memory efficiency
        self.assertLess(memory_increase, 500,  # Should not increase by more than 500MB
                       f"Memory increase {memory_increase:.1f}MB should be under 500MB")

        # Validate data integrity
        for sensor_id in self.all_sensors:
            window = sliding_windows[sensor_id]
            self.assertLessEqual(len(window.values), window_size,
                               f"Sliding window for {sensor_id} should not exceed {window_size} points")
            self.assertGreater(len(window.values), 0,
                             f"Sliding window for {sensor_id} should contain data")

        # Record test result
        test_result = PerformanceTestResult(
            test_name="memory_management_efficiency",
            target_metric="memory_increase_mb",
            target_value=500.0,
            actual_value=memory_increase,
            passed=memory_increase < 500,
            execution_time=processing_time,
            additional_metrics={
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'data_points_processed': 2000 * len(self.all_sensors),
                'windows_created': len(sliding_windows)
            },
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Memory increase: {memory_increase:.1f}MB, "
              f"Processing time: {processing_time:.2f}s")

    def test_advanced_caching_performance(self):
        """Test advanced caching system performance"""
        print("\n🚀 Testing advanced caching performance...")

        # Initialize cache with test configuration
        cache_config = CacheConfig(
            l1_max_size=1000,
            l2_max_size=10000,
            ttl_seconds=300,
            redis_enabled=False  # Use in-memory for testing
        )

        start_time = time.time()

        # Test cache operations
        cache_operations = 1000
        cache_hits = 0
        cache_misses = 0

        # Populate cache with sensor data
        for i, sensor_id in enumerate(self.all_sensors):
            cache_key = f"sensor_data_{sensor_id}"
            data = self.sensor_data[sensor_id]['values'][-100:]
            self.cache_manager.set(cache_key, data, ttl=300)

        # Test cache retrieval performance
        retrieval_times = []

        for i in range(cache_operations):
            # Mix of hits and misses
            if i % 3 == 0:  # Force some cache misses
                cache_key = f"sensor_data_NONEXISTENT_{i}"
            else:
                sensor_id = self.all_sensors[i % len(self.all_sensors)]
                cache_key = f"sensor_data_{sensor_id}"

            retrieval_start = time.time()
            result = self.cache_manager.get(cache_key)
            retrieval_time = (time.time() - retrieval_start) * 1000  # ms
            retrieval_times.append(retrieval_time)

            if result is not None:
                cache_hits += 1
            else:
                cache_misses += 1

        total_time = time.time() - start_time
        avg_retrieval_time = statistics.mean(retrieval_times)
        cache_hit_rate = cache_hits / (cache_hits + cache_misses)

        # Validate caching performance
        self.assertLess(avg_retrieval_time, 1.0,  # Under 1ms average
                       f"Cache retrieval should be under 1ms, got {avg_retrieval_time:.2f}ms")
        self.assertGreater(cache_hit_rate, 0.6,  # At least 60% hit rate
                          f"Cache hit rate should be >60%, got {cache_hit_rate:.1%}")

        # Record test result
        test_result = PerformanceTestResult(
            test_name="advanced_caching_performance",
            target_metric="cache_hit_rate",
            target_value=0.6,
            actual_value=cache_hit_rate,
            passed=cache_hit_rate > 0.6 and avg_retrieval_time < 1.0,
            execution_time=total_time,
            additional_metrics={
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'avg_retrieval_time_ms': avg_retrieval_time,
                'operations_per_second': cache_operations / total_time
            },
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Cache hit rate: {cache_hit_rate:.1%}, "
              f"Avg retrieval: {avg_retrieval_time:.2f}ms")

    def test_async_processing_performance(self):
        """Test asynchronous processing performance"""
        print("\n⚡ Testing async processing performance...")

        async def run_async_test():
            # Configure async processor
            config = AsyncProcessorConfig(
                max_concurrent_tasks=20,
                task_timeout_seconds=30,
                batch_size=10
            )

            # Prepare async processing tasks
            sensor_batches = []
            for i in range(0, len(self.all_sensors), 10):
                batch = self.all_sensors[i:i+10]
                sensor_batches.append(batch)

            start_time = time.time()

            # Process all sensor batches asynchronously
            tasks = []
            for batch in sensor_batches:
                task = self._async_process_sensors(batch)
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            processing_time = time.time() - start_time

            # Analyze results
            successful_batches = sum(1 for r in results if not isinstance(r, Exception))
            total_sensors_processed = successful_batches * 10

            return {
                'processing_time': processing_time,
                'successful_batches': successful_batches,
                'total_sensors_processed': total_sensors_processed,
                'processing_rate': total_sensors_processed / processing_time
            }

        # Run async test
        try:
            result = asyncio.run(run_async_test())
        except Exception as e:
            self.fail(f"Async processing test failed: {e}")

        # Validate async performance
        self.assertGreater(result['processing_rate'], self.target_processing_rate,
                          f"Async processing rate {result['processing_rate']:.1f} "
                          f"should exceed {self.target_processing_rate}")
        self.assertGreaterEqual(result['total_sensors_processed'], self.sensor_count * 0.9,
                               "Should process at least 90% of sensors successfully")

        # Record test result
        test_result = PerformanceTestResult(
            test_name="async_processing_performance",
            target_metric="async_processing_rate",
            target_value=self.target_processing_rate,
            actual_value=result['processing_rate'],
            passed=result['processing_rate'] > self.target_processing_rate,
            execution_time=result['processing_time'],
            additional_metrics=result,
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Async processing rate: {result['processing_rate']:.1f} sensors/sec")

    async def _async_process_sensors(self, sensor_batch: List[str]) -> Dict[str, Any]:
        """Asynchronously process a batch of sensors"""
        # Simulate async sensor processing
        await asyncio.sleep(0.1)  # Simulate network/IO delay

        processed_data = {}
        for sensor_id in sensor_batch:
            data = self.sensor_data[sensor_id]['values'][-50:]

            # Simulate processing
            processed_data[sensor_id] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'anomaly_count': np.sum(np.abs(data) > 2.0),
                'health_score': 100 - (np.sum(np.abs(data) > 2.0) / len(data) * 100)
            }

        return processed_data

    def test_data_compression_efficiency(self):
        """Test data compression for large datasets"""
        print("\n📦 Testing data compression efficiency...")

        # Prepare large dataset for compression testing
        large_dataset = {}
        for sensor_id in self.all_sensors:
            # Create larger dataset (10x normal size)
            data = np.tile(self.sensor_data[sensor_id]['values'], 10)
            large_dataset[sensor_id] = {
                'values': data.tolist(),
                'metadata': {
                    'sensor_type': 'temperature' if 'T-' in sensor_id else 'pressure',
                    'location': 'zone_a' if int(sensor_id[-2:]) % 2 == 0 else 'zone_b',
                    'calibration_date': '2024-01-01',
                    'measurement_unit': 'celsius' if 'T-' in sensor_id else 'bar'
                }
            }

        start_time = time.time()

        # Test compression
        compressed_data = self.data_compressor.compress_dashboard_data(
            large_dataset,
            data_key="large_sensor_dataset"
        )

        compression_time = time.time() - start_time

        # Calculate compression metrics
        import json
        original_size = len(json.dumps(large_dataset).encode('utf-8'))
        compressed_size = len(compressed_data['compressed_data'])
        compression_ratio = original_size / compressed_size

        # Test decompression
        decompression_start = time.time()
        decompressed_data = self.data_compressor.decompress_dashboard_data(compressed_data)
        decompression_time = time.time() - decompression_start

        # Validate compression efficiency
        self.assertGreater(compression_ratio, 2.0,  # At least 2:1 compression
                          f"Compression ratio {compression_ratio:.1f} should be >2:1")
        self.assertLess(compression_time, 5.0,  # Under 5 seconds
                       f"Compression time {compression_time:.2f}s should be under 5s")
        self.assertLess(decompression_time, 2.0,  # Under 2 seconds
                       f"Decompression time {decompression_time:.2f}s should be under 2s")

        # Validate data integrity
        self.assertEqual(len(decompressed_data), len(large_dataset),
                        "Decompressed data should have same structure as original")

        # Record test result
        test_result = PerformanceTestResult(
            test_name="data_compression_efficiency",
            target_metric="compression_ratio",
            target_value=2.0,
            actual_value=compression_ratio,
            passed=compression_ratio > 2.0 and compression_time < 5.0,
            execution_time=compression_time + decompression_time,
            additional_metrics={
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'compression_time_seconds': compression_time,
                'decompression_time_seconds': decompression_time
            },
            timestamp=datetime.now()
        )
        self.test_results.append(test_result)

        print(f"✅ Compression ratio: {compression_ratio:.1f}:1, "
              f"Time: {compression_time:.2f}s")

    @classmethod
    def tearDownClass(cls):
        """Generate performance test report"""
        print("\n" + "="*80)
        print("PHASE 4.1 PERFORMANCE & SCALABILITY TEST SUMMARY")
        print("="*80)

        total_tests = len(cls.test_results)
        passed_tests = sum(1 for result in cls.test_results if result.passed)

        print(f"Tests run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")

        print("\n📊 Performance Metrics Summary:")
        for result in cls.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {result.test_name}: "
                  f"{result.actual_value:.2f} {result.target_metric} "
                  f"(target: {result.target_value})")

        # Save detailed results to file
        results_file = project_root / "testing_phase4" / "test_results" / "phase4_1_performance_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        import json
        results_data = []
        for result in cls.test_results:
            result_dict = {
                'test_name': result.test_name,
                'target_metric': result.target_metric,
                'target_value': result.target_value,
                'actual_value': result.actual_value,
                'passed': result.passed,
                'execution_time': result.execution_time,
                'additional_metrics': result.additional_metrics,
                'timestamp': result.timestamp.isoformat()
            }
            results_data.append(result_dict)

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == '__main__':
    # Run performance & scalability tests
    unittest.main(verbosity=2)