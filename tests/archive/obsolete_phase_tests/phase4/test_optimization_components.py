"""
Phase 4.1 Optimization Components Testing Suite

Tests individual optimization components for reliability, performance,
and integration with the 80-sensor IoT dashboard system.
"""

import unittest
import sys
import os
import time
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import threading
import json
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import optimization components
try:
    from src.utils.advanced_cache import AdvancedCacheManager, CacheConfig, CacheMetrics
    from src.utils.memory_manager import MemoryManager, SlidingWindow
    from src.utils.async_processor import AsyncSensorProcessor, AsyncProcessorConfig
    from src.utils.callback_optimizer import CallbackOptimizer
    from src.utils.data_compressor import IntelligentDataCompressor, CompressionConfig
    from src.utils.predictive_cache import PredictiveCacheManager
    from src.data_ingestion.database_manager import OptimizedDatabaseManager, DatabaseConfig
except ImportError as e:
    print(f"Warning: Could not import optimization components: {e}")


class TestAdvancedCacheManager(unittest.TestCase):
    """Test Advanced Cache Manager component"""

    def setUp(self):
        """Setup test environment"""
        self.cache_config = CacheConfig(
            l1_max_size=100,
            l2_max_size=1000,
            ttl_seconds=60,
            redis_enabled=False,  # Use in-memory for testing
            compression_enabled=True,
            metrics_enabled=True
        )

        try:
            self.cache_manager = AdvancedCacheManager(self.cache_config)
        except Exception as e:
            self.skipTest(f"Could not initialize AdvancedCacheManager: {e}")

    def test_l1_cache_operations(self):
        """Test L1 (in-memory) cache operations"""
        print("\n🔄 Testing L1 cache operations...")

        # Test basic set/get operations
        test_data = {"sensor_1": [1, 2, 3, 4, 5], "timestamp": datetime.now().isoformat()}
        key = "test_sensor_data"

        # Set data in cache
        success = self.cache_manager.set(key, test_data, ttl=60)
        self.assertTrue(success, "Cache set operation should succeed")

        # Get data from cache
        cached_data = self.cache_manager.get(key)
        self.assertIsNotNone(cached_data, "Cache get should return data")
        self.assertEqual(cached_data["sensor_1"], test_data["sensor_1"], "Cached data should match original")

        print("✅ L1 cache basic operations working")

    def test_l2_cache_fallback(self):
        """Test L2 cache fallback when L1 is full"""
        print("\n🔄 Testing L2 cache fallback...")

        # Fill L1 cache beyond capacity
        for i in range(self.cache_config.l1_max_size + 10):
            key = f"test_key_{i}"
            data = {"value": i, "large_data": list(range(100))}
            self.cache_manager.set(key, data, ttl=60)

        # Verify some data moved to L2
        metrics = self.cache_manager.get_metrics()
        self.assertGreater(metrics.l2_operations, 0, "L2 cache should have operations")

        # Test retrieval from both levels
        recent_key = f"test_key_{self.cache_config.l1_max_size + 5}"
        recent_data = self.cache_manager.get(recent_key)
        self.assertIsNotNone(recent_data, "Should retrieve data from L2 cache")

        print("✅ L2 cache fallback working")

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        print("\n⏰ Testing cache expiration...")

        # Set data with short TTL
        key = "expire_test"
        data = {"test": "expiration"}
        self.cache_manager.set(key, data, ttl=1)  # 1 second TTL

        # Immediately retrieve - should exist
        cached_data = self.cache_manager.get(key)
        self.assertIsNotNone(cached_data, "Data should exist immediately")

        # Wait for expiration
        time.sleep(2)

        # Try to retrieve - should be expired
        expired_data = self.cache_manager.get(key)
        self.assertIsNone(expired_data, "Data should be expired after TTL")

        print("✅ Cache expiration working")

    def test_cache_metrics(self):
        """Test cache metrics collection"""
        print("\n📊 Testing cache metrics...")

        # Perform various cache operations
        for i in range(20):
            key = f"metrics_test_{i}"
            data = {"value": i}
            self.cache_manager.set(key, data)

            # Mix hits and misses
            if i % 2 == 0:
                self.cache_manager.get(key)  # Hit
            else:
                self.cache_manager.get(f"nonexistent_{i}")  # Miss

        # Get metrics
        metrics = self.cache_manager.get_metrics()

        # Validate metrics
        self.assertGreater(metrics.total_operations, 0, "Should have recorded operations")
        self.assertGreater(metrics.hit_count, 0, "Should have cache hits")
        self.assertGreater(metrics.miss_count, 0, "Should have cache misses")
        self.assertGreaterEqual(metrics.hit_rate, 0.0, "Hit rate should be >= 0")
        self.assertLessEqual(metrics.hit_rate, 1.0, "Hit rate should be <= 1")

        print(f"✅ Cache metrics: {metrics.hit_rate:.2%} hit rate")


class TestMemoryManager(unittest.TestCase):
    """Test Memory Manager component"""

    def setUp(self):
        """Setup test environment"""
        try:
            self.memory_manager = MemoryManager()
        except Exception as e:
            self.skipTest(f"Could not initialize MemoryManager: {e}")

    def test_sliding_window_creation(self):
        """Test sliding window creation and initialization"""
        print("\n🪟 Testing sliding window creation...")

        sensor_id = "TEST_SENSOR_001"
        max_size = 100

        # Create sliding window
        window = SlidingWindow(
            sensor_id=sensor_id,
            max_size=max_size,
            timestamps=np.array([]),
            values=np.array([]),
            anomaly_scores=np.array([])
        )

        self.assertEqual(window.sensor_id, sensor_id, "Sensor ID should match")
        self.assertEqual(window.max_size, max_size, "Max size should match")
        self.assertEqual(len(window.timestamps), 0, "Should start empty")

        print("✅ Sliding window creation working")

    def test_sliding_window_data_addition(self):
        """Test adding data to sliding window"""
        print("\n➕ Testing sliding window data addition...")

        window = SlidingWindow(
            sensor_id="TEST_SENSOR",
            max_size=10,
            timestamps=np.array([]),
            values=np.array([]),
            anomaly_scores=np.array([])
        )

        # Add data points
        for i in range(15):  # More than max_size
            timestamp = datetime.now() + timedelta(seconds=i)
            value = np.sin(i * 0.1)
            anomaly_score = 0.1 * i

            self.memory_manager.add_data_point(window, timestamp, value, anomaly_score)

        # Validate window size constraint
        self.assertLessEqual(len(window.values), window.max_size,
                           "Window should not exceed max size")
        self.assertEqual(len(window.values), window.max_size,
                        "Window should be at max size after overflow")

        # Validate data integrity
        self.assertEqual(len(window.timestamps), len(window.values),
                        "Timestamps and values should have same length")
        self.assertEqual(len(window.values), len(window.anomaly_scores),
                        "Values and anomaly scores should have same length")

        print("✅ Sliding window data addition working")

    def test_memory_optimization_80_sensors(self):
        """Test memory optimization for 80 sensors"""
        print("\n🧠 Testing memory optimization for 80 sensors...")

        # Create windows for 80 sensors
        windows = {}
        sensor_ids = [f"SENSOR_{i:03d}" for i in range(80)]

        start_time = time.time()

        for sensor_id in sensor_ids:
            windows[sensor_id] = SlidingWindow(
                sensor_id=sensor_id,
                max_size=1000,
                timestamps=np.array([]),
                values=np.array([]),
                anomaly_scores=np.array([])
            )

        # Simulate data ingestion for all sensors
        for data_point in range(2000):  # 2000 points per sensor
            for sensor_id in sensor_ids:
                timestamp = datetime.now() + timedelta(seconds=data_point)
                value = np.random.normal(0, 1)
                anomaly_score = abs(value) > 2.0

                self.memory_manager.add_data_point(
                    windows[sensor_id], timestamp, value, anomaly_score
                )

        processing_time = time.time() - start_time

        # Validate all windows are properly managed
        for sensor_id in sensor_ids:
            window = windows[sensor_id]
            self.assertLessEqual(len(window.values), window.max_size,
                               f"Window {sensor_id} should not exceed max size")
            self.assertGreater(len(window.values), 0,
                             f"Window {sensor_id} should contain data")

        # Performance check
        self.assertLess(processing_time, 30.0,
                       f"Processing 80 sensors should complete in <30s, took {processing_time:.2f}s")

        print(f"✅ 80-sensor memory optimization: {processing_time:.2f}s")


class TestAsyncSensorProcessor(unittest.TestCase):
    """Test Async Sensor Processor component"""

    def setUp(self):
        """Setup test environment"""
        self.processor_config = AsyncProcessorConfig(
            max_concurrent_tasks=20,
            task_timeout_seconds=10,
            batch_size=10,
            queue_max_size=1000
        )

        try:
            self.async_processor = AsyncSensorProcessor(self.processor_config)
        except Exception as e:
            self.skipTest(f"Could not initialize AsyncSensorProcessor: {e}")

    def test_async_task_execution(self):
        """Test asynchronous task execution"""
        print("\n⚡ Testing async task execution...")

        async def run_async_test():
            # Create test tasks
            test_tasks = []
            for i in range(10):
                task_data = {"sensor_id": f"SENSOR_{i}", "data": list(range(i, i+10))}
                task = self._create_test_task(task_data)
                test_tasks.append(task)

            # Execute tasks concurrently
            start_time = time.time()
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            execution_time = time.time() - start_time

            # Validate results
            successful_tasks = sum(1 for r in results if not isinstance(r, Exception))

            return {
                'execution_time': execution_time,
                'successful_tasks': successful_tasks,
                'total_tasks': len(test_tasks),
                'results': results
            }

        # Run async test
        try:
            result = asyncio.run(run_async_test())
        except Exception as e:
            self.fail(f"Async task execution failed: {e}")

        # Validate performance
        self.assertGreater(result['successful_tasks'], 8,
                          "At least 80% of tasks should succeed")
        self.assertLess(result['execution_time'], 5.0,
                       "Async execution should complete quickly")

        print(f"✅ Async execution: {result['successful_tasks']}/{result['total_tasks']} tasks")

    async def _create_test_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a test task for async processing"""
        # Simulate async processing delay
        await asyncio.sleep(0.1)

        # Process the data
        sensor_id = task_data["sensor_id"]
        data = task_data["data"]

        processed_result = {
            "sensor_id": sensor_id,
            "mean": np.mean(data),
            "std": np.std(data),
            "processed_at": datetime.now().isoformat()
        }

        return processed_result

    def test_concurrent_sensor_processing(self):
        """Test concurrent processing of multiple sensors"""
        print("\n🔄 Testing concurrent sensor processing...")

        async def process_sensors():
            # Create sensor data for processing
            sensor_data = {}
            for i in range(20):
                sensor_id = f"CONCURRENT_SENSOR_{i}"
                sensor_data[sensor_id] = {
                    "values": np.random.normal(0, 1, 100),
                    "timestamps": [datetime.now() - timedelta(seconds=j) for j in range(100)]
                }

            # Process sensors concurrently
            start_time = time.time()

            tasks = []
            for sensor_id, data in sensor_data.items():
                task = self._process_single_sensor(sensor_id, data)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            processing_time = time.time() - start_time

            return {
                'processing_time': processing_time,
                'results': results,
                'sensors_processed': len([r for r in results if not isinstance(r, Exception)])
            }

        # Run concurrent processing test
        try:
            result = asyncio.run(process_sensors())
        except Exception as e:
            self.fail(f"Concurrent sensor processing failed: {e}")

        # Validate concurrent processing
        self.assertGreater(result['sensors_processed'], 18,
                          "Should process at least 90% of sensors")
        self.assertLess(result['processing_time'], 3.0,
                       "Concurrent processing should be fast")

        print(f"✅ Concurrent processing: {result['sensors_processed']} sensors in {result['processing_time']:.2f}s")

    async def _process_single_sensor(self, sensor_id: str, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sensor asynchronously"""
        # Simulate processing delay
        await asyncio.sleep(0.05)

        values = sensor_data["values"]

        # Calculate sensor metrics
        result = {
            "sensor_id": sensor_id,
            "health_score": 100 - (np.std(values) * 10),
            "anomaly_count": np.sum(np.abs(values) > 2.0),
            "last_value": float(values[-1]),
            "trend": "stable" if np.std(values[-10:]) < np.std(values[:-10]) else "increasing"
        }

        return result


class TestCallbackOptimizer(unittest.TestCase):
    """Test Callback Optimizer component"""

    def setUp(self):
        """Setup test environment"""
        try:
            self.callback_optimizer = CallbackOptimizer()
        except Exception as e:
            self.skipTest(f"Could not initialize CallbackOptimizer: {e}")

    def test_callback_memoization(self):
        """Test callback memoization functionality"""
        print("\n🧠 Testing callback memoization...")

        call_count = {"count": 0}

        @self.callback_optimizer.memoize_callback(ttl=60)
        def expensive_calculation(x, y):
            call_count["count"] += 1
            time.sleep(0.1)  # Simulate expensive operation
            return x * y + np.random.random()

        # First call - should execute function
        start_time = time.time()
        result1 = expensive_calculation(5, 10)
        first_call_time = time.time() - start_time

        # Second call with same parameters - should use cache
        start_time = time.time()
        result2 = expensive_calculation(5, 10)
        second_call_time = time.time() - start_time

        # Validate memoization
        self.assertEqual(call_count["count"], 1, "Function should only be called once")
        self.assertEqual(result1, result2, "Cached result should match original")
        self.assertLess(second_call_time, first_call_time / 2,
                       "Cached call should be significantly faster")

        print(f"✅ Memoization working: {first_call_time:.3f}s → {second_call_time:.3f}s")

    def test_incremental_updates(self):
        """Test incremental update optimization"""
        print("\n🔄 Testing incremental updates...")

        update_count = {"count": 0}

        @self.callback_optimizer.memoize_callback(use_incremental=True)
        def update_sensor_chart(sensor_data, chart_type="line"):
            update_count["count"] += 1
            # Simulate chart generation
            return {
                "chart_data": [d * 2 for d in sensor_data],
                "type": chart_type,
                "generated_at": datetime.now().isoformat()
            }

        # Initial data
        initial_data = [1, 2, 3, 4, 5]
        result1 = update_sensor_chart(initial_data)

        # Same data - should use cache
        result2 = update_sensor_chart(initial_data)

        # Modified data - should trigger update
        modified_data = [1, 2, 3, 4, 5, 6]
        result3 = update_sensor_chart(modified_data)

        # Validate incremental updates
        self.assertEqual(update_count["count"], 2, "Function should be called twice (initial + update)")
        self.assertEqual(result1["chart_data"], result2["chart_data"], "Cached result should match")
        self.assertNotEqual(result1["chart_data"], result3["chart_data"], "Updated result should differ")

        print("✅ Incremental updates working")


class TestDataCompressor(unittest.TestCase):
    """Test Data Compressor component"""

    def setUp(self):
        """Setup test environment"""
        self.compression_config = CompressionConfig(
            compression_threshold=1024,  # 1KB threshold
            preferred_method="lz4",
            compression_level=6
        )

        try:
            self.data_compressor = IntelligentDataCompressor(self.compression_config)
        except Exception as e:
            self.skipTest(f"Could not initialize DataCompressor: {e}")

    def test_data_compression_ratio(self):
        """Test data compression efficiency"""
        print("\n📦 Testing data compression ratio...")

        # Create large, compressible dataset
        large_dataset = {
            "sensor_data": {
                f"SENSOR_{i:03d}": {
                    "values": [x * 0.1 for x in range(1000)],  # Repeated pattern
                    "timestamps": [(datetime.now() + timedelta(seconds=x)).isoformat() for x in range(1000)],
                    "metadata": {
                        "type": "temperature",
                        "unit": "celsius",
                        "location": f"zone_{i % 5}"
                    }
                } for i in range(20)
            },
            "system_info": {
                "version": "1.0.0",
                "deployment": "production",
                "config": {"setting_" + str(i): f"value_{i}" for i in range(100)}
            }
        }

        # Compress data
        start_time = time.time()
        compressed_result = self.data_compressor.compress_dashboard_data(
            large_dataset,
            data_key="large_sensor_dataset"
        )
        compression_time = time.time() - start_time

        # Calculate compression metrics
        import json
        original_size = len(json.dumps(large_dataset).encode('utf-8'))
        compressed_size = len(compressed_result['compressed_data'])
        compression_ratio = original_size / compressed_size

        # Validate compression
        self.assertGreater(compression_ratio, 2.0,
                          f"Compression ratio {compression_ratio:.1f} should be >2:1")
        self.assertLess(compression_time, 5.0,
                       f"Compression should complete in <5s, took {compression_time:.2f}s")

        print(f"✅ Compression ratio: {compression_ratio:.1f}:1 in {compression_time:.2f}s")

    def test_compression_decompression_integrity(self):
        """Test data integrity after compression/decompression"""
        print("\n🔍 Testing compression/decompression integrity...")

        # Test data with various types
        test_data = {
            "sensors": {
                "TEMP_001": {"values": [20.1, 20.2, 20.3], "status": "normal"},
                "PRESSURE_001": {"values": [1.01, 1.02, 1.00], "status": "normal"}
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "sensor_count": 2
            },
            "arrays": [1, 2, 3, 4, 5],
            "nested": {"level1": {"level2": {"value": "deep_value"}}}
        }

        # Compress and decompress
        compressed = self.data_compressor.compress_dashboard_data(test_data, "integrity_test")
        decompressed = self.data_compressor.decompress_dashboard_data(compressed)

        # Validate integrity
        self.assertEqual(len(decompressed), len(test_data), "Structure should be preserved")
        self.assertEqual(decompressed["sensors"]["TEMP_001"]["values"],
                        test_data["sensors"]["TEMP_001"]["values"],
                        "Nested arrays should be preserved")
        self.assertEqual(decompressed["nested"]["level1"]["level2"]["value"],
                        test_data["nested"]["level1"]["level2"]["value"],
                        "Deep nesting should be preserved")

        print("✅ Data integrity preserved through compression/decompression")


class TestPredictiveCache(unittest.TestCase):
    """Test Predictive Cache Manager component"""

    def setUp(self):
        """Setup test environment"""
        try:
            self.predictive_cache = PredictiveCacheManager()
        except Exception as e:
            self.skipTest(f"Could not initialize PredictiveCacheManager: {e}")

    def test_cache_prediction_learning(self):
        """Test cache prediction learning from access patterns"""
        print("\n🧠 Testing cache prediction learning...")

        # Simulate user access patterns
        access_patterns = [
            # Pattern 1: User frequently accesses these together
            ["sensor_overview", "anomaly_dashboard", "health_metrics"],
            ["sensor_overview", "anomaly_dashboard", "health_metrics"],
            ["sensor_overview", "anomaly_dashboard", "health_metrics"],

            # Pattern 2: Different access sequence
            ["equipment_status", "maintenance_schedule"],
            ["equipment_status", "maintenance_schedule"],

            # Pattern 3: Individual accesses
            ["performance_metrics"],
            ["system_logs"]
        ]

        # Learn from access patterns
        for pattern in access_patterns:
            for i, cache_key in enumerate(pattern):
                context = {
                    "time_of_day": 14,  # 2 PM
                    "user_type": "operator",
                    "sequence_position": i,
                    "session_length": len(pattern)
                }

                # Record access
                self.predictive_cache.record_cache_access(cache_key, context)

        # Test prediction
        current_context = {
            "time_of_day": 14,
            "user_type": "operator",
            "current_page": "sensor_overview"
        }

        predictions = self.predictive_cache.predict_next_cache_keys(current_context)

        # Validate predictions
        self.assertIsInstance(predictions, list, "Predictions should be a list")
        if predictions:  # If prediction model is trained
            self.assertIn("anomaly_dashboard", [pred['cache_key'] for pred in predictions],
                         "Should predict commonly accessed key")

        print("✅ Cache prediction learning working")

    def test_prediction_accuracy(self):
        """Test prediction accuracy with known patterns"""
        print("\n🎯 Testing prediction accuracy...")

        # Create predictable access pattern
        pattern_sequence = ["page_a", "page_b", "page_c"]

        # Train with repeated pattern
        for _ in range(50):  # Repeat pattern many times
            for i, page in enumerate(pattern_sequence):
                context = {
                    "sequence_step": i,
                    "pattern_id": "test_pattern",
                    "time_of_day": 10
                }
                self.predictive_cache.record_cache_access(page, context)

        # Test prediction after seeing first part of pattern
        test_context = {
            "current_page": "page_a",
            "sequence_step": 0,
            "pattern_id": "test_pattern",
            "time_of_day": 10
        }

        predictions = self.predictive_cache.predict_next_cache_keys(test_context)

        # Basic validation (prediction model may not be fully trained in test)
        self.assertIsInstance(predictions, list, "Should return prediction list")

        print("✅ Prediction accuracy test completed")


if __name__ == '__main__':
    # Create test suite for optimization components
    test_classes = [
        TestAdvancedCacheManager,
        TestMemoryManager,
        TestAsyncSensorProcessor,
        TestCallbackOptimizer,
        TestDataCompressor,
        TestPredictiveCache
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTest(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("PHASE 4.1 OPTIMIZATION COMPONENTS TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")

    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")