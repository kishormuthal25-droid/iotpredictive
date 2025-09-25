#!/usr/bin/env python3
"""
MLflow Model Registry Integration Test Suite
Tests model registration, versioning, and lifecycle management
"""

import unittest
import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.model_registry.model_manager import get_model_manager, ModelManager
    from src.model_registry.model_registry import ModelRegistry
    MLFLOW_AVAILABLE = True
except ImportError as e:
    MLFLOW_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestModelRegistryIntegration(unittest.TestCase):
    """Test MLflow model registry integration with our system"""

    def setUp(self):
        """Setup test environment"""
        if not MLFLOW_AVAILABLE:
            self.skipTest(f"MLflow not available: {IMPORT_ERROR}")

        self.model_manager = None
        self.temp_dir = None

    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_manager_initialization(self):
        """Test that model manager initializes correctly with MLflow"""
        print("\n🔧 Testing Model Manager Initialization...")

        start_time = time.time()
        self.model_manager = get_model_manager()
        init_time = time.time() - start_time

        self.assertIsNotNone(self.model_manager)
        self.assertIsInstance(self.model_manager, ModelManager)

        # Initialization should be fast (lazy loading)
        self.assertLess(init_time, 1.0, f"Initialization too slow: {init_time:.3f}s")

        print(f"   ✅ Model manager initialized in {init_time:.3f}s")

    def test_available_models_discovery(self):
        """Test discovery of available models from MLflow"""
        print("\n🔍 Testing Available Models Discovery...")

        self.model_manager = get_model_manager()

        discovery_start = time.time()
        available_models = self.model_manager.get_available_models()
        discovery_time = time.time() - discovery_start

        self.assertIsInstance(available_models, dict)

        print(f"   📊 Found {len(available_models)} available models")
        print(f"   ⏱️  Discovery time: {discovery_time:.3f}s")

        # Test model metadata
        for model_id, model_info in list(available_models.items())[:3]:  # Test first 3
            print(f"   📋 Model: {model_id}")
            self.assertIn('name', model_info)
            self.assertIn('version', model_info)

            if 'description' in model_info:
                print(f"      Description: {model_info['description'][:50]}...")
            if 'tags' in model_info:
                print(f"      Tags: {model_info.get('tags', {})}")

    def test_model_loading_and_caching(self):
        """Test model loading performance and caching mechanism"""
        print("\n📦 Testing Model Loading and Caching...")

        self.model_manager = get_model_manager()
        available_models = self.model_manager.get_available_models()

        if not available_models:
            self.skipTest("No models available for testing")

        # Select first available model
        model_id = list(available_models.keys())[0]
        print(f"   🎯 Testing with model: {model_id}")

        # First load (cold start)
        cold_start = time.time()
        model_info_1 = self.model_manager.get_model_info(model_id)
        cold_time = time.time() - cold_start

        self.assertIsNotNone(model_info_1)
        print(f"   ❄️  Cold load time: {cold_time:.3f}s")

        # Second load (should be cached)
        warm_start = time.time()
        model_info_2 = self.model_manager.get_model_info(model_id)
        warm_time = time.time() - warm_start

        self.assertIsNotNone(model_info_2)
        print(f"   🔥 Warm load time: {warm_time:.3f}s")

        # Verify caching works
        self.assertEqual(model_info_1, model_info_2, "Cached model should be identical")

        # Warm load should be significantly faster
        if cold_time > 0.01:  # Only test if cold load was measurable
            speedup = cold_time / warm_time if warm_time > 0 else float('inf')
            print(f"   ⚡ Cache speedup: {speedup:.1f}x")
            self.assertGreater(speedup, 2.0, "Caching should provide at least 2x speedup")

    def test_model_metadata_validation(self):
        """Test that model metadata is properly validated and structured"""
        print("\n✅ Testing Model Metadata Validation...")

        self.model_manager = get_model_manager()
        available_models = self.model_manager.get_available_models()

        if not available_models:
            self.skipTest("No models available for metadata testing")

        # Test metadata structure for multiple models
        tested_models = 0
        for model_id, model_info in available_models.items():
            if tested_models >= 5:  # Limit to first 5 models
                break

            print(f"   🔍 Validating {model_id}...")

            # Required fields
            self.assertIn('name', model_info, f"Model {model_id} missing 'name'")
            self.assertIn('version', model_info, f"Model {model_id} missing 'version'")

            # Validate field types
            self.assertIsInstance(model_info['name'], str)

            # Optional but expected fields
            expected_fields = ['description', 'tags', 'stage', 'creation_timestamp']
            present_fields = [field for field in expected_fields if field in model_info]

            print(f"      ✅ Present fields: {present_fields}")

            # At least some metadata should be present
            self.assertGreater(len(model_info), 2,
                             f"Model {model_id} has insufficient metadata")

            tested_models += 1

        print(f"   📊 Validated {tested_models} models successfully")

    def test_error_handling_robustness(self):
        """Test error handling for various failure scenarios"""
        print("\n🛡️  Testing Error Handling Robustness...")

        self.model_manager = get_model_manager()

        # Test invalid model ID
        print("   🚫 Testing invalid model ID...")
        invalid_model_info = self.model_manager.get_model_info("invalid_model_id_12345")
        self.assertIsNone(invalid_model_info, "Should return None for invalid model")

        # Test empty model ID
        print("   🚫 Testing empty model ID...")
        empty_model_info = self.model_manager.get_model_info("")
        self.assertIsNone(empty_model_info, "Should return None for empty model ID")

        # Test None model ID
        print("   🚫 Testing None model ID...")
        none_model_info = self.model_manager.get_model_info(None)
        self.assertIsNone(none_model_info, "Should return None for None model ID")

        print("   ✅ Error handling working correctly")

    def test_memory_management(self):
        """Test that model registry doesn't leak memory"""
        print("\n🧠 Testing Memory Management...")

        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            self.skipTest("psutil not available for memory testing")

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        self.model_manager = get_model_manager()
        available_models = self.model_manager.get_available_models()

        # Load several models
        loaded_count = 0
        for model_id in list(available_models.keys())[:5]:  # Load first 5
            model_info = self.model_manager.get_model_info(model_id)
            if model_info:
                loaded_count += 1

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"   📊 Initial memory: {initial_memory:.1f}MB")
        print(f"   📊 Final memory: {final_memory:.1f}MB")
        print(f"   📊 Memory increase: {memory_increase:.1f}MB")
        print(f"   📊 Models loaded: {loaded_count}")

        # Memory increase should be reasonable (less than 50MB per model)
        if loaded_count > 0:
            memory_per_model = memory_increase / loaded_count
            print(f"   📊 Memory per model: {memory_per_model:.1f}MB")

            self.assertLess(memory_per_model, 50.0,
                           f"Memory usage per model too high: {memory_per_model:.1f}MB")

    def test_concurrent_access_safety(self):
        """Test that concurrent access to model registry is thread-safe"""
        print("\n🔒 Testing Concurrent Access Safety...")

        import threading
        import queue

        self.model_manager = get_model_manager()
        available_models = self.model_manager.get_available_models()

        if len(available_models) < 2:
            self.skipTest("Need at least 2 models for concurrent testing")

        model_ids = list(available_models.keys())[:3]
        results_queue = queue.Queue()
        errors_queue = queue.Queue()

        def worker(model_id, worker_id):
            """Worker function for concurrent testing"""
            try:
                for i in range(3):  # 3 operations per worker
                    start_time = time.time()
                    model_info = self.model_manager.get_model_info(model_id)
                    operation_time = time.time() - start_time

                    results_queue.put({
                        'worker_id': worker_id,
                        'model_id': model_id,
                        'operation': i,
                        'time': operation_time,
                        'success': model_info is not None
                    })

                    time.sleep(0.1)  # Small delay between operations
            except Exception as e:
                errors_queue.put(f"Worker {worker_id}: {str(e)}")

        # Start concurrent workers
        threads = []
        for i, model_id in enumerate(model_ids):
            thread = threading.Thread(target=worker, args=(model_id, i))
            threads.append(thread)
            thread.start()

        # Wait for all workers
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())

        print(f"   📊 Completed operations: {len(results)}")
        print(f"   📊 Errors: {len(errors)}")

        # No errors should occur
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

        # All operations should succeed
        successful_ops = sum(1 for r in results if r['success'])
        self.assertEqual(successful_ops, len(results), "All operations should succeed")

        # Average operation time should be reasonable
        avg_time = sum(r['time'] for r in results) / len(results)
        print(f"   ⏱️  Average operation time: {avg_time:.3f}s")
        self.assertLess(avg_time, 1.0, "Operations should be fast even under concurrency")


class TestModelRegistryConfiguration(unittest.TestCase):
    """Test model registry configuration and settings"""

    def setUp(self):
        """Setup test environment"""
        if not MLFLOW_AVAILABLE:
            self.skipTest(f"MLflow not available: {IMPORT_ERROR}")

    def test_registry_configuration(self):
        """Test that model registry is properly configured"""
        print("\n⚙️  Testing Model Registry Configuration...")

        model_manager = get_model_manager()

        # Test configuration attributes
        config_attrs = [
            'mlflow_tracking_uri', 'mlflow_registry_uri',
            'model_cache_size', 'lazy_loading_enabled'
        ]

        present_configs = []
        for attr in config_attrs:
            if hasattr(model_manager, attr):
                value = getattr(model_manager, attr)
                present_configs.append(f"{attr}: {value}")

        print(f"   📋 Configuration: {present_configs}")

        # Registry should be properly initialized
        self.assertTrue(hasattr(model_manager, 'get_available_models'))
        self.assertTrue(callable(getattr(model_manager, 'get_available_models')))


def run_comprehensive_mlflow_test():
    """Run comprehensive MLflow model registry test suite"""
    print("=" * 80)
    print("🗃️  MLFLOW MODEL REGISTRY COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    # System info
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")

    if not MLFLOW_AVAILABLE:
        print(f"❌ MLflow not available: {IMPORT_ERROR}")
        return False

    print()

    # Create test suite
    suite = unittest.TestSuite()

    # Core functionality tests
    suite.addTest(TestModelRegistryIntegration('test_model_manager_initialization'))
    suite.addTest(TestModelRegistryIntegration('test_available_models_discovery'))
    suite.addTest(TestModelRegistryIntegration('test_model_loading_and_caching'))
    suite.addTest(TestModelRegistryIntegration('test_model_metadata_validation'))

    # Robustness tests
    suite.addTest(TestModelRegistryIntegration('test_error_handling_robustness'))
    suite.addTest(TestModelRegistryIntegration('test_memory_management'))
    suite.addTest(TestModelRegistryIntegration('test_concurrent_access_safety'))

    # Configuration tests
    suite.addTest(TestModelRegistryConfiguration('test_registry_configuration'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL MLFLOW MODEL REGISTRY TESTS PASSED!")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        for test, error in result.failures + result.errors:
            print(f"   ❌ {test}: {error.split(chr(10))[0]}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_mlflow_test()
    sys.exit(0 if success else 1)