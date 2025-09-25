#!/usr/bin/env python3
"""
MLflow Lazy Loading Performance Test Suite
Tests the critical startup performance optimization (<2s startup)
"""

import unittest
import sys
import os
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import memory_profiler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_registry.model_manager import get_model_manager
from src.dashboard.mlflow_model_manager import MLFlowDashboardModelManager


class TestMLflowLazyLoading(unittest.TestCase):
    """Test MLflow lazy loading performance and memory optimization"""

    def setUp(self):
        """Setup test environment"""
        self.start_time = time.time()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def test_startup_performance_under_2_seconds(self):
        """Test that model manager initializes in under 2 seconds (CRITICAL REQUIREMENT)"""
        print("\n🚀 Testing MLflow Model Manager Startup Performance...")

        startup_start = time.time()

        # Initialize model manager (should be lazy loaded)
        model_manager = get_model_manager()

        startup_time = time.time() - startup_start

        # CRITICAL: Must be under 2 seconds
        self.assertLess(startup_time, 2.0,
                       f"Startup time {startup_time:.2f}s exceeds 2s requirement!")

        print(f"   ✅ Startup time: {startup_time:.3f}s (Target: <2.0s)")

        # Verify lazy loading behavior
        self.assertIsNotNone(model_manager)
        available_models = model_manager.get_available_models()

        print(f"   ✅ Available models: {len(available_models)} (lazy loaded)")

    def test_memory_usage_under_512mb_baseline(self):
        """Test that initial memory usage stays under 512MB (PERFORMANCE TARGET)"""
        print("\n🧠 Testing Memory Usage After Model Manager Init...")

        # Initialize model manager
        model_manager = get_model_manager()

        # Wait for initialization to complete
        time.sleep(0.5)

        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory

        print(f"   📊 Initial memory: {self.initial_memory:.1f}MB")
        print(f"   📊 Current memory: {current_memory:.1f}MB")
        print(f"   📊 Memory increase: {memory_increase:.1f}MB")

        # Target: Stay under 512MB total, with reasonable increase
        self.assertLess(current_memory, 512,
                       f"Memory usage {current_memory:.1f}MB exceeds 512MB target!")
        self.assertLess(memory_increase, 100,
                       f"Memory increase {memory_increase:.1f}MB too high!")

    def test_lazy_model_loading_behavior(self):
        """Test that models are actually loaded lazily, not all at once"""
        print("\n⏳ Testing Lazy Loading Behavior...")

        model_manager = get_model_manager()

        # Get available models (should not load them)
        available_models = model_manager.get_available_models()

        memory_before = self.process.memory_info().rss / 1024 / 1024

        # Load first model (should trigger lazy loading)
        if available_models:
            first_model_id = list(available_models.keys())[0]
            print(f"   🔄 Loading first model: {first_model_id}")

            load_start = time.time()
            model_info = model_manager.get_model_info(first_model_id)
            load_time = time.time() - load_start

            memory_after = self.process.memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before

            print(f"   ⏱️  Load time: {load_time:.3f}s")
            print(f"   📊 Memory increase: {memory_increase:.1f}MB")

            # Verify model was loaded
            self.assertIsNotNone(model_info)

            # Load time should be reasonable (under 500ms per model)
            self.assertLess(load_time, 0.5,
                           f"Model load time {load_time:.3f}s too slow!")

    def test_concurrent_model_access(self):
        """Test that multiple models can be accessed concurrently without issues"""
        print("\n🔄 Testing Concurrent Model Access...")

        model_manager = get_model_manager()
        available_models = model_manager.get_available_models()

        if len(available_models) < 2:
            self.skipTest("Need at least 2 models for concurrent testing")

        model_ids = list(available_models.keys())[:3]  # Test with 3 models
        results = {}
        errors = []

        def load_model(model_id):
            try:
                start_time = time.time()
                model_info = model_manager.get_model_info(model_id)
                load_time = time.time() - start_time
                results[model_id] = {
                    'success': True,
                    'load_time': load_time,
                    'model_info': model_info is not None
                }
            except Exception as e:
                errors.append(f"Model {model_id}: {str(e)}")
                results[model_id] = {'success': False, 'error': str(e)}

        # Start concurrent loading
        threads = []
        concurrent_start = time.time()

        for model_id in model_ids:
            thread = threading.Thread(target=load_model, args=(model_id,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        total_concurrent_time = time.time() - concurrent_start

        print(f"   ⏱️  Total concurrent time: {total_concurrent_time:.3f}s")

        # Verify all models loaded successfully
        self.assertEqual(len(errors), 0, f"Concurrent loading errors: {errors}")

        for model_id, result in results.items():
            self.assertTrue(result['success'], f"Model {model_id} failed to load")
            print(f"   ✅ {model_id}: {result['load_time']:.3f}s")

        # Concurrent loading should be efficient
        self.assertLess(total_concurrent_time, 2.0,
                       "Concurrent loading too slow!")

    def test_model_cache_efficiency(self):
        """Test that model caching works efficiently (repeated access)"""
        print("\n💾 Testing Model Cache Efficiency...")

        model_manager = get_model_manager()
        available_models = model_manager.get_available_models()

        if not available_models:
            self.skipTest("No models available for caching test")

        model_id = list(available_models.keys())[0]

        # First access (should load from storage)
        first_start = time.time()
        model_info_1 = model_manager.get_model_info(model_id)
        first_time = time.time() - first_start

        # Second access (should use cache)
        second_start = time.time()
        model_info_2 = model_manager.get_model_info(model_id)
        second_time = time.time() - second_start

        print(f"   🔄 First access: {first_time:.3f}s")
        print(f"   ⚡ Second access: {second_time:.3f}s")

        # Verify caching effectiveness
        self.assertEqual(model_info_1, model_info_2, "Model info should be identical")

        # Cache should be significantly faster (at least 50% improvement)
        cache_improvement = (first_time - second_time) / first_time
        self.assertGreater(cache_improvement, 0.5,
                          f"Cache improvement {cache_improvement:.1%} insufficient!")

        print(f"   ✅ Cache improvement: {cache_improvement:.1%}")


class TestMLflowDashboardIntegration(unittest.TestCase):
    """Test MLflow integration with dashboard components"""

    def test_dashboard_model_manager_init(self):
        """Test dashboard-specific model manager initialization"""
        print("\n🎯 Testing Dashboard Model Manager...")

        try:
            dashboard_model_manager = MLFlowDashboardModelManager()
            self.assertIsNotNone(dashboard_model_manager)

            # Test model loading for dashboard
            models = dashboard_model_manager.load_models()
            self.assertIsInstance(models, dict)

            print(f"   ✅ Dashboard models loaded: {len(models)}")

        except ImportError as e:
            self.skipTest(f"Dashboard model manager not available: {e}")

    def test_model_switching_performance(self):
        """Test performance of switching models in dashboard context"""
        print("\n🔄 Testing Model Switching Performance...")

        try:
            dashboard_model_manager = MLFlowDashboardModelManager()
            available_models = dashboard_model_manager.get_available_models()

            if len(available_models) < 2:
                self.skipTest("Need at least 2 models for switching test")

            model_ids = list(available_models.keys())[:2]

            switch_times = []
            for i, model_id in enumerate(model_ids):
                switch_start = time.time()
                result = dashboard_model_manager.switch_to_model(model_id)
                switch_time = time.time() - switch_start

                switch_times.append(switch_time)
                print(f"   🔄 Switch to {model_id}: {switch_time:.3f}s")

                self.assertTrue(result, f"Failed to switch to {model_id}")

            # Model switching should be fast (under 200ms)
            avg_switch_time = sum(switch_times) / len(switch_times)
            self.assertLess(avg_switch_time, 0.2,
                           f"Average switch time {avg_switch_time:.3f}s too slow!")

            print(f"   ✅ Average switch time: {avg_switch_time:.3f}s")

        except ImportError:
            self.skipTest("Dashboard model manager not available")


def run_performance_benchmark():
    """Run comprehensive MLflow performance benchmark"""
    print("=" * 70)
    print("*** MLFLOW LAZY LOADING PERFORMANCE BENCHMARK ***")
    print("=" * 70)

    # Record system info
    print(f"🖥️  System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // 1024**3}GB RAM")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run tests
    suite = unittest.TestSuite()

    # Add performance-critical tests
    suite.addTest(TestMLflowLazyLoading('test_startup_performance_under_2_seconds'))
    suite.addTest(TestMLflowLazyLoading('test_memory_usage_under_512mb_baseline'))
    suite.addTest(TestMLflowLazyLoading('test_lazy_model_loading_behavior'))
    suite.addTest(TestMLflowLazyLoading('test_concurrent_model_access'))
    suite.addTest(TestMLflowLazyLoading('test_model_cache_efficiency'))

    # Add dashboard integration tests
    suite.addTest(TestMLflowDashboardIntegration('test_dashboard_model_manager_init'))
    suite.addTest(TestMLflowDashboardIntegration('test_model_switching_performance'))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("🎉 ALL MLFLOW PERFORMANCE TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED - PERFORMANCE ISSUES DETECTED")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == '__main__':
    # Run as performance benchmark
    success = run_performance_benchmark()
    sys.exit(0 if success else 1)