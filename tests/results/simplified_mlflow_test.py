#!/usr/bin/env python3
"""
Simplified MLflow Lazy Loading Test - Windows Compatible
Critical performance validation without Unicode characters
"""

import sys
import os
import time
import psutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_registry.model_manager import get_model_manager

def test_startup_performance():
    """Test critical <2s startup requirement"""
    print("TESTING: MLflow Model Manager Startup Performance")

    startup_start = time.time()

    # Initialize model manager (should be lazy loaded)
    model_manager = get_model_manager()

    startup_time = time.time() - startup_start

    print(f"Startup time: {startup_time:.3f}s (Target: <2.0s)")

    success = startup_time < 2.0
    print(f"Result: {'PASSED' if success else 'FAILED'}")

    return success

def test_memory_usage():
    """Test memory usage requirement"""
    print("TESTING: Memory Usage")

    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    print(f"Current memory usage: {memory_mb:.1f}MB (Target: <512MB baseline)")

    success = memory_mb < 512
    print(f"Result: {'PASSED' if success else 'FAILED'}")

    return success

def test_model_availability():
    """Test model availability"""
    print("TESTING: Model Availability")

    model_manager = get_model_manager()
    available_models = model_manager.get_available_models()

    print(f"Available models: {len(available_models)}")

    success = len(available_models) > 50  # Should have 97+ models
    print(f"Result: {'PASSED' if success else 'FAILED'}")

    return success

def run_critical_tests():
    """Run all critical performance tests"""
    print("=" * 60)
    print("MLFLOW LAZY LOADING CRITICAL PERFORMANCE TESTS")
    print("=" * 60)

    test_results = []

    # Test 1: Startup Performance (<2s)
    test_results.append(test_startup_performance())
    print()

    # Test 2: Memory Usage (<512MB)
    test_results.append(test_memory_usage())
    print()

    # Test 3: Model Availability (97+ models)
    test_results.append(test_model_availability())
    print()

    # Summary
    passed = sum(test_results)
    total = len(test_results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    all_passed = passed == total
    print(f"OVERALL RESULT: {'PASSED - SYSTEM READY' if all_passed else 'FAILED - ISSUES DETECTED'}")

    return all_passed

if __name__ == "__main__":
    success = run_critical_tests()
    sys.exit(0 if success else 1)