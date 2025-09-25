#!/usr/bin/env python3
"""
System Validation Script for Phase 3.3 Final Cleanup
Tests core system functionality without running the full dashboard
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test critical import functionality"""
    print("Testing critical imports...")

    try:
        # Test basic utilities
        from src.utils import ml_fallback
        print("[OK] ML fallback utilities")

        # Test model registry (should work with fallback)
        from src.model_registry.model_manager import get_model_manager
        print("[OK] Model manager")

        # Test data ingestion
        from src.data_ingestion.nasa_data_ingestion_service import NASADataIngestionService
        print("[OK] NASA data ingestion")

        # Test anomaly detection
        from src.anomaly_detection.nasa_anomaly_engine import NASAAnomalyEngine
        print("[OK] NASA anomaly engine")

        # Test dashboard components
        from src.dashboard.layouts import overview
        print("[OK] Dashboard layouts")

        return True

    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return False

def test_model_manager():
    """Test model manager functionality"""
    print("\nTesting model manager...")

    try:
        from src.model_registry.model_manager import get_model_manager

        # Get manager instance
        manager = get_model_manager()
        print("[OK] Model manager instance created")

        # Test getting available models
        models = manager.get_available_models()
        print(f"[OK] Available models: {len(models)}")

        # Test cache stats
        if hasattr(manager, 'cache'):
            stats = manager.cache.get_stats()
            print(f"[OK] Cache stats: {stats['size']}/{stats['max_size']} models cached")

        return True

    except Exception as e:
        print(f"[ERROR] Model manager test failed: {e}")
        return False

def test_data_services():
    """Test data services"""
    print("\nTesting data services...")

    try:
        from src.data_ingestion.nasa_data_ingestion_service import NASADataIngestionService

        service = NASADataIngestionService()
        print("[OK] NASA data service created")

        # Test getting datasets
        datasets = service.get_available_datasets()
        print(f"[OK] Available datasets: {len(datasets)}")

        return True

    except Exception as e:
        print(f"[ERROR] Data services test failed: {e}")
        return False

def test_system_health():
    """Test system health monitoring"""
    print("\nTesting system health...")

    try:
        # Test memory monitoring (without starting full monitoring)
        import psutil
        memory = psutil.virtual_memory()
        print(f"[OK] System memory: {memory.used / 1024 / 1024:.1f}MB used ({memory.percent:.1f}%)")

        # Test CPU
        cpu = psutil.cpu_percent(interval=1)
        print(f"[OK] CPU usage: {cpu:.1f}%")

        # Test disk space
        disk = psutil.disk_usage('.')
        print(f"[OK] Disk usage: {disk.used / 1024 / 1024 / 1024:.1f}GB used ({disk.percent:.1f}%)")

        return True

    except Exception as e:
        print(f"[ERROR] System health test failed: {e}")
        return False

def test_performance_targets():
    """Test if system meets performance targets"""
    print("\nTesting performance targets...")

    try:
        import psutil
        from pathlib import Path

        # Test memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        memory_target = 512  # MB
        memory_ok = memory_mb < memory_target

        print(f"[OK] Memory usage: {memory_mb:.1f}MB {'(GOOD)' if memory_ok else '(HIGH)'}")

        # Test startup speed (simulate)
        start_time = time.time()

        # Simulate some initialization work
        from src.model_registry.model_manager import get_model_manager
        manager = get_model_manager()
        manager.get_available_models()

        startup_time = time.time() - start_time
        startup_target = 3.0  # seconds
        startup_ok = startup_time < startup_target

        print(f"[OK] Startup simulation: {startup_time:.1f}s {'(FAST)' if startup_ok else '(SLOW)'}")

        return memory_ok and startup_ok

    except Exception as e:
        print(f"[ERROR] Performance test failed: {e}")
        return False

def main():
    """Run system validation"""
    print("IoT Predictive Maintenance System - Validation")
    print("=" * 55)
    print("Phase 3.3: Final System Validation")
    print()

    tests = [
        ("Critical Imports", test_imports),
        ("Model Manager", test_model_manager),
        ("Data Services", test_data_services),
        ("System Health", test_system_health),
        ("Performance Targets", test_performance_targets),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()

    # Summary
    print("=" * 55)
    print("VALIDATION RESULTS")
    print("=" * 55)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1

    print()
    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        return 0
    elif passed >= total * 0.8:
        print("[WARNING] MOST TESTS PASSED - SYSTEM FUNCTIONAL WITH MINOR ISSUES")
        return 1
    else:
        print("[ERROR] MULTIPLE TEST FAILURES - SYSTEM NEEDS ATTENTION")
        return 2

if __name__ == '__main__':
    sys.exit(main())