#!/usr/bin/env python3
"""
Simple MLFlow Dashboard Test
Tests core MLFlow functionality without unicode issues
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_mlflow_basic():
    """Test basic MLFlow functionality"""
    print("Testing MLFlow Basic Functionality")
    print("=" * 50)

    try:
        # Test MLFlow model manager
        print("1. Testing MLFlow Model Manager...")
        from src.model_registry.model_manager import get_model_manager

        model_manager = get_model_manager()
        available_models = model_manager.get_available_models()

        print(f"   [OK] Model manager initialized")
        print(f"   [OK] Available models: {len(available_models)}")

        return True

    except Exception as e:
        print(f"   [ERROR] MLFlow error: {e}")
        return False

def test_dashboard_manager():
    """Test dashboard model manager"""
    print("\n2. Testing Dashboard Model Manager...")

    try:
        from src.dashboard.model_manager import pretrained_model_manager

        stats = pretrained_model_manager.get_inference_stats()
        print(f"   [OK] Dashboard model manager ready")
        print(f"   [OK] Available models: {stats.get('available_models_count', 0)}")
        print(f"   [OK] Currently loaded: {stats.get('loaded_models_count', 0)}")

        return True

    except Exception as e:
        print(f"   [ERROR] Dashboard manager error: {e}")
        return False

def test_layouts():
    """Test dashboard layouts"""
    print("\n3. Testing Dashboard Layouts...")

    try:
        # Test layout imports
        from src.dashboard.layouts import overview, anomaly_monitor, forecast_view
        from src.dashboard.layouts import maintenance_scheduler, work_orders, iot_system_structure

        print("   [OK] All 6 dashboard layouts imported successfully")

        # Test layout creation
        overview_layout = overview.create_layout()
        print("   [OK] Overview layout created")

        return True

    except Exception as e:
        print(f"   [ERROR] Layouts error: {e}")
        return False

def test_unified_controller():
    """Test unified data controller"""
    print("\n4. Testing Unified Data Controller...")

    try:
        from src.data_ingestion.unified_data_controller import get_unified_controller

        controller = get_unified_controller()
        overview = controller.get_system_overview()

        print(f"   [OK] System overview retrieved")
        print(f"   [OK] Equipment count: {overview.get('total_equipment', 0)}")

        return True

    except Exception as e:
        print(f"   [ERROR] Controller error: {e}")
        return False

def run_test():
    """Run the complete test suite"""
    print("MLFlow Dashboard Test Suite")
    print("=" * 50)

    start_time = time.time()

    tests = [
        ("MLFlow Basic", test_mlflow_basic),
        ("Dashboard Manager", test_dashboard_manager),
        ("Dashboard Layouts", test_layouts),
        ("Unified Controller", test_unified_controller)
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    end_time = time.time()
    duration = end_time - start_time

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    passed = 0
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    print(f"Duration: {duration:.2f} seconds")

    if passed == len(tests):
        print("\nSUCCESS: All tests passed!")
        print("MLFlow dashboard integration is working correctly.")
        return True
    else:
        print(f"\nFAILED: {len(tests) - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_test()
    exit_code = 0 if success else 1
    sys.exit(exit_code)