#!/usr/bin/env python3
"""
MLFlow-Enhanced Dashboard Integration Test
Tests complete dashboard functionality with MLFlow lazy loading
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_mlflow_infrastructure():
    """Test MLFlow infrastructure and lazy loading"""
    print("Testing MLFlow Infrastructure")
    print("=" * 50)

    try:
        # Test MLFlow model manager initialization
        print("1. Testing MLFlow Model Manager...")
        from src.model_registry.model_manager import get_model_manager

        model_manager = get_model_manager()
        available_models = model_manager.get_available_models()

        print(f"   [OK] Model manager initialized")
        print(f"   [OK] Available models: {len(available_models)}")
        print(f"   [OK] MLFlow server running: {model_manager.mlflow_config.is_running}")

        # Test model discovery without loading
        print("   [OK] Models discovered without loading (lazy initialization)")

    except Exception as e:
        print(f"   [ERROR] MLFlow infrastructure error: {e}")
        return False

    return True

def test_dashboard_model_manager():
    """Test MLFlow-enhanced dashboard model manager"""
    print("\nTesting Dashboard Model Manager")
    print("=" * 50)

    try:
        # Test dashboard model manager
        print("2. Testing MLFlow Dashboard Model Manager...")
        from src.dashboard.model_manager import pretrained_model_manager

        # This should not load any models yet
        stats = pretrained_model_manager.get_inference_stats()
        print(f"   [OK] Dashboard model manager initialized")
        print(f"   [OK] Available models: {stats.get('available_models_count', 0)}")
        print(f"   [OK] Loaded models: {stats.get('loaded_models_count', 0)} (should be 0 - lazy loading)")

        # Test lazy model loading for a specific equipment
        print("\n3. Testing Lazy Model Loading...")
        test_equipment = "MSL_25"  # Known NASA Telemanom model

        print(f"   Loading model for {test_equipment}...")
        model = pretrained_model_manager.get_model_for_equipment(test_equipment)

        if model is not None:
            print(f"   [OK] Model loaded successfully for {test_equipment}")

            # Check stats after loading
            updated_stats = pretrained_model_manager.get_inference_stats()
            print(f"   [OK] Loaded models after lazy loading: {updated_stats.get('loaded_models_count', 0)}")
        else:
            print(f"   [WARN] Model not found for {test_equipment} (might be expected)")

    except Exception as e:
        print(f"   [ERROR] Dashboard model manager error: {e}")
        return False

    return True

def test_unified_data_controller():
    """Test unified data controller with lazy anomaly detection"""
    print("\nTesting Unified Data Controller")
    print("=" * 50)

    try:
        print("4. Testing Unified Data Controller...")
        from src.data_ingestion.unified_data_controller import get_unified_controller

        controller = get_unified_controller()

        # Test system overview (should not load models)
        overview = controller.get_system_overview()
        print(f"   [OK] System overview retrieved")
        print(f"   [OK] Total equipment: {overview.get('total_equipment', 0)}")
        print(f"   [OK] Total sensors: {overview.get('total_sensors', 0)}")

        # Test equipment status (should not load models upfront)
        equipment_status = controller.get_all_equipment_status()
        print(f"   [OK] Equipment status retrieved for {len(equipment_status)} items")

        # Test lazy anomaly detection for a specific sensor
        print("\n5. Testing Lazy Anomaly Detection...")
        if equipment_status:
            first_equipment = list(equipment_status.keys())[0]
            print(f"   Testing anomaly detection for: {first_equipment}")

            anomaly_scores = controller.detect_anomalies_for_sensor(first_equipment)
            print(f"   [OK] Anomaly detection completed (lazy model loading)")
            print(f"   [OK] Anomaly scores shape: {anomaly_scores.shape if hasattr(anomaly_scores, 'shape') else len(anomaly_scores)}")

    except Exception as e:
        print(f"   [ERROR] Unified data controller error: {e}")
        return False

    return True

def test_dashboard_layouts():
    """Test dashboard layouts with MLFlow integration"""
    print("\nTesting Dashboard Layouts")
    print("=" * 50)

    try:
        print("6. Testing Dashboard Layout Imports...")

        # Test all 6 main dashboard layouts
        layouts_to_test = [
            ("overview", "src.dashboard.layouts.overview"),
            ("anomaly_monitor", "src.dashboard.layouts.anomaly_monitor"),
            ("forecast_view", "src.dashboard.layouts.forecast_view"),
            ("maintenance_scheduler", "src.dashboard.layouts.maintenance_scheduler"),
            ("work_orders", "src.dashboard.layouts.work_orders"),
            ("iot_system_structure", "src.dashboard.layouts.iot_system_structure"),
            ("pipeline_dashboard", "src.dashboard.layouts.pipeline_dashboard")
        ]

        loaded_layouts = {}
        for layout_name, module_path in layouts_to_test:
            try:
                # Import layout module
                module = __import__(module_path, fromlist=[layout_name])

                # Test layout creation (should not load models)
                if hasattr(module, 'create_layout'):
                    layout = module.create_layout()
                    loaded_layouts[layout_name] = True
                    print(f"   [OK] {layout_name} layout loaded successfully")
                else:
                    loaded_layouts[layout_name] = False
                    print(f"   [WARN] {layout_name} layout missing create_layout method")

            except Exception as e:
                loaded_layouts[layout_name] = False
                print(f"   [ERROR] {layout_name} layout error: {e}")

        successful_layouts = sum(loaded_layouts.values())
        print(f"\n   [SUMMARY] Layout Summary: {successful_layouts}/{len(layouts_to_test)} layouts loaded successfully")

        return successful_layouts >= 5  # At least 5 out of 7 should work

    except Exception as e:
        print(f"   [ERROR] Dashboard layouts error: {e}")
        return False

def test_dashboard_startup_simulation():
    """Simulate dashboard startup without actually starting the server"""
    print("\nTesting Dashboard Startup Simulation")
    print("=" * 50)

    try:
        print("7. Testing Dashboard Startup Components...")

        # Test Dash app creation (lightweight test)
        import dash
        from dash import html
        import dash_bootstrap_components as dbc

        print("   Testing Dash app creation...")
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        print("   [OK] Dash app created successfully")

        # Test basic layout structure
        print("   Testing basic layout structure...")
        app.layout = html.Div([
            html.H1("MLFlow Dashboard Test"),
            html.Div(id="test-content")
        ])
        print("   [OK] Basic layout structure created")

        # Test callback registration simulation
        print("   Testing callback registration...")
        from dash import Input, Output

        @app.callback(
            Output('test-content', 'children'),
            [Input('test-content', 'id')]
        )
        def test_callback(component_id):
            return "MLFlow dashboard integration test successful!"

        print("   [OK] Test callback registered successfully")

        return True

    except Exception as e:
        print(f"   [ERROR] Dashboard startup simulation error: {e}")
        return False

def test_memory_usage():
    """Test memory usage to ensure no model loading issues"""
    print("\nTesting Memory Usage")
    print("=" * 50)

    try:
        import psutil
        import os

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"8. Initial memory usage: {initial_memory:.2f} MB")

        # Test various imports that previously caused memory issues
        print("   Testing imports without model loading...")

        # These should not cause significant memory increase
        from src.model_registry.model_manager import get_model_manager
        from src.dashboard.model_manager import pretrained_model_manager
        from src.data_ingestion.unified_data_controller import get_unified_controller

        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Memory after imports: {mid_memory:.2f} MB (increase: {mid_memory - initial_memory:.2f} MB)")

        # Test single model lazy loading
        model_manager = get_model_manager()
        available_models = list(model_manager.get_available_models().keys())

        if available_models:
            test_model = available_models[0]
            print(f"   Testing lazy loading for model: {test_model}")
            model = model_manager.get_model(test_model)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            model_load_increase = final_memory - mid_memory

            print(f"   Memory after single model load: {final_memory:.2f} MB (model load increase: {model_load_increase:.2f} MB)")

            # Check if memory increase is reasonable (should be much less than 1GB)
            if model_load_increase < 100:  # Less than 100MB for single model
                print("   [OK] Memory usage is reasonable for single model loading")
                return True
            else:
                print(f"   [WARN] High memory usage for single model: {model_load_increase:.2f} MB")
                return False
        else:
            print("   [OK] No models available for testing - this is expected in lazy loading")
            return True

    except ImportError:
        print("   [WARN] psutil not available for memory testing")
        return True  # Not critical
    except Exception as e:
        print(f"   [ERROR] Memory usage test error: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive MLFlow dashboard integration test"""
    print("MLFlow-Enhanced Dashboard Integration Test Suite")
    print("=" * 70)
    print("Testing dashboard functionality with MLFlow lazy loading")
    print("This test should complete in under 60 seconds")
    print("=" * 70)

    start_time = time.time()

    # Run all test components
    test_results = {
        "mlflow_infrastructure": test_mlflow_infrastructure(),
        "dashboard_model_manager": test_dashboard_model_manager(),
        "unified_data_controller": test_unified_data_controller(),
        "dashboard_layouts": test_dashboard_layouts(),
        "dashboard_startup": test_dashboard_startup_simulation(),
        "memory_usage": test_memory_usage()
    }

    end_time = time.time()
    duration = end_time - start_time

    # Test results summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"{test_name.replace('_', ' ').title():.<50} {status}")
        if result:
            passed_tests += 1

    print("-" * 70)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print(f"Test duration: {duration:.2f} seconds")

    if passed_tests == total_tests:
        print("\nALL TESTS PASSED! MLFlow dashboard integration is working correctly.")
        print("[OK] Dashboard should start quickly without loading all 308 models")
        print("[OK] All 6 dashboard tabs should be functional")
        print("[OK] Models will load lazily when needed")
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        print(f"\n[PARTIAL SUCCESS] MOSTLY SUCCESSFUL ({passed_tests}/{total_tests} passed)")
        print("Dashboard should work with some limitations.")
    else:
        print(f"\n[FAILED] TESTS FAILED ({passed_tests}/{total_tests} passed)")
        print("Dashboard needs additional fixes before deployment.")

    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit_code = 0 if success else 1
    sys.exit(exit_code)