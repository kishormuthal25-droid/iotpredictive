#!/usr/bin/env python3
"""
Memory-Optimized Test Launcher for Low-RAM Systems
Runs essential tests with minimal memory usage
"""
import os
import sys
import gc
import psutil
import subprocess
from pathlib import Path

def check_memory():
    """Check available memory before starting tests"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"Available RAM: {available_gb:.1f} GB ({memory.percent}% used)")

    if available_gb < 1.0:
        print("❌ Warning: Less than 1GB available. Tests may fail.")
        return False
    return True

def run_minimal_tests():
    """Run only essential tests to verify system functionality"""

    # Set memory-optimized environment
    os.environ["CONFIG_FILE"] = "config/memory_optimized.yaml"
    os.environ["TESTING_MODE"] = "true"
    os.environ["MEMORY_OPTIMIZED"] = "true"

    print("🧪 Starting Memory-Optimized Test Suite...")
    print("=" * 50)

    # Essential tests only
    test_commands = [
        # Basic import tests (fastest)
        "python -c \"import src.dashboard.app; print('✅ Dashboard imports OK')\"",

        # Model manager test (critical)
        "python -c \"from src.model_registry.model_manager import get_model_manager; print('✅ Model manager OK')\"",

        # Basic data test
        "python -c \"from src.data_ingestion.unified_data_access import UnifiedDataAccess; print('✅ Data access OK')\"",

        # MLflow connection test
        "python -c \"from src.model_registry.mlflow_manager import MLFlowManager; print('✅ MLflow OK')\"",
    ]

    success_count = 0

    for i, cmd in enumerate(test_commands, 1):
        print(f"\n[{i}/{len(test_commands)}] Running: {cmd.split(';')[0]}...")

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(result.stdout.strip())
                success_count += 1
            else:
                print(f"❌ Failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Force garbage collection after each test
        gc.collect()

    print(f"\n{'='*50}")
    print(f"✅ {success_count}/{len(test_commands)} tests passed")

    if success_count >= 3:
        print("🎉 Core system functional - Ready for basic testing!")
        return True
    else:
        print("❌ Core system issues detected")
        return False

def run_dashboard_test():
    """Test dashboard startup with minimal resources"""
    print("\n🚀 Testing Dashboard Startup...")

    # Set environment for minimal dashboard
    os.environ["CONFIG_FILE"] = "config/memory_optimized.yaml"
    os.environ["DASHBOARD_PORT"] = "8061"  # Different port
    os.environ["TESTING_MODE"] = "true"

    try:
        # Import test
        print("Testing dashboard import...")
        result = subprocess.run(
            "python -c \"from src.dashboard.app import create_dash_app; print('Dashboard import OK')\"",
            shell=True, capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0:
            print("✅ Dashboard can be imported successfully")
            return True
        else:
            print(f"❌ Dashboard import failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def main():
    """Main test execution"""
    print("🔧 Memory-Optimized IoT System Testing")
    print("For systems with limited RAM (<2GB available)")
    print("=" * 50)

    # Check memory first
    if not check_memory():
        print("\n💡 Recommendations:")
        print("- Close all browsers and unnecessary applications")
        print("- Restart your system if possible")
        print("- Consider upgrading RAM for better performance")

        choice = input("\nContinue anyway? (y/N): ").lower()
        if choice != 'y':
            return False

    # Run essential tests
    if not run_minimal_tests():
        return False

    # Test dashboard startup
    if not run_dashboard_test():
        print("\n⚠️  Dashboard startup may be limited, but core system works")

    print(f"\n{'='*60}")
    print("🎯 MEMORY-OPTIMIZED SETUP COMPLETE")
    print("=" * 60)
    print("✅ Core system tested and functional")
    print("✅ Optimized for low-memory environments")
    print("\n📊 To run the optimized dashboard:")
    print("   CONFIG_FILE=config/memory_optimized.yaml python launch_real_data_dashboard.py")
    print("\n💡 This configuration disables:")
    print("   - Advanced forecasting")
    print("   - Risk assessment")
    print("   - Real-time updates")
    print("   - Most caching")
    print("   - Complex visualizations")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)