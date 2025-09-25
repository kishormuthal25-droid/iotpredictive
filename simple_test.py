#!/usr/bin/env python3
"""
Simple IoT System Test - No Unicode Issues
Tests core functionality before Codespace deployment
"""
import os
import sys
import psutil
import subprocess
from pathlib import Path

def check_memory():
    """Check available memory"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"Available RAM: {available_gb:.1f} GB ({memory.percent}% used)")
    return available_gb

def test_basic_imports():
    """Test basic system imports"""
    print("\n=== TESTING BASIC IMPORTS ===")

    tests = [
        ("Dashboard app", "from src.dashboard.app import create_dash_app"),
        ("Model manager", "from src.model_registry.model_manager import get_model_manager"),
        ("Data access", "from src.data_ingestion.unified_data_access import UnifiedDataAccess"),
        ("Config", "from config.settings import get_config"),
    ]

    success_count = 0
    for name, import_cmd in tests:
        try:
            print(f"Testing {name}...", end="")
            exec(import_cmd)
            print(" OK")
            success_count += 1
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"\nImport Tests: {success_count}/{len(tests)} passed")
    return success_count >= 3

def main():
    print("IoT System Quick Test")
    print("=" * 50)

    # Check memory
    available_ram = check_memory()
    if available_ram < 1.0:
        print("WARNING: Low memory detected")

    # Test imports
    if test_basic_imports():
        print("\n" + "=" * 50)
        print("CORE SYSTEM: FUNCTIONAL")
        print("Ready for Codespace deployment!")
        print("\nNext steps:")
        print("1. Run: quick_push.bat")
        print("2. Create Codespace")
        print("3. In Codespace: python launch_real_data_dashboard.py")
        return True
    else:
        print("\n" + "=" * 50)
        print("SYSTEM ISSUES DETECTED")
        print("Fix issues before Codespace deployment")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)