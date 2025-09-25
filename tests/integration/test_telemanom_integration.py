#!/usr/bin/env python3
"""
Test NASA Telemanom Integration with Dashboard
Verify that our integration layer works correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.anomaly_detection.telemanom_integration import telemanom_integration

def test_telemanom_integration():
    """Test the Telemanom integration layer"""
    print("Testing NASA Telemanom Integration Layer")
    print("="*50)

    # Check model loading
    status = telemanom_integration.get_model_status()
    print(f"Total models loaded: {status['total_models']}")
    print(f"Available sensors: {status['loaded_models']}")

    if status['total_models'] == 0:
        print("ERROR: No models loaded!")
        return False

    # Test sensor data detection
    print(f"\nTesting anomaly detection...")

    # Create test sensor data for our trained sensors
    test_sensor_data = {}
    for sensor_id in telemanom_integration.get_available_sensors()[:2]:  # Test first 2 sensors
        # Create realistic test data
        test_data = np.random.randn(100, 1) * 5 + 30  # Voltage-like data
        test_sensor_data[sensor_id] = test_data

    # Run detection
    results = telemanom_integration.detect_anomalies(test_sensor_data)

    print(f"Detection results: {len(results)} sensors processed")
    for result in results:
        print(f"  {result.sensor_id}: score={result.anomaly_score:.3f}, anomaly={result.is_anomaly}")

    return len(results) > 0

def test_dashboard_integration_compatibility():
    """Test compatibility with dashboard integration format"""
    print("\nTesting Dashboard Integration Compatibility")
    print("="*50)

    # Test data service integration
    try:
        from src.data_ingestion.nasa_data_service import nasa_data_service

        # Test creating service (should load our integration)
        print("NASA Data Service initialization: SUCCESS")

        # Check if our Telemanom integration is loaded
        if hasattr(nasa_data_service, 'telemanom_integration'):
            print("Telemanom integration loaded in data service: SUCCESS")
            integration_status = nasa_data_service.telemanom_integration.get_model_status()
            print(f"  Models available in service: {integration_status['total_models']}")
        else:
            print("Telemanom integration NOT found in data service: FAILED")
            return False

        return True

    except Exception as e:
        print(f"Dashboard integration test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing NASA Telemanom Dashboard Integration\n")

    # Test integration layer
    test1_passed = test_telemanom_integration()

    # Test dashboard compatibility
    test2_passed = test_dashboard_integration_compatibility()

    print("\n" + "="*50)
    print("INTEGRATION TEST RESULTS")
    print("="*50)

    if test1_passed and test2_passed:
        print("SUCCESS: NASA Telemanom integration is ready for dashboard!")
        print("Real models will be used for anomaly detection")
    else:
        print("FAILED: Integration needs debugging")

    print("\nNext: Launch dashboard and verify real model usage")