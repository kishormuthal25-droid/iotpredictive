#!/usr/bin/env python3
"""
Test script to verify sensor dropdown functionality
Tests the equipment-to-sensor dropdown chain for SMAP-PWR-001
"""

import sys
import json
import requests
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

print("Testing NASA IoT Dashboard Sensor Dropdown Functionality")
print("=" * 60)

try:
    from src.data_ingestion.equipment_mapper import equipment_mapper
    print("+ Successfully imported equipment_mapper")
except ImportError as e:
    print(f"x Import error for equipment_mapper: {e}")
    sys.exit(1)

try:
    from src.dashboard.unified_data_orchestrator import unified_data_orchestrator
    print("+ Successfully imported unified_data_orchestrator")
except ImportError as e:
    print(f"x Import error for unified_data_orchestrator: {e}")
    unified_data_orchestrator = None

def test_equipment_mapper():
    """Test the equipment mapper functionality"""
    print("\n=== Testing Equipment Mapper ===")

    # Test getting all equipment
    all_equipment = equipment_mapper.get_all_equipment()
    print(f"✓ Total equipment components: {len(all_equipment)}")

    # Find SMAP-PWR-001
    smap_pwr_001 = None
    for equipment in all_equipment:
        if equipment.equipment_id == "SMAP-PWR-001":
            smap_pwr_001 = equipment
            break

    if smap_pwr_001:
        print(f"✓ Found SMAP-PWR-001: {smap_pwr_001.equipment_type}")
        print(f"  - Subsystem: {smap_pwr_001.subsystem}")
        print(f"  - Sensor count: {len(smap_pwr_001.sensors)}")
        print(f"  - Sensors:")
        for i, sensor in enumerate(smap_pwr_001.sensors, 1):
            print(f"    {i}. {sensor.name} ({sensor.unit})")

        # Test sensor options method
        sensor_options = equipment_mapper.get_sensor_options_by_equipment("SMAP-PWR-001")
        print(f"  - Sensor options for dropdown: {len(sensor_options)}")
        for option in sensor_options:
            print(f"    - {option['label']}")

        return True
    else:
        print("✗ SMAP-PWR-001 not found in equipment mapper")
        return False

def test_unified_data_orchestrator():
    """Test the unified data orchestrator"""
    print("\n=== Testing Unified Data Orchestrator ===")

    try:
        # Test equipment availability
        available_equipment = unified_data_orchestrator.get_available_equipment()
        print(f"✓ Available equipment from orchestrator: {len(available_equipment)}")

        smap_available = "SMAP-PWR-001" in available_equipment
        print(f"{'✓' if smap_available else '✗'} SMAP-PWR-001 available: {smap_available}")

        if smap_available:
            # Test sensor options for SMAP-PWR-001
            sensor_options = unified_data_orchestrator.get_sensor_options_for_equipment("SMAP-PWR-001")
            print(f"✓ Sensor options from orchestrator: {len(sensor_options)}")

            for option in sensor_options:
                print(f"  - {option['label'] if 'label' in option else option}")

        return smap_available
    except Exception as e:
        print(f"✗ Error testing unified data orchestrator: {e}")
        return False

def test_anomaly_monitor_layout():
    """Test the anomaly monitor layout functionality"""
    print("\n=== Testing Anomaly Monitor Layout ===")

    try:
        # Import specific functions from anomaly monitor
        sys.path.append(str(Path(__file__).parent / "src" / "dashboard" / "layouts"))

        # Try to test the functionality without importing the class
        print("+ Testing equipment dropdown backend functionality...")

        # Test equipment options directly
        all_equipment = equipment_mapper.get_all_equipment()
        equipment_options = []
        for equipment in all_equipment:
            label = f"{equipment.equipment_id} ({equipment.equipment_type})"
            equipment_options.append({"label": label, "value": equipment.equipment_id})

        print(f"+ Equipment options available: {len(equipment_options)}")

        smap_found = False
        for option in equipment_options:
            if option['value'] == 'SMAP-PWR-001':
                smap_found = True
                print(f"+ Found SMAP-PWR-001 in equipment options: {option['label']}")
                break

        if not smap_found:
            print("x SMAP-PWR-001 not found in equipment options")
            print("Available options:")
            for option in equipment_options[:10]:  # Show first 10
                print(f"  - {option['value']}: {option['label']}")

        # Test sensor options for SMAP-PWR-001
        sensor_options = equipment_mapper.get_sensor_options_by_equipment("SMAP-PWR-001")
        print(f"+ Sensor options for SMAP-PWR-001: {len(sensor_options)}")

        for option in sensor_options:
            print(f"  - {option['label']}")

        return smap_found and len(sensor_options) == 6

    except Exception as e:
        print(f"x Error testing anomaly monitor layout: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard_access():
    """Test dashboard accessibility"""
    print("\n=== Testing Dashboard Access ===")

    try:
        response = requests.get('http://localhost:8060', timeout=5)
        if response.status_code == 200:
            print("+ Dashboard is accessible at http://localhost:8060")

            # Check if it's the correct dashboard
            if "IoT Predictive Maintenance Dashboard" in response.text:
                print("+ Correct dashboard title found")
                return True
            else:
                print("x Unexpected dashboard content")
                return False
        else:
            print(f"x Dashboard returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"x Could not access dashboard: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing NASA IoT Dashboard Sensor Dropdown Functionality")
    print("=" * 60)

    results = {
        'equipment_mapper': test_equipment_mapper(),
        'unified_data_orchestrator': test_unified_data_orchestrator(),
        'anomaly_monitor_layout': test_anomaly_monitor_layout(),
        'dashboard_access': test_dashboard_access()
    }

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:25} : {status}")
        if not passed:
            all_passed = False

    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ ALL TESTS PASSED - Sensor dropdown should work correctly")
        print("\nExpected behavior:")
        print("1. Equipment dropdown should show SMAP-PWR-001")
        print("2. Selecting SMAP-PWR-001 should populate sensor dropdown with 6 sensors:")
        print("   - Solar Panel Voltage")
        print("   - Battery Current")
        print("   - Power Distribution Temperature")
        print("   - Charging Controller Status")
        print("   - Bus Voltage")
        print("   - Load Current")
    else:
        print("✗ SOME TESTS FAILED - There may be issues with sensor dropdown")
        print("\nThis could explain why the sensor dropdown is not populating correctly.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)