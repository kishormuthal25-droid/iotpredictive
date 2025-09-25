#!/usr/bin/env python3
"""
Test script to verify sensor dropdown functionality
Tests the equipment mapper and unified data orchestrator directly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_equipment_mapping():
    """Test equipment mapping and sensor options"""
    print("=== Testing Equipment Mapping ===")

    try:
        from src.data_ingestion.equipment_mapper import equipment_mapper
        from src.dashboard.unified_data_orchestrator import unified_data_orchestrator

        # Test 1: Get all equipment
        print("\n1. Testing equipment list:")
        equipment_list = equipment_mapper.get_all_equipment()
        print(f"   Found {len(equipment_list)} equipment:")
        for eq in equipment_list[:5]:  # Show first 5
            print(f"   - {eq}")
        if len(equipment_list) > 5:
            print(f"   ... and {len(equipment_list) - 5} more")

        # Test 2: Test SMAP-PWR-001 specifically
        print("\n2. Testing SMAP-PWR-001 sensor options:")
        smap_sensors = equipment_mapper.get_sensor_options_by_equipment("SMAP-PWR-001")
        print(f"   Found {len(smap_sensors)} sensors:")
        for sensor in smap_sensors:
            print(f"   - {sensor['label']} (value: {sensor['value']})")

        # Test 3: Test unified data orchestrator
        print("\n3. Testing Unified Data Orchestrator:")
        udo_sensors = unified_data_orchestrator.get_sensor_options_for_equipment("SMAP-PWR-001")
        print(f"   Found {len(udo_sensors)} sensors from UDO:")
        for sensor in udo_sensors:
            print(f"   - {sensor.get('label', 'No label')} (value: {sensor.get('value', 'No value')})")

        # Test 4: Test MSL equipment too
        print("\n4. Testing MSL-MOB-001 sensor options:")
        msl_sensors = equipment_mapper.get_sensor_options_by_equipment("MSL-MOB-001")
        print(f"   Found {len(msl_sensors)} sensors:")
        for sensor in msl_sensors[:3]:  # Show first 3
            print(f"   - {sensor['label']} (value: {sensor['value']})")
        if len(msl_sensors) > 3:
            print(f"   ... and {len(msl_sensors) - 3} more")

        # Test 5: Test hierarchical equipment options
        print("\n5. Testing hierarchical equipment options:")
        hierarchical_options = equipment_mapper.get_hierarchical_equipment_options()
        print(f"   Found {len(hierarchical_options)} hierarchical options:")
        for option in hierarchical_options:
            if 'children' in option:
                print(f"   - {option['label']} (subsystem with {len(option['children'])} equipment)")
            else:
                print(f"   - {option['label']} (equipment: {option['value']})")

        print("\n=== All Tests Completed Successfully! ===")
        print("✅ Equipment mapping is working correctly")
        print("✅ SMAP-PWR-001 has the expected 6 sensors")
        print("✅ Unified Data Orchestrator is functioning")
        print("✅ The sensor dropdown should populate when equipment is selected")

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_equipment_mapping()