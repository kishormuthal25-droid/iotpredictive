"""
IoT Equipment Mapping System
Maps NASA SMAP and MSL raw sensor data to realistic aerospace equipment components
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorSpec:
    """Specification for a single sensor"""
    name: str
    unit: str
    min_value: float
    max_value: float
    nominal_value: float
    critical_threshold: float
    equipment_type: str
    subsystem: str


@dataclass
class EquipmentComponent:
    """Represents a single equipment component with its sensors"""
    equipment_id: str
    equipment_type: str
    subsystem: str
    location: str
    sensors: List[SensorSpec]
    criticality: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'


class IoTEquipmentMapper:
    """Maps NASA raw data to realistic IoT equipment structure"""

    def __init__(self):
        self.smap_equipment = self._define_smap_equipment()
        self.msl_equipment = self._define_msl_equipment()

    def _define_smap_equipment(self) -> List[EquipmentComponent]:
        """Define SMAP satellite equipment components (25 sensors total)"""
        equipment = []

        # Power Systems (6 sensors: 0-5)
        power_sensors = [
            SensorSpec("Solar Panel Voltage", "V", 25.0, 35.0, 30.0, 26.0, "SATELLITE", "POWER"),
            SensorSpec("Battery Current", "A", 0.0, 15.0, 8.0, 12.0, "SATELLITE", "POWER"),
            SensorSpec("Power Distribution Temperature", "°C", -20.0, 60.0, 25.0, 55.0, "SATELLITE", "POWER"),
            SensorSpec("Charging Controller Status", "%", 0.0, 100.0, 95.0, 80.0, "SATELLITE", "POWER"),
            SensorSpec("Bus Voltage", "V", 27.0, 30.0, 28.5, 27.5, "SATELLITE", "POWER"),
            SensorSpec("Load Current", "A", 5.0, 12.0, 8.5, 11.0, "SATELLITE", "POWER")
        ]
        equipment.append(EquipmentComponent("SMAP-PWR-001", "Power System", "POWER", "Satellite Bus", power_sensors, "CRITICAL"))

        # Communication Systems (5 sensors: 6-10)
        comm_sensors = [
            SensorSpec("Antenna Orientation", "degrees", 0.0, 360.0, 180.0, 350.0, "SATELLITE", "COMMUNICATION"),
            SensorSpec("Signal Strength", "dBm", -80.0, -30.0, -50.0, -75.0, "SATELLITE", "COMMUNICATION"),
            SensorSpec("Data Transmission Rate", "Mbps", 0.1, 50.0, 25.0, 5.0, "SATELLITE", "COMMUNICATION"),
            SensorSpec("Receiver Temperature", "°C", -30.0, 70.0, 20.0, 65.0, "SATELLITE", "COMMUNICATION"),
            SensorSpec("Transmitter Power", "W", 10.0, 100.0, 50.0, 20.0, "SATELLITE", "COMMUNICATION")
        ]
        equipment.append(EquipmentComponent("SMAP-COM-001", "Communication System", "COMMUNICATION", "Satellite Bus", comm_sensors, "HIGH"))

        # Attitude Control (6 sensors: 11-16)
        attitude_sensors = [
            SensorSpec("Gyroscope X", "°/s", -10.0, 10.0, 0.0, 8.0, "SATELLITE", "ATTITUDE"),
            SensorSpec("Gyroscope Y", "°/s", -10.0, 10.0, 0.0, 8.0, "SATELLITE", "ATTITUDE"),
            SensorSpec("Gyroscope Z", "°/s", -10.0, 10.0, 0.0, 8.0, "SATELLITE", "ATTITUDE"),
            SensorSpec("Accelerometer X", "m/s²", -2.0, 2.0, 0.0, 1.5, "SATELLITE", "ATTITUDE"),
            SensorSpec("Accelerometer Y", "m/s²", -2.0, 2.0, 0.0, 1.5, "SATELLITE", "ATTITUDE"),
            SensorSpec("Accelerometer Z", "m/s²", -2.0, 2.0, 0.0, 1.5, "SATELLITE", "ATTITUDE")
        ]
        equipment.append(EquipmentComponent("SMAP-ATT-001", "Attitude Control", "ATTITUDE", "Satellite Bus", attitude_sensors, "CRITICAL"))

        # Thermal Control (4 sensors: 17-20)
        thermal_sensors = [
            SensorSpec("Internal Temperature", "°C", -40.0, 60.0, 20.0, 55.0, "SATELLITE", "THERMAL"),
            SensorSpec("Radiator Temperature", "°C", -80.0, 40.0, -20.0, 35.0, "SATELLITE", "THERMAL"),
            SensorSpec("Heat Exchanger Efficiency", "%", 70.0, 100.0, 95.0, 75.0, "SATELLITE", "THERMAL"),
            SensorSpec("Thermal Controller Power", "W", 50.0, 200.0, 100.0, 180.0, "SATELLITE", "THERMAL")
        ]
        equipment.append(EquipmentComponent("SMAP-THM-001", "Thermal Management", "THERMAL", "Satellite Bus", thermal_sensors, "HIGH"))

        # Payload Sensors (4 sensors: 21-24)
        payload_sensors = [
            SensorSpec("Soil Moisture Detector Status", "%", 0.0, 100.0, 98.0, 85.0, "SATELLITE", "PAYLOAD"),
            SensorSpec("Radar Antenna Position", "degrees", 0.0, 360.0, 180.0, 350.0, "SATELLITE", "PAYLOAD"),
            SensorSpec("Data Processing Unit Load", "%", 0.0, 100.0, 45.0, 90.0, "SATELLITE", "PAYLOAD"),
            SensorSpec("Payload Power Consumption", "W", 100.0, 300.0, 200.0, 280.0, "SATELLITE", "PAYLOAD")
        ]
        equipment.append(EquipmentComponent("SMAP-PAY-001", "Soil Moisture Payload", "PAYLOAD", "Satellite Bus", payload_sensors, "HIGH"))

        return equipment

    def _define_msl_equipment(self) -> List[EquipmentComponent]:
        """Define MSL Mars rover equipment components (55 sensors total)"""
        equipment = []

        # Mobility Systems - Front Wheels (12 sensors: 0-11)
        front_mobility_sensors = [
            SensorSpec("FL Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("FL Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("FL Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY"),
            SensorSpec("FR Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("FR Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("FR Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY"),
            SensorSpec("ML Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("ML Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("ML Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY"),
            SensorSpec("MR Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("MR Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("MR Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY")
        ]
        equipment.append(EquipmentComponent("MSL-MOB-001", "Mobility System Front", "MOBILITY", "Mars Surface", front_mobility_sensors, "CRITICAL"))

        # Mobility Systems - Rear Wheels (6 sensors: 12-17)
        rear_mobility_sensors = [
            SensorSpec("RL Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("RL Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("RL Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY"),
            SensorSpec("RR Wheel Motor Current", "A", 0.0, 20.0, 8.0, 18.0, "ROVER", "MOBILITY"),
            SensorSpec("RR Wheel Motor Temperature", "°C", -50.0, 80.0, 20.0, 75.0, "ROVER", "MOBILITY"),
            SensorSpec("RR Suspension Angle", "degrees", -30.0, 30.0, 0.0, 25.0, "ROVER", "MOBILITY")
        ]
        equipment.append(EquipmentComponent("MSL-MOB-002", "Mobility System Rear", "MOBILITY", "Mars Surface", rear_mobility_sensors, "CRITICAL"))

        # Power Systems (8 sensors: 18-25)
        power_sensors = [
            SensorSpec("RTG Power Output", "W", 100.0, 150.0, 125.0, 110.0, "ROVER", "POWER"),
            SensorSpec("Primary Battery Voltage", "V", 24.0, 32.0, 28.0, 25.0, "ROVER", "POWER"),
            SensorSpec("Secondary Battery Voltage", "V", 24.0, 32.0, 28.0, 25.0, "ROVER", "POWER"),
            SensorSpec("Backup Battery Voltage", "V", 24.0, 32.0, 28.0, 25.0, "ROVER", "POWER"),
            SensorSpec("System Battery Voltage", "V", 24.0, 32.0, 28.0, 25.0, "ROVER", "POWER"),
            SensorSpec("Power Distribution Temp 1", "°C", -40.0, 60.0, 20.0, 55.0, "ROVER", "POWER"),
            SensorSpec("Power Distribution Temp 2", "°C", -40.0, 60.0, 20.0, 55.0, "ROVER", "POWER"),
            SensorSpec("Power Distribution Temp 3", "°C", -40.0, 60.0, 20.0, 55.0, "ROVER", "POWER")
        ]
        equipment.append(EquipmentComponent("MSL-PWR-001", "RTG Power System", "POWER", "Mars Surface", power_sensors, "CRITICAL"))

        # Environmental Sensors (12 sensors: 26-37)
        env_sensors = [
            SensorSpec("Atmospheric Pressure", "Pa", 400.0, 1200.0, 800.0, 450.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Wind Speed", "m/s", 0.0, 30.0, 10.0, 25.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Ambient Temperature 1", "°C", -90.0, 20.0, -40.0, -80.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Ambient Temperature 2", "°C", -90.0, 20.0, -40.0, -80.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Ambient Temperature 3", "°C", -90.0, 20.0, -40.0, -80.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Ground Temperature", "°C", -90.0, 20.0, -30.0, -80.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Internal Temperature", "°C", -20.0, 50.0, 20.0, 45.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Dust Level 1", "particles/m³", 0.0, 1000.0, 200.0, 800.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Dust Level 2", "particles/m³", 0.0, 1000.0, 200.0, 800.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Dust Level 3", "particles/m³", 0.0, 1000.0, 200.0, 800.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("Humidity", "%", 0.0, 10.0, 2.0, 8.0, "ROVER", "ENVIRONMENTAL"),
            SensorSpec("UV Radiation", "W/m²", 0.0, 50.0, 20.0, 45.0, "ROVER", "ENVIRONMENTAL")
        ]
        equipment.append(EquipmentComponent("MSL-ENV-001", "Environmental Monitoring", "ENVIRONMENTAL", "Mars Surface", env_sensors, "MEDIUM"))

        # Scientific Instruments (10 sensors: 38-47)
        science_sensors = [
            SensorSpec("ChemCam Spectrometer Status", "%", 80.0, 100.0, 98.0, 85.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("MAHLI Camera Status", "%", 80.0, 100.0, 98.0, 85.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("MARDI Camera Status", "%", 80.0, 100.0, 98.0, 85.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("SAM Spectrometer Temperature", "°C", 20.0, 80.0, 50.0, 75.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("APXS Deployment Angle", "degrees", 0.0, 180.0, 90.0, 170.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("Drill System Status", "%", 80.0, 100.0, 95.0, 85.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("Sample Analysis Unit Load", "%", 0.0, 100.0, 30.0, 90.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("Mastcam Temperature", "°C", -30.0, 40.0, 10.0, 35.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("Laser Power", "W", 5.0, 15.0, 10.0, 6.0, "ROVER", "SCIENTIFIC"),
            SensorSpec("Instrument Power Consumption", "W", 50.0, 200.0, 100.0, 180.0, "ROVER", "SCIENTIFIC")
        ]
        equipment.append(EquipmentComponent("MSL-SCI-001", "Scientific Instruments", "SCIENTIFIC", "Mars Surface", science_sensors, "HIGH"))

        # Communication Systems (6 sensors: 48-53)
        comm_sensors = [
            SensorSpec("High-Gain Antenna Pointing", "degrees", 0.0, 360.0, 180.0, 350.0, "ROVER", "COMMUNICATION"),
            SensorSpec("Low-Gain Antenna Status", "%", 80.0, 100.0, 98.0, 85.0, "ROVER", "COMMUNICATION"),
            SensorSpec("Data Buffer Usage 1", "%", 0.0, 100.0, 40.0, 95.0, "ROVER", "COMMUNICATION"),
            SensorSpec("Data Buffer Usage 2", "%", 0.0, 100.0, 40.0, 95.0, "ROVER", "COMMUNICATION"),
            SensorSpec("Data Buffer Usage 3", "%", 0.0, 100.0, 40.0, 95.0, "ROVER", "COMMUNICATION"),
            SensorSpec("Communication Power", "W", 20.0, 100.0, 50.0, 90.0, "ROVER", "COMMUNICATION")
        ]
        equipment.append(EquipmentComponent("MSL-COM-001", "Communication System", "COMMUNICATION", "Mars Surface", comm_sensors, "HIGH"))

        # Navigation & Orientation (1 sensor: 54)
        nav_sensors = [
            SensorSpec("IMU System Status", "%", 80.0, 100.0, 98.0, 85.0, "ROVER", "NAVIGATION")
        ]
        equipment.append(EquipmentComponent("MSL-NAV-001", "Navigation Computer", "NAVIGATION", "Mars Surface", nav_sensors, "CRITICAL"))

        return equipment

    def map_raw_data_to_equipment(self, smap_data: np.ndarray, msl_data: np.ndarray,
                                timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Map raw NASA data to realistic equipment telemetry"""
        telemetry_records = []

        # Validate input dimensions
        if smap_data.shape[1] != 25:
            raise ValueError(f"Expected SMAP data with 25 features, got {smap_data.shape[1]}")
        if msl_data.shape[1] != 55:
            raise ValueError(f"Expected MSL data with 55 features, got {msl_data.shape[1]}")

        # Process SMAP data
        smap_sensor_idx = 0
        for equipment in self.smap_equipment:
            for i, sample in enumerate(smap_data):
                timestamp = timestamps[i] if i < len(timestamps) else timestamps[-1]

                sensor_values = {}
                for j, sensor in enumerate(equipment.sensors):
                    if smap_sensor_idx + j < smap_data.shape[1]:
                        # Normalize and scale to realistic sensor ranges
                        raw_value = sample[smap_sensor_idx + j]
                        scaled_value = self._scale_sensor_value(raw_value, sensor)
                        sensor_values[sensor.name] = scaled_value

                telemetry_records.append({
                    'timestamp': timestamp,
                    'equipment_id': equipment.equipment_id,
                    'equipment_type': equipment.equipment_type,
                    'subsystem': equipment.subsystem,
                    'location': equipment.location,
                    'criticality': equipment.criticality,
                    'sensor_values': sensor_values,
                    'data_source': 'SMAP'
                })

            smap_sensor_idx += len(equipment.sensors)

        # Process MSL data
        msl_sensor_idx = 0
        for equipment in self.msl_equipment:
            for i, sample in enumerate(msl_data):
                timestamp = timestamps[i] if i < len(timestamps) else timestamps[-1]

                sensor_values = {}
                for j, sensor in enumerate(equipment.sensors):
                    if msl_sensor_idx + j < msl_data.shape[1]:
                        # Normalize and scale to realistic sensor ranges
                        raw_value = sample[msl_sensor_idx + j]
                        scaled_value = self._scale_sensor_value(raw_value, sensor)
                        sensor_values[sensor.name] = scaled_value

                telemetry_records.append({
                    'timestamp': timestamp,
                    'equipment_id': equipment.equipment_id,
                    'equipment_type': equipment.equipment_type,
                    'subsystem': equipment.subsystem,
                    'location': equipment.location,
                    'criticality': equipment.criticality,
                    'sensor_values': sensor_values,
                    'data_source': 'MSL'
                })

            msl_sensor_idx += len(equipment.sensors)

        return telemetry_records

    def _scale_sensor_value(self, raw_value: float, sensor: SensorSpec) -> float:
        """Scale raw NASA telemetry value to realistic sensor range while preserving signal patterns"""
        # IMPORTANT: Preserve the actual NASA signal patterns and variations
        # The raw_value comes directly from NASA telemetry data and contains
        # real aerospace signal patterns, anomalies, and fluctuations

        # Treat raw_value as already normalized NASA telemetry data [0, 1] range
        # Directly map the NASA signal patterns to the sensor's realistic range
        range_span = sensor.max_value - sensor.min_value

        # Preserve NASA signal patterns by using the full dynamic range
        # Instead of centering around nominal, use the full sensor range
        scaled_value = sensor.min_value + (raw_value * range_span)

        # Ensure value stays within sensor physical limits
        scaled_value = max(sensor.min_value, min(sensor.max_value, scaled_value))

        # Do NOT add artificial noise - the NASA data already contains
        # real signal variations, noise, and anomalies from actual spacecraft

        return round(scaled_value, 3)

    def get_all_equipment(self) -> List[EquipmentComponent]:
        """Get all equipment components (SMAP + MSL)"""
        return self.smap_equipment + self.msl_equipment

    def get_equipment_by_id(self, equipment_id: str) -> EquipmentComponent:
        """Get equipment component by ID"""
        all_equipment = self.get_all_equipment()
        for equipment in all_equipment:
            if equipment.equipment_id == equipment_id:
                return equipment
        raise ValueError(f"Equipment {equipment_id} not found")

    def get_equipment_by_subsystem(self, subsystem: str) -> List[EquipmentComponent]:
        """Get all equipment for a specific subsystem"""
        all_equipment = self.get_all_equipment()
        return [eq for eq in all_equipment if eq.subsystem == subsystem]

    def get_equipment_summary(self) -> Dict[str, Any]:
        """Get summary of all equipment"""
        all_equipment = self.get_all_equipment()

        summary = {
            'total_equipment': len(all_equipment),
            'smap_equipment': len(self.smap_equipment),
            'msl_equipment': len(self.msl_equipment),
            'total_sensors': sum(len(eq.sensors) for eq in all_equipment),
            'subsystems': {},
            'criticality_levels': {}
        }

        # Count by subsystem
        for equipment in all_equipment:
            subsystem = equipment.subsystem
            if subsystem not in summary['subsystems']:
                summary['subsystems'][subsystem] = 0
            summary['subsystems'][subsystem] += 1

            # Count by criticality
            criticality = equipment.criticality
            if criticality not in summary['criticality_levels']:
                summary['criticality_levels'][criticality] = 0
            summary['criticality_levels'][criticality] += 1

        return summary

    def get_equipment_info(self, equipment_id: str) -> Dict[str, Any]:
        """Get equipment information by ID for model manager compatibility

        Args:
            equipment_id: Equipment identifier

        Returns:
            Equipment information dictionary
        """
        try:
            equipment = self.get_equipment_by_id(equipment_id)
            return {
                'equipment_id': equipment.equipment_id,
                'equipment_type': equipment.equipment_type,
                'subsystem': equipment.subsystem,
                'location': equipment.location,
                'criticality': equipment.criticality,
                'sensors': [
                    {
                        'name': sensor.name,
                        'unit': sensor.unit,
                        'min_value': sensor.min_value,
                        'max_value': sensor.max_value,
                        'nominal_value': sensor.nominal_value,
                        'critical_threshold': sensor.critical_threshold
                    }
                    for sensor in equipment.sensors
                ],
                'sensor_count': len(equipment.sensors)
            }
        except Exception:
            # Return default info if equipment not found
            return {
                'equipment_id': equipment_id,
                'equipment_type': 'Unknown',
                'subsystem': 'Unknown',
                'location': 'Unknown',
                'criticality': 'MEDIUM',
                'sensors': [],
                'sensor_count': 0
            }

    def get_hierarchical_equipment_options(self) -> List[Dict[str, Any]]:
        """Get hierarchical equipment options for enhanced dropdown selection

        Returns:
            Hierarchical list of equipment options organized by mission and subsystem
        """
        options = []

        # Add mission-level groupings
        options.append({
            'label': '🛰️ NASA SMAP (Satellite) - 25 Sensors',
            'value': 'SMAP_MISSION',
            'disabled': True,
            'type': 'mission_header'
        })

        # Group SMAP equipment by subsystem
        smap_subsystems = {}
        for equipment in self.smap_equipment:
            if equipment.subsystem not in smap_subsystems:
                smap_subsystems[equipment.subsystem] = []
            smap_subsystems[equipment.subsystem].append(equipment)

        for subsystem, equipment_list in smap_subsystems.items():
            sensor_count = sum(len(eq.sensors) for eq in equipment_list)
            options.append({
                'label': f'├─ {subsystem} Systems ({sensor_count} sensors)',
                'value': f'SMAP_{subsystem}',
                'disabled': False,
                'type': 'subsystem_group',
                'mission': 'SMAP'
            })

            for equipment in equipment_list:
                options.append({
                    'label': f'│  └─ {equipment.equipment_id} ({len(equipment.sensors)} sensors)',
                    'value': equipment.equipment_id,
                    'disabled': False,
                    'type': 'equipment',
                    'mission': 'SMAP',
                    'subsystem': equipment.subsystem,
                    'sensor_count': len(equipment.sensors)
                })

        # Add MSL mission
        options.append({
            'label': '🚗 NASA MSL (Mars Rover) - 55 Sensors',
            'value': 'MSL_MISSION',
            'disabled': True,
            'type': 'mission_header'
        })

        # Group MSL equipment by subsystem
        msl_subsystems = {}
        for equipment in self.msl_equipment:
            if equipment.subsystem not in msl_subsystems:
                msl_subsystems[equipment.subsystem] = []
            msl_subsystems[equipment.subsystem].append(equipment)

        for subsystem, equipment_list in msl_subsystems.items():
            sensor_count = sum(len(eq.sensors) for eq in equipment_list)
            options.append({
                'label': f'├─ {subsystem} Systems ({sensor_count} sensors)',
                'value': f'MSL_{subsystem}',
                'disabled': False,
                'type': 'subsystem_group',
                'mission': 'MSL'
            })

            for equipment in equipment_list:
                options.append({
                    'label': f'│  └─ {equipment.equipment_id} ({len(equipment.sensors)} sensors)',
                    'value': equipment.equipment_id,
                    'disabled': False,
                    'type': 'equipment',
                    'mission': 'MSL',
                    'subsystem': equipment.subsystem,
                    'sensor_count': len(equipment.sensors)
                })

        return options

    def get_sensor_options_by_equipment(self, equipment_id: str) -> List[Dict[str, Any]]:
        """Get sensor options for specific equipment

        Args:
            equipment_id: Equipment identifier

        Returns:
            List of sensor options for dropdown
        """
        try:
            equipment = self.get_equipment_by_id(equipment_id)
            sensor_options = []

            for i, sensor in enumerate(equipment.sensors):
                # Create unique sensor identifier
                sensor_id = f"{equipment_id}_{sensor.name.replace(' ', '_').replace('.', '').lower()}"

                # Create display label with unit and range info
                range_info = f"{sensor.min_value:.1f}-{sensor.max_value:.1f} {sensor.unit}"
                label = f"{sensor.name} ({range_info})"

                sensor_options.append({
                    'label': label,
                    'value': sensor_id,
                    'sensor_name': sensor.name,
                    'unit': sensor.unit,
                    'min_value': sensor.min_value,
                    'max_value': sensor.max_value,
                    'nominal_value': sensor.nominal_value,
                    'critical_threshold': sensor.critical_threshold,
                    'equipment_id': equipment_id,
                    'subsystem': equipment.subsystem,
                    'criticality': equipment.criticality
                })

            return sensor_options

        except Exception as e:
            logger.warning(f"Could not get sensors for equipment {equipment_id}: {e}")
            return []

    def get_all_sensor_options(self) -> List[Dict[str, Any]]:
        """Get all sensor options organized by equipment

        Returns:
            Complete list of all 80 sensor options
        """
        all_sensor_options = []

        # Process SMAP equipment
        for equipment in self.smap_equipment:
            sensors = self.get_sensor_options_by_equipment(equipment.equipment_id)
            all_sensor_options.extend(sensors)

        # Process MSL equipment
        for equipment in self.msl_equipment:
            sensors = self.get_sensor_options_by_equipment(equipment.equipment_id)
            all_sensor_options.extend(sensors)

        return all_sensor_options

    def get_sensors_by_subsystem(self, subsystem: str) -> List[Dict[str, Any]]:
        """Get all sensors for a specific subsystem across both missions

        Args:
            subsystem: Subsystem name (POWER, COMMUNICATION, etc.)

        Returns:
            List of sensors in the subsystem
        """
        subsystem_sensors = []

        # Get equipment for this subsystem
        equipment_list = self.get_equipment_by_subsystem(subsystem)

        for equipment in equipment_list:
            sensors = self.get_sensor_options_by_equipment(equipment.equipment_id)
            subsystem_sensors.extend(sensors)

        return subsystem_sensors

    def get_sensor_metadata(self, sensor_id: str) -> Dict[str, Any]:
        """Get detailed metadata for a specific sensor

        Args:
            sensor_id: Sensor identifier (format: EQUIPMENT-ID_sensor_name)

        Returns:
            Sensor metadata dictionary
        """
        try:
            # Parse equipment ID from sensor ID
            parts = sensor_id.split('_')
            if len(parts) < 2:
                raise ValueError(f"Invalid sensor ID format: {sensor_id}")

            equipment_id = parts[0]
            sensor_name_parts = '_'.join(parts[1:])

            # Get equipment
            equipment = self.get_equipment_by_id(equipment_id)

            # Find matching sensor
            for sensor in equipment.sensors:
                normalized_name = sensor.name.replace(' ', '_').replace('.', '').lower()
                if normalized_name == sensor_name_parts:
                    return {
                        'sensor_id': sensor_id,
                        'sensor_name': sensor.name,
                        'equipment_id': equipment_id,
                        'equipment_type': equipment.equipment_type,
                        'subsystem': equipment.subsystem,
                        'location': equipment.location,
                        'criticality': equipment.criticality,
                        'unit': sensor.unit,
                        'min_value': sensor.min_value,
                        'max_value': sensor.max_value,
                        'nominal_value': sensor.nominal_value,
                        'critical_threshold': sensor.critical_threshold,
                        'value_range': sensor.max_value - sensor.min_value,
                        'mission': 'SMAP' if 'SMAP' in equipment_id else 'MSL'
                    }

            raise ValueError(f"Sensor not found: {sensor_id}")

        except Exception as e:
            logger.error(f"Error getting sensor metadata for {sensor_id}: {e}")
            return {
                'sensor_id': sensor_id,
                'sensor_name': 'Unknown',
                'equipment_id': 'Unknown',
                'error': str(e)
            }

    def get_comprehensive_sensor_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all sensors organized by various criteria

        Returns:
            Detailed sensor summary with multiple organizational views
        """
        all_equipment = self.get_all_equipment()

        # Basic counts
        total_sensors = sum(len(eq.sensors) for eq in all_equipment)
        smap_sensors = sum(len(eq.sensors) for eq in self.smap_equipment)
        msl_sensors = sum(len(eq.sensors) for eq in self.msl_equipment)

        # Group by subsystem
        subsystem_breakdown = {}
        for equipment in all_equipment:
            subsystem = equipment.subsystem
            if subsystem not in subsystem_breakdown:
                subsystem_breakdown[subsystem] = {
                    'equipment_count': 0,
                    'sensor_count': 0,
                    'equipment_list': [],
                    'smap_sensors': 0,
                    'msl_sensors': 0
                }

            subsystem_breakdown[subsystem]['equipment_count'] += 1
            subsystem_breakdown[subsystem]['sensor_count'] += len(equipment.sensors)
            subsystem_breakdown[subsystem]['equipment_list'].append({
                'equipment_id': equipment.equipment_id,
                'sensor_count': len(equipment.sensors),
                'mission': 'SMAP' if 'SMAP' in equipment.equipment_id else 'MSL'
            })

            if 'SMAP' in equipment.equipment_id:
                subsystem_breakdown[subsystem]['smap_sensors'] += len(equipment.sensors)
            else:
                subsystem_breakdown[subsystem]['msl_sensors'] += len(equipment.sensors)

        # Group by criticality
        criticality_breakdown = {}
        for equipment in all_equipment:
            crit = equipment.criticality
            if crit not in criticality_breakdown:
                criticality_breakdown[crit] = {
                    'equipment_count': 0,
                    'sensor_count': 0
                }
            criticality_breakdown[crit]['equipment_count'] += 1
            criticality_breakdown[crit]['sensor_count'] += len(equipment.sensors)

        return {
            'total_sensors': total_sensors,
            'smap_sensors': smap_sensors,
            'msl_sensors': msl_sensors,
            'total_equipment': len(all_equipment),
            'smap_equipment': len(self.smap_equipment),
            'msl_equipment': len(self.msl_equipment),
            'subsystem_breakdown': subsystem_breakdown,
            'criticality_breakdown': criticality_breakdown,
            'sensor_distribution': {
                'SMAP': {
                    'POWER': 6, 'COMMUNICATION': 5, 'ATTITUDE': 6,
                    'THERMAL': 4, 'PAYLOAD': 4
                },
                'MSL': {
                    'MOBILITY': 18, 'POWER': 8, 'ENVIRONMENTAL': 12,
                    'SCIENTIFIC': 10, 'COMMUNICATION': 6, 'NAVIGATION': 1
                }
            }
        }

    def get_equipment_by_channel(self, channel: str) -> EquipmentComponent:
        """Get equipment component by channel name for database integration

        Args:
            channel: Channel identifier (e.g., 'A-1', 'T-5', etc.)

        Returns:
            EquipmentComponent: The equipment associated with this channel

        Raises:
            ValueError: If channel not found
        """
        # Map NASA channel names to our equipment IDs
        channel_mapping = {
            # SMAP channels (A-1 to T-13 based on NASA dataset)
            'A-1': 'SMAP-PWR-001', 'A-2': 'SMAP-PWR-001', 'A-3': 'SMAP-PWR-001',
            'A-4': 'SMAP-PWR-001', 'A-5': 'SMAP-PWR-001', 'A-6': 'SMAP-PWR-001',
            'A-7': 'SMAP-COM-001', 'A-8': 'SMAP-COM-001', 'A-9': 'SMAP-COM-001',
            'B-1': 'SMAP-COM-001', 'C-1': 'SMAP-COM-001', 'C-2': 'SMAP-ATT-001',
            'D-1': 'SMAP-ATT-001', 'D-2': 'SMAP-ATT-001', 'D-3': 'SMAP-ATT-001',
            'D-4': 'SMAP-ATT-001', 'D-5': 'SMAP-ATT-001', 'D-6': 'SMAP-THM-001',
            'D-7': 'SMAP-THM-001', 'D-8': 'SMAP-THM-001', 'D-9': 'SMAP-THM-001',
            'D-11': 'SMAP-PAY-001', 'D-12': 'SMAP-PAY-001', 'D-13': 'SMAP-PAY-001',
            'D-14': 'SMAP-PAY-001', 'D-15': 'SMAP-PAY-001', 'D-16': 'SMAP-PAY-001',
            'E-1': 'MSL-MOB-001', 'E-2': 'MSL-MOB-001', 'E-3': 'MSL-MOB-001',
            'E-4': 'MSL-MOB-001', 'E-5': 'MSL-MOB-001', 'E-6': 'MSL-MOB-001',
            'E-7': 'MSL-MOB-001', 'E-8': 'MSL-MOB-001', 'E-9': 'MSL-MOB-001',
            'E-10': 'MSL-MOB-001', 'E-11': 'MSL-MOB-001', 'E-12': 'MSL-MOB-001',
            'E-13': 'MSL-MOB-001', 'F-1': 'MSL-MOB-002', 'F-2': 'MSL-MOB-002',
            'F-3': 'MSL-MOB-002', 'F-4': 'MSL-MOB-002', 'F-5': 'MSL-MOB-002',
            'F-7': 'MSL-MOB-002', 'F-8': 'MSL-MOB-002', 'G-1': 'MSL-PWR-001',
            'G-2': 'MSL-PWR-001', 'G-3': 'MSL-PWR-001', 'G-4': 'MSL-PWR-001',
            'G-6': 'MSL-PWR-001', 'G-7': 'MSL-PWR-001', 'M-1': 'MSL-ENV-001',
            'M-2': 'MSL-ENV-001', 'M-3': 'MSL-ENV-001', 'M-4': 'MSL-ENV-001',
            'M-5': 'MSL-ENV-001', 'M-6': 'MSL-ENV-001', 'M-7': 'MSL-ENV-001',
            'P-1': 'MSL-SCI-001', 'P-2': 'MSL-SCI-001', 'P-3': 'MSL-SCI-001',
            'P-4': 'MSL-SCI-001', 'P-7': 'MSL-SCI-001', 'P-10': 'MSL-SCI-001',
            'P-11': 'MSL-SCI-001', 'P-14': 'MSL-SCI-001', 'P-15': 'MSL-SCI-001',
            'R-1': 'MSL-COM-001', 'S-1': 'MSL-COM-001', 'S-2': 'MSL-COM-001',
            'T-1': 'MSL-COM-001', 'T-2': 'MSL-COM-001', 'T-3': 'MSL-COM-001',
            'T-4': 'MSL-NAV-001', 'T-5': 'MSL-NAV-001', 'T-8': 'MSL-NAV-001',
            'T-9': 'MSL-NAV-001', 'T-10': 'MSL-NAV-001', 'T-12': 'MSL-NAV-001',
            'T-13': 'MSL-NAV-001'
        }

        # Get equipment ID from channel mapping
        equipment_id = channel_mapping.get(channel)
        if not equipment_id:
            raise ValueError(f"Channel '{channel}' not found in equipment mapping")

        # Return the equipment component
        return self.get_equipment_by_id(equipment_id)

    def get_channel_list(self) -> List[str]:
        """Get list of all available channels for database integration"""
        return [
            'A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'A-7', 'A-8', 'A-9',
            'B-1', 'C-1', 'C-2', 'D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6',
            'D-7', 'D-8', 'D-9', 'D-11', 'D-12', 'D-13', 'D-14', 'D-15', 'D-16',
            'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9',
            'E-10', 'E-11', 'E-12', 'E-13', 'F-1', 'F-2', 'F-3', 'F-4', 'F-5',
            'F-7', 'F-8', 'G-1', 'G-2', 'G-3', 'G-4', 'G-6', 'G-7', 'M-1',
            'M-2', 'M-3', 'M-4', 'M-5', 'M-6', 'M-7', 'P-1', 'P-2', 'P-3',
            'P-4', 'P-7', 'P-10', 'P-11', 'P-14', 'P-15', 'R-1', 'S-1', 'S-2',
            'T-1', 'T-2', 'T-3', 'T-4', 'T-5', 'T-8', 'T-9', 'T-10', 'T-12', 'T-13'
        ]


# Initialize global mapper instance
equipment_mapper = IoTEquipmentMapper()