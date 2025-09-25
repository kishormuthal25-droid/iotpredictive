"""
Dropdown State Manager
Centralized management for all dropdown interactions across the dashboard
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DropdownOption:
    """Standard structure for dropdown options"""
    label: str
    value: str
    disabled: bool = False
    title: str = ""  # Tooltip text
    group: str = ""  # For grouping options


@dataclass
class EquipmentInfo:
    """Equipment information for dropdown display"""
    equipment_id: str
    equipment_type: str
    subsystem: str
    total_sensors: int
    criticality: str
    spacecraft: str  # 'smap' or 'msl'


class DropdownStateManager:
    """
    Centralized manager for all dropdown state and interactions
    Handles Equipment → Sensor → Metric cascading dropdowns
    """

    def __init__(self):
        """Initialize the dropdown state manager"""
        self.equipment_mapper = None
        self.unified_controller = None
        self._equipment_cache = {}
        self._sensor_cache = {}
        self._metric_cache = {}
        self._load_dependencies()

        # NASA Equipment Hierarchy
        self.nasa_equipment_hierarchy = {
            'SMAP': {
                'name': '🛰️ SMAP Satellite',
                'subsystems': ['POWER', 'COMMUNICATION', 'ATTITUDE', 'THERMAL', 'PAYLOAD'],
                'total_sensors': 25,
                'criticality_map': {
                    'POWER': 'CRITICAL',
                    'ATTITUDE': 'CRITICAL',
                    'COMMUNICATION': 'HIGH',
                    'THERMAL': 'HIGH',
                    'PAYLOAD': 'HIGH'
                }
            },
            'MSL': {
                'name': '🤖 MSL Mars Rover',
                'subsystems': ['MOBILITY', 'POWER', 'ENVIRONMENTAL', 'SCIENTIFIC', 'COMMUNICATION', 'NAVIGATION'],
                'total_sensors': 55,
                'criticality_map': {
                    'MOBILITY': 'CRITICAL',
                    'POWER': 'CRITICAL',
                    'NAVIGATION': 'CRITICAL',
                    'COMMUNICATION': 'HIGH',
                    'SCIENTIFIC': 'HIGH',
                    'ENVIRONMENTAL': 'MEDIUM'
                }
            }
        }

        # Standard metric types for sensor data
        self.standard_metrics = [
            {'label': 'Raw Value', 'value': 'raw_value'},
            {'label': 'Anomaly Score', 'value': 'anomaly_score'},
            {'label': 'Health Score', 'value': 'health_score'},
            {'label': 'Temperature', 'value': 'temperature'},
            {'label': 'Pressure', 'value': 'pressure'},
            {'label': 'Vibration', 'value': 'vibration'},
            {'label': 'Current', 'value': 'current'},
            {'label': 'Voltage', 'value': 'voltage'},
            {'label': 'Power', 'value': 'power'},
            {'label': 'Frequency', 'value': 'frequency'}
        ]

        logger.info("DropdownStateManager initialized successfully")

    def _load_dependencies(self):
        """Load required dependencies with error handling"""
        try:
            from src.data_ingestion.equipment_mapper import equipment_mapper
            self.equipment_mapper = equipment_mapper
            logger.info("Equipment mapper loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to load equipment mapper: {e}")

        try:
            from src.data_ingestion.unified_data_controller import get_unified_controller
            self.unified_controller = get_unified_controller()
            logger.info("Unified controller loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to load unified controller: {e}")

    def get_equipment_options(self, include_all: bool = True,
                            filter_by_spacecraft: Optional[str] = None,
                            filter_by_criticality: Optional[str] = None) -> List[DropdownOption]:
        """
        Get all equipment options for dropdown

        Args:
            include_all: Include 'All Equipment' option
            filter_by_spacecraft: Filter by 'smap' or 'msl'
            filter_by_criticality: Filter by 'CRITICAL', 'HIGH', 'MEDIUM'

        Returns:
            List of DropdownOption objects for equipment
        """
        try:
            options = []

            # Add 'All Equipment' option if requested
            if include_all:
                options.append(DropdownOption(
                    label="🌐 All Equipment",
                    value="ALL",
                    title="Show data from all equipment"
                ))

            # Get equipment from mapper
            if self.equipment_mapper:
                all_equipment = self.equipment_mapper.get_all_equipment()

                # Group equipment by spacecraft and subsystem
                grouped_equipment = defaultdict(lambda: defaultdict(list))

                for equipment in all_equipment:
                    spacecraft = equipment.equipment_id.split('-')[0]  # SMAP or MSL
                    subsystem = equipment.subsystem

                    # Apply filters
                    if filter_by_spacecraft and spacecraft.lower() != filter_by_spacecraft.lower():
                        continue

                    criticality = self._get_equipment_criticality(spacecraft, subsystem)
                    if filter_by_criticality and criticality != filter_by_criticality:
                        continue

                    grouped_equipment[spacecraft][subsystem].append(equipment)

                # Create options with proper grouping
                for spacecraft, subsystems in grouped_equipment.items():
                    spacecraft_info = self.nasa_equipment_hierarchy.get(spacecraft, {})
                    spacecraft_name = spacecraft_info.get('name', spacecraft)

                    for subsystem, equipment_list in subsystems.items():
                        for equipment in equipment_list:
                            criticality = self._get_equipment_criticality(spacecraft, subsystem)
                            criticality_icon = self._get_criticality_icon(criticality)

                            label = f"{criticality_icon} {equipment.equipment_id} ({equipment.equipment_type})"
                            title = f"{spacecraft_name} - {subsystem} - {criticality} Priority - {len(equipment.sensors)} sensors"

                            options.append(DropdownOption(
                                label=label,
                                value=equipment.equipment_id,
                                title=title,
                                group=f"{spacecraft_name} - {subsystem}"
                            ))

            # Cache the results
            cache_key = f"{include_all}_{filter_by_spacecraft}_{filter_by_criticality}"
            self._equipment_cache[cache_key] = options

            logger.info(f"Generated {len(options)} equipment options")
            return options

        except Exception as e:
            logger.error(f"Error getting equipment options: {e}")
            return [DropdownOption(label="Error loading equipment", value="ERROR", disabled=True)]

    def get_sensor_options_for_equipment(self, equipment_id: str,
                                       include_all: bool = True) -> List[DropdownOption]:
        """
        Get sensor options for specific equipment

        Args:
            equipment_id: Equipment ID to get sensors for
            include_all: Include 'All Sensors' option

        Returns:
            List of DropdownOption objects for sensors
        """
        try:
            options = []

            # Handle 'ALL' equipment case
            if equipment_id == "ALL":
                if include_all:
                    options.append(DropdownOption(
                        label="🌐 All Sensors",
                        value="ALL",
                        title="Show data from all sensors across all equipment"
                    ))

                # Get all sensors from all equipment
                if self.equipment_mapper:
                    all_equipment = self.equipment_mapper.get_all_equipment()
                    for equipment in all_equipment:
                        for sensor in equipment.sensors:
                            sensor_label = f"{equipment.equipment_id} - {sensor.name}"
                            options.append(DropdownOption(
                                label=sensor_label,
                                value=f"{equipment.equipment_id}::{sensor.name}",
                                title=f"{sensor.description} ({sensor.unit})",
                                group=equipment.equipment_id
                            ))

                return options

            # Add 'All Sensors' option for specific equipment
            if include_all:
                options.append(DropdownOption(
                    label="📊 All Sensors",
                    value="ALL",
                    title=f"Show data from all sensors on {equipment_id}"
                ))

            # Get sensors for specific equipment
            if self.equipment_mapper:
                equipment_sensors = self.equipment_mapper.get_sensor_options_by_equipment(equipment_id)

                for sensor_option in equipment_sensors:
                    # Get additional sensor details
                    sensor_details = self._get_sensor_details(equipment_id, sensor_option['value'])

                    options.append(DropdownOption(
                        label=sensor_option['label'],
                        value=sensor_option['value'],
                        title=sensor_details.get('description', ''),
                        group=equipment_id
                    ))

            # Cache the results
            self._sensor_cache[equipment_id] = options

            logger.info(f"Generated {len(options)} sensor options for {equipment_id}")
            return options

        except Exception as e:
            logger.error(f"Error getting sensor options for {equipment_id}: {e}")
            return [DropdownOption(label="Error loading sensors", value="ERROR", disabled=True)]

    def get_metric_options_for_sensor(self, equipment_id: str, sensor_id: str,
                                    include_calculated: bool = True) -> List[DropdownOption]:
        """
        Get metric options for specific sensor

        Args:
            equipment_id: Equipment ID
            sensor_id: Sensor ID
            include_calculated: Include calculated metrics (anomaly scores, health scores)

        Returns:
            List of DropdownOption objects for metrics
        """
        try:
            options = []

            # Standard metrics available for all sensors
            for metric in self.standard_metrics:
                # Skip calculated metrics if not requested
                if not include_calculated and metric['value'] in ['anomaly_score', 'health_score']:
                    continue

                options.append(DropdownOption(
                    label=metric['label'],
                    value=metric['value'],
                    title=f"{metric['label']} measurement for {sensor_id}"
                ))

            # Add sensor-specific metrics based on sensor type
            specific_metrics = self._get_sensor_specific_metrics(equipment_id, sensor_id)
            options.extend(specific_metrics)

            # Cache the results
            cache_key = f"{equipment_id}::{sensor_id}"
            self._metric_cache[cache_key] = options

            logger.info(f"Generated {len(options)} metric options for {equipment_id}::{sensor_id}")
            return options

        except Exception as e:
            logger.error(f"Error getting metric options for {equipment_id}::{sensor_id}: {e}")
            return [DropdownOption(label="Raw Value", value="raw_value")]

    def get_equipment_info(self, equipment_id: str) -> Optional[EquipmentInfo]:
        """
        Get detailed equipment information

        Args:
            equipment_id: Equipment ID to get info for

        Returns:
            EquipmentInfo object or None if not found
        """
        try:
            if self.equipment_mapper:
                all_equipment = self.equipment_mapper.get_all_equipment()

                for equipment in all_equipment:
                    if equipment.equipment_id == equipment_id:
                        spacecraft = equipment_id.split('-')[0]
                        criticality = self._get_equipment_criticality(spacecraft, equipment.subsystem)

                        return EquipmentInfo(
                            equipment_id=equipment.equipment_id,
                            equipment_type=equipment.equipment_type,
                            subsystem=equipment.subsystem,
                            total_sensors=len(equipment.sensors),
                            criticality=criticality,
                            spacecraft=spacecraft.lower()
                        )

            return None

        except Exception as e:
            logger.error(f"Error getting equipment info for {equipment_id}: {e}")
            return None

    def validate_selection(self, equipment_id: str, sensor_id: str = None,
                         metric_id: str = None) -> Dict[str, bool]:
        """
        Validate dropdown selections

        Args:
            equipment_id: Selected equipment ID
            sensor_id: Selected sensor ID (optional)
            metric_id: Selected metric ID (optional)

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'equipment_valid': False,
            'sensor_valid': False,
            'metric_valid': False,
            'combination_valid': False
        }

        try:
            # Validate equipment
            if equipment_id == "ALL" or self.get_equipment_info(equipment_id):
                validation_result['equipment_valid'] = True

            # Validate sensor if provided
            if sensor_id:
                sensor_options = self.get_sensor_options_for_equipment(equipment_id, include_all=True)
                if any(opt.value == sensor_id for opt in sensor_options):
                    validation_result['sensor_valid'] = True

            # Validate metric if provided
            if metric_id and sensor_id:
                metric_options = self.get_metric_options_for_sensor(equipment_id, sensor_id)
                if any(opt.value == metric_id for opt in metric_options):
                    validation_result['metric_valid'] = True

            # Overall combination validity
            validation_result['combination_valid'] = (
                validation_result['equipment_valid'] and
                (not sensor_id or validation_result['sensor_valid']) and
                (not metric_id or validation_result['metric_valid'])
            )

        except Exception as e:
            logger.error(f"Error validating selection: {e}")

        return validation_result

    def _get_equipment_criticality(self, spacecraft: str, subsystem: str) -> str:
        """Get criticality level for equipment"""
        spacecraft_info = self.nasa_equipment_hierarchy.get(spacecraft.upper(), {})
        criticality_map = spacecraft_info.get('criticality_map', {})
        return criticality_map.get(subsystem, 'MEDIUM')

    def _get_criticality_icon(self, criticality: str) -> str:
        """Get icon for criticality level"""
        icons = {
            'CRITICAL': '🔴',
            'HIGH': '🟡',
            'MEDIUM': '🟢',
            'LOW': '⚪'
        }
        return icons.get(criticality, '⚪')

    def _get_sensor_details(self, equipment_id: str, sensor_id: str) -> Dict[str, str]:
        """Get detailed sensor information"""
        try:
            if self.equipment_mapper:
                all_equipment = self.equipment_mapper.get_all_equipment()

                for equipment in all_equipment:
                    if equipment.equipment_id == equipment_id:
                        for sensor in equipment.sensors:
                            if sensor.name == sensor_id:
                                return {
                                    'description': getattr(sensor, 'description', sensor.name),
                                    'unit': getattr(sensor, 'unit', ''),
                                    'range': getattr(sensor, 'range', ''),
                                    'type': getattr(sensor, 'sensor_type', 'numeric')
                                }
        except Exception as e:
            logger.error(f"Error getting sensor details: {e}")

        return {'description': sensor_id, 'unit': '', 'range': '', 'type': 'numeric'}

    def _get_sensor_specific_metrics(self, equipment_id: str, sensor_id: str) -> List[DropdownOption]:
        """Get sensor-specific metric options"""
        specific_metrics = []

        # Add metrics based on sensor name patterns
        sensor_lower = sensor_id.lower()

        if 'temperature' in sensor_lower or 'thermal' in sensor_lower:
            specific_metrics.extend([
                DropdownOption(label="Temperature Trend", value="temp_trend"),
                DropdownOption(label="Temperature Rate", value="temp_rate")
            ])

        if 'current' in sensor_lower:
            specific_metrics.extend([
                DropdownOption(label="Current RMS", value="current_rms"),
                DropdownOption(label="Current Peak", value="current_peak")
            ])

        if 'voltage' in sensor_lower:
            specific_metrics.extend([
                DropdownOption(label="Voltage RMS", value="voltage_rms"),
                DropdownOption(label="Voltage Ripple", value="voltage_ripple")
            ])

        if 'vibration' in sensor_lower:
            specific_metrics.extend([
                DropdownOption(label="Vibration FFT", value="vibration_fft"),
                DropdownOption(label="Vibration Envelope", value="vibration_envelope")
            ])

        return specific_metrics

    def clear_cache(self):
        """Clear all cached dropdown data"""
        self._equipment_cache.clear()
        self._sensor_cache.clear()
        self._metric_cache.clear()
        logger.info("Dropdown caches cleared")

    def get_cache_status(self) -> Dict[str, int]:
        """Get cache status information"""
        return {
            'equipment_cache_size': len(self._equipment_cache),
            'sensor_cache_size': len(self._sensor_cache),
            'metric_cache_size': len(self._metric_cache)
        }


# Global instance for use across the dashboard
dropdown_state_manager = DropdownStateManager()