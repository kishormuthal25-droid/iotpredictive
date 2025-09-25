"""
Quick Select Manager
NASA mission-specific equipment quick select presets for rapid navigation
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    """NASA mission phases"""
    LAUNCH = "launch"
    CRUISE = "cruise"
    OPERATIONS = "operations"
    EXTENDED_MISSION = "extended"
    SAFING = "safing"
    EMERGENCY = "emergency"
    COMMUNICATION = "communication"
    MAINTENANCE = "maintenance"


class SystemCriticality(Enum):
    """System criticality levels"""
    MISSION_CRITICAL = "mission_critical"
    HIGH_PRIORITY = "high_priority"
    NOMINAL = "nominal"
    MONITORING_ONLY = "monitoring_only"


@dataclass
class QuickSelectPreset:
    """Quick select preset configuration"""
    id: str
    name: str
    description: str
    icon: str
    equipment_ids: List[str]
    sensor_categories: List[str]
    mission_phase: MissionPhase
    criticality: SystemCriticality
    spacecraft: List[str]  # ['smap', 'msl'] or ['both']
    subsystems: List[str]

    # Display options
    priority: int = 0  # Higher priority shows first
    color: str = "primary"
    shortcut_key: Optional[str] = None

    # Contextual information
    use_cases: List[str] = field(default_factory=list)
    related_presets: List[str] = field(default_factory=list)
    estimated_data_points: Optional[int] = None


class QuickSelectManager:
    """
    NASA mission-specific equipment quick select presets
    Provides rapid equipment selection for common operational scenarios
    """

    def __init__(self):
        """Initialize quick select manager"""
        # Import managers
        self.dropdown_manager = None
        self.filter_manager = None
        self.shared_state = None
        self._initialize_managers()

        # NASA equipment specifications
        self.nasa_equipment_specs = self._initialize_nasa_specs()

        # Mission-specific presets
        self.mission_presets = self._create_mission_presets()

        # Equipment groupings
        self.equipment_groups = self._create_equipment_groups()

        # Quick action presets
        self.quick_actions = self._create_quick_actions()

        logger.info("QuickSelectManager initialized with NASA mission presets")

    def _initialize_managers(self):
        """Initialize manager dependencies"""
        try:
            from .dropdown_manager import dropdown_state_manager
            from .filter_manager import filter_manager as fm
            from ..state.shared_state import shared_state_manager

            self.dropdown_manager = dropdown_state_manager
            self.filter_manager = fm
            self.shared_state = shared_state_manager

        except ImportError as e:
            logger.warning(f"Could not import managers: {e}")

    def _initialize_nasa_specs(self) -> Dict[str, Any]:
        """Initialize NASA equipment specifications"""
        return {
            'smap': {
                'name': '🛰️ SMAP Satellite',
                'mission_start': '2015-01-31',
                'equipment_count': 5,
                'sensor_count': 25,
                'subsystems': {
                    'POWER': {
                        'criticality': 'MISSION_CRITICAL',
                        'sensors': ['Solar Panel Voltage', 'Battery Current', 'Power Distribution Temperature',
                                  'Charging Controller Status', 'Bus Voltage', 'Load Current'],
                        'equipment_ids': ['SMAP-PWR-001']
                    },
                    'COMMUNICATION': {
                        'criticality': 'HIGH_PRIORITY',
                        'sensors': ['Antenna Gain', 'Signal Strength', 'Data Transmission Rate',
                                  'Communication Temperature', 'Uplink Quality'],
                        'equipment_ids': ['SMAP-COM-001']
                    },
                    'ATTITUDE': {
                        'criticality': 'MISSION_CRITICAL',
                        'sensors': ['Gyroscope X', 'Gyroscope Y', 'Gyroscope Z',
                                  'Accelerometer X', 'Accelerometer Y', 'Star Tracker'],
                        'equipment_ids': ['SMAP-ATT-001']
                    },
                    'THERMAL': {
                        'criticality': 'HIGH_PRIORITY',
                        'sensors': ['Thermal Radiator Temperature', 'Heat Exchanger Status',
                                  'Instrument Temperature', 'Electronics Temperature'],
                        'equipment_ids': ['SMAP-THM-001']
                    },
                    'PAYLOAD': {
                        'criticality': 'HIGH_PRIORITY',
                        'sensors': ['Soil Moisture Sensor', 'Antenna Reflector',
                                  'Calibration Reference', 'Data Processor'],
                        'equipment_ids': ['SMAP-PAY-001']
                    }
                }
            },
            'msl': {
                'name': '🤖 MSL Mars Rover',
                'mission_start': '2011-11-26',
                'equipment_count': 7,
                'sensor_count': 55,
                'subsystems': {
                    'MOBILITY': {
                        'criticality': 'MISSION_CRITICAL',
                        'sensors': ['Front Left Motor', 'Front Right Motor', 'Middle Left Motor',
                                  'Middle Right Motor', 'Rear Left Motor', 'Rear Right Motor',
                                  'Suspension Front', 'Suspension Rear', 'Wheel Torque',
                                  'Drive System', 'Steering Actuator', 'Mobility Health'],
                        'equipment_ids': ['MSL-MOB-001', 'MSL-MOB-002']
                    },
                    'POWER': {
                        'criticality': 'MISSION_CRITICAL',
                        'sensors': ['RTG Power Output', 'Battery Level', 'Power Distribution',
                                  'Thermal Power', 'Battery Temperature', 'Power Bus Voltage',
                                  'Load Management', 'Backup Power'],
                        'equipment_ids': ['MSL-PWR-001']
                    },
                    'ENVIRONMENTAL': {
                        'criticality': 'NOMINAL',
                        'sensors': ['Atmospheric Pressure', 'Air Temperature', 'Wind Speed',
                                  'Wind Direction', 'Humidity', 'UV Radiation', 'Dust Level',
                                  'Atmospheric Composition', 'Weather Station', 'Environmental Monitor',
                                  'Dust Storm Detector', 'Seasonal Monitor'],
                        'equipment_ids': ['MSL-ENV-001']
                    },
                    'SCIENTIFIC': {
                        'criticality': 'HIGH_PRIORITY',
                        'sensors': ['ChemCam Laser', 'MAHLI Camera', 'APXS Spectrometer',
                                  'SAM Analysis', 'CheMin Drill', 'Rock Abrasion',
                                  'Sample Collection', 'Lab Analysis', 'Instrument Arm',
                                  'Scientific Processor'],
                        'equipment_ids': ['MSL-SCI-001']
                    },
                    'COMMUNICATION': {
                        'criticality': 'HIGH_PRIORITY',
                        'sensors': ['UHF Antenna', 'High Gain Antenna', 'Low Gain Antenna',
                                  'Communication Processor', 'Data Buffer', 'Signal Processing'],
                        'equipment_ids': ['MSL-COM-001']
                    },
                    'NAVIGATION': {
                        'criticality': 'MISSION_CRITICAL',
                        'sensors': ['IMU Navigation'],
                        'equipment_ids': ['MSL-NAV-001']
                    }
                }
            }
        }

    def _create_mission_presets(self) -> Dict[str, QuickSelectPreset]:
        """Create mission-specific presets"""
        presets = {}

        # Critical systems monitoring
        presets['mission_critical'] = QuickSelectPreset(
            id='mission_critical',
            name='🚨 Mission Critical Systems',
            description='Monitor all mission-critical equipment for both SMAP and MSL',
            icon='🚨',
            equipment_ids=self._get_critical_equipment(),
            sensor_categories=['power', 'attitude', 'mobility', 'navigation'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.MISSION_CRITICAL,
            spacecraft=['both'],
            subsystems=['POWER', 'ATTITUDE', 'MOBILITY', 'NAVIGATION'],
            priority=100,
            color='danger',
            shortcut_key='Ctrl+1',
            use_cases=['Emergency response', 'System health checks', 'Anomaly investigation'],
            estimated_data_points=800
        )

        # SMAP operations
        presets['smap_operations'] = QuickSelectPreset(
            id='smap_operations',
            name='🛰️ SMAP Operations',
            description='SMAP satellite operational monitoring',
            icon='🛰️',
            equipment_ids=self._get_smap_equipment(),
            sensor_categories=['power', 'thermal', 'attitude', 'payload', 'communication'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.HIGH_PRIORITY,
            spacecraft=['smap'],
            subsystems=['POWER', 'THERMAL', 'ATTITUDE', 'PAYLOAD', 'COMMUNICATION'],
            priority=90,
            color='primary',
            shortcut_key='Ctrl+2',
            use_cases=['Daily operations', 'Soil moisture monitoring', 'Orbit maintenance'],
            estimated_data_points=300
        )

        # MSL operations
        presets['msl_operations'] = QuickSelectPreset(
            id='msl_operations',
            name='🤖 MSL Rover Operations',
            description='Mars Science Laboratory rover operational monitoring',
            icon='🤖',
            equipment_ids=self._get_msl_equipment(),
            sensor_categories=['mobility', 'power', 'scientific', 'environmental', 'communication'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.HIGH_PRIORITY,
            spacecraft=['msl'],
            subsystems=['MOBILITY', 'POWER', 'SCIENTIFIC', 'ENVIRONMENTAL', 'COMMUNICATION'],
            priority=90,
            color='warning',
            shortcut_key='Ctrl+3',
            use_cases=['Daily operations', 'Scientific exploration', 'Terrain navigation'],
            estimated_data_points=650
        )

        # Power systems focus
        presets['power_systems'] = QuickSelectPreset(
            id='power_systems',
            name='⚡ Power Systems',
            description='Monitor power generation, distribution, and consumption',
            icon='⚡',
            equipment_ids=self._get_power_equipment(),
            sensor_categories=['power', 'thermal'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.MISSION_CRITICAL,
            spacecraft=['both'],
            subsystems=['POWER'],
            priority=80,
            color='success',
            shortcut_key='Ctrl+P',
            use_cases=['Power budget analysis', 'Battery health monitoring', 'Solar panel performance'],
            estimated_data_points=200
        )

        # Communication systems
        presets['communication'] = QuickSelectPreset(
            id='communication',
            name='📡 Communication Systems',
            description='Monitor communication links and data transmission',
            icon='📡',
            equipment_ids=self._get_communication_equipment(),
            sensor_categories=['communication', 'signal'],
            mission_phase=MissionPhase.COMMUNICATION,
            criticality=SystemCriticality.HIGH_PRIORITY,
            spacecraft=['both'],
            subsystems=['COMMUNICATION'],
            priority=75,
            color='info',
            shortcut_key='Ctrl+C',
            use_cases=['Communication windows', 'Data downlink', 'Signal quality assessment'],
            estimated_data_points=150
        )

        # Emergency response
        presets['emergency_response'] = QuickSelectPreset(
            id='emergency_response',
            name='🚨 Emergency Response',
            description='Critical systems for emergency response and safing',
            icon='🚨',
            equipment_ids=self._get_emergency_equipment(),
            sensor_categories=['power', 'attitude', 'mobility', 'thermal'],
            mission_phase=MissionPhase.EMERGENCY,
            criticality=SystemCriticality.MISSION_CRITICAL,
            spacecraft=['both'],
            subsystems=['POWER', 'ATTITUDE', 'MOBILITY'],
            priority=95,
            color='danger',
            shortcut_key='Ctrl+E',
            use_cases=['System safing', 'Fault isolation', 'Recovery operations'],
            estimated_data_points=400
        )

        # Scientific instruments
        presets['scientific_instruments'] = QuickSelectPreset(
            id='scientific_instruments',
            name='🔬 Scientific Instruments',
            description='Monitor scientific instruments and data collection',
            icon='🔬',
            equipment_ids=self._get_scientific_equipment(),
            sensor_categories=['scientific', 'payload', 'environmental'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.HIGH_PRIORITY,
            spacecraft=['both'],
            subsystems=['SCIENTIFIC', 'PAYLOAD', 'ENVIRONMENTAL'],
            priority=70,
            color='secondary',
            shortcut_key='Ctrl+S',
            use_cases=['Data collection', 'Instrument calibration', 'Science operations'],
            estimated_data_points=500
        )

        # Recent anomalies
        presets['recent_anomalies'] = QuickSelectPreset(
            id='recent_anomalies',
            name='⚠️ Recent Anomalies',
            description='Equipment with recent anomaly detections',
            icon='⚠️',
            equipment_ids=[],  # Will be populated dynamically
            sensor_categories=['all'],
            mission_phase=MissionPhase.OPERATIONS,
            criticality=SystemCriticality.HIGH_PRIORITY,
            spacecraft=['both'],
            subsystems=[],
            priority=85,
            color='warning',
            shortcut_key='Ctrl+A',
            use_cases=['Anomaly investigation', 'Fault analysis', 'Trending'],
            estimated_data_points=0  # Dynamic
        )

        return presets

    def _create_equipment_groups(self) -> Dict[str, Dict[str, Any]]:
        """Create logical equipment groupings"""
        return {
            'critical_systems': {
                'name': '🔴 Critical Systems',
                'equipment_ids': self._get_critical_equipment(),
                'description': 'Mission-critical equipment that must be monitored continuously'
            },
            'power_generation': {
                'name': '🔋 Power Generation',
                'equipment_ids': ['SMAP-PWR-001', 'MSL-PWR-001'],
                'description': 'Power generation and management systems'
            },
            'navigation_control': {
                'name': '🧭 Navigation & Control',
                'equipment_ids': ['SMAP-ATT-001', 'MSL-MOB-001', 'MSL-MOB-002', 'MSL-NAV-001'],
                'description': 'Navigation, attitude control, and mobility systems'
            },
            'data_collection': {
                'name': '📊 Data Collection',
                'equipment_ids': ['SMAP-PAY-001', 'MSL-SCI-001', 'MSL-ENV-001'],
                'description': 'Scientific instruments and data collection systems'
            },
            'thermal_management': {
                'name': '🌡️ Thermal Management',
                'equipment_ids': ['SMAP-THM-001'],
                'description': 'Thermal control and heat management systems'
            }
        }

    def _create_quick_actions(self) -> Dict[str, Dict[str, Any]]:
        """Create quick action presets"""
        return {
            'health_check': {
                'name': '🏥 System Health Check',
                'description': 'Quick overview of all systems health',
                'action': 'apply_preset',
                'preset_id': 'mission_critical',
                'time_range': '1h',
                'chart_type': 'line'
            },
            'power_analysis': {
                'name': '⚡ Power Analysis',
                'description': 'Analyze power consumption and generation',
                'action': 'apply_preset',
                'preset_id': 'power_systems',
                'time_range': '24h',
                'chart_type': 'area'
            },
            'anomaly_hunt': {
                'name': '🔍 Anomaly Hunt',
                'description': 'Search for anomalies across all systems',
                'action': 'apply_preset',
                'preset_id': 'recent_anomalies',
                'time_range': '6h',
                'chart_type': 'scatter'
            },
            'comm_window': {
                'name': '📡 Communication Window',
                'description': 'Prepare for communication window',
                'action': 'apply_preset',
                'preset_id': 'communication',
                'time_range': '15m',
                'chart_type': 'line'
            }
        }

    def get_mission_presets(self, mission_phase: Optional[MissionPhase] = None,
                           spacecraft: Optional[str] = None) -> List[QuickSelectPreset]:
        """
        Get mission presets filtered by criteria

        Args:
            mission_phase: Filter by mission phase
            spacecraft: Filter by spacecraft ('smap', 'msl', 'both')

        Returns:
            List of matching presets
        """
        presets = list(self.mission_presets.values())

        # Filter by mission phase
        if mission_phase:
            presets = [p for p in presets if p.mission_phase == mission_phase]

        # Filter by spacecraft
        if spacecraft:
            presets = [p for p in presets
                      if spacecraft in p.spacecraft or 'both' in p.spacecraft]

        # Sort by priority
        presets.sort(key=lambda x: x.priority, reverse=True)

        return presets

    def get_preset_options(self, include_shortcuts: bool = True) -> List[Dict[str, Any]]:
        """
        Get preset options for dropdown

        Args:
            include_shortcuts: Include keyboard shortcuts in labels

        Returns:
            List of preset options for dropdown
        """
        options = []

        for preset in self.get_mission_presets():
            label = f"{preset.icon} {preset.name}"
            if include_shortcuts and preset.shortcut_key:
                label += f" ({preset.shortcut_key})"

            options.append({
                'label': label,
                'value': preset.id,
                'title': preset.description,
                'disabled': False
            })

        return options

    def apply_preset(self, preset_id: str, component_id: str = "quick_select") -> bool:
        """
        Apply a quick select preset

        Args:
            preset_id: ID of preset to apply
            component_id: Component applying the preset

        Returns:
            True if successful
        """
        try:
            if preset_id not in self.mission_presets:
                logger.error(f"Unknown preset: {preset_id}")
                return False

            preset = self.mission_presets[preset_id]

            # Handle dynamic presets
            if preset_id == 'recent_anomalies':
                preset.equipment_ids = self._get_recent_anomaly_equipment()

            # Apply equipment filter
            if self.filter_manager and preset.equipment_ids:
                self.filter_manager.set_equipment_filter(preset.equipment_ids, component_id)

            # Update shared state
            if self.shared_state:
                updates = {
                    'selections.equipment_id': preset.equipment_ids[0] if preset.equipment_ids else None,
                    'ui.current_preset': preset.id,
                    'ui.preset_applied_at': datetime.now().isoformat()
                }

                self.shared_state.update_multiple(updates, component_id)

            logger.info(f"Applied preset '{preset.name}' by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error applying preset {preset_id}: {e}")
            return False

    def apply_quick_action(self, action_id: str, component_id: str = "quick_select") -> bool:
        """
        Apply a quick action

        Args:
            action_id: ID of action to apply
            component_id: Component applying the action

        Returns:
            True if successful
        """
        try:
            if action_id not in self.quick_actions:
                logger.error(f"Unknown quick action: {action_id}")
                return False

            action = self.quick_actions[action_id]

            # Apply preset
            if action['action'] == 'apply_preset':
                success = self.apply_preset(action['preset_id'], component_id)

                if success and self.shared_state:
                    # Apply additional settings
                    updates = {}

                    if 'time_range' in action:
                        updates['filters.time_range'] = action['time_range']

                    if 'chart_type' in action:
                        updates['filters.chart_type'] = action['chart_type']

                    if updates:
                        self.shared_state.update_multiple(updates, component_id)

                return success

            return False

        except Exception as e:
            logger.error(f"Error applying quick action {action_id}: {e}")
            return False

    def get_equipment_summary(self, preset_id: str) -> Dict[str, Any]:
        """
        Get summary of equipment in preset

        Args:
            preset_id: Preset ID

        Returns:
            Equipment summary
        """
        try:
            if preset_id not in self.mission_presets:
                return {}

            preset = self.mission_presets[preset_id]

            # Count by spacecraft
            smap_count = len([eq for eq in preset.equipment_ids if eq.startswith('SMAP')])
            msl_count = len([eq for eq in preset.equipment_ids if eq.startswith('MSL')])

            # Estimate sensor count
            estimated_sensors = 0
            for equipment_id in preset.equipment_ids:
                spacecraft = equipment_id.split('-')[0].lower()
                if spacecraft in self.nasa_equipment_specs:
                    for subsystem in preset.subsystems:
                        if subsystem in self.nasa_equipment_specs[spacecraft]['subsystems']:
                            estimated_sensors += len(
                                self.nasa_equipment_specs[spacecraft]['subsystems'][subsystem]['sensors']
                            )

            return {
                'total_equipment': len(preset.equipment_ids),
                'smap_equipment': smap_count,
                'msl_equipment': msl_count,
                'estimated_sensors': estimated_sensors,
                'subsystems': preset.subsystems,
                'criticality': preset.criticality.value,
                'mission_phase': preset.mission_phase.value
            }

        except Exception as e:
            logger.error(f"Error getting equipment summary: {e}")
            return {}

    def _get_critical_equipment(self) -> List[str]:
        """Get all mission-critical equipment IDs"""
        critical_equipment = []

        for spacecraft, specs in self.nasa_equipment_specs.items():
            for subsystem, info in specs['subsystems'].items():
                if info['criticality'] == 'MISSION_CRITICAL':
                    critical_equipment.extend(info['equipment_ids'])

        return critical_equipment

    def _get_smap_equipment(self) -> List[str]:
        """Get all SMAP equipment IDs"""
        smap_equipment = []

        for subsystem, info in self.nasa_equipment_specs['smap']['subsystems'].items():
            smap_equipment.extend(info['equipment_ids'])

        return smap_equipment

    def _get_msl_equipment(self) -> List[str]:
        """Get all MSL equipment IDs"""
        msl_equipment = []

        for subsystem, info in self.nasa_equipment_specs['msl']['subsystems'].items():
            msl_equipment.extend(info['equipment_ids'])

        return msl_equipment

    def _get_power_equipment(self) -> List[str]:
        """Get power system equipment IDs"""
        power_equipment = []

        for spacecraft, specs in self.nasa_equipment_specs.items():
            if 'POWER' in specs['subsystems']:
                power_equipment.extend(specs['subsystems']['POWER']['equipment_ids'])

        return power_equipment

    def _get_communication_equipment(self) -> List[str]:
        """Get communication equipment IDs"""
        comm_equipment = []

        for spacecraft, specs in self.nasa_equipment_specs.items():
            if 'COMMUNICATION' in specs['subsystems']:
                comm_equipment.extend(specs['subsystems']['COMMUNICATION']['equipment_ids'])

        return comm_equipment

    def _get_scientific_equipment(self) -> List[str]:
        """Get scientific instrument equipment IDs"""
        sci_equipment = []

        # SMAP payload
        if 'PAYLOAD' in self.nasa_equipment_specs['smap']['subsystems']:
            sci_equipment.extend(self.nasa_equipment_specs['smap']['subsystems']['PAYLOAD']['equipment_ids'])

        # MSL scientific instruments
        if 'SCIENTIFIC' in self.nasa_equipment_specs['msl']['subsystems']:
            sci_equipment.extend(self.nasa_equipment_specs['msl']['subsystems']['SCIENTIFIC']['equipment_ids'])

        # MSL environmental monitoring
        if 'ENVIRONMENTAL' in self.nasa_equipment_specs['msl']['subsystems']:
            sci_equipment.extend(self.nasa_equipment_specs['msl']['subsystems']['ENVIRONMENTAL']['equipment_ids'])

        return sci_equipment

    def _get_emergency_equipment(self) -> List[str]:
        """Get emergency response equipment IDs"""
        # Return critical systems that are essential for emergency response
        return self._get_critical_equipment()

    def _get_recent_anomaly_equipment(self) -> List[str]:
        """Get equipment with recent anomalies (dynamic)"""
        # This would query the anomaly detection system for recent anomalies
        # For now, return a subset as example
        return ['SMAP-PWR-001', 'MSL-MOB-001']


# Global instance for use across the dashboard
quick_select_manager = QuickSelectManager()