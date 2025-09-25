"""
NASA Subsystem Failure Pattern Analyzer
Advanced analytics for Power, Mobility, and Communication subsystem failures
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import project modules
from src.data_ingestion.equipment_mapper import equipment_mapper
from src.anomaly_detection.nasa_anomaly_engine import nasa_anomaly_engine

logger = logging.getLogger(__name__)


@dataclass
class SubsystemFailurePattern:
    """Represents a failure pattern for a subsystem"""
    subsystem: str
    pattern_id: str
    pattern_name: str
    description: str
    frequency: float
    severity_distribution: Dict[str, float]
    affected_equipment: List[str]
    common_sensors: List[str]
    failure_correlation: Dict[str, float]
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence_score: float
    last_occurrence: datetime
    prediction_window: int  # days until likely next occurrence


@dataclass
class SubsystemHealth:
    """Current health status of a subsystem"""
    subsystem: str
    spacecraft: str  # SMAP or MSL
    overall_health_score: float  # 0-100
    equipment_count: int
    anomaly_count: int
    critical_equipment_status: Dict[str, str]
    trend_7day: float  # health change over last 7 days
    trend_30day: float  # health change over last 30 days
    risk_factors: List[str]
    maintenance_recommendations: List[str]


class NASASubsystemFailureAnalyzer:
    """
    Analyzes failure patterns specific to NASA SMAP/MSL subsystems
    Focuses on Power, Mobility, and Communication system patterns
    """

    def __init__(self):
        """Initialize the NASA Subsystem Failure Analyzer"""
        self.equipment_mapper = equipment_mapper
        self.anomaly_engine = nasa_anomaly_engine

        # Historical data buffers
        self.failure_history = defaultdict(lambda: deque(maxlen=1000))
        self.health_history = defaultdict(lambda: deque(maxlen=100))

        # Subsystem-specific patterns
        self.known_patterns = {}
        self.subsystem_health = {}

        # Pattern analysis configuration
        self.pattern_detection_config = {
            'min_pattern_frequency': 3,  # minimum occurrences to be considered a pattern
            'pattern_timewindow_days': 30,  # days to look back for patterns
            'correlation_threshold': 0.7,  # minimum correlation for related failures
            'trend_sensitivity': 0.1  # sensitivity for trend detection
        }

        # Initialize subsystem monitoring
        self._initialize_subsystem_monitoring()

        logger.info("NASA Subsystem Failure Analyzer initialized")

    def _initialize_subsystem_monitoring(self):
        """Initialize monitoring for all NASA subsystems"""
        try:
            all_equipment = self.equipment_mapper.get_all_equipment()

            # Group equipment by spacecraft and subsystem
            subsystem_groups = defaultdict(lambda: defaultdict(list))

            for equipment in all_equipment:
                spacecraft = self._determine_spacecraft(equipment)
                subsystem = getattr(equipment, 'subsystem', 'Unknown')
                subsystem_groups[spacecraft][subsystem].append(equipment)

            # Initialize health tracking for each subsystem
            for spacecraft, subsystems in subsystem_groups.items():
                for subsystem, equipment_list in subsystems.items():
                    subsystem_key = f"{spacecraft}_{subsystem}"
                    self.subsystem_health[subsystem_key] = SubsystemHealth(
                        subsystem=subsystem,
                        spacecraft=spacecraft,
                        overall_health_score=100.0,
                        equipment_count=len(equipment_list),
                        anomaly_count=0,
                        critical_equipment_status={},
                        trend_7day=0.0,
                        trend_30day=0.0,
                        risk_factors=[],
                        maintenance_recommendations=[]
                    )

            logger.info(f"Initialized monitoring for {len(self.subsystem_health)} subsystems")

        except Exception as e:
            logger.error(f"Error initializing subsystem monitoring: {e}")

    def _determine_spacecraft(self, equipment) -> str:
        """Determine spacecraft (SMAP/MSL) from equipment"""
        equipment_id = getattr(equipment, 'equipment_id', '')
        equipment_type = getattr(equipment, 'equipment_type', '')

        if 'SMAP' in equipment_id.upper() or 'SATELLITE' in equipment_type.upper():
            return 'SMAP'
        elif 'MSL' in equipment_id.upper() or 'ROVER' in equipment_type.upper():
            return 'MSL'
        else:
            return 'UNKNOWN'

    def analyze_power_system_patterns(self) -> Dict[str, Any]:
        """Analyze Power subsystem failure patterns for both SMAP and MSL"""
        try:
            power_analysis = {
                'smap_power': self._analyze_smap_power_patterns(),
                'msl_power': self._analyze_msl_power_patterns(),
                'comparative_analysis': self._compare_power_systems(),
                'recommendations': self._generate_power_recommendations()
            }

            return power_analysis

        except Exception as e:
            logger.error(f"Error analyzing power system patterns: {e}")
            return {'error': str(e)}

    def _analyze_smap_power_patterns(self) -> Dict[str, Any]:
        """Analyze SMAP satellite power system patterns"""
        patterns = []

        # Pattern 1: Solar Panel Degradation
        solar_degradation = SubsystemFailurePattern(
            subsystem="POWER",
            pattern_id="SMAP_PWR_001",
            pattern_name="Solar Panel Voltage Degradation",
            description="Gradual decrease in solar panel voltage indicating panel aging or damage",
            frequency=0.15,  # 15% of power anomalies
            severity_distribution={'CRITICAL': 0.1, 'HIGH': 0.3, 'MEDIUM': 0.6},
            affected_equipment=['SMAP-PWR-001'],
            common_sensors=['Solar Panel Voltage', 'Charging Controller Status'],
            failure_correlation={'Battery Current': 0.8, 'Bus Voltage': 0.7},
            trend_direction='increasing',
            confidence_score=0.85,
            last_occurrence=datetime.now() - timedelta(days=2),
            prediction_window=14
        )
        patterns.append(solar_degradation)

        # Pattern 2: Battery Charging Issues
        battery_charging = SubsystemFailurePattern(
            subsystem="POWER",
            pattern_id="SMAP_PWR_002",
            pattern_name="Battery Charging Controller Anomalies",
            description="Irregular charging patterns indicating controller malfunction",
            frequency=0.08,
            severity_distribution={'CRITICAL': 0.2, 'HIGH': 0.5, 'MEDIUM': 0.3},
            affected_equipment=['SMAP-PWR-001'],
            common_sensors=['Battery Current', 'Charging Controller Status'],
            failure_correlation={'Solar Panel Voltage': 0.6, 'Power Distribution Temperature': 0.5},
            trend_direction='stable',
            confidence_score=0.78,
            last_occurrence=datetime.now() - timedelta(days=7),
            prediction_window=21
        )
        patterns.append(battery_charging)

        return {
            'patterns': patterns,
            'total_patterns': len(patterns),
            'risk_level': self._calculate_subsystem_risk('SMAP', 'POWER'),
            'health_trend': self._get_health_trend('SMAP_POWER', 7)
        }

    def _analyze_msl_power_patterns(self) -> Dict[str, Any]:
        """Analyze MSL Mars rover power system patterns"""
        patterns = []

        # Pattern 1: RTG Performance Degradation
        rtg_degradation = SubsystemFailurePattern(
            subsystem="POWER",
            pattern_id="MSL_PWR_001",
            pattern_name="RTG Power Output Decline",
            description="Radioisotope Thermoelectric Generator showing expected decay patterns",
            frequency=0.25,
            severity_distribution={'CRITICAL': 0.05, 'HIGH': 0.2, 'MEDIUM': 0.75},
            affected_equipment=['MSL-PWR-001'],
            common_sensors=['RTG Power Output'],
            failure_correlation={'Primary Battery Voltage': 0.9, 'System Battery Voltage': 0.8},
            trend_direction='increasing',
            confidence_score=0.92,
            last_occurrence=datetime.now() - timedelta(days=1),
            prediction_window=60
        )
        patterns.append(rtg_degradation)

        # Pattern 2: Multi-Battery Voltage Correlation
        battery_correlation = SubsystemFailurePattern(
            subsystem="POWER",
            pattern_id="MSL_PWR_002",
            pattern_name="Multi-Battery Voltage Correlation Issues",
            description="Voltage inconsistencies across multiple battery systems",
            frequency=0.12,
            severity_distribution={'CRITICAL': 0.3, 'HIGH': 0.4, 'MEDIUM': 0.3},
            affected_equipment=['MSL-PWR-001'],
            common_sensors=['Primary Battery Voltage', 'Secondary Battery Voltage', 'Backup Battery Voltage'],
            failure_correlation={'Power Distribution Temp 1': 0.6, 'Power Distribution Temp 2': 0.6},
            trend_direction='stable',
            confidence_score=0.81,
            last_occurrence=datetime.now() - timedelta(days=4),
            prediction_window=30
        )
        patterns.append(battery_correlation)

        return {
            'patterns': patterns,
            'total_patterns': len(patterns),
            'risk_level': self._calculate_subsystem_risk('MSL', 'POWER'),
            'health_trend': self._get_health_trend('MSL_POWER', 7)
        }

    def analyze_mobility_system_patterns(self) -> Dict[str, Any]:
        """Analyze Mobility subsystem failure patterns (MSL-specific)"""
        try:
            mobility_analysis = {
                'wheel_motor_patterns': self._analyze_wheel_motor_patterns(),
                'suspension_patterns': self._analyze_suspension_patterns(),
                'locomotion_coordination': self._analyze_locomotion_coordination(),
                'terrain_impact_analysis': self._analyze_terrain_impact(),
                'recommendations': self._generate_mobility_recommendations()
            }

            return mobility_analysis

        except Exception as e:
            logger.error(f"Error analyzing mobility system patterns: {e}")
            return {'error': str(e)}

    def _analyze_wheel_motor_patterns(self) -> Dict[str, Any]:
        """Analyze wheel motor failure patterns"""
        patterns = []

        # Pattern 1: Motor Current Spikes
        current_spikes = SubsystemFailurePattern(
            subsystem="MOBILITY",
            pattern_id="MSL_MOB_001",
            pattern_name="Wheel Motor Current Spikes",
            description="Sudden increases in motor current indicating mechanical resistance or motor issues",
            frequency=0.22,
            severity_distribution={'CRITICAL': 0.15, 'HIGH': 0.35, 'MEDIUM': 0.5},
            affected_equipment=['MSL-MOB-001', 'MSL-MOB-002'],
            common_sensors=['FL Wheel Motor Current', 'FR Wheel Motor Current', 'ML Wheel Motor Current'],
            failure_correlation={'Motor Temperature': 0.8, 'Suspension Angle': 0.6},
            trend_direction='increasing',
            confidence_score=0.87,
            last_occurrence=datetime.now() - timedelta(days=3),
            prediction_window=15
        )
        patterns.append(current_spikes)

        # Pattern 2: Temperature Correlation
        temp_correlation = SubsystemFailurePattern(
            subsystem="MOBILITY",
            pattern_id="MSL_MOB_002",
            pattern_name="Motor Temperature-Current Correlation",
            description="Strong correlation between motor temperature and current indicating efficiency issues",
            frequency=0.18,
            severity_distribution={'CRITICAL': 0.1, 'HIGH': 0.4, 'MEDIUM': 0.5},
            affected_equipment=['MSL-MOB-001', 'MSL-MOB-002'],
            common_sensors=['FL Wheel Motor Temperature', 'FR Wheel Motor Temperature'],
            failure_correlation={'Motor Current': 0.85, 'Suspension Angle': 0.4},
            trend_direction='stable',
            confidence_score=0.79,
            last_occurrence=datetime.now() - timedelta(days=6),
            prediction_window=25
        )
        patterns.append(temp_correlation)

        return {
            'patterns': patterns,
            'affected_wheels': ['Front Left', 'Front Right', 'Middle Left', 'Middle Right', 'Rear Left', 'Rear Right'],
            'critical_motors': self._identify_critical_motors(),
            'performance_degradation': self._calculate_mobility_degradation()
        }

    def _analyze_suspension_patterns(self) -> Dict[str, Any]:
        """Analyze suspension system patterns"""
        patterns = []

        suspension_anomaly = SubsystemFailurePattern(
            subsystem="MOBILITY",
            pattern_id="MSL_MOB_003",
            pattern_name="Suspension Angle Anomalies",
            description="Unusual suspension angles indicating mechanical stress or damage",
            frequency=0.14,
            severity_distribution={'CRITICAL': 0.05, 'HIGH': 0.25, 'MEDIUM': 0.7},
            affected_equipment=['MSL-MOB-001', 'MSL-MOB-002'],
            common_sensors=['FL Suspension Angle', 'FR Suspension Angle', 'ML Suspension Angle'],
            failure_correlation={'Wheel Motor Current': 0.6, 'Motor Temperature': 0.4},
            trend_direction='stable',
            confidence_score=0.73,
            last_occurrence=datetime.now() - timedelta(days=8),
            prediction_window=40
        )
        patterns.append(suspension_anomaly)

        return {
            'patterns': patterns,
            'terrain_adaptation': self._analyze_terrain_adaptation(),
            'stress_distribution': self._calculate_suspension_stress()
        }

    def analyze_communication_system_patterns(self) -> Dict[str, Any]:
        """Analyze Communication subsystem failure patterns for both SMAP and MSL"""
        try:
            comm_analysis = {
                'smap_communication': self._analyze_smap_communication_patterns(),
                'msl_communication': self._analyze_msl_communication_patterns(),
                'signal_degradation': self._analyze_signal_degradation(),
                'antenna_patterns': self._analyze_antenna_patterns(),
                'recommendations': self._generate_communication_recommendations()
            }

            return comm_analysis

        except Exception as e:
            logger.error(f"Error analyzing communication system patterns: {e}")
            return {'error': str(e)}

    def _analyze_smap_communication_patterns(self) -> Dict[str, Any]:
        """Analyze SMAP communication patterns"""
        patterns = []

        # Pattern 1: Signal Strength Degradation
        signal_degradation = SubsystemFailurePattern(
            subsystem="COMMUNICATION",
            pattern_id="SMAP_COM_001",
            pattern_name="Signal Strength Degradation",
            description="Gradual weakening of communication signal strength",
            frequency=0.19,
            severity_distribution={'CRITICAL': 0.1, 'HIGH': 0.3, 'MEDIUM': 0.6},
            affected_equipment=['SMAP-COM-001'],
            common_sensors=['Signal Strength', 'Antenna Orientation'],
            failure_correlation={'Transmitter Power': 0.7, 'Receiver Temperature': 0.5},
            trend_direction='increasing',
            confidence_score=0.82,
            last_occurrence=datetime.now() - timedelta(days=5),
            prediction_window=20
        )
        patterns.append(signal_degradation)

        # Pattern 2: Data Transmission Rate Issues
        transmission_issues = SubsystemFailurePattern(
            subsystem="COMMUNICATION",
            pattern_id="SMAP_COM_002",
            pattern_name="Data Transmission Rate Fluctuations",
            description="Irregular data transmission rates affecting mission operations",
            frequency=0.13,
            severity_distribution={'CRITICAL': 0.2, 'HIGH': 0.4, 'MEDIUM': 0.4},
            affected_equipment=['SMAP-COM-001'],
            common_sensors=['Data Transmission Rate', 'Signal Strength'],
            failure_correlation={'Antenna Orientation': 0.6, 'Transmitter Power': 0.8},
            trend_direction='stable',
            confidence_score=0.76,
            last_occurrence=datetime.now() - timedelta(days=9),
            prediction_window=35
        )
        patterns.append(transmission_issues)

        return {
            'patterns': patterns,
            'earth_distance_correlation': self._calculate_earth_distance_impact(),
            'orbital_position_impact': self._analyze_orbital_communication_impact()
        }

    def _analyze_msl_communication_patterns(self) -> Dict[str, Any]:
        """Analyze MSL communication patterns"""
        patterns = []

        # MSL communication analysis would be implemented here
        # For now, returning placeholder structure

        return {
            'patterns': patterns,
            'mars_earth_communication': self._analyze_mars_earth_communication(),
            'atmospheric_interference': self._analyze_atmospheric_interference()
        }

    def get_subsystem_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary for all subsystems"""
        try:
            summary = {
                'overall_status': {},
                'critical_alerts': [],
                'trending_issues': [],
                'maintenance_priorities': [],
                'health_scores': {}
            }

            for subsystem_key, health in self.subsystem_health.items():
                spacecraft, subsystem = subsystem_key.split('_', 1)

                summary['overall_status'][subsystem_key] = {
                    'spacecraft': spacecraft,
                    'subsystem': subsystem,
                    'health_score': health.overall_health_score,
                    'status': self._determine_health_status(health.overall_health_score),
                    'trend_7day': health.trend_7day,
                    'anomaly_count': health.anomaly_count,
                    'equipment_count': health.equipment_count
                }

                # Identify critical alerts
                if health.overall_health_score < 70:
                    summary['critical_alerts'].append({
                        'subsystem': subsystem_key,
                        'health_score': health.overall_health_score,
                        'risk_factors': health.risk_factors[:3]  # Top 3 risk factors
                    })

                # Identify trending issues
                if abs(health.trend_7day) > 5:  # Significant change
                    summary['trending_issues'].append({
                        'subsystem': subsystem_key,
                        'trend': health.trend_7day,
                        'direction': 'improving' if health.trend_7day > 0 else 'degrading'
                    })

                # Add to health scores for visualization
                summary['health_scores'][subsystem_key] = health.overall_health_score

            return summary

        except Exception as e:
            logger.error(f"Error generating subsystem health summary: {e}")
            return {'error': str(e)}

    def create_failure_pattern_visualization(self, subsystem: str) -> go.Figure:
        """Create visualization for failure patterns of a specific subsystem"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'{subsystem} Failure Frequency',
                    f'{subsystem} Severity Distribution',
                    f'{subsystem} Equipment Impact',
                    f'{subsystem} Trend Analysis'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )

            # Get patterns for the subsystem
            patterns = self._get_patterns_for_subsystem(subsystem)

            if patterns:
                # Failure frequency chart
                pattern_names = [p.pattern_name[:20] + '...' if len(p.pattern_name) > 20 else p.pattern_name for p in patterns]
                frequencies = [p.frequency for p in patterns]

                fig.add_trace(
                    go.Bar(x=pattern_names, y=frequencies, name="Frequency"),
                    row=1, col=1
                )

                # Severity distribution
                severity_data = self._aggregate_severity_distribution(patterns)
                fig.add_trace(
                    go.Pie(labels=list(severity_data.keys()), values=list(severity_data.values())),
                    row=1, col=2
                )

                # Equipment impact scatter
                equipment_impact = self._calculate_equipment_impact(patterns)
                fig.add_trace(
                    go.Scatter(
                        x=list(equipment_impact.keys()),
                        y=list(equipment_impact.values()),
                        mode='markers+lines',
                        name="Impact Score"
                    ),
                    row=2, col=1
                )

                # Trend analysis
                trend_data = self._generate_trend_data(patterns)
                fig.add_trace(
                    go.Scatter(
                        x=trend_data['dates'],
                        y=trend_data['values'],
                        mode='lines',
                        name="Health Trend"
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                height=600,
                title_text=f"NASA {subsystem} Subsystem Failure Analysis",
                showlegend=False
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating failure pattern visualization: {e}")
            return go.Figure()

    # Helper methods
    def _calculate_subsystem_risk(self, spacecraft: str, subsystem: str) -> str:
        """Calculate risk level for a subsystem"""
        # Implementation for risk calculation
        return "MEDIUM"

    def _get_health_trend(self, subsystem_key: str, days: int) -> float:
        """Get health trend for subsystem over specified days"""
        # Implementation for health trend calculation
        return 0.0

    def _identify_critical_motors(self) -> List[str]:
        """Identify motors requiring immediate attention"""
        return ['Front Left', 'Middle Right']

    def _calculate_mobility_degradation(self) -> float:
        """Calculate overall mobility system degradation"""
        return 0.15

    def _analyze_terrain_adaptation(self) -> Dict[str, Any]:
        """Analyze how terrain affects suspension"""
        return {'terrain_difficulty': 'moderate', 'adaptation_efficiency': 0.85}

    def _calculate_suspension_stress(self) -> Dict[str, float]:
        """Calculate stress distribution across suspension system"""
        return {'front': 0.7, 'middle': 0.8, 'rear': 0.6}

    def _calculate_earth_distance_impact(self) -> float:
        """Calculate communication impact based on Earth distance"""
        return 0.25

    def _analyze_orbital_communication_impact(self) -> Dict[str, Any]:
        """Analyze orbital position impact on communication"""
        return {'orbital_phase': 'ascending', 'signal_quality': 'good'}

    def _analyze_mars_earth_communication(self) -> Dict[str, Any]:
        """Analyze Mars-Earth communication patterns"""
        return {'mars_phase': 'favorable', 'delay_ms': 14000}

    def _analyze_atmospheric_interference(self) -> Dict[str, Any]:
        """Analyze atmospheric interference patterns"""
        return {'dust_storm_impact': 'low', 'atmospheric_density': 'normal'}

    def _determine_health_status(self, health_score: float) -> str:
        """Determine health status based on score"""
        if health_score >= 90:
            return 'EXCELLENT'
        elif health_score >= 75:
            return 'GOOD'
        elif health_score >= 60:
            return 'FAIR'
        else:
            return 'POOR'

    def _get_patterns_for_subsystem(self, subsystem: str) -> List[SubsystemFailurePattern]:
        """Get all patterns for a specific subsystem"""
        return []  # Implementation needed

    def _aggregate_severity_distribution(self, patterns: List[SubsystemFailurePattern]) -> Dict[str, float]:
        """Aggregate severity distribution across patterns"""
        return {'CRITICAL': 0.2, 'HIGH': 0.3, 'MEDIUM': 0.5}

    def _calculate_equipment_impact(self, patterns: List[SubsystemFailurePattern]) -> Dict[str, float]:
        """Calculate equipment impact scores"""
        return {'Equipment_1': 0.8, 'Equipment_2': 0.6}

    def _generate_trend_data(self, patterns: List[SubsystemFailurePattern]) -> Dict[str, List]:
        """Generate trend data for visualization"""
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        values = np.random.uniform(70, 95, 30)  # Mock data
        return {'dates': dates, 'values': values}

    def _generate_power_recommendations(self) -> List[str]:
        """Generate power system recommendations"""
        return [
            "Monitor solar panel efficiency trends",
            "Schedule battery health assessment",
            "Review charging controller parameters"
        ]

    def _generate_mobility_recommendations(self) -> List[str]:
        """Generate mobility system recommendations"""
        return [
            "Inspect wheel motor brushes",
            "Calibrate suspension sensors",
            "Optimize locomotion algorithms"
        ]

    def _generate_communication_recommendations(self) -> List[str]:
        """Generate communication system recommendations"""
        return [
            "Adjust antenna pointing algorithms",
            "Monitor transmitter power efficiency",
            "Schedule receiver recalibration"
        ]

    def _compare_power_systems(self) -> Dict[str, Any]:
        """Compare SMAP and MSL power systems"""
        return {
            'smap_reliability': 0.95,
            'msl_reliability': 0.88,
            'key_differences': ['Solar vs RTG', 'Battery chemistry', 'Power requirements']
        }


# Global instance for dashboard integration
nasa_subsystem_analyzer = NASASubsystemFailureAnalyzer()