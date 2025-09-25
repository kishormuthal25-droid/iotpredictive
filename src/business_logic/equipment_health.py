"""
Equipment Health Scoring System for IoT Predictive Maintenance

This module provides comprehensive health scoring for equipment subsystems,
individual sensors, and overall system health. Implements multi-dimensional
health metrics with predictive degradation modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import math

# Local imports
from .failure_classification import FailureClassificationEngine, FailureMode, Severity, SensorMapping
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Equipment health status levels"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-90%
    FAIR = "fair"               # 60-75%
    POOR = "poor"               # 40-60%
    CRITICAL = "critical"       # 0-40%


@dataclass
class HealthMetrics:
    """Individual health metric components"""
    # Core metrics (0-1 scale)
    performance_score: float = 1.0      # Performance vs baseline
    reliability_score: float = 1.0      # Failure rate history
    condition_score: float = 1.0        # Current sensor readings
    trend_score: float = 1.0            # Degradation trends

    # Auxiliary metrics
    maintenance_score: float = 1.0      # Maintenance history impact
    environmental_score: float = 1.0    # Environmental stress
    utilization_score: float = 1.0     # Usage intensity impact

    # Metadata
    data_quality: float = 1.0           # Data completeness/quality
    confidence: float = 1.0             # Overall confidence in scoring


@dataclass
class SensorHealth:
    """Health information for individual sensor"""
    sensor_id: str
    health_score: float                 # Overall health (0-100)
    status: HealthStatus
    metrics: HealthMetrics

    # Anomaly information
    current_anomaly_score: float = 0.0
    anomaly_trend: float = 0.0          # Positive = getting worse

    # Degradation modeling
    degradation_rate: float = 0.0       # Rate of health decline
    estimated_rul: Optional[float] = None  # Remaining useful life (hours)

    # Recent history
    health_history: List[float] = field(default_factory=list)
    last_maintenance: Optional[datetime] = None

    # Recommendations
    maintenance_urgency: int = 0         # 0-5 scale
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class SubsystemHealth:
    """Health information for equipment subsystem"""
    subsystem_name: str
    health_score: float                 # Overall subsystem health (0-100)
    status: HealthStatus

    # Component breakdown
    sensor_healths: Dict[str, SensorHealth]
    critical_sensors: List[str]         # Sensors below threshold

    # Impact assessment
    operational_impact: float = 0.0     # Impact on operations (0-1)
    safety_impact: float = 0.0          # Safety implications (0-1)
    mission_criticality: float = 0.0    # Mission criticality (0-1)

    # Maintenance planning
    next_maintenance_due: Optional[datetime] = None
    maintenance_cost_estimate: float = 0.0
    downtime_estimate_hours: float = 0.0


@dataclass
class SystemHealth:
    """Overall system health assessment"""
    overall_health_score: float         # Weighted system health (0-100)
    status: HealthStatus

    # Subsystem breakdown
    subsystem_healths: Dict[str, SubsystemHealth]

    # System-wide metrics
    availability: float = 1.0           # System availability (0-1)
    performance_efficiency: float = 1.0 # Performance vs rated capacity
    mtbf: Optional[float] = None        # Mean time between failures
    mttr: Optional[float] = None        # Mean time to repair

    # Risk assessment
    failure_risk_30d: float = 0.0       # 30-day failure probability
    cascade_failure_risk: float = 0.0   # Risk of cascading failures

    # Economic indicators
    total_maintenance_cost_30d: float = 0.0
    potential_loss_avoidance: float = 0.0


class EquipmentHealthScorer:
    """
    Comprehensive equipment health scoring system.

    Features:
    - Multi-dimensional health assessment
    - Predictive degradation modeling
    - Subsystem impact analysis
    - Risk-based maintenance prioritization
    - Economic impact quantification
    """

    def __init__(self, failure_classifier: FailureClassificationEngine = None):
        self.config = get_config()
        self.failure_classifier = failure_classifier or FailureClassificationEngine()

        # Health calculation parameters
        self.weights = self._load_health_weights()
        self.thresholds = self._load_health_thresholds()
        self.baselines = self._load_baseline_metrics()

        # Historical data storage
        self.health_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.maintenance_history: Dict[str, List[datetime]] = {}

        logger.info("Initialized EquipmentHealthScorer")

    def _load_health_weights(self) -> Dict[str, Dict[str, float]]:
        """Load weighting factors for health calculations"""
        return {
            'sensor_level': {
                'performance_score': 0.25,
                'reliability_score': 0.20,
                'condition_score': 0.25,
                'trend_score': 0.15,
                'maintenance_score': 0.10,
                'environmental_score': 0.05
            },
            'subsystem_level': {
                'sensor_average': 0.60,
                'critical_sensor_penalty': 0.25,
                'operational_impact': 0.10,
                'safety_impact': 0.05
            },
            'system_level': {
                'subsystem_weighted': 0.70,
                'availability': 0.15,
                'performance_efficiency': 0.10,
                'cascade_risk_penalty': 0.05
            },
            'subsystem_criticality': {
                # SMAP subsystems
                'soil_monitoring': 0.30,
                'radar_system': 0.25,
                'power_system': 0.20,
                'thermal_monitoring': 0.10,
                'communication': 0.10,
                'navigation': 0.05,

                # MSL subsystems
                'mobility': 0.25,
                'robotic_arm': 0.20,
                'science_instruments': 0.20,
                'attitude_control': 0.15,
                'mechanical_system': 0.10,
                'environmental': 0.05
            }
        }

    def _load_health_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load health status thresholds"""
        return {
            'health_status': {
                'excellent': 90.0,
                'good': 75.0,
                'fair': 60.0,
                'poor': 40.0,
                'critical': 0.0
            },
            'anomaly_impact': {
                'low': 0.5,      # Minimal health impact
                'medium': 0.7,   # Moderate health impact
                'high': 0.85,    # Significant health impact
                'critical': 0.95 # Severe health impact
            },
            'maintenance_urgency': {
                'routine': 0,    # Normal schedule
                'elevated': 1,   # Monitor closely
                'high': 2,       # Schedule soon
                'urgent': 3,     # Schedule immediately
                'emergency': 4,  # Stop operation
                'critical': 5    # Emergency response
            }
        }

    def _load_baseline_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load baseline performance metrics for comparison"""
        return {
            # SMAP baseline metrics
            'soil_monitoring': {
                'data_collection_rate': 0.95,
                'measurement_accuracy': 0.98,
                'response_time_ms': 100
            },
            'radar_system': {
                'signal_strength_dbm': -30,
                'noise_floor_dbm': -70,
                'detection_accuracy': 0.92
            },
            'power_system': {
                'efficiency': 0.85,
                'voltage_stability': 0.98,
                'power_quality': 0.95
            },

            # MSL baseline metrics
            'mobility': {
                'power_efficiency': 0.80,
                'navigation_accuracy': 0.95,
                'terrain_adaptability': 0.90
            },
            'robotic_arm': {
                'positioning_accuracy': 0.98,
                'force_control': 0.95,
                'operation_success_rate': 0.97
            },
            'science_instruments': {
                'data_quality': 0.96,
                'measurement_precision': 0.94,
                'instrument_availability': 0.92
            }
        }

    def calculate_sensor_health(self, sensor_id: str, current_data: Dict[str, float],
                              anomaly_scores: Dict[str, float],
                              time_window_data: Optional[Dict[str, np.ndarray]] = None) -> SensorHealth:
        """Calculate comprehensive health score for individual sensor"""

        if sensor_id not in self.failure_classifier.sensor_mappings:
            logger.warning(f"Unknown sensor ID: {sensor_id}")
            return self._create_default_sensor_health(sensor_id)

        mapping = self.failure_classifier.sensor_mappings[sensor_id]
        current_value = current_data.get(sensor_id, 0.0)
        anomaly_score = anomaly_scores.get(sensor_id, 0.0)

        # Calculate individual health metrics
        metrics = self._calculate_health_metrics(sensor_id, mapping, current_value,
                                               anomaly_score, time_window_data)

        # Calculate overall health score
        weights = self.weights['sensor_level']
        health_score = (
            metrics.performance_score * weights['performance_score'] +
            metrics.reliability_score * weights['reliability_score'] +
            metrics.condition_score * weights['condition_score'] +
            metrics.trend_score * weights['trend_score'] +
            metrics.maintenance_score * weights['maintenance_score'] +
            metrics.environmental_score * weights['environmental_score']
        ) * 100

        # Apply data quality penalty
        health_score *= metrics.data_quality

        # Determine health status
        status = self._determine_health_status(health_score)

        # Calculate degradation metrics
        degradation_rate = self._calculate_degradation_rate(sensor_id, health_score)
        estimated_rul = self._estimate_remaining_useful_life(sensor_id, health_score, degradation_rate)

        # Generate maintenance recommendations
        maintenance_urgency = self._calculate_maintenance_urgency(health_score, anomaly_score, degradation_rate)
        recommended_actions = self._generate_health_recommendations(sensor_id, mapping, metrics, anomaly_score)

        # Update health history
        self._update_health_history(sensor_id, health_score)

        return SensorHealth(
            sensor_id=sensor_id,
            health_score=health_score,
            status=status,
            metrics=metrics,
            current_anomaly_score=anomaly_score,
            anomaly_trend=self._calculate_anomaly_trend(sensor_id, anomaly_score),
            degradation_rate=degradation_rate,
            estimated_rul=estimated_rul,
            health_history=self._get_recent_health_history(sensor_id),
            last_maintenance=self._get_last_maintenance(sensor_id),
            maintenance_urgency=maintenance_urgency,
            recommended_actions=recommended_actions
        )

    def _calculate_health_metrics(self, sensor_id: str, mapping: SensorMapping,
                                current_value: float, anomaly_score: float,
                                time_window_data: Optional[Dict[str, np.ndarray]]) -> HealthMetrics:
        """Calculate individual health metric components"""

        # Performance Score: How well sensor performs vs baseline
        performance_score = self._calculate_performance_score(mapping, current_value)

        # Reliability Score: Based on failure history and current stability
        reliability_score = self._calculate_reliability_score(sensor_id, anomaly_score)

        # Condition Score: Current operational condition
        condition_score = self._calculate_condition_score(mapping, current_value, anomaly_score)

        # Trend Score: Degradation trends analysis
        trend_score = self._calculate_trend_score(sensor_id, time_window_data)

        # Maintenance Score: Impact of maintenance history
        maintenance_score = self._calculate_maintenance_score(sensor_id)

        # Environmental Score: Environmental stress factors
        environmental_score = self._calculate_environmental_score(mapping, current_value)

        # Utilization Score: Usage intensity impact
        utilization_score = self._calculate_utilization_score(sensor_id, time_window_data)

        # Data Quality: Completeness and consistency of sensor data
        data_quality = self._calculate_data_quality(sensor_id, time_window_data)

        # Overall confidence in health assessment
        confidence = self._calculate_health_confidence(sensor_id, data_quality, time_window_data)

        return HealthMetrics(
            performance_score=performance_score,
            reliability_score=reliability_score,
            condition_score=condition_score,
            trend_score=trend_score,
            maintenance_score=maintenance_score,
            environmental_score=environmental_score,
            utilization_score=utilization_score,
            data_quality=data_quality,
            confidence=confidence
        )

    def _calculate_performance_score(self, mapping: SensorMapping, current_value: float) -> float:
        """Calculate performance score based on operational parameters"""
        # Normalize current value within expected range
        normal_min, normal_max = mapping.normal_range
        range_size = normal_max - normal_min

        if range_size == 0:
            return 1.0

        # Calculate how close to optimal (center of range)
        optimal_value = (normal_min + normal_max) / 2
        deviation = abs(current_value - optimal_value) / (range_size / 2)

        # Performance degrades with deviation from optimal
        performance = max(0.0, 1.0 - deviation)

        # Apply equipment-specific performance modifiers
        if mapping.equipment_type == 'temperature_sensor':
            # Temperature sensors perform better when stable
            performance *= 0.95 if abs(current_value - optimal_value) < range_size * 0.1 else 0.85
        elif mapping.equipment_type in ['voltage_sensor', 'current_sensor']:
            # Electrical sensors need tight tolerance
            performance *= 0.98 if deviation < 0.1 else 0.80

        return max(0.0, min(1.0, performance))

    def _calculate_reliability_score(self, sensor_id: str, anomaly_score: float) -> float:
        """Calculate reliability score based on anomaly history"""
        base_reliability = 1.0 - anomaly_score

        # Factor in historical anomaly patterns
        if sensor_id in self.health_history:
            recent_health = [h for _, h in self.health_history[sensor_id][-10:]]
            if recent_health:
                stability = 1.0 - np.std(recent_health) / 100.0  # Penalize volatility
                base_reliability = (base_reliability + stability) / 2

        return max(0.0, min(1.0, base_reliability))

    def _calculate_condition_score(self, mapping: SensorMapping, current_value: float, anomaly_score: float) -> float:
        """Calculate current operational condition score"""
        # Base condition on threshold violations
        condition = 1.0

        # Check critical thresholds
        if current_value > mapping.critical_thresholds['high']:
            violation_severity = (current_value - mapping.critical_thresholds['high']) / mapping.critical_thresholds['high']
            condition -= min(0.5, violation_severity * 0.3)
        elif current_value < mapping.critical_thresholds['low'] and mapping.critical_thresholds['low'] > 0:
            violation_severity = (mapping.critical_thresholds['low'] - current_value) / mapping.critical_thresholds['low']
            condition -= min(0.5, violation_severity * 0.3)

        # Factor in anomaly score
        condition *= (1.0 - anomaly_score * 0.4)

        return max(0.0, min(1.0, condition))

    def _calculate_trend_score(self, sensor_id: str, time_window_data: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate trend score based on degradation patterns"""
        if not time_window_data or sensor_id not in time_window_data:
            return 0.8  # Neutral score when no trend data

        data = time_window_data[sensor_id]
        if len(data) < 5:
            return 0.8

        # Calculate trend using linear regression
        x = np.arange(len(data))
        trend_slope = np.polyfit(x, data, 1)[0]

        # Normalize trend relative to data range
        data_range = np.max(data) - np.min(data)
        if data_range == 0:
            return 1.0

        normalized_trend = abs(trend_slope) / data_range

        # Good trend score means stable or improving values
        trend_score = max(0.0, 1.0 - normalized_trend * 10)

        return min(1.0, trend_score)

    def _calculate_maintenance_score(self, sensor_id: str) -> float:
        """Calculate maintenance impact score"""
        if sensor_id not in self.maintenance_history:
            return 0.7  # Penalty for unknown maintenance history

        last_maintenance = self._get_last_maintenance(sensor_id)
        if not last_maintenance:
            return 0.7

        # Score based on time since last maintenance
        days_since_maintenance = (datetime.now() - last_maintenance).days

        # Optimal maintenance interval (equipment-dependent)
        optimal_interval = 90  # days

        if days_since_maintenance <= optimal_interval:
            return 1.0
        else:
            # Gradual degradation after optimal interval
            overdue_factor = (days_since_maintenance - optimal_interval) / optimal_interval
            return max(0.3, 1.0 - overdue_factor * 0.1)

    def _calculate_environmental_score(self, mapping: SensorMapping, current_value: float) -> float:
        """Calculate environmental stress impact score"""
        # Base environmental score
        env_score = 1.0

        # Temperature-based environmental stress
        if mapping.measurement_type == 'temperature':
            # Extreme temperatures reduce component life
            normal_min, normal_max = mapping.normal_range
            if current_value > normal_max * 1.1 or current_value < normal_min * 0.9:
                env_score *= 0.9

        # Equipment-specific environmental factors
        if 'outdoor' in mapping.location or 'external' in mapping.location:
            env_score *= 0.95  # Outdoor equipment faces more stress

        if mapping.subsystem == 'mobility':
            env_score *= 0.92  # Mobile equipment faces more stress

        return env_score

    def _calculate_utilization_score(self, sensor_id: str, time_window_data: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate utilization intensity impact"""
        if not time_window_data or sensor_id not in time_window_data:
            return 1.0

        data = time_window_data[sensor_id]
        if len(data) < 10:
            return 1.0

        # Calculate usage intensity (variation in readings)
        usage_intensity = np.std(data) / (np.mean(np.abs(data)) + 1e-6)

        # High usage intensity gradually reduces health
        utilization_score = max(0.8, 1.0 - usage_intensity * 0.1)

        return utilization_score

    def _calculate_data_quality(self, sensor_id: str, time_window_data: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate data quality score"""
        if not time_window_data or sensor_id not in time_window_data:
            return 0.7  # Reduced confidence without historical data

        data = time_window_data[sensor_id]

        # Check data completeness
        expected_points = 100  # Expected data points in window
        completeness = min(1.0, len(data) / expected_points)

        # Check for data consistency (no extreme outliers)
        if len(data) > 1:
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            outliers = np.sum((data < (q25 - 1.5 * iqr)) | (data > (q75 + 1.5 * iqr)))
            consistency = max(0.5, 1.0 - outliers / len(data))
        else:
            consistency = 1.0

        return (completeness + consistency) / 2

    def _calculate_health_confidence(self, sensor_id: str, data_quality: float,
                                   time_window_data: Optional[Dict[str, np.ndarray]]) -> float:
        """Calculate confidence in health assessment"""
        confidence = data_quality

        # Reduce confidence if limited historical data
        if not time_window_data or sensor_id not in time_window_data:
            confidence *= 0.8
        elif len(time_window_data[sensor_id]) < 20:
            confidence *= 0.9

        # Factor in maintenance history knowledge
        if sensor_id not in self.maintenance_history:
            confidence *= 0.9

        return confidence

    def calculate_subsystem_health(self, subsystem_name: str, sensor_healths: Dict[str, SensorHealth],
                                 operational_context: Optional[Dict] = None) -> SubsystemHealth:
        """Calculate health score for equipment subsystem"""

        if not sensor_healths:
            logger.warning(f"No sensor health data for subsystem: {subsystem_name}")
            return self._create_default_subsystem_health(subsystem_name)

        # Calculate base subsystem health from sensor average
        sensor_scores = [sh.health_score for sh in sensor_healths.values()]
        base_health = np.mean(sensor_scores)

        # Identify critical sensors (below threshold)
        critical_threshold = self.thresholds['health_status']['fair']
        critical_sensors = [sid for sid, sh in sensor_healths.items()
                          if sh.health_score < critical_threshold]

        # Apply penalties for critical sensors
        weights = self.weights['subsystem_level']
        critical_penalty = len(critical_sensors) / len(sensor_healths) * weights['critical_sensor_penalty']

        # Calculate impact scores
        operational_impact = self._calculate_operational_impact(subsystem_name, sensor_healths, operational_context)
        safety_impact = self._calculate_safety_impact(subsystem_name, critical_sensors)
        mission_criticality = self._calculate_mission_criticality(subsystem_name)

        # Final subsystem health score
        subsystem_health = (
            base_health * weights['sensor_average'] -
            critical_penalty * 100 +
            operational_impact * weights['operational_impact'] * 100 +
            safety_impact * weights['safety_impact'] * 100
        )

        subsystem_health = max(0.0, min(100.0, subsystem_health))

        # Determine health status
        status = self._determine_health_status(subsystem_health)

        # Calculate maintenance planning metrics
        next_maintenance, cost_estimate, downtime_estimate = self._calculate_maintenance_planning(
            subsystem_name, sensor_healths, critical_sensors
        )

        return SubsystemHealth(
            subsystem_name=subsystem_name,
            health_score=subsystem_health,
            status=status,
            sensor_healths=sensor_healths,
            critical_sensors=critical_sensors,
            operational_impact=operational_impact,
            safety_impact=safety_impact,
            mission_criticality=mission_criticality,
            next_maintenance_due=next_maintenance,
            maintenance_cost_estimate=cost_estimate,
            downtime_estimate_hours=downtime_estimate
        )

    def calculate_system_health(self, subsystem_healths: Dict[str, SubsystemHealth],
                              system_metrics: Optional[Dict] = None) -> SystemHealth:
        """Calculate overall system health"""

        if not subsystem_healths:
            logger.warning("No subsystem health data available")
            return self._create_default_system_health()

        # Calculate weighted subsystem health
        criticality_weights = self.weights['subsystem_criticality']
        total_weight = 0
        weighted_health = 0

        for subsystem_name, subsystem_health in subsystem_healths.items():
            weight = criticality_weights.get(subsystem_name, 0.1)  # Default weight for unknown subsystems
            weighted_health += subsystem_health.health_score * weight
            total_weight += weight

        if total_weight > 0:
            base_system_health = weighted_health / total_weight
        else:
            base_system_health = np.mean([sh.health_score for sh in subsystem_healths.values()])

        # Factor in system-level metrics
        system_weights = self.weights['system_level']
        availability = system_metrics.get('availability', 1.0) if system_metrics else 1.0
        performance_efficiency = system_metrics.get('performance_efficiency', 1.0) if system_metrics else 1.0

        # Calculate cascade failure risk
        cascade_risk = self._calculate_cascade_failure_risk(subsystem_healths)

        # Final system health score
        system_health = (
            base_system_health * system_weights['subsystem_weighted'] +
            availability * 100 * system_weights['availability'] +
            performance_efficiency * 100 * system_weights['performance_efficiency'] -
            cascade_risk * 100 * system_weights['cascade_risk_penalty']
        )

        system_health = max(0.0, min(100.0, system_health))

        # Determine system status
        status = self._determine_health_status(system_health)

        # Calculate system-wide reliability metrics
        mtbf, mttr = self._calculate_reliability_metrics(subsystem_healths)

        # Calculate risk assessments
        failure_risk_30d = self._calculate_30day_failure_risk(subsystem_healths)

        # Calculate economic indicators
        total_maintenance_cost, potential_loss_avoidance = self._calculate_economic_indicators(subsystem_healths)

        return SystemHealth(
            overall_health_score=system_health,
            status=status,
            subsystem_healths=subsystem_healths,
            availability=availability,
            performance_efficiency=performance_efficiency,
            mtbf=mtbf,
            mttr=mttr,
            failure_risk_30d=failure_risk_30d,
            cascade_failure_risk=cascade_risk,
            total_maintenance_cost_30d=total_maintenance_cost,
            potential_loss_avoidance=potential_loss_avoidance
        )

    # Helper methods for health calculations

    def _determine_health_status(self, health_score: float) -> HealthStatus:
        """Determine health status from score"""
        thresholds = self.thresholds['health_status']

        if health_score >= thresholds['excellent']:
            return HealthStatus.EXCELLENT
        elif health_score >= thresholds['good']:
            return HealthStatus.GOOD
        elif health_score >= thresholds['fair']:
            return HealthStatus.FAIR
        elif health_score >= thresholds['poor']:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _calculate_degradation_rate(self, sensor_id: str, current_health: float) -> float:
        """Calculate health degradation rate"""
        if sensor_id not in self.health_history:
            return 0.0

        history = self.health_history[sensor_id]
        if len(history) < 2:
            return 0.0

        # Calculate degradation over recent history
        recent_history = history[-10:]  # Last 10 measurements
        if len(recent_history) < 2:
            return 0.0

        times = [t.timestamp() for t, _ in recent_history]
        healths = [h for _, h in recent_history]

        # Linear regression to find degradation rate
        if len(times) > 1:
            slope = np.polyfit(times, healths, 1)[0]
            return max(0.0, -slope * 3600)  # Convert to degradation per hour

        return 0.0

    def _estimate_remaining_useful_life(self, sensor_id: str, current_health: float, degradation_rate: float) -> Optional[float]:
        """Estimate remaining useful life in hours"""
        if degradation_rate <= 0:
            return None

        # Estimate time to reach critical threshold
        critical_threshold = self.thresholds['health_status']['poor']
        health_margin = current_health - critical_threshold

        if health_margin <= 0:
            return 0.0  # Already below threshold

        rul_hours = health_margin / degradation_rate
        return min(8760.0, max(1.0, rul_hours))  # Between 1 hour and 1 year

    def _calculate_maintenance_urgency(self, health_score: float, anomaly_score: float, degradation_rate: float) -> int:
        """Calculate maintenance urgency level (0-5)"""
        urgency = 0

        # Health-based urgency
        if health_score < 40:
            urgency = 5  # Critical
        elif health_score < 60:
            urgency = 4  # Emergency
        elif health_score < 75:
            urgency = 3  # Urgent
        elif health_score < 85:
            urgency = 2  # High
        elif health_score < 95:
            urgency = 1  # Elevated

        # Factor in anomaly score
        if anomaly_score > 0.8:
            urgency = min(5, urgency + 1)

        # Factor in degradation rate
        if degradation_rate > 1.0:  # More than 1% per hour
            urgency = min(5, urgency + 1)

        return urgency

    def _generate_health_recommendations(self, sensor_id: str, mapping: SensorMapping,
                                       metrics: HealthMetrics, anomaly_score: float) -> List[str]:
        """Generate health-based maintenance recommendations"""
        recommendations = []

        # Performance-based recommendations
        if metrics.performance_score < 0.8:
            recommendations.append(f"Calibrate {mapping.equipment_type} for optimal performance")

        # Reliability-based recommendations
        if metrics.reliability_score < 0.7:
            recommendations.append(f"Investigate reliability issues in {sensor_id}")
            if anomaly_score > 0.8:
                recommendations.append("Consider sensor replacement due to recurring anomalies")

        # Condition-based recommendations
        if metrics.condition_score < 0.6:
            recommendations.append(f"Inspect {mapping.equipment_type} for physical damage")
            recommendations.append("Check sensor mounting and connections")

        # Trend-based recommendations
        if metrics.trend_score < 0.7:
            recommendations.append("Monitor sensor trends closely for accelerating degradation")

        # Maintenance-based recommendations
        if metrics.maintenance_score < 0.8:
            recommendations.append("Schedule preventive maintenance")

        # Equipment-specific recommendations
        if mapping.equipment_type == 'temperature_sensor' and metrics.environmental_score < 0.9:
            recommendations.append("Check thermal management system")
        elif 'motor' in mapping.equipment_type and metrics.condition_score < 0.8:
            recommendations.append("Check motor bearings and lubrication")
        elif mapping.equipment_type == 'comm_unit' and metrics.performance_score < 0.8:
            recommendations.append("Verify antenna alignment and signal path")

        return recommendations

    def _calculate_operational_impact(self, subsystem_name: str, sensor_healths: Dict[str, SensorHealth],
                                    operational_context: Optional[Dict]) -> float:
        """Calculate operational impact of subsystem health"""
        # Base impact on average sensor health
        avg_health = np.mean([sh.health_score for sh in sensor_healths.values()])
        base_impact = (100 - avg_health) / 100

        # Subsystem-specific impact multipliers
        impact_multipliers = {
            'power_system': 1.5,      # Power issues affect everything
            'communication': 1.3,     # Communication is critical for operations
            'mobility': 1.2,          # Mobility affects mission capability
            'radar_system': 1.1,      # Important for SMAP mission
            'science_instruments': 1.0, # Affects science return
            'thermal_system': 0.9,    # Important but has redundancy
        }

        multiplier = impact_multipliers.get(subsystem_name, 1.0)
        return min(1.0, base_impact * multiplier)

    def _calculate_safety_impact(self, subsystem_name: str, critical_sensors: List[str]) -> float:
        """Calculate safety impact of subsystem degradation"""
        # Safety-critical subsystems
        safety_criticality = {
            'power_system': 0.9,      # Power failures can be hazardous
            'thermal_system': 0.8,    # Overheating risks
            'mobility': 0.7,          # Mobility failures can strand equipment
            'communication': 0.6,     # Loss of control communication
            'robotic_arm': 0.5,       # Arm malfunctions
            'science_instruments': 0.2, # Lower safety impact
        }

        base_safety_impact = safety_criticality.get(subsystem_name, 0.3)

        # Increase impact based on number of critical sensors
        if critical_sensors:
            critical_factor = len(critical_sensors) / 10  # Assume max 10 sensors per subsystem
            return min(1.0, base_safety_impact + critical_factor * 0.2)

        return base_safety_impact * 0.5  # Reduce if no critical sensors

    def _calculate_mission_criticality(self, subsystem_name: str) -> float:
        """Calculate mission criticality of subsystem"""
        # Mission criticality weights (already defined in health weights)
        return self.weights['subsystem_criticality'].get(subsystem_name, 0.1)

    def _calculate_maintenance_planning(self, subsystem_name: str, sensor_healths: Dict[str, SensorHealth],
                                      critical_sensors: List[str]) -> Tuple[Optional[datetime], float, float]:
        """Calculate maintenance planning metrics"""
        # Find earliest maintenance requirement
        next_maintenance = None
        urgencies = [sh.maintenance_urgency for sh in sensor_healths.values()]
        max_urgency = max(urgencies) if urgencies else 0

        if max_urgency >= 4:  # Emergency or critical
            next_maintenance = datetime.now() + timedelta(hours=6)
        elif max_urgency >= 3:  # Urgent
            next_maintenance = datetime.now() + timedelta(days=1)
        elif max_urgency >= 2:  # High
            next_maintenance = datetime.now() + timedelta(days=7)
        elif max_urgency >= 1:  # Elevated
            next_maintenance = datetime.now() + timedelta(days=30)

        # Estimate maintenance cost
        base_costs = {
            'power_system': 5000,
            'robotic_arm': 8000,
            'mobility': 6000,
            'science_instruments': 10000,
            'communication': 3000,
            'thermal_system': 4000,
        }

        base_cost = base_costs.get(subsystem_name, 2000)
        urgency_multiplier = 1.0 + max_urgency * 0.2
        critical_multiplier = 1.0 + len(critical_sensors) * 0.15
        cost_estimate = base_cost * urgency_multiplier * critical_multiplier

        # Estimate downtime
        base_downtime = {
            'power_system': 8,        # hours
            'robotic_arm': 12,
            'mobility': 6,
            'science_instruments': 16,
            'communication': 4,
            'thermal_system': 10,
        }

        downtime = base_downtime.get(subsystem_name, 4)
        downtime_estimate = downtime * (1.0 + len(critical_sensors) * 0.1)

        return next_maintenance, cost_estimate, downtime_estimate

    def _calculate_cascade_failure_risk(self, subsystem_healths: Dict[str, SubsystemHealth]) -> float:
        """Calculate risk of cascading failures"""
        # Identify interdependencies
        dependencies = {
            'power_system': ['thermal_system', 'communication', 'science_instruments'],
            'thermal_system': ['power_system'],
            'communication': ['navigation'],
            'mobility': ['navigation', 'robotic_arm'],
        }

        cascade_risk = 0.0

        for subsystem_name, subsystem_health in subsystem_healths.items():
            if subsystem_health.health_score < 60:  # Failing subsystem
                dependent_systems = dependencies.get(subsystem_name, [])
                for dep_system in dependent_systems:
                    if dep_system in subsystem_healths:
                        # Risk increases if dependent system is also unhealthy
                        dep_health = subsystem_healths[dep_system].health_score
                        cascade_risk += (100 - subsystem_health.health_score) * (100 - dep_health) / 10000

        return min(1.0, cascade_risk)

    def _calculate_reliability_metrics(self, subsystem_healths: Dict[str, SubsystemHealth]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate MTBF and MTTR"""
        # Simplified calculation based on health scores
        # In practice, this would use detailed failure history

        health_scores = [sh.health_score for sh in subsystem_healths.values()]
        avg_health = np.mean(health_scores)

        # Estimate MTBF (hours) based on average health
        if avg_health > 80:
            mtbf = 8760  # 1 year
        elif avg_health > 60:
            mtbf = 4380  # 6 months
        elif avg_health > 40:
            mtbf = 2190  # 3 months
        else:
            mtbf = 720   # 1 month

        # Estimate MTTR (hours) based on worst subsystem
        min_health = min(health_scores)
        if min_health < 40:
            mttr = 24    # 1 day for critical issues
        elif min_health < 60:
            mttr = 12    # 12 hours
        elif min_health < 80:
            mttr = 6     # 6 hours
        else:
            mttr = 2     # 2 hours

        return mtbf, mttr

    def _calculate_30day_failure_risk(self, subsystem_healths: Dict[str, SubsystemHealth]) -> float:
        """Calculate 30-day failure probability"""
        # Risk based on current health and degradation trends
        risks = []

        for subsystem_health in subsystem_healths.values():
            health_score = subsystem_health.health_score

            # Base risk from health score
            if health_score < 40:
                base_risk = 0.8
            elif health_score < 60:
                base_risk = 0.4
            elif health_score < 80:
                base_risk = 0.1
            else:
                base_risk = 0.02

            # Increase risk for critical sensors
            if subsystem_health.critical_sensors:
                critical_factor = len(subsystem_health.critical_sensors) / 10
                base_risk *= (1.0 + critical_factor)

            risks.append(min(1.0, base_risk))

        # System failure probability (not all subsystems need to fail)
        return 1.0 - np.prod([1.0 - risk for risk in risks])

    def _calculate_economic_indicators(self, subsystem_healths: Dict[str, SubsystemHealth]) -> Tuple[float, float]:
        """Calculate economic impact indicators"""
        total_cost = sum(sh.maintenance_cost_estimate for sh in subsystem_healths.values())

        # Potential loss avoidance based on preventing failures
        loss_values = {
            'science_instruments': 100000,  # High value science missions
            'robotic_arm': 80000,
            'mobility': 60000,
            'power_system': 50000,
            'communication': 40000,
            'thermal_system': 30000,
        }

        potential_loss = 0
        for subsystem_name, subsystem_health in subsystem_healths.items():
            if subsystem_health.health_score < 60:
                loss_value = loss_values.get(subsystem_name, 20000)
                failure_prob = (60 - subsystem_health.health_score) / 60
                potential_loss += loss_value * failure_prob

        return total_cost, potential_loss

    # Default object creation methods

    def _create_default_sensor_health(self, sensor_id: str) -> SensorHealth:
        """Create default sensor health for unknown sensors"""
        return SensorHealth(
            sensor_id=sensor_id,
            health_score=50.0,
            status=HealthStatus.FAIR,
            metrics=HealthMetrics(),
            recommended_actions=["Verify sensor configuration and data availability"]
        )

    def _create_default_subsystem_health(self, subsystem_name: str) -> SubsystemHealth:
        """Create default subsystem health"""
        return SubsystemHealth(
            subsystem_name=subsystem_name,
            health_score=50.0,
            status=HealthStatus.FAIR,
            sensor_healths={},
            critical_sensors=[]
        )

    def _create_default_system_health(self) -> SystemHealth:
        """Create default system health"""
        return SystemHealth(
            overall_health_score=50.0,
            status=HealthStatus.FAIR,
            subsystem_healths={}
        )

    # History management methods

    def _update_health_history(self, sensor_id: str, health_score: float):
        """Update health history for sensor"""
        if sensor_id not in self.health_history:
            self.health_history[sensor_id] = []

        self.health_history[sensor_id].append((datetime.now(), health_score))

        # Keep only recent history (last 100 measurements)
        if len(self.health_history[sensor_id]) > 100:
            self.health_history[sensor_id] = self.health_history[sensor_id][-100:]

    def _get_recent_health_history(self, sensor_id: str, hours: int = 24) -> List[float]:
        """Get recent health history"""
        if sensor_id not in self.health_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_history = [(t, h) for t, h in self.health_history[sensor_id] if t > cutoff_time]

        return [h for _, h in recent_history]

    def _get_last_maintenance(self, sensor_id: str) -> Optional[datetime]:
        """Get last maintenance date for sensor"""
        if sensor_id not in self.maintenance_history:
            return None

        maintenance_dates = self.maintenance_history[sensor_id]
        return max(maintenance_dates) if maintenance_dates else None

    def _calculate_anomaly_trend(self, sensor_id: str, current_anomaly: float) -> float:
        """Calculate anomaly trend (positive = worsening)"""
        # This would typically use historical anomaly scores
        # For now, return neutral trend
        return 0.0

    def update_maintenance_history(self, sensor_id: str, maintenance_date: datetime = None):
        """Update maintenance history for sensor"""
        if maintenance_date is None:
            maintenance_date = datetime.now()

        if sensor_id not in self.maintenance_history:
            self.maintenance_history[sensor_id] = []

        self.maintenance_history[sensor_id].append(maintenance_date)

        # Keep only recent maintenance history (last 10 records)
        if len(self.maintenance_history[sensor_id]) > 10:
            self.maintenance_history[sensor_id] = self.maintenance_history[sensor_id][-10:]