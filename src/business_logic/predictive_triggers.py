"""
Predictive Maintenance Triggers System

This module implements intelligent trigger mechanisms for automated predictive
maintenance scheduling based on sensor anomalies, health scores, failure predictions,
and business rules. Integrates with equipment health scoring and cost-benefit analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import math

# Local imports
from .failure_classification import FailureClassificationEngine, FailureMode, Severity
from .equipment_health import EquipmentHealthScorer, SensorHealth, SubsystemHealth, HealthStatus
from ..utils.config import get_config


logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of maintenance triggers"""
    THRESHOLD_BASED = "threshold_based"           # Simple threshold violations
    TREND_BASED = "trend_based"                   # Degradation trend analysis
    PREDICTIVE_MODEL = "predictive_model"         # ML model predictions
    CORRELATION_BASED = "correlation_based"       # Multi-sensor correlations
    TIME_BASED = "time_based"                     # Scheduled maintenance
    EVENT_BASED = "event_based"                   # Specific event triggers
    HEALTH_BASED = "health_based"                 # Equipment health scores
    FAILURE_MODE = "failure_mode"                 # Specific failure mode detection


class MaintenanceType(Enum):
    """Types of maintenance activities"""
    INSPECTION = "inspection"                     # Visual/basic inspection
    CALIBRATION = "calibration"                   # Sensor calibration
    CLEANING = "cleaning"                         # Cleaning/debris removal
    LUBRICATION = "lubrication"                   # Lubrication service
    REPLACEMENT = "replacement"                   # Component replacement
    REPAIR = "repair"                             # Corrective repair
    OVERHAUL = "overhaul"                         # Major overhaul
    EMERGENCY = "emergency"                       # Emergency intervention


class Priority(Enum):
    """Maintenance priority levels"""
    ROUTINE = 1                                   # Normal schedule
    LOW = 2                                       # Can be delayed
    MEDIUM = 3                                    # Standard priority
    HIGH = 4                                      # Schedule soon
    URGENT = 5                                    # Immediate attention
    CRITICAL = 6                                  # Emergency response


@dataclass
class TriggerCondition:
    """Individual trigger condition definition"""
    condition_id: str
    trigger_type: TriggerType
    sensor_ids: List[str]                         # Sensors involved

    # Threshold conditions
    threshold_value: Optional[float] = None
    comparison_operator: str = ">"                # >, <, >=, <=, ==, !=

    # Trend conditions
    trend_window_hours: int = 24
    trend_threshold: float = 0.1                  # Rate of change threshold

    # Correlation conditions
    correlation_threshold: float = 0.8
    min_correlated_sensors: int = 2

    # Time conditions
    maintenance_interval_hours: int = 720         # 30 days default
    last_maintenance: Optional[datetime] = None

    # Health conditions
    health_threshold: float = 75.0                # Health score threshold
    consecutive_violations: int = 3               # Required consecutive violations

    # Failure mode conditions
    target_failure_modes: List[FailureMode] = field(default_factory=list)
    failure_probability_threshold: float = 0.3


@dataclass
class MaintenanceAction:
    """Maintenance action definition"""
    action_id: str
    maintenance_type: MaintenanceType
    priority: Priority

    # Resource requirements
    estimated_duration_hours: float
    required_technicians: int = 1
    required_skills: List[str] = field(default_factory=list)
    required_parts: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0

    # Scheduling constraints
    can_defer_hours: int = 0                      # How long can this be delayed
    requires_system_shutdown: bool = False
    preferred_time_windows: List[Tuple[int, int]] = field(default_factory=list)  # (start_hour, end_hour)

    # Safety and operational impact
    safety_criticality: int = 1                  # 1-5 scale
    operational_impact: float = 0.1              # 0-1 scale

    # Dependencies
    prerequisite_actions: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)


@dataclass
class TriggerRule:
    """Complete trigger rule definition"""
    rule_id: str
    rule_name: str
    description: str

    # Trigger conditions (AND logic between conditions)
    conditions: List[TriggerCondition]

    # Actions to trigger
    actions: List[MaintenanceAction]

    # Rule metadata
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    # Rule evaluation parameters
    cooldown_hours: int = 24                      # Minimum time between triggers
    max_triggers_per_day: int = 3                 # Prevent excessive triggering
    confidence_threshold: float = 0.7             # Minimum confidence to trigger


@dataclass
class TriggerEvent:
    """Triggered maintenance event"""
    event_id: str
    rule_id: str
    trigger_timestamp: datetime

    # Triggering data
    triggering_sensors: List[str]
    trigger_values: Dict[str, float]
    anomaly_scores: Dict[str, float]
    health_scores: Dict[str, float]

    # Event assessment
    confidence: float
    severity: Severity
    predicted_failure_modes: List[FailureMode]

    # Recommended actions
    recommended_actions: List[MaintenanceAction]
    estimated_impact: Dict[str, float]            # Cost, downtime, etc.

    # Status tracking
    is_acknowledged: bool = False
    is_scheduled: bool = False
    scheduled_date: Optional[datetime] = None
    is_completed: bool = False
    completion_date: Optional[datetime] = None


class PredictiveMaintenanceTriggerSystem:
    """
    Comprehensive predictive maintenance trigger system.

    Features:
    - Multi-condition trigger rules
    - Automated maintenance scheduling
    - Resource and constraint management
    - Priority-based action sequencing
    - Cost-benefit optimization
    - Integration with health scoring and failure classification
    """

    def __init__(self,
                 failure_classifier: FailureClassificationEngine = None,
                 health_scorer: EquipmentHealthScorer = None):
        self.config = get_config()
        self.failure_classifier = failure_classifier or FailureClassificationEngine()
        self.health_scorer = health_scorer or EquipmentHealthScorer()

        # Load trigger rules and actions
        self.trigger_rules: Dict[str, TriggerRule] = {}
        self.maintenance_actions: Dict[str, MaintenanceAction] = {}

        # Event tracking
        self.trigger_events: List[TriggerEvent] = []
        self.pending_events: List[TriggerEvent] = []

        # Initialize system
        self._initialize_default_rules()
        self._initialize_default_actions()

        logger.info("Initialized PredictiveMaintenanceTriggerSystem")

    def _initialize_default_rules(self):
        """Initialize default trigger rules for the 80-sensor system"""

        # Rule 1: Critical Health Score Trigger
        critical_health_rule = TriggerRule(
            rule_id="critical_health_rule",
            rule_name="Critical Health Score Trigger",
            description="Triggers when equipment health drops below critical threshold",
            conditions=[
                TriggerCondition(
                    condition_id="health_critical",
                    trigger_type=TriggerType.HEALTH_BASED,
                    sensor_ids=["*"],  # All sensors
                    health_threshold=40.0,
                    consecutive_violations=2
                )
            ],
            actions=["emergency_inspection", "system_diagnostic"]
        )

        # Rule 2: Temperature Anomaly Trigger (SMAP/MSL thermal sensors)
        temp_anomaly_rule = TriggerRule(
            rule_id="temperature_anomaly_rule",
            rule_name="Temperature Anomaly Trigger",
            description="Triggers on thermal system anomalies",
            conditions=[
                TriggerCondition(
                    condition_id="temp_threshold",
                    trigger_type=TriggerType.THRESHOLD_BASED,
                    sensor_ids=["T-1", "T-2", "T-3", "T-1", "T-2", "T-3", "T-4", "T-5", "T-6", "T-7", "T-8"],
                    threshold_value=60.0,
                    comparison_operator=">",
                    consecutive_violations=3
                )
            ],
            actions=["thermal_inspection", "cooling_system_check"]
        )

        # Rule 3: Power System Voltage Fluctuation (SMAP/MSL power sensors)
        power_anomaly_rule = TriggerRule(
            rule_id="power_system_rule",
            rule_name="Power System Voltage Trigger",
            description="Triggers on power system voltage anomalies",
            conditions=[
                TriggerCondition(
                    condition_id="voltage_fluctuation",
                    trigger_type=TriggerType.CORRELATION_BASED,
                    sensor_ids=["V-1", "V-2", "V-3", "P-1", "P-2", "P-3", "P-4", "P-5", "P-6", "P-7"],
                    correlation_threshold=0.7,
                    min_correlated_sensors=3
                )
            ],
            actions=["power_system_diagnostic", "electrical_inspection"]
        )

        # Rule 4: Mobility System Degradation (MSL mobility sensors)
        mobility_rule = TriggerRule(
            rule_id="mobility_degradation_rule",
            rule_name="Mobility System Degradation",
            description="Triggers on mobility system performance degradation",
            conditions=[
                TriggerCondition(
                    condition_id="mobility_performance",
                    trigger_type=TriggerType.TREND_BASED,
                    sensor_ids=["M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7", "M-8", "M-9", "M-10"],
                    trend_window_hours=48,
                    trend_threshold=0.15
                )
            ],
            actions=["mobility_inspection", "wheel_motor_service"]
        )

        # Rule 5: Robotic Arm Stress (MSL arm sensors)
        arm_stress_rule = TriggerRule(
            rule_id="robotic_arm_stress_rule",
            rule_name="Robotic Arm Stress Detection",
            description="Triggers on high robotic arm joint stress",
            conditions=[
                TriggerCondition(
                    condition_id="arm_torque_high",
                    trigger_type=TriggerType.THRESHOLD_BASED,
                    sensor_ids=["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8"],
                    threshold_value=70.0,  # High torque threshold
                    comparison_operator=">",
                    consecutive_violations=5
                )
            ],
            actions=["arm_joint_inspection", "lubrication_service"]
        )

        # Rule 6: Communication Signal Degradation
        comm_degradation_rule = TriggerRule(
            rule_id="communication_degradation_rule",
            rule_name="Communication Signal Degradation",
            description="Triggers on communication system signal quality issues",
            conditions=[
                TriggerCondition(
                    condition_id="signal_quality_low",
                    trigger_type=TriggerType.TREND_BASED,
                    sensor_ids=["C-1", "C-2", "C-3", "C-4", "C-5"],
                    trend_window_hours=12,
                    trend_threshold=-0.1,  # Negative trend (degrading)
                    consecutive_violations=2
                )
            ],
            actions=["communication_diagnostic", "antenna_alignment"]
        )

        # Rule 7: Science Instrument Performance (MSL)
        science_performance_rule = TriggerRule(
            rule_id="science_instrument_rule",
            rule_name="Science Instrument Performance",
            description="Triggers on science instrument performance issues",
            conditions=[
                TriggerCondition(
                    condition_id="instrument_health",
                    trigger_type=TriggerType.HEALTH_BASED,
                    sensor_ids=["S-1", "S-2", "S-3", "S-4", "S-5", "S-6", "S-7", "S-8", "S-9", "S-10", "S-11", "S-12"],
                    health_threshold=70.0,
                    consecutive_violations=1
                )
            ],
            actions=["instrument_calibration", "optical_cleaning"]
        )

        # Rule 8: Scheduled Preventive Maintenance
        preventive_rule = TriggerRule(
            rule_id="preventive_maintenance_rule",
            rule_name="Scheduled Preventive Maintenance",
            description="Time-based preventive maintenance scheduling",
            conditions=[
                TriggerCondition(
                    condition_id="scheduled_maintenance",
                    trigger_type=TriggerType.TIME_BASED,
                    sensor_ids=["*"],
                    maintenance_interval_hours=2160  # 90 days
                )
            ],
            actions=["routine_inspection", "system_calibration"]
        )

        # Add rules to system
        for rule in [critical_health_rule, temp_anomaly_rule, power_anomaly_rule,
                    mobility_rule, arm_stress_rule, comm_degradation_rule,
                    science_performance_rule, preventive_rule]:
            self.trigger_rules[rule.rule_id] = rule

    def _initialize_default_actions(self):
        """Initialize default maintenance actions"""

        actions = [
            # Emergency Actions
            MaintenanceAction(
                action_id="emergency_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.CRITICAL,
                estimated_duration_hours=2.0,
                required_technicians=2,
                required_skills=["systems_diagnosis", "emergency_response"],
                estimated_cost=1000.0,
                safety_criticality=5,
                operational_impact=0.8,
                requires_system_shutdown=True
            ),

            MaintenanceAction(
                action_id="system_diagnostic",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.URGENT,
                estimated_duration_hours=4.0,
                required_technicians=1,
                required_skills=["systems_diagnosis", "data_analysis"],
                estimated_cost=800.0,
                safety_criticality=3,
                operational_impact=0.3
            ),

            # Thermal System Actions
            MaintenanceAction(
                action_id="thermal_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.HIGH,
                estimated_duration_hours=3.0,
                required_technicians=1,
                required_skills=["thermal_systems", "instrumentation"],
                estimated_cost=600.0,
                safety_criticality=4,
                operational_impact=0.4
            ),

            MaintenanceAction(
                action_id="cooling_system_check",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.HIGH,
                estimated_duration_hours=2.5,
                required_technicians=1,
                required_skills=["thermal_systems", "HVAC"],
                required_parts=["thermal_sensors", "coolant"],
                estimated_cost=500.0,
                safety_criticality=3,
                operational_impact=0.2
            ),

            # Power System Actions
            MaintenanceAction(
                action_id="power_system_diagnostic",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.URGENT,
                estimated_duration_hours=3.5,
                required_technicians=1,
                required_skills=["electrical_systems", "power_electronics"],
                estimated_cost=700.0,
                safety_criticality=5,
                operational_impact=0.6,
                requires_system_shutdown=True
            ),

            MaintenanceAction(
                action_id="electrical_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.HIGH,
                estimated_duration_hours=2.0,
                required_technicians=1,
                required_skills=["electrical_systems"],
                estimated_cost=400.0,
                safety_criticality=4,
                operational_impact=0.2
            ),

            # Mobility System Actions
            MaintenanceAction(
                action_id="mobility_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.MEDIUM,
                estimated_duration_hours=4.0,
                required_technicians=2,
                required_skills=["mechanical_systems", "robotics"],
                estimated_cost=800.0,
                safety_criticality=3,
                operational_impact=0.5
            ),

            MaintenanceAction(
                action_id="wheel_motor_service",
                maintenance_type=MaintenanceType.LUBRICATION,
                priority=Priority.MEDIUM,
                estimated_duration_hours=6.0,
                required_technicians=2,
                required_skills=["mechanical_systems", "motor_control"],
                required_parts=["lubricants", "motor_brushes"],
                estimated_cost=1200.0,
                safety_criticality=2,
                operational_impact=0.7,
                can_defer_hours=72
            ),

            # Robotic Arm Actions
            MaintenanceAction(
                action_id="arm_joint_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.MEDIUM,
                estimated_duration_hours=3.0,
                required_technicians=1,
                required_skills=["robotics", "mechanical_systems"],
                estimated_cost=600.0,
                safety_criticality=3,
                operational_impact=0.4
            ),

            MaintenanceAction(
                action_id="lubrication_service",
                maintenance_type=MaintenanceType.LUBRICATION,
                priority=Priority.LOW,
                estimated_duration_hours=2.0,
                required_technicians=1,
                required_skills=["mechanical_systems"],
                required_parts=["joint_lubricants"],
                estimated_cost=300.0,
                safety_criticality=1,
                operational_impact=0.1,
                can_defer_hours=168  # 1 week
            ),

            # Communication Actions
            MaintenanceAction(
                action_id="communication_diagnostic",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.HIGH,
                estimated_duration_hours=2.5,
                required_technicians=1,
                required_skills=["communication_systems", "RF_systems"],
                estimated_cost=500.0,
                safety_criticality=2,
                operational_impact=0.5
            ),

            MaintenanceAction(
                action_id="antenna_alignment",
                maintenance_type=MaintenanceType.CALIBRATION,
                priority=Priority.MEDIUM,
                estimated_duration_hours=1.5,
                required_technicians=1,
                required_skills=["communication_systems", "antenna_systems"],
                estimated_cost=300.0,
                safety_criticality=1,
                operational_impact=0.2,
                can_defer_hours=48
            ),

            # Science Instrument Actions
            MaintenanceAction(
                action_id="instrument_calibration",
                maintenance_type=MaintenanceType.CALIBRATION,
                priority=Priority.MEDIUM,
                estimated_duration_hours=4.0,
                required_technicians=1,
                required_skills=["instrumentation", "calibration"],
                required_parts=["calibration_standards"],
                estimated_cost=800.0,
                safety_criticality=1,
                operational_impact=0.3,
                can_defer_hours=120
            ),

            MaintenanceAction(
                action_id="optical_cleaning",
                maintenance_type=MaintenanceType.CLEANING,
                priority=Priority.LOW,
                estimated_duration_hours=1.0,
                required_technicians=1,
                required_skills=["optics", "precision_cleaning"],
                required_parts=["optical_cleaners", "lint_free_cloths"],
                estimated_cost=200.0,
                safety_criticality=1,
                operational_impact=0.1,
                can_defer_hours=240
            ),

            # Routine Actions
            MaintenanceAction(
                action_id="routine_inspection",
                maintenance_type=MaintenanceType.INSPECTION,
                priority=Priority.ROUTINE,
                estimated_duration_hours=8.0,
                required_technicians=2,
                required_skills=["systems_diagnosis", "preventive_maintenance"],
                estimated_cost=1500.0,
                safety_criticality=2,
                operational_impact=0.3,
                can_defer_hours=720,  # 30 days
                preferred_time_windows=[(22, 6)]  # Night shift
            ),

            MaintenanceAction(
                action_id="system_calibration",
                maintenance_type=MaintenanceType.CALIBRATION,
                priority=Priority.ROUTINE,
                estimated_duration_hours=6.0,
                required_technicians=1,
                required_skills=["calibration", "systems_diagnosis"],
                required_parts=["calibration_standards"],
                estimated_cost=1000.0,
                safety_criticality=1,
                operational_impact=0.2,
                can_defer_hours=1440,  # 60 days
                preferred_time_windows=[(20, 8)]  # Off-peak hours
            ),
        ]

        # Add actions to system
        for action in actions:
            self.maintenance_actions[action.action_id] = action

    def evaluate_triggers(self,
                         sensor_data: Dict[str, float],
                         anomaly_scores: Dict[str, float],
                         health_scores: Dict[str, SensorHealth],
                         time_window_data: Optional[Dict[str, np.ndarray]] = None) -> List[TriggerEvent]:
        """
        Evaluate all trigger rules and generate maintenance events

        Args:
            sensor_data: Current sensor readings {sensor_id: value}
            anomaly_scores: Anomaly scores {sensor_id: score}
            health_scores: Health assessment results {sensor_id: SensorHealth}
            time_window_data: Historical data for trend analysis

        Returns:
            List of triggered maintenance events
        """
        triggered_events = []
        current_time = datetime.now()

        logger.info(f"Evaluating {len(self.trigger_rules)} trigger rules")

        for rule_id, rule in self.trigger_rules.items():
            if not rule.is_active:
                continue

            # Check cooldown period
            if (rule.last_triggered and
                (current_time - rule.last_triggered).total_seconds() < rule.cooldown_hours * 3600):
                continue

            # Check daily trigger limit
            today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            today_triggers = sum(1 for event in self.trigger_events
                               if (event.rule_id == rule_id and
                                   event.trigger_timestamp >= today_start))

            if today_triggers >= rule.max_triggers_per_day:
                continue

            # Evaluate rule conditions
            rule_triggered, confidence, triggering_data = self._evaluate_rule_conditions(
                rule, sensor_data, anomaly_scores, health_scores, time_window_data
            )

            if rule_triggered and confidence >= rule.confidence_threshold:
                # Create trigger event
                event = self._create_trigger_event(
                    rule, triggering_data, confidence, sensor_data, anomaly_scores, health_scores
                )

                triggered_events.append(event)

                # Update rule statistics
                rule.last_triggered = current_time
                rule.trigger_count += 1

                logger.info(f"Rule '{rule.rule_name}' triggered with confidence {confidence:.2f}")

        # Store triggered events
        self.trigger_events.extend(triggered_events)
        self.pending_events.extend([e for e in triggered_events if not e.is_acknowledged])

        return triggered_events

    def _evaluate_rule_conditions(self,
                                 rule: TriggerRule,
                                 sensor_data: Dict[str, float],
                                 anomaly_scores: Dict[str, float],
                                 health_scores: Dict[str, SensorHealth],
                                 time_window_data: Optional[Dict[str, np.ndarray]]) -> Tuple[bool, float, Dict]:
        """Evaluate all conditions for a trigger rule"""

        condition_results = []
        triggering_data = {}

        for condition in rule.conditions:
            result, confidence, data = self._evaluate_single_condition(
                condition, sensor_data, anomaly_scores, health_scores, time_window_data
            )

            condition_results.append((result, confidence))
            triggering_data.update(data)

        # All conditions must be true (AND logic)
        rule_triggered = all(result for result, _ in condition_results)

        # Average confidence across conditions
        total_confidence = np.mean([conf for _, conf in condition_results]) if condition_results else 0.0

        return rule_triggered, total_confidence, triggering_data

    def _evaluate_single_condition(self,
                                  condition: TriggerCondition,
                                  sensor_data: Dict[str, float],
                                  anomaly_scores: Dict[str, float],
                                  health_scores: Dict[str, SensorHealth],
                                  time_window_data: Optional[Dict[str, np.ndarray]]) -> Tuple[bool, float, Dict]:
        """Evaluate a single trigger condition"""

        # Get relevant sensors
        if condition.sensor_ids == ["*"]:
            relevant_sensors = list(sensor_data.keys())
        else:
            relevant_sensors = [sid for sid in condition.sensor_ids if sid in sensor_data]

        if not relevant_sensors:
            return False, 0.0, {}

        triggering_data = {}

        if condition.trigger_type == TriggerType.THRESHOLD_BASED:
            return self._evaluate_threshold_condition(condition, relevant_sensors, sensor_data, triggering_data)

        elif condition.trigger_type == TriggerType.TREND_BASED:
            return self._evaluate_trend_condition(condition, relevant_sensors, time_window_data, triggering_data)

        elif condition.trigger_type == TriggerType.CORRELATION_BASED:
            return self._evaluate_correlation_condition(condition, relevant_sensors, anomaly_scores, triggering_data)

        elif condition.trigger_type == TriggerType.HEALTH_BASED:
            return self._evaluate_health_condition(condition, relevant_sensors, health_scores, triggering_data)

        elif condition.trigger_type == TriggerType.TIME_BASED:
            return self._evaluate_time_condition(condition, triggering_data)

        elif condition.trigger_type == TriggerType.FAILURE_MODE:
            return self._evaluate_failure_mode_condition(condition, relevant_sensors, health_scores, triggering_data)

        else:
            return False, 0.0, {}

    def _evaluate_threshold_condition(self, condition: TriggerCondition, sensors: List[str],
                                    sensor_data: Dict[str, float], triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate threshold-based condition"""
        violations = 0
        total_sensors = len(sensors)

        for sensor_id in sensors:
            value = sensor_data.get(sensor_id, 0.0)

            # Evaluate threshold condition
            if condition.comparison_operator == ">":
                violated = value > condition.threshold_value
            elif condition.comparison_operator == "<":
                violated = value < condition.threshold_value
            elif condition.comparison_operator == ">=":
                violated = value >= condition.threshold_value
            elif condition.comparison_operator == "<=":
                violated = value <= condition.threshold_value
            elif condition.comparison_operator == "==":
                violated = abs(value - condition.threshold_value) < 1e-6
            elif condition.comparison_operator == "!=":
                violated = abs(value - condition.threshold_value) >= 1e-6
            else:
                violated = False

            if violated:
                violations += 1
                triggering_data[sensor_id] = value

        # Check if enough sensors violated threshold
        violation_ratio = violations / total_sensors
        triggered = violation_ratio >= 0.5  # At least 50% of sensors must violate

        confidence = violation_ratio if triggered else 0.0

        return triggered, confidence, triggering_data

    def _evaluate_trend_condition(self, condition: TriggerCondition, sensors: List[str],
                                time_window_data: Optional[Dict[str, np.ndarray]],
                                triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate trend-based condition"""
        if not time_window_data:
            return False, 0.0, {}

        significant_trends = 0
        total_trends = 0

        for sensor_id in sensors:
            if sensor_id not in time_window_data:
                continue

            data = time_window_data[sensor_id]
            if len(data) < 5:  # Need minimum data points
                continue

            # Calculate trend slope
            x = np.arange(len(data))
            trend_slope = np.polyfit(x, data, 1)[0]

            # Normalize by data range to make threshold meaningful
            data_range = np.max(data) - np.min(data)
            if data_range > 0:
                normalized_trend = abs(trend_slope) / data_range

                if normalized_trend >= condition.trend_threshold:
                    significant_trends += 1
                    triggering_data[sensor_id] = trend_slope

            total_trends += 1

        if total_trends == 0:
            return False, 0.0, {}

        trend_ratio = significant_trends / total_trends
        triggered = trend_ratio >= 0.3  # At least 30% show significant trends
        confidence = trend_ratio if triggered else 0.0

        return triggered, confidence, triggering_data

    def _evaluate_correlation_condition(self, condition: TriggerCondition, sensors: List[str],
                                      anomaly_scores: Dict[str, float],
                                      triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate correlation-based condition"""
        # Get anomaly scores for relevant sensors
        sensor_anomalies = {sid: anomaly_scores.get(sid, 0.0) for sid in sensors}

        # Count sensors with high anomaly scores
        high_anomaly_sensors = [sid for sid, score in sensor_anomalies.items()
                               if score >= condition.correlation_threshold]

        triggered = len(high_anomaly_sensors) >= condition.min_correlated_sensors

        if triggered:
            for sensor_id in high_anomaly_sensors:
                triggering_data[sensor_id] = sensor_anomalies[sensor_id]

        confidence = len(high_anomaly_sensors) / len(sensors) if triggered else 0.0

        return triggered, confidence, triggering_data

    def _evaluate_health_condition(self, condition: TriggerCondition, sensors: List[str],
                                 health_scores: Dict[str, SensorHealth],
                                 triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate health-based condition"""
        unhealthy_sensors = 0
        total_sensors = len(sensors)

        for sensor_id in sensors:
            if sensor_id in health_scores:
                health_score = health_scores[sensor_id].health_score

                if health_score < condition.health_threshold:
                    unhealthy_sensors += 1
                    triggering_data[sensor_id] = health_score

        health_violation_ratio = unhealthy_sensors / total_sensors
        triggered = health_violation_ratio >= 0.2  # At least 20% unhealthy
        confidence = health_violation_ratio if triggered else 0.0

        return triggered, confidence, triggering_data

    def _evaluate_time_condition(self, condition: TriggerCondition, triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate time-based condition"""
        if condition.last_maintenance is None:
            # No maintenance history - trigger immediately
            triggered = True
            confidence = 1.0
        else:
            time_since_maintenance = datetime.now() - condition.last_maintenance
            hours_since = time_since_maintenance.total_seconds() / 3600

            triggered = hours_since >= condition.maintenance_interval_hours
            confidence = min(1.0, hours_since / condition.maintenance_interval_hours)

        triggering_data['time_since_maintenance'] = hours_since if condition.last_maintenance else float('inf')

        return triggered, confidence, triggering_data

    def _evaluate_failure_mode_condition(self, condition: TriggerCondition, sensors: List[str],
                                       health_scores: Dict[str, SensorHealth],
                                       triggering_data: Dict) -> Tuple[bool, float, Dict]:
        """Evaluate failure mode condition"""
        # This would integrate with failure classification engine
        # For now, use health scores as proxy

        critical_sensors = 0
        for sensor_id in sensors:
            if sensor_id in health_scores:
                health = health_scores[sensor_id]

                # Consider sensors with poor health as having potential failure modes
                if health.health_score < 60:
                    critical_sensors += 1
                    triggering_data[sensor_id] = health.health_score

        triggered = critical_sensors >= 1
        confidence = min(1.0, critical_sensors / len(sensors)) if triggered else 0.0

        return triggered, confidence, triggering_data

    def _create_trigger_event(self, rule: TriggerRule, triggering_data: Dict, confidence: float,
                             sensor_data: Dict[str, float], anomaly_scores: Dict[str, float],
                             health_scores: Dict[str, SensorHealth]) -> TriggerEvent:
        """Create a trigger event from rule evaluation"""

        event_id = f"event_{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Determine severity based on confidence and health scores
        if confidence > 0.9:
            severity = Severity.CRITICAL
        elif confidence > 0.8:
            severity = Severity.HIGH
        elif confidence > 0.6:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        # Get triggering sensors
        triggering_sensors = list(triggering_data.keys())

        # Extract relevant sensor values and scores
        trigger_values = {sid: sensor_data.get(sid, 0.0) for sid in triggering_sensors}
        trigger_anomalies = {sid: anomaly_scores.get(sid, 0.0) for sid in triggering_sensors}
        trigger_health = {sid: health_scores[sid].health_score for sid in triggering_sensors
                         if sid in health_scores}

        # Get recommended actions from rule
        recommended_actions = [self.maintenance_actions[action_id]
                             for action_id in rule.actions
                             if action_id in self.maintenance_actions]

        # Estimate impact
        estimated_impact = self._estimate_event_impact(recommended_actions, severity)

        # Predict failure modes (placeholder)
        predicted_failure_modes = [FailureMode.PERFORMANCE_DEGRADATION]

        return TriggerEvent(
            event_id=event_id,
            rule_id=rule.rule_id,
            trigger_timestamp=datetime.now(),
            triggering_sensors=triggering_sensors,
            trigger_values=trigger_values,
            anomaly_scores=trigger_anomalies,
            health_scores=trigger_health,
            confidence=confidence,
            severity=severity,
            predicted_failure_modes=predicted_failure_modes,
            recommended_actions=recommended_actions,
            estimated_impact=estimated_impact
        )

    def _estimate_event_impact(self, actions: List[MaintenanceAction], severity: Severity) -> Dict[str, float]:
        """Estimate the impact of maintenance actions"""
        total_cost = sum(action.estimated_cost for action in actions)
        total_downtime = sum(action.estimated_duration_hours for action in actions)
        max_operational_impact = max([action.operational_impact for action in actions], default=0.0)

        # Severity multipliers
        severity_multipliers = {
            Severity.LOW: 1.0,
            Severity.MEDIUM: 1.2,
            Severity.HIGH: 1.5,
            Severity.CRITICAL: 2.0
        }

        multiplier = severity_multipliers.get(severity, 1.0)

        return {
            'estimated_cost': total_cost * multiplier,
            'estimated_downtime_hours': total_downtime,
            'operational_impact': max_operational_impact,
            'severity_multiplier': multiplier
        }

    def get_pending_events(self, priority_filter: Optional[Priority] = None) -> List[TriggerEvent]:
        """Get pending maintenance events, optionally filtered by priority"""
        events = [event for event in self.pending_events if not event.is_acknowledged]

        if priority_filter:
            events = [event for event in events
                     if any(action.priority == priority_filter for action in event.recommended_actions)]

        # Sort by severity and timestamp
        severity_order = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
        events.sort(key=lambda e: (severity_order.get(e.severity, 0), e.trigger_timestamp), reverse=True)

        return events

    def acknowledge_event(self, event_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge a triggered event"""
        for event in self.pending_events:
            if event.event_id == event_id:
                event.is_acknowledged = True
                logger.info(f"Event {event_id} acknowledged by {acknowledged_by}")
                return True

        return False

    def schedule_maintenance(self, event_id: str, scheduled_date: datetime) -> bool:
        """Schedule maintenance for a triggered event"""
        for event in self.trigger_events:
            if event.event_id == event_id:
                event.is_scheduled = True
                event.scheduled_date = scheduled_date
                logger.info(f"Maintenance scheduled for event {event_id} on {scheduled_date}")
                return True

        return False

    def add_custom_rule(self, rule: TriggerRule) -> bool:
        """Add a custom trigger rule"""
        if rule.rule_id in self.trigger_rules:
            logger.warning(f"Rule {rule.rule_id} already exists")
            return False

        self.trigger_rules[rule.rule_id] = rule
        logger.info(f"Added custom rule: {rule.rule_name}")
        return True

    def get_system_status(self) -> Dict[str, any]:
        """Get overall trigger system status"""
        total_rules = len(self.trigger_rules)
        active_rules = sum(1 for rule in self.trigger_rules.values() if rule.is_active)
        pending_events_count = len(self.get_pending_events())

        # Recent activity
        recent_events = [e for e in self.trigger_events
                        if (datetime.now() - e.trigger_timestamp).total_seconds() < 24 * 3600]

        # Priority breakdown
        priority_counts = {}
        for event in self.get_pending_events():
            for action in event.recommended_actions:
                priority = action.priority
                priority_counts[priority.value] = priority_counts.get(priority.value, 0) + 1

        return {
            'total_rules': total_rules,
            'active_rules': active_rules,
            'pending_events': pending_events_count,
            'recent_events_24h': len(recent_events),
            'priority_breakdown': priority_counts,
            'system_health': "operational"
        }