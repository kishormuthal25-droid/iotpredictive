"""
Failure Mode Classification Engine for IoT Predictive Maintenance System

This module implements intelligent failure mode classification for 80+ sensors,
mapping sensor anomalies to specific equipment failure types and root causes.
Supports both SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory) datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Local imports
from ..utils.config import get_config
from ..anomaly_detection.models.lstm_predictor import LSTMPredictor


logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Enumeration of equipment failure modes"""
    # Mechanical Failures
    BEARING_WEAR = "bearing_wear"
    VIBRATION_ANOMALY = "vibration_anomaly"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"

    # Thermal Failures
    OVERHEATING = "overheating"
    THERMAL_CYCLING = "thermal_cycling"
    INSULATION_BREAKDOWN = "insulation_breakdown"

    # Electrical Failures
    VOLTAGE_FLUCTUATION = "voltage_fluctuation"
    CURRENT_SPIKE = "current_spike"
    GROUND_FAULT = "ground_fault"
    PHASE_IMBALANCE = "phase_imbalance"

    # Fluid/Flow Failures
    FLOW_BLOCKAGE = "flow_blockage"
    PRESSURE_DROP = "pressure_drop"
    LEAKAGE = "leakage"
    CAVITATION = "cavitation"

    # Communication Failures
    SENSOR_DRIFT = "sensor_drift"
    SIGNAL_LOSS = "signal_loss"
    DATA_CORRUPTION = "data_corruption"

    # Environmental Failures
    CORROSION = "corrosion"
    CONTAMINATION = "contamination"
    MOISTURE_INGRESS = "moisture_ingress"

    # System Failures
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EFFICIENCY_LOSS = "efficiency_loss"
    CONTROL_LOOP_INSTABILITY = "control_loop_instability"


class Severity(Enum):
    """Failure severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class FailureClassification:
    """Result of failure mode classification"""
    sensor_id: str
    failure_modes: List[FailureMode]
    probabilities: List[float]
    severity: Severity
    confidence: float
    root_cause: str
    recommended_actions: List[str]
    time_to_failure_hours: Optional[float] = None
    affected_subsystems: List[str] = field(default_factory=list)


@dataclass
class SensorMapping:
    """Sensor to equipment mapping configuration"""
    sensor_id: str
    equipment_type: str
    subsystem: str
    location: str
    measurement_type: str
    normal_range: Tuple[float, float]
    critical_thresholds: Dict[str, float]
    related_sensors: List[str] = field(default_factory=list)


class FailureClassificationEngine:
    """
    Advanced failure mode classification engine for 80-sensor IoT system.

    Features:
    - Multi-output classification for simultaneous failure mode detection
    - Sensor correlation analysis for root cause identification
    - Time-series pattern recognition for early failure detection
    - Subsystem impact assessment
    - Severity scoring with confidence intervals
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config()
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Load sensor mappings
        self.sensor_mappings = self._load_sensor_mappings()

        # Initialize failure mode rules
        self.failure_rules = self._initialize_failure_rules()

        # Load or initialize models
        self._load_models()

        logger.info(f"Initialized FailureClassificationEngine for {len(self.sensor_mappings)} sensors")

    def _load_sensor_mappings(self) -> Dict[str, SensorMapping]:
        """Load sensor to equipment mappings for SMAP and MSL datasets"""
        mappings = {}

        # SMAP Sensor Mappings (25 sensors - Soil Moisture Active Passive)
        smap_sensors = {
            # Soil Moisture Sensors
            'P-1': SensorMapping('P-1', 'moisture_probe', 'soil_monitoring', 'zone_a', 'soil_moisture', (0.1, 0.6), {'low': 0.05, 'high': 0.8}),
            'P-2': SensorMapping('P-2', 'moisture_probe', 'soil_monitoring', 'zone_b', 'soil_moisture', (0.1, 0.6), {'low': 0.05, 'high': 0.8}),
            'P-3': SensorMapping('P-3', 'moisture_probe', 'soil_monitoring', 'zone_c', 'soil_moisture', (0.1, 0.6), {'low': 0.05, 'high': 0.8}),
            'P-4': SensorMapping('P-4', 'moisture_probe', 'soil_monitoring', 'zone_d', 'soil_moisture', (0.1, 0.6), {'low': 0.05, 'high': 0.8}),

            # Temperature Sensors
            'T-1': SensorMapping('T-1', 'temperature_sensor', 'thermal_monitoring', 'surface', 'temperature', (-40, 60), {'low': -50, 'high': 70}),
            'T-2': SensorMapping('T-2', 'temperature_sensor', 'thermal_monitoring', 'subsurface', 'temperature', (-20, 40), {'low': -30, 'high': 50}),
            'T-3': SensorMapping('T-3', 'temperature_sensor', 'thermal_monitoring', 'electronics', 'temperature', (0, 45), {'low': -10, 'high': 55}),

            # Radar System Sensors
            'R-1': SensorMapping('R-1', 'radar_unit', 'radar_system', 'antenna_1', 'signal_strength', (-100, -20), {'low': -120, 'high': -10}),
            'R-2': SensorMapping('R-2', 'radar_unit', 'radar_system', 'antenna_2', 'signal_strength', (-100, -20), {'low': -120, 'high': -10}),
            'R-3': SensorMapping('R-3', 'radar_unit', 'radar_system', 'receiver', 'noise_level', (-80, -40), {'low': -90, 'high': -30}),

            # Power System Sensors
            'V-1': SensorMapping('V-1', 'voltage_sensor', 'power_system', 'solar_panel_1', 'voltage', (24, 30), {'low': 20, 'high': 35}),
            'V-2': SensorMapping('V-2', 'voltage_sensor', 'power_system', 'solar_panel_2', 'voltage', (24, 30), {'low': 20, 'high': 35}),
            'V-3': SensorMapping('V-3', 'voltage_sensor', 'power_system', 'battery', 'voltage', (22, 28), {'low': 18, 'high': 32}),
            'I-1': SensorMapping('I-1', 'current_sensor', 'power_system', 'main_bus', 'current', (0.5, 5.0), {'low': 0.1, 'high': 8.0}),
            'I-2': SensorMapping('I-2', 'current_sensor', 'power_system', 'radar_bus', 'current', (1.0, 3.0), {'low': 0.2, 'high': 5.0}),

            # Vibration Sensors
            'A-1': SensorMapping('A-1', 'accelerometer', 'mechanical_system', 'antenna_gimbal', 'acceleration', (0, 2.0), {'low': 0, 'high': 5.0}),
            'A-2': SensorMapping('A-2', 'accelerometer', 'mechanical_system', 'platform', 'acceleration', (0, 1.0), {'low': 0, 'high': 3.0}),

            # Pressure Sensors
            'Pr-1': SensorMapping('Pr-1', 'pressure_sensor', 'environmental', 'internal', 'pressure', (950, 1050), {'low': 900, 'high': 1100}),
            'Pr-2': SensorMapping('Pr-2', 'pressure_sensor', 'environmental', 'external', 'pressure', (0.1, 1013), {'low': 0, 'high': 1100}),

            # Communication Sensors
            'C-1': SensorMapping('C-1', 'comm_unit', 'communication', 'transmitter', 'signal_quality', (0.8, 1.0), {'low': 0.5, 'high': 1.0}),
            'C-2': SensorMapping('C-2', 'comm_unit', 'communication', 'receiver', 'signal_quality', (0.8, 1.0), {'low': 0.5, 'high': 1.0}),

            # GPS Sensors
            'G-1': SensorMapping('G-1', 'gps_unit', 'navigation', 'primary', 'position_accuracy', (0, 5), {'low': 0, 'high': 20}),
            'G-2': SensorMapping('G-2', 'gps_unit', 'navigation', 'backup', 'position_accuracy', (0, 5), {'low': 0, 'high': 20}),

            # Attitude Sensors
            'At-1': SensorMapping('At-1', 'gyroscope', 'attitude_control', 'roll', 'angular_velocity', (-10, 10), {'low': -20, 'high': 20}),
            'At-2': SensorMapping('At-2', 'gyroscope', 'attitude_control', 'pitch', 'angular_velocity', (-10, 10), {'low': -20, 'high': 20}),
            'At-3': SensorMapping('At-3', 'gyroscope', 'attitude_control', 'yaw', 'angular_velocity', (-10, 10), {'low': -20, 'high': 20}),
        }

        # MSL Sensor Mappings (55 sensors - Mars Science Laboratory)
        msl_sensors = {
            # Rover Mobility System (10 sensors)
            'M-1': SensorMapping('M-1', 'wheel_motor', 'mobility', 'front_left_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-2': SensorMapping('M-2', 'wheel_motor', 'mobility', 'front_right_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-3': SensorMapping('M-3', 'wheel_motor', 'mobility', 'middle_left_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-4': SensorMapping('M-4', 'wheel_motor', 'mobility', 'middle_right_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-5': SensorMapping('M-5', 'wheel_motor', 'mobility', 'rear_left_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-6': SensorMapping('M-6', 'wheel_motor', 'mobility', 'rear_right_wheel', 'motor_current', (0.5, 3.0), {'low': 0.1, 'high': 5.0}),
            'M-7': SensorMapping('M-7', 'suspension', 'mobility', 'rocker_bogie_left', 'joint_angle', (-45, 45), {'low': -60, 'high': 60}),
            'M-8': SensorMapping('M-8', 'suspension', 'mobility', 'rocker_bogie_right', 'joint_angle', (-45, 45), {'low': -60, 'high': 60}),
            'M-9': SensorMapping('M-9', 'odometry', 'mobility', 'navigation', 'position_error', (0, 0.5), {'low': 0, 'high': 2.0}),
            'M-10': SensorMapping('M-10', 'traction', 'mobility', 'surface_grip', 'slip_ratio', (0, 0.1), {'low': 0, 'high': 0.5}),

            # Robotic Arm System (8 sensors)
            'A-1': SensorMapping('A-1', 'arm_joint', 'robotic_arm', 'shoulder_azimuth', 'torque', (0, 50), {'low': 0, 'high': 80}),
            'A-2': SensorMapping('A-2', 'arm_joint', 'robotic_arm', 'shoulder_elevation', 'torque', (0, 50), {'low': 0, 'high': 80}),
            'A-3': SensorMapping('A-3', 'arm_joint', 'robotic_arm', 'elbow', 'torque', (0, 30), {'low': 0, 'high': 50}),
            'A-4': SensorMapping('A-4', 'arm_joint', 'robotic_arm', 'wrist', 'torque', (0, 20), {'low': 0, 'high': 35}),
            'A-5': SensorMapping('A-5', 'arm_joint', 'robotic_arm', 'turret', 'torque', (0, 25), {'low': 0, 'high': 40}),
            'A-6': SensorMapping('A-6', 'position_sensor', 'robotic_arm', 'end_effector', 'position_accuracy', (0, 2), {'low': 0, 'high': 10}),
            'A-7': SensorMapping('A-7', 'force_sensor', 'robotic_arm', 'gripper', 'grip_force', (0, 100), {'low': 0, 'high': 150}),
            'A-8': SensorMapping('A-8', 'temperature_sensor', 'robotic_arm', 'actuators', 'temperature', (-40, 40), {'low': -50, 'high': 60}),

            # Scientific Instruments (12 sensors)
            'S-1': SensorMapping('S-1', 'spectrometer', 'science_instruments', 'chemcam_laser', 'laser_power', (80, 100), {'low': 70, 'high': 110}),
            'S-2': SensorMapping('S-2', 'spectrometer', 'science_instruments', 'chemcam_detector', 'signal_intensity', (1000, 50000), {'low': 500, 'high': 60000}),
            'S-3': SensorMapping('S-3', 'drill', 'science_instruments', 'drill_motor', 'motor_current', (0.5, 4.0), {'low': 0.1, 'high': 6.0}),
            'S-4': SensorMapping('S-4', 'drill', 'science_instruments', 'drill_depth', 'penetration_depth', (0, 50), {'low': 0, 'high': 60}),
            'S-5': SensorMapping('S-5', 'camera', 'science_instruments', 'mastcam_left', 'image_quality', (0.8, 1.0), {'low': 0.5, 'high': 1.0}),
            'S-6': SensorMapping('S-6', 'camera', 'science_instruments', 'mastcam_right', 'image_quality', (0.8, 1.0), {'low': 0.5, 'high': 1.0}),
            'S-7': SensorMapping('S-7', 'camera', 'science_instruments', 'navcam_left', 'image_quality', (0.7, 1.0), {'low': 0.4, 'high': 1.0}),
            'S-8': SensorMapping('S-8', 'camera', 'science_instruments', 'navcam_right', 'image_quality', (0.7, 1.0), {'low': 0.4, 'high': 1.0}),
            'S-9': SensorMapping('S-9', 'atmospheric_sensor', 'science_instruments', 'rems_pressure', 'atmospheric_pressure', (400, 1200), {'low': 300, 'high': 1500}),
            'S-10': SensorMapping('S-10', 'atmospheric_sensor', 'science_instruments', 'rems_humidity', 'relative_humidity', (0, 100), {'low': 0, 'high': 100}),
            'S-11': SensorMapping('S-11', 'atmospheric_sensor', 'science_instruments', 'rems_wind_speed', 'wind_speed', (0, 30), {'low': 0, 'high': 50}),
            'S-12': SensorMapping('S-12', 'radiation_sensor', 'science_instruments', 'rad_detector', 'radiation_level', (0, 1000), {'low': 0, 'high': 2000}),

            # Power and Thermal Management (15 sensors)
            'P-1': SensorMapping('P-1', 'rtg', 'power_system', 'radioisotope_generator', 'power_output', (100, 120), {'low': 80, 'high': 140}),
            'P-2': SensorMapping('P-2', 'battery', 'power_system', 'lithium_ion_battery', 'voltage', (22, 29), {'low': 18, 'high': 32}),
            'P-3': SensorMapping('P-3', 'battery', 'power_system', 'lithium_ion_battery', 'current', (-10, 15), {'low': -20, 'high': 25}),
            'P-4': SensorMapping('P-4', 'battery', 'power_system', 'lithium_ion_battery', 'temperature', (-40, 40), {'low': -50, 'high': 60}),
            'P-5': SensorMapping('P-5', 'power_distribution', 'power_system', 'main_bus', 'voltage', (28, 32), {'low': 24, 'high': 36}),
            'P-6': SensorMapping('P-6', 'power_distribution', 'power_system', 'payload_bus', 'voltage', (28, 32), {'low': 24, 'high': 36}),
            'P-7': SensorMapping('P-7', 'power_distribution', 'power_system', 'mobility_bus', 'voltage', (28, 32), {'low': 24, 'high': 36}),
            'T-1': SensorMapping('T-1', 'temperature_sensor', 'thermal_system', 'electronics_box', 'temperature', (-40, 40), {'low': -50, 'high': 60}),
            'T-2': SensorMapping('T-2', 'temperature_sensor', 'thermal_system', 'battery_heater', 'temperature', (-30, 30), {'low': -40, 'high': 50}),
            'T-3': SensorMapping('T-3', 'temperature_sensor', 'thermal_system', 'instrument_deck', 'temperature', (-80, 20), {'low': -100, 'high': 40}),
            'T-4': SensorMapping('T-4', 'temperature_sensor', 'thermal_system', 'mobility_motors', 'temperature', (-40, 60), {'low': -50, 'high': 80}),
            'T-5': SensorMapping('T-5', 'heater', 'thermal_system', 'survival_heater_1', 'power_consumption', (10, 50), {'low': 0, 'high': 80}),
            'T-6': SensorMapping('T-6', 'heater', 'thermal_system', 'survival_heater_2', 'power_consumption', (10, 50), {'low': 0, 'high': 80}),
            'T-7': SensorMapping('T-7', 'thermal_radiator', 'thermal_system', 'heat_rejection', 'heat_flow', (20, 100), {'low': 0, 'high': 150}),
            'T-8': SensorMapping('T-8', 'insulation', 'thermal_system', 'thermal_blanket', 'thermal_resistance', (5, 15), {'low': 2, 'high': 20}),

            # Communication System (5 sensors)
            'C-1': SensorMapping('C-1', 'antenna', 'communication', 'high_gain_antenna', 'signal_strength', (-120, -80), {'low': -140, 'high': -60}),
            'C-2': SensorMapping('C-2', 'antenna', 'communication', 'low_gain_antenna', 'signal_strength', (-130, -90), {'low': -150, 'high': -70}),
            'C-3': SensorMapping('C-3', 'transmitter', 'communication', 'x_band_transmitter', 'output_power', (15, 25), {'low': 10, 'high': 30}),
            'C-4': SensorMapping('C-4', 'receiver', 'communication', 'x_band_receiver', 'noise_figure', (2, 5), {'low': 1, 'high': 8}),
            'C-5': SensorMapping('C-5', 'modem', 'communication', 'data_link', 'bit_error_rate', (0, 0.001), {'low': 0, 'high': 0.01}),

            # Navigation and Attitude (5 sensors)
            'N-1': SensorMapping('N-1', 'imu', 'navigation', 'inertial_measurement', 'angular_velocity', (-180, 180), {'low': -360, 'high': 360}),
            'N-2': SensorMapping('N-2', 'imu', 'navigation', 'inertial_measurement', 'linear_acceleration', (-20, 20), {'low': -50, 'high': 50}),
            'N-3': SensorMapping('N-3', 'star_tracker', 'navigation', 'attitude_determination', 'tracking_accuracy', (0, 0.1), {'low': 0, 'high': 1.0}),
            'N-4': SensorMapping('N-4', 'sun_sensor', 'navigation', 'sun_position', 'sun_angle', (0, 180), {'low': 0, 'high': 180}),
            'N-5': SensorMapping('N-5', 'hazard_camera', 'navigation', 'terrain_assessment', 'obstacle_detection', (0, 1), {'low': 0, 'high': 1}),
        }

        # Combine all mappings
        mappings.update(smap_sensors)
        mappings.update(msl_sensors)

        # Set up sensor relationships
        self._setup_sensor_relationships(mappings)

        return mappings

    def _setup_sensor_relationships(self, mappings: Dict[str, SensorMapping]):
        """Establish relationships between related sensors"""
        # Group sensors by subsystem for correlation analysis
        subsystem_groups = {}
        for sensor_id, mapping in mappings.items():
            subsystem = mapping.subsystem
            if subsystem not in subsystem_groups:
                subsystem_groups[subsystem] = []
            subsystem_groups[subsystem].append(sensor_id)

        # Set related sensors within each subsystem
        for subsystem, sensor_ids in subsystem_groups.items():
            for sensor_id in sensor_ids:
                mappings[sensor_id].related_sensors = [s for s in sensor_ids if s != sensor_id]

    def _initialize_failure_rules(self) -> Dict[str, Dict]:
        """Initialize failure mode classification rules"""
        return {
            # Electrical system rules
            'voltage_anomaly': {
                'conditions': [
                    {'sensor_type': 'voltage_sensor', 'threshold_type': 'range', 'severity': Severity.HIGH},
                    {'sensor_type': 'current_sensor', 'correlation': 'high', 'severity': Severity.MEDIUM}
                ],
                'failure_modes': [FailureMode.VOLTAGE_FLUCTUATION, FailureMode.PHASE_IMBALANCE],
                'root_causes': ['power_supply_instability', 'load_imbalance', 'grid_disturbance']
            },

            # Thermal system rules
            'thermal_anomaly': {
                'conditions': [
                    {'sensor_type': 'temperature_sensor', 'threshold_type': 'high', 'severity': Severity.HIGH},
                    {'sensor_type': 'heater', 'correlation': 'high', 'severity': Severity.MEDIUM}
                ],
                'failure_modes': [FailureMode.OVERHEATING, FailureMode.THERMAL_CYCLING],
                'root_causes': ['insufficient_cooling', 'heater_malfunction', 'thermal_barrier_failure']
            },

            # Mechanical system rules
            'mechanical_anomaly': {
                'conditions': [
                    {'sensor_type': 'accelerometer', 'threshold_type': 'high', 'severity': Severity.HIGH},
                    {'sensor_type': 'wheel_motor', 'correlation': 'medium', 'severity': Severity.MEDIUM}
                ],
                'failure_modes': [FailureMode.VIBRATION_ANOMALY, FailureMode.BEARING_WEAR, FailureMode.MISALIGNMENT],
                'root_causes': ['bearing_degradation', 'mechanical_wear', 'improper_installation']
            },

            # Communication system rules
            'communication_anomaly': {
                'conditions': [
                    {'sensor_type': 'comm_unit', 'threshold_type': 'low', 'severity': Severity.MEDIUM},
                    {'sensor_type': 'antenna', 'correlation': 'high', 'severity': Severity.HIGH}
                ],
                'failure_modes': [FailureMode.SIGNAL_LOSS, FailureMode.DATA_CORRUPTION],
                'root_causes': ['antenna_misalignment', 'interference', 'hardware_degradation']
            }
        }

    def _load_models(self):
        """Load or initialize classification models"""
        model_path = Path(self.config.get('model_save_path', 'data/models'))
        model_path.mkdir(parents=True, exist_ok=True)

        try:
            # Load primary classification model
            self.models['primary'] = joblib.load(model_path / 'failure_classification_model.pkl')
            self.scalers['primary'] = joblib.load(model_path / 'failure_classification_scaler.pkl')
            logger.info("Loaded existing failure classification models")
        except FileNotFoundError:
            # Initialize new models
            self.models['primary'] = MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
            self.scalers['primary'] = StandardScaler()
            logger.info("Initialized new failure classification models")

    def classify_failure(self, sensor_data: Dict[str, float],
                        anomaly_scores: Dict[str, float],
                        time_window_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, FailureClassification]:
        """
        Classify failure modes for anomalous sensors

        Args:
            sensor_data: Current sensor readings {sensor_id: value}
            anomaly_scores: Anomaly scores for each sensor {sensor_id: score}
            time_window_data: Historical data for pattern analysis

        Returns:
            Dictionary of failure classifications {sensor_id: FailureClassification}
        """
        classifications = {}

        # Identify anomalous sensors (score > threshold)
        anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        anomalous_sensors = {sid: score for sid, score in anomaly_scores.items()
                           if score > anomaly_threshold}

        if not anomalous_sensors:
            return classifications

        logger.info(f"Classifying failures for {len(anomalous_sensors)} anomalous sensors")

        for sensor_id, anomaly_score in anomalous_sensors.items():
            if sensor_id not in self.sensor_mappings:
                logger.warning(f"Unknown sensor ID: {sensor_id}")
                continue

            classification = self._classify_single_sensor(
                sensor_id, sensor_data, anomaly_score, time_window_data
            )

            if classification:
                classifications[sensor_id] = classification

        # Perform cross-sensor correlation analysis
        self._enhance_with_correlation_analysis(classifications, sensor_data, anomaly_scores)

        return classifications

    def _classify_single_sensor(self, sensor_id: str, sensor_data: Dict[str, float],
                               anomaly_score: float,
                               time_window_data: Optional[Dict[str, np.ndarray]]) -> Optional[FailureClassification]:
        """Classify failure mode for a single sensor"""
        mapping = self.sensor_mappings[sensor_id]
        current_value = sensor_data.get(sensor_id, 0.0)

        # Extract features for classification
        features = self._extract_features(sensor_id, current_value, anomaly_score,
                                        time_window_data, sensor_data)

        # Rule-based classification
        rule_results = self._apply_rule_based_classification(mapping, current_value, anomaly_score)

        # ML-based classification (if model is trained)
        ml_results = self._apply_ml_classification(features)

        # Combine results
        failure_modes, probabilities = self._combine_classification_results(rule_results, ml_results)

        if not failure_modes:
            return None

        # Determine severity
        severity = self._calculate_severity(mapping, current_value, anomaly_score)

        # Calculate confidence
        confidence = self._calculate_confidence(anomaly_score, probabilities)

        # Identify root cause
        root_cause = self._identify_root_cause(mapping, failure_modes, current_value, anomaly_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(failure_modes, severity, mapping)

        # Estimate time to failure
        ttf = self._estimate_time_to_failure(sensor_id, anomaly_score, time_window_data)

        # Identify affected subsystems
        affected_subsystems = self._identify_affected_subsystems(mapping, failure_modes)

        return FailureClassification(
            sensor_id=sensor_id,
            failure_modes=failure_modes,
            probabilities=probabilities,
            severity=severity,
            confidence=confidence,
            root_cause=root_cause,
            recommended_actions=recommendations,
            time_to_failure_hours=ttf,
            affected_subsystems=affected_subsystems
        )

    def _extract_features(self, sensor_id: str, current_value: float, anomaly_score: float,
                         time_window_data: Optional[Dict[str, np.ndarray]],
                         sensor_data: Dict[str, float]) -> np.ndarray:
        """Extract features for ML classification"""
        mapping = self.sensor_mappings[sensor_id]
        features = []

        # Basic features
        features.extend([
            current_value,
            anomaly_score,
            (current_value - mapping.normal_range[0]) / (mapping.normal_range[1] - mapping.normal_range[0]),  # normalized position
            1.0 if current_value > mapping.critical_thresholds['high'] else 0.0,  # critical high
            1.0 if current_value < mapping.critical_thresholds['low'] else 0.0,   # critical low
        ])

        # Time-series features
        if time_window_data and sensor_id in time_window_data:
            data = time_window_data[sensor_id]
            if len(data) > 1:
                features.extend([
                    np.std(data),           # volatility
                    np.mean(np.diff(data)), # trend
                    len(data),              # data points available
                    np.max(data) - np.min(data),  # range
                ])
            else:
                features.extend([0.0, 0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # Related sensor features
        related_anomaly_count = 0
        related_anomaly_avg = 0.0
        for related_id in mapping.related_sensors[:5]:  # Limit to prevent feature explosion
            if related_id in sensor_data:
                related_mapping = self.sensor_mappings.get(related_id)
                if related_mapping:
                    related_value = sensor_data[related_id]
                    is_anomalous = (related_value > related_mapping.critical_thresholds['high'] or
                                  related_value < related_mapping.critical_thresholds['low'])
                    if is_anomalous:
                        related_anomaly_count += 1
                        related_anomaly_avg += abs(related_value - np.mean(related_mapping.normal_range))

        if related_anomaly_count > 0:
            related_anomaly_avg /= related_anomaly_count

        features.extend([related_anomaly_count, related_anomaly_avg])

        # Equipment type encoding (one-hot for major types)
        equipment_types = ['temperature_sensor', 'voltage_sensor', 'current_sensor', 'accelerometer',
                          'wheel_motor', 'comm_unit', 'camera', 'pressure_sensor']
        for eq_type in equipment_types:
            features.append(1.0 if mapping.equipment_type == eq_type else 0.0)

        return np.array(features)

    def _apply_rule_based_classification(self, mapping: SensorMapping, current_value: float,
                                       anomaly_score: float) -> Tuple[List[FailureMode], List[float]]:
        """Apply rule-based classification logic"""
        failure_modes = []
        probabilities = []

        # Temperature-based rules
        if mapping.measurement_type == 'temperature':
            if current_value > mapping.critical_thresholds['high']:
                failure_modes.append(FailureMode.OVERHEATING)
                probabilities.append(min(0.9, anomaly_score))
            elif current_value < mapping.critical_thresholds['low']:
                failure_modes.append(FailureMode.THERMAL_CYCLING)
                probabilities.append(min(0.8, anomaly_score))

        # Voltage-based rules
        elif mapping.measurement_type == 'voltage':
            if abs(current_value - np.mean(mapping.normal_range)) > (mapping.normal_range[1] - mapping.normal_range[0]) * 0.3:
                failure_modes.append(FailureMode.VOLTAGE_FLUCTUATION)
                probabilities.append(min(0.85, anomaly_score))

        # Current-based rules
        elif mapping.measurement_type == 'current' or 'motor_current' in mapping.measurement_type:
            if current_value > mapping.critical_thresholds['high']:
                failure_modes.append(FailureMode.CURRENT_SPIKE)
                probabilities.append(min(0.8, anomaly_score))
                if 'motor' in mapping.equipment_type:
                    failure_modes.append(FailureMode.BEARING_WEAR)
                    probabilities.append(min(0.7, anomaly_score))

        # Vibration-based rules
        elif mapping.measurement_type == 'acceleration':
            if current_value > mapping.normal_range[1]:
                failure_modes.append(FailureMode.VIBRATION_ANOMALY)
                probabilities.append(min(0.9, anomaly_score))
                if current_value > mapping.critical_thresholds['high']:
                    failure_modes.append(FailureMode.MISALIGNMENT)
                    probabilities.append(min(0.75, anomaly_score))

        # Pressure-based rules
        elif mapping.measurement_type == 'pressure' or 'atmospheric_pressure' in mapping.measurement_type:
            if current_value < mapping.critical_thresholds['low']:
                failure_modes.append(FailureMode.PRESSURE_DROP)
                probabilities.append(min(0.8, anomaly_score))

        # Communication-based rules
        elif 'signal' in mapping.measurement_type or mapping.equipment_type == 'comm_unit':
            if current_value < mapping.normal_range[0]:
                failure_modes.append(FailureMode.SIGNAL_LOSS)
                probabilities.append(min(0.85, anomaly_score))

        # General sensor drift detection
        if anomaly_score > 0.8 and not failure_modes:
            failure_modes.append(FailureMode.SENSOR_DRIFT)
            probabilities.append(anomaly_score)

        return failure_modes, probabilities

    def _apply_ml_classification(self, features: np.ndarray) -> Tuple[List[FailureMode], List[float]]:
        """Apply machine learning classification (placeholder for trained model)"""
        # This would use the trained model once available
        # For now, return empty results
        return [], []

    def _combine_classification_results(self, rule_results: Tuple[List[FailureMode], List[float]],
                                      ml_results: Tuple[List[FailureMode], List[float]]) -> Tuple[List[FailureMode], List[float]]:
        """Combine rule-based and ML classification results"""
        failure_modes, probabilities = rule_results
        ml_modes, ml_probs = ml_results

        # For now, prioritize rule-based results
        # In a full implementation, this would intelligently combine both approaches
        return failure_modes, probabilities

    def _calculate_severity(self, mapping: SensorMapping, current_value: float, anomaly_score: float) -> Severity:
        """Calculate failure severity level"""
        # Critical threshold violations
        if (current_value > mapping.critical_thresholds['high'] or
            current_value < mapping.critical_thresholds['low']):
            return Severity.CRITICAL if anomaly_score > 0.9 else Severity.HIGH

        # High anomaly scores
        if anomaly_score > 0.8:
            return Severity.HIGH
        elif anomaly_score > 0.6:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _calculate_confidence(self, anomaly_score: float, probabilities: List[float]) -> float:
        """Calculate confidence in the classification"""
        if not probabilities:
            return 0.0

        # Base confidence on anomaly score and probability consistency
        max_prob = max(probabilities)
        prob_consistency = 1.0 - np.std(probabilities) if len(probabilities) > 1 else 1.0

        confidence = (anomaly_score * 0.4 + max_prob * 0.4 + prob_consistency * 0.2)
        return min(1.0, confidence)

    def _identify_root_cause(self, mapping: SensorMapping, failure_modes: List[FailureMode],
                           current_value: float, anomaly_score: float) -> str:
        """Identify the most likely root cause"""
        # Equipment-specific root cause analysis
        if mapping.equipment_type == 'temperature_sensor':
            if FailureMode.OVERHEATING in failure_modes:
                return "thermal_management_failure"
            else:
                return "sensor_calibration_drift"

        elif mapping.equipment_type in ['voltage_sensor', 'current_sensor']:
            return "electrical_system_instability"

        elif mapping.equipment_type == 'wheel_motor':
            if FailureMode.BEARING_WEAR in failure_modes:
                return "mechanical_wear_progression"
            else:
                return "motor_controller_malfunction"

        elif mapping.equipment_type == 'accelerometer':
            return "mechanical_vibration_source"

        elif mapping.equipment_type == 'comm_unit':
            return "communication_hardware_degradation"

        else:
            return "unknown_degradation_pattern"

    def _generate_recommendations(self, failure_modes: List[FailureMode], severity: Severity,
                                mapping: SensorMapping) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []

        # Severity-based recommendations
        if severity == Severity.CRITICAL:
            recommendations.append("IMMEDIATE ACTION REQUIRED - Stop equipment operation")
            recommendations.append("Dispatch emergency maintenance team")
        elif severity == Severity.HIGH:
            recommendations.append("Schedule urgent maintenance within 24 hours")
        elif severity == Severity.MEDIUM:
            recommendations.append("Schedule maintenance within 1 week")
        else:
            recommendations.append("Monitor closely and schedule routine maintenance")

        # Failure mode specific recommendations
        failure_mode_actions = {
            FailureMode.OVERHEATING: ["Check cooling system", "Verify thermal barriers", "Inspect heat sinks"],
            FailureMode.VOLTAGE_FLUCTUATION: ["Check power supply", "Inspect electrical connections", "Verify load balance"],
            FailureMode.BEARING_WEAR: ["Replace bearings", "Check lubrication", "Inspect alignment"],
            FailureMode.VIBRATION_ANOMALY: ["Check mounting bolts", "Verify balance", "Inspect for loose components"],
            FailureMode.SIGNAL_LOSS: ["Check antenna alignment", "Inspect cables", "Verify signal processing unit"],
            FailureMode.PRESSURE_DROP: ["Check for leaks", "Inspect seals", "Verify pump operation"],
            FailureMode.SENSOR_DRIFT: ["Recalibrate sensor", "Check sensor mounting", "Replace if necessary"]
        }

        for mode in failure_modes:
            if mode in failure_mode_actions:
                recommendations.extend(failure_mode_actions[mode])

        # Equipment-specific recommendations
        if mapping.equipment_type == 'wheel_motor':
            recommendations.append("Check wheel alignment and tire condition")
        elif mapping.equipment_type == 'comm_unit':
            recommendations.append("Run communication system diagnostics")

        return list(set(recommendations))  # Remove duplicates

    def _estimate_time_to_failure(self, sensor_id: str, anomaly_score: float,
                                time_window_data: Optional[Dict[str, np.ndarray]]) -> Optional[float]:
        """Estimate time to failure in hours"""
        if not time_window_data or sensor_id not in time_window_data:
            # Base estimate on anomaly score only
            if anomaly_score > 0.9:
                return 24.0  # 1 day
            elif anomaly_score > 0.8:
                return 72.0  # 3 days
            elif anomaly_score > 0.7:
                return 168.0  # 1 week
            else:
                return None

        # More sophisticated prediction using historical data
        data = time_window_data[sensor_id]
        if len(data) < 10:
            return None

        # Simple trend-based estimation
        recent_trend = np.polyfit(range(len(data)), data, 1)[0]
        mapping = self.sensor_mappings[sensor_id]

        if recent_trend != 0:
            # Estimate when value will reach critical threshold
            current_value = data[-1]
            if recent_trend > 0:
                # Increasing trend
                critical_value = mapping.critical_thresholds['high']
                if current_value < critical_value:
                    hours_to_critical = (critical_value - current_value) / recent_trend
                    return max(1.0, min(8760.0, hours_to_critical))  # Between 1 hour and 1 year
            else:
                # Decreasing trend
                critical_value = mapping.critical_thresholds['low']
                if current_value > critical_value:
                    hours_to_critical = (current_value - critical_value) / abs(recent_trend)
                    return max(1.0, min(8760.0, hours_to_critical))

        return None

    def _identify_affected_subsystems(self, mapping: SensorMapping, failure_modes: List[FailureMode]) -> List[str]:
        """Identify subsystems that could be affected by the failure"""
        affected = [mapping.subsystem]

        # Cross-subsystem impact analysis
        impact_map = {
            'power_system': ['thermal_system', 'communication', 'science_instruments'],
            'thermal_system': ['power_system', 'science_instruments'],
            'communication': ['navigation', 'science_instruments'],
            'mobility': ['navigation', 'robotic_arm'],
            'robotic_arm': ['science_instruments'],
            'science_instruments': [],  # Generally isolated
        }

        if mapping.subsystem in impact_map:
            # High severity failures can cascade
            if any(mode in [FailureMode.OVERHEATING, FailureMode.VOLTAGE_FLUCTUATION,
                          FailureMode.CURRENT_SPIKE] for mode in failure_modes):
                affected.extend(impact_map[mapping.subsystem])

        return list(set(affected))

    def _enhance_with_correlation_analysis(self, classifications: Dict[str, FailureClassification],
                                         sensor_data: Dict[str, float],
                                         anomaly_scores: Dict[str, float]):
        """Enhance classifications with cross-sensor correlation analysis"""
        # Group classifications by subsystem
        subsystem_failures = {}
        for sensor_id, classification in classifications.items():
            mapping = self.sensor_mappings[sensor_id]
            subsystem = mapping.subsystem
            if subsystem not in subsystem_failures:
                subsystem_failures[subsystem] = []
            subsystem_failures[subsystem].append((sensor_id, classification))

        # Analyze patterns within each subsystem
        for subsystem, failures in subsystem_failures.items():
            if len(failures) > 1:
                self._analyze_subsystem_failure_patterns(failures, sensor_data, anomaly_scores)

    def _analyze_subsystem_failure_patterns(self, failures: List[Tuple[str, FailureClassification]],
                                          sensor_data: Dict[str, float],
                                          anomaly_scores: Dict[str, float]):
        """Analyze failure patterns within a subsystem"""
        # Check for systematic failures
        common_modes = {}
        for sensor_id, classification in failures:
            for mode in classification.failure_modes:
                if mode not in common_modes:
                    common_modes[mode] = 0
                common_modes[mode] += 1

        # If multiple sensors show the same failure mode, increase confidence
        for sensor_id, classification in failures:
            for i, mode in enumerate(classification.failure_modes):
                if common_modes[mode] > 1:
                    # Boost confidence for correlated failures
                    classification.confidence = min(1.0, classification.confidence * 1.2)
                    # Update root cause to indicate systematic issue
                    if "systematic" not in classification.root_cause:
                        classification.root_cause = f"systematic_{classification.root_cause}"

    def train_model(self, training_data: pd.DataFrame, labels: pd.DataFrame):
        """Train the ML classification model"""
        logger.info("Training failure classification model...")

        # Prepare features
        feature_columns = [col for col in training_data.columns if col != 'sensor_id']
        X = training_data[feature_columns].values
        y = labels.values

        # Scale features
        X_scaled = self.scalers['primary'].fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        self.models['primary'].fit(X_train, y_train)

        # Evaluate
        y_pred = self.models['primary'].predict(X_test)

        logger.info("Model training completed")
        logger.info(f"Test set accuracy: {np.mean(y_pred == y_test):.3f}")

        # Save model
        model_path = Path(self.config.get('model_save_path', 'data/models'))
        joblib.dump(self.models['primary'], model_path / 'failure_classification_model.pkl')
        joblib.dump(self.scalers['primary'], model_path / 'failure_classification_scaler.pkl')

    def get_sensor_health_summary(self) -> Dict[str, Dict]:
        """Get health summary for all sensors"""
        summary = {}

        for sensor_id, mapping in self.sensor_mappings.items():
            summary[sensor_id] = {
                'equipment_type': mapping.equipment_type,
                'subsystem': mapping.subsystem,
                'location': mapping.location,
                'measurement_type': mapping.measurement_type,
                'normal_range': mapping.normal_range,
                'related_sensors': mapping.related_sensors[:3]  # Top 3 related sensors
            }

        return summary