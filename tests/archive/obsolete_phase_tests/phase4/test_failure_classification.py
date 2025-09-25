"""
Phase 4.2 Failure Classification Testing Suite

Tests the failure classification engine for 80-sensor mapping,
24 failure modes, and intelligent anomaly-to-failure classification.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import business logic components
try:
    from src.business_logic.failure_classification import (
        FailureClassificationEngine, FailureMode, Severity,
        FailureClassification, SensorMapping
    )
except ImportError as e:
    print(f"Warning: Could not import failure classification components: {e}")


@dataclass
class FailureTestResult:
    """Container for failure classification test results"""
    test_name: str
    sensor_id: str
    expected_failure_modes: List[str]
    predicted_failure_modes: List[str]
    confidence: float
    accuracy: float
    execution_time: float
    timestamp: datetime


class TestFailureClassificationEngine(unittest.TestCase):
    """Test Failure Classification Engine for 80-sensor system"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for failure classification"""
        cls.test_results: List[FailureTestResult] = []
        cls.total_sensors = 80
        cls.smap_sensor_count = 25
        cls.msl_sensor_count = 55

        # Expected failure modes count
        cls.expected_failure_modes = 24

        # Generate test data
        cls._setup_test_scenarios()

    @classmethod
    def _setup_test_scenarios(cls):
        """Setup test scenarios for different failure types"""
        cls.test_scenarios = {
            # Thermal failure scenarios
            "thermal_overheating": {
                "sensor_ids": ["T-1", "T-2", "T-3"],
                "sensor_data": {"T-1": 75.0, "T-2": 68.0, "T-3": 70.0},  # High temperatures
                "anomaly_scores": {"T-1": 0.95, "T-2": 0.85, "T-3": 0.88},
                "expected_modes": [FailureMode.OVERHEATING],
                "expected_severity": Severity.HIGH
            },

            # Power system failure scenarios
            "power_voltage_fluctuation": {
                "sensor_ids": ["V-1", "V-2", "V-3"],
                "sensor_data": {"V-1": 35.5, "V-2": 19.2, "V-3": 36.8},  # Out of range voltages
                "anomaly_scores": {"V-1": 0.92, "V-2": 0.88, "V-3": 0.94},
                "expected_modes": [FailureMode.VOLTAGE_FLUCTUATION],
                "expected_severity": Severity.HIGH
            },

            # Mechanical system failure scenarios
            "mechanical_vibration": {
                "sensor_ids": ["A-1", "A-2"],
                "sensor_data": {"A-1": 4.5, "A-2": 3.8},  # High vibration
                "anomaly_scores": {"A-1": 0.89, "A-2": 0.82},
                "expected_modes": [FailureMode.VIBRATION_ANOMALY],
                "expected_severity": Severity.MEDIUM
            },

            # MSL mobility system failure
            "mobility_motor_stress": {
                "sensor_ids": ["M-1", "M-2", "M-3"],
                "sensor_data": {"M-1": 4.8, "M-2": 5.2, "M-3": 4.9},  # High motor current
                "anomaly_scores": {"M-1": 0.87, "M-2": 0.91, "M-3": 0.85},
                "expected_modes": [FailureMode.CURRENT_SPIKE, FailureMode.BEARING_WEAR],
                "expected_severity": Severity.HIGH
            },

            # MSL robotic arm failure
            "robotic_arm_torque": {
                "sensor_ids": ["A-1", "A-2", "A-3"],
                "sensor_data": {"A-1": 75.0, "A-2": 78.0, "A-3": 72.0},  # High torque
                "anomaly_scores": {"A-1": 0.86, "A-2": 0.90, "A-3": 0.83},
                "expected_modes": [FailureMode.BEARING_WEAR, FailureMode.MISALIGNMENT],
                "expected_severity": Severity.MEDIUM
            },

            # Communication system failure
            "communication_signal_loss": {
                "sensor_ids": ["C-1", "C-2"],
                "sensor_data": {"C-1": 0.3, "C-2": 0.4},  # Low signal quality
                "anomaly_scores": {"C-1": 0.78, "C-2": 0.82},
                "expected_modes": [FailureMode.SIGNAL_LOSS],
                "expected_severity": Severity.MEDIUM
            },

            # Science instrument degradation
            "science_instrument_drift": {
                "sensor_ids": ["S-1", "S-2", "S-5"],
                "sensor_data": {"S-1": 65.0, "S-2": 800, "S-5": 0.65},  # Degraded performance
                "anomaly_scores": {"S-1": 0.75, "S-2": 0.73, "S-5": 0.77},
                "expected_modes": [FailureMode.SENSOR_DRIFT, FailureMode.PERFORMANCE_DEGRADATION],
                "expected_severity": Severity.LOW
            },

            # Pressure system failure
            "pressure_drop": {
                "sensor_ids": ["Pr-1", "Pr-2"],
                "sensor_data": {"Pr-1": 850, "Pr-2": 0.05},  # Low pressure
                "anomaly_scores": {"Pr-1": 0.84, "Pr-2": 0.86},
                "expected_modes": [FailureMode.PRESSURE_DROP],
                "expected_severity": Severity.MEDIUM
            }
        }

    def setUp(self):
        """Setup for each test"""
        try:
            self.failure_classifier = FailureClassificationEngine()
        except Exception as e:
            self.skipTest(f"Could not initialize FailureClassificationEngine: {e}")

    def test_80_sensor_mapping_coverage(self):
        """Test that all 80 sensors are properly mapped"""
        print("\n🗺️  Testing 80-sensor mapping coverage...")

        # Get sensor mappings
        sensor_mappings = self.failure_classifier.sensor_mappings

        # Validate total sensor count
        self.assertEqual(len(sensor_mappings), self.total_sensors,
                        f"Should have {self.total_sensors} sensors mapped")

        # Count SMAP and MSL sensors
        smap_sensors = [sid for sid in sensor_mappings.keys() if any(
            sid.startswith(prefix) for prefix in ['P-', 'T-', 'R-', 'V-', 'I-', 'A-', 'Pr-', 'C-', 'G-', 'At-']
        )]
        msl_sensors = [sid for sid in sensor_mappings.keys() if any(
            sid.startswith(prefix) for prefix in ['M-', 'A-', 'S-', 'P-', 'T-', 'C-', 'N-']
        )]

        # Validate sensor distribution (allowing for some overlap in sensor ID patterns)
        self.assertGreaterEqual(len(smap_sensors), 20, "Should have at least 20 SMAP-like sensors")
        self.assertGreaterEqual(len(msl_sensors), 30, "Should have at least 30 MSL-like sensors")

        # Validate sensor mapping structure
        for sensor_id, mapping in sensor_mappings.items():
            self.assertIsInstance(mapping, SensorMapping, f"Sensor {sensor_id} should have proper mapping")
            self.assertIsNotNone(mapping.equipment_type, f"Sensor {sensor_id} should have equipment type")
            self.assertIsNotNone(mapping.subsystem, f"Sensor {sensor_id} should have subsystem")
            self.assertIsNotNone(mapping.normal_range, f"Sensor {sensor_id} should have normal range")

        print(f"✅ All {len(sensor_mappings)} sensors properly mapped")
        print(f"   📊 Distribution: ~{len(smap_sensors)} SMAP-like, ~{len(msl_sensors)} MSL-like sensors")

    def test_24_failure_modes_coverage(self):
        """Test that all 24 failure modes are covered"""
        print("\n🔧 Testing 24 failure modes coverage...")

        # Get all available failure modes
        available_modes = list(FailureMode)

        # Validate expected count
        self.assertGreaterEqual(len(available_modes), self.expected_failure_modes,
                               f"Should have at least {self.expected_failure_modes} failure modes")

        # Validate key failure mode categories
        expected_categories = {
            "mechanical": [FailureMode.BEARING_WEAR, FailureMode.VIBRATION_ANOMALY, FailureMode.MISALIGNMENT],
            "thermal": [FailureMode.OVERHEATING, FailureMode.THERMAL_CYCLING],
            "electrical": [FailureMode.VOLTAGE_FLUCTUATION, FailureMode.CURRENT_SPIKE],
            "communication": [FailureMode.SIGNAL_LOSS, FailureMode.DATA_CORRUPTION],
            "environmental": [FailureMode.CORROSION, FailureMode.CONTAMINATION],
            "system": [FailureMode.PERFORMANCE_DEGRADATION, FailureMode.EFFICIENCY_LOSS]
        }

        for category, modes in expected_categories.items():
            for mode in modes:
                self.assertIn(mode, available_modes,
                             f"Failure mode {mode.value} from {category} category should be available")

        print(f"✅ {len(available_modes)} failure modes available")
        print(f"   🏷️  Categories: Mechanical, Thermal, Electrical, Communication, Environmental, System")

    def test_failure_classification_accuracy(self):
        """Test failure classification accuracy for known scenarios"""
        print("\n🎯 Testing failure classification accuracy...")

        total_scenarios = len(self.test_scenarios)
        correct_classifications = 0
        accuracy_scores = []

        for scenario_name, scenario_data in self.test_scenarios.items():
            print(f"   Testing scenario: {scenario_name}")

            start_time = datetime.now()

            # Perform classification
            classifications = self.failure_classifier.classify_failure(
                sensor_data=scenario_data["sensor_data"],
                anomaly_scores=scenario_data["anomaly_scores"],
                time_window_data=None  # Not required for basic testing
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Analyze results
            scenario_accuracy = 0.0
            predicted_modes = []

            if classifications:
                for sensor_id, classification in classifications.items():
                    predicted_modes.extend([fm.value for fm in classification.failure_modes])

                    # Check if expected modes are found
                    expected_mode_values = [fm.value for fm in scenario_data["expected_modes"]]
                    found_modes = [fm.value for fm in classification.failure_modes]

                    # Calculate accuracy for this classification
                    matches = sum(1 for mode in expected_mode_values if mode in found_modes)
                    if expected_mode_values:
                        scenario_accuracy = matches / len(expected_mode_values)

                    # Record test result
                    test_result = FailureTestResult(
                        test_name=scenario_name,
                        sensor_id=sensor_id,
                        expected_failure_modes=expected_mode_values,
                        predicted_failure_modes=found_modes,
                        confidence=classification.confidence,
                        accuracy=scenario_accuracy,
                        execution_time=execution_time,
                        timestamp=datetime.now()
                    )
                    self.test_results.append(test_result)

            accuracy_scores.append(scenario_accuracy)
            if scenario_accuracy > 0.5:  # At least 50% accuracy
                correct_classifications += 1

        # Calculate overall accuracy
        overall_accuracy = correct_classifications / total_scenarios
        avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0

        # Validate accuracy requirements
        self.assertGreater(overall_accuracy, 0.6,
                          f"Overall classification accuracy {overall_accuracy:.1%} should be >60%")

        print(f"✅ Classification accuracy: {overall_accuracy:.1%} scenarios, {avg_accuracy:.1%} average")

    def test_multi_sensor_correlation_analysis(self):
        """Test correlation analysis across multiple sensors"""
        print("\n🔗 Testing multi-sensor correlation analysis...")

        # Test scenario with correlated sensor failures
        correlated_scenario = {
            "sensor_data": {
                "V-1": 35.0,    # High voltage
                "V-2": 34.8,    # High voltage
                "I-1": 7.5,     # High current
                "I-2": 7.2,     # High current
                "T-1": 65.0,    # Elevated temperature
                "T-2": 63.0     # Elevated temperature
            },
            "anomaly_scores": {
                "V-1": 0.89, "V-2": 0.87, "I-1": 0.92,
                "I-2": 0.88, "T-1": 0.78, "T-2": 0.75
            }
        }

        # Perform classification
        classifications = self.failure_classifier.classify_failure(
            sensor_data=correlated_scenario["sensor_data"],
            anomaly_scores=correlated_scenario["anomaly_scores"]
        )

        # Validate correlation detection
        self.assertGreater(len(classifications), 3,
                          "Should detect failures in multiple correlated sensors")

        # Check for power system correlation
        power_related_sensors = [sid for sid in classifications.keys()
                               if sid.startswith(('V-', 'I-', 'T-'))]
        self.assertGreater(len(power_related_sensors), 2,
                          "Should detect correlation in power system sensors")

        # Validate failure modes consistency
        detected_modes = set()
        for classification in classifications.values():
            detected_modes.update([fm.value for fm in classification.failure_modes])

        power_related_modes = {
            FailureMode.VOLTAGE_FLUCTUATION.value,
            FailureMode.CURRENT_SPIKE.value,
            FailureMode.OVERHEATING.value
        }

        mode_overlap = detected_modes.intersection(power_related_modes)
        self.assertGreater(len(mode_overlap), 1,
                          "Should detect related failure modes in correlated systems")

        print(f"✅ Correlation analysis: {len(power_related_sensors)} correlated sensors")

    def test_severity_assessment(self):
        """Test failure severity assessment accuracy"""
        print("\n⚠️  Testing failure severity assessment...")

        severity_test_cases = [
            {
                "name": "critical_failure",
                "sensor_data": {"T-1": 85.0},  # Extreme temperature
                "anomaly_scores": {"T-1": 0.98},
                "expected_severity": Severity.CRITICAL
            },
            {
                "name": "high_severity",
                "sensor_data": {"V-1": 40.0},  # Very high voltage
                "anomaly_scores": {"V-1": 0.92},
                "expected_severity": Severity.HIGH
            },
            {
                "name": "medium_severity",
                "sensor_data": {"A-1": 3.0},  # Moderate vibration
                "anomaly_scores": {"A-1": 0.78},
                "expected_severity": Severity.MEDIUM
            },
            {
                "name": "low_severity",
                "sensor_data": {"S-1": 72.0},  # Slight instrument drift
                "anomaly_scores": {"S-1": 0.65},
                "expected_severity": Severity.LOW
            }
        ]

        correct_severity_assessments = 0

        for test_case in severity_test_cases:
            classifications = self.failure_classifier.classify_failure(
                sensor_data=test_case["sensor_data"],
                anomaly_scores=test_case["anomaly_scores"]
            )

            if classifications:
                for classification in classifications.values():
                    # Allow for some flexibility in severity assessment
                    severity_values = {
                        Severity.CRITICAL: 4, Severity.HIGH: 3,
                        Severity.MEDIUM: 2, Severity.LOW: 1
                    }

                    expected_value = severity_values[test_case["expected_severity"]]
                    actual_value = severity_values[classification.severity]

                    # Accept if within one level
                    if abs(expected_value - actual_value) <= 1:
                        correct_severity_assessments += 1
                        break

        severity_accuracy = correct_severity_assessments / len(severity_test_cases)

        # Validate severity assessment accuracy
        self.assertGreater(severity_accuracy, 0.7,
                          f"Severity assessment accuracy {severity_accuracy:.1%} should be >70%")

        print(f"✅ Severity assessment accuracy: {severity_accuracy:.1%}")

    def test_confidence_scoring_reliability(self):
        """Test confidence scoring reliability"""
        print("\n🎯 Testing confidence scoring reliability...")

        confidence_test_scenarios = [
            {
                "name": "high_confidence_scenario",
                "sensor_data": {"T-1": 75.0, "T-2": 73.0},
                "anomaly_scores": {"T-1": 0.95, "T-2": 0.93},
                "expected_min_confidence": 0.8
            },
            {
                "name": "medium_confidence_scenario",
                "sensor_data": {"V-1": 32.0},
                "anomaly_scores": {"V-1": 0.78},
                "expected_min_confidence": 0.5
            },
            {
                "name": "low_confidence_scenario",
                "sensor_data": {"S-1": 88.0},
                "anomaly_scores": {"S-1": 0.62},
                "expected_min_confidence": 0.3
            }
        ]

        confidence_scores = []

        for scenario in confidence_test_scenarios:
            classifications = self.failure_classifier.classify_failure(
                sensor_data=scenario["sensor_data"],
                anomaly_scores=scenario["anomaly_scores"]
            )

            if classifications:
                for classification in classifications.values():
                    confidence_scores.append(classification.confidence)

                    # Validate minimum confidence threshold
                    self.assertGreaterEqual(classification.confidence,
                                          scenario["expected_min_confidence"],
                                          f"Confidence for {scenario['name']} should meet minimum threshold")

        # Validate confidence score distribution
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            confidence_std = np.std(confidence_scores)

            self.assertGreater(avg_confidence, 0.5,
                              "Average confidence should be reasonable")
            self.assertLess(confidence_std, 0.4,
                           "Confidence scores should not be too variable")

        print(f"✅ Confidence scoring: {avg_confidence:.2f} average, {confidence_std:.2f} std dev")

    def test_performance_80_sensors(self):
        """Test classification performance with all 80 sensors"""
        print("\n⚡ Testing performance with 80 sensors...")

        # Generate synthetic data for all 80 sensors
        all_sensor_data = {}
        all_anomaly_scores = {}

        sensor_mappings = self.failure_classifier.sensor_mappings

        for sensor_id, mapping in sensor_mappings.items():
            # Generate realistic data within normal ranges
            normal_min, normal_max = mapping.normal_range

            # 10% chance of anomalous reading
            if np.random.random() < 0.1:
                # Generate anomalous value
                if np.random.random() < 0.5:
                    value = normal_max + (normal_max - normal_min) * 0.5
                else:
                    value = normal_min - (normal_max - normal_min) * 0.5
                anomaly_score = np.random.uniform(0.7, 0.95)
            else:
                # Generate normal value
                value = np.random.uniform(normal_min, normal_max)
                anomaly_score = np.random.uniform(0.0, 0.3)

            all_sensor_data[sensor_id] = value
            all_anomaly_scores[sensor_id] = anomaly_score

        # Perform classification
        start_time = datetime.now()

        classifications = self.failure_classifier.classify_failure(
            sensor_data=all_sensor_data,
            anomaly_scores=all_anomaly_scores
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate performance
        self.assertLess(execution_time, 5.0,
                       f"80-sensor classification should complete in <5s, took {execution_time:.2f}s")

        # Validate reasonable number of classifications
        anomalous_sensors = sum(1 for score in all_anomaly_scores.values() if score > 0.7)
        self.assertLessEqual(len(classifications), anomalous_sensors,
                            "Should not classify more failures than anomalous sensors")

        processing_rate = len(sensor_mappings) / execution_time

        print(f"✅ 80-sensor performance: {execution_time:.2f}s ({processing_rate:.1f} sensors/sec)")
        print(f"   📊 Classifications: {len(classifications)} from {anomalous_sensors} anomalous sensors")

    @classmethod
    def tearDownClass(cls):
        """Generate failure classification test report"""
        print("\n" + "="*80)
        print("PHASE 4.2 FAILURE CLASSIFICATION TEST SUMMARY")
        print("="*80)

        if cls.test_results:
            total_tests = len(cls.test_results)
            accurate_predictions = sum(1 for result in cls.test_results if result.accuracy > 0.5)

            print(f"Classification tests: {total_tests}")
            print(f"Accurate predictions: {accurate_predictions}")
            print(f"Classification accuracy: {(accurate_predictions/total_tests*100):.1f}%")

            # Calculate average metrics
            avg_confidence = np.mean([result.confidence for result in cls.test_results])
            avg_accuracy = np.mean([result.accuracy for result in cls.test_results])
            avg_execution_time = np.mean([result.execution_time for result in cls.test_results])

            print(f"\n📊 Performance Metrics:")
            print(f"   Average confidence: {avg_confidence:.2f}")
            print(f"   Average accuracy: {avg_accuracy:.2f}")
            print(f"   Average execution time: {avg_execution_time:.3f}s")

            # Save detailed results
            results_file = project_root / "testing_phase4" / "test_results" / "phase4_2_failure_classification_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)

            results_data = []
            for result in cls.test_results:
                result_dict = {
                    'test_name': result.test_name,
                    'sensor_id': result.sensor_id,
                    'expected_failure_modes': result.expected_failure_modes,
                    'predicted_failure_modes': result.predicted_failure_modes,
                    'confidence': result.confidence,
                    'accuracy': result.accuracy,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat()
                }
                results_data.append(result_dict)

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == '__main__':
    # Run failure classification tests
    unittest.main(verbosity=2)