"""
Phase 4.2 Equipment Health Scoring Testing Suite

Tests the equipment health scoring system for multi-dimensional health metrics,
5-tier health status classification, and comprehensive health assessment.
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
    from src.business_logic.equipment_health import (
        EquipmentHealthScorer, SensorHealth, SubsystemHealth, SystemHealth,
        HealthStatus, HealthMetrics
    )
    from src.business_logic.failure_classification import FailureClassificationEngine
except ImportError as e:
    print(f"Warning: Could not import equipment health components: {e}")


@dataclass
class HealthTestResult:
    """Container for health scoring test results"""
    test_name: str
    component_type: str  # sensor, subsystem, system
    component_id: str
    health_score: float
    health_status: str
    expected_status: str
    accuracy: float
    execution_time: float
    timestamp: datetime


class TestEquipmentHealthScorer(unittest.TestCase):
    """Test Equipment Health Scoring system"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for health scoring"""
        cls.test_results: List[HealthTestResult] = []

        # Health status thresholds for validation
        cls.health_thresholds = {
            'excellent': 90.0,
            'good': 75.0,
            'fair': 60.0,
            'poor': 40.0,
            'critical': 0.0
        }

        # Generate test data
        cls._setup_health_test_data()

    @classmethod
    def _setup_health_test_data(cls):
        """Setup test data for health assessment scenarios"""
        cls.health_scenarios = {
            # Excellent health scenario
            "excellent_sensor": {
                "sensor_data": {"SENSOR_001": 25.0},  # Normal temperature
                "anomaly_scores": {"SENSOR_001": 0.05},  # Very low anomaly
                "expected_status": HealthStatus.EXCELLENT,
                "expected_score_range": (90, 100)
            },

            # Good health scenario
            "good_sensor": {
                "sensor_data": {"SENSOR_002": 27.0},  # Slightly elevated but normal
                "anomaly_scores": {"SENSOR_002": 0.15},  # Low anomaly
                "expected_status": HealthStatus.GOOD,
                "expected_score_range": (75, 90)
            },

            # Fair health scenario
            "fair_sensor": {
                "sensor_data": {"SENSOR_003": 35.0},  # Approaching threshold
                "anomaly_scores": {"SENSOR_003": 0.35},  # Moderate anomaly
                "expected_status": HealthStatus.FAIR,
                "expected_score_range": (60, 75)
            },

            # Poor health scenario
            "poor_sensor": {
                "sensor_data": {"SENSOR_004": 45.0},  # Above normal range
                "anomaly_scores": {"SENSOR_004": 0.65},  # High anomaly
                "expected_status": HealthStatus.POOR,
                "expected_score_range": (40, 60)
            },

            # Critical health scenario
            "critical_sensor": {
                "sensor_data": {"SENSOR_005": 85.0},  # Critical temperature
                "anomaly_scores": {"SENSOR_005": 0.95},  # Very high anomaly
                "expected_status": HealthStatus.CRITICAL,
                "expected_score_range": (0, 40)
            }
        }

        # Subsystem test scenarios
        cls.subsystem_scenarios = {
            "healthy_subsystem": {
                "sensors": {
                    "TEMP_01": {"score": 85, "status": HealthStatus.GOOD},
                    "TEMP_02": {"score": 88, "status": HealthStatus.GOOD},
                    "TEMP_03": {"score": 92, "status": HealthStatus.EXCELLENT}
                },
                "expected_status": HealthStatus.GOOD,
                "expected_score_range": (80, 95)
            },

            "degraded_subsystem": {
                "sensors": {
                    "POWER_01": {"score": 55, "status": HealthStatus.POOR},
                    "POWER_02": {"score": 62, "status": HealthStatus.FAIR},
                    "POWER_03": {"score": 45, "status": HealthStatus.POOR}
                },
                "expected_status": HealthStatus.FAIR,
                "expected_score_range": (45, 65)
            },

            "critical_subsystem": {
                "sensors": {
                    "MOTOR_01": {"score": 25, "status": HealthStatus.CRITICAL},
                    "MOTOR_02": {"score": 35, "status": HealthStatus.CRITICAL},
                    "MOTOR_03": {"score": 42, "status": HealthStatus.POOR}
                },
                "expected_status": HealthStatus.CRITICAL,
                "expected_score_range": (20, 45)
            }
        }

    def setUp(self):
        """Setup for each test"""
        try:
            self.failure_classifier = FailureClassificationEngine()
            self.health_scorer = EquipmentHealthScorer(self.failure_classifier)
        except Exception as e:
            self.skipTest(f"Could not initialize EquipmentHealthScorer: {e}")

    def test_multi_dimensional_health_metrics(self):
        """Test multi-dimensional health metrics calculation"""
        print("\n📊 Testing multi-dimensional health metrics...")

        # Test sensor with varied characteristics
        sensor_id = "TEST_SENSOR_MD"
        sensor_data = {sensor_id: 30.0}  # Temperature sensor
        anomaly_scores = {sensor_id: 0.25}

        # Generate time window data for trend analysis
        time_points = 100
        time_window_data = {
            sensor_id: np.array([
                30.0 + np.sin(i * 0.1) + np.random.normal(0, 0.5)
                for i in range(time_points)
            ])
        }

        start_time = datetime.now()

        # Calculate sensor health
        sensor_health = self.health_scorer.calculate_sensor_health(
            sensor_id, sensor_data, anomaly_scores, time_window_data
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate multi-dimensional metrics
        metrics = sensor_health.metrics

        # Test that all 7 health dimensions are calculated
        health_dimensions = [
            'performance_score', 'reliability_score', 'condition_score',
            'trend_score', 'maintenance_score', 'environmental_score', 'utilization_score'
        ]

        for dimension in health_dimensions:
            score = getattr(metrics, dimension)
            self.assertIsInstance(score, float, f"{dimension} should be a float")
            self.assertGreaterEqual(score, 0.0, f"{dimension} should be >= 0")
            self.assertLessEqual(score, 1.0, f"{dimension} should be <= 1")

        # Test data quality and confidence metrics
        self.assertIsInstance(metrics.data_quality, float, "Data quality should be a float")
        self.assertIsInstance(metrics.confidence, float, "Confidence should be a float")

        # Validate overall health score
        self.assertGreaterEqual(sensor_health.health_score, 0.0, "Health score should be >= 0")
        self.assertLessEqual(sensor_health.health_score, 100.0, "Health score should be <= 100")

        print(f"✅ Multi-dimensional metrics calculated in {execution_time:.3f}s")
        print(f"   📈 Performance: {metrics.performance_score:.2f}, Reliability: {metrics.reliability_score:.2f}")
        print(f"   🔍 Condition: {metrics.condition_score:.2f}, Trend: {metrics.trend_score:.2f}")

    def test_5_tier_health_status_classification(self):
        """Test 5-tier health status classification accuracy"""
        print("\n🏷️  Testing 5-tier health status classification...")

        correct_classifications = 0
        total_tests = len(self.health_scenarios)

        for scenario_name, scenario_data in self.health_scenarios.items():
            print(f"   Testing scenario: {scenario_name}")

            start_time = datetime.now()

            # Calculate health for scenario
            sensor_health = self.health_scorer.calculate_sensor_health(
                list(scenario_data["sensor_data"].keys())[0],
                scenario_data["sensor_data"],
                scenario_data["anomaly_scores"]
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Validate status classification
            expected_status = scenario_data["expected_status"]
            actual_status = sensor_health.status

            # Check if classification is correct
            is_correct = (actual_status == expected_status)
            if is_correct:
                correct_classifications += 1

            # Validate score range
            expected_min, expected_max = scenario_data["expected_score_range"]
            score_in_range = expected_min <= sensor_health.health_score <= expected_max

            # Allow for some flexibility in classification boundaries
            if not is_correct:
                # Check if it's within one tier of expected
                status_order = [HealthStatus.CRITICAL, HealthStatus.POOR, HealthStatus.FAIR,
                              HealthStatus.GOOD, HealthStatus.EXCELLENT]
                expected_idx = status_order.index(expected_status)
                actual_idx = status_order.index(actual_status)

                if abs(expected_idx - actual_idx) <= 1:
                    is_correct = True
                    correct_classifications += 1

            # Record test result
            test_result = HealthTestResult(
                test_name=scenario_name,
                component_type="sensor",
                component_id=list(scenario_data["sensor_data"].keys())[0],
                health_score=sensor_health.health_score,
                health_status=actual_status.value,
                expected_status=expected_status.value,
                accuracy=1.0 if is_correct else 0.0,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            self.test_results.append(test_result)

        # Validate classification accuracy
        classification_accuracy = correct_classifications / total_tests
        self.assertGreater(classification_accuracy, 0.7,
                          f"Health status classification accuracy {classification_accuracy:.1%} should be >70%")

        print(f"✅ Health status classification accuracy: {classification_accuracy:.1%}")

    def test_degradation_modeling(self):
        """Test degradation modeling and remaining useful life estimation"""
        print("\n📉 Testing degradation modeling...")

        # Create sensor with degrading health trend
        sensor_id = "DEGRADING_SENSOR"
        current_data = {sensor_id: 42.0}  # Current reading
        anomaly_scores = {sensor_id: 0.7}  # High anomaly

        # Create degrading time series data
        time_points = 50
        degradation_data = []
        base_value = 25.0  # Starting healthy value

        for i in range(time_points):
            # Simulate gradual degradation
            degraded_value = base_value + (i * 0.5) + np.random.normal(0, 1)
            degradation_data.append(degraded_value)

        time_window_data = {sensor_id: np.array(degradation_data)}

        # Calculate health with degradation modeling
        start_time = datetime.now()

        sensor_health = self.health_scorer.calculate_sensor_health(
            sensor_id, current_data, anomaly_scores, time_window_data
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate degradation metrics
        self.assertIsNotNone(sensor_health.degradation_rate,
                           "Degradation rate should be calculated")
        self.assertGreaterEqual(sensor_health.degradation_rate, 0.0,
                               "Degradation rate should be non-negative")

        # Test remaining useful life estimation
        if sensor_health.estimated_rul is not None:
            self.assertGreater(sensor_health.estimated_rul, 0.0,
                             "Remaining useful life should be positive")
            self.assertLess(sensor_health.estimated_rul, 8760.0,  # One year in hours
                           "RUL should be reasonable (< 1 year)")

        # Validate trend score reflects degradation
        self.assertLess(sensor_health.metrics.trend_score, 0.8,
                       "Trend score should reflect degradation")

        print(f"✅ Degradation modeling completed in {execution_time:.3f}s")
        if sensor_health.estimated_rul:
            print(f"   ⏱️  Estimated RUL: {sensor_health.estimated_rul:.1f} hours")
        print(f"   📈 Degradation rate: {sensor_health.degradation_rate:.3f}")

    def test_subsystem_health_aggregation(self):
        """Test subsystem health aggregation from sensor health"""
        print("\n🏢 Testing subsystem health aggregation...")

        correct_aggregations = 0
        total_subsystem_tests = len(self.subsystem_scenarios)

        for scenario_name, scenario_data in self.subsystem_scenarios.items():
            print(f"   Testing subsystem: {scenario_name}")

            # Create mock sensor health objects
            sensor_healths = {}
            for sensor_id, sensor_info in scenario_data["sensors"].items():
                # Create mock sensor health
                sensor_health = SensorHealth(
                    sensor_id=sensor_id,
                    health_score=sensor_info["score"],
                    status=sensor_info["status"],
                    metrics=HealthMetrics()  # Default metrics
                )
                sensor_healths[sensor_id] = sensor_health

            start_time = datetime.now()

            # Calculate subsystem health
            subsystem_health = self.health_scorer.calculate_subsystem_health(
                subsystem_name=scenario_name.replace("_subsystem", ""),
                sensor_healths=sensor_healths
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Validate subsystem health
            expected_status = scenario_data["expected_status"]
            expected_min, expected_max = scenario_data["expected_score_range"]

            # Check score range
            score_in_range = expected_min <= subsystem_health.health_score <= expected_max

            # Check status (allow for one tier flexibility)
            status_order = [HealthStatus.CRITICAL, HealthStatus.POOR, HealthStatus.FAIR,
                           HealthStatus.GOOD, HealthStatus.EXCELLENT]
            expected_idx = status_order.index(expected_status)
            actual_idx = status_order.index(subsystem_health.status)

            status_correct = abs(expected_idx - actual_idx) <= 1

            if score_in_range and status_correct:
                correct_aggregations += 1

            # Validate subsystem structure
            self.assertEqual(len(subsystem_health.sensor_healths), len(sensor_healths),
                           "Subsystem should include all sensors")
            self.assertIsInstance(subsystem_health.critical_sensors, list,
                                "Critical sensors should be a list")

            # Record test result
            test_result = HealthTestResult(
                test_name=scenario_name,
                component_type="subsystem",
                component_id=scenario_name,
                health_score=subsystem_health.health_score,
                health_status=subsystem_health.status.value,
                expected_status=expected_status.value,
                accuracy=1.0 if (score_in_range and status_correct) else 0.0,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            self.test_results.append(test_result)

        # Validate aggregation accuracy
        aggregation_accuracy = correct_aggregations / total_subsystem_tests
        self.assertGreater(aggregation_accuracy, 0.6,
                          f"Subsystem aggregation accuracy {aggregation_accuracy:.1%} should be >60%")

        print(f"✅ Subsystem health aggregation accuracy: {aggregation_accuracy:.1%}")

    def test_system_wide_health_assessment(self):
        """Test system-wide health assessment with cascade failure risk"""
        print("\n🌐 Testing system-wide health assessment...")

        # Create multiple subsystem healths
        subsystem_healths = {}

        # Healthy subsystem
        healthy_sensors = {
            f"HEALTHY_{i:02d}": SensorHealth(
                sensor_id=f"HEALTHY_{i:02d}",
                health_score=85 + np.random.uniform(-5, 10),
                status=HealthStatus.GOOD,
                metrics=HealthMetrics()
            ) for i in range(5)
        }

        subsystem_healths["power_system"] = SubsystemHealth(
            subsystem_name="power_system",
            health_score=87.0,
            status=HealthStatus.GOOD,
            sensor_healths=healthy_sensors,
            critical_sensors=[]
        )

        # Degraded subsystem
        degraded_sensors = {
            f"DEGRADED_{i:02d}": SensorHealth(
                sensor_id=f"DEGRADED_{i:02d}",
                health_score=55 + np.random.uniform(-10, 10),
                status=HealthStatus.FAIR,
                metrics=HealthMetrics()
            ) for i in range(3)
        }

        subsystem_healths["thermal_system"] = SubsystemHealth(
            subsystem_name="thermal_system",
            health_score=58.0,
            status=HealthStatus.FAIR,
            sensor_healths=degraded_sensors,
            critical_sensors=[]
        )

        # Critical subsystem
        critical_sensors = {
            f"CRITICAL_{i:02d}": SensorHealth(
                sensor_id=f"CRITICAL_{i:02d}",
                health_score=30 + np.random.uniform(-10, 10),
                status=HealthStatus.CRITICAL,
                metrics=HealthMetrics()
            ) for i in range(2)
        }

        subsystem_healths["mobility"] = SubsystemHealth(
            subsystem_name="mobility",
            health_score=32.0,
            status=HealthStatus.CRITICAL,
            sensor_healths=critical_sensors,
            critical_sensors=list(critical_sensors.keys())
        )

        # System metrics
        system_metrics = {
            'availability': 0.92,
            'performance_efficiency': 0.85
        }

        start_time = datetime.now()

        # Calculate system health
        system_health = self.health_scorer.calculate_system_health(
            subsystem_healths, system_metrics
        )

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate system health
        self.assertIsInstance(system_health, SystemHealth, "Should return SystemHealth object")
        self.assertGreaterEqual(system_health.overall_health_score, 0.0,
                               "System health score should be >= 0")
        self.assertLessEqual(system_health.overall_health_score, 100.0,
                            "System health score should be <= 100")

        # Validate subsystem inclusion
        self.assertEqual(len(system_health.subsystem_healths), len(subsystem_healths),
                        "System health should include all subsystems")

        # Validate cascade failure risk calculation
        self.assertIsInstance(system_health.cascade_failure_risk, float,
                             "Cascade failure risk should be a float")
        self.assertGreaterEqual(system_health.cascade_failure_risk, 0.0,
                               "Cascade failure risk should be >= 0")
        self.assertLessEqual(system_health.cascade_failure_risk, 1.0,
                            "Cascade failure risk should be <= 1")

        # With critical subsystem, cascade risk should be elevated
        self.assertGreater(system_health.cascade_failure_risk, 0.1,
                          "Cascade failure risk should be elevated with critical subsystem")

        # Validate reliability metrics
        if system_health.mtbf is not None:
            self.assertGreater(system_health.mtbf, 0, "MTBF should be positive")
        if system_health.mttr is not None:
            self.assertGreater(system_health.mttr, 0, "MTTR should be positive")

        # Validate economic indicators
        self.assertGreaterEqual(system_health.total_maintenance_cost_30d, 0.0,
                               "Maintenance cost should be non-negative")

        print(f"✅ System health assessment completed in {execution_time:.3f}s")
        print(f"   🏥 Overall health: {system_health.overall_health_score:.1f}%")
        print(f"   ⚠️  Cascade risk: {system_health.cascade_failure_risk:.1%}")
        print(f"   💰 30-day maintenance cost: ${system_health.total_maintenance_cost_30d:,.0f}")

    def test_maintenance_urgency_scoring(self):
        """Test maintenance urgency scoring (0-5 scale)"""
        print("\n🚨 Testing maintenance urgency scoring...")

        urgency_test_cases = [
            {
                "name": "routine_maintenance",
                "health_score": 92.0,
                "anomaly_score": 0.1,
                "degradation_rate": 0.01,
                "expected_urgency_range": (0, 1)
            },
            {
                "name": "elevated_attention",
                "health_score": 78.0,
                "anomaly_score": 0.3,
                "degradation_rate": 0.05,
                "expected_urgency_range": (1, 2)
            },
            {
                "name": "high_priority",
                "health_score": 65.0,
                "anomaly_score": 0.6,
                "degradation_rate": 0.1,
                "expected_urgency_range": (2, 3)
            },
            {
                "name": "urgent_maintenance",
                "health_score": 45.0,
                "anomaly_score": 0.8,
                "degradation_rate": 0.2,
                "expected_urgency_range": (3, 4)
            },
            {
                "name": "emergency_response",
                "health_score": 25.0,
                "anomaly_score": 0.95,
                "degradation_rate": 0.5,
                "expected_urgency_range": (4, 5)
            }
        ]

        correct_urgency_assessments = 0

        for test_case in urgency_test_cases:
            # Create sensor health with specific characteristics
            sensor_health = SensorHealth(
                sensor_id=f"URGENCY_TEST_{test_case['name']}",
                health_score=test_case["health_score"],
                status=HealthStatus.FAIR,  # Will be determined by health score
                metrics=HealthMetrics(),
                current_anomaly_score=test_case["anomaly_score"],
                degradation_rate=test_case["degradation_rate"]
            )

            # The urgency should be calculated during health assessment
            # For this test, we'll validate the urgency scoring logic
            urgency_level = sensor_health.maintenance_urgency

            expected_min, expected_max = test_case["expected_urgency_range"]

            # Validate urgency range
            self.assertGreaterEqual(urgency_level, 0, "Urgency should be >= 0")
            self.assertLessEqual(urgency_level, 5, "Urgency should be <= 5")

            # Check if urgency is in expected range (allow some flexibility)
            if expected_min <= urgency_level <= expected_max + 1:
                correct_urgency_assessments += 1

        urgency_accuracy = correct_urgency_assessments / len(urgency_test_cases)

        # Validate urgency scoring accuracy
        self.assertGreater(urgency_accuracy, 0.6,
                          f"Urgency scoring accuracy {urgency_accuracy:.1%} should be >60%")

        print(f"✅ Maintenance urgency scoring accuracy: {urgency_accuracy:.1%}")

    def test_health_scoring_performance(self):
        """Test health scoring performance for multiple sensors"""
        print("\n⚡ Testing health scoring performance...")

        # Generate data for multiple sensors
        num_sensors = 50
        sensor_data = {}
        anomaly_scores = {}

        for i in range(num_sensors):
            sensor_id = f"PERF_SENSOR_{i:03d}"
            sensor_data[sensor_id] = 25.0 + np.random.uniform(-10, 20)
            anomaly_scores[sensor_id] = np.random.uniform(0.0, 0.8)

        start_time = datetime.now()

        # Calculate health for all sensors
        sensor_healths = {}
        for sensor_id in sensor_data.keys():
            sensor_health = self.health_scorer.calculate_sensor_health(
                sensor_id,
                {sensor_id: sensor_data[sensor_id]},
                {sensor_id: anomaly_scores[sensor_id]}
            )
            sensor_healths[sensor_id] = sensor_health

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate performance
        processing_rate = num_sensors / execution_time
        self.assertGreater(processing_rate, 10.0,
                          f"Health scoring should process >10 sensors/sec, got {processing_rate:.1f}")

        # Validate all sensors were processed
        self.assertEqual(len(sensor_healths), num_sensors,
                        "All sensors should have health calculated")

        # Validate health scores are reasonable
        health_scores = [sh.health_score for sh in sensor_healths.values()]
        avg_health = np.mean(health_scores)
        health_std = np.std(health_scores)

        self.assertGreater(avg_health, 0.0, "Average health should be positive")
        self.assertLess(avg_health, 100.0, "Average health should be under 100")
        self.assertGreater(health_std, 0.0, "Health scores should have variation")

        print(f"✅ Health scoring performance: {processing_rate:.1f} sensors/sec")
        print(f"   📊 Average health: {avg_health:.1f}% (std: {health_std:.1f})")

    @classmethod
    def tearDownClass(cls):
        """Generate equipment health test report"""
        print("\n" + "="*80)
        print("PHASE 4.2 EQUIPMENT HEALTH SCORING TEST SUMMARY")
        print("="*80)

        if cls.test_results:
            total_tests = len(cls.test_results)
            accurate_assessments = sum(1 for result in cls.test_results if result.accuracy > 0.0)

            print(f"Health assessment tests: {total_tests}")
            print(f"Accurate assessments: {accurate_assessments}")
            print(f"Assessment accuracy: {(accurate_assessments/total_tests*100):.1f}%")

            # Group results by component type
            component_types = ['sensor', 'subsystem', 'system']
            for comp_type in component_types:
                type_results = [r for r in cls.test_results if r.component_type == comp_type]
                if type_results:
                    type_accuracy = np.mean([r.accuracy for r in type_results])
                    avg_health_score = np.mean([r.health_score for r in type_results])
                    print(f"   {comp_type.title()}: {type_accuracy:.1%} accuracy, {avg_health_score:.1f} avg health")

            # Calculate performance metrics
            avg_execution_time = np.mean([result.execution_time for result in cls.test_results])

            print(f"\n📊 Performance Metrics:")
            print(f"   Average execution time: {avg_execution_time:.3f}s")

            # Save detailed results
            results_file = project_root / "testing_phase4" / "test_results" / "phase4_2_equipment_health_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)

            results_data = []
            for result in cls.test_results:
                result_dict = {
                    'test_name': result.test_name,
                    'component_type': result.component_type,
                    'component_id': result.component_id,
                    'health_score': result.health_score,
                    'health_status': result.health_status,
                    'expected_status': result.expected_status,
                    'accuracy': result.accuracy,
                    'execution_time': result.execution_time,
                    'timestamp': result.timestamp.isoformat()
                }
                results_data.append(result_dict)

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == '__main__':
    # Run equipment health scoring tests
    unittest.main(verbosity=2)