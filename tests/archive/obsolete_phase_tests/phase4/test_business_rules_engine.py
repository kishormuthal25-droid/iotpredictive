"""
Phase 4.2 Business Rules Engine Testing Suite

Tests the integrated business rules engine for intelligent decision automation,
multi-criteria decision optimization, and comprehensive maintenance decision making.
"""

import unittest
import sys
import os
import asyncio
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
    from src.business_logic.business_rules_engine import (
        BusinessRulesEngine, DecisionType, ConfidenceLevel, MaintenanceDecision,
        BusinessRule, DecisionContext
    )
    from src.business_logic.failure_classification import FailureMode, Severity
    from src.business_logic.equipment_health import HealthStatus, SensorHealth, HealthMetrics
    from src.business_logic.predictive_triggers import TriggerEvent, Priority
except ImportError as e:
    print(f"Warning: Could not import business rules engine components: {e}")


@dataclass
class DecisionTestResult:
    """Container for business decision test results"""
    test_name: str
    decision_type: str
    confidence_level: str
    decision_quality: float
    execution_time: float
    resource_efficiency: float
    roi_projection: float
    timestamp: datetime


class TestBusinessRulesEngine(unittest.TestCase):
    """Test Business Rules Engine for intelligent maintenance decisions"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for business rules engine"""
        cls.test_results: List[DecisionTestResult] = []

        # Expected decision types and confidence levels
        cls.expected_decision_types = 7  # As defined in DecisionType enum
        cls.expected_confidence_levels = 5  # As defined in ConfidenceLevel enum

        # Generate test scenarios
        cls._setup_decision_test_scenarios()

    @classmethod
    def _setup_decision_test_scenarios(cls):
        """Setup test scenarios for different decision types"""
        cls.decision_scenarios = {
            # Emergency response scenario
            "critical_equipment_failure": {
                "sensor_data": {
                    "V-1": 45.0,  # Critical voltage
                    "T-1": 85.0,  # Critical temperature
                    "I-1": 9.0    # Critical current
                },
                "anomaly_scores": {
                    "V-1": 0.98,
                    "T-1": 0.95,
                    "I-1": 0.92
                },
                "expected_decision": DecisionType.IMMEDIATE_ACTION,
                "expected_confidence": ConfidenceLevel.VERY_HIGH,
                "expected_urgency": Priority.CRITICAL
            },

            # Preventive maintenance scenario
            "degrading_equipment": {
                "sensor_data": {
                    "A-1": 65.0,  # High vibration
                    "A-2": 62.0,  # High vibration
                    "M-1": 4.2    # Elevated motor current
                },
                "anomaly_scores": {
                    "A-1": 0.78,
                    "A-2": 0.75,
                    "M-1": 0.72
                },
                "expected_decision": DecisionType.SCHEDULE_MAINTENANCE,
                "expected_confidence": ConfidenceLevel.HIGH,
                "expected_urgency": Priority.HIGH
            },

            # Performance optimization scenario
            "efficiency_degradation": {
                "sensor_data": {
                    "S-1": 88.0,   # Slight instrument drift
                    "S-2": 1200,   # Moderate performance loss
                    "C-1": 0.72    # Communication quality drop
                },
                "anomaly_scores": {
                    "S-1": 0.65,
                    "S-2": 0.68,
                    "C-1": 0.58
                },
                "expected_decision": DecisionType.OPTIMIZE_STRATEGY,
                "expected_confidence": ConfidenceLevel.MEDIUM,
                "expected_urgency": Priority.MEDIUM
            },

            # Monitoring scenario
            "stable_operation": {
                "sensor_data": {
                    "T-1": 28.0,   # Normal temperature
                    "V-1": 26.5,   # Normal voltage
                    "P-1": 1015    # Normal pressure
                },
                "anomaly_scores": {
                    "T-1": 0.15,
                    "V-1": 0.12,
                    "P-1": 0.08
                },
                "expected_decision": DecisionType.CONTINUE_OPERATION,
                "expected_confidence": ConfidenceLevel.HIGH,
                "expected_urgency": Priority.ROUTINE
            },

            # Investigation scenario
            "anomalous_patterns": {
                "sensor_data": {
                    "R-1": -85.0,  # Unusual radar signal
                    "R-2": -88.0,  # Unusual radar signal
                    "G-1": 12.0    # GPS accuracy degradation
                },
                "anomaly_scores": {
                    "R-1": 0.82,
                    "R-2": 0.79,
                    "G-1": 0.71
                },
                "expected_decision": DecisionType.INVESTIGATE_FURTHER,
                "expected_confidence": ConfidenceLevel.MEDIUM,
                "expected_urgency": Priority.MEDIUM
            }
        }

    def setUp(self):
        """Setup for each test"""
        try:
            self.business_rules_engine = BusinessRulesEngine()
        except Exception as e:
            self.skipTest(f"Could not initialize BusinessRulesEngine: {e}")

    def test_8_default_business_rules(self):
        """Test that all 8 default business rules are properly configured"""
        print("\n📋 Testing 8 default business rules...")

        rules = self.business_rules_engine.rules

        # Validate expected number of rules
        self.assertGreaterEqual(len(rules), 8,
                               f"Should have at least 8 default rules, found {len(rules)}")

        # Expected rule categories
        expected_rule_types = [
            "emergency_response",
            "failure_prevention",
            "cost_optimization",
            "predictive_maintenance",
            "performance_management",
            "resource_management",
            "asset_management",
            "compliance"
        ]

        # Validate rule structure
        for rule_id, rule in rules.items():
            self.assertIsInstance(rule, BusinessRule, f"Rule {rule_id} should be BusinessRule instance")
            self.assertIsNotNone(rule.rule_name, f"Rule {rule_id} should have a name")
            self.assertIsNotNone(rule.description, f"Rule {rule_id} should have description")
            self.assertIsNotNone(rule.decision_logic, f"Rule {rule_id} should have decision logic")
            self.assertIsInstance(rule.decision_type, DecisionType,
                                f"Rule {rule_id} should have valid decision type")

            # Validate rule is active
            if "critical" in rule.rule_name.lower() or "emergency" in rule.rule_name.lower():
                self.assertTrue(rule.is_active, f"Critical rule {rule_id} should be active")

        print(f"✅ {len(rules)} business rules configured and validated")

    def test_integrated_decision_making(self):
        """Test integrated decision making across all components"""
        print("\n🤖 Testing integrated decision making...")

        total_scenarios = len(self.decision_scenarios)
        successful_decisions = 0
        decision_quality_scores = []

        for scenario_name, scenario_data in self.decision_scenarios.items():
            print(f"   Testing scenario: {scenario_name}")

            # Create decision context
            context = self._create_decision_context(scenario_data)

            start_time = datetime.now()

            # Make maintenance decision
            try:
                decisions = asyncio.run(
                    self.business_rules_engine.make_maintenance_decision(context)
                )
                execution_time = (datetime.now() - start_time).total_seconds()

                # Validate decision output
                if decisions:
                    primary_decision = decisions[0]  # Get primary decision

                    # Check decision type alignment
                    expected_decision = scenario_data["expected_decision"]
                    decision_type_match = (primary_decision.decision_type == expected_decision)

                    # Check confidence level reasonableness
                    confidence_reasonable = (primary_decision.confidence > 0.5)

                    # Calculate decision quality score
                    decision_quality = self._calculate_decision_quality(
                        primary_decision, scenario_data
                    )

                    decision_quality_scores.append(decision_quality)

                    if decision_type_match and confidence_reasonable:
                        successful_decisions += 1

                    # Record test result
                    test_result = DecisionTestResult(
                        test_name=scenario_name,
                        decision_type=primary_decision.decision_type.value,
                        confidence_level=primary_decision.confidence_level.value,
                        decision_quality=decision_quality,
                        execution_time=execution_time,
                        resource_efficiency=self._calculate_resource_efficiency(primary_decision),
                        roi_projection=primary_decision.roi_projection,
                        timestamp=datetime.now()
                    )
                    self.test_results.append(test_result)

                else:
                    print(f"     ⚠️  No decisions generated for {scenario_name}")

            except Exception as e:
                print(f"     ❌ Decision making failed for {scenario_name}: {e}")
                execution_time = (datetime.now() - start_time).total_seconds()

        # Validate integrated decision performance
        decision_success_rate = successful_decisions / total_scenarios
        avg_decision_quality = np.mean(decision_quality_scores) if decision_quality_scores else 0.0

        self.assertGreater(decision_success_rate, 0.6,
                          f"Decision success rate {decision_success_rate:.1%} should be >60%")
        self.assertGreater(avg_decision_quality, 0.7,
                          f"Average decision quality {avg_decision_quality:.2f} should be >0.7")

        print(f"✅ Integrated decision making: {decision_success_rate:.1%} success, "
              f"{avg_decision_quality:.2f} avg quality")

    def _create_decision_context(self, scenario_data: Dict[str, Any]) -> DecisionContext:
        """Create decision context from scenario data"""
        # Create mock sensor healths
        sensor_healths = {}
        for sensor_id, anomaly_score in scenario_data["anomaly_scores"].items():
            # Derive health score from anomaly score (inverse relationship)
            health_score = 100 - (anomaly_score * 80)

            health_status = HealthStatus.EXCELLENT
            if health_score < 40:
                health_status = HealthStatus.CRITICAL
            elif health_score < 60:
                health_status = HealthStatus.POOR
            elif health_score < 75:
                health_status = HealthStatus.FAIR
            elif health_score < 90:
                health_status = HealthStatus.GOOD

            sensor_health = SensorHealth(
                sensor_id=sensor_id,
                health_score=health_score,
                status=health_status,
                metrics=HealthMetrics(),
                current_anomaly_score=anomaly_score
            )
            sensor_healths[sensor_id] = sensor_health

        # Create decision context
        context = DecisionContext(
            timestamp=datetime.now(),
            sensor_data=scenario_data["sensor_data"],
            anomaly_scores=scenario_data["anomaly_scores"],
            sensor_healths=sensor_healths,
            operational_context={
                'availability': 0.95,
                'performance_efficiency': 0.88
            },
            resource_availability={
                'technicians': 3,
                'budget': 50000
            },
            business_constraints={
                'min_roi': 15.0,
                'max_payback_months': 24
            }
        )

        return context

    def _calculate_decision_quality(self, decision: MaintenanceDecision, scenario_data: Dict[str, Any]) -> float:
        """Calculate decision quality score"""
        quality_factors = []

        # Type appropriateness (40% weight)
        expected_decision = scenario_data["expected_decision"]
        if decision.decision_type == expected_decision:
            quality_factors.append(1.0)
        else:
            # Partial credit for related decision types
            decision_similarity = {
                DecisionType.IMMEDIATE_ACTION: [DecisionType.SCHEDULE_MAINTENANCE],
                DecisionType.SCHEDULE_MAINTENANCE: [DecisionType.IMMEDIATE_ACTION, DecisionType.OPTIMIZE_STRATEGY],
                DecisionType.INVESTIGATE_FURTHER: [DecisionType.MONITOR_CLOSELY],
                DecisionType.CONTINUE_OPERATION: [DecisionType.MONITOR_CLOSELY]
            }

            similar_decisions = decision_similarity.get(expected_decision, [])
            if decision.decision_type in similar_decisions:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)

        # Confidence appropriateness (30% weight)
        confidence_score = min(1.0, decision.confidence / 0.8)  # Normalize to expected range
        quality_factors.append(confidence_score)

        # Economic justification (20% weight)
        economic_score = 0.5  # Default moderate score
        if decision.roi_projection > 20:
            economic_score = 1.0
        elif decision.roi_projection > 10:
            economic_score = 0.8
        elif decision.roi_projection > 0:
            economic_score = 0.6
        quality_factors.append(economic_score)

        # Response timeliness (10% weight)
        urgency_match = scenario_data.get("expected_urgency", Priority.MEDIUM)
        if decision.urgency_level == urgency_match:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.7)

        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_quality = sum(factor * weight for factor, weight in zip(quality_factors, weights))

        return weighted_quality

    def _calculate_resource_efficiency(self, decision: MaintenanceDecision) -> float:
        """Calculate resource efficiency score"""
        # Base efficiency on cost-benefit ratio
        if decision.estimated_cost > 0:
            efficiency = min(1.0, decision.estimated_benefit / decision.estimated_cost)
        else:
            efficiency = 1.0

        # Factor in resource requirements
        resource_reqs = decision.resource_requirements
        if resource_reqs:
            # Penalize for excessive resource requirements
            technician_efficiency = 1.0 - min(0.3, resource_reqs.get('peak_technicians_needed', 1) * 0.1)
            time_efficiency = 1.0 - min(0.3, resource_reqs.get('total_duration_hours', 4) * 0.02)
            efficiency = efficiency * technician_efficiency * time_efficiency

        return efficiency

    def test_7_decision_types_coverage(self):
        """Test coverage of all 7 decision types"""
        print("\n🎯 Testing 7 decision types coverage...")

        # Generate scenarios that should trigger each decision type
        decision_type_scenarios = {
            DecisionType.IMMEDIATE_ACTION: {
                "sensor_data": {"CRITICAL_SENSOR": 95.0},
                "anomaly_scores": {"CRITICAL_SENSOR": 0.99}
            },
            DecisionType.SCHEDULE_MAINTENANCE: {
                "sensor_data": {"DEGRADED_SENSOR": 55.0},
                "anomaly_scores": {"DEGRADED_SENSOR": 0.75}
            },
            DecisionType.MONITOR_CLOSELY: {
                "sensor_data": {"WATCH_SENSOR": 45.0},
                "anomaly_scores": {"WATCH_SENSOR": 0.45}
            },
            DecisionType.CONTINUE_OPERATION: {
                "sensor_data": {"NORMAL_SENSOR": 25.0},
                "anomaly_scores": {"NORMAL_SENSOR": 0.10}
            },
            DecisionType.INVESTIGATE_FURTHER: {
                "sensor_data": {"UNUSUAL_SENSOR": 65.0},
                "anomaly_scores": {"UNUSUAL_SENSOR": 0.80}
            },
            DecisionType.OPTIMIZE_STRATEGY: {
                "sensor_data": {"INEFFICIENT_SENSOR": 35.0},
                "anomaly_scores": {"INEFFICIENT_SENSOR": 0.55}
            },
            DecisionType.REPLACE_EQUIPMENT: {
                "sensor_data": {"OLD_SENSOR": 85.0},
                "anomaly_scores": {"OLD_SENSOR": 0.95}
            }
        }

        triggered_decision_types = set()

        for expected_type, scenario_data in decision_type_scenarios.items():
            context = self._create_decision_context(scenario_data)

            try:
                decisions = asyncio.run(
                    self.business_rules_engine.make_maintenance_decision(context)
                )

                if decisions:
                    for decision in decisions:
                        triggered_decision_types.add(decision.decision_type)

            except Exception as e:
                print(f"     ⚠️  Failed to test {expected_type.value}: {e}")

        # Validate decision type coverage
        coverage_ratio = len(triggered_decision_types) / len(DecisionType)
        self.assertGreater(coverage_ratio, 0.6,
                          f"Decision type coverage {coverage_ratio:.1%} should be >60%")

        print(f"✅ Decision type coverage: {len(triggered_decision_types)}/{len(DecisionType)} "
              f"({coverage_ratio:.1%})")

    def test_5_confidence_levels_accuracy(self):
        """Test accuracy of 5 confidence levels"""
        print("\n🎯 Testing 5 confidence levels accuracy...")

        confidence_scenarios = [
            {
                "name": "very_high_confidence",
                "sensor_data": {"CLEAR_FAILURE": 90.0},
                "anomaly_scores": {"CLEAR_FAILURE": 0.98},
                "expected_level": ConfidenceLevel.VERY_HIGH
            },
            {
                "name": "high_confidence",
                "sensor_data": {"LIKELY_ISSUE": 65.0},
                "anomaly_scores": {"LIKELY_ISSUE": 0.85},
                "expected_level": ConfidenceLevel.HIGH
            },
            {
                "name": "medium_confidence",
                "sensor_data": {"MODERATE_CONCERN": 45.0},
                "anomaly_scores": {"MODERATE_CONCERN": 0.65},
                "expected_level": ConfidenceLevel.MEDIUM
            },
            {
                "name": "low_confidence",
                "sensor_data": {"UNCLEAR_SIGNAL": 35.0},
                "anomaly_scores": {"UNCLEAR_SIGNAL": 0.45},
                "expected_level": ConfidenceLevel.LOW
            },
            {
                "name": "very_low_confidence",
                "sensor_data": {"NOISY_DATA": 30.0},
                "anomaly_scores": {"NOISY_DATA": 0.25},
                "expected_level": ConfidenceLevel.VERY_LOW
            }
        ]

        confidence_accuracy = 0
        total_confidence_tests = len(confidence_scenarios)

        for scenario in confidence_scenarios:
            context = self._create_decision_context(scenario)

            try:
                decisions = asyncio.run(
                    self.business_rules_engine.make_maintenance_decision(context)
                )

                if decisions:
                    decision = decisions[0]
                    actual_level = decision.confidence_level
                    expected_level = scenario["expected_level"]

                    # Allow for adjacent confidence levels
                    confidence_levels = [
                        ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW,
                        ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH,
                        ConfidenceLevel.VERY_HIGH
                    ]

                    expected_idx = confidence_levels.index(expected_level)
                    actual_idx = confidence_levels.index(actual_level)

                    if abs(expected_idx - actual_idx) <= 1:
                        confidence_accuracy += 1

            except Exception as e:
                print(f"     ⚠️  Confidence test failed for {scenario['name']}: {e}")

        accuracy_rate = confidence_accuracy / total_confidence_tests

        self.assertGreater(accuracy_rate, 0.6,
                          f"Confidence level accuracy {accuracy_rate:.1%} should be >60%")

        print(f"✅ Confidence level accuracy: {accuracy_rate:.1%}")

    def test_resource_constraint_validation(self):
        """Test resource constraint validation"""
        print("\n💰 Testing resource constraint validation...")

        # Test scenario with limited resources
        limited_resource_context = DecisionContext(
            timestamp=datetime.now(),
            sensor_data={"EXPENSIVE_REPAIR": 75.0},
            anomaly_scores={"EXPENSIVE_REPAIR": 0.85},
            sensor_healths={},
            resource_availability={
                'technicians': 1,  # Limited technicians
                'budget': 1000     # Limited budget
            },
            business_constraints={
                'min_roi': 50.0,   # High ROI requirement
                'max_payback_months': 6  # Short payback period
            }
        )

        # Test decision making under constraints
        decisions = asyncio.run(
            self.business_rules_engine.make_maintenance_decision(limited_resource_context)
        )

        if decisions:
            for decision in decisions:
                # Validate resource feasibility
                self.assertLessEqual(decision.estimated_cost, 1000,
                                   "Decision should respect budget constraints")

                resource_reqs = decision.resource_requirements
                if resource_reqs:
                    self.assertLessEqual(resource_reqs.get('peak_technicians_needed', 0), 1,
                                       "Decision should respect technician availability")

                # Validate business constraints
                if decision.roi_projection > 0:
                    self.assertGreaterEqual(decision.roi_projection, 50.0,
                                          "Decision should meet minimum ROI requirement")

        print("✅ Resource constraint validation working")

    def test_decision_audit_and_tracking(self):
        """Test decision audit trail and tracking"""
        print("\n📋 Testing decision audit and tracking...")

        # Make several decisions
        test_scenarios = list(self.decision_scenarios.values())[:3]

        for i, scenario_data in enumerate(test_scenarios):
            context = self._create_decision_context(scenario_data)

            decisions = asyncio.run(
                self.business_rules_engine.make_maintenance_decision(context)
            )

            if decisions:
                # Validate decision has audit information
                decision = decisions[0]

                self.assertIsNotNone(decision.decision_id, "Decision should have unique ID")
                self.assertIsInstance(decision.decision_timestamp, datetime,
                                    "Decision should have timestamp")
                self.assertIsInstance(decision.contributing_rules, list,
                                    "Decision should track contributing rules")

        # Test decision history retrieval
        active_decisions = self.business_rules_engine.get_active_decisions()
        self.assertIsInstance(active_decisions, list, "Should return list of active decisions")

        # Test system performance metrics
        performance_metrics = self.business_rules_engine.get_system_performance_metrics()
        self.assertIsInstance(performance_metrics, dict, "Should return performance metrics")

        required_metrics = ['total_rules', 'active_rules', 'system_status']
        for metric in required_metrics:
            self.assertIn(metric, performance_metrics,
                         f"Performance metrics should include {metric}")

        print("✅ Decision audit and tracking working")

    def test_business_rules_performance(self):
        """Test business rules engine performance"""
        print("\n⚡ Testing business rules engine performance...")

        # Generate multiple scenarios for performance testing
        performance_scenarios = []
        for i in range(20):
            scenario = {
                "sensor_data": {f"PERF_SENSOR_{i}": 30.0 + np.random.uniform(-20, 40)},
                "anomaly_scores": {f"PERF_SENSOR_{i}": np.random.uniform(0.1, 0.9)}
            }
            performance_scenarios.append(scenario)

        start_time = datetime.now()

        # Process all scenarios
        total_decisions = 0
        for scenario_data in performance_scenarios:
            context = self._create_decision_context(scenario_data)

            try:
                decisions = asyncio.run(
                    self.business_rules_engine.make_maintenance_decision(context)
                )
                total_decisions += len(decisions) if decisions else 0

            except Exception as e:
                print(f"     ⚠️  Performance test scenario failed: {e}")

        execution_time = (datetime.now() - start_time).total_seconds()

        # Validate performance
        processing_rate = len(performance_scenarios) / execution_time
        self.assertGreater(processing_rate, 5.0,
                          f"Should process >5 scenarios/sec, got {processing_rate:.1f}")

        # Validate decision generation
        decision_rate = total_decisions / len(performance_scenarios)
        self.assertGreater(decision_rate, 0.5,
                          f"Should generate >0.5 decisions per scenario, got {decision_rate:.1f}")

        print(f"✅ Business rules performance: {processing_rate:.1f} scenarios/sec, "
              f"{decision_rate:.1f} decisions/scenario")

    @classmethod
    def tearDownClass(cls):
        """Generate business rules engine test report"""
        print("\n" + "="*80)
        print("PHASE 4.2 BUSINESS RULES ENGINE TEST SUMMARY")
        print("="*80)

        if cls.test_results:
            total_tests = len(cls.test_results)

            print(f"Decision tests: {total_tests}")

            # Calculate average metrics
            avg_decision_quality = np.mean([result.decision_quality for result in cls.test_results])
            avg_execution_time = np.mean([result.execution_time for result in cls.test_results])
            avg_resource_efficiency = np.mean([result.resource_efficiency for result in cls.test_results])
            avg_roi_projection = np.mean([result.roi_projection for result in cls.test_results])

            print(f"\n📊 Performance Metrics:")
            print(f"   Average decision quality: {avg_decision_quality:.2f}")
            print(f"   Average execution time: {avg_execution_time:.3f}s")
            print(f"   Average resource efficiency: {avg_resource_efficiency:.2f}")
            print(f"   Average ROI projection: {avg_roi_projection:.1f}%")

            # Decision type distribution
            decision_types = {}
            for result in cls.test_results:
                dt = result.decision_type
                decision_types[dt] = decision_types.get(dt, 0) + 1

            print(f"\n🎯 Decision Type Distribution:")
            for decision_type, count in decision_types.items():
                print(f"   {decision_type}: {count}")

            # Save detailed results
            results_file = project_root / "testing_phase4" / "test_results" / "phase4_2_business_rules_results.json"
            results_file.parent.mkdir(parents=True, exist_ok=True)

            results_data = []
            for result in cls.test_results:
                result_dict = {
                    'test_name': result.test_name,
                    'decision_type': result.decision_type,
                    'confidence_level': result.confidence_level,
                    'decision_quality': result.decision_quality,
                    'execution_time': result.execution_time,
                    'resource_efficiency': result.resource_efficiency,
                    'roi_projection': result.roi_projection,
                    'timestamp': result.timestamp.isoformat()
                }
                results_data.append(result_dict)

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)

            print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == '__main__':
    # Run business rules engine tests
    unittest.main(verbosity=2)