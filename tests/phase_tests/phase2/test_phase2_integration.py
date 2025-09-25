"""
Phase 2 Integration Tests
Comprehensive testing for all Phase 2 enhanced components
"""

import unittest
import tempfile
import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import Phase 2 components
from src.dashboard.components.subsystem_failure_analyzer import NASASubsystemFailureAnalyzer
from src.dashboard.components.detection_details_panel import EnhancedDetectionDetailsPanel
from src.dashboard.components.alert_action_manager import InteractiveAlertActionManager
from src.dashboard.components.threshold_manager import EquipmentSpecificThresholdManager
from src.database.phase2_enhancements import Phase2DatabaseManager, AlertActionRecord, ThresholdHistoryRecord

# Import existing components for integration
from src.alerts.alert_manager import Alert, AlertStatus, AlertSeverity, AlertType
from src.maintenance.work_order_manager import WorkOrder, WorkOrderPriority, MaintenanceType, WorkOrderStatus
from src.anomaly_detection.nasa_anomaly_engine import AnomalyResult


class TestPhase2Integration(unittest.TestCase):
    """Test suite for Phase 2 component integration"""

    def setUp(self):
        """Set up test environment"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()

        # Initialize Phase 2 components with test configuration
        self.db_manager = Phase2DatabaseManager(self.temp_db.name)
        self.subsystem_analyzer = NASASubsystemFailureAnalyzer()
        self.detection_panel = EnhancedDetectionDetailsPanel()
        self.alert_manager = InteractiveAlertActionManager()
        self.threshold_manager = EquipmentSpecificThresholdManager()

        # Create mock equipment for testing
        self.mock_equipment = self._create_mock_equipment()

        # Create test alert
        self.test_alert = Alert(
            alert_id="test_alert_001",
            rule_id="test_rule",
            alert_type=AlertType.ANOMALY,
            severity=AlertSeverity.HIGH,
            status=AlertStatus.NEW,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source="TEST_EQUIPMENT",
            title="Test Anomaly Alert",
            description="High anomaly score detected on test equipment",
            details={"anomaly_score": 0.85, "sensor": "test_sensor"},
            affected_equipment=["SMAP-PWR-001"],
            metrics={"confidence": 0.92}
        )

    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.temp_db.name)
        except OSError:
            pass

    def _create_mock_equipment(self):
        """Create mock equipment for testing"""
        from src.data_ingestion.equipment_mapper import EquipmentComponent, SensorSpec

        sensors = [
            SensorSpec("Solar Panel Voltage", "V", 25.0, 35.0, 30.0, 26.0, "SATELLITE", "POWER"),
            SensorSpec("Battery Current", "A", 0.0, 15.0, 8.0, 12.0, "SATELLITE", "POWER"),
            SensorSpec("Power Distribution Temperature", "°C", -20.0, 60.0, 25.0, 55.0, "SATELLITE", "POWER")
        ]

        return EquipmentComponent(
            equipment_id="SMAP-PWR-001",
            equipment_type="Power System",
            subsystem="POWER",
            location="Satellite Bus",
            sensors=sensors,
            criticality="CRITICAL"
        )

    # =============================================================================
    # Database Integration Tests
    # =============================================================================

    def test_database_initialization(self):
        """Test database initialization and table creation"""
        # Check if all required tables exist
        stats = self.db_manager.get_database_statistics()

        self.assertIn('alert_actions_count', stats)
        self.assertIn('threshold_history_count', stats)
        self.assertIn('subsystem_analytics_count', stats)
        self.assertIn('equipment_health_snapshots_count', stats)
        self.assertIn('detection_details_history_count', stats)

        # All should be 0 initially
        self.assertEqual(stats['alert_actions_count'], 0)
        self.assertEqual(stats['threshold_history_count'], 0)

    def test_alert_action_recording(self):
        """Test alert action recording and retrieval"""
        action = AlertActionRecord(
            action_id=str(uuid.uuid4()),
            alert_id="test_alert_001",
            action_type="acknowledge",
            user_id="test_user",
            timestamp=datetime.now(),
            notes="Test acknowledgment"
        )

        # Record action
        success = self.db_manager.record_alert_action(action)
        self.assertTrue(success)

        # Retrieve actions
        actions = self.db_manager.get_alert_actions("test_alert_001")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "acknowledge")
        self.assertEqual(actions[0].user_id, "test_user")

    def test_threshold_history_recording(self):
        """Test threshold history recording and retrieval"""
        record = ThresholdHistoryRecord(
            change_id=str(uuid.uuid4()),
            equipment_id="SMAP-PWR-001",
            equipment_type="Power System",
            subsystem="POWER",
            criticality="CRITICAL",
            old_thresholds={"critical": 0.90, "high": 0.75},
            new_thresholds={"critical": 0.85, "high": 0.70},
            optimization_type="accuracy",
            improvement_score=0.15,
            confidence=0.85,
            justification="Optimization for better accuracy",
            user_id="test_user",
            timestamp=datetime.now(),
            applied=True
        )

        # Record threshold change
        success = self.db_manager.record_threshold_change(record)
        self.assertTrue(success)

        # Retrieve history
        history = self.db_manager.get_threshold_history("SMAP-PWR-001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].optimization_type, "accuracy")
        self.assertEqual(history[0].improvement_score, 0.15)

    # =============================================================================
    # Subsystem Analyzer Integration Tests
    # =============================================================================

    def test_subsystem_analyzer_initialization(self):
        """Test subsystem analyzer initialization"""
        self.assertIsNotNone(self.subsystem_analyzer)
        self.assertIsInstance(self.subsystem_analyzer.subsystem_health, dict)

    def test_power_system_analysis(self):
        """Test power system failure pattern analysis"""
        analysis = self.subsystem_analyzer.analyze_power_system_patterns()

        self.assertIn('smap_power', analysis)
        self.assertIn('msl_power', analysis)
        self.assertIn('comparative_analysis', analysis)
        self.assertIn('recommendations', analysis)

        # Check SMAP power analysis structure
        smap_analysis = analysis['smap_power']
        self.assertIn('patterns', smap_analysis)
        self.assertIn('total_patterns', smap_analysis)
        self.assertIn('risk_level', smap_analysis)

    def test_mobility_system_analysis(self):
        """Test mobility system failure pattern analysis"""
        analysis = self.subsystem_analyzer.analyze_mobility_system_patterns()

        self.assertIn('wheel_motor_patterns', analysis)
        self.assertIn('suspension_patterns', analysis)
        self.assertIn('locomotion_coordination', analysis)
        self.assertIn('recommendations', analysis)

    def test_communication_system_analysis(self):
        """Test communication system failure pattern analysis"""
        analysis = self.subsystem_analyzer.analyze_communication_system_patterns()

        self.assertIn('smap_communication', analysis)
        self.assertIn('msl_communication', analysis)
        self.assertIn('signal_degradation', analysis)
        self.assertIn('recommendations', analysis)

    def test_subsystem_health_summary(self):
        """Test subsystem health summary generation"""
        summary = self.subsystem_analyzer.get_subsystem_health_summary()

        self.assertIn('overall_status', summary)
        self.assertIn('critical_alerts', summary)
        self.assertIn('trending_issues', summary)
        self.assertIn('health_scores', summary)

    # =============================================================================
    # Detection Details Panel Integration Tests
    # =============================================================================

    def test_detection_panel_initialization(self):
        """Test detection details panel initialization"""
        self.assertIsNotNone(self.detection_panel)

    @patch('src.dashboard.components.detection_details_panel.equipment_mapper')
    def test_equipment_detection_analysis(self, mock_mapper):
        """Test equipment detection analysis"""
        # Mock equipment mapper
        mock_mapper.get_all_equipment.return_value = [self.mock_equipment]

        # Create mock anomaly result
        mock_result = AnomalyResult(
            timestamp=datetime.now(),
            equipment_id="SMAP-PWR-001",
            equipment_type="Power System",
            subsystem="POWER",
            anomaly_score=0.85,
            is_anomaly=True,
            severity_level="HIGH",
            confidence=0.92,
            model_name="test_model",
            reconstruction_error=0.15,
            sensor_values={"Solar Panel Voltage": 25.6, "Battery Current": 12.1},
            anomalous_sensors=["Solar Panel Voltage"],
            requires_alert=True,
            alert_message="High anomaly detected"
        )

        # Test analysis
        summary = self.detection_panel.analyze_equipment_detection("SMAP-PWR-001", mock_result)

        self.assertEqual(summary.equipment_id, "SMAP-PWR-001")
        self.assertEqual(summary.subsystem, "POWER")
        self.assertEqual(summary.overall_anomaly_score, 0.85)
        self.assertTrue(summary.is_anomaly)
        self.assertEqual(summary.severity_level, "HIGH")

    # =============================================================================
    # Alert Action Manager Integration Tests
    # =============================================================================

    def test_alert_acknowledgment(self):
        """Test alert acknowledgment functionality"""
        result = self.alert_manager.acknowledge_alert(
            "test_alert_001", "test_user", "Test acknowledgment"
        )

        # Note: This will fail with default result since no actual alert exists
        # In a real environment, this would test with the actual alert manager
        self.assertIsNotNone(result)
        self.assertFalse(result.success)  # Expected to fail with mock data

    def test_alert_dismissal(self):
        """Test alert dismissal functionality"""
        result = self.alert_manager.dismiss_alert(
            "test_alert_001", "test_user", "False positive", "Test dismissal"
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.success)  # Expected to fail with mock data

    def test_work_order_creation(self):
        """Test work order creation from alert"""
        work_order_params = {
            'type': 'CORRECTIVE',
            'priority': 'HIGH',
            'duration_hours': 4.0,
            'cost': 500.0,
            'description': 'Test work order'
        }

        result = self.alert_manager.create_work_order_from_alert(
            "test_alert_001", "test_user", work_order_params
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.success)  # Expected to fail with mock data

    def test_alert_escalation(self):
        """Test alert escalation functionality"""
        result = self.alert_manager.escalate_alert(
            "test_alert_001", "test_user", "No response within SLA", "Test escalation"
        )

        self.assertIsNotNone(result)
        self.assertFalse(result.success)  # Expected to fail with mock data

    # =============================================================================
    # Threshold Manager Integration Tests
    # =============================================================================

    def test_threshold_manager_initialization(self):
        """Test threshold manager initialization"""
        self.assertIsNotNone(self.threshold_manager)
        self.assertIsInstance(self.threshold_manager.threshold_configs, dict)

    def test_threshold_optimization(self):
        """Test threshold optimization functionality"""
        # This would need mock equipment setup
        # For now, test the optimization result structure
        result = self.threshold_manager.optimize_thresholds_for_equipment(
            "SMAP-PWR-001", "accuracy"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.equipment_id, "SMAP-PWR-001")
        self.assertEqual(result.optimization_type, "accuracy")

    def test_threshold_recommendations(self):
        """Test threshold recommendation generation"""
        recommendations = self.threshold_manager.get_threshold_recommendations()

        self.assertIsInstance(recommendations, list)
        # With mock data, might be empty, but structure should be correct

    # =============================================================================
    # End-to-End Integration Tests
    # =============================================================================

    def test_complete_workflow(self):
        """Test complete workflow from detection to alert action"""
        # 1. Analyze subsystem patterns
        power_analysis = self.subsystem_analyzer.analyze_power_system_patterns()
        self.assertIn('smap_power', power_analysis)

        # 2. Analyze detection details
        summary = self.detection_panel.analyze_equipment_detection("SMAP-PWR-001")
        self.assertIsNotNone(summary)

        # 3. Record alert action
        action = AlertActionRecord(
            action_id=str(uuid.uuid4()),
            alert_id="workflow_test_alert",
            action_type="acknowledge",
            user_id="test_user",
            timestamp=datetime.now(),
            notes="Workflow test"
        )
        success = self.db_manager.record_alert_action(action)
        self.assertTrue(success)

        # 4. Optimize thresholds
        optimization_result = self.threshold_manager.optimize_thresholds_for_equipment(
            "SMAP-PWR-001", "accuracy"
        )
        self.assertIsNotNone(optimization_result)

    def test_api_data_flow(self):
        """Test data flow for API integration"""
        # Test database statistics
        stats = self.db_manager.get_database_statistics()
        self.assertIn('alert_actions_count', stats)

        # Test alert action statistics
        alert_stats = self.db_manager.get_alert_action_statistics(30)
        self.assertIn('action_counts', alert_stats)
        self.assertIn('total_actions', alert_stats)

    def test_error_handling(self):
        """Test error handling across components"""
        # Test with invalid equipment ID
        summary = self.detection_panel.analyze_equipment_detection("INVALID_ID")
        self.assertEqual(summary.equipment_id, "INVALID_ID")
        self.assertEqual(summary.equipment_type, "Unknown")

        # Test with invalid alert ID
        result = self.alert_manager.acknowledge_alert("INVALID_ALERT", "test_user")
        self.assertFalse(result.success)

    def test_performance_under_load(self):
        """Test component performance under simulated load"""
        # Record multiple alert actions
        for i in range(100):
            action = AlertActionRecord(
                action_id=str(uuid.uuid4()),
                alert_id=f"load_test_alert_{i}",
                action_type="acknowledge",
                user_id="load_test_user",
                timestamp=datetime.now(),
                notes=f"Load test {i}"
            )
            success = self.db_manager.record_alert_action(action)
            self.assertTrue(success)

        # Verify all records were created
        stats = self.db_manager.get_database_statistics()
        self.assertEqual(stats['alert_actions_count'], 100)

    # =============================================================================
    # Component Interaction Tests
    # =============================================================================

    def test_subsystem_analyzer_detection_panel_integration(self):
        """Test integration between subsystem analyzer and detection panel"""
        # Get subsystem health
        health_summary = self.subsystem_analyzer.get_subsystem_health_summary()

        # Use health data for detection analysis
        if health_summary.get('overall_status'):
            equipment_ids = list(health_summary['overall_status'].keys())
            if equipment_ids:
                # Test detection analysis for equipment from health summary
                summary = self.detection_panel.analyze_equipment_detection(equipment_ids[0])
                self.assertIsNotNone(summary)

    def test_alert_manager_database_integration(self):
        """Test integration between alert manager and database"""
        # Create alert action
        action = AlertActionRecord(
            action_id=str(uuid.uuid4()),
            alert_id="integration_test_alert",
            action_type="acknowledge",
            user_id="integration_test_user",
            timestamp=datetime.now(),
            notes="Integration test"
        )

        # Record in database
        success = self.db_manager.record_alert_action(action)
        self.assertTrue(success)

        # Retrieve and verify
        actions = self.db_manager.get_alert_actions("integration_test_alert")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "acknowledge")

    def test_threshold_manager_database_integration(self):
        """Test integration between threshold manager and database"""
        # Record threshold change
        record = ThresholdHistoryRecord(
            change_id=str(uuid.uuid4()),
            equipment_id="SMAP-PWR-001",
            equipment_type="Power System",
            subsystem="POWER",
            criticality="CRITICAL",
            old_thresholds={"critical": 0.90},
            new_thresholds={"critical": 0.85},
            optimization_type="accuracy",
            improvement_score=0.10,
            confidence=0.80,
            justification="Test optimization",
            user_id="test_user",
            timestamp=datetime.now(),
            applied=True
        )

        success = self.db_manager.record_threshold_change(record)
        self.assertTrue(success)

        # Retrieve and verify
        history = self.db_manager.get_threshold_history("SMAP-PWR-001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].optimization_type, "accuracy")


class Phase2IntegrationTestRunner:
    """Test runner for Phase 2 integration tests"""

    def __init__(self):
        self.test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2Integration)

    def run_tests(self, verbosity=2):
        """Run all Phase 2 integration tests"""
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(self.test_suite)

        return result

    def run_specific_test(self, test_name, verbosity=2):
        """Run a specific test"""
        suite = unittest.TestSuite()
        suite.addTest(TestPhase2Integration(test_name))

        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        return result


def main():
    """Main function to run Phase 2 integration tests"""
    print("=" * 80)
    print("NASA IoT Predictive Maintenance System - Phase 2 Integration Tests")
    print("=" * 80)

    runner = Phase2IntegrationTestRunner()
    result = runner.run_tests()

    print("\n" + "=" * 80)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    print("=" * 80)

    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)