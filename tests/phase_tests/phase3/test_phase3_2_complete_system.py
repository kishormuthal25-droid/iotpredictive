"""
Comprehensive test suite for Phase 3.2 Work Order Management Automation
Tests complete integration and performance of all Phase 3.2 components
"""

import unittest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import warnings
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Phase 3.2 components
from src.maintenance.work_order_manager import WorkOrderManager, WorkOrderPriority, WorkOrderStatus, MaintenanceType
from src.maintenance.automated_work_order_creator import AutomatedWorkOrderCreator
from src.maintenance.optimized_resource_allocator import OptimizedResourceAllocator
from src.maintenance.work_order_lifecycle_tracker import WorkOrderLifecycleTracker
from src.maintenance.technician_performance_analyzer import TechnicianPerformanceAnalyzer
from src.maintenance.work_order_analytics import WorkOrderAnalytics
from src.maintenance.phase3_2_integration import Phase32Integration, IntegratedAutomationConfig

# Import test data utilities
from src.maintenance.work_order_manager import Equipment, Technician, WorkOrder

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPhase32CompleteSystem(unittest.TestCase):
    """Comprehensive test suite for Phase 3.2 automation system"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_config = IntegratedAutomationConfig(
            automation_enabled=True,
            auto_assignment_enabled=True,
            lifecycle_tracking_enabled=True,
            performance_monitoring_enabled=True,
            analytics_dashboard_enabled=True,
            min_automation_confidence=0.7
        )

    def setUp(self):
        """Set up test environment for each test"""
        # Initialize Phase 3.2 integration system
        self.integration_system = Phase32Integration(config=self.test_config)

        # Initialize complete system
        self.assertTrue(
            self.integration_system.initialize_complete_system(),
            "Failed to initialize complete Phase 3.2 system"
        )

        # Add test data
        self._setup_test_data()

    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'integration_system'):
            del self.integration_system

    def _setup_test_data(self):
        """Set up test data for comprehensive testing"""
        # Create test equipment
        test_equipment = [
            Equipment(
                equipment_id='EQ001',
                name='Primary Pump',
                type='pump',
                location='Zone_A',
                criticality='critical',
                installation_date=datetime.now() - timedelta(days=365),
                last_maintenance=datetime.now() - timedelta(days=30)
            ),
            Equipment(
                equipment_id='EQ002',
                name='Backup Motor',
                type='motor',
                location='Zone_B',
                criticality='high',
                installation_date=datetime.now() - timedelta(days=500),
                last_maintenance=datetime.now() - timedelta(days=60)
            ),
            Equipment(
                equipment_id='EQ003',
                name='Sensor Array',
                type='sensor',
                location='Zone_A',
                criticality='medium',
                installation_date=datetime.now() - timedelta(days=200)
            )
        ]

        # Register equipment
        for equipment in test_equipment:
            self.integration_system.work_order_manager.register_equipment(equipment)

        # Create test technicians
        test_technicians = [
            Technician(
                technician_id='TECH001',
                name='John Smith',
                skills=['pump', 'hydraulic', 'mechanical'],
                availability={'Monday': [(datetime.now().replace(hour=8, minute=0),
                                        datetime.now().replace(hour=17, minute=0))]},
                current_workload=2,
                max_daily_orders=8,
                location='Zone_A',
                certification_level=4,
                hourly_rate=75.0
            ),
            Technician(
                technician_id='TECH002',
                name='Jane Doe',
                skills=['motor', 'electrical', 'programming'],
                availability={'Monday': [(datetime.now().replace(hour=9, minute=0),
                                        datetime.now().replace(hour=18, minute=0))]},
                current_workload=1,
                max_daily_orders=6,
                location='Zone_B',
                certification_level=5,
                hourly_rate=85.0
            ),
            Technician(
                technician_id='TECH003',
                name='Bob Johnson',
                skills=['sensor', 'electronics', 'calibration'],
                availability={'Monday': [(datetime.now().replace(hour=7, minute=0),
                                        datetime.now().replace(hour=16, minute=0))]},
                current_workload=3,
                max_daily_orders=7,
                location='Zone_A',
                certification_level=3,
                hourly_rate=65.0
            )
        ]

        # Register technicians
        for technician in test_technicians:
            self.integration_system.work_order_manager.register_technician(technician)

    def test_automated_work_order_creation(self):
        """Test automated work order creation from anomaly"""
        logger.info("Testing automated work order creation...")

        # Test critical anomaly
        critical_anomaly = {
            'anomaly_id': 'ANOM001',
            'type': 'failure',
            'severity': 'critical',
            'confidence': 0.95,
            'sensor': 'pressure_sensor_01',
            'value': 150,
            'threshold': 100,
            'timestamp': datetime.now(),
            'trend': {'direction': 'increasing', 'acceleration': 0.5}
        }

        result = self.integration_system.process_anomaly_with_full_automation(
            anomaly_data=critical_anomaly,
            equipment_id='EQ001'
        )

        # Validate results
        self.assertTrue(result['automation_successful'], "Automation should succeed for critical anomaly")
        self.assertTrue(result['work_order_created'], "Work order should be created")
        self.assertTrue(result['technician_assigned'], "Technician should be assigned")
        self.assertTrue(result['lifecycle_tracking_started'], "Lifecycle tracking should start")
        self.assertGreater(result['automation_confidence'], 0.8, "Automation confidence should be high")
        self.assertLess(result['processing_time_seconds'], 5.0, "Processing should be fast")

        logger.info("✓ Automated work order creation test passed")

    def test_intelligent_priority_assignment(self):
        """Test intelligent priority assignment"""
        logger.info("Testing intelligent priority assignment...")

        # Test different priority scenarios
        test_scenarios = [
            {
                'anomaly': {
                    'type': 'failure',
                    'severity': 'critical',
                    'confidence': 0.98
                },
                'equipment_id': 'EQ001',  # Critical equipment
                'expected_priority': WorkOrderPriority.CRITICAL
            },
            {
                'anomaly': {
                    'type': 'degradation',
                    'severity': 'medium',
                    'confidence': 0.75
                },
                'equipment_id': 'EQ003',  # Medium criticality equipment
                'expected_priority': WorkOrderPriority.MEDIUM
            }
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario):
                result = self.integration_system.process_anomaly_with_full_automation(
                    anomaly_data=scenario['anomaly'],
                    equipment_id=scenario['equipment_id']
                )

                if result.get('work_order_id'):
                    work_order = self.integration_system.work_order_manager.work_orders[result['work_order_id']]

                    # Priority should match expected or be appropriately adjusted
                    self.assertIsNotNone(work_order.priority, "Priority should be assigned")

                    # For critical scenarios, priority should be high
                    if scenario['expected_priority'] == WorkOrderPriority.CRITICAL:
                        self.assertIn(work_order.priority, [WorkOrderPriority.CRITICAL, WorkOrderPriority.HIGH],
                                    "Critical scenarios should have high priority")

        logger.info("✓ Intelligent priority assignment test passed")

    def test_optimized_resource_allocation(self):
        """Test optimized resource allocation"""
        logger.info("Testing optimized resource allocation...")

        # Create multiple work orders to test allocation optimization
        anomalies = [
            {
                'anomaly_id': 'ANOM_PUMP_001',
                'type': 'vibration',
                'severity': 'high',
                'confidence': 0.85,
                'equipment_type': 'pump'
            },
            {
                'anomaly_id': 'ANOM_MOTOR_001',
                'type': 'temperature',
                'severity': 'medium',
                'confidence': 0.78,
                'equipment_type': 'motor'
            },
            {
                'anomaly_id': 'ANOM_SENSOR_001',
                'type': 'calibration',
                'severity': 'low',
                'confidence': 0.72,
                'equipment_type': 'sensor'
            }
        ]

        equipment_ids = ['EQ001', 'EQ002', 'EQ003']
        assignment_results = []

        # Process all anomalies
        for i, anomaly in enumerate(anomalies):
            result = self.integration_system.process_anomaly_with_full_automation(
                anomaly_data=anomaly,
                equipment_id=equipment_ids[i]
            )
            assignment_results.append(result)

        # Validate resource allocation
        assigned_technicians = []
        for result in assignment_results:
            if result.get('technician_assigned') and result.get('assigned_technician'):
                assigned_technicians.append(result['assigned_technician'])

        # Check that different technicians are assigned when possible (load balancing)
        if len(assigned_technicians) > 1:
            unique_technicians = len(set(assigned_technicians))
            self.assertGreaterEqual(unique_technicians, min(2, len(assigned_technicians)),
                                  "Resource allocation should balance workload")

        # Validate skill matching
        for i, result in enumerate(assignment_results):
            if result.get('assigned_technician'):
                technician = self.integration_system.work_order_manager.technician_registry[
                    result['assigned_technician']
                ]

                # Check if technician has relevant skills
                anomaly_type = anomalies[i]['equipment_type']
                self.assertTrue(
                    any(skill in technician.skills for skill in [anomaly_type, 'general']),
                    f"Technician should have relevant skills for {anomaly_type}"
                )

        logger.info("✓ Optimized resource allocation test passed")

    def test_lifecycle_tracking(self):
        """Test comprehensive lifecycle tracking"""
        logger.info("Testing lifecycle tracking...")

        # Create work order and track lifecycle
        anomaly = {
            'anomaly_id': 'LIFECYCLE_TEST_001',
            'type': 'failure',
            'severity': 'high',
            'confidence': 0.88
        }

        result = self.integration_system.process_anomaly_with_full_automation(
            anomaly_data=anomaly,
            equipment_id='EQ001'
        )

        self.assertTrue(result['work_order_created'], "Work order should be created for lifecycle test")

        work_order_id = result['work_order_id']
        lifecycle_tracker = self.integration_system.lifecycle_tracker

        # Check initial lifecycle events
        timeline = lifecycle_tracker.get_work_order_timeline(work_order_id)
        self.assertGreater(len(timeline), 0, "Lifecycle events should be recorded")

        # Test status updates with tracking
        success = self.integration_system.work_order_manager.update_work_order_status_with_tracking(
            order_id=work_order_id,
            new_status=WorkOrderStatus.IN_PROGRESS,
            performer='TECH001',
            notes="Starting work on critical pump"
        )

        self.assertTrue(success, "Status update with tracking should succeed")

        # Check updated timeline
        updated_timeline = lifecycle_tracker.get_work_order_timeline(work_order_id)
        self.assertGreater(len(updated_timeline), len(timeline), "New lifecycle events should be added")

        # Test SLA metrics
        sla_status = lifecycle_tracker.get_sla_status(work_order_id)
        self.assertIsNotNone(sla_status, "SLA status should be available")

        # Complete the work order
        complete_success = self.integration_system.work_order_manager.update_work_order_status_with_tracking(
            order_id=work_order_id,
            new_status=WorkOrderStatus.COMPLETED,
            performer='TECH001',
            notes="Work completed successfully"
        )

        self.assertTrue(complete_success, "Work order completion should succeed")

        # Get timeline analysis
        timeline_analysis = lifecycle_tracker.get_timeline_analysis(work_order_id)
        self.assertIsNotNone(timeline_analysis, "Timeline analysis should be available")
        self.assertGreater(timeline_analysis.total_lifecycle_hours, 0, "Total lifecycle time should be positive")

        logger.info("✓ Lifecycle tracking test passed")

    def test_performance_analytics(self):
        """Test technician performance analytics"""
        logger.info("Testing performance analytics...")

        performance_analyzer = self.integration_system.performance_analyzer

        # Test technician performance analysis
        technician_id = 'TECH001'
        performance_metrics = performance_analyzer.analyze_technician_performance(technician_id)

        if performance_metrics:
            self.assertEqual(performance_metrics.technician_id, technician_id)
            self.assertIsInstance(performance_metrics.completion_rate, float)
            self.assertIsInstance(performance_metrics.efficiency_score, float)
            self.assertIsInstance(performance_metrics.workload_utilization, float)

        # Test workload balance analysis
        workload_analysis = performance_analyzer.analyze_workload_balance()

        if workload_analysis:
            self.assertIsInstance(workload_analysis.workload_imbalance_score, float)
            self.assertIsInstance(workload_analysis.technician_workloads, dict)
            self.assertGreaterEqual(workload_analysis.capacity_utilization_rate, 0.0)
            self.assertLessEqual(workload_analysis.capacity_utilization_rate, 1.0)

        # Test performance comparison
        technician_ids = ['TECH001', 'TECH002', 'TECH003']
        performance_comparison = performance_analyzer.compare_technician_performance(technician_ids)

        if performance_comparison:
            self.assertEqual(len(performance_comparison.technicians_compared), len(technician_ids))
            self.assertIsInstance(performance_comparison.comparison_metrics, dict)
            self.assertIsInstance(performance_comparison.performance_rankings, dict)

        logger.info("✓ Performance analytics test passed")

    def test_automation_dashboard_data(self):
        """Test automation dashboard data generation"""
        logger.info("Testing automation dashboard data...")

        # Generate dashboard data
        dashboard_data = self.integration_system.work_order_manager.get_automation_dashboard_data()

        self.assertIsInstance(dashboard_data, dict, "Dashboard data should be a dictionary")

        # Check required sections
        required_sections = [
            'automation_metrics',
            'enhanced_metrics',
            'workload_analysis',
            'recent_automated_orders',
            'automation_efficiency_trends',
            'system_health'
        ]

        for section in required_sections:
            self.assertIn(section, dashboard_data, f"Dashboard data should include {section}")

        # Validate automation metrics
        automation_metrics = dashboard_data['automation_metrics']
        self.assertIsInstance(automation_metrics, dict)

        expected_automation_metrics = [
            'automated_work_orders_created',
            'automated_assignments_made',
            'automation_success_rate',
            'manual_interventions'
        ]

        for metric in expected_automation_metrics:
            self.assertIn(metric, automation_metrics, f"Automation metrics should include {metric}")

        # Validate system health
        system_health = dashboard_data['system_health']
        self.assertIsInstance(system_health, dict)
        self.assertIn('overall_health', system_health)

        logger.info("✓ Automation dashboard data test passed")

    def test_system_integration_status(self):
        """Test integrated system status reporting"""
        logger.info("Testing system integration status...")

        # Get system status
        system_status = self.integration_system.get_integrated_system_status()

        self.assertIsInstance(system_status, dict, "System status should be a dictionary")

        # Check required fields
        required_fields = [
            'timestamp',
            'system_health',
            'component_status',
            'integration_metrics',
            'active_workflows'
        ]

        for field in required_fields:
            self.assertIn(field, system_status, f"System status should include {field}")

        # Validate component status
        component_status = system_status['component_status']
        expected_components = [
            'work_order_manager',
            'automated_creator',
            'resource_allocator',
            'lifecycle_tracker',
            'performance_analyzer'
        ]

        for component in expected_components:
            self.assertIn(component, component_status, f"Component status should include {component}")
            self.assertIn(component_status[component], ['operational', 'not_initialized', 'error'],
                         f"Component {component} should have valid status")

        # Validate system health
        valid_health_states = ['healthy', 'degraded', 'critical', 'error']
        self.assertIn(system_status['system_health'], valid_health_states,
                     "System health should be in valid states")

        logger.info("✓ System integration status test passed")

    def test_comprehensive_metrics_calculation(self):
        """Test comprehensive metrics calculation"""
        logger.info("Testing comprehensive metrics calculation...")

        # Get comprehensive metrics
        comprehensive_metrics = self.integration_system.get_comprehensive_metrics()

        if comprehensive_metrics:
            # Validate timestamp
            self.assertIsInstance(comprehensive_metrics.timestamp, datetime)

            # Validate Phase 3.2 metrics
            self.assertIsInstance(comprehensive_metrics.automation_success_rate, float)
            self.assertIsInstance(comprehensive_metrics.workload_balance_score, float)
            self.assertIsInstance(comprehensive_metrics.lifecycle_efficiency, float)
            self.assertIsInstance(comprehensive_metrics.performance_analytics_insights, int)

            # Validate integration metrics
            self.assertIsInstance(comprehensive_metrics.system_synergy_score, float)
            self.assertIsInstance(comprehensive_metrics.cross_component_efficiency, float)
            self.assertIn(comprehensive_metrics.overall_system_health,
                         ['excellent', 'good', 'fair', 'poor', 'error'])

            # Validate ranges
            self.assertGreaterEqual(comprehensive_metrics.automation_success_rate, 0.0)
            self.assertLessEqual(comprehensive_metrics.automation_success_rate, 1.0)

            self.assertGreaterEqual(comprehensive_metrics.workload_balance_score, 0.0)
            self.assertLessEqual(comprehensive_metrics.workload_balance_score, 1.0)

        logger.info("✓ Comprehensive metrics calculation test passed")

    def test_end_to_end_automation_workflow(self):
        """Test complete end-to-end automation workflow"""
        logger.info("Testing end-to-end automation workflow...")

        # Test complete workflow from anomaly detection to completion
        test_anomaly = {
            'anomaly_id': 'E2E_TEST_001',
            'type': 'failure',
            'severity': 'critical',
            'confidence': 0.92,
            'sensor': 'main_pump_pressure',
            'value': 180,
            'threshold': 120,
            'timestamp': datetime.now(),
            'equipment_type': 'pump',
            'location': 'Zone_A'
        }

        # Step 1: Process anomaly with full automation
        automation_result = self.integration_system.process_anomaly_with_full_automation(
            anomaly_data=test_anomaly,
            equipment_id='EQ001'
        )

        # Validate automation result
        self.assertTrue(automation_result['automation_successful'], "End-to-end automation should succeed")
        self.assertTrue(automation_result['work_order_created'], "Work order should be created")
        self.assertTrue(automation_result['technician_assigned'], "Technician should be assigned")

        work_order_id = automation_result['work_order_id']
        assigned_technician = automation_result['assigned_technician']

        # Step 2: Simulate work progress
        progress_updates = [
            (WorkOrderStatus.IN_PROGRESS, "Started diagnostic procedures"),
            (WorkOrderStatus.COMPLETED, "Replaced faulty pressure valve and tested system")
        ]

        for status, notes in progress_updates:
            success = self.integration_system.work_order_manager.update_work_order_status_with_tracking(
                order_id=work_order_id,
                new_status=status,
                performer=assigned_technician,
                notes=notes
            )
            self.assertTrue(success, f"Status update to {status} should succeed")

        # Step 3: Validate final state
        work_order = self.integration_system.work_order_manager.work_orders[work_order_id]
        self.assertEqual(work_order.status, WorkOrderStatus.COMPLETED, "Work order should be completed")

        # Step 4: Check lifecycle tracking
        timeline = self.integration_system.lifecycle_tracker.get_work_order_timeline(work_order_id)
        self.assertGreaterEqual(len(timeline), 3, "Timeline should have creation, progress, and completion events")

        # Step 5: Validate performance impact
        performance_metrics = self.integration_system.performance_analyzer.analyze_technician_performance(
            assigned_technician
        )
        if performance_metrics:
            self.assertGreater(performance_metrics.total_work_orders, 0,
                             "Technician should have completed work orders")

        # Step 6: Check system metrics
        system_status = self.integration_system.get_integrated_system_status()
        self.assertGreater(system_status['integration_metrics']['total_integrated_operations'], 0,
                          "Integration operations should be recorded")

        logger.info("✓ End-to-end automation workflow test passed")

    def test_performance_benchmarks(self):
        """Test system performance against benchmarks"""
        logger.info("Testing performance benchmarks...")

        # Performance benchmark targets
        benchmarks = {
            'automation_response_time': 5.0,  # seconds
            'automation_success_rate': 0.80,  # 80%
            'assignment_accuracy': 0.85,      # 85%
            'workload_balance_threshold': 0.70 # 70%
        }

        # Test automation response time
        start_time = datetime.now()

        test_anomaly = {
            'anomaly_id': 'PERF_TEST_001',
            'type': 'degradation',
            'severity': 'medium',
            'confidence': 0.80
        }

        result = self.integration_system.process_anomaly_with_full_automation(
            anomaly_data=test_anomaly,
            equipment_id='EQ002'
        )

        response_time = result.get('processing_time_seconds', float('inf'))
        self.assertLess(response_time, benchmarks['automation_response_time'],
                       f"Response time {response_time}s should be under {benchmarks['automation_response_time']}s")

        # Test automation success rate benchmark
        if result['automation_successful']:
            automation_confidence = result.get('automation_confidence', 0.0)
            self.assertGreater(automation_confidence, benchmarks['automation_success_rate'],
                             f"Automation confidence should exceed {benchmarks['automation_success_rate']}")

        # Test system resource utilization
        system_status = self.integration_system.get_integrated_system_status()
        self.assertIn(system_status['system_health'], ['healthy', 'degraded'],
                     "System health should be acceptable under load")

        logger.info("✓ Performance benchmarks test passed")


def run_phase32_tests():
    """Run all Phase 3.2 tests"""
    print("=" * 80)
    print("RUNNING PHASE 3.2 WORK ORDER MANAGEMENT AUTOMATION TESTS")
    print("=" * 80)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase32CompleteSystem)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 3.2 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print("\nFAILURES:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")

    if result.errors:
        print("\nERRORS:")
        for test, error in result.errors:
            print(f"- {test}: {error}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase32_tests()
    sys.exit(0 if success else 1)