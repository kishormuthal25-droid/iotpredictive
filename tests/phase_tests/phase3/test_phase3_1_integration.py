"""
Phase 3.1 Integration Testing Suite
Comprehensive tests for Enhanced Calendar Views, Resource Optimization,
Cost Forecasting, Parts Inventory Management, and Regulatory Compliance Tracking
"""

import unittest
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Phase 3.1 components
from src.maintenance.advanced_resource_optimizer import AdvancedResourceOptimizer, TechnicianProfile, EquipmentProfile, SkillRequirement
from src.maintenance.intelligent_cost_forecaster import IntelligentCostForecaster, CostForecast
from src.maintenance.parts_inventory_manager import PartsInventoryManager, PartSpecification, PartCategory
from src.maintenance.regulatory_compliance_tracker import RegulatoryComplianceTracker, ComplianceStatus, ComplianceCategory
from src.maintenance.phase3_1_integration import Phase31Integration, IntegratedMaintenanceTask


class TestAdvancedResourceOptimizer(unittest.TestCase):
    """Test Advanced Resource Optimization System"""

    def setUp(self):
        """Setup test environment"""
        self.optimizer = AdvancedResourceOptimizer()

    def test_optimizer_initialization(self):
        """Test resource optimizer initialization"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsInstance(self.optimizer.technicians, dict)
        self.assertIsInstance(self.optimizer.equipment, dict)

    def test_technician_management(self):
        """Test technician profile management"""
        technician = TechnicianProfile(
            'test_tech', 'Test Technician',
            {'electrical': 5, 'mechanical': 3},
            ['electrical_cert'], 5.0, 80.0, 1.1, {}, 'morning', 'Zone_A'
        )

        # Add technician
        self.optimizer.add_technician(technician)
        self.assertIn('test_tech', self.optimizer.technicians)
        self.assertEqual(self.optimizer.technicians['test_tech'].name, 'Test Technician')

    def test_equipment_management(self):
        """Test equipment profile management"""
        skill_req = SkillRequirement('electrical', 4, 0.8)
        equipment = EquipmentProfile(
            'test_eq', 'Test Equipment', 'pump', 'Zone_A', 5,
            [skill_req], 3, ['safety'], {}, {}, 1000, 50
        )

        # Add equipment
        self.optimizer.add_equipment(equipment)
        self.assertIn('test_eq', self.optimizer.equipment)
        self.assertEqual(self.optimizer.equipment['test_eq'].name, 'Test Equipment')

    def test_skill_matching(self):
        """Test skill matching algorithm"""
        technician = TechnicianProfile(
            'tech1', 'Tech 1', {'electrical': 5, 'mechanical': 3},
            [], 5.0, 80.0, 1.0, {}, 'morning', 'Zone_A'
        )

        skill_requirements = [
            SkillRequirement('electrical', 4, 0.8),
            SkillRequirement('mechanical', 2, 0.4)
        ]

        score = self.optimizer.calculate_skill_match_score(technician, skill_requirements)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.5)  # Can exceed 1.0 due to over-qualification bonus

    def test_assignment_optimization(self):
        """Test assignment optimization"""
        # Add sample data
        technician = TechnicianProfile(
            'tech1', 'Tech 1', {'electrical': 5}, [], 5.0, 80.0, 1.0, {}, 'morning', 'Zone_A'
        )
        self.optimizer.add_technician(technician)

        tasks = [{
            'task_id': 'task1',
            'equipment_id': 'eq1',
            'duration_hours': 4.0,
            'scheduled_date': datetime.now().date()
        }]

        result = self.optimizer.optimize_assignments(tasks)
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('assignments', result)

    def test_workload_balancing(self):
        """Test workload balancing functionality"""
        assignments = [
            {'technician_id': 'tech1', 'task_duration': 8.0},
            {'technician_id': 'tech2', 'task_duration': 2.0}
        ]

        balanced = self.optimizer.balance_workload(assignments)
        self.assertIsInstance(balanced, list)
        self.assertEqual(len(balanced), len(assignments))


class TestIntelligentCostForecaster(unittest.TestCase):
    """Test Intelligent Cost Forecasting System"""

    def setUp(self):
        """Setup test environment"""
        self.forecaster = IntelligentCostForecaster()

    def test_forecaster_initialization(self):
        """Test cost forecaster initialization"""
        self.assertIsNotNone(self.forecaster)
        self.assertIsInstance(self.forecaster.models, dict)
        self.assertIsInstance(self.forecaster.cost_components, dict)

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now() - timedelta(days=1)

        data = self.forecaster.generate_synthetic_data(start_date, end_date)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('total_cost', data.columns)
        self.assertIn('date', data.columns)

    def test_model_training(self):
        """Test model training"""
        # Generate synthetic data first
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now() - timedelta(days=1)
        self.forecaster.generate_synthetic_data(start_date, end_date)

        # Train models
        metrics = self.forecaster.train_models(test_size=0.3)
        self.assertIsInstance(metrics, dict)

        # Check that at least one model was trained
        self.assertGreater(len(metrics), 0)

        # Verify metrics structure
        for model_name, model_metrics in metrics.items():
            self.assertIn('mae', model_metrics)
            self.assertIn('rmse', model_metrics)
            self.assertIn('r2', model_metrics)

    def test_cost_forecasting(self):
        """Test cost forecasting functionality"""
        # Generate and train with synthetic data
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now() - timedelta(days=1)
        self.forecaster.generate_synthetic_data(start_date, end_date)
        self.forecaster.train_models()

        # Generate forecast
        forecast_date = datetime.now()
        forecast = self.forecaster.forecast_costs(forecast_date, 30)

        self.assertIsInstance(forecast, CostForecast)
        self.assertGreater(forecast.predicted_cost, 0)
        self.assertIsInstance(forecast.component_breakdown, dict)
        self.assertGreaterEqual(forecast.model_confidence, 0.0)
        self.assertLessEqual(forecast.model_confidence, 1.0)

    def test_budget_optimization(self):
        """Test budget optimization"""
        # Setup forecaster with data
        start_date = datetime.now() - timedelta(days=100)
        end_date = datetime.now() - timedelta(days=1)
        self.forecaster.generate_synthetic_data(start_date, end_date)
        self.forecaster.train_models()

        current_budget = 50000.0
        optimization = self.forecaster.optimize_budget(current_budget, 90)

        self.assertIsNotNone(optimization)
        self.assertEqual(optimization.current_budget, current_budget)
        self.assertGreaterEqual(optimization.recommended_budget, 0)
        self.assertIsInstance(optimization.recommendations, list)


class TestPartsInventoryManager(unittest.TestCase):
    """Test Parts Inventory Management System"""

    def setUp(self):
        """Setup test environment"""
        self.manager = PartsInventoryManager()

    def test_manager_initialization(self):
        """Test inventory manager initialization"""
        self.assertIsNotNone(self.manager)
        self.assertIsInstance(self.manager.parts_catalog, dict)
        self.assertIsInstance(self.manager.inventory, dict)
        self.assertIsInstance(self.manager.suppliers, dict)

    def test_parts_catalog_management(self):
        """Test parts catalog management"""
        part_spec = PartSpecification(
            'TEST_001', 'TEST-PART', 'Test Part',
            'Test part for unit testing',
            PartCategory.STANDARD, 'TestCorp', 'TEST-001',
            {'spec1': 'value1'}, ['EQ_TEST'], 1.0
        )

        # Add part to catalog
        success = self.manager.add_part_to_catalog(part_spec)
        self.assertTrue(success)
        self.assertIn('TEST_001', self.manager.parts_catalog)

    def test_inventory_level_updates(self):
        """Test inventory level updates"""
        # Use existing part from sample data
        part_id = list(self.manager.inventory.keys())[0]
        initial_stock = self.manager.inventory[part_id].current_stock

        # Update inventory
        success = self.manager.update_inventory_levels(part_id, -2, "usage")
        self.assertTrue(success)

        # Verify update
        new_stock = self.manager.inventory[part_id].current_stock
        self.assertEqual(new_stock, max(0, initial_stock - 2))

    def test_demand_forecasting(self):
        """Test demand forecasting"""
        # Use existing part
        part_id = list(self.manager.inventory.keys())[0]

        forecast = self.manager.forecast_demand(part_id, 60)
        self.assertIsNotNone(forecast)
        self.assertEqual(forecast.part_id, part_id)
        self.assertGreater(forecast.predicted_demand, 0)
        self.assertIsInstance(forecast.confidence_interval, tuple)

    def test_inventory_optimization(self):
        """Test inventory optimization"""
        recommendations = self.manager.optimize_inventory_levels()
        self.assertIsInstance(recommendations, dict)

        # Check recommendation structure
        for part_id, recommendation in recommendations.items():
            self.assertIn('recommended_eoq', recommendation)
            self.assertIn('recommended_reorder_point', recommendation)
            self.assertIn('optimization_potential', recommendation)

    def test_purchase_order_creation(self):
        """Test purchase order creation"""
        part_requirements = [
            {'part_id': list(self.manager.inventory.keys())[0], 'quantity': 5}
        ]

        order = self.manager.create_purchase_order(part_requirements)
        self.assertIsNotNone(order)
        self.assertGreater(order.total_value, 0)
        self.assertEqual(len(order.items), 1)

    def test_reorder_report_generation(self):
        """Test reorder report generation"""
        # Artificially lower stock levels to trigger reorders
        for part_id, item in self.manager.inventory.items():
            item.current_stock = item.reorder_point - 1

        report = self.manager.generate_reorder_report()
        self.assertIsInstance(report, list)
        self.assertGreater(len(report), 0)

        # Check report structure
        for recommendation in report:
            self.assertIn('part_id', recommendation)
            self.assertIn('recommended_quantity', recommendation)
            self.assertIn('urgency', recommendation)


class TestRegulatoryComplianceTracker(unittest.TestCase):
    """Test Regulatory Compliance Tracking System"""

    def setUp(self):
        """Setup test environment"""
        self.tracker = RegulatoryComplianceTracker()

    def test_tracker_initialization(self):
        """Test compliance tracker initialization"""
        self.assertIsNotNone(self.tracker)
        self.assertIsInstance(self.tracker.requirements, dict)
        self.assertIsInstance(self.tracker.compliance_records, dict)
        self.assertIsInstance(self.tracker.certifications, dict)

        # Should have sample requirements
        self.assertGreater(len(self.tracker.requirements), 0)

    def test_compliance_record_creation(self):
        """Test compliance record creation"""
        # Use existing requirement
        req_id = list(self.tracker.requirements.keys())[0]

        record = self.tracker.create_compliance_record(
            req_id, 'EQ_001', 'test_auditor',
            ['Test finding'], ComplianceStatus.COMPLIANT
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.requirement_id, req_id)
        self.assertEqual(record.status, ComplianceStatus.COMPLIANT)
        self.assertIn(record.record_id, self.tracker.compliance_records)

    def test_compliance_check_conduct(self):
        """Test conducting compliance checks"""
        req_id = list(self.tracker.requirements.keys())[0]

        record = self.tracker.conduct_compliance_check(
            req_id, 'EQ_001', 'inspector', 95.0,
            ['Minor documentation gap'], ['evidence.pdf']
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.score, 95.0)
        self.assertEqual(record.status, ComplianceStatus.COMPLIANT)

    def test_audit_scheduling(self):
        """Test audit scheduling"""
        from src.maintenance.regulatory_compliance_tracker import AuditType

        audit_date = datetime.now() + timedelta(days=30)
        audit = self.tracker.schedule_audit(
            AuditType.INTERNAL, 'Test Internal Audit',
            'test_auditor', audit_date, ['EQ_001'], 2
        )

        self.assertIsNotNone(audit)
        self.assertEqual(audit.audit_type, AuditType.INTERNAL)
        self.assertEqual(audit.auditor, 'test_auditor')
        self.assertIn(audit.audit_id, self.tracker.audits)

    def test_certification_status_update(self):
        """Test certification status updates"""
        # Use existing certification
        cert_id = list(self.tracker.certifications.keys())[0]

        success = self.tracker.update_certification_status(cert_id)
        self.assertTrue(success)

        # Verify status was updated
        certification = self.tracker.certifications[cert_id]
        self.assertIn(certification.status, [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.WARNING,
            ComplianceStatus.EXPIRED
        ])

    def test_compliance_report_generation(self):
        """Test compliance report generation"""
        # Create some test records first
        req_id = list(self.tracker.requirements.keys())[0]
        self.tracker.conduct_compliance_check(req_id, 'EQ_001', 'auditor', 90.0)

        report = self.tracker.generate_compliance_report()
        self.assertIsInstance(report, dict)
        self.assertIn('compliance_summary', report)
        self.assertIn('certification_status', report)
        self.assertIn('recommendations', report)

    def test_dashboard_data_retrieval(self):
        """Test dashboard data retrieval"""
        dashboard_data = self.tracker.get_compliance_dashboard_data()
        self.assertIsInstance(dashboard_data, dict)
        self.assertIn('current_status', dashboard_data)
        self.assertIn('upcoming_deadlines', dashboard_data)
        self.assertIn('performance_metrics', dashboard_data)

    def test_auto_status_updates(self):
        """Test automatic status updates"""
        # Should not raise exceptions
        self.tracker.auto_update_statuses()

        # Verify system is still functional
        self.assertIsInstance(self.tracker.certifications, dict)
        self.assertIsInstance(self.tracker.alerts, dict)


class TestPhase31Integration(unittest.TestCase):
    """Test Phase 3.1 Integration System"""

    def setUp(self):
        """Setup test environment"""
        self.integration = Phase31Integration()

    def test_integration_initialization(self):
        """Test Phase 3.1 integration initialization"""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.resource_optimizer)
        self.assertIsNotNone(self.integration.cost_forecaster)
        self.assertIsNotNone(self.integration.inventory_manager)
        self.assertIsNotNone(self.integration.compliance_tracker)

    def test_integrated_task_creation(self):
        """Test integrated maintenance task creation"""
        task_data = {
            'task_id': 'INTEGRATION_TEST_001',
            'equipment_id': 'EQ_001',
            'task_type': 'preventive',
            'priority': 'medium',
            'scheduled_date': datetime.now() + timedelta(days=1),
            'duration_hours': 4.0,
            'required_parts': [{'part_id': 'BRG001', 'quantity': 1}]
        }

        integrated_task = self.integration.create_integrated_maintenance_task(task_data)
        self.assertIsNotNone(integrated_task)
        self.assertIsInstance(integrated_task, IntegratedMaintenanceTask)
        self.assertEqual(integrated_task.task_id, 'INTEGRATION_TEST_001')
        self.assertGreaterEqual(integrated_task.optimization_score, 0)
        self.assertLessEqual(integrated_task.optimization_score, 100)

    def test_schedule_optimization(self):
        """Test integrated schedule optimization"""
        tasks = [
            {
                'task_id': 'OPT_TEST_001',
                'equipment_id': 'EQ_001',
                'task_type': 'preventive',
                'priority': 'medium',
                'scheduled_date': datetime.now() + timedelta(days=1),
                'duration_hours': 4.0,
                'required_parts': []
            },
            {
                'task_id': 'OPT_TEST_002',
                'equipment_id': 'EQ_002',
                'task_type': 'corrective',
                'priority': 'high',
                'scheduled_date': datetime.now() + timedelta(days=2),
                'duration_hours': 6.0,
                'required_parts': []
            }
        ]

        result = self.integration.optimize_integrated_schedule(tasks, 30)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.optimized_tasks), 2)
        self.assertGreaterEqual(result.total_cost_savings, 0)
        self.assertGreaterEqual(result.confidence_score, 0)
        self.assertLessEqual(result.confidence_score, 100)

    def test_dashboard_data_integration(self):
        """Test integrated dashboard data retrieval"""
        dashboard_data = self.integration.get_integration_dashboard_data()
        self.assertIsInstance(dashboard_data, dict)
        self.assertIn('integration_overview', dashboard_data)
        self.assertIn('resource_optimization', dashboard_data)
        self.assertIn('cost_forecasting', dashboard_data)
        self.assertIn('inventory_status', dashboard_data)
        self.assertIn('compliance_status', dashboard_data)

    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking"""
        # Create and optimize some tasks to generate metrics
        tasks = [{
            'task_id': 'PERF_TEST_001',
            'equipment_id': 'EQ_001',
            'task_type': 'preventive',
            'priority': 'medium',
            'scheduled_date': datetime.now() + timedelta(days=1),
            'duration_hours': 2.0,
            'required_parts': []
        }]

        result = self.integration.optimize_integrated_schedule(tasks)
        self.assertIsNotNone(result)

        # Check performance metrics
        metrics = self.integration.performance_metrics
        self.assertIn('total_optimizations', metrics)
        self.assertIn('average_cost_savings', metrics)
        self.assertIn('average_efficiency_gain', metrics)
        self.assertIn('integration_success_rate', metrics)

    def test_comprehensive_report_generation(self):
        """Test comprehensive Phase 3.1 report generation"""
        report = self.integration.generate_phase31_report()
        self.assertIsInstance(report, dict)
        self.assertIn('report_metadata', report)
        self.assertIn('executive_summary', report)
        self.assertIn('component_performance', report)
        self.assertIn('integration_analysis', report)
        self.assertIn('recommendations', report)

    def test_component_synergy_calculation(self):
        """Test component synergy calculation"""
        # Create a task to have data for synergy calculation
        task_data = {
            'equipment_id': 'EQ_001',
            'task_type': 'preventive',
            'priority': 'medium',
            'scheduled_date': datetime.now() + timedelta(days=1),
            'duration_hours': 4.0,
            'required_parts': [{'part_id': 'BRG001', 'quantity': 1}]
        }

        self.integration.create_integrated_maintenance_task(task_data)

        synergy_score = self.integration._calculate_component_synergy()
        self.assertIsInstance(synergy_score, float)
        self.assertGreaterEqual(synergy_score, 0.0)
        self.assertLessEqual(synergy_score, 100.0)


class TestPhase31SystemIntegration(unittest.TestCase):
    """Test end-to-end Phase 3.1 system integration"""

    def setUp(self):
        """Setup comprehensive test environment"""
        self.integration = Phase31Integration()

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Step 1: Create maintenance tasks
        tasks = [
            {
                'equipment_id': 'EQ_001',
                'task_type': 'preventive',
                'priority': 'medium',
                'scheduled_date': datetime.now() + timedelta(days=1),
                'duration_hours': 4.0,
                'required_parts': [{'part_id': 'BRG001', 'quantity': 2}]
            },
            {
                'equipment_id': 'EQ_002',
                'task_type': 'corrective',
                'priority': 'high',
                'scheduled_date': datetime.now() + timedelta(days=2),
                'duration_hours': 6.0,
                'required_parts': [{'part_id': 'SNS001', 'quantity': 1}]
            }
        ]

        # Step 2: Optimize schedule
        optimization_result = self.integration.optimize_integrated_schedule(tasks)
        self.assertIsNotNone(optimization_result)

        # Step 3: Verify all components were utilized
        self.assertEqual(len(optimization_result.optimized_tasks), 2)

        # Check that each task has all enhancement data
        for task in optimization_result.optimized_tasks:
            self.assertIsNotNone(task.assigned_technician)
            self.assertGreater(task.estimated_cost, 0)
            self.assertIsInstance(task.parts_availability, dict)
            self.assertIsInstance(task.compliance_requirements, list)

        # Step 4: Generate comprehensive report
        report = self.integration.generate_phase31_report()
        self.assertIn('executive_summary', report)
        self.assertGreater(report['executive_summary']['total_integrated_tasks'], 0)

    def test_data_consistency_across_components(self):
        """Test data consistency across all Phase 3.1 components"""
        # Create task and verify data flows correctly between components
        task_data = {
            'equipment_id': 'EQ_001',
            'task_type': 'preventive',
            'priority': 'critical',
            'scheduled_date': datetime.now() + timedelta(days=1),
            'duration_hours': 8.0,
            'required_parts': [{'part_id': 'BRG001', 'quantity': 3}]
        }

        integrated_task = self.integration.create_integrated_maintenance_task(task_data)
        self.assertIsNotNone(integrated_task)

        # Verify resource optimization data
        self.assertIsInstance(integrated_task.skill_match_score, float)
        self.assertGreaterEqual(integrated_task.skill_match_score, 0.0)

        # Verify cost forecasting data
        self.assertGreater(integrated_task.estimated_cost, 0)
        self.assertIsInstance(integrated_task.cost_breakdown, dict)

        # Verify parts inventory data
        self.assertIsInstance(integrated_task.parts_availability, dict)
        self.assertGreater(integrated_task.parts_cost, 0)

        # Verify compliance data
        self.assertIsInstance(integrated_task.compliance_requirements, list)
        self.assertIn(integrated_task.compliance_status, [
            ComplianceStatus.COMPLIANT,
            ComplianceStatus.WARNING,
            ComplianceStatus.NON_COMPLIANT,
            ComplianceStatus.PENDING_REVIEW,
            ComplianceStatus.NOT_APPLICABLE
        ])

    def test_performance_under_load(self):
        """Test system performance under load"""
        # Create a larger number of tasks to test performance
        large_task_set = []
        for i in range(20):
            task = {
                'equipment_id': f'EQ_{i % 3 + 1:03d}',
                'task_type': ['preventive', 'corrective', 'predictive'][i % 3],
                'priority': ['low', 'medium', 'high', 'critical'][i % 4],
                'scheduled_date': datetime.now() + timedelta(days=i % 7 + 1),
                'duration_hours': float(2 + i % 6),
                'required_parts': [{'part_id': 'BRG001', 'quantity': i % 3 + 1}] if i % 2 == 0 else []
            }
            large_task_set.append(task)

        # Measure optimization time
        start_time = time.time()
        result = self.integration.optimize_integrated_schedule(large_task_set)
        optimization_time = time.time() - start_time

        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(len(result.optimized_tasks), 20)
        self.assertLess(optimization_time, 30.0)  # Should complete within 30 seconds

        # Verify optimization quality
        avg_score = np.mean([task.optimization_score for task in result.optimized_tasks])
        self.assertGreater(avg_score, 50.0)  # Average score should be reasonable


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add Phase 3.1 component tests
    suite.addTest(unittest.makeSuite(TestAdvancedResourceOptimizer))
    suite.addTest(unittest.makeSuite(TestIntelligentCostForecaster))
    suite.addTest(unittest.makeSuite(TestPartsInventoryManager))
    suite.addTest(unittest.makeSuite(TestRegulatoryComplianceTracker))
    suite.addTest(unittest.makeSuite(TestPhase31Integration))
    suite.addTest(unittest.makeSuite(TestPhase31SystemIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*80}")
    print(f"PHASE 3.1 INTEGRATION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    if success_rate >= 95:
        print("PHASE 3.1 INTEGRATION READY FOR DEPLOYMENT")
    elif success_rate >= 80:
        print("PHASE 3.1 INTEGRATION NEEDS MINOR FIXES")
    else:
        print("PHASE 3.1 INTEGRATION NEEDS SIGNIFICANT WORK")

    # Exit with appropriate code
    exit_code = 0 if len(result.failures) + len(result.errors) == 0 else 1
    sys.exit(exit_code)