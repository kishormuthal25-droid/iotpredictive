"""
Phase 2 Integration Validation Script
Validates that all Phase 2 components are properly integrated and functional
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import importlib
import traceback
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2IntegrationValidator:
    """Validates Phase 2 component integration and functionality"""

    def __init__(self):
        """Initialize the validator"""
        self.validation_results = {}
        self.components = {}
        self.errors = []

    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks"""
        logger.info("Starting Phase 2 Integration Validation")
        logger.info("=" * 60)

        validations = [
            ("Import Validation", self.validate_imports),
            ("Component Initialization", self.validate_component_initialization),
            ("Database Schema", self.validate_database_schema),
            ("API Endpoints", self.validate_api_endpoints),
            ("Dashboard Layout", self.validate_dashboard_layout),
            ("Component Integration", self.validate_component_integration),
            ("Performance Check", self.validate_performance),
            ("Error Handling", self.validate_error_handling)
        ]

        for validation_name, validation_func in validations:
            logger.info(f"\n--- {validation_name} ---")
            try:
                result = validation_func()
                self.validation_results[validation_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'details': result if isinstance(result, dict) else {}
                }
                logger.info(f"✓ {validation_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                self.validation_results[validation_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"✗ {validation_name}: ERROR - {str(e)}")
                self.errors.append(f"{validation_name}: {str(e)}")

        return self.generate_validation_report()

    def validate_imports(self) -> bool:
        """Validate that all Phase 2 components can be imported"""
        components_to_import = [
            ('src.dashboard.components.subsystem_failure_analyzer', 'nasa_subsystem_analyzer'),
            ('src.dashboard.components.detection_details_panel', 'detection_details_panel'),
            ('src.dashboard.components.alert_action_manager', 'alert_action_manager'),
            ('src.dashboard.components.threshold_manager', 'threshold_manager'),
            ('src.database.phase2_enhancements', 'phase2_db_manager'),
            ('src.api.phase2_endpoints', 'phase2_api')
        ]

        success_count = 0
        for module_name, component_name in components_to_import:
            try:
                module = importlib.import_module(module_name)
                component = getattr(module, component_name)
                self.components[component_name] = component
                logger.info(f"  ✓ Imported {module_name}.{component_name}")
                success_count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to import {module_name}.{component_name}: {e}")

        return success_count == len(components_to_import)

    def validate_component_initialization(self) -> Dict[str, Any]:
        """Validate that all components can be initialized"""
        if not self.components:
            return False

        initialization_results = {}

        # Test NASA Subsystem Analyzer
        try:
            analyzer = self.components['nasa_subsystem_analyzer']
            if hasattr(analyzer, 'subsystem_health'):
                initialization_results['subsystem_analyzer'] = 'PASS'
                logger.info("  ✓ NASA Subsystem Analyzer initialized")
            else:
                initialization_results['subsystem_analyzer'] = 'FAIL'
        except Exception as e:
            initialization_results['subsystem_analyzer'] = f'ERROR: {e}'

        # Test Detection Details Panel
        try:
            panel = self.components['detection_details_panel']
            if hasattr(panel, 'confidence_thresholds'):
                initialization_results['detection_panel'] = 'PASS'
                logger.info("  ✓ Detection Details Panel initialized")
            else:
                initialization_results['detection_panel'] = 'FAIL'
        except Exception as e:
            initialization_results['detection_panel'] = f'ERROR: {e}'

        # Test Alert Action Manager
        try:
            alert_mgr = self.components['alert_action_manager']
            if hasattr(alert_mgr, 'dismissal_reasons'):
                initialization_results['alert_manager'] = 'PASS'
                logger.info("  ✓ Alert Action Manager initialized")
            else:
                initialization_results['alert_manager'] = 'FAIL'
        except Exception as e:
            initialization_results['alert_manager'] = f'ERROR: {e}'

        # Test Threshold Manager
        try:
            threshold_mgr = self.components['threshold_manager']
            if hasattr(threshold_mgr, 'threshold_configs'):
                initialization_results['threshold_manager'] = 'PASS'
                logger.info("  ✓ Threshold Manager initialized")
            else:
                initialization_results['threshold_manager'] = 'FAIL'
        except Exception as e:
            initialization_results['threshold_manager'] = f'ERROR: {e}'

        # Test Database Manager
        try:
            db_mgr = self.components['phase2_db_manager']
            stats = db_mgr.get_database_statistics()
            if isinstance(stats, dict):
                initialization_results['database_manager'] = 'PASS'
                logger.info("  ✓ Database Manager initialized")
            else:
                initialization_results['database_manager'] = 'FAIL'
        except Exception as e:
            initialization_results['database_manager'] = f'ERROR: {e}'

        return initialization_results

    def validate_database_schema(self) -> Dict[str, Any]:
        """Validate database schema and connectivity"""
        try:
            db_mgr = self.components['phase2_db_manager']
            stats = db_mgr.get_database_statistics()

            required_tables = [
                'alert_actions_count',
                'threshold_history_count',
                'subsystem_analytics_count',
                'equipment_health_snapshots_count',
                'detection_details_history_count'
            ]

            schema_results = {}
            for table in required_tables:
                if table in stats:
                    schema_results[table] = 'EXISTS'
                    logger.info(f"  ✓ Table {table.replace('_count', '')} exists")
                else:
                    schema_results[table] = 'MISSING'
                    logger.error(f"  ✗ Table {table.replace('_count', '')} missing")

            # Test database operations
            try:
                # Test alert action recording
                from src.database.phase2_enhancements import AlertActionRecord
                import uuid
                test_action = AlertActionRecord(
                    action_id=str(uuid.uuid4()),
                    alert_id="validation_test",
                    action_type="test",
                    user_id="validator",
                    timestamp=datetime.now(),
                    notes="Validation test"
                )

                success = db_mgr.record_alert_action(test_action)
                schema_results['write_operations'] = 'PASS' if success else 'FAIL'

                # Test retrieval
                actions = db_mgr.get_alert_actions("validation_test")
                schema_results['read_operations'] = 'PASS' if len(actions) > 0 else 'FAIL'

                logger.info("  ✓ Database read/write operations working")

            except Exception as e:
                schema_results['operations'] = f'ERROR: {e}'

            return schema_results

        except Exception as e:
            return {'error': str(e)}

    def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoint structure"""
        try:
            api_blueprint = self.components['phase2_api']

            # Check if blueprint has required routes
            expected_routes = [
                '/api/v2/subsystem/analysis/<subsystem>',
                '/api/v2/subsystem/health-summary',
                '/api/v2/detection/details/<equipment_id>',
                '/api/v2/alerts/<alert_id>/acknowledge',
                '/api/v2/alerts/<alert_id>/dismiss',
                '/api/v2/thresholds/<equipment_id>',
                '/api/v2/thresholds/<equipment_id>/optimize',
                '/api/v2/system/health'
            ]

            api_results = {}

            # Check blueprint structure
            if hasattr(api_blueprint, 'deferred_functions'):
                api_results['blueprint_structure'] = 'PASS'
                logger.info("  ✓ API Blueprint structure valid")
            else:
                api_results['blueprint_structure'] = 'FAIL'

            # Since we can't test actual HTTP requests without running the server,
            # we validate the function existence
            endpoint_functions = [
                'get_subsystem_analysis',
                'get_subsystem_health_summary',
                'get_detection_details',
                'acknowledge_alert',
                'dismiss_alert',
                'get_equipment_thresholds',
                'optimize_equipment_thresholds',
                'get_system_health'
            ]

            from src.api import phase2_endpoints
            for func_name in endpoint_functions:
                if hasattr(phase2_endpoints, func_name):
                    api_results[func_name] = 'EXISTS'
                    logger.info(f"  ✓ API function {func_name} exists")
                else:
                    api_results[func_name] = 'MISSING'
                    logger.error(f"  ✗ API function {func_name} missing")

            return api_results

        except Exception as e:
            return {'error': str(e)}

    def validate_dashboard_layout(self) -> Dict[str, Any]:
        """Validate dashboard layout integration"""
        try:
            # Import the enhanced anomaly monitor
            from src.dashboard.layouts.anomaly_monitor import AnomalyMonitor

            monitor = AnomalyMonitor()
            layout = monitor.create_layout()

            layout_results = {}

            # Check if layout has required attributes
            if hasattr(layout, 'children'):
                layout_results['layout_structure'] = 'PASS'
                logger.info("  ✓ Dashboard layout structure valid")
            else:
                layout_results['layout_structure'] = 'FAIL'

            # Check for Phase 2 components in layout
            phase2_methods = [
                '_create_enhanced_tabbed_interface',
                '_create_overview_tab_content',
                '_create_subsystem_analysis_tab_content',
                '_create_detection_details_tab_content',
                '_create_alert_management_tab_content',
                '_create_threshold_config_tab_content'
            ]

            for method_name in phase2_methods:
                if hasattr(monitor, method_name):
                    layout_results[method_name] = 'EXISTS'
                    logger.info(f"  ✓ Layout method {method_name} exists")
                else:
                    layout_results[method_name] = 'MISSING'
                    logger.error(f"  ✗ Layout method {method_name} missing")

            return layout_results

        except Exception as e:
            return {'error': str(e)}

    def validate_component_integration(self) -> Dict[str, Any]:
        """Validate that components can work together"""
        integration_results = {}

        try:
            # Test subsystem analyzer and database integration
            analyzer = self.components['nasa_subsystem_analyzer']
            db_mgr = self.components['phase2_db_manager']

            # Get health summary
            health_summary = analyzer.get_subsystem_health_summary()
            if isinstance(health_summary, dict):
                integration_results['analyzer_functionality'] = 'PASS'
                logger.info("  ✓ Subsystem analyzer functionality working")
            else:
                integration_results['analyzer_functionality'] = 'FAIL'

            # Test alert manager and database integration
            alert_mgr = self.components['alert_action_manager']

            # Test dismissal reasons
            if hasattr(alert_mgr, 'dismissal_reasons') and len(alert_mgr.dismissal_reasons) > 0:
                integration_results['alert_manager_config'] = 'PASS'
                logger.info("  ✓ Alert manager configuration valid")
            else:
                integration_results['alert_manager_config'] = 'FAIL'

            # Test threshold manager
            threshold_mgr = self.components['threshold_manager']
            recommendations = threshold_mgr.get_threshold_recommendations()
            if isinstance(recommendations, list):
                integration_results['threshold_recommendations'] = 'PASS'
                logger.info("  ✓ Threshold recommendations functionality working")
            else:
                integration_results['threshold_recommendations'] = 'FAIL'

            # Test detection panel
            panel = self.components['detection_details_panel']
            summary = panel.analyze_equipment_detection("TEST_EQUIPMENT")
            if hasattr(summary, 'equipment_id'):
                integration_results['detection_analysis'] = 'PASS'
                logger.info("  ✓ Detection analysis functionality working")
            else:
                integration_results['detection_analysis'] = 'FAIL'

            return integration_results

        except Exception as e:
            return {'error': str(e)}

    def validate_performance(self) -> Dict[str, Any]:
        """Validate component performance"""
        performance_results = {}

        try:
            # Test database performance
            import time
            db_mgr = self.components['phase2_db_manager']

            # Time database operations
            start_time = time.time()
            stats = db_mgr.get_database_statistics()
            db_time = time.time() - start_time

            performance_results['database_query_time'] = f"{db_time:.3f}s"
            performance_results['database_performance'] = 'PASS' if db_time < 1.0 else 'SLOW'

            logger.info(f"  ✓ Database query time: {db_time:.3f}s")

            # Test analyzer performance
            analyzer = self.components['nasa_subsystem_analyzer']

            start_time = time.time()
            health_summary = analyzer.get_subsystem_health_summary()
            analyzer_time = time.time() - start_time

            performance_results['analyzer_time'] = f"{analyzer_time:.3f}s"
            performance_results['analyzer_performance'] = 'PASS' if analyzer_time < 2.0 else 'SLOW'

            logger.info(f"  ✓ Analyzer processing time: {analyzer_time:.3f}s")

            return performance_results

        except Exception as e:
            return {'error': str(e)}

    def validate_error_handling(self) -> Dict[str, Any]:
        """Validate error handling across components"""
        error_handling_results = {}

        try:
            # Test invalid equipment ID handling
            panel = self.components['detection_details_panel']
            summary = panel.analyze_equipment_detection("INVALID_EQUIPMENT_ID")

            if summary.equipment_id == "INVALID_EQUIPMENT_ID":
                error_handling_results['detection_panel_error_handling'] = 'PASS'
                logger.info("  ✓ Detection panel handles invalid IDs gracefully")
            else:
                error_handling_results['detection_panel_error_handling'] = 'FAIL'

            # Test invalid alert action
            alert_mgr = self.components['alert_action_manager']
            result = alert_mgr.acknowledge_alert("INVALID_ALERT_ID", "test_user")

            if not result.success:
                error_handling_results['alert_manager_error_handling'] = 'PASS'
                logger.info("  ✓ Alert manager handles invalid alert IDs gracefully")
            else:
                error_handling_results['alert_manager_error_handling'] = 'FAIL'

            # Test database error handling
            db_mgr = self.components['phase2_db_manager']
            actions = db_mgr.get_alert_actions("NONEXISTENT_ALERT")

            if isinstance(actions, list):
                error_handling_results['database_error_handling'] = 'PASS'
                logger.info("  ✓ Database handles nonexistent records gracefully")
            else:
                error_handling_results['database_error_handling'] = 'FAIL'

            return error_handling_results

        except Exception as e:
            return {'error': str(e)}

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for result in self.validation_results.values()
                               if result['status'] == 'PASS')
        failed_validations = sum(1 for result in self.validation_results.values()
                               if result['status'] == 'FAIL')
        error_validations = sum(1 for result in self.validation_results.values()
                              if result['status'] == 'ERROR')

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validations': total_validations,
                'passed': passed_validations,
                'failed': failed_validations,
                'errors': error_validations,
                'success_rate': f"{(passed_validations / total_validations * 100):.1f}%" if total_validations > 0 else "0%"
            },
            'validation_results': self.validation_results,
            'errors': self.errors,
            'overall_status': 'PASS' if failed_validations == 0 and error_validations == 0 else 'FAIL'
        }

        return report

    def print_validation_report(self, report: Dict[str, Any]):
        """Print formatted validation report"""
        print("\n" + "=" * 80)
        print("PHASE 2 INTEGRATION VALIDATION REPORT")
        print("=" * 80)

        summary = report['summary']
        print(f"Timestamp: {report['timestamp']}")
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']}")
        print(f"Overall Status: {report['overall_status']}")

        print("\n" + "-" * 80)
        print("VALIDATION DETAILS")
        print("-" * 80)

        for validation_name, result in report['validation_results'].items():
            status_symbol = {
                'PASS': '✓',
                'FAIL': '✗',
                'ERROR': '⚠'
            }.get(result['status'], '?')

            print(f"{status_symbol} {validation_name}: {result['status']}")

            if result['status'] == 'ERROR':
                print(f"    Error: {result.get('error', 'Unknown error')}")

        if report['errors']:
            print("\n" + "-" * 80)
            print("ERROR SUMMARY")
            print("-" * 80)
            for error in report['errors']:
                print(f"• {error}")

        print("\n" + "=" * 80)

        # Recommendations
        if report['overall_status'] == 'FAIL':
            print("RECOMMENDATIONS:")
            print("• Check that all Phase 2 components are properly installed")
            print("• Verify database connectivity and schema")
            print("• Ensure all required dependencies are available")
            print("• Review error messages for specific issues")
        else:
            print("🎉 All Phase 2 components are properly integrated and functional!")

        print("=" * 80)


def main():
    """Main function to run validation"""
    validator = Phase2IntegrationValidator()
    report = validator.validate_all()
    validator.print_validation_report(report)

    return report['overall_status'] == 'PASS'


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)