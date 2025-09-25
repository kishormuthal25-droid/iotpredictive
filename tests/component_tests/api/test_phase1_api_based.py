#!/usr/bin/env python3
"""
Phase 1 API-Based Testing Suite
Tests Core Dashboard Infrastructure functionality without browser automation
"""

import sys
import os
import warnings
import unittest
import time
import requests
import json
from pathlib import Path
from datetime import datetime, timedelta
import threading
import subprocess
import signal
import importlib.util

# Add project root to path
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

class Phase1APITestSuite(unittest.TestCase):
    """API-based Phase 1 testing suite"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.dashboard_url = "http://localhost:8060"
        cls.dashboard_process = None
        cls.test_results = {
            'infrastructure': {},
            'components': {},
            'data_pipeline': {},
            'performance': {},
            'errors': []
        }

        print("Starting Phase 1 API-Based Testing...")
        print("=" * 60)

        # Test project structure
        cls._test_project_structure()

        # Test component imports
        cls._test_component_imports()

        # Start dashboard for API testing
        cls._start_dashboard_for_testing()

    @classmethod
    def _test_project_structure(cls):
        """Test project structure and file existence"""
        print("\nTesting Project Structure...")

        required_files = [
            "launch_real_data_dashboard.py",
            "src/dashboard/app.py",
            "src/dashboard/layouts",
            "src/dashboard/components",
            "src/data_ingestion",
            "src/anomaly_detection",
            "config"
        ]

        structure_results = {}

        for file_path in required_files:
            if os.path.exists(file_path):
                structure_results[file_path] = "EXISTS"
                print(f"[PASS] {file_path}: EXISTS")
            else:
                structure_results[file_path] = "MISSING"
                print(f"[FAIL] {file_path}: MISSING")

        cls.test_results['infrastructure']['project_structure'] = structure_results

    @classmethod
    def _test_component_imports(cls):
        """Test component import capabilities"""
        print("\n Testing Component Imports...")

        components_to_test = [
            ("src.dashboard.app", "Dashboard App"),
            ("src.data_ingestion.nasa_data_service", "NASA Data Service"),
            ("src.anomaly_detection.nasa_anomaly_engine", "Anomaly Engine"),
            ("src.dashboard.components.dropdown_manager", "Dropdown Manager"),
            ("src.dashboard.components.chart_manager", "Chart Manager"),
            ("src.dashboard.components.anomaly_heatmap", "Anomaly Heatmap"),
            ("src.dashboard.components.pipeline_status_monitor", "Pipeline Status Monitor")
        ]

        import_results = {}

        for module_path, component_name in components_to_test:
            try:
                spec = importlib.util.spec_from_file_location(
                    module_path.replace('.', '_'),
                    module_path.replace('.', '/') + '.py'
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    import_results[component_name] = "IMPORT_SUCCESS"
                    print(f" {component_name}: IMPORT_SUCCESS")
                else:
                    import_results[component_name] = "SPEC_FAILED"
                    print(f" {component_name}: SPEC_FAILED")
            except Exception as e:
                import_results[component_name] = f"IMPORT_FAILED: {str(e)[:50]}"
                print(f" {component_name}: IMPORT_FAILED - {str(e)[:50]}")

        cls.test_results['components']['import_tests'] = import_results

    @classmethod
    def _start_dashboard_for_testing(cls):
        """Start dashboard for API testing"""
        try:
            print("\n Starting dashboard for API testing...")

            # Check if dashboard is already running
            try:
                response = requests.get(cls.dashboard_url, timeout=5)
                if response.status_code == 200:
                    print(" Dashboard already running")
                    return
            except requests.RequestException:
                pass

            # Start dashboard process
            cls.dashboard_process = subprocess.Popen(
                [sys.executable, "launch_real_data_dashboard.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            # Wait for dashboard to start
            max_attempts = 20
            for attempt in range(max_attempts):
                try:
                    response = requests.get(cls.dashboard_url, timeout=5)
                    if response.status_code == 200:
                        print(" Dashboard started successfully")
                        time.sleep(5)  # Additional wait for full initialization
                        return
                except requests.RequestException:
                    pass

                print(f" Waiting for dashboard... ({attempt + 1}/{max_attempts})")
                time.sleep(3)

            print(" Dashboard startup timeout - proceeding with limited testing")

        except Exception as e:
            print(f" Failed to start dashboard: {e}")
            cls.test_results['errors'].append(f"Dashboard startup failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        print("\n Cleaning up test environment...")

        if cls.dashboard_process:
            try:
                if os.name == 'nt':  # Windows
                    cls.dashboard_process.terminate()
                else:  # Unix-like
                    os.killpg(os.getpgid(cls.dashboard_process.pid), signal.SIGTERM)
                cls.dashboard_process.wait(timeout=10)
                print(" Dashboard process terminated")
            except Exception as e:
                print(f" Error terminating dashboard: {e}")

        # Generate final report
        cls._generate_test_report()

    def test_01_dashboard_accessibility(self):
        """Test dashboard basic accessibility"""
        print("\n Testing Dashboard Accessibility...")

        try:
            # Test if dashboard is reachable
            response = requests.get(self.dashboard_url, timeout=10)

            if response.status_code == 200:
                self.test_results['infrastructure']['dashboard_accessible'] = 'PASS'
                print(" Dashboard Accessibility: PASS")

                # Check if response contains expected content
                content = response.text.lower()
                expected_elements = ['dash', 'plotly', 'bootstrap', 'equipment', 'anomaly']

                found_elements = []
                for element in expected_elements:
                    if element in content:
                        found_elements.append(element)

                self.test_results['infrastructure']['dashboard_content'] = {
                    'expected': expected_elements,
                    'found': found_elements,
                    'coverage': f"{len(found_elements)}/{len(expected_elements)}"
                }

                print(f" Content elements found: {len(found_elements)}/{len(expected_elements)}")

            else:
                self.test_results['infrastructure']['dashboard_accessible'] = f'FAIL - Status {response.status_code}'
                print(f" Dashboard Accessibility: FAIL - Status {response.status_code}")

        except Exception as e:
            self.test_results['infrastructure']['dashboard_accessible'] = f'FAIL - {str(e)}'
            print(f" Dashboard Accessibility: FAIL - {e}")
            self.test_results['errors'].append(f"Dashboard accessibility test failed: {e}")

    def test_02_dashboard_components_presence(self):
        """Test presence of dashboard components"""
        print("\n Testing Dashboard Components Presence...")

        try:
            # Test component files existence
            component_files = [
                "src/dashboard/components/dropdown_manager.py",
                "src/dashboard/components/chart_manager.py",
                "src/dashboard/components/anomaly_heatmap.py",
                "src/dashboard/components/pipeline_status_monitor.py",
                "src/dashboard/components/model_status_panel.py",
                "src/dashboard/components/alerts_pipeline.py"
            ]

            component_results = {}

            for component_file in component_files:
                if os.path.exists(component_file):
                    component_results[os.path.basename(component_file)] = "EXISTS"
                    print(f" {os.path.basename(component_file)}: EXISTS")

                    # Check file size (non-empty)
                    file_size = os.path.getsize(component_file)
                    if file_size > 100:  # At least 100 bytes
                        component_results[f"{os.path.basename(component_file)}_content"] = f"NON_EMPTY ({file_size} bytes)"
                    else:
                        component_results[f"{os.path.basename(component_file)}_content"] = "EMPTY_OR_SMALL"
                else:
                    component_results[os.path.basename(component_file)] = "MISSING"
                    print(f" {os.path.basename(component_file)}: MISSING")

            self.test_results['components']['file_presence'] = component_results

        except Exception as e:
            self.test_results['components']['file_presence'] = f'FAIL - {str(e)}'
            print(f" Component Presence Test: FAIL - {e}")
            self.test_results['errors'].append(f"Component presence test failed: {e}")

    def test_03_data_pipeline_components(self):
        """Test data pipeline components"""
        print("\n Testing Data Pipeline Components...")

        try:
            # Test data ingestion components
            pipeline_files = [
                "src/data_ingestion/nasa_data_service.py",
                "src/data_ingestion/equipment_mapper.py",
                "src/data_ingestion/data_loader.py",
                "src/data_ingestion/unified_data_controller.py"
            ]

            pipeline_results = {}

            for pipeline_file in pipeline_files:
                if os.path.exists(pipeline_file):
                    pipeline_results[os.path.basename(pipeline_file)] = "EXISTS"
                    print(f" {os.path.basename(pipeline_file)}: EXISTS")

                    # Try to import and check for key classes/functions
                    try:
                        module_name = os.path.basename(pipeline_file).replace('.py', '')
                        spec = importlib.util.spec_from_file_location(module_name, pipeline_file)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            pipeline_results[f"{module_name}_import"] = "SUCCESS"
                            print(f" {module_name}: IMPORT_SUCCESS")
                        else:
                            pipeline_results[f"{module_name}_import"] = "SPEC_FAILED"
                    except Exception as import_error:
                        pipeline_results[f"{module_name}_import"] = f"FAILED: {str(import_error)[:30]}"
                        print(f" {module_name}: IMPORT_FAILED")

                else:
                    pipeline_results[os.path.basename(pipeline_file)] = "MISSING"
                    print(f" {os.path.basename(pipeline_file)}: MISSING")

            self.test_results['data_pipeline']['components'] = pipeline_results

        except Exception as e:
            self.test_results['data_pipeline']['components'] = f'FAIL - {str(e)}'
            print(f" Data Pipeline Test: FAIL - {e}")
            self.test_results['errors'].append(f"Data pipeline test failed: {e}")

    def test_04_anomaly_detection_components(self):
        """Test anomaly detection components"""
        print("\n Testing Anomaly Detection Components...")

        try:
            # Test anomaly detection files
            anomaly_files = [
                "src/anomaly_detection/nasa_anomaly_engine.py",
                "src/anomaly_detection/lstm_autoencoder.py",
                "src/anomaly_detection/lstm_vae.py",
                "src/anomaly_detection/telemanom_integration.py"
            ]

            anomaly_results = {}

            for anomaly_file in anomaly_files:
                if os.path.exists(anomaly_file):
                    anomaly_results[os.path.basename(anomaly_file)] = "EXISTS"
                    print(f" {os.path.basename(anomaly_file)}: EXISTS")

                    # Check file content for key terms
                    try:
                        with open(anomaly_file, 'r', encoding='utf-8') as f:
                            content = f.read().lower()

                        key_terms = ['anomaly', 'detection', 'model', 'predict', 'lstm']
                        found_terms = [term for term in key_terms if term in content]

                        anomaly_results[f"{os.path.basename(anomaly_file)}_content"] = f"Terms: {len(found_terms)}/{len(key_terms)}"
                        print(f" {os.path.basename(anomaly_file)}: Contains {len(found_terms)}/{len(key_terms)} key terms")

                    except Exception as content_error:
                        anomaly_results[f"{os.path.basename(anomaly_file)}_content"] = "READ_FAILED"

                else:
                    anomaly_results[os.path.basename(anomaly_file)] = "MISSING"
                    print(f" {os.path.basename(anomaly_file)}: MISSING")

            self.test_results['data_pipeline']['anomaly_detection'] = anomaly_results

        except Exception as e:
            self.test_results['data_pipeline']['anomaly_detection'] = f'FAIL - {str(e)}'
            print(f" Anomaly Detection Test: FAIL - {e}")
            self.test_results['errors'].append(f"Anomaly detection test failed: {e}")

    def test_05_configuration_system(self):
        """Test configuration system"""
        print("\n Testing Configuration System...")

        try:
            config_results = {}

            # Test config directory and files
            config_files = [
                "config/config.yaml",
                "config/settings.py"
            ]

            for config_file in config_files:
                if os.path.exists(config_file):
                    config_results[os.path.basename(config_file)] = "EXISTS"
                    print(f" {os.path.basename(config_file)}: EXISTS")

                    # For YAML files, try to parse
                    if config_file.endswith('.yaml'):
                        try:
                            import yaml
                            with open(config_file, 'r') as f:
                                yaml_content = yaml.safe_load(f)
                            config_results[f"{os.path.basename(config_file)}_valid"] = "VALID_YAML"
                            print(f" {os.path.basename(config_file)}: VALID_YAML")
                        except Exception:
                            config_results[f"{os.path.basename(config_file)}_valid"] = "INVALID_YAML"
                            print(f" {os.path.basename(config_file)}: INVALID_YAML")

                else:
                    config_results[os.path.basename(config_file)] = "MISSING"
                    print(f" {os.path.basename(config_file)}: MISSING")

            self.test_results['infrastructure']['configuration'] = config_results

        except Exception as e:
            self.test_results['infrastructure']['configuration'] = f'FAIL - {str(e)}'
            print(f" Configuration Test: FAIL - {e}")
            self.test_results['errors'].append(f"Configuration test failed: {e}")

    def test_06_data_availability(self):
        """Test data availability"""
        print("\n Testing Data Availability...")

        try:
            data_results = {}

            # Test data directories
            data_dirs = [
                "data",
                "data/models",
                "data/processed",
                "data/raw"
            ]

            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    data_results[data_dir] = "EXISTS"
                    print(f" {data_dir}: EXISTS")

                    # Count files in directory
                    try:
                        file_count = len([f for f in os.listdir(data_dir)
                                        if os.path.isfile(os.path.join(data_dir, f))])
                        data_results[f"{data_dir}_files"] = f"{file_count} files"
                        print(f" {data_dir}: {file_count} files")
                    except Exception:
                        data_results[f"{data_dir}_files"] = "COUNT_FAILED"

                else:
                    data_results[data_dir] = "MISSING"
                    print(f" {data_dir}: MISSING")

            self.test_results['data_pipeline']['data_availability'] = data_results

        except Exception as e:
            self.test_results['data_pipeline']['data_availability'] = f'FAIL - {str(e)}'
            print(f" Data Availability Test: FAIL - {e}")
            self.test_results['errors'].append(f"Data availability test failed: {e}")

    def test_07_performance_baseline(self):
        """Test performance baseline"""
        print("\n Testing Performance Baseline...")

        try:
            performance_results = {}

            # Test dashboard response time
            start_time = time.time()
            try:
                response = requests.get(self.dashboard_url, timeout=30)
                response_time = time.time() - start_time
                performance_results['dashboard_response_time'] = f"{response_time:.2f}s"

                if response_time < 10:
                    print(f" Dashboard response time: {response_time:.2f}s (Good)")
                else:
                    print(f" Dashboard response time: {response_time:.2f}s (Slow)")

            except Exception as e:
                performance_results['dashboard_response_time'] = f"FAILED: {str(e)}"
                print(f" Dashboard response time: FAILED")

            # Test file system performance
            start_time = time.time()
            test_data = "x" * 1000  # 1KB test data
            with open("temp_performance_test.txt", "w") as f:
                f.write(test_data)
            os.remove("temp_performance_test.txt")
            file_io_time = time.time() - start_time
            performance_results['file_io_time'] = f"{file_io_time:.4f}s"
            print(f" File I/O time: {file_io_time:.4f}s")

            self.test_results['performance']['baseline'] = performance_results

        except Exception as e:
            self.test_results['performance']['baseline'] = f'FAIL - {str(e)}'
            print(f" Performance Baseline Test: FAIL - {e}")
            self.test_results['errors'].append(f"Performance baseline test failed: {e}")

    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print(" PHASE 1 API-BASED TESTING REPORT")
        print("=" * 70)

        # Count tests
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        # Infrastructure tests
        print("\n INFRASTRUCTURE TESTS:")
        for category, results in cls.test_results['infrastructure'].items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if result == 'PASS' or result == 'EXISTS' or 'SUCCESS' in str(result):
                        passed_tests += 1
                        print(f"   {category}.{test_name}: {result}")
                    else:
                        failed_tests += 1
                        print(f"   {category}.{test_name}: {result}")
            else:
                total_tests += 1
                if results == 'PASS' or 'SUCCESS' in str(results):
                    passed_tests += 1
                    print(f"   {category}: {results}")
                else:
                    failed_tests += 1
                    print(f"   {category}: {results}")

        # Component tests
        print("\n COMPONENT TESTS:")
        for category, results in cls.test_results['components'].items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if result == 'PASS' or result == 'EXISTS' or 'SUCCESS' in str(result):
                        passed_tests += 1
                        print(f"   {category}.{test_name}: {result}")
                    else:
                        failed_tests += 1
                        print(f"   {category}.{test_name}: {result}")

        # Data pipeline tests
        print("\n DATA PIPELINE TESTS:")
        for category, results in cls.test_results['data_pipeline'].items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    total_tests += 1
                    if result == 'PASS' or result == 'EXISTS' or 'SUCCESS' in str(result):
                        passed_tests += 1
                        print(f"   {category}.{test_name}: {result}")
                    else:
                        failed_tests += 1
                        print(f"   {category}.{test_name}: {result}")

        # Performance tests
        print("\n PERFORMANCE TESTS:")
        for category, results in cls.test_results['performance'].items():
            if isinstance(results, dict):
                for test_name, result in results.items():
                    print(f"   {category}.{test_name}: {result}")

        # Errors
        print("\n ERRORS ENCOUNTERED:")
        if cls.test_results['errors']:
            for error in cls.test_results['errors']:
                print(f"   {error}")
        else:
            print("   No critical errors encountered")

        # Summary
        print("\n SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        success_rate = (passed_tests/total_tests*100) if total_tests > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")

        # Grade the overall system
        if success_rate >= 90:
            grade = " EXCELLENT"
        elif success_rate >= 75:
            grade = " GOOD"
        elif success_rate >= 60:
            grade = " FAIR"
        else:
            grade = " NEEDS_IMPROVEMENT"

        print(f"  Overall Grade: {grade}")

        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 1 - Core Dashboard Infrastructure (API-Based)',
            'test_results': cls.test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'grade': grade
            }
        }

        with open('phase1_api_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n Detailed report saved to: phase1_api_test_report.json")
        print("=" * 70)

if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2, exit=False)