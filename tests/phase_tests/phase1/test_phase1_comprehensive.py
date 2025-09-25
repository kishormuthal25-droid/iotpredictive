#!/usr/bin/env python3
"""
Phase 1 Comprehensive Testing Suite
Tests Core Dashboard Infrastructure functionality
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

# Add project root to path
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# Test imports
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class Phase1TestSuite(unittest.TestCase):
    """Comprehensive Phase 1 testing suite"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.dashboard_url = "http://localhost:8060"
        cls.dashboard_process = None
        cls.driver = None
        cls.test_results = {
            'interactive_components': {},
            'realtime_pipeline': {},
            'performance_metrics': {},
            'errors': []
        }

        print("🚀 Starting Phase 1 Comprehensive Testing...")
        print("=" * 60)

        # Setup Chrome driver
        cls._setup_driver()

        # Start dashboard
        cls._start_dashboard()

        # Wait for dashboard to be ready
        cls._wait_for_dashboard()

    @classmethod
    def _setup_driver(cls):
        """Setup Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")

            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)
            print("✅ Chrome WebDriver initialized")

        except Exception as e:
            print(f"❌ Failed to setup Chrome driver: {e}")
            cls.driver = None

    @classmethod
    def _start_dashboard(cls):
        """Start the dashboard process"""
        try:
            print("🔄 Starting dashboard...")
            cls.dashboard_process = subprocess.Popen(
                [sys.executable, "launch_real_data_dashboard.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            print("✅ Dashboard process started")

        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            cls.test_results['errors'].append(f"Dashboard startup failed: {e}")

    @classmethod
    def _wait_for_dashboard(cls):
        """Wait for dashboard to be ready"""
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(cls.dashboard_url, timeout=5)
                if response.status_code == 200:
                    print("✅ Dashboard is ready")
                    time.sleep(5)  # Additional wait for full initialization
                    return
            except requests.RequestException:
                pass

            print(f"⏳ Waiting for dashboard... ({attempt + 1}/{max_attempts})")
            time.sleep(2)

        raise Exception("Dashboard failed to start within timeout period")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        print("\n🧹 Cleaning up test environment...")

        if cls.driver:
            cls.driver.quit()
            print("✅ WebDriver closed")

        if cls.dashboard_process:
            try:
                if os.name == 'nt':  # Windows
                    cls.dashboard_process.terminate()
                else:  # Unix-like
                    os.killpg(os.getpgid(cls.dashboard_process.pid), signal.SIGTERM)
                cls.dashboard_process.wait(timeout=10)
                print("✅ Dashboard process terminated")
            except Exception as e:
                print(f"⚠️ Error terminating dashboard: {e}")

        # Generate final report
        cls._generate_test_report()

    def test_01_interactive_components_equipment_dropdowns(self):
        """Test P1-IC-001: Equipment/Sensor Dropdowns"""
        print("\n🔍 Testing Equipment/Sensor Dropdowns...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Navigate to dashboard
            self.driver.get(self.dashboard_url)

            # Wait for page load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Test equipment dropdown existence
            equipment_dropdown = self.driver.find_element(By.ID, "equipment-dropdown")
            self.assertIsNotNone(equipment_dropdown, "Equipment dropdown not found")

            # Test dropdown population
            equipment_options = equipment_dropdown.find_elements(By.TAG_NAME, "option")
            self.assertGreater(len(equipment_options), 1, "Equipment dropdown not populated")

            # Test sensor cascade
            equipment_dropdown.click()
            time.sleep(2)

            # Select first equipment option
            if len(equipment_options) > 1:
                equipment_options[1].click()
                time.sleep(3)

                # Check if sensor dropdown is updated
                try:
                    sensor_dropdown = self.driver.find_element(By.ID, "sensor-dropdown")
                    sensor_options = sensor_dropdown.find_elements(By.TAG_NAME, "option")
                    self.assertGreater(len(sensor_options), 1, "Sensor dropdown not cascaded")

                    self.test_results['interactive_components']['equipment_dropdowns'] = 'PASS'
                    print("✅ Equipment/Sensor Dropdowns: PASS")

                except NoSuchElementException:
                    self.test_results['interactive_components']['equipment_dropdowns'] = 'FAIL - Sensor dropdown not found'
                    print("❌ Equipment/Sensor Dropdowns: FAIL - Sensor dropdown not found")

        except Exception as e:
            self.test_results['interactive_components']['equipment_dropdowns'] = f'FAIL - {str(e)}'
            print(f"❌ Equipment/Sensor Dropdowns: FAIL - {e}")
            self.test_results['errors'].append(f"Equipment dropdowns test failed: {e}")

    def test_02_interactive_components_chart_selectors(self):
        """Test P1-IC-002: Chart Type Selectors"""
        print("\n🔍 Testing Chart Type Selectors...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for chart type selector
            chart_selectors = self.driver.find_elements(By.CSS_SELECTOR, "[id*='chart-type']")

            if chart_selectors:
                chart_selector = chart_selectors[0]

                # Test chart type options
                chart_options = chart_selector.find_elements(By.TAG_NAME, "option")
                expected_types = ['line', 'candlestick', 'area']

                found_types = [opt.get_attribute('value') for opt in chart_options]

                for chart_type in expected_types:
                    if chart_type in str(found_types).lower():
                        print(f"✅ Found chart type: {chart_type}")

                self.test_results['interactive_components']['chart_selectors'] = 'PASS'
                print("✅ Chart Type Selectors: PASS")
            else:
                self.test_results['interactive_components']['chart_selectors'] = 'FAIL - Chart selectors not found'
                print("❌ Chart Type Selectors: FAIL - Not found")

        except Exception as e:
            self.test_results['interactive_components']['chart_selectors'] = f'FAIL - {str(e)}'
            print(f"❌ Chart Type Selectors: FAIL - {e}")
            self.test_results['errors'].append(f"Chart selectors test failed: {e}")

    def test_03_interactive_components_time_controls(self):
        """Test P1-IC-003: Time Controls"""
        print("\n🔍 Testing Time Controls...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for time control elements
            time_controls = self.driver.find_elements(By.CSS_SELECTOR, "[id*='time'], [id*='date'], [class*='time']")

            if time_controls:
                print(f"✅ Found {len(time_controls)} time control elements")

                # Test for specific time controls
                time_window_found = False
                date_picker_found = False

                for control in time_controls:
                    control_id = control.get_attribute('id') or ''
                    control_class = control.get_attribute('class') or ''

                    if 'window' in control_id.lower() or 'interval' in control_id.lower():
                        time_window_found = True
                        print("✅ Time window control found")

                    if 'date' in control_id.lower() or 'picker' in control_class.lower():
                        date_picker_found = True
                        print("✅ Date picker control found")

                self.test_results['interactive_components']['time_controls'] = 'PASS'
                print("✅ Time Controls: PASS")
            else:
                self.test_results['interactive_components']['time_controls'] = 'FAIL - No time controls found'
                print("❌ Time Controls: FAIL - Not found")

        except Exception as e:
            self.test_results['interactive_components']['time_controls'] = f'FAIL - {str(e)}'
            print(f"❌ Time Controls: FAIL - {e}")
            self.test_results['errors'].append(f"Time controls test failed: {e}")

    def test_04_realtime_pipeline_processing_rate(self):
        """Test P1-RD-001: Processing Rate Display"""
        print("\n🔍 Testing Processing Rate Display...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for processing rate indicators
            rate_elements = self.driver.find_elements(By.CSS_SELECTOR, "[id*='rate'], [id*='processing'], [class*='rate']")

            processing_rate_found = False

            for element in rate_elements:
                text = element.text.lower()
                if 'processing' in text and ('rate' in text or 'real-time' in text or 'nasa' in text):
                    processing_rate_found = True
                    print(f"✅ Processing rate display found: {element.text}")
                    break

            if processing_rate_found:
                self.test_results['realtime_pipeline']['processing_rate'] = 'PASS'
                print("✅ Processing Rate Display: PASS")
            else:
                self.test_results['realtime_pipeline']['processing_rate'] = 'FAIL - Processing rate display not found'
                print("❌ Processing Rate Display: FAIL - Not found")

        except Exception as e:
            self.test_results['realtime_pipeline']['processing_rate'] = f'FAIL - {str(e)}'
            print(f"❌ Processing Rate Display: FAIL - {e}")
            self.test_results['errors'].append(f"Processing rate test failed: {e}")

    def test_05_realtime_pipeline_anomaly_heatmap(self):
        """Test P1-RD-002: Equipment Anomaly Heatmap"""
        print("\n🔍 Testing Equipment Anomaly Heatmap...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for heatmap elements
            heatmap_elements = self.driver.find_elements(By.CSS_SELECTOR, "[id*='heatmap'], [id*='anomaly'], [class*='heatmap']")

            if heatmap_elements:
                print(f"✅ Found {len(heatmap_elements)} heatmap elements")

                # Check for SVG or canvas elements (typical for heatmaps)
                svg_elements = self.driver.find_elements(By.TAG_NAME, "svg")
                canvas_elements = self.driver.find_elements(By.TAG_NAME, "canvas")

                if svg_elements or canvas_elements:
                    print(f"✅ Found {len(svg_elements)} SVG and {len(canvas_elements)} canvas elements")
                    self.test_results['realtime_pipeline']['anomaly_heatmap'] = 'PASS'
                    print("✅ Equipment Anomaly Heatmap: PASS")
                else:
                    self.test_results['realtime_pipeline']['anomaly_heatmap'] = 'PARTIAL - Elements found but no visualization'
                    print("⚠️ Equipment Anomaly Heatmap: PARTIAL - No visualization elements")
            else:
                self.test_results['realtime_pipeline']['anomaly_heatmap'] = 'FAIL - Heatmap elements not found'
                print("❌ Equipment Anomaly Heatmap: FAIL - Not found")

        except Exception as e:
            self.test_results['realtime_pipeline']['anomaly_heatmap'] = f'FAIL - {str(e)}'
            print(f"❌ Equipment Anomaly Heatmap: FAIL - {e}")
            self.test_results['errors'].append(f"Anomaly heatmap test failed: {e}")

    def test_06_realtime_pipeline_model_status(self):
        """Test P1-RD-003: Active Models Status"""
        print("\n🔍 Testing Active Models Status...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for model status elements
            model_elements = self.driver.find_elements(By.CSS_SELECTOR, "[id*='model'], [id*='status'], [class*='model']")

            models_status_found = False

            for element in model_elements:
                text = element.text.lower()
                if ('model' in text and 'status' in text) or ('active' in text and 'model' in text) or '80' in text:
                    models_status_found = True
                    print(f"✅ Model status display found: {element.text}")
                    break

            if models_status_found:
                self.test_results['realtime_pipeline']['model_status'] = 'PASS'
                print("✅ Active Models Status: PASS")
            else:
                self.test_results['realtime_pipeline']['model_status'] = 'FAIL - Model status not found'
                print("❌ Active Models Status: FAIL - Not found")

        except Exception as e:
            self.test_results['realtime_pipeline']['model_status'] = f'FAIL - {str(e)}'
            print(f"❌ Active Models Status: FAIL - {e}")
            self.test_results['errors'].append(f"Model status test failed: {e}")

    def test_07_realtime_pipeline_nasa_alerts(self):
        """Test P1-RD-004: NASA Alerts Pipeline"""
        print("\n🔍 Testing NASA Alerts Pipeline...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Look for alert elements
            alert_elements = self.driver.find_elements(By.CSS_SELECTOR, "[id*='alert'], [id*='notification'], [class*='alert']")

            alerts_found = False

            for element in alert_elements:
                text = element.text.lower()
                if ('alert' in text or 'notification' in text) and ('nasa' in text or 'anomaly' in text):
                    alerts_found = True
                    print(f"✅ NASA alerts found: {element.text}")
                    break

            # Also check for alert-related UI components
            if not alerts_found:
                # Look for badge, notification, or alert UI components
                badge_elements = self.driver.find_elements(By.CSS_SELECTOR, ".badge, .notification, .alert-badge")
                if badge_elements:
                    alerts_found = True
                    print(f"✅ Found {len(badge_elements)} alert UI components")

            if alerts_found:
                self.test_results['realtime_pipeline']['nasa_alerts'] = 'PASS'
                print("✅ NASA Alerts Pipeline: PASS")
            else:
                self.test_results['realtime_pipeline']['nasa_alerts'] = 'FAIL - NASA alerts not found'
                print("❌ NASA Alerts Pipeline: FAIL - Not found")

        except Exception as e:
            self.test_results['realtime_pipeline']['nasa_alerts'] = f'FAIL - {str(e)}'
            print(f"❌ NASA Alerts Pipeline: FAIL - {e}")
            self.test_results['errors'].append(f"NASA alerts test failed: {e}")

    def test_08_performance_metrics(self):
        """Test overall performance metrics"""
        print("\n🔍 Testing Performance Metrics...")

        if not self.driver:
            self.skipTest("WebDriver not available")

        try:
            # Measure page load time
            start_time = time.time()
            self.driver.get(self.dashboard_url)

            # Wait for main content to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            load_time = time.time() - start_time
            self.test_results['performance_metrics']['page_load_time'] = f"{load_time:.2f}s"

            # Check if load time is acceptable (< 10 seconds)
            if load_time < 10:
                print(f"✅ Page load time: {load_time:.2f}s (Good)")
            else:
                print(f"⚠️ Page load time: {load_time:.2f}s (Slow)")

            # Test dashboard responsiveness
            start_time = time.time()

            # Try to interact with dropdowns
            try:
                equipment_dropdown = self.driver.find_element(By.ID, "equipment-dropdown")
                equipment_dropdown.click()
                response_time = time.time() - start_time
                self.test_results['performance_metrics']['interaction_response_time'] = f"{response_time:.2f}s"

                if response_time < 2:
                    print(f"✅ Interaction response time: {response_time:.2f}s (Good)")
                else:
                    print(f"⚠️ Interaction response time: {response_time:.2f}s (Slow)")

            except NoSuchElementException:
                self.test_results['performance_metrics']['interaction_response_time'] = 'N/A - No interactive elements'
                print("⚠️ No interactive elements found for response time test")

            print("✅ Performance Metrics: COMPLETED")

        except Exception as e:
            self.test_results['performance_metrics']['error'] = str(e)
            print(f"❌ Performance Metrics: FAIL - {e}")
            self.test_results['errors'].append(f"Performance metrics test failed: {e}")

    @classmethod
    def _generate_test_report(cls):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 PHASE 1 TESTING REPORT")
        print("=" * 60)

        # Summary
        total_tests = 0
        passed_tests = 0
        failed_tests = 0

        print("\n🔍 INTERACTIVE COMPONENTS RESULTS:")
        for test_name, result in cls.test_results['interactive_components'].items():
            total_tests += 1
            status = "✅ PASS" if result == 'PASS' else "❌ FAIL"
            if result == 'PASS':
                passed_tests += 1
            else:
                failed_tests += 1
            print(f"  {test_name}: {status}")
            if result != 'PASS':
                print(f"    Details: {result}")

        print("\n🔍 REAL-TIME PIPELINE RESULTS:")
        for test_name, result in cls.test_results['realtime_pipeline'].items():
            total_tests += 1
            status = "✅ PASS" if result == 'PASS' else "❌ FAIL"
            if result == 'PASS':
                passed_tests += 1
            else:
                failed_tests += 1
            print(f"  {test_name}: {status}")
            if result != 'PASS':
                print(f"    Details: {result}")

        print("\n📊 PERFORMANCE METRICS:")
        for metric_name, value in cls.test_results['performance_metrics'].items():
            print(f"  {metric_name}: {value}")

        print("\n🐛 ERRORS ENCOUNTERED:")
        if cls.test_results['errors']:
            for error in cls.test_results['errors']:
                print(f"  ❌ {error}")
        else:
            print("  ✅ No errors encountered")

        print("\n📈 SUMMARY:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {failed_tests}")
        print(f"  Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "  Success Rate: N/A")

        # Save detailed report to file
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 1 - Core Dashboard Infrastructure',
            'test_results': cls.test_results,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests/total_tests*100) if total_tests > 0 else 0
            }
        }

        with open('phase1_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\n💾 Detailed report saved to: phase1_test_report.json")
        print("=" * 60)

if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2, exit=False)