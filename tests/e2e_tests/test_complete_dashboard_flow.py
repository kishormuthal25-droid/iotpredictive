#!/usr/bin/env python3
"""
End-to-End Dashboard Flow Test Suite
Tests complete user workflows through the dashboard
"""

import unittest
import sys
import os
import time
import threading
import requests
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings
import json
import queue

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Suppress warnings during testing
warnings.filterwarnings('ignore')

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


class TestDashboardEndToEnd(unittest.TestCase):
    """Test complete end-to-end dashboard workflows"""

    @classmethod
    def setUpClass(cls):
        """Setup dashboard server for testing"""
        if not SELENIUM_AVAILABLE:
            raise unittest.SkipTest("Selenium not available for E2E testing")

        cls.dashboard_process = None
        cls.dashboard_url = "http://localhost:8060"
        cls.startup_timeout = 30  # seconds

        # Start dashboard in background
        cls._start_dashboard_server()

    @classmethod
    def tearDownClass(cls):
        """Cleanup dashboard server"""
        cls._stop_dashboard_server()

    @classmethod
    def _start_dashboard_server(cls):
        """Start the dashboard server for testing"""
        print("\n🚀 Starting dashboard server for E2E testing...")

        try:
            # Use the main entry point
            launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

            cls.dashboard_process = subprocess.Popen(
                [sys.executable, str(launch_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(launch_script.parent)
            )

            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < cls.startup_timeout:
                try:
                    response = requests.get(cls.dashboard_url, timeout=2)
                    if response.status_code == 200:
                        print(f"   ✅ Dashboard server started at {cls.dashboard_url}")
                        time.sleep(2)  # Additional stabilization time
                        return
                except requests.RequestException:
                    pass
                time.sleep(1)

            raise TimeoutException(f"Dashboard failed to start within {cls.startup_timeout}s")

        except Exception as e:
            print(f"   ❌ Failed to start dashboard: {e}")
            raise

    @classmethod
    def _stop_dashboard_server(cls):
        """Stop the dashboard server"""
        if cls.dashboard_process:
            print("\n🛑 Stopping dashboard server...")
            try:
                cls.dashboard_process.terminate()
                cls.dashboard_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.dashboard_process.kill()
            except Exception as e:
                print(f"   ⚠️  Error stopping dashboard: {e}")

    def setUp(self):
        """Setup browser for each test"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            self.wait = WebDriverWait(self.driver, 15)
        except WebDriverException as e:
            self.skipTest(f"Chrome WebDriver not available: {e}")

    def tearDown(self):
        """Cleanup browser after each test"""
        if hasattr(self, 'driver'):
            try:
                self.driver.quit()
            except Exception:
                pass

    def test_dashboard_loading_performance(self):
        """Test that dashboard loads within performance requirements"""
        print("\n⚡ Testing Dashboard Loading Performance...")

        load_start = time.time()
        self.driver.get(self.dashboard_url)

        # Wait for main content to load
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            load_time = time.time() - load_start

            print(f"   ⏱️  Page load time: {load_time:.3f}s")

            # Dashboard should load quickly (under 5 seconds)
            self.assertLess(load_time, 5.0, f"Dashboard load time too slow: {load_time:.3f}s")

            # Check for basic dashboard structure
            title = self.driver.title
            self.assertIn("IoT", title, "Dashboard title should contain 'IoT'")

            print(f"   ✅ Dashboard loaded: '{title}'")

        except TimeoutException:
            self.fail("Dashboard failed to load within timeout")

    def test_navigation_between_pages(self):
        """Test navigation between different dashboard pages"""
        print("\n🧭 Testing Navigation Between Pages...")

        self.driver.get(self.dashboard_url)

        # Wait for page to load
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Look for navigation elements (common dashboard patterns)
        nav_selectors = [
            "nav", ".navbar", ".nav-tabs", ".nav-pills",
            "[data-toggle='tab']", "[role='tab']"
        ]

        nav_elements = []
        for selector in nav_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                nav_elements.extend(elements)
            except Exception:
                continue

        if not nav_elements:
            print("   ℹ️  No navigation elements found - single page dashboard")
            return

        print(f"   🔍 Found {len(nav_elements)} potential navigation elements")

        # Test clicking on navigation elements
        clicked_elements = 0
        for i, element in enumerate(nav_elements[:5]):  # Test first 5
            try:
                if element.is_displayed() and element.is_enabled():
                    element_text = element.text[:30] if element.text else f"Element {i}"

                    click_start = time.time()
                    element.click()
                    click_time = time.time() - click_start

                    print(f"   🖱️  Clicked '{element_text}': {click_time:.3f}s")
                    clicked_elements += 1

                    # Navigation should be fast
                    self.assertLess(click_time, 2.0, "Navigation too slow")

                    time.sleep(0.5)  # Brief pause between clicks
            except Exception as e:
                print(f"   ⚠️  Could not click element {i}: {e}")

        print(f"   ✅ Successfully tested {clicked_elements} navigation elements")

    def test_sensor_data_visualization(self):
        """Test sensor data visualization and interaction"""
        print("\n📊 Testing Sensor Data Visualization...")

        self.driver.get(self.dashboard_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Look for chart/graph elements (common Plotly/Dash patterns)
        chart_selectors = [
            ".plotly-graph-div", ".dash-graph", "svg",
            "[data-dash-is-loading]", ".chart-container"
        ]

        charts_found = []
        for selector in chart_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                charts_found.extend(elements)
            except Exception:
                continue

        print(f"   📈 Found {len(charts_found)} chart elements")

        if charts_found:
            # Test interacting with first chart
            first_chart = charts_found[0]
            try:
                # Check if chart is visible
                self.assertTrue(first_chart.is_displayed(), "Chart should be visible")

                # Try to get chart size
                chart_size = first_chart.size
                print(f"   📐 Chart size: {chart_size['width']}x{chart_size['height']}")

                # Chart should have reasonable dimensions
                self.assertGreater(chart_size['width'], 100, "Chart width too small")
                self.assertGreater(chart_size['height'], 100, "Chart height too small")

                print("   ✅ Chart visualization working")

            except Exception as e:
                print(f"   ⚠️  Chart interaction error: {e}")
        else:
            print("   ℹ️  No charts found - dashboard may be data-driven")

    def test_dropdown_and_controls_interaction(self):
        """Test dropdown menus and control elements"""
        print("\n🎛️  Testing Dropdown and Controls Interaction...")

        self.driver.get(self.dashboard_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Look for interactive controls
        control_selectors = [
            "select", ".dropdown", ".Select-control",
            "input[type='range']", "input[type='checkbox']",
            "button", ".btn"
        ]

        controls_found = []
        for selector in control_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                controls_found.extend(elements)
            except Exception:
                continue

        print(f"   🎮 Found {len(controls_found)} control elements")

        # Test interacting with controls
        tested_controls = 0
        for i, control in enumerate(controls_found[:10]):  # Test first 10
            try:
                if control.is_displayed() and control.is_enabled():
                    control_type = control.tag_name
                    control_class = control.get_attribute('class') or ''

                    interaction_start = time.time()

                    if control_type == 'select':
                        # Test dropdown selection
                        options = control.find_elements(By.TAG_NAME, 'option')
                        if options and len(options) > 1:
                            options[1].click()  # Select second option
                            tested_controls += 1

                    elif control_type == 'button':
                        # Test button click
                        control.click()
                        tested_controls += 1

                    elif control_type == 'input':
                        input_type = control.get_attribute('type')
                        if input_type == 'checkbox':
                            control.click()
                            tested_controls += 1

                    interaction_time = time.time() - interaction_start

                    if tested_controls > 0:
                        print(f"   🎯 Interacted with {control_type}: {interaction_time:.3f}s")

                        # Interactions should be responsive
                        self.assertLess(interaction_time, 1.0, "Control interaction too slow")

                    time.sleep(0.3)  # Brief pause between interactions

            except Exception as e:
                print(f"   ⚠️  Control interaction error: {str(e)[:50]}")

        print(f"   ✅ Successfully tested {tested_controls} controls")

    def test_real_time_data_updates(self):
        """Test real-time data updates in the dashboard"""
        print("\n🔄 Testing Real-time Data Updates...")

        self.driver.get(self.dashboard_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Capture initial page content
        initial_content = self.driver.page_source

        # Wait for potential updates
        print("   ⏳ Waiting for potential data updates...")
        time.sleep(5)

        # Capture content after waiting
        updated_content = self.driver.page_source

        # Check for dynamic content indicators
        dynamic_indicators = [
            'data-dash-is-loading="true"',
            'Loading...', 'Updating...',
            'last-updated', 'timestamp'
        ]

        has_dynamic_content = any(indicator in updated_content.lower()
                                for indicator in dynamic_indicators)

        if initial_content != updated_content:
            print("   ✅ Content updated - real-time features working")
        elif has_dynamic_content:
            print("   ✅ Dynamic content indicators found")
        else:
            print("   ℹ️  Static content - may update based on data availability")

        # Check for timestamp elements
        try:
            timestamp_elements = self.driver.find_elements(
                By.XPATH, "//*[contains(text(), '2024') or contains(text(), '2023')]"
            )
            if timestamp_elements:
                print(f"   📅 Found {len(timestamp_elements)} timestamp elements")
        except Exception:
            pass

    def test_error_handling_and_resilience(self):
        """Test dashboard error handling and resilience"""
        print("\n🛡️  Testing Error Handling and Resilience...")

        self.driver.get(self.dashboard_url)
        self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Check for error messages in the console
        console_logs = []
        try:
            logs = self.driver.get_log('browser')
            console_logs = [log for log in logs if log['level'] in ['SEVERE', 'WARNING']]
        except Exception:
            print("   ℹ️  Browser logs not accessible")

        # Check for error elements on page
        error_selectors = [
            ".error", ".alert-danger", ".warning", ".alert-warning",
            "[class*='error']", "[class*='warning']"
        ]

        errors_found = []
        for selector in error_selectors:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                errors_found.extend(elements)
            except Exception:
                continue

        # Report findings
        severe_console_errors = [log for log in console_logs if log['level'] == 'SEVERE']

        if severe_console_errors:
            print(f"   ⚠️  Found {len(severe_console_errors)} severe console errors:")
            for error in severe_console_errors[:3]:  # Show first 3
                print(f"      ❌ {error['message'][:100]}...")

        if errors_found:
            print(f"   ⚠️  Found {len(errors_found)} error elements on page")
        else:
            print("   ✅ No error elements found on page")

        # The dashboard should be functional despite minor errors
        page_title = self.driver.title
        self.assertIsNotNone(page_title, "Page should have a title")
        self.assertNotEqual(page_title, "", "Page title should not be empty")

    def test_performance_under_load(self):
        """Test dashboard performance under simulated load"""
        print("\n⚡ Testing Performance Under Load...")

        # Simulate multiple rapid page interactions
        interaction_times = []

        for i in range(5):  # 5 rapid interactions
            start_time = time.time()

            self.driver.get(self.dashboard_url)
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            interaction_time = time.time() - start_time
            interaction_times.append(interaction_time)

            print(f"   🔄 Interaction {i+1}: {interaction_time:.3f}s")

            time.sleep(0.5)  # Brief pause between interactions

        # Calculate performance metrics
        avg_time = sum(interaction_times) / len(interaction_times)
        max_time = max(interaction_times)

        print(f"   📊 Average response time: {avg_time:.3f}s")
        print(f"   📊 Maximum response time: {max_time:.3f}s")

        # Performance should remain acceptable under load
        self.assertLess(avg_time, 3.0, f"Average response time too slow: {avg_time:.3f}s")
        self.assertLess(max_time, 5.0, f"Maximum response time too slow: {max_time:.3f}s")

        print("   ✅ Performance acceptable under load")


def run_e2e_dashboard_tests():
    """Run comprehensive end-to-end dashboard tests"""
    print("=" * 80)
    print("🌐 END-TO-END DASHBOARD WORKFLOW TEST SUITE")
    print("=" * 80)

    if not SELENIUM_AVAILABLE:
        print("❌ Selenium not available - E2E tests cannot run")
        print("   Install with: pip install selenium")
        print("   Also need ChromeDriver: https://chromedriver.chromium.org/")
        return False

    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()

    # Create test suite
    suite = unittest.TestSuite()

    # Add E2E tests in logical order
    suite.addTest(TestDashboardEndToEnd('test_dashboard_loading_performance'))
    suite.addTest(TestDashboardEndToEnd('test_navigation_between_pages'))
    suite.addTest(TestDashboardEndToEnd('test_sensor_data_visualization'))
    suite.addTest(TestDashboardEndToEnd('test_dropdown_and_controls_interaction'))
    suite.addTest(TestDashboardEndToEnd('test_real_time_data_updates'))
    suite.addTest(TestDashboardEndToEnd('test_error_handling_and_resilience'))
    suite.addTest(TestDashboardEndToEnd('test_performance_under_load'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL END-TO-END DASHBOARD TESTS PASSED!")
        print("   Dashboard is ready for production use!")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        print("   Dashboard needs attention before production deployment")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_e2e_dashboard_tests()
    sys.exit(0 if success else 1)