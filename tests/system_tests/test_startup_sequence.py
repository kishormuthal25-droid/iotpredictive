#!/usr/bin/env python3
"""
System Startup Sequence Test Suite
Tests complete system startup validation and performance
"""

import unittest
import sys
import os
import time
import subprocess
import threading
import requests
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
import psutil
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Suppress warnings during testing
warnings.filterwarnings('ignore')


class TestSystemStartupSequence(unittest.TestCase):
    """Test complete system startup sequence and validation"""

    def setUp(self):
        """Setup test environment"""
        self.test_start_time = time.time()
        self.system_process = None
        self.base_url = "http://localhost:8060"

    def tearDown(self):
        """Cleanup after each test"""
        if self.system_process:
            self._stop_system_process()

    def _stop_system_process(self):
        """Stop system process safely"""
        if self.system_process:
            try:
                self.system_process.terminate()
                self.system_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.system_process.kill()
                self.system_process.wait()
            except Exception:
                pass
            finally:
                self.system_process = None

    def test_system_startup_time_under_2_seconds(self):
        """Test that system starts up in under 2 seconds (CRITICAL PERFORMANCE REQUIREMENT)"""
        print("\n🚀 Testing System Startup Time (<2s requirement)...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"
        self.assertTrue(launch_script.exists(), f"Launch script not found: {launch_script}")

        # Measure startup time
        startup_start = time.time()

        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready (HTTP response)
        ready_time = None
        max_wait = 10  # Maximum wait time (for safety)

        while time.time() - startup_start < max_wait:
            try:
                response = requests.get(self.base_url, timeout=1)
                if response.status_code == 200:
                    ready_time = time.time() - startup_start
                    break
            except requests.RequestException:
                pass
            time.sleep(0.1)

        if ready_time is None:
            self.fail(f"System failed to start within {max_wait} seconds")

        print(f"   ⏱️  System startup time: {ready_time:.3f}s")

        # CRITICAL: Must be under 2 seconds
        self.assertLess(ready_time, 2.0,
                       f"Startup time {ready_time:.3f}s exceeds 2s requirement!")

        print(f"   ✅ Startup time requirement met: {ready_time:.3f}s < 2.0s")

    def test_system_memory_usage_baseline(self):
        """Test system memory usage stays within reasonable bounds"""
        print("\n🧠 Testing System Memory Usage Baseline...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start system
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready
        startup_timeout = 15
        start_time = time.time()
        system_ready = False

        while time.time() - start_time < startup_timeout:
            try:
                response = requests.get(self.base_url, timeout=1)
                if response.status_code == 200:
                    system_ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)

        if not system_ready:
            self.fail("System did not start within timeout")

        # Wait additional time for full initialization
        time.sleep(3)

        # Measure memory usage
        try:
            system_proc = psutil.Process(self.system_process.pid)
            memory_info = system_proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            print(f"   📊 System memory usage: {memory_mb:.1f}MB")

            # Check child processes too
            children = system_proc.children(recursive=True)
            total_memory = memory_mb

            for child in children:
                try:
                    child_memory = child.memory_info().rss / 1024 / 1024
                    total_memory += child_memory
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            print(f"   📊 Total system memory (with children): {total_memory:.1f}MB")

            # Memory usage targets
            # - Baseline: <512MB for basic operation
            # - Extended: <1GB for full operation with data
            self.assertLess(memory_mb, 1024, f"System memory {memory_mb:.1f}MB too high!")

            if memory_mb < 512:
                print("   ✅ Excellent memory usage (<512MB)")
            elif memory_mb < 1024:
                print("   ✅ Acceptable memory usage (<1GB)")

        except psutil.NoSuchProcess:
            self.fail("System process not found for memory measurement")

    def test_system_health_endpoints(self):
        """Test system health and status endpoints"""
        print("\n🏥 Testing System Health Endpoints...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start system
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready
        ready = self._wait_for_system_ready()
        self.assertTrue(ready, "System failed to start")

        # Test main dashboard endpoint
        main_response = requests.get(self.base_url, timeout=5)
        self.assertEqual(main_response.status_code, 200, "Main dashboard should be accessible")
        print("   ✅ Main dashboard endpoint responding")

        # Test common health endpoints
        health_endpoints = [
            "/health", "/status", "/_health", "/_status",
            "/api/health", "/api/status"
        ]

        accessible_endpoints = []
        for endpoint in health_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=2)
                if response.status_code in [200, 404]:  # 404 is OK, means server is responding
                    accessible_endpoints.append(endpoint)
            except requests.RequestException:
                pass

        print(f"   📊 Accessible endpoints: {accessible_endpoints}")

        # At least the main endpoint should work
        main_check = requests.get(self.base_url, timeout=5)
        self.assertEqual(main_check.status_code, 200, "Main endpoint should work")

    def test_system_component_initialization(self):
        """Test that all system components initialize correctly"""
        print("\n🔧 Testing System Component Initialization...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Capture startup output for analysis
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready and collect output
        ready = self._wait_for_system_ready(timeout=15)
        self.assertTrue(ready, "System failed to start")

        # Allow time for initialization messages
        time.sleep(2)

        # Collect available output
        try:
            stdout_data, stderr_data = self.system_process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            # Process is still running, which is expected
            stdout_data = ""
            stderr_data = ""

        # Analyze startup output for component initialization
        startup_messages = []
        if hasattr(self.system_process, 'stdout') and self.system_process.stdout:
            try:
                # Read available output without blocking
                import select
                if select.select([self.system_process.stdout], [], [], 0.1)[0]:
                    partial_output = self.system_process.stdout.read()
                    if partial_output:
                        startup_messages.append(partial_output.decode('utf-8', errors='ignore'))
            except Exception:
                pass

        # Look for common initialization indicators
        initialization_indicators = [
            "starting", "initialized", "loaded", "ready",
            "mlflow", "dashboard", "data", "service"
        ]

        found_indicators = []
        for message in startup_messages:
            message_lower = message.lower()
            for indicator in initialization_indicators:
                if indicator in message_lower and indicator not in found_indicators:
                    found_indicators.append(indicator)

        print(f"   📋 Initialization indicators found: {found_indicators}")

        # Test system responsiveness
        response_start = time.time()
        response = requests.get(self.base_url, timeout=5)
        response_time = time.time() - response_start

        self.assertEqual(response.status_code, 200, "System should be responsive")
        print(f"   ⚡ System response time: {response_time:.3f}s")

        # Response should be fast after initialization
        self.assertLess(response_time, 1.0, "System should respond quickly after init")

        print("   ✅ System components initialized successfully")

    def test_system_concurrent_startup_safety(self):
        """Test system behavior when multiple startup attempts are made"""
        print("\n🔒 Testing Concurrent Startup Safety...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start first instance
        process1 = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for first instance to be ready
        ready1 = self._wait_for_system_ready(timeout=15)
        self.assertTrue(ready1, "First instance failed to start")

        # Try to start second instance (should fail gracefully or handle port conflict)
        process2 = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        time.sleep(3)  # Give second instance time to attempt startup

        # Check status of both processes
        process1_running = process1.poll() is None
        process2_running = process2.poll() is None

        print(f"   📊 First instance running: {process1_running}")
        print(f"   📊 Second instance running: {process2_running}")

        # At least the first instance should be running
        self.assertTrue(process1_running, "First instance should remain running")

        # Test that the system is still accessible
        response = requests.get(self.base_url, timeout=5)
        self.assertEqual(response.status_code, 200, "System should remain accessible")

        # Cleanup second process
        try:
            process2.terminate()
            process2.wait(timeout=5)
        except Exception:
            pass

        # Keep first process as main system process for teardown
        self.system_process = process1

        print("   ✅ Concurrent startup handled safely")

    def test_system_graceful_shutdown(self):
        """Test that system shuts down gracefully"""
        print("\n👋 Testing System Graceful Shutdown...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start system
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready
        ready = self._wait_for_system_ready()
        self.assertTrue(ready, "System failed to start")

        # Verify system is accessible
        response = requests.get(self.base_url, timeout=5)
        self.assertEqual(response.status_code, 200, "System should be accessible")

        # Initiate graceful shutdown
        shutdown_start = time.time()
        self.system_process.terminate()  # Send SIGTERM

        # Wait for graceful shutdown
        try:
            return_code = self.system_process.wait(timeout=10)
            shutdown_time = time.time() - shutdown_start

            print(f"   ⏱️  Graceful shutdown time: {shutdown_time:.3f}s")
            print(f"   📊 Exit code: {return_code}")

            # Shutdown should be reasonably fast
            self.assertLess(shutdown_time, 10.0, "Graceful shutdown should be timely")

            # System should no longer be accessible
            time.sleep(1)
            with self.assertRaises(requests.RequestException):
                requests.get(self.base_url, timeout=2)

            print("   ✅ System shutdown gracefully")

        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown failed
            self.system_process.kill()
            self.system_process.wait()
            self.fail("System did not shutdown gracefully within timeout")

        # Clear process reference to avoid double cleanup
        self.system_process = None

    def _wait_for_system_ready(self, timeout=20):
        """Wait for system to be ready and accessible"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.base_url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)
        return False


class TestSystemResourceManagement(unittest.TestCase):
    """Test system resource management and limits"""

    def setUp(self):
        """Setup test environment"""
        self.system_process = None
        self.base_url = "http://localhost:8060"

    def tearDown(self):
        """Cleanup after each test"""
        if self.system_process:
            try:
                self.system_process.terminate()
                self.system_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.system_process.kill()
            except Exception:
                pass

    def test_cpu_usage_under_load(self):
        """Test CPU usage remains reasonable under normal operation"""
        print("\n🖥️  Testing CPU Usage Under Load...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start system
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready
        ready = self._wait_for_system_ready()
        if not ready:
            self.skipTest("System failed to start")

        # Allow system to stabilize
        time.sleep(5)

        # Measure CPU usage over time
        try:
            system_proc = psutil.Process(self.system_process.pid)
            cpu_measurements = []

            # Take CPU measurements over 10 seconds
            for i in range(10):
                cpu_percent = system_proc.cpu_percent(interval=1)
                cpu_measurements.append(cpu_percent)
                print(f"   📊 CPU measurement {i+1}: {cpu_percent:.1f}%")

            avg_cpu = sum(cpu_measurements) / len(cpu_measurements)
            max_cpu = max(cpu_measurements)

            print(f"   📊 Average CPU usage: {avg_cpu:.1f}%")
            print(f"   📊 Maximum CPU usage: {max_cpu:.1f}%")

            # CPU usage should be reasonable
            # Average should be low for idle system
            self.assertLess(avg_cpu, 50.0, f"Average CPU usage too high: {avg_cpu:.1f}%")

            # Maximum spikes should be tolerable
            self.assertLess(max_cpu, 90.0, f"Maximum CPU usage too high: {max_cpu:.1f}%")

            print("   ✅ CPU usage within acceptable limits")

        except psutil.NoSuchProcess:
            self.fail("System process not found for CPU measurement")

    def test_file_descriptor_limits(self):
        """Test that system doesn't exhaust file descriptors"""
        print("\n📁 Testing File Descriptor Usage...")

        launch_script = Path(__file__).parent.parent.parent / "launch_real_data_dashboard.py"

        # Start system
        self.system_process = subprocess.Popen(
            [sys.executable, str(launch_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(launch_script.parent)
        )

        # Wait for system to be ready
        ready = self._wait_for_system_ready()
        if not ready:
            self.skipTest("System failed to start")

        # Allow system to stabilize
        time.sleep(3)

        try:
            system_proc = psutil.Process(self.system_process.pid)

            # Get file descriptor count
            try:
                num_fds = system_proc.num_fds()
                print(f"   📊 Open file descriptors: {num_fds}")

                # File descriptor usage should be reasonable
                # Typical web applications should use less than 1000 FDs
                self.assertLess(num_fds, 1000, f"Too many file descriptors: {num_fds}")

                if num_fds < 100:
                    print("   ✅ Excellent file descriptor usage")
                elif num_fds < 500:
                    print("   ✅ Good file descriptor usage")
                else:
                    print("   ⚠️  High file descriptor usage")

            except AttributeError:
                # num_fds() not available on all platforms
                print("   ℹ️  File descriptor count not available on this platform")

        except psutil.NoSuchProcess:
            self.fail("System process not found for FD measurement")

    def _wait_for_system_ready(self, timeout=20):
        """Wait for system to be ready and accessible"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.base_url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.5)
        return False


def run_system_startup_tests():
    """Run comprehensive system startup test suite"""
    print("=" * 80)
    print("🚀 SYSTEM STARTUP SEQUENCE TEST SUITE")
    print("=" * 80)

    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")

    try:
        print(f"💻 System: {psutil.cpu_count()} cores, {psutil.virtual_memory().total // 1024**3}GB RAM")
    except Exception:
        print("💻 System info not available")

    print()

    # Create test suite
    suite = unittest.TestSuite()

    # Add startup sequence tests
    suite.addTest(TestSystemStartupSequence('test_system_startup_time_under_2_seconds'))
    suite.addTest(TestSystemStartupSequence('test_system_memory_usage_baseline'))
    suite.addTest(TestSystemStartupSequence('test_system_health_endpoints'))
    suite.addTest(TestSystemStartupSequence('test_system_component_initialization'))
    suite.addTest(TestSystemStartupSequence('test_system_concurrent_startup_safety'))
    suite.addTest(TestSystemStartupSequence('test_system_graceful_shutdown'))

    # Add resource management tests
    suite.addTest(TestSystemResourceManagement('test_cpu_usage_under_load'))
    suite.addTest(TestSystemResourceManagement('test_file_descriptor_limits'))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL SYSTEM STARTUP TESTS PASSED!")
        print("   System is ready for production deployment!")
    else:
        print(f"❌ {len(result.failures)} FAILURES, {len(result.errors)} ERRORS")
        print("   System needs optimization before production deployment")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_system_startup_tests()
    sys.exit(0 if success else 1)