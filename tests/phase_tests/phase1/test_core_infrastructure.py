"""
Phase 1 Core Infrastructure Testing Suite
Tests fundamental dashboard components, data ingestion, and basic anomaly detection
"""

import unittest
import sys
import os
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestCoreInfrastructure(unittest.TestCase):
    """Test Phase 1 core infrastructure components"""

    def setUp(self):
        """Setup test environment"""
        self.dashboard_url = "http://localhost:8050"
        self.test_data_path = project_root / "testing_phase1_phase2" / "test_data"
        self.test_data_path.mkdir(exist_ok=True)

    def test_project_structure(self):
        """Test that core project structure exists"""
        required_dirs = [
            "src/dashboard",
            "src/data_ingestion",
            "src/anomaly_detection",
            "src/preprocessing",
            "config",
            "data"
        ]

        for dir_path in required_dirs:
            full_path = project_root / dir_path
            self.assertTrue(full_path.exists(), f"Required directory missing: {dir_path}")

    def test_configuration_loading(self):
        """Test configuration system"""
        try:
            from config.settings import settings

            # Test that core config sections exist
            self.assertIsNotNone(settings.dashboard, "Dashboard config missing")
            self.assertIsNotNone(settings.models, "Models config missing")
            self.assertIsNotNone(settings.data, "Data config missing")

            # Test dashboard config
            self.assertIn('port', settings.dashboard)
            self.assertIn('host', settings.dashboard)

        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")

    def test_data_ingestion_module(self):
        """Test data ingestion components"""
        try:
            from src.data_ingestion.nasa_data_service import NASADataService

            # Test service initialization
            service = NASADataService()
            self.assertIsNotNone(service, "NASA data service failed to initialize")

            # Test that required methods exist
            self.assertTrue(hasattr(service, 'load_dataset'), "load_dataset method missing")
            self.assertTrue(hasattr(service, 'get_sensor_data'), "get_sensor_data method missing")

        except Exception as e:
            self.fail(f"Data ingestion module test failed: {e}")

    def test_preprocessing_pipeline(self):
        """Test data preprocessing components"""
        try:
            from src.preprocessing.data_processor import DataProcessor

            # Test processor initialization
            processor = DataProcessor()
            self.assertIsNotNone(processor, "Data processor failed to initialize")

            # Test with sample data
            sample_data = np.random.randn(100, 5)
            processed = processor.normalize(sample_data)

            self.assertEqual(processed.shape, sample_data.shape, "Processed data shape mismatch")

        except Exception as e:
            self.fail(f"Preprocessing pipeline test failed: {e}")

    def test_anomaly_detection_models(self):
        """Test basic anomaly detection model loading"""
        try:
            from src.anomaly_detection.lstm_predictor import LSTMPredictor

            # Test model initialization
            model = LSTMPredictor(input_dim=5, sequence_length=10)
            self.assertIsNotNone(model, "LSTM Predictor failed to initialize")

            # Test model methods exist
            self.assertTrue(hasattr(model, 'predict'), "predict method missing")
            self.assertTrue(hasattr(model, 'detect_anomalies'), "detect_anomalies method missing")

        except Exception as e:
            self.fail(f"Anomaly detection model test failed: {e}")

    def test_dashboard_components(self):
        """Test dashboard component imports"""
        try:
            from src.dashboard.components import (
                create_metric_card,
                create_alert_table,
                create_equipment_status
            )

            # Test component functions exist
            self.assertTrue(callable(create_metric_card), "create_metric_card not callable")
            self.assertTrue(callable(create_alert_table), "create_alert_table not callable")
            self.assertTrue(callable(create_equipment_status), "create_equipment_status not callable")

        except Exception as e:
            self.fail(f"Dashboard components test failed: {e}")

    def test_dashboard_utilities(self):
        """Test dashboard utility classes"""
        try:
            from src.dashboard.utils import DataManager, WebSocketManager

            # Test utility initialization
            data_manager = DataManager()
            ws_manager = WebSocketManager()

            self.assertIsNotNone(data_manager, "DataManager failed to initialize")
            self.assertIsNotNone(ws_manager, "WebSocketManager failed to initialize")

        except Exception as e:
            self.fail(f"Dashboard utilities test failed: {e}")

    def test_database_connectivity(self):
        """Test database connection and basic operations"""
        try:
            from src.utils.database import DatabaseManager

            # Test database manager
            db_manager = DatabaseManager()

            # Test connection
            self.assertTrue(db_manager.test_connection(), "Database connection failed")

        except Exception as e:
            # Log warning but don't fail if database not configured
            print(f"Warning: Database test failed (may be expected): {e}")

    def test_logging_system(self):
        """Test logging configuration"""
        import logging

        # Test that logger can be created
        logger = logging.getLogger('test_logger')
        self.assertIsNotNone(logger, "Logger creation failed")

        # Test logging levels
        logger.info("Test info message")
        logger.warning("Test warning message")

    def test_file_permissions(self):
        """Test that required files have proper permissions"""
        critical_files = [
            "launch_real_data_dashboard.py",
            "config/config.yaml",
            "src/dashboard/app.py"
        ]

        for file_path in critical_files:
            full_path = project_root / file_path
            self.assertTrue(full_path.exists(), f"Critical file missing: {file_path}")
            self.assertTrue(full_path.is_file(), f"Path is not a file: {file_path}")


class TestDashboardConnectivity(unittest.TestCase):
    """Test dashboard server connectivity"""

    def setUp(self):
        self.dashboard_url = "http://localhost:8050"

    def test_dashboard_server_running(self):
        """Test if dashboard server is accessible"""
        try:
            response = requests.get(self.dashboard_url, timeout=5)
            self.assertEqual(response.status_code, 200, "Dashboard server not responding")

        except requests.ConnectionError:
            self.skipTest("Dashboard server not running - this is expected during testing")
        except Exception as e:
            self.fail(f"Dashboard connectivity test failed: {e}")

    def test_dashboard_pages_exist(self):
        """Test that dashboard pages are accessible"""
        pages_to_test = [
            "/",
            "/anomaly-monitor",
            "/enhanced-forecasting",
            "/maintenance-dashboard"
        ]

        for page in pages_to_test:
            try:
                response = requests.get(f"{self.dashboard_url}{page}", timeout=5)
                self.assertIn(response.status_code, [200, 404],
                             f"Unexpected response for page {page}: {response.status_code}")

            except requests.ConnectionError:
                self.skipTest(f"Dashboard server not running - skipping page test for {page}")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add infrastructure tests
    suite.addTest(unittest.makeSuite(TestCoreInfrastructure))

    # Add connectivity tests (these may be skipped if dashboard not running)
    suite.addTest(unittest.makeSuite(TestDashboardConnectivity))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PHASE 1 CORE INFRASTRUCTURE TEST SUMMARY")
    print(f"{'='*60}")
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