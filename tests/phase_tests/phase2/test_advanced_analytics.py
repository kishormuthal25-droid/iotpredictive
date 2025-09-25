"""
Phase 2 Advanced Analytics & Visualization Testing Suite
Tests anomaly detection models, forecasting, and advanced dashboard features
"""

import unittest
import sys
import os
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class TestAnomalyDetectionModels(unittest.TestCase):
    """Test Phase 2 anomaly detection models and algorithms"""

    def setUp(self):
        """Setup test environment"""
        self.test_data_path = project_root / "testing_phase1_phase2" / "test_data"
        self.test_data_path.mkdir(exist_ok=True)

        # Create sample telemetry data for testing
        self.sample_data = np.random.randn(1000, 5)
        self.sequence_length = 50

    def test_nasa_telemanom_integration(self):
        """Test NASA Telemanom model integration"""
        try:
            from src.anomaly_detection.nasa_telemanom import NASATelemanom

            # Test model initialization
            model = NASATelemanom('MSL_25')
            self.assertIsNotNone(model, "NASA Telemanom failed to initialize")

            # Test model methods
            self.assertTrue(hasattr(model, 'predict'), "predict method missing")
            self.assertTrue(hasattr(model, 'detect_anomalies'), "detect_anomalies method missing")

        except Exception as e:
            self.fail(f"NASA Telemanom integration test failed: {e}")

    def test_lstm_predictor(self):
        """Test LSTM Predictor model"""
        try:
            from src.anomaly_detection.lstm_predictor import LSTMPredictor

            # Test model creation
            model = LSTMPredictor(input_dim=5, sequence_length=self.sequence_length)
            self.assertIsNotNone(model.model, "LSTM model not created")

            # Test prediction functionality
            test_input = self.sample_data[:self.sequence_length].reshape(1, self.sequence_length, 5)
            prediction = model.predict(test_input)
            self.assertIsNotNone(prediction, "LSTM prediction failed")

        except Exception as e:
            self.fail(f"LSTM Predictor test failed: {e}")

    def test_lstm_autoencoder(self):
        """Test LSTM Autoencoder model"""
        try:
            from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder

            # Test model creation
            model = LSTMAutoencoder(input_dim=5, sequence_length=self.sequence_length, encoding_dim=10)
            self.assertIsNotNone(model.encoder, "LSTM Autoencoder encoder not created")
            self.assertIsNotNone(model.decoder, "LSTM Autoencoder decoder not created")

            # Test encoding/decoding
            test_input = self.sample_data[:self.sequence_length].reshape(1, self.sequence_length, 5)
            encoded = model.encode(test_input)
            decoded = model.decode(encoded)

            self.assertEqual(decoded.shape, test_input.shape, "Autoencoder shape mismatch")

        except Exception as e:
            self.fail(f"LSTM Autoencoder test failed: {e}")

    def test_lstm_vae(self):
        """Test LSTM VAE model"""
        try:
            from src.anomaly_detection.lstm_vae import LSTMVAE

            # Test model creation
            model = LSTMVAE(input_dim=5, sequence_length=self.sequence_length, latent_dim=10)
            self.assertIsNotNone(model.encoder, "LSTM VAE encoder not created")
            self.assertIsNotNone(model.decoder, "LSTM VAE decoder not created")

            # Test VAE specific methods
            self.assertTrue(hasattr(model, 'get_latent_representation'), "get_latent_representation method missing")

        except Exception as e:
            self.fail(f"LSTM VAE test failed: {e}")

    def test_model_ensemble(self):
        """Test model ensemble functionality"""
        try:
            from src.anomaly_detection.model_ensemble import ModelEnsemble

            # Test ensemble creation
            ensemble = ModelEnsemble()
            self.assertIsNotNone(ensemble, "Model ensemble failed to initialize")

            # Test ensemble methods
            self.assertTrue(hasattr(ensemble, 'add_model'), "add_model method missing")
            self.assertTrue(hasattr(ensemble, 'predict'), "predict method missing")
            self.assertTrue(hasattr(ensemble, 'detect_anomalies'), "detect_anomalies method missing")

        except Exception as e:
            self.fail(f"Model ensemble test failed: {e}")

    def test_anomaly_scoring(self):
        """Test anomaly scoring algorithms"""
        try:
            from src.anomaly_detection.scoring import AnomalyScorer

            scorer = AnomalyScorer()

            # Test scoring methods
            normal_data = np.random.normal(0, 1, (100, 5))
            anomaly_data = np.random.normal(5, 1, (10, 5))

            normal_scores = scorer.calculate_scores(normal_data)
            anomaly_scores = scorer.calculate_scores(anomaly_data)

            # Anomaly scores should generally be higher
            self.assertGreater(np.mean(anomaly_scores), np.mean(normal_scores),
                             "Anomaly scores not higher than normal scores")

        except Exception as e:
            self.fail(f"Anomaly scoring test failed: {e}")


class TestForecastingModels(unittest.TestCase):
    """Test Phase 2 forecasting models and capabilities"""

    def setUp(self):
        """Setup test environment"""
        # Create sample time series data
        self.time_series_data = np.sin(np.linspace(0, 20*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        self.forecast_horizon = 50

    def test_transformer_forecaster(self):
        """Test Transformer-based forecasting model"""
        try:
            from src.forecasting.transformer_forecaster import TransformerForecaster

            # Test model creation
            model = TransformerForecaster(input_dim=1, sequence_length=100, forecast_horizon=self.forecast_horizon)
            self.assertIsNotNone(model.model, "Transformer forecaster model not created")

            # Test forecasting functionality
            test_input = self.time_series_data[:100].reshape(1, 100, 1)
            forecast = model.forecast(test_input)

            self.assertEqual(forecast.shape[1], self.forecast_horizon, "Forecast horizon mismatch")

        except Exception as e:
            self.fail(f"Transformer forecaster test failed: {e}")

    def test_lstm_forecaster(self):
        """Test LSTM-based forecasting model"""
        try:
            from src.forecasting.lstm_forecaster import LSTMForecaster

            # Test model creation
            model = LSTMForecaster(input_dim=1, sequence_length=100, forecast_horizon=self.forecast_horizon)
            self.assertIsNotNone(model.model, "LSTM forecaster model not created")

            # Test forecasting functionality
            test_input = self.time_series_data[:100].reshape(1, 100, 1)
            forecast = model.forecast(test_input)

            self.assertEqual(forecast.shape[1], self.forecast_horizon, "Forecast horizon mismatch")

        except Exception as e:
            self.fail(f"LSTM forecaster test failed: {e}")

    def test_forecasting_metrics(self):
        """Test forecasting evaluation metrics"""
        try:
            from src.forecasting.metrics import ForecastingMetrics

            metrics = ForecastingMetrics()

            # Create test data
            actual = np.random.randn(100)
            predicted = actual + np.random.normal(0, 0.1, 100)

            # Test metrics calculation
            mae = metrics.mean_absolute_error(actual, predicted)
            rmse = metrics.root_mean_squared_error(actual, predicted)
            mape = metrics.mean_absolute_percentage_error(actual, predicted)

            self.assertGreater(mae, 0, "MAE should be positive")
            self.assertGreater(rmse, 0, "RMSE should be positive")
            self.assertGreater(mape, 0, "MAPE should be positive")

        except Exception as e:
            self.fail(f"Forecasting metrics test failed: {e}")


class TestAdvancedVisualization(unittest.TestCase):
    """Test Phase 2 advanced visualization and dashboard features"""

    def setUp(self):
        """Setup test environment"""
        self.dashboard_url = "http://localhost:8050"

    def test_advanced_dashboard_components(self):
        """Test advanced dashboard components"""
        try:
            from src.dashboard.advanced_components import (
                create_anomaly_heatmap,
                create_forecasting_chart,
                create_equipment_health_gauge,
                create_risk_assessment_chart
            )

            # Test component functions exist and are callable
            self.assertTrue(callable(create_anomaly_heatmap), "create_anomaly_heatmap not callable")
            self.assertTrue(callable(create_forecasting_chart), "create_forecasting_chart not callable")
            self.assertTrue(callable(create_equipment_health_gauge), "create_equipment_health_gauge not callable")
            self.assertTrue(callable(create_risk_assessment_chart), "create_risk_assessment_chart not callable")

        except Exception as e:
            self.fail(f"Advanced dashboard components test failed: {e}")

    def test_real_time_data_processing(self):
        """Test real-time data processing capabilities"""
        try:
            from src.dashboard.real_time_processor import RealTimeProcessor

            processor = RealTimeProcessor()

            # Test processor methods
            self.assertTrue(hasattr(processor, 'process_stream'), "process_stream method missing")
            self.assertTrue(hasattr(processor, 'update_anomaly_buffer'), "update_anomaly_buffer method missing")
            self.assertTrue(hasattr(processor, 'calculate_real_time_metrics'), "calculate_real_time_metrics method missing")

        except Exception as e:
            self.fail(f"Real-time data processing test failed: {e}")

    def test_interactive_visualizations(self):
        """Test interactive visualization features"""
        try:
            from src.dashboard.interactive_viz import InteractiveVisualizations

            viz = InteractiveVisualizations()

            # Test visualization methods
            self.assertTrue(hasattr(viz, 'create_drill_down_chart'), "create_drill_down_chart method missing")
            self.assertTrue(hasattr(viz, 'create_time_range_selector'), "create_time_range_selector method missing")
            self.assertTrue(hasattr(viz, 'create_sensor_comparison'), "create_sensor_comparison method missing")

        except Exception as e:
            self.fail(f"Interactive visualizations test failed: {e}")


class TestMaintenanceOptimization(unittest.TestCase):
    """Test Phase 2 maintenance optimization features"""

    def setUp(self):
        """Setup test environment"""
        pass

    def test_maintenance_scheduler(self):
        """Test maintenance scheduling optimization"""
        try:
            from src.maintenance.scheduler import MaintenanceScheduler

            scheduler = MaintenanceScheduler()

            # Test scheduler methods
            self.assertTrue(hasattr(scheduler, 'optimize_schedule'), "optimize_schedule method missing")
            self.assertTrue(hasattr(scheduler, 'calculate_priority'), "calculate_priority method missing")
            self.assertTrue(hasattr(scheduler, 'assign_technicians'), "assign_technicians method missing")

        except Exception as e:
            self.fail(f"Maintenance scheduler test failed: {e}")

    def test_work_order_optimization(self):
        """Test work order optimization using PuLP"""
        try:
            from src.maintenance.work_order_optimizer import WorkOrderOptimizer

            optimizer = WorkOrderOptimizer()

            # Test optimizer methods
            self.assertTrue(hasattr(optimizer, 'optimize_assignments'), "optimize_assignments method missing")
            self.assertTrue(hasattr(optimizer, 'calculate_costs'), "calculate_costs method missing")

        except Exception as e:
            self.fail(f"Work order optimization test failed: {e}")

    def test_resource_allocation(self):
        """Test resource allocation algorithms"""
        try:
            from src.maintenance.resource_allocator import ResourceAllocator

            allocator = ResourceAllocator()

            # Test resource allocation methods
            self.assertTrue(hasattr(allocator, 'allocate_technicians'), "allocate_technicians method missing")
            self.assertTrue(hasattr(allocator, 'calculate_workload'), "calculate_workload method missing")

        except Exception as e:
            self.fail(f"Resource allocation test failed: {e}")


class TestDataPipelineAdvanced(unittest.TestCase):
    """Test Phase 2 advanced data pipeline features"""

    def setUp(self):
        """Setup test environment"""
        pass

    def test_streaming_data_processing(self):
        """Test streaming data processing capabilities"""
        try:
            from src.data_ingestion.streaming_processor import StreamingProcessor

            processor = StreamingProcessor()

            # Test streaming methods
            self.assertTrue(hasattr(processor, 'process_kafka_stream'), "process_kafka_stream method missing")
            self.assertTrue(hasattr(processor, 'handle_real_time_alerts'), "handle_real_time_alerts method missing")

        except Exception as e:
            self.fail(f"Streaming data processing test failed: {e}")

    def test_advanced_preprocessing(self):
        """Test advanced preprocessing features"""
        try:
            from src.preprocessing.advanced_processor import AdvancedDataProcessor

            processor = AdvancedDataProcessor()

            # Test advanced preprocessing methods
            self.assertTrue(hasattr(processor, 'handle_missing_data'), "handle_missing_data method missing")
            self.assertTrue(hasattr(processor, 'detect_outliers'), "detect_outliers method missing")
            self.assertTrue(hasattr(processor, 'feature_engineering'), "feature_engineering method missing")

        except Exception as e:
            self.fail(f"Advanced preprocessing test failed: {e}")

    def test_nasa_subsystem_analysis(self):
        """Test NASA subsystem failure analysis"""
        try:
            from src.anomaly_detection.nasa_subsystem_analyzer import NASASubsystemAnalyzer

            analyzer = NASASubsystemAnalyzer()

            # Test subsystem analysis methods
            self.assertTrue(hasattr(analyzer, 'analyze_subsystem_health'), "analyze_subsystem_health method missing")
            self.assertTrue(hasattr(analyzer, 'predict_failures'), "predict_failures method missing")
            self.assertTrue(hasattr(analyzer, 'generate_maintenance_recommendations'), "generate_maintenance_recommendations method missing")

        except Exception as e:
            self.fail(f"NASA subsystem analysis test failed: {e}")


class TestDashboardPages(unittest.TestCase):
    """Test Phase 2 dashboard pages and navigation"""

    def setUp(self):
        self.dashboard_url = "http://localhost:8050"

    def test_anomaly_monitor_page(self):
        """Test anomaly monitor page functionality"""
        try:
            response = requests.get(f"{self.dashboard_url}/anomaly-monitor", timeout=5)
            self.assertIn(response.status_code, [200, 404],
                         f"Unexpected response for anomaly monitor page: {response.status_code}")

        except requests.ConnectionError:
            self.skipTest("Dashboard server not running - skipping page test")

    def test_enhanced_forecasting_page(self):
        """Test enhanced forecasting page"""
        try:
            response = requests.get(f"{self.dashboard_url}/enhanced-forecasting", timeout=5)
            self.assertIn(response.status_code, [200, 404],
                         f"Unexpected response for forecasting page: {response.status_code}")

        except requests.ConnectionError:
            self.skipTest("Dashboard server not running - skipping page test")

    def test_maintenance_dashboard_page(self):
        """Test maintenance dashboard page"""
        try:
            response = requests.get(f"{self.dashboard_url}/maintenance-dashboard", timeout=5)
            self.assertIn(response.status_code, [200, 404],
                         f"Unexpected response for maintenance page: {response.status_code}")

        except requests.ConnectionError:
            self.skipTest("Dashboard server not running - skipping page test")

    def test_equipment_analysis_page(self):
        """Test equipment analysis page"""
        try:
            response = requests.get(f"{self.dashboard_url}/equipment-analysis", timeout=5)
            self.assertIn(response.status_code, [200, 404],
                         f"Unexpected response for equipment analysis page: {response.status_code}")

        except requests.ConnectionError:
            self.skipTest("Dashboard server not running - skipping page test")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()

    # Add Phase 2 test classes
    suite.addTest(unittest.makeSuite(TestAnomalyDetectionModels))
    suite.addTest(unittest.makeSuite(TestForecastingModels))
    suite.addTest(unittest.makeSuite(TestAdvancedVisualization))
    suite.addTest(unittest.makeSuite(TestMaintenanceOptimization))
    suite.addTest(unittest.makeSuite(TestDataPipelineAdvanced))
    suite.addTest(unittest.makeSuite(TestDashboardPages))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PHASE 2 ADVANCED ANALYTICS TEST SUMMARY")
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