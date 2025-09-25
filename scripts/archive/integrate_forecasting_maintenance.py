#!/usr/bin/env python3
"""
Phase 4: Forecasting & Maintenance Integration
Integrates Transformer forecasting with trained NASA Telemanom models for predictive maintenance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import json
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.forecasting.transformer_forecaster import TransformerForecaster
from src.maintenance.work_order_manager import WorkOrderManager, WorkOrderPriority
from src.anomaly_detection.nasa_telemanom import NASATelemanom
from src.data_ingestion.nasa_data_service import NASADataService

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NASA_PredictiveMaintenanceEngine:
    """
    Comprehensive predictive maintenance engine integrating:
    - NASA Telemanom anomaly detection
    - Transformer-based forecasting
    - Automated work order generation
    """

    def __init__(self):
        self.nasa_data_service = None
        self.forecaster = None
        self.work_order_manager = None
        self.telemanom_models = {}
        self.sensor_data = {}
        self.forecasts = {}

    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing NASA Predictive Maintenance Engine")

        # Initialize NASA data service
        logger.info("Initializing NASA Data Service...")
        self.nasa_data_service = NASADataService()

        # Initialize transformer forecaster
        logger.info("Initializing Transformer Forecaster...")
        self.forecaster = TransformerForecaster(
            sequence_length=100,
            forecast_horizon=24,  # 24-step ahead forecasting
            d_model=128,
            num_heads=8,
            num_encoder_layers=4,
            dff=512,
            dropout_rate=0.1
        )

        # Initialize work order manager
        logger.info("Initializing Work Order Manager...")
        self.work_order_manager = WorkOrderManager()

        logger.info("NASA Predictive Maintenance Engine initialized successfully")

    def load_trained_telemanom_models(self):
        """Load all trained NASA Telemanom models"""
        logger.info("Loading trained NASA Telemanom models...")
        models_dir = Path("data/models/telemanom")

        loaded_count = 0
        for model_file in models_dir.glob("*.pkl"):
            if "_model" in model_file.name:
                continue  # Skip old format

            sensor_id = model_file.stem
            if sensor_id.startswith(("SMAP_", "MSL_")):
                try:
                    # Create Telemanom instance and load model
                    telemanom = NASATelemanom(sensor_id=sensor_id)
                    telemanom.load_model(str(model_file))
                    self.telemanom_models[sensor_id] = telemanom
                    loaded_count += 1

                    if loaded_count % 10 == 0:
                        logger.info(f"Loaded {loaded_count} Telemanom models...")

                except Exception as e:
                    logger.warning(f"Failed to load model {sensor_id}: {e}")

        logger.info(f"Successfully loaded {loaded_count} NASA Telemanom models")
        return loaded_count

    def generate_nasa_equipment_forecasts(self, equipment_types=['POWER', 'MOBILITY', 'COMMUNICATION', 'SCIENCE']):
        """Generate forecasts for critical NASA equipment"""
        logger.info("Generating NASA equipment forecasts...")

        # Get current sensor data
        current_data = self.nasa_data_service.get_real_time_data()

        equipment_forecasts = {}

        for equipment_type in equipment_types:
            logger.info(f"Forecasting for {equipment_type} systems...")

            # Get sensors for this equipment type
            equipment_sensors = self._get_equipment_sensors(equipment_type)

            for sensor_id in equipment_sensors:
                if sensor_id in self.telemanom_models and sensor_id in current_data:
                    try:
                        # Get historical data for forecasting
                        sensor_data = current_data[sensor_id]

                        # Train forecaster on recent data
                        self.forecaster.fit(sensor_data, epochs=10, verbose=0)

                        # Generate forecast
                        forecast = self.forecaster.predict(sensor_data)

                        # Calculate confidence intervals
                        forecast_mean = forecast.mean(axis=0)
                        forecast_std = forecast.std(axis=0)

                        equipment_forecasts[sensor_id] = {
                            'forecast_mean': forecast_mean.tolist(),
                            'forecast_std': forecast_std.tolist(),
                            'confidence_lower': (forecast_mean - 1.96 * forecast_std).tolist(),
                            'confidence_upper': (forecast_mean + 1.96 * forecast_std).tolist(),
                            'equipment_type': equipment_type,
                            'timestamp': datetime.now().isoformat()
                        }

                    except Exception as e:
                        logger.warning(f"Failed to forecast {sensor_id}: {e}")

        self.forecasts = equipment_forecasts
        logger.info(f"Generated forecasts for {len(equipment_forecasts)} sensors")
        return equipment_forecasts

    def _get_equipment_sensors(self, equipment_type):
        """Get sensor IDs for specific equipment type"""
        equipment_mapping = {
            'POWER': [f'SMAP_{i:02d}' for i in range(0, 5)] + [f'MSL_{i:02d}' for i in range(25, 33)],
            'MOBILITY': [f'MSL_{i:02d}' for i in range(33, 51)],
            'COMMUNICATION': [f'SMAP_{i:02d}' for i in range(5, 10)] + [f'MSL_{i:02d}' for i in range(73, 79)],
            'SCIENCE': [f'MSL_{i:02d}' for i in range(63, 73)],
            'THERMAL': [f'SMAP_{i:02d}' for i in range(15, 20)],
            'ATTITUDE': [f'SMAP_{i:02d}' for i in range(10, 15)],
            'PAYLOAD': [f'SMAP_{i:02d}' for i in range(20, 25)],
            'ENVIRONMENTAL': [f'MSL_{i:02d}' for i in range(51, 63)]
        }
        return equipment_mapping.get(equipment_type, [])

    def detect_anomalies_and_predict_failures(self):
        """Run comprehensive anomaly detection and failure prediction"""
        logger.info("Running comprehensive anomaly detection and failure prediction...")

        current_data = self.nasa_data_service.get_real_time_data()
        anomalies_detected = {}
        failure_predictions = {}

        for sensor_id, model in self.telemanom_models.items():
            if sensor_id in current_data:
                try:
                    # Detect current anomalies
                    sensor_data = current_data[sensor_id]
                    anomaly_score = model.detect_anomalies(sensor_data)
                    is_anomaly = anomaly_score > model.error_threshold

                    if is_anomaly:
                        anomalies_detected[sensor_id] = {
                            'anomaly_score': float(anomaly_score),
                            'threshold': float(model.error_threshold),
                            'severity': self._calculate_severity(anomaly_score, model.error_threshold),
                            'timestamp': datetime.now().isoformat()
                        }

                    # Predict future failures using forecasts
                    if sensor_id in self.forecasts:
                        forecast_data = self.forecasts[sensor_id]
                        predicted_failures = self._predict_failures_from_forecast(
                            forecast_data, model.error_threshold
                        )

                        if predicted_failures:
                            failure_predictions[sensor_id] = predicted_failures

                except Exception as e:
                    logger.warning(f"Failed to process {sensor_id}: {e}")

        logger.info(f"Detected {len(anomalies_detected)} current anomalies")
        logger.info(f"Predicted failures for {len(failure_predictions)} sensors")

        return anomalies_detected, failure_predictions

    def _calculate_severity(self, anomaly_score, threshold):
        """Calculate anomaly severity level"""
        ratio = anomaly_score / threshold
        if ratio > 3.0:
            return "CRITICAL"
        elif ratio > 2.0:
            return "HIGH"
        elif ratio > 1.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _predict_failures_from_forecast(self, forecast_data, threshold):
        """Predict potential failures from forecast data"""
        forecast_mean = np.array(forecast_data['forecast_mean'])
        forecast_upper = np.array(forecast_data['confidence_upper'])

        # Check if forecast exceeds threshold
        failure_points = np.where(forecast_upper > threshold * 1.2)[0]  # 20% margin

        if len(failure_points) > 0:
            return {
                'failure_likely': True,
                'failure_timeframes': failure_points.tolist(),
                'max_predicted_value': float(forecast_upper.max()),
                'threshold': float(threshold),
                'confidence': 'HIGH' if forecast_upper.max() > threshold * 1.5 else 'MEDIUM'
            }

        return None

    def generate_maintenance_work_orders(self, anomalies, failure_predictions):
        """Generate automated work orders based on anomalies and predictions"""
        logger.info("Generating automated maintenance work orders...")

        work_orders_created = []

        # Process current anomalies
        for sensor_id, anomaly_data in anomalies.items():
            severity = anomaly_data['severity']

            # Determine priority
            if severity == "CRITICAL":
                priority = WorkOrderPriority.CRITICAL
            elif severity == "HIGH":
                priority = WorkOrderPriority.HIGH
            else:
                priority = WorkOrderPriority.MEDIUM

            # Create work order
            work_order = self.work_order_manager.create_work_order(
                title=f"Anomaly Detected - {sensor_id}",
                description=f"Anomaly detected on sensor {sensor_id}. "
                           f"Anomaly score: {anomaly_data['anomaly_score']:.3f}, "
                           f"Threshold: {anomaly_data['threshold']:.3f}, "
                           f"Severity: {severity}",
                priority=priority,
                equipment_id=sensor_id,
                estimated_duration=self._estimate_repair_duration(severity),
                required_skills=self._get_required_skills(sensor_id)
            )

            work_orders_created.append(work_order)

        # Process failure predictions
        for sensor_id, prediction_data in failure_predictions.items():
            work_order = self.work_order_manager.create_work_order(
                title=f"Preventive Maintenance - {sensor_id}",
                description=f"Predictive maintenance required for sensor {sensor_id}. "
                           f"Failure predicted with {prediction_data['confidence']} confidence. "
                           f"Predicted max value: {prediction_data['max_predicted_value']:.3f}",
                priority=WorkOrderPriority.MEDIUM,
                equipment_id=sensor_id,
                estimated_duration=self._estimate_maintenance_duration(sensor_id),
                required_skills=self._get_required_skills(sensor_id),
                due_date=datetime.now() + timedelta(days=7)  # Preventive - not urgent
            )

            work_orders_created.append(work_order)

        logger.info(f"Created {len(work_orders_created)} automated work orders")
        return work_orders_created

    def _estimate_repair_duration(self, severity):
        """Estimate repair duration based on severity"""
        duration_map = {
            "CRITICAL": 4.0,  # 4 hours
            "HIGH": 2.0,      # 2 hours
            "MEDIUM": 1.0,    # 1 hour
            "LOW": 0.5        # 30 minutes
        }
        return duration_map.get(severity, 1.0)

    def _estimate_maintenance_duration(self, sensor_id):
        """Estimate maintenance duration based on sensor type"""
        if sensor_id.startswith("SMAP"):
            return 1.5  # Satellite maintenance - 1.5 hours
        else:
            return 2.0  # Mars rover maintenance - 2 hours

    def _get_required_skills(self, sensor_id):
        """Get required skills for sensor maintenance"""
        if "PWR" in sensor_id or sensor_id.endswith(("00", "01", "02", "03", "04")):
            return ["electrical", "power_systems"]
        elif "COM" in sensor_id or sensor_id.endswith(("05", "06", "07", "08", "09")):
            return ["communications", "electronics"]
        elif "MOB" in sensor_id:
            return ["mechanical", "robotics"]
        elif "SCI" in sensor_id:
            return ["scientific_instruments", "calibration"]
        else:
            return ["general_maintenance"]

    def run_comprehensive_analysis(self):
        """Run complete predictive maintenance analysis"""
        logger.info("=== Starting Comprehensive NASA Predictive Maintenance Analysis ===")

        # Generate forecasts
        forecasts = self.generate_nasa_equipment_forecasts()

        # Detect anomalies and predict failures
        anomalies, predictions = self.detect_anomalies_and_predict_failures()

        # Generate work orders
        work_orders = self.generate_maintenance_work_orders(anomalies, predictions)

        # Create summary report
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_sensors_analyzed': len(self.telemanom_models),
            'forecasts_generated': len(forecasts),
            'current_anomalies': len(anomalies),
            'failure_predictions': len(predictions),
            'work_orders_created': len(work_orders),
            'forecasts': forecasts,
            'anomalies': anomalies,
            'predictions': predictions,
            'work_orders': [wo.to_dict() for wo in work_orders]
        }

        # Save summary
        summary_path = Path("data/maintenance_analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Comprehensive analysis complete. Summary saved to {summary_path}")

        return summary

def main():
    """Main function to run Phase 4 integration"""
    print("=== Phase 4: NASA Predictive Forecasting & Maintenance ===")
    print("Integrating trained Telemanom models with forecasting and maintenance")
    print("=" * 70)

    # Initialize engine
    engine = NASA_PredictiveMaintenanceEngine()
    engine.initialize()

    # Load trained models
    models_loaded = engine.load_trained_telemanom_models()
    print(f"Loaded {models_loaded} trained NASA Telemanom models")

    # Run comprehensive analysis
    summary = engine.run_comprehensive_analysis()

    print("\n=== Analysis Summary ===")
    print(f"Total sensors analyzed: {summary['total_sensors_analyzed']}")
    print(f"Forecasts generated: {summary['forecasts_generated']}")
    print(f"Current anomalies detected: {summary['current_anomalies']}")
    print(f"Failure predictions made: {summary['failure_predictions']}")
    print(f"Work orders created: {summary['work_orders_created']}")
    print("=" * 70)
    print("Phase 4 integration complete!")

if __name__ == "__main__":
    main()