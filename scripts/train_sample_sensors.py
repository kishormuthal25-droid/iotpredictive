"""
Train Sample Sensors Script
Train NASA Telemanom models for a few sample sensors to validate integration
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.anomaly_detection.nasa_telemanom import NASATelemanom, Telemanom_Config
from src.data_ingestion.equipment_mapper import IoTEquipmentMapper
from src.data_ingestion.unified_data_controller import UnifiedDataController

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_realistic_sensor_data(sensor_info: dict, n_samples: int = 2000) -> np.ndarray:
    """Generate realistic sensor data based on sensor specifications

    Args:
        sensor_info: Sensor specification from equipment mapper
        n_samples: Number of samples to generate

    Returns:
        Generated sensor data
    """
    # Get sensor characteristics
    min_val = sensor_info.get('min_value', 0)
    max_val = sensor_info.get('max_value', 100)
    nominal = sensor_info.get('nominal_value', (min_val + max_val) / 2)

    # Generate time series
    t = np.linspace(0, n_samples / 100, n_samples)  # 100 samples per time unit

    # Create realistic patterns based on sensor type
    if 'temperature' in sensor_info.get('sensor_name', '').lower():
        # Temperature patterns - daily cycles with noise
        data = nominal + (max_val - min_val) * 0.1 * np.sin(0.1 * t) + np.random.normal(0, (max_val - min_val) * 0.02, n_samples)
    elif 'voltage' in sensor_info.get('sensor_name', '').lower() or 'current' in sensor_info.get('sensor_name', '').lower():
        # Power patterns - stable with occasional drifts
        data = nominal + (max_val - min_val) * 0.05 * np.sin(0.05 * t) + np.random.normal(0, (max_val - min_val) * 0.01, n_samples)
    elif 'pressure' in sensor_info.get('sensor_name', '').lower():
        # Pressure patterns - gradual changes with noise
        data = nominal + (max_val - min_val) * 0.08 * np.sin(0.03 * t) + np.random.normal(0, (max_val - min_val) * 0.03, n_samples)
    else:
        # Generic sensor pattern
        data = nominal + (max_val - min_val) * 0.1 * np.sin(0.07 * t) + np.random.normal(0, (max_val - min_val) * 0.02, n_samples)

    # Ensure data stays within bounds
    data = np.clip(data, min_val, max_val)

    return data.reshape(-1, 1)


def train_sample_sensors():
    """Train models for first 5 sensors"""
    logger.info("Training sample sensors with NASA Telemanom")

    # Initialize equipment mapper
    mapper = IoTEquipmentMapper()

    # Create models directory
    models_dir = Path("data/models/telemanom")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Get first 5 sensors
    sample_sensors = []
    sensor_id = 0

    # Get SMAP sensors
    for equipment in mapper.smap_equipment:
        for sensor in equipment.sensors:
            if len(sample_sensors) >= 5:
                break

            sensor_key = f"SMAP_{sensor_id:02d}_{sensor.name.replace(' ', '_')}"
            sensor_info = {
                'sensor_key': sensor_key,
                'equipment_id': equipment.equipment_id,
                'sensor_name': sensor.name,
                'sensor_unit': sensor.unit,
                'min_value': sensor.min_value,
                'max_value': sensor.max_value,
                'nominal_value': sensor.nominal_value,
                'criticality': equipment.criticality,
                'subsystem': equipment.subsystem
            }
            sample_sensors.append(sensor_info)
            sensor_id += 1

        if len(sample_sensors) >= 5:
            break

    logger.info(f"Training {len(sample_sensors)} sample sensors")

    # Train each sensor
    results = {}

    for i, sensor_info in enumerate(sample_sensors):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training sensor {i+1}/5: {sensor_info['sensor_name']}")
        logger.info(f"{'='*50}")

        try:
            # Generate training data
            training_data = generate_realistic_sensor_data(sensor_info, 1500)
            logger.info(f"Generated {len(training_data)} training samples")

            # Configure model based on criticality
            if sensor_info['criticality'] == 'CRITICAL':
                config = Telemanom_Config(
                    epochs=10,
                    sequence_length=80,
                    lstm_units=[64, 32],
                    batch_size=32
                )
            else:
                config = Telemanom_Config(
                    epochs=5,
                    sequence_length=50,
                    lstm_units=[32, 16],
                    batch_size=16
                )

            # Create and train model
            model = NASATelemanom(sensor_info['sensor_key'], config)

            # Split data
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]

            # Train
            history = model.fit(train_data, val_data)

            # Save model
            model_path = models_dir / f"{sensor_info['sensor_key']}.pkl"
            model.save_model(str(model_path))

            # Test anomaly detection
            test_data = generate_realistic_sensor_data(sensor_info, 500)
            anomaly_results = model.predict_anomalies(test_data)

            # Store results
            results[sensor_info['sensor_key']] = {
                'success': True,
                'sensor_info': sensor_info,
                'model_path': str(model_path),
                'threshold': model.error_threshold,
                'training_loss': history['loss'][-1],
                'val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
                'test_anomalies': int(np.sum(anomaly_results['anomalies'])),
                'max_score': float(np.max(anomaly_results['scores']))
            }

            logger.info(f"✅ Successfully trained {sensor_info['sensor_name']}")
            logger.info(f"   Threshold: {model.error_threshold:.4f}")
            logger.info(f"   Training Loss: {history['loss'][-1]:.4f}")
            logger.info(f"   Test Anomalies: {results[sensor_info['sensor_key']]['test_anomalies']}")

        except Exception as e:
            logger.error(f"❌ Failed to train {sensor_info['sensor_name']}: {e}")
            results[sensor_info['sensor_key']] = {
                'success': False,
                'sensor_info': sensor_info,
                'error': str(e)
            }

    # Save training summary
    summary = {
        'trained_sensors': len([r for r in results.values() if r['success']]),
        'failed_sensors': len([r for r in results.values() if not r['success']]),
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    summary_path = models_dir / f"sample_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Successful: {summary['trained_sensors']}/5")
    logger.info(f"Failed: {summary['failed_sensors']}/5")
    logger.info(f"Summary saved to: {summary_path}")

    return summary


def test_trained_models():
    """Test the trained models"""
    logger.info("Testing trained models")

    models_dir = Path("data/models/telemanom")
    model_files = list(models_dir.glob("SMAP_*.pkl"))

    if not model_files:
        logger.warning("No trained models found")
        return

    logger.info(f"Found {len(model_files)} trained models")

    for model_file in model_files:
        try:
            # Load model
            model = NASATelemanom.load_model(str(model_file))

            # Generate test data
            test_data = np.random.randn(500, model.n_features)

            # Run detection
            results = model.predict_anomalies(test_data)

            logger.info(f"✅ {model.sensor_id}: {np.sum(results['anomalies'])} anomalies detected")

        except Exception as e:
            logger.error(f"❌ Failed to test {model_file.name}: {e}")


if __name__ == "__main__":
    # Train sample sensors
    summary = train_sample_sensors()

    # Test trained models
    if summary['trained_sensors'] > 0:
        logger.info("\nTesting trained models...")
        test_trained_models()

    logger.info("\nSample training completed!")