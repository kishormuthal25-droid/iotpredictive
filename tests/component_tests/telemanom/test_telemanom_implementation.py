"""
Test NASA Telemanom Implementation
Test the Telemanom implementation with synthetic data to verify it works
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from typing import Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.anomaly_detection.nasa_telemanom import NASATelemanom, Telemanom_Config, quick_train_telemanom

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_sensor_data(n_samples: int = 2000, n_features: int = 1,
                                  anomaly_rate: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sensor data with anomalies

    Args:
        n_samples: Number of time steps
        n_features: Number of sensor features
        anomaly_rate: Proportion of anomalous samples

    Returns:
        Tuple of (data, labels) where labels indicate anomalies
    """
    # Generate normal data with some patterns
    t = np.linspace(0, 100, n_samples)
    data = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Create realistic sensor patterns
        base_signal = np.sin(0.1 * t + i) + 0.3 * np.sin(0.5 * t + i * 2)
        noise = np.random.normal(0, 0.1, n_samples)
        data[:, i] = base_signal + noise

    # Add anomalies
    labels = np.zeros(n_samples, dtype=bool)
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Create anomalous behavior
        data[idx:idx+5] += np.random.normal(2, 0.5, (min(5, n_samples-idx), n_features))
        labels[idx:idx+5] = True

    return data, labels


def test_single_sensor():
    """Test Telemanom with a single sensor"""
    logger.info("Testing single sensor Telemanom model")

    # Generate test data
    data, true_labels = generate_synthetic_sensor_data(1500, 1, 0.05)

    # Split data
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_labels = true_labels[split_idx:]

    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")

    # Quick train model
    try:
        model = quick_train_telemanom("TEST_SENSOR_01", train_data, epochs=3)
        logger.info("Model training completed successfully")

        # Test anomaly detection
        results = model.predict_anomalies(test_data)
        logger.info(f"Anomaly detection completed")
        logger.info(f"Detected {np.sum(results['anomalies'])} anomalies out of {len(test_data)} samples")
        logger.info(f"Threshold: {results['threshold']:.4f}")
        logger.info(f"Max score: {np.max(results['scores']):.4f}")

        # Calculate basic metrics
        predicted = results['anomalies']
        if len(test_labels) == len(predicted):
            tp = np.sum(predicted & test_labels)
            fp = np.sum(predicted & ~test_labels)
            fn = np.sum(~predicted & test_labels)
            tn = np.sum(~predicted & ~test_labels)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            logger.info(f"Metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # Test model save/load
        model_path = "test_model.pkl"
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Load and test
        loaded_model = NASATelemanom.load_model(model_path)
        loaded_results = loaded_model.predict_anomalies(test_data[:100])
        logger.info(f"Loaded model test: {np.sum(loaded_results['anomalies'])} anomalies detected")

        # Clean up
        Path(model_path).unlink(missing_ok=True)
        model_h5 = Path(model_path).parent / f"{Path(model_path).stem}_model.h5"
        model_h5.unlink(missing_ok=True)

        return True

    except Exception as e:
        logger.error(f"Single sensor test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_multi_sensor():
    """Test Telemanom with multi-sensor data"""
    logger.info("Testing multi-sensor Telemanom model")

    # Generate multi-sensor data (like spacecraft subsystem)
    data, true_labels = generate_synthetic_sensor_data(1200, 3, 0.03)

    # Split data
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    logger.info(f"Multi-sensor data shape: {train_data.shape}")

    try:
        # Create model with custom config
        config = Telemanom_Config(
            epochs=3,
            sequence_length=30,
            lstm_units=[32, 16],
            batch_size=16
        )

        model = NASATelemanom("TEST_MULTI_SENSOR", config)
        history = model.fit(train_data)

        logger.info("Multi-sensor model training completed")

        # Test detection
        results = model.predict_anomalies(test_data)
        logger.info(f"Multi-sensor detection: {np.sum(results['anomalies'])} anomalies")

        return True

    except Exception as e:
        logger.error(f"Multi-sensor test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_training_pipeline():
    """Test the training pipeline components"""
    logger.info("Testing training pipeline components")

    try:
        # Test equipment mapper
        from src.data_ingestion.equipment_mapper import IoTEquipmentMapper
        mapper = IoTEquipmentMapper()

        logger.info(f"SMAP equipment: {len(mapper.smap_equipment)} components")
        logger.info(f"MSL equipment: {len(mapper.msl_equipment)} components")

        # Count total sensors
        total_sensors = 0
        for equipment in mapper.smap_equipment + mapper.msl_equipment:
            total_sensors += len(equipment.sensors)

        logger.info(f"Total sensors mapped: {total_sensors}")

        # Test pipeline class import
        from scripts.train_telemanom_models import TelemanoMTrainingPipeline
        pipeline = TelemanoMTrainingPipeline("test_models")

        # Test sensor mapping
        sensor_mapping = pipeline.get_sensor_mapping()
        logger.info(f"Pipeline sensor mapping: {len(sensor_mapping)} sensors")

        # Show sample sensors
        sample_sensors = list(sensor_mapping.items())[:5]
        for sensor_key, sensor_info in sample_sensors:
            logger.info(f"Sample sensor: {sensor_key} - {sensor_info['sensor_name']}")

        return True

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all tests"""
    logger.info("Starting NASA Telemanom implementation tests")

    tests = [
        ("Single Sensor Test", test_single_sensor),
        ("Multi-Sensor Test", test_multi_sensor),
        ("Training Pipeline Test", test_training_pipeline)
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = test_func()
            results[test_name] = result
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: FAILED with exception: {e}")

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! NASA Telemanom implementation is working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total-passed} tests failed. Check the logs for details.")
        return False


if __name__ == "__main__":
    main()