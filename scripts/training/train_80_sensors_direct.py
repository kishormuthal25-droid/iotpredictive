#!/usr/bin/env python3
"""
Direct NASA Data Training for 80 Sensors
Trains all 80 sensors using real NASA SMAP/MSL data efficiently
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.anomaly_detection.nasa_telemanom import NASATelemanom, Telemanom_Config
from src.data_ingestion.equipment_mapper import IoTEquipmentMapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nasa_raw_data():
    """Load NASA raw datasets directly"""
    try:
        # Load SMAP data
        smap_train = np.load('data/raw/smap/train.npy')
        smap_test = np.load('data/raw/smap/test.npy')

        # Load MSL data
        msl_train = np.load('data/raw/msl/train.npy')
        msl_test = np.load('data/raw/msl/test.npy')

        logger.info(f"Loaded NASA data - SMAP: {smap_train.shape}, MSL: {msl_train.shape}")

        return {
            'smap_train': smap_train,
            'smap_test': smap_test,
            'msl_train': msl_train,
            'msl_test': msl_test
        }
    except Exception as e:
        logger.error(f"Failed to load NASA data: {e}")
        return None

def train_sensor_model(sensor_id: str, data: np.ndarray, models_dir: Path, quick_mode=True):
    """Train a single sensor model with NASA Telemanom"""
    try:
        # Ensure we have enough data
        if len(data) < 100:
            logger.warning(f"Insufficient data for {sensor_id}: {len(data)} samples")
            return None

        logger.info(f"Training {sensor_id} with {len(data)} samples")

        # Create model configuration for quick training
        config = Telemanom_Config(
            batch_size=64,
            epochs=5 if quick_mode else 15,
            lstm_units=[50, 50],
            dropout_rate=0.2,
            prediction_length=10,
            sequence_length=100,
            smoothing_window=30
        )

        # Initialize and train model
        model = NASATelemanom(sensor_id=sensor_id, config=config)
        model.fit(data)

        # Save model
        model_path = models_dir / f"{sensor_id}.pkl"
        model.save_model(str(model_path))

        logger.info(f"Successfully trained {sensor_id} - threshold: {model.error_threshold:.4f}")

        return {
            'sensor_id': sensor_id,
            'model_path': str(model_path),
            'threshold': model.error_threshold,
            'data_samples': len(data),
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Failed to train {sensor_id}: {e}")
        return {
            'sensor_id': sensor_id,
            'status': 'failed',
            'error': str(e)
        }

def train_all_80_sensors():
    """Train all 80 NASA sensors with real data"""
    logger.info("Starting comprehensive 80-sensor training with real NASA data")

    # Create models directory
    models_dir = Path("data/models/telemanom")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load NASA data
    nasa_data = load_nasa_raw_data()
    if not nasa_data:
        logger.error("Could not load NASA data")
        return

    # Get equipment mapper for sensor definitions
    mapper = IoTEquipmentMapper()

    # Training results
    results = []
    successful = 0
    failed = 0

    # Train SMAP sensors (0-24)
    logger.info("Training SMAP sensors (0-24)")
    smap_data = nasa_data['smap_train']

    for i in range(25):
        if i >= smap_data.shape[1]:
            logger.warning(f"SMAP sensor {i} not available in data")
            continue

        sensor_id = f"SMAP_{i:02d}"
        sensor_data = smap_data[:, i]

        result = train_sensor_model(sensor_id, sensor_data, models_dir, quick_mode=True)
        results.append(result)

        if result and result['status'] == 'success':
            successful += 1
        else:
            failed += 1

        logger.info(f"Progress: {i+1}/25 SMAP sensors")

    # Train MSL sensors (25-79)
    logger.info("Training MSL sensors (25-79)")
    msl_data = nasa_data['msl_train']

    for i in range(55):
        if i >= msl_data.shape[1]:
            logger.warning(f"MSL sensor {i} not available in data")
            continue

        sensor_id = f"MSL_{i+25:02d}"
        sensor_data = msl_data[:, i]

        result = train_sensor_model(sensor_id, sensor_data, models_dir, quick_mode=True)
        results.append(result)

        if result and result['status'] == 'success':
            successful += 1
        else:
            failed += 1

        logger.info(f"Progress: {i+1}/55 MSL sensors, Total: {25+i+1}/80")

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_sensors': 80,
        'successful': successful,
        'failed': failed,
        'success_rate': successful / 80 * 100,
        'results': results
    }

    summary_path = models_dir / f"comprehensive_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Training completed: {successful}/{80} successful ({successful/80*100:.1f}%)")
    logger.info(f"Summary saved to: {summary_path}")

    return summary

if __name__ == "__main__":
    print("=== NASA 80-Sensor Training Pipeline ===")
    print("Training all NASA SMAP/MSL sensors with real data")
    print("=" * 50)

    summary = train_all_80_sensors()

    print("\n=== Training Summary ===")
    print(f"Total sensors: {summary['total_sensors']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print("=" * 50)