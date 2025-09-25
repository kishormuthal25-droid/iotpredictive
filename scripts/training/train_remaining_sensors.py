#!/usr/bin/env python3
"""
Smart Incremental NASA Sensor Training
Only trains sensors that don't already have trained models
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_existing_models(models_dir: Path):
    """Check which sensors already have trained models"""
    existing_models = set()

    for model_file in models_dir.glob("*.pkl"):
        if "_model" in model_file.name:
            continue  # Skip old format models

        sensor_id = model_file.stem
        if sensor_id.startswith(("SMAP_", "MSL_")) and len(sensor_id) >= 6:
            existing_models.add(sensor_id)

    logger.info(f"Found {len(existing_models)} existing models: {sorted(existing_models)}")
    return existing_models

def get_missing_sensors(existing_models):
    """Determine which sensors still need training"""
    # Expected sensors: SMAP_00 to SMAP_24, MSL_25 to MSL_79
    expected_smap = {f"SMAP_{i:02d}" for i in range(25)}
    expected_msl = {f"MSL_{i:02d}" for i in range(25, 80)}
    expected_all = expected_smap | expected_msl

    missing = expected_all - existing_models

    smap_missing = [s for s in missing if s.startswith("SMAP")]
    msl_missing = [s for s in missing if s.startswith("MSL")]

    logger.info(f"Missing SMAP sensors: {len(smap_missing)} - {sorted(smap_missing)}")
    logger.info(f"Missing MSL sensors: {len(msl_missing)} - {sorted(msl_missing)}")

    return sorted(smap_missing), sorted(msl_missing)

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

def train_missing_sensors():
    """Train only the sensors that don't already have models"""
    logger.info("Starting incremental training for missing sensors only")

    # Create models directory
    models_dir = Path("data/models/telemanom")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check existing models
    existing_models = check_existing_models(models_dir)
    smap_missing, msl_missing = get_missing_sensors(existing_models)

    total_missing = len(smap_missing) + len(msl_missing)
    if total_missing == 0:
        logger.info("All sensors already trained! No work needed.")
        return

    logger.info(f"Need to train {total_missing} missing sensors")

    # Load NASA data
    nasa_data = load_nasa_raw_data()
    if not nasa_data:
        logger.error("Could not load NASA data")
        return

    # Training results
    results = []
    successful = 0
    failed = 0

    # Train missing SMAP sensors
    if smap_missing:
        logger.info(f"Training {len(smap_missing)} missing SMAP sensors")
        smap_data = nasa_data['smap_train']

        for sensor_id in smap_missing:
            # Extract sensor index (e.g., "SMAP_20" -> 20)
            sensor_idx = int(sensor_id.split('_')[1])

            if sensor_idx >= smap_data.shape[1]:
                logger.warning(f"SMAP sensor {sensor_idx} not available in data")
                continue

            sensor_data = smap_data[:, sensor_idx]

            result = train_sensor_model(sensor_id, sensor_data, models_dir, quick_mode=True)
            results.append(result)

            if result and result['status'] == 'success':
                successful += 1
            else:
                failed += 1

            logger.info(f"SMAP Progress: {successful + failed}/{len(smap_missing)} completed")

    # Train missing MSL sensors
    if msl_missing:
        logger.info(f"Training {len(msl_missing)} missing MSL sensors")
        msl_data = nasa_data['msl_train']

        for sensor_id in msl_missing:
            # Extract sensor index (e.g., "MSL_25" -> 25, map to MSL column 0)
            sensor_idx = int(sensor_id.split('_')[1]) - 25  # MSL_25 maps to column 0

            if sensor_idx >= msl_data.shape[1] or sensor_idx < 0:
                logger.warning(f"MSL sensor {sensor_id} (column {sensor_idx}) not available in data")
                continue

            sensor_data = msl_data[:, sensor_idx]

            result = train_sensor_model(sensor_id, sensor_data, models_dir, quick_mode=True)
            results.append(result)

            if result and result['status'] == 'success':
                successful += 1
            else:
                failed += 1

            logger.info(f"MSL Progress: {successful + failed - len(smap_missing)}/{len(msl_missing)} completed")

    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_missing': total_missing,
        'smap_missing': len(smap_missing),
        'msl_missing': len(msl_missing),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / total_missing * 100 if total_missing > 0 else 100,
        'results': results
    }

    summary_path = models_dir / f"incremental_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Incremental training completed: {successful}/{total_missing} successful ({successful/total_missing*100:.1f}%)")
    logger.info(f"Summary saved to: {summary_path}")

    return summary

if __name__ == "__main__":
    print("=== NASA Incremental Sensor Training ===")
    print("Training only missing sensors to complete 80-sensor coverage")
    print("=" * 50)

    summary = train_missing_sensors()

    if summary:
        print("\n=== Training Summary ===")
        print(f"Total missing sensors: {summary['total_missing']}")
        print(f"SMAP missing: {summary['smap_missing']}")
        print(f"MSL missing: {summary['msl_missing']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
    print("=" * 50)