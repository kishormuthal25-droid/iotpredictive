"""
Telemanom Model Training Pipeline
Train NASA Telemanom models for all 80 sensors in SMAP/MSL systems
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import json
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.anomaly_detection.nasa_telemanom import NASATelemanom, Telemanom_Config, quick_train_telemanom
from src.data_ingestion.equipment_mapper import IoTEquipmentMapper
from src.data_ingestion.nasa_data_ingestion_service import NASADataIngestionService
from config.settings import get_data_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelemanoMTrainingPipeline:
    """Training pipeline for NASA Telemanom models"""

    def __init__(self, models_dir: str = "data/models/telemanom"):
        """Initialize training pipeline

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data services
        self.equipment_mapper = IoTEquipmentMapper()
        self.data_service = NASADataIngestionService()

        # Get equipment definitions
        self.smap_equipment = self.equipment_mapper.smap_equipment
        self.msl_equipment = self.equipment_mapper.msl_equipment

        # Training results
        self.training_results = {}
        self.failed_sensors = []

        logger.info(f"Initialized training pipeline. Models will be saved to: {self.models_dir}")

    def get_sensor_mapping(self) -> Dict[str, Dict]:
        """Get mapping of all 80 sensors to their equipment

        Returns:
            Dictionary mapping sensor IDs to equipment info
        """
        sensor_mapping = {}
        sensor_id = 0

        # SMAP sensors (25 total)
        for equipment in self.smap_equipment:
            for sensor in equipment.sensors:
                sensor_key = f"SMAP_{sensor_id:02d}_{sensor.name.replace(' ', '_')}"
                sensor_mapping[sensor_key] = {
                    'equipment_id': equipment.equipment_id,
                    'equipment_type': equipment.equipment_type,
                    'subsystem': equipment.subsystem,
                    'sensor_name': sensor.name,
                    'sensor_unit': sensor.unit,
                    'sensor_id': sensor_id,
                    'criticality': equipment.criticality,
                    'dataset': 'SMAP'
                }
                sensor_id += 1

        # MSL sensors (55 total)
        for equipment in self.msl_equipment:
            for sensor in equipment.sensors:
                sensor_key = f"MSL_{sensor_id:02d}_{sensor.name.replace(' ', '_')}"
                sensor_mapping[sensor_key] = {
                    'equipment_id': equipment.equipment_id,
                    'equipment_type': equipment.equipment_type,
                    'subsystem': equipment.subsystem,
                    'sensor_name': sensor.name,
                    'sensor_unit': sensor.unit,
                    'sensor_id': sensor_id,
                    'criticality': equipment.criticality,
                    'dataset': 'MSL'
                }
                sensor_id += 1

        logger.info(f"Mapped {len(sensor_mapping)} sensors total")
        return sensor_mapping

    def load_sensor_data(self, sensor_info: Dict, sample_size: int = 10000) -> Optional[np.ndarray]:
        """Load training data for a specific sensor

        Args:
            sensor_info: Sensor information dictionary
            sample_size: Number of samples to load for training

        Returns:
            Training data array or None if failed
        """
        try:
            dataset = sensor_info['dataset']
            sensor_id = sensor_info['sensor_id']

            # Load NASA data (this would be adapted based on your actual data format)
            if dataset == 'SMAP':
                # Load SMAP data - adapt this to your actual data loading
                data = self.data_service.get_smap_sensor_data(sensor_id, limit=sample_size)
            else:  # MSL
                # Load MSL data - adapt this to your actual data loading
                data = self.data_service.get_msl_sensor_data(sensor_id, limit=sample_size)

            if data is None or len(data) < 500:  # Minimum data requirement
                logger.warning(f"Insufficient data for sensor {sensor_info['sensor_name']}: {len(data) if data is not None else 0} samples")
                return None

            # Convert to numpy array and handle preprocessing
            if isinstance(data, pd.DataFrame):
                # Assume first column is timestamp, rest are sensor values
                if data.shape[1] > 1:
                    sensor_data = data.iloc[:, 1:].values  # Skip timestamp
                else:
                    sensor_data = data.values
            else:
                sensor_data = np.array(data)

            # Ensure 2D array
            if len(sensor_data.shape) == 1:
                sensor_data = sensor_data.reshape(-1, 1)

            logger.info(f"Loaded {len(sensor_data)} samples for {sensor_info['sensor_name']}")
            return sensor_data

        except Exception as e:
            logger.error(f"Failed to load data for sensor {sensor_info['sensor_name']}: {e}")
            return None

    def train_single_sensor(self, sensor_key: str, sensor_info: Dict,
                           quick_mode: bool = False) -> Optional[NASATelemanom]:
        """Train Telemanom model for a single sensor

        Args:
            sensor_key: Sensor identifier
            sensor_info: Sensor information
            quick_mode: Use quick training (5 epochs) or full training

        Returns:
            Trained model or None if failed
        """
        try:
            logger.info(f"Training model for sensor: {sensor_key}")

            # Load training data
            training_data = self.load_sensor_data(sensor_info)
            if training_data is None:
                return None

            # Configure model
            if quick_mode:
                config = Telemanom_Config(
                    epochs=5,
                    sequence_length=50,  # Shorter for quick training
                    batch_size=32
                )
            else:
                # Full training configuration
                config = Telemanom_Config(
                    epochs=35,  # NASA default
                    sequence_length=100,  # Shorter than NASA default for efficiency
                    batch_size=70
                )

                # Adjust based on equipment criticality
                if sensor_info['criticality'] == 'CRITICAL':
                    config.epochs = 50
                    config.sequence_length = 150

            # Create and train model
            model = NASATelemanom(sensor_key, config)

            # Split data for training/validation
            split_idx = int(len(training_data) * 0.8)
            train_data = training_data[:split_idx]
            val_data = training_data[split_idx:]

            # Train model
            history = model.fit(train_data, val_data)

            # Save model
            model_path = self.models_dir / f"{sensor_key}.pkl"
            model.save_model(str(model_path))

            # Store training results
            self.training_results[sensor_key] = {
                'success': True,
                'sensor_info': sensor_info,
                'model_path': str(model_path),
                'training_loss': history['loss'][-1] if 'loss' in history else None,
                'val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
                'threshold': model.error_threshold,
                'training_samples': len(train_data),
                'model_params': model.model.count_params() if model.model else 0
            }

            logger.info(f"Successfully trained {sensor_key} - Threshold: {model.error_threshold:.4f}")
            return model

        except Exception as e:
            logger.error(f"Failed to train sensor {sensor_key}: {e}")
            logger.error(traceback.format_exc())

            self.training_results[sensor_key] = {
                'success': False,
                'sensor_info': sensor_info,
                'error': str(e)
            }
            self.failed_sensors.append(sensor_key)
            return None

    def train_all_sensors(self, quick_mode: bool = False,
                         sensor_range: Optional[Tuple[int, int]] = None) -> Dict:
        """Train models for all sensors

        Args:
            quick_mode: Use quick training mode
            sensor_range: Optional tuple (start, end) to train only specific sensors

        Returns:
            Training results summary
        """
        logger.info(f"Starting {'quick' if quick_mode else 'full'} training for all sensors")

        # Get sensor mapping
        sensor_mapping = self.get_sensor_mapping()
        sensors_to_train = list(sensor_mapping.items())

        # Filter by range if specified
        if sensor_range:
            start, end = sensor_range
            sensors_to_train = [(k, v) for k, v in sensors_to_train
                              if start <= v['sensor_id'] <= end]

        logger.info(f"Training {len(sensors_to_train)} sensors")

        # Train each sensor
        successful_models = 0
        for sensor_key, sensor_info in sensors_to_train:
            model = self.train_single_sensor(sensor_key, sensor_info, quick_mode)
            if model:
                successful_models += 1

            # Log progress
            progress = ((sensor_info['sensor_id'] - (sensor_range[0] if sensor_range else 0)) + 1)
            total = len(sensors_to_train)
            logger.info(f"Progress: {progress}/{total} sensors processed")

        # Save training summary
        summary = {
            'total_sensors': len(sensors_to_train),
            'successful_models': successful_models,
            'failed_models': len(self.failed_sensors),
            'quick_mode': quick_mode,
            'sensor_range': sensor_range,
            'timestamp': datetime.now().isoformat(),
            'results': self.training_results
        }

        # Save summary to file
        summary_path = self.models_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training completed: {successful_models}/{len(sensors_to_train)} successful")
        logger.info(f"Summary saved to: {summary_path}")

        return summary

    def test_trained_models(self, sample_sensors: int = 5) -> Dict:
        """Test a sample of trained models

        Args:
            sample_sensors: Number of models to test

        Returns:
            Test results
        """
        logger.info(f"Testing {sample_sensors} trained models")

        # Find trained models
        model_files = list(self.models_dir.glob("*.pkl"))
        if len(model_files) == 0:
            logger.warning("No trained models found")
            return {'error': 'No trained models found'}

        # Test sample of models
        test_results = {}
        tested_count = 0

        for model_file in model_files[:sample_sensors]:
            try:
                # Load model
                model = NASATelemanom.load_model(str(model_file))

                # Generate test data (synthetic for testing)
                test_data = np.random.randn(1000, model.n_features)

                # Run anomaly detection
                results = model.predict_anomalies(test_data)

                test_results[model.sensor_id] = {
                    'model_loaded': True,
                    'anomalies_detected': int(np.sum(results['anomalies'])),
                    'max_score': float(np.max(results['scores'])),
                    'threshold': float(model.error_threshold),
                    'model_info': model.get_model_info()
                }

                tested_count += 1
                logger.info(f"Test passed for {model.sensor_id}")

            except Exception as e:
                test_results[model_file.stem] = {
                    'model_loaded': False,
                    'error': str(e)
                }
                logger.error(f"Test failed for {model_file.stem}: {e}")

        summary = {
            'tested_models': tested_count,
            'successful_tests': sum(1 for r in test_results.values() if r.get('model_loaded', False)),
            'results': test_results
        }

        logger.info(f"Testing completed: {summary['successful_tests']}/{tested_count} models passed")
        return summary


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train NASA Telemanom models')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (5 epochs)')
    parser.add_argument('--sensor-range', type=str,
                       help='Sensor range to train (e.g., "0-10")')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing models')
    parser.add_argument('--models-dir', type=str, default='data/models/telemanom',
                       help='Directory to save models')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TelemanoMTrainingPipeline(args.models_dir)

    if args.test_only:
        # Test existing models
        results = pipeline.test_trained_models()
        print("\nTest Results:")
        print(f"Successful tests: {results.get('successful_tests', 0)}")
        return

    # Parse sensor range
    sensor_range = None
    if args.sensor_range:
        try:
            start, end = map(int, args.sensor_range.split('-'))
            sensor_range = (start, end)
        except ValueError:
            logger.error("Invalid sensor range format. Use 'start-end' (e.g., '0-10')")
            return

    # Train models
    try:
        results = pipeline.train_all_sensors(
            quick_mode=args.quick,
            sensor_range=sensor_range
        )

        print("\nTraining Summary:")
        print(f"Total sensors: {results['total_sensors']}")
        print(f"Successful: {results['successful_models']}")
        print(f"Failed: {results['failed_models']}")

        if results['failed_models'] > 0:
            print(f"Failed sensors: {pipeline.failed_sensors}")

        # Test a few models
        if results['successful_models'] > 0:
            print("\nTesting trained models...")
            test_results = pipeline.test_trained_models(3)
            print(f"Test results: {test_results['successful_tests']}/{test_results['tested_models']} passed")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()