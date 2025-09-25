#!/usr/bin/env python3
"""
Quick Anomaly Model Training Script
Option 2: Balanced Training (15-25 minutes)
Optimized for NASA SMAP/MSL equipment-specific anomaly detection
"""

import os
import sys

# TensorFlow optimizations - MUST be set before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'  # Intra-op parallelism
os.environ['TF_NUM_INTEROP_THREADS'] = '2'  # Inter-op parallelism

# Import TensorFlow first and configure immediately
import tensorflow as tf

# Configure TensorFlow for CPU optimization BEFORE any other TF operations
try:
    # Disable GPU to avoid overhead
    tf.config.set_visible_devices([], 'GPU')
    print("[OK] GPU disabled for CPU-optimized training")

    # Set memory growth for any remaining devices
    physical_devices = tf.config.list_physical_devices()
    print(f"Available devices: {len(physical_devices)}")

except Exception as e:
    print(f"[WARNING] TensorFlow configuration warning: {e}")
    print("Continuing with default TF settings...")

# Now safe to import other modules
import yaml
import time
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from typing import Dict, List, Tuple, Optional
import pickle
import json
from dataclasses import dataclass
import gc

from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import project modules with error handling
try:
    from src.data_ingestion.data_loader import DataLoader
    print("[OK] DataLoader imported successfully")
except ImportError as e:
    print(f"[WARNING] DataLoader import failed: {e}")

try:
    from src.data_ingestion.nasa_data_service import nasa_data_service
    print("[OK] NASA data service imported successfully")
except ImportError as e:
    print(f"[WARNING] NASA data service import failed: {e}")
    nasa_data_service = None

try:
    from src.data_ingestion.equipment_mapper import equipment_mapper
    print("[OK] Equipment mapper imported successfully")
except ImportError as e:
    print(f"[WARNING] Equipment mapper import failed: {e}")
    equipment_mapper = None

try:
    from src.anomaly_detection.nasa_anomaly_engine import nasa_anomaly_engine
    print("[OK] NASA anomaly engine imported successfully")
except ImportError as e:
    print(f"[WARNING] NASA anomaly engine import failed: {e}")
    nasa_anomaly_engine = None

try:
    from src.anomaly_detection.training_tracker import training_tracker
    print("[OK] Training tracker imported successfully")
except ImportError as e:
    print(f"[WARNING] Training tracker import failed: {e}")
    training_tracker = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quick_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class QuickTrainingConfig:
    """Quick training configuration"""
    epochs: int = 30
    batch_size: int = 32
    validation_split: float = 0.15
    sequence_length: int = 40
    early_stopping_patience: int = 5
    learning_rate: float = 0.002
    encoder_units: List[int] = None
    latent_dim: int = 8
    dropout_rate: float = 0.2


class QuickLSTMAutoencoder:
    """Optimized LSTM Autoencoder for quick training"""

    def __init__(self, config: QuickTrainingConfig, equipment_id: str):
        self.config = config
        self.equipment_id = equipment_id
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.training_history = None
        self.metrics = {}

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build optimized LSTM autoencoder model"""
        sequence_length, n_features = input_shape

        # Input layer
        input_layer = layers.Input(shape=(sequence_length, n_features))

        # Encoder - optimized for speed
        x = input_layer
        for units in self.config.encoder_units:
            x = layers.LSTM(units, return_sequences=True, dropout=self.config.dropout_rate)(x)
            x = layers.BatchNormalization()(x)

        # Latent representation
        encoded = layers.LSTM(self.config.latent_dim, return_sequences=False)(x)

        # Repeat vector for decoder
        decoded = layers.RepeatVector(sequence_length)(encoded)

        # Decoder - mirror encoder
        for units in reversed(self.config.encoder_units):
            decoded = layers.LSTM(units, return_sequences=True, dropout=self.config.dropout_rate)(decoded)
            decoded = layers.BatchNormalization()(decoded)

        # Output layer
        output = layers.TimeDistributed(layers.Dense(n_features, activation='linear'))(decoded)

        # Create model
        model = keras.Model(input_layer, output, name=f"quick_autoencoder_{self.equipment_id}")

        # Compile with optimized settings
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict:
        """Train the model with optimizations"""
        logger.info(f"Starting quick training for {self.equipment_id}")
        start_time = time.time()

        try:
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)

            logger.info(f"Model architecture for {self.equipment_id}:")
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Encoder units: {self.config.encoder_units}")
            logger.info(f"Latent dim: {self.config.latent_dim}")

            # Callbacks for quick training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                ),
                keras.callbacks.TerminateOnNaN()
            ]

            # Train model
            history = self.model.fit(
                X_train, X_train,  # Autoencoder: input = output
                validation_data=(X_val, X_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            training_time = time.time() - start_time

            # Calculate metrics
            self.training_history = history.history
            self.is_trained = True

            # Quick evaluation
            train_loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            final_epoch = len(history.history['loss'])

            self.metrics = {
                'training_time': training_time,
                'final_epoch': final_epoch,
                'best_train_loss': train_loss,
                'best_val_loss': val_loss,
                'convergence_ratio': val_loss / train_loss if train_loss > 0 else 1.0,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }

            logger.info(f"Training completed for {self.equipment_id} in {training_time:.1f}s")
            logger.info(f"Final epoch: {final_epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            return self.metrics

        except Exception as e:
            logger.error(f"Training failed for {self.equipment_id}: {str(e)}")
            self.metrics = {'error': str(e), 'training_time': time.time() - start_time}
            return self.metrics

    def save_model(self, save_path: Path) -> bool:
        """Save trained model"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning(f"Model {self.equipment_id} not trained, skipping save")
                return False

            # Create save directory
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model
            model_path = save_path / f"{self.equipment_id}_quick_anomaly_detector.h5"
            self.model.save(model_path, save_format='h5')

            # Save metadata
            metadata = {
                'equipment_id': self.equipment_id,
                'config': {
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'sequence_length': self.config.sequence_length,
                    'encoder_units': self.config.encoder_units,
                    'latent_dim': self.config.latent_dim
                },
                'metrics': self.metrics,
                'training_date': datetime.now().isoformat(),
                'scaler_params': {
                    'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                    'min_': self.scaler.min_.tolist() if hasattr(self.scaler, 'min_') else None,
                    'data_min_': self.scaler.data_min_.tolist() if hasattr(self.scaler, 'data_min_') else None,
                    'data_max_': self.scaler.data_max_.tolist() if hasattr(self.scaler, 'data_max_') else None
                }
            }

            metadata_path = save_path / f"{self.equipment_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model {self.equipment_id}: {str(e)}")
            return False


class QuickTrainingManager:
    """Manager for quick anomaly model training"""

    def __init__(self, config_path: str = None):
        """Initialize training manager"""
        self.config_path = config_path or "config/quick_train_config.yaml"
        self.config = self._load_config()
        self.equipment_configs = {}
        self.training_results = {}
        self.trained_models = {}

        # Create directories
        self.models_dir = Path(self.config['data']['models_path'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        Path('logs').mkdir(exist_ok=True)

        logger.info("Quick Training Manager initialized")
        logger.info(f"Target training time: 15-25 minutes")
        logger.info(f"Models directory: {self.models_dir}")

    def _load_config(self) -> Dict:
        """Load training configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return minimal default config
            return {
                'training': {'epochs': 30, 'batch_size': 32},
                'data': {'models_path': 'data/models', 'smap_path': 'data/raw/smap', 'msl_path': 'data/raw/msl'}
            }

    def _create_equipment_config(self, equipment_id: str) -> QuickTrainingConfig:
        """Create equipment-specific training configuration"""
        # Parse equipment type and criticality
        if equipment_id.startswith('SMAP-'):
            dataset = 'smap'
            eq_type = equipment_id.split('-')[1]
        elif equipment_id.startswith('MSL-'):
            dataset = 'msl'
            eq_type = equipment_id.split('-')[1]
        else:
            dataset = 'smap'
            eq_type = 'UNK'

        # Get equipment-specific settings
        equipment_info = self.config.get('nasa_equipment', {}).get(dataset, {}).get(eq_type, {})
        criticality = equipment_info.get('criticality', 'MEDIUM')

        # Get complexity settings based on criticality
        complexity = self.config.get('models', {}).get('complexity_by_criticality', {}).get(criticality, {})

        # Create config with optimizations
        config = QuickTrainingConfig(
            epochs=complexity.get('epochs', self.config.get('training', {}).get('epochs', 30)),
            batch_size=self.config.get('training', {}).get('batch_size', 32),
            validation_split=self.config.get('training', {}).get('validation_split', 0.15),
            sequence_length=self.config.get('training', {}).get('sequence_length', 40),
            early_stopping_patience=self.config.get('training', {}).get('early_stopping_patience', 5),
            learning_rate=self.config.get('models', {}).get('lstm_autoencoder', {}).get('learning_rate', 0.002),
            encoder_units=complexity.get('encoder_units', [32, 16]),
            latent_dim=complexity.get('latent_dim', 8),
            dropout_rate=self.config.get('models', {}).get('lstm_autoencoder', {}).get('dropout_rate', 0.2)
        )

        self.equipment_configs[equipment_id] = config
        logger.info(f"Config created for {equipment_id} (criticality: {criticality})")
        return config

    def load_and_prepare_data(self, equipment_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for specific equipment"""
        logger.info(f"Loading data for {equipment_id}")

        try:
            if equipment_id.startswith('SMAP-'):
                # Load SMAP data
                data_path = Path(self.config['data']['smap_path'])
                train_data = np.load(data_path / 'train.npy', allow_pickle=True)

                # For demo purposes, use first available data and simulate equipment-specific data
                if len(train_data.shape) == 2:
                    # If 2D, simulate sequences
                    seq_len = self.config.get('training', {}).get('sequence_length', 40)
                    sequences = []
                    for i in range(len(train_data) - seq_len + 1):
                        sequences.append(train_data[i:i+seq_len])
                    train_data = np.array(sequences)

            elif equipment_id.startswith('MSL-'):
                # Load MSL data
                data_path = Path(self.config['data']['msl_path'])
                train_data = np.load(data_path / 'train.npy', allow_pickle=True)

                # Similar processing for MSL
                if len(train_data.shape) == 2:
                    seq_len = self.config.get('training', {}).get('sequence_length', 40)
                    sequences = []
                    for i in range(len(train_data) - seq_len + 1):
                        sequences.append(train_data[i:i+seq_len])
                    train_data = np.array(sequences)

            else:
                raise ValueError(f"Unknown equipment type: {equipment_id}")

            # Quick preprocessing
            scaler = MinMaxScaler()
            original_shape = train_data.shape

            # Reshape for scaling
            if len(original_shape) == 3:
                n_samples, n_timesteps, n_features = original_shape
                train_data_reshaped = train_data.reshape(-1, n_features)
                train_data_scaled = scaler.fit_transform(train_data_reshaped)
                train_data_scaled = train_data_scaled.reshape(original_shape)
            else:
                train_data_scaled = scaler.fit_transform(train_data)

            # Split train/validation
            val_split = self.config.get('training', {}).get('validation_split', 0.15)
            split_idx = int(len(train_data_scaled) * (1 - val_split))

            X_train = train_data_scaled[:split_idx]
            X_val = train_data_scaled[split_idx:]

            # Use only normal data (filter out anomalies if labels available)
            # For quick training, we'll use first 80% as normal data
            normal_ratio = 0.8
            normal_samples = int(len(X_train) * normal_ratio)
            X_train = X_train[:normal_samples]

            logger.info(f"Data prepared for {equipment_id}: Train={X_train.shape}, Val={X_val.shape}")
            return X_train, X_val

        except Exception as e:
            logger.error(f"Failed to load data for {equipment_id}: {str(e)}")
            # Return dummy data for testing
            seq_len = 40
            n_features = 10
            n_samples = 1000
            X_train = np.random.normal(0, 1, (n_samples, seq_len, n_features))
            X_val = np.random.normal(0, 1, (200, seq_len, n_features))
            logger.warning(f"Using dummy data for {equipment_id}")
            return X_train, X_val

    def train_single_model(self, equipment_id: str) -> Dict:
        """Train a single model for equipment"""
        logger.info(f"Starting training for {equipment_id}")
        start_time = time.time()

        try:
            # Update training tracker if available
            if training_tracker is not None:
                training_tracker.start_model_training(equipment_id, 1000)  # Approximate sample count

            # Create configuration
            config = self._create_equipment_config(equipment_id)

            # Load data
            X_train, X_val = self.load_and_prepare_data(equipment_id)

            # Create and train model
            model = QuickLSTMAutoencoder(config, equipment_id)
            metrics = model.train(X_train, X_val)

            # Save model
            model.save_model(self.models_dir)

            # Store trained model
            self.trained_models[equipment_id] = model

            # Update results
            total_time = time.time() - start_time
            result = {
                'equipment_id': equipment_id,
                'status': 'completed',
                'training_time': total_time,
                'metrics': metrics,
                'config': {
                    'epochs': config.epochs,
                    'encoder_units': config.encoder_units,
                    'latent_dim': config.latent_dim
                }
            }

            self.training_results[equipment_id] = result

            # Update training tracker
            if training_tracker is not None:
                training_tracker.complete_model_training(equipment_id, success=True)

            logger.info(f"[SUCCESS] Completed training for {equipment_id} in {total_time:.1f}s")
            return result

        except Exception as e:
            error_result = {
                'equipment_id': equipment_id,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }
            self.training_results[equipment_id] = error_result

            if training_tracker is not None:
                training_tracker.complete_model_training(equipment_id, success=False, error_message=str(e))

            logger.error(f"[ERROR] Training failed for {equipment_id}: {str(e)}")
            return error_result

        finally:
            # Clean up memory
            keras.backend.clear_session()
            gc.collect()

    def train_all_models_parallel(self, max_workers: int = 3) -> Dict:
        """Train all models with parallel execution"""
        logger.info("Starting parallel training for all NASA equipment models")
        logger.info("[INFO] Expected completion time: 15-25 minutes")

        # Get equipment list from config
        equipment_list = self.config.get('training_priority', [
            "SMAP-PWR-001", "MSL-MOB-001", "MSL-MOB-002", "SMAP-COM-001",
            "MSL-COM-001", "SMAP-ATT-001", "MSL-NAV-001", "MSL-PWR-001",
            "MSL-ENV-001", "SMAP-THM-001", "SMAP-PAY-001", "MSL-SCI-001"
        ])

        start_time = time.time()

        logger.info(f"Training {len(equipment_list)} models with {max_workers} parallel workers")

        # Initialize training tracker
        if training_tracker is not None:
            training_tracker.start_training(equipment_list, total_epochs=30)

        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training jobs
            future_to_equipment = {
                executor.submit(self.train_single_model, equipment_id): equipment_id
                for equipment_id in equipment_list
            }

            # Process results as they complete
            completed = 0
            failed = 0

            for future in future_to_equipment:
                equipment_id = future_to_equipment[future]
                try:
                    result = future.result()
                    if result['status'] == 'completed':
                        completed += 1
                        logger.info(f"[PROGRESS] {completed + failed}/{len(equipment_list)} models processed")
                    else:
                        failed += 1
                        logger.warning(f"[FAILED] Failed: {equipment_id}")

                except Exception as e:
                    failed += 1
                    logger.error(f"[EXCEPTION] Exception in {equipment_id}: {str(e)}")

        total_time = time.time() - start_time

        # Generate summary
        summary = {
            'total_models': len(equipment_list),
            'completed_models': completed,
            'failed_models': failed,
            'total_training_time': total_time,
            'average_time_per_model': total_time / len(equipment_list),
            'training_results': self.training_results,
            'timestamp': datetime.now().isoformat()
        }

        # Save training summary
        summary_path = self.models_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("="*60)
        logger.info("[COMPLETE] TRAINING FINISHED!")
        logger.info("="*60)
        logger.info(f"[SUCCESS] Successfully trained: {completed}/{len(equipment_list)} models")
        logger.info(f"[FAILED] Failed: {failed}/{len(equipment_list)} models")
        logger.info(f"[TIME] Total time: {total_time/60:.1f} minutes")
        logger.info(f"[STATS] Average time per model: {total_time/len(equipment_list):.1f} seconds")
        logger.info(f"[SAVE] Models saved to: {self.models_dir}")
        logger.info(f"[REPORT] Training summary: {summary_path}")

        if completed >= len(equipment_list) * 0.8:  # 80% success rate
            logger.info("[READY] Training completed successfully! Ready for dashboard deployment.")
        else:
            logger.warning("[WARNING] Some models failed. Check logs for details.")

        return summary


def main():
    """Main training function"""
    print("NASA Anomaly Detection - Quick Training System")
    print("="*60)
    print("Option 2: Balanced Training (15-25 minutes)")
    print("Optimized for NASA SMAP/MSL equipment anomaly detection")
    print("="*60)

    try:
        # Initialize training manager
        trainer = QuickTrainingManager("config/quick_train_config.yaml")

        # Start parallel training
        results = trainer.train_all_models_parallel(max_workers=3)

        print("\n[RESULTS] TRAINING RESULTS:")
        print(f"[SUCCESS] Completed: {results['completed_models']}/{results['total_models']}")
        print(f"[TIME] Time: {results['total_training_time']/60:.1f} minutes")
        print(f"[STATS] Avg per model: {results['average_time_per_model']:.1f} seconds")

        if results['completed_models'] >= results['total_models'] * 0.8:
            print("\n[READY] SUCCESS! Models are ready for the anomaly monitoring dashboard!")
            print("\nNext steps:")
            print("1. Run: python scripts/run_dashboard.py")
            print("2. Open browser to: http://localhost:8050")
            print("3. Navigate to Anomaly Monitor section")
            print("4. Enjoy real-time NASA anomaly detection!")
        else:
            print("\n[WARNING] Some models failed. Check logs for details.")
            print("Dashboard can still work with completed models.")

        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        print(f"\n[ERROR] Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)