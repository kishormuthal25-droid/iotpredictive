"""
NASA Telemanom Implementation
Official NASA algorithm for spacecraft telemetry anomaly detection
Based on https://github.com/khundman/telemanom
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class Telemanom_Config:
    """Configuration for NASA Telemanom model"""
    # Model architecture
    sequence_length: int = 250  # NASA default l_s
    lstm_units: List[int] = None  # Will default to [80, 80]
    dropout_rate: float = 0.3
    prediction_length: int = 10  # Prediction window

    # Training parameters
    epochs: int = 35  # NASA default
    batch_size: int = 70  # NASA default
    learning_rate: float = 0.001
    validation_split: float = 0.2

    # Anomaly detection
    error_buffer: int = 100  # Buffer for error distribution
    smoothing_window: int = 30  # Smoothing window for errors
    contamination: float = 0.05  # Expected anomaly rate

    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [80, 80]


class NASATelemanom:
    """
    NASA Telemanom Anomaly Detection Algorithm

    Implementation of the official NASA spacecraft telemetry anomaly detection
    system using LSTM neural networks with dynamic thresholding.
    """

    def __init__(self, sensor_id: str, config: Optional[Telemanom_Config] = None):
        """Initialize Telemanom model for specific sensor

        Args:
            sensor_id: Unique identifier for the sensor
            config: Model configuration
        """
        self.sensor_id = sensor_id
        self.config = config or Telemanom_Config()

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Training data and statistics
        self.training_errors = None
        self.error_threshold = None
        self.smoothed_errors = None

        # Model metadata
        self.n_features = None
        self.training_history = None

        logger.info(f"Initialized NASA Telemanom for sensor {sensor_id}")

    def _build_model(self) -> keras.Model:
        """Build LSTM model following NASA Telemanom architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                units=self.config.lstm_units[0],
                return_sequences=True,
                input_shape=(self.config.sequence_length, self.n_features),
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),

            # Second LSTM layer (if specified)
            layers.LSTM(
                units=self.config.lstm_units[1] if len(self.config.lstm_units) > 1 else self.config.lstm_units[0],
                return_sequences=False,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.dropout_rate
            ),

            # Dense layers for prediction
            layers.Dense(units=self.n_features * self.config.prediction_length, activation='linear'),
            layers.Reshape((self.config.prediction_length, self.n_features))
        ])

        # Compile with Adam optimizer (NASA default)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for training

        Args:
            data: Input time series data [timesteps, features]

        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = []

        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_length

        for i in range(len(data) - seq_len - pred_len + 1):
            # Input sequence
            sequence = data[i:i + seq_len]
            # Target is next pred_len timesteps
            target = data[i + seq_len:i + seq_len + pred_len]

            sequences.append(sequence)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def fit(self, data: np.ndarray, validation_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train NASA Telemanom model on normal data

        Args:
            data: Training data [timesteps, features]
            validation_data: Optional validation data

        Returns:
            Training history
        """
        logger.info(f"Training Telemanom model for sensor {self.sensor_id}")

        # Store data characteristics
        self.n_features = data.shape[1] if len(data.shape) > 1 else 1
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Normalize data
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self._create_sequences(data_scaled)

        if len(X) == 0:
            raise ValueError(f"Insufficient data for sequence length {self.config.sequence_length}")

        logger.info(f"Created {len(X)} training sequences")

        # Build model
        self.model = self._build_model()

        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            if len(validation_data.shape) == 1:
                validation_data = validation_data.reshape(-1, 1)
            val_scaled = self.scaler.transform(validation_data)
            X_val, y_val = self._create_sequences(val_scaled)
            if len(X_val) > 0:
                val_data = (X_val, y_val)

        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_data else 'loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_data else 'loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]

        # Train model
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=val_data,
            validation_split=self.config.validation_split if val_data is None else 0,
            callbacks=callbacks,
            verbose=1
        )

        # Calculate training errors for threshold computation
        predictions = self.model.predict(X, verbose=0)

        # Reshape for error calculation
        y_flat = y.reshape(-1, self.n_features)
        pred_flat = predictions.reshape(-1, self.n_features)

        # Calculate point-wise errors
        errors = np.mean(np.square(y_flat - pred_flat), axis=1)
        self.training_errors = errors

        # Compute dynamic threshold using NASA's approach
        self._compute_threshold()

        self.is_trained = True
        self.training_history = history.history

        logger.info(f"Model training completed. Threshold: {self.error_threshold:.4f}")

        return history.history

    def _compute_threshold(self):
        """Compute dynamic threshold using NASA Telemanom approach"""
        if self.training_errors is None:
            raise ValueError("No training errors available for threshold computation")

        # Smooth errors
        errors = self.training_errors
        smoothed = self._smooth_errors(errors)
        self.smoothed_errors = smoothed

        # Use percentile-based threshold (NASA approach)
        # This is a simplified version of their nonparametric method
        threshold_percentile = (1 - self.config.contamination) * 100
        self.error_threshold = np.percentile(smoothed, threshold_percentile)

        # Ensure minimum threshold
        min_threshold = np.mean(smoothed) + 2 * np.std(smoothed)
        self.error_threshold = max(self.error_threshold, min_threshold)

        logger.info(f"Computed threshold: {self.error_threshold:.4f} "
                   f"(mean: {np.mean(smoothed):.4f}, std: {np.std(smoothed):.4f})")

    def _smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """Apply smoothing to errors using moving average"""
        window = min(self.config.smoothing_window, len(errors) // 5)
        if window < 2:
            return errors

        # Apply moving average
        smoothed = np.convolve(errors, np.ones(window)/window, mode='same')

        # Handle edge effects
        for i in range(window//2):
            smoothed[i] = np.mean(errors[:i+window//2+1])
            smoothed[-(i+1)] = np.mean(errors[-(i+window//2+1):])

        return smoothed

    def predict_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in new data

        Args:
            data: Input data [timesteps, features]

        Returns:
            Dictionary with anomaly results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before anomaly detection")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        # Scale data
        data_scaled = self.scaler.transform(data)

        # Create sequences
        X, y = self._create_sequences(data_scaled)

        if len(X) == 0:
            return {
                'anomalies': np.array([]),
                'errors': np.array([]),
                'scores': np.array([]),
                'threshold': self.error_threshold
            }

        # Get predictions
        predictions = self.model.predict(X, verbose=0)

        # Calculate errors
        y_flat = y.reshape(-1, self.n_features)
        pred_flat = predictions.reshape(-1, self.n_features)
        errors = np.mean(np.square(y_flat - pred_flat), axis=1)

        # Smooth errors
        smoothed_errors = self._smooth_errors(errors)

        # Detect anomalies
        anomalies = smoothed_errors > self.error_threshold

        # Calculate anomaly scores (normalized by threshold)
        scores = smoothed_errors / self.error_threshold

        # Map back to original time indices
        seq_len = self.config.sequence_length
        pred_len = self.config.prediction_length

        # Create full-length arrays
        full_anomalies = np.zeros(len(data), dtype=bool)
        full_errors = np.zeros(len(data))
        full_scores = np.zeros(len(data))

        # Fill in the detected values
        for i, (anom, error, score) in enumerate(zip(anomalies, smoothed_errors, scores)):
            # Map sequence to its corresponding time range
            start_idx = i + seq_len
            end_idx = min(start_idx + pred_len, len(data))

            if anom:
                full_anomalies[start_idx:end_idx] = True
            full_errors[start_idx:end_idx] = error
            full_scores[start_idx:end_idx] = score

        return {
            'anomalies': full_anomalies,
            'errors': full_errors,
            'scores': full_scores,
            'threshold': self.error_threshold,
            'raw_errors': errors,
            'smoothed_errors': smoothed_errors
        }

    def get_anomaly_score(self, data: np.ndarray) -> float:
        """Get single anomaly score for data window

        Args:
            data: Input data window

        Returns:
            Anomaly score (higher = more anomalous)
        """
        results = self.predict_anomalies(data)
        if len(results['scores']) == 0:
            return 0.0
        return np.max(results['scores'])

    def save_model(self, filepath: str):
        """Save complete model state

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model architecture and weights
        if self.model:
            model_path = filepath.parent / f"{filepath.stem}_model.h5"
            self.model.save(model_path)

        # Save complete state
        state = {
            'sensor_id': self.sensor_id,
            'config': self.config,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'training_errors': self.training_errors,
            'error_threshold': self.error_threshold,
            'smoothed_errors': self.smoothed_errors,
            'n_features': self.n_features,
            'training_history': self.training_history,
            'model_path': str(model_path) if self.model else None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'NASATelemanom':
        """Load complete model state

        Args:
            filepath: Path to saved model

        Returns:
            Loaded NASATelemanom instance
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance
        instance = cls(state['sensor_id'], state['config'])

        # Restore state
        instance.scaler = state['scaler']
        instance.is_trained = state['is_trained']
        instance.training_errors = state['training_errors']
        instance.error_threshold = state['error_threshold']
        instance.smoothed_errors = state['smoothed_errors']
        instance.n_features = state['n_features']
        instance.training_history = state['training_history']

        # Load model if available
        if state['model_path'] and Path(state['model_path']).exists():
            instance.model = keras.models.load_model(state['model_path'])

        logger.info(f"Model loaded from {filepath}")

        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics

        Returns:
            Model information dictionary
        """
        info = {
            'sensor_id': self.sensor_id,
            'is_trained': self.is_trained,
            'config': self.config.__dict__,
            'n_features': self.n_features,
            'error_threshold': self.error_threshold
        }

        if self.training_errors is not None:
            info.update({
                'training_error_stats': {
                    'mean': float(np.mean(self.training_errors)),
                    'std': float(np.std(self.training_errors)),
                    'min': float(np.min(self.training_errors)),
                    'max': float(np.max(self.training_errors))
                }
            })

        if self.model:
            info.update({
                'model_params': self.model.count_params(),
                'model_layers': len(self.model.layers)
            })

        return info


def quick_train_telemanom(sensor_id: str, data: np.ndarray,
                         epochs: int = 5) -> NASATelemanom:
    """Quick training function for testing

    Args:
        sensor_id: Sensor identifier
        data: Training data
        epochs: Number of epochs (default 5 for quick training)

    Returns:
        Trained Telemanom model
    """
    config = Telemanom_Config(epochs=epochs, sequence_length=50)  # Shorter for quick training
    model = NASATelemanom(sensor_id, config)
    model.fit(data)
    return model