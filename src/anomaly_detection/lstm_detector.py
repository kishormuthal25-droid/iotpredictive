"""
LSTM Detector Module for Anomaly Detection
LSTM prediction-based anomaly detection (Telemanom approach)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Input, RepeatVector, TimeDistributed, Bidirectional,
    GRU, Conv1D, MaxPooling1D, Flatten, Attention
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, Callback
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.anomaly_detection.base_detector import (
    BaseAnomalyDetector, ThresholdStrategy, PercentileThreshold
)
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')


@dataclass
class LSTMConfig:
    """Configuration for LSTM detector"""
    # Architecture
    lstm_units: List[int] = None
    dropout_rate: float = 0.2
    use_bidirectional: bool = False
    use_attention: bool = False
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    
    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'mse'
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Sequence parameters
    sequence_length: int = 100
    prediction_length: int = 10
    stride: int = 1
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    use_batch_norm: bool = False
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]


class AnomalyCallback(Callback):
    """Custom callback for tracking anomaly detection metrics during training"""
    
    def __init__(self, validation_data=None):
        super().__init__()
        self.validation_data = validation_data
        self.history = {'loss': [], 'val_loss': [], 'lr': []}
    
    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['lr'].append(self.model.optimizer.learning_rate.numpy())
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: loss={logs.get('loss'):.4f}, "
                       f"val_loss={logs.get('val_loss'):.4f}")


class LSTMDetector(BaseAnomalyDetector):
    """
    LSTM-based anomaly detector using prediction error
    Implements the Telemanom approach for time series anomaly detection
    """
    
    def __init__(self,
                 name: str = "LSTMDetector",
                 config: Optional[LSTMConfig] = None,
                 threshold_strategy: Optional[ThresholdStrategy] = None,
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize LSTM detector
        
        Args:
            name: Detector name
            config: LSTM configuration
            threshold_strategy: Threshold determination strategy
            contamination: Expected anomaly ratio
            random_state: Random seed
        """
        super().__init__(
            name=name,
            threshold_strategy=threshold_strategy or PercentileThreshold(100 * (1 - contamination)),
            contamination=contamination,
            window_size=config.sequence_length if config else 100,
            random_state=random_state
        )
        
        self.config = config or LSTMConfig()
        
        # Model components
        self.model: Optional[Model] = None
        self.scaler: Optional[MinMaxScaler] = None
        
        # Training state
        self.training_history = None
        self.best_model_path = None
        
        # Error statistics
        self.error_stats = {'mean': 0, 'std': 1}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info(f"LSTMDetector initialized with sequence_length={self.config.sequence_length}")
    
    def _build_model(self):
        """Build LSTM model architecture"""
        inputs = Input(shape=(self.config.sequence_length, self.n_features))
        x = inputs
        
        # Add convolutional layers for feature extraction (optional)
        if self.n_features > 1:
            x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x) if self.config.use_batch_norm else x
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            # Regularization
            if self.config.l1_reg > 0 or self.config.l2_reg > 0:
                regularizer = keras.regularizers.L1L2(l1=self.config.l1_reg, l2=self.config.l2_reg)
            else:
                regularizer = None
            
            # Return sequences for all but last layer
            return_sequences = (i < len(self.config.lstm_units) - 1)
            
            # Create LSTM layer
            if self.config.use_bidirectional:
                lstm = Bidirectional(LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer
                ))(x)
            else:
                lstm = LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    recurrent_regularizer=regularizer
                )(x)
            
            # Batch normalization
            if self.config.use_batch_norm:
                lstm = BatchNormalization()(lstm)
            
            # Dropout
            if self.config.dropout_rate > 0:
                lstm = Dropout(self.config.dropout_rate)(lstm)
            
            x = lstm
        
        # Attention mechanism (optional)
        if self.config.use_attention and len(self.config.lstm_units) > 1:
            # Need to reshape for attention
            x = RepeatVector(self.config.prediction_length)(x)
            x = LSTM(self.config.lstm_units[-1], return_sequences=True)(x)
            attention = Attention()([x, x])
            x = layers.GlobalAveragePooling1D()(attention)
        
        # Output layers for prediction
        if self.config.prediction_length > 1:
            # Multi-step prediction
            x = RepeatVector(self.config.prediction_length)(x)
            x = LSTM(self.config.lstm_units[-1], return_sequences=True)(x)
            outputs = TimeDistributed(Dense(self.n_features))(x)
        else:
            # Single-step prediction
            outputs = Dense(self.n_features)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self._compile_model()
        
        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")
    
    def _compile_model(self):
        """Compile the model with optimizer and loss"""
        # Select optimizer
        if self.config.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.config.learning_rate)
        
        # Select loss
        if self.config.loss == 'mse':
            loss = MeanSquaredError()
        elif self.config.loss == 'mae':
            loss = MeanAbsoluteError()
        else:
            loss = MeanSquaredError()
        
        # Compile
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse']
        )
    
    def _create_sequences(self, 
                         data: np.ndarray,
                         shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data
            shuffle: Whether to shuffle sequences
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(0, len(data) - self.config.sequence_length - self.config.prediction_length + 1, 
                      self.config.stride):
            X.append(data[i:i + self.config.sequence_length])
            
            if self.config.prediction_length == 1:
                y.append(data[i + self.config.sequence_length])
            else:
                y.append(data[i + self.config.sequence_length:i + self.config.sequence_length + self.config.prediction_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle if requested
        if shuffle:
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        return X, y
    
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit LSTM model
        
        Args:
            X: Training data
            y: Not used (unsupervised)
        """
        # Initialize scaler if not exists
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        
        # Ensure 2D for scaling
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 3:
            # Already sequenced, reshape for scaling
            original_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
            X = self.scaler.fit_transform(X)
            X = X.reshape(original_shape)
        else:
            X = self.scaler.fit_transform(X)
        
        # Create sequences if not already sequenced
        if X.ndim == 2:
            X_seq, y_seq = self._create_sequences(X, shuffle=True)
        else:
            X_seq = X
            # Create targets by shifting
            y_seq = np.roll(X, -1, axis=1)[:, -self.config.prediction_length:, :]
        
        # Split data for validation
        split_idx = int(len(X_seq) * (1 - self.config.validation_split))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Callbacks
        callbacks_list = self._get_callbacks()
        
        # Train model
        logger.info(f"Training LSTM on {len(X_train)} sequences")
        
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate error statistics on training data
        train_predictions = self.model.predict(X_train, verbose=0)
        train_errors = self._calculate_errors(X_train, train_predictions)
        self.error_stats['mean'] = np.mean(train_errors)
        self.error_stats['std'] = np.std(train_errors)
        
        logger.info(f"Training complete. Final loss: {self.training_history.history['loss'][-1]:.4f}")
    
    def _get_callbacks(self) -> List[Callback]:
        """Get training callbacks"""
        callbacks_list = []
        
        # Custom anomaly callback
        callbacks_list.append(AnomalyCallback())
        
        # Early stopping
        callbacks_list.append(EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        ))
        
        # Reduce learning rate
        callbacks_list.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=0
        ))
        
        # Model checkpoint
        checkpoint_path = get_data_path('models') / f'{self.name}_best.h5'
        callbacks_list.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ))
        self.best_model_path = checkpoint_path
        
        return callbacks_list
    
    def _calculate_errors(self, X: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate prediction errors
        
        Args:
            X: Input sequences
            predictions: Model predictions
            
        Returns:
            Error scores
        """
        # Get the actual next values
        if self.config.prediction_length == 1:
            # Single-step prediction
            actual = X[:, -1, :]  # Last timestep
            if predictions.ndim == 3:
                predictions = predictions.squeeze(1)
            errors = np.mean(np.abs(actual - predictions), axis=1)
        else:
            # Multi-step prediction
            actual = X[:, -self.config.prediction_length:, :]
            errors = np.mean(np.abs(actual - predictions), axis=(1, 2))
        
        return errors
    
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on prediction error
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores
        """
        # Scale data
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            # Create sequences
            X_seq, _ = self._create_sequences(X_scaled, shuffle=False)
            
            if len(X_seq) == 0:
                # Not enough data for sequences, pad
                padding = self.config.sequence_length - len(X_scaled)
                X_padded = np.pad(X_scaled, ((padding, 0), (0, 0)), mode='edge')
                X_seq = X_padded.reshape(1, self.config.sequence_length, -1)
        else:
            # Already sequenced
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X_seq = X_scaled.reshape(original_shape)
        
        # Get predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Calculate errors
        errors = self._calculate_errors(X_seq, predictions)
        
        # Normalize errors using training statistics
        if self.error_stats['std'] > 0:
            normalized_errors = (errors - self.error_stats['mean']) / self.error_stats['std']
        else:
            normalized_errors = errors
        
        # Convert to anomaly scores (higher score = more anomalous)
        scores = np.abs(normalized_errors)
        
        # If input was shorter than sequence length, repeat score
        if len(scores) < len(X):
            scores = np.repeat(scores[-1], len(X))
        
        return scores
    
    def predict_next(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next values in sequence
        
        Args:
            X: Input sequence
            
        Returns:
            Predicted next values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Prepare input
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_seq = X_scaled[-self.config.sequence_length:].reshape(1, self.config.sequence_length, -1)
        else:
            X_seq = X
        
        # Predict
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Inverse scale
        if predictions.ndim == 3:
            predictions = predictions.reshape(-1, self.n_features)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def detect_anomalies_in_sequence(self, 
                                    X: np.ndarray,
                                    smoothing_window: int = 5) -> np.ndarray:
        """
        Detect anomalies in a continuous sequence
        
        Args:
            X: Time series data
            smoothing_window: Window for error smoothing
            
        Returns:
            Binary anomaly labels for each timestep
        """
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Smooth scores if requested
        if smoothing_window > 1:
            from scipy.signal import savgol_filter
            scores = savgol_filter(scores, smoothing_window, 1)
        
        # Apply threshold
        anomalies = scores > self.threshold
        
        return anomalies
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving"""
        state = {
            'config': self.config,
            'error_stats': self.error_stats,
            'model_weights': self.model.get_weights() if self.model else None,
            'scaler': self.scaler
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state for loading"""
        self.config = state['config']
        self.error_stats = state['error_stats']
        self.scaler = state['scaler']
        
        if state['model_weights'] and self.model:
            self.model.set_weights(state['model_weights'])
    
    def plot_predictions(self, 
                        X: np.ndarray,
                        n_samples: int = 200,
                        save_path: Optional[Path] = None):
        """
        Plot predictions vs actual values
        
        Args:
            X: Input data
            n_samples: Number of samples to plot
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        # Get predictions
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_seq, _ = self._create_sequences(X_scaled, shuffle=False)
        else:
            X_seq = X
        
        predictions = self.model.predict(X_seq[:n_samples], verbose=0)
        
        # Calculate errors
        errors = self._calculate_errors(X_seq[:n_samples], predictions)
        
        # Create plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted (first feature)
        actual = X_seq[:n_samples, -1, 0]
        if self.config.prediction_length == 1:
            pred = predictions[:n_samples, 0] if predictions.ndim == 2 else predictions[:n_samples]
        else:
            pred = predictions[:n_samples, 0, 0]
        
        axes[0].plot(actual, label='Actual', alpha=0.7)
        axes[0].plot(pred, label='Predicted', alpha=0.7)
        axes[0].set_title('Actual vs Predicted Values (First Feature)')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        axes[1].plot(errors, label='Prediction Error', color='red', alpha=0.7)
        axes[1].axhline(self.threshold, color='green', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[1].set_title('Prediction Errors')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Anomaly Detection
        anomalies = errors > self.threshold
        axes[2].scatter(np.arange(len(errors))[~anomalies], 
                       errors[~anomalies], 
                       c='blue', alpha=0.5, s=10, label='Normal')
        axes[2].scatter(np.arange(len(errors))[anomalies],
                       errors[anomalies],
                       c='red', alpha=0.7, s=20, label='Anomaly')
        axes[2].axhline(self.threshold, color='green', linestyle='--', alpha=0.5)
        axes[2].set_title('Anomaly Detection Results')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Error Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet")


class BiLSTMDetector(LSTMDetector):
    """Bidirectional LSTM detector"""
    
    def __init__(self, **kwargs):
        super().__init__(name="BiLSTMDetector", **kwargs)
        self.config.use_bidirectional = True


class AttentionLSTMDetector(LSTMDetector):
    """LSTM with attention mechanism"""
    
    def __init__(self, **kwargs):
        super().__init__(name="AttentionLSTMDetector", **kwargs)
        self.config.use_attention = True


class StackedLSTMDetector(LSTMDetector):
    """Deep stacked LSTM detector"""
    
    def __init__(self, n_layers: int = 5, **kwargs):
        super().__init__(name="StackedLSTMDetector", **kwargs)
        # Create deep architecture
        units = [256] * n_layers
        for i in range(n_layers):
            units[i] = units[i] // (2 ** (i // 2))  # Gradually decrease units
        self.config.lstm_units = units


if __name__ == "__main__":
    # Test LSTM detector
    print("\n" + "="*60)
    print("Testing LSTM Anomaly Detector")
    print("="*60)
    
    # Create synthetic time series data
    np.random.seed(42)
    t = np.linspace(0, 100, 2000)
    
    # Normal pattern: sine wave with noise
    normal_signal = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t))
    
    # Add anomalies
    anomaly_indices = [500, 1000, 1500]
    for idx in anomaly_indices:
        # Different types of anomalies
        if idx == 500:
            # Amplitude anomaly
            normal_signal[idx:idx+50] *= 3
        elif idx == 1000:
            # Frequency anomaly
            normal_signal[idx:idx+50] = np.sin(2 * np.pi * 0.5 * t[idx:idx+50])
        else:
            # Random noise anomaly
            normal_signal[idx:idx+50] = np.random.randn(50) * 2
    
    # Reshape for univariate time series
    data = normal_signal.reshape(-1, 1)
    
    # Create labels for evaluation
    labels = np.zeros(len(data))
    for idx in anomaly_indices:
        labels[idx:idx+50] = 1
    
    # Test 1: Basic LSTM Detector
    print("\n1. Testing Basic LSTM Detector...")
    
    config = LSTMConfig(
        lstm_units=[64, 32],
        sequence_length=50,
        prediction_length=1,
        epochs=20,
        batch_size=32
    )
    
    detector = LSTMDetector(config=config, contamination=0.05)
    
    # Fit detector
    print("   Training LSTM...")
    detector.fit(data[:1500], labels[:1500])
    
    # Test detection
    print("\n   Testing on remaining data...")
    test_data = data[1500:]
    test_labels = labels[1500:]
    
    predictions = detector.predict(test_data)
    scores = detector.score_samples(test_data)
    
    # Calculate simple accuracy
    if len(predictions) == len(test_labels[:len(predictions)]):
        accuracy = np.mean(predictions == test_labels[:len(predictions)])
        print(f"   Test accuracy: {accuracy:.3f}")
    
    # Test 2: Bidirectional LSTM
    print("\n2. Testing Bidirectional LSTM...")
    bi_detector = BiLSTMDetector(config=config, contamination=0.05)
    bi_detector.fit(data[:1500])
    bi_predictions = bi_detector.predict(test_data)
    print(f"   Detected {np.sum(bi_predictions)} anomalies in test data")
    
    # Test 3: Multi-step prediction
    print("\n3. Testing Multi-step Prediction...")
    config_multi = LSTMConfig(
        lstm_units=[64, 32],
        sequence_length=50,
        prediction_length=10,  # Predict 10 steps ahead
        epochs=20
    )
    
    multi_detector = LSTMDetector(
        name="MultiStepLSTM",
        config=config_multi,
        contamination=0.05
    )
    multi_detector.fit(data[:1500])
    
    # Predict next values
    next_values = multi_detector.predict_next(data[1000:1050])
    print(f"   Predicted next {config_multi.prediction_length} values shape: {next_values.shape}")
    
    # Test 4: Continuous sequence anomaly detection
    print("\n4. Testing Sequence Anomaly Detection...")
    sequence_anomalies = detector.detect_anomalies_in_sequence(data, smoothing_window=5)
    print(f"   Detected {np.sum(sequence_anomalies)} anomalous points")
    
    # Test 5: Visualization
    print("\n5. Generating prediction plots...")
    detector.plot_predictions(data[:500], n_samples=100)
    
    # Test 6: Model summary
    print("\n6. Model Architecture:")
    detector.summary()
    
    # Test 7: Save and load
    print("\n7. Testing save/load...")
    save_path = get_data_path('models') / 'test_lstm_detector.pkl'
    detector.save(save_path)
    
    # Create new detector and load
    new_detector = LSTMDetector(config=config)
    new_detector.load(save_path)
    
    # Verify loaded model works
    loaded_predictions = new_detector.predict(test_data[:10])
    print(f"   Loaded model predictions: {loaded_predictions}")
    
    print("\n" + "="*60)
    print("LSTM detector test complete")
    print("="*60)
