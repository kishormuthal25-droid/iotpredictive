"""
LSTM Forecaster Module for Time Series Anomaly Detection
Forecasting-based anomaly detection using LSTM models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
import json
import pickle

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Input, Bidirectional, GRU, Conv1D, MaxPooling1D,
    Flatten, Reshape, Layer, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, Callback, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
class LSTMForecasterConfig:
    """Configuration for LSTM Forecaster"""
    # Model architecture
    lstm_units: List[int] = None
    dense_units: List[int] = None
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2
    use_bidirectional: bool = True
    use_attention: bool = False
    model_type: str = 'vanilla'  # 'vanilla', 'stacked', 'seq2seq', 'attention'
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'mse'
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Forecasting parameters
    lookback_steps: int = 50
    forecast_horizon: int = 10
    multi_step: bool = False
    
    # Preprocessing
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust'
    use_differencing: bool = False
    seasonal_period: Optional[int] = None
    
    # Anomaly detection
    threshold_method: str = 'dynamic'  # 'dynamic', 'static', 'adaptive'
    anomaly_threshold: float = 3.0  # Number of std deviations
    use_prediction_interval: bool = True
    confidence_level: float = 0.95
    
    # CPU optimization
    use_mixed_precision: bool = False
    prefetch_buffer: int = 2
    
    def __post_init__(self):
        """Set default values if not provided"""
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
        if self.dense_units is None:
            self.dense_units = [32, 16]


class CustomAttention(Layer):
    """Custom attention layer for time series"""
    
    def __init__(self, units=128, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_vector',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(CustomAttention, self).build(input_shape)
        
    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, tf.expand_dims(self.u, axis=1)), axis=1)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)


class LSTMForecaster(BaseAnomalyDetector):
    """LSTM-based time series forecaster for anomaly detection"""
    
    def __init__(self, config: Optional[LSTMForecasterConfig] = None):
        """Initialize LSTM Forecaster
        
        Args:
            config: Configuration object for the forecaster
        """
        super().__init__()
        self.config = config or LSTMForecasterConfig()
        self.model = None
        self.scaler = None
        self.history = None
        self.forecast_errors = []
        self.prediction_intervals = {}
        self._initialize_scaler()
        
        # Training metrics storage
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
        logger.info(f"Initialized LSTM Forecaster with config: {self.config}")
        
    def _initialize_scaler(self):
        """Initialize the appropriate scaler"""
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM forecasting model
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        if self.config.model_type == 'vanilla':
            return self._build_vanilla_lstm(input_shape)
        elif self.config.model_type == 'stacked':
            return self._build_stacked_lstm(input_shape)
        elif self.config.model_type == 'seq2seq':
            return self._build_seq2seq_lstm(input_shape)
        elif self.config.model_type == 'attention':
            return self._build_attention_lstm(input_shape)
        else:
            return self._build_vanilla_lstm(input_shape)
            
    def _build_vanilla_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Build vanilla LSTM model"""
        model = Sequential()
        
        # First LSTM layer
        if self.config.use_bidirectional:
            model.add(Bidirectional(
                LSTM(self.config.lstm_units[0], 
                     return_sequences=len(self.config.lstm_units) > 1,
                     dropout=self.config.dropout_rate,
                     recurrent_dropout=self.config.recurrent_dropout),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                self.config.lstm_units[0],
                return_sequences=len(self.config.lstm_units) > 1,
                dropout=self.config.dropout_rate,
                recurrent_dropout=self.config.recurrent_dropout,
                input_shape=input_shape
            ))
        
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(self.config.lstm_units[1:], 1):
            return_seq = i < len(self.config.lstm_units) - 1
            if self.config.use_bidirectional:
                model.add(Bidirectional(
                    LSTM(units, 
                         return_sequences=return_seq,
                         dropout=self.config.dropout_rate,
                         recurrent_dropout=self.config.recurrent_dropout)
                ))
            else:
                model.add(LSTM(
                    units,
                    return_sequences=return_seq,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                ))
            model.add(BatchNormalization())
            
        # Dense layers
        for units in self.config.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.config.dropout_rate))
            
        # Output layer
        if self.config.multi_step:
            model.add(Dense(self.config.forecast_horizon * input_shape[1]))
            model.add(Reshape((self.config.forecast_horizon, input_shape[1])))
        else:
            model.add(Dense(input_shape[1]))
            
        # Compile model
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss(),
            metrics=['mae', 'mse']
        )
        
        return model
        
    def _build_stacked_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Build stacked LSTM with skip connections"""
        inputs = Input(shape=input_shape)
        x = inputs
        
        # Stacked LSTM with residual connections
        for i, units in enumerate(self.config.lstm_units):
            return_seq = i < len(self.config.lstm_units) - 1
            
            if self.config.use_bidirectional:
                lstm_out = Bidirectional(
                    LSTM(units, 
                         return_sequences=True,
                         dropout=self.config.dropout_rate,
                         recurrent_dropout=self.config.recurrent_dropout)
                )(x)
            else:
                lstm_out = LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                )(x)
                
            lstm_out = BatchNormalization()(lstm_out)
            
            # Skip connection if dimensions match
            if x.shape[-1] == lstm_out.shape[-1]:
                x = Add()([x, lstm_out])
            else:
                x = lstm_out
                
            if not return_seq:
                x = Lambda(lambda x: x[:, -1, :])(x)
                
        # Dense layers
        for units in self.config.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.config.dropout_rate)(x)
            
        # Output
        if self.config.multi_step:
            outputs = Dense(self.config.forecast_horizon * input_shape[1])(x)
            outputs = Reshape((self.config.forecast_horizon, input_shape[1]))(outputs)
        else:
            outputs = Dense(input_shape[1])(x)
            
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss(),
            metrics=['mae', 'mse']
        )
        
        return model
        
    def _build_seq2seq_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Build sequence-to-sequence LSTM"""
        # Encoder
        encoder_inputs = Input(shape=input_shape)
        encoder = LSTM(self.config.lstm_units[0], 
                       return_state=True,
                       dropout=self.config.dropout_rate)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.config.forecast_horizon, input_shape[1]))
        decoder_lstm = LSTM(self.config.lstm_units[0], 
                           return_sequences=True,
                           return_state=True,
                           dropout=self.config.dropout_rate)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # Dense layer
        decoder_dense = TimeDistributed(Dense(input_shape[1]))
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Create model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss(),
            metrics=['mae', 'mse']
        )
        
        return model
        
    def _build_attention_lstm(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        x = inputs
        for i, units in enumerate(self.config.lstm_units):
            if self.config.use_bidirectional:
                x = Bidirectional(
                    LSTM(units, 
                         return_sequences=True,
                         dropout=self.config.dropout_rate,
                         recurrent_dropout=self.config.recurrent_dropout)
                )(x)
            else:
                x = LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.config.dropout_rate,
                    recurrent_dropout=self.config.recurrent_dropout
                )(x)
            x = BatchNormalization()(x)
            
        # Attention layer
        if self.config.use_attention:
            attention_output = CustomAttention(units=self.config.lstm_units[-1])(x)
        else:
            attention_output = Lambda(lambda x: x[:, -1, :])(x)
            
        # Dense layers
        for units in self.config.dense_units:
            attention_output = Dense(units, activation='relu')(attention_output)
            attention_output = Dropout(self.config.dropout_rate)(attention_output)
            
        # Output
        if self.config.multi_step:
            outputs = Dense(self.config.forecast_horizon * input_shape[1])(attention_output)
            outputs = Reshape((self.config.forecast_horizon, input_shape[1]))(outputs)
        else:
            outputs = Dense(input_shape[1])(attention_output)
            
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=self._get_optimizer(),
            loss=self._get_loss(),
            metrics=['mae', 'mse']
        )
        
        return model
        
    def _get_optimizer(self):
        """Get optimizer based on configuration"""
        if self.config.optimizer == 'adam':
            return Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            return RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'sgd':
            return SGD(learning_rate=self.config.learning_rate, momentum=0.9)
        else:
            return Adam(learning_rate=self.config.learning_rate)
            
    def _get_loss(self):
        """Get loss function based on configuration"""
        if self.config.loss == 'mse':
            return MeanSquaredError()
        elif self.config.loss == 'mae':
            return MeanAbsoluteError()
        elif self.config.loss == 'huber':
            return Huber()
        else:
            return MeanSquaredError()
            
    def create_sequences(self, data: np.ndarray, 
                         lookback: int, 
                         horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting
        
        Args:
            data: Input time series data
            lookback: Number of timesteps to look back
            horizon: Number of timesteps to forecast
            
        Returns:
            X and y sequences for training
        """
        X, y = [], []
        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i - lookback:i])
            if self.config.multi_step:
                y.append(data[i:i + horizon])
            else:
                y.append(data[i])
        return np.array(X), np.array(y)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LSTMForecaster':
        """Train the LSTM forecaster
        
        Args:
            X: Training data
            y: Target values (optional, will be created from X if not provided)
            
        Returns:
            Self
        """
        logger.info("Training LSTM Forecaster...")
        
        # Prepare data
        if len(X.shape) == 2:
            # Reshape to 3D if needed
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        # Scale data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(
            X_scaled, 
            self.config.lookback_steps,
            self.config.forecast_horizon
        )
        
        # Build model
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/lstm_forecaster_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_seq, y_seq,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store training history
        self.training_history['loss'] = self.history.history['loss']
        self.training_history['val_loss'] = self.history.history['val_loss']
        
        # Calculate prediction intervals if enabled
        if self.config.use_prediction_interval:
            self._calculate_prediction_intervals(X_seq, y_seq)
            
        logger.info("LSTM Forecaster training completed")
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Prepare data
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        # Scale data
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Create sequences
        X_seq, _ = self.create_sequences(
            X_scaled,
            self.config.lookback_steps,
            self.config.forecast_horizon
        )
        
        if X_seq.shape[0] == 0:
            # Not enough data for sequence creation
            return np.array([])
            
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Inverse transform
        if self.config.multi_step:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            predictions = self.scaler.inverse_transform(predictions)
            predictions = predictions.reshape(-1, self.config.forecast_horizon, predictions.shape[-1])
        else:
            predictions = self.scaler.inverse_transform(predictions)
            
        return predictions
        
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies based on forecasting errors
        
        Args:
            X: Input data
            
        Returns:
            Binary array indicating anomalies (1 for anomaly, 0 for normal)
        """
        predictions = self.predict(X)
        
        if predictions.shape[0] == 0:
            return np.zeros(X.shape[0])
            
        # Calculate forecast errors
        actual_values = X[self.config.lookback_steps:]
        if self.config.multi_step:
            actual_values = actual_values[:predictions.shape[0]]
            errors = np.mean(np.abs(actual_values[:, :self.config.forecast_horizon] - predictions), axis=(1, 2))
        else:
            actual_values = actual_values[:predictions.shape[0]]
            errors = np.abs(actual_values - predictions).mean(axis=1)
            
        # Determine threshold
        if self.config.threshold_method == 'dynamic':
            threshold = np.mean(errors) + self.config.anomaly_threshold * np.std(errors)
        elif self.config.threshold_method == 'static':
            threshold = self.config.anomaly_threshold
        elif self.config.threshold_method == 'adaptive':
            # Use rolling statistics
            window_size = min(100, len(errors) // 10)
            rolling_mean = pd.Series(errors).rolling(window_size, min_periods=1).mean()
            rolling_std = pd.Series(errors).rolling(window_size, min_periods=1).std()
            threshold = rolling_mean + self.config.anomaly_threshold * rolling_std
        else:
            threshold = np.percentile(errors, 95)
            
        # Detect anomalies
        anomalies = np.zeros(X.shape[0])
        if isinstance(threshold, (int, float)):
            anomalies[self.config.lookback_steps:self.config.lookback_steps + len(errors)] = (errors > threshold).astype(int)
        else:
            anomalies[self.config.lookback_steps:self.config.lookback_steps + len(errors)] = (errors > threshold.values).astype(int)
            
        return anomalies
        
    def _calculate_prediction_intervals(self, X_train: np.ndarray, y_train: np.ndarray):
        """Calculate prediction intervals for uncertainty quantification"""
        # Make predictions on training data
        train_predictions = self.model.predict(X_train, verbose=0)
        
        # Calculate residuals
        residuals = y_train - train_predictions
        
        # Calculate standard deviation of residuals
        if self.config.multi_step:
            std_residuals = np.std(residuals, axis=0)
        else:
            std_residuals = np.std(residuals, axis=0)
            
        # Store for later use
        self.prediction_intervals = {
            'std_residuals': std_residuals,
            'confidence_level': self.config.confidence_level
        }
        
    def get_prediction_with_intervals(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions with confidence intervals
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with predictions and confidence bounds
        """
        predictions = self.predict(X)
        
        if not self.config.use_prediction_interval or 'std_residuals' not in self.prediction_intervals:
            return {
                'predictions': predictions,
                'lower_bound': None,
                'upper_bound': None
            }
            
        # Calculate confidence intervals
        z_score = 1.96 if self.config.confidence_level == 0.95 else 2.576  # For 99% confidence
        std_residuals = self.prediction_intervals['std_residuals']
        
        lower_bound = predictions - z_score * std_residuals
        upper_bound = predictions + z_score * std_residuals
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
    def rolling_forecast(self, X: np.ndarray, steps_ahead: int) -> np.ndarray:
        """Perform rolling forecast for multiple steps ahead
        
        Args:
            X: Historical data
            steps_ahead: Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        forecasts = []
        current_batch = X.copy()
        
        for _ in range(steps_ahead):
            # Get next prediction
            next_pred = self.predict(current_batch[-self.config.lookback_steps:])
            
            if next_pred.shape[0] == 0:
                break
                
            if self.config.multi_step:
                forecasts.append(next_pred[0, 0])
                # Update batch with prediction
                current_batch = np.vstack([current_batch, next_pred[0, 0]])
            else:
                forecasts.append(next_pred[-1])
                # Update batch with prediction
                current_batch = np.vstack([current_batch, next_pred[-1]])
                
        return np.array(forecasts)
        
    def save_model(self, filepath: str):
        """Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save scaler
        with open(f"{filepath}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save config
        with open(f"{filepath}_config.json", 'w') as f:
            config_dict = {
                'lstm_units': self.config.lstm_units,
                'dense_units': self.config.dense_units,
                'dropout_rate': self.config.dropout_rate,
                'lookback_steps': self.config.lookback_steps,
                'forecast_horizon': self.config.forecast_horizon,
                'model_type': self.config.model_type,
                'scaler_type': self.config.scaler_type,
                'threshold_method': self.config.threshold_method,
                'anomaly_threshold': self.config.anomaly_threshold
            }
            json.dump(config_dict, f, indent=2)
            
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load a saved model
        
        Args:
            filepath: Path to load the model from
        """
        # Load model
        self.model = keras.models.load_model(
            f"{filepath}_model.h5",
            custom_objects={'CustomAttention': CustomAttention}
        )
        
        # Load scaler
        with open(f"{filepath}_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load config
        with open(f"{filepath}_config.json", 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self.config, key, value)
                
        logger.info(f"Model loaded from {filepath}")
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (approximation for LSTM)
        
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # For LSTM, we can use gradient-based importance
        # This is a simplified version
        importance_scores = {}
        
        # Get the weights of the first LSTM layer
        first_layer_weights = self.model.layers[0].get_weights()[0]
        
        # Calculate importance as sum of absolute weights
        feature_importance = np.abs(first_layer_weights).mean(axis=0)
        
        for i, importance in enumerate(feature_importance):
            importance_scores[f'feature_{i}'] = float(importance)
            
        return importance_scores
        
    def get_model_summary(self) -> str:
        """Get model architecture summary
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not yet built"
            
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)