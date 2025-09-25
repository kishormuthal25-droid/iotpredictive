"""
Transformer-based Time Series Forecaster for IoT Anomaly Detection
This module implements a Transformer architecture for multi-step ahead forecasting
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Input, Embedding
)
import logging
from typing import Dict, List, Tuple, Optional, Union
import joblib
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(layers.Layer):
    """
    Positional encoding layer for Transformer
    Adds positional information to the input embeddings
    """
    
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config
    
    def positional_encoding(self, length, depth):
        """
        Create positional encoding matrix
        """
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1
        )
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        length = tf.shape(inputs)[1]
        # Add positional encoding
        return inputs + self.pos_encoding[tf.newaxis, :length, :]


class TransformerBlock(layers.Layer):
    """
    Single Transformer block with multi-head attention and feed-forward network
    """
    
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dropout(dropout_rate),
            Dense(d_model),
            Dropout(dropout_rate)
        ])
        
        # Layer normalization
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def call(self, inputs, training=False, mask=None):
        """Forward pass through transformer block"""
        # Multi-head attention
        attn_output = self.attention(
            inputs, inputs, 
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        
        return output


class TransformerForecaster:
    """
    Transformer-based time series forecaster for IoT sensor data
    """
    
    def __init__(self, config):
        """
        Initialize Transformer Forecaster
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self.model = None
        self.history = None
        self.scaler = None
        
        # Extract transformer config
        self.transformer_config = config.forecasting.get('transformer', {})
        
        # Model architecture parameters
        self.d_model = self.transformer_config.get('architecture', {}).get('d_model', 128)
        self.n_heads = self.transformer_config.get('architecture', {}).get('n_heads', 8)
        self.n_layers = self.transformer_config.get('architecture', {}).get('n_layers', 4)
        self.d_ff = self.transformer_config.get('architecture', {}).get('d_ff', 512)
        self.dropout = self.transformer_config.get('architecture', {}).get('dropout', 0.1)
        self.max_seq_length = self.transformer_config.get('architecture', {}).get('max_seq_length', 200)
        
        # Training parameters
        self.epochs = self.transformer_config.get('training', {}).get('epochs', 100)
        self.batch_size = self.transformer_config.get('training', {}).get('batch_size', 32)
        self.learning_rate = self.transformer_config.get('training', {}).get('learning_rate', 0.001)
        self.warmup_steps = self.transformer_config.get('training', {}).get('warmup_steps', 1000)
        
        # Forecasting parameters
        self.forecast_horizon = config.forecasting.get('general', {}).get('forecast_horizon', 24)
        
        logger.info(f"TransformerForecaster initialized with d_model={self.d_model}, "
                   f"n_heads={self.n_heads}, n_layers={self.n_layers}")
    
    def build_model(self, input_shape: Tuple[int, int], output_shape: int) -> models.Model:
        """
        Build Transformer model architecture
        
        Args:
            input_shape: Shape of input (sequence_length, n_features)
            output_shape: Number of output features to predict
        
        Returns:
            Compiled Keras model
        """
        sequence_length, n_features = input_shape
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Linear projection to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = PositionalEncoding(sequence_length, self.d_model)
        x = pos_encoding(x)
        
        # Initial dropout
        x = Dropout(self.dropout)(x)
        
        # Stack of Transformer blocks
        for i in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.n_heads,
                ff_dim=self.d_ff,
                dropout_rate=self.dropout,
                name=f'transformer_block_{i}'
            )(x)
        
        # Global pooling or take last timestep
        # Option 1: Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Option 2: Take last timestep (uncomment if preferred)
        # x = x[:, -1, :]
        
        # Final dense layers for forecasting
        x = Dense(256, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        
        # Output layer - predict next `forecast_horizon` timesteps
        outputs = Dense(output_shape, activation='linear')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Custom learning rate schedule
        lr_schedule = self._create_learning_rate_schedule()
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        logger.info(f"Model built successfully. Total parameters: {model.count_params():,}")
        
        return model
    
    def _create_learning_rate_schedule(self):
        """
        Create learning rate schedule with warmup
        """
        def schedule(step):
            """Learning rate schedule function"""
            arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
            arg2 = step * (self.warmup_steps ** -1.5)
            
            return tf.math.rsqrt(tf.cast(self.d_model, tf.float32)) * \
                   tf.math.minimum(arg1, arg2)
        
        return optimizers.schedules.LearningRateSchedule(schedule) \
               if self.warmup_steps > 0 else self.learning_rate
    
    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    validation_split: float = 0.2) -> Tuple:
        """
        Prepare data for training or prediction
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            y: Target values of shape (n_samples, forecast_horizon, n_features)
            validation_split: Fraction of data to use for validation
        
        Returns:
            Prepared data tuple
        """
        # Normalize data if scaler doesn't exist
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
            # Reshape for scaling
            n_samples, seq_len, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(n_samples, seq_len, n_features)
            
            if y is not None:
                y_reshaped = y.reshape(-1, y.shape[-1])
                y_scaled = self.scaler.transform(y_reshaped)
                y = y_scaled.reshape(y.shape)
        else:
            # Use existing scaler
            n_samples, seq_len, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped)
            X = X_scaled.reshape(n_samples, seq_len, n_features)
            
            if y is not None:
                y_reshaped = y.reshape(-1, y.shape[-1])
                y_scaled = self.scaler.transform(y_reshaped)
                y = y_scaled.reshape(y.shape)
        
        if y is not None and validation_split > 0:
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            return (X_train, y_train), (X_val, y_val)
        
        return X, y
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int,
                        forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting
        
        Args:
            data: Time series data of shape (n_timesteps, n_features)
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
        
        Returns:
            X: Input sequences
            y: Target sequences
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             validation_split: float = 0.2) -> Dict:
        """
        Train the Transformer model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            validation_split: Validation split if X_val not provided
        
        Returns:
            Training history dictionary
        """
        logger.info("Starting Transformer training...")
        
        # If y_train is None, create autoregressive targets
        if y_train is None:
            logger.info("Creating autoregressive targets...")
            # For autoregressive: predict next forecast_horizon steps
            y_train = np.zeros((len(X_train), self.forecast_horizon * X_train.shape[-1]))
            for i in range(len(X_train) - self.forecast_horizon):
                y_train[i] = X_train[i + 1:i + 1 + self.forecast_horizon].flatten()
        
        # Prepare data
        if X_val is None:
            (X_train, y_train), (X_val, y_val) = self.prepare_data(
                X_train, y_train, validation_split
            )
        else:
            X_train, y_train = self.prepare_data(X_train, y_train, validation_split=0)
            X_val, y_val = self.prepare_data(X_val, y_val, validation_split=0)
        
        # Build model if not exists
        if self.model is None:
            input_shape = X_train.shape[1:]  # (sequence_length, n_features)
            output_shape = y_train.shape[1]  # forecast_horizon * n_features
            self.build_model(input_shape, output_shape)
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='transformer_forecaster_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Add TensorBoard if in config
        if self.config.system.get('debug', False):
            callback_list.append(
                callbacks.TensorBoard(
                    log_dir=f'./logs/transformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    histogram_freq=1
                )
            )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Training completed successfully")
        
        return self.history.history
    
    def predict(self, X: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
            return_sequences: If True, reshape output to (n_samples, forecast_horizon, n_features)
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        X_scaled, _ = self.prepare_data(X, validation_split=0)
        
        # Make predictions
        predictions = self.model.predict(X_scaled, batch_size=self.batch_size)
        
        # Inverse transform predictions
        if self.scaler is not None:
            predictions = self.scaler.inverse_transform(
                predictions.reshape(-1, predictions.shape[-1] // self.forecast_horizon)
            ).reshape(predictions.shape)
        
        # Reshape if needed
        if return_sequences:
            n_features = X.shape[-1]
            predictions = predictions.reshape(-1, self.forecast_horizon, n_features)
        
        return predictions
    
    def forecast(self, X: np.ndarray, n_steps: Optional[int] = None) -> np.ndarray:
        """
        Forecast future values
        
        Args:
            X: Historical data of shape (sequence_length, n_features)
            n_steps: Number of steps to forecast (default: forecast_horizon)
        
        Returns:
            Forecasted values
        """
        if n_steps is None:
            n_steps = self.forecast_horizon
        
        # Ensure X has batch dimension
        if len(X.shape) == 2:
            X = X[np.newaxis, ...]
        
        forecasts = []
        current_sequence = X.copy()
        
        # Multi-step forecasting
        steps_predicted = 0
        while steps_predicted < n_steps:
            # Predict next horizon
            next_pred = self.predict(current_sequence, return_sequences=True)
            
            # Determine how many steps to use
            steps_to_use = min(self.forecast_horizon, n_steps - steps_predicted)
            forecasts.append(next_pred[0, :steps_to_use, :])
            
            # Update sequence for next prediction
            current_sequence = np.concatenate([
                current_sequence[:, steps_to_use:, :],
                next_pred[:, :steps_to_use, :]
            ], axis=1)
            
            steps_predicted += steps_to_use
        
        return np.concatenate(forecasts, axis=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test targets
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare data
        X_test_scaled, y_test_scaled = self.prepare_data(X_test, y_test, validation_split=0)
        
        # Evaluate
        metrics = self.model.evaluate(X_test_scaled, y_test_scaled, 
                                     batch_size=self.batch_size,
                                     verbose=0)
        
        # Create metrics dictionary
        metric_names = ['loss'] + [m.name for m in self.model.metrics]
        metrics_dict = dict(zip(metric_names, metrics))
        
        # Additional custom metrics
        predictions = self.predict(X_test)
        
        # Calculate additional metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_test_flat = y_test.reshape(-1)
        predictions_flat = predictions.reshape(-1)
        
        metrics_dict['rmse'] = np.sqrt(mean_squared_error(y_test_flat, predictions_flat))
        metrics_dict['mae'] = mean_absolute_error(y_test_flat, predictions_flat)
        metrics_dict['r2'] = r2_score(y_test_flat, predictions_flat)
        
        logger.info(f"Evaluation metrics: {metrics_dict}")
        
        return metrics_dict
    
    def save(self, filepath: str):
        """
        Save model and scaler
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save model
        model_path = filepath.replace('.h5', '_model.h5')
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save configuration
        config_path = filepath.replace('.h5', '_config.json')
        config_dict = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'forecast_horizon': self.forecast_horizon,
            'max_seq_length': self.max_seq_length
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def load(self, filepath: str):
        """
        Load saved model and scaler
        
        Args:
            filepath: Path to load the model from
        """
        # Load model
        model_path = filepath.replace('.h5', '_model.h5')
        if os.path.exists(model_path):
            # Register custom layers
            custom_objects = {
                'PositionalEncoding': PositionalEncoding,
                'TransformerBlock': TransformerBlock
            }
            self.model = models.load_model(model_path, custom_objects=custom_objects)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load scaler
        scaler_path = filepath.replace('.h5', '_scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load configuration
        config_path = filepath.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(self, key, value)
            logger.info(f"Configuration loaded from {config_path}")
    
    def get_attention_weights(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Extract attention weights for visualization
        
        Args:
            X: Input sequences
        
        Returns:
            List of attention weight matrices
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        attention_weights = []
        
        # Get intermediate layer outputs
        for layer in self.model.layers:
            if isinstance(layer, TransformerBlock):
                # Create intermediate model
                intermediate_model = models.Model(
                    inputs=self.model.input,
                    outputs=layer.attention.output
                )
                
                # Get attention weights
                weights = intermediate_model.predict(X)
                attention_weights.append(weights)
        
        return attention_weights
    
    def summary(self):
        """Print model summary"""
        if self.model is not None:
            self.model.summary()
        else:
            logger.warning("No model built yet. Call build_model() or train() first.")


# Utility functions for standalone usage
def create_transformer_forecaster(config: Dict) -> TransformerForecaster:
    """
    Factory function to create TransformerForecaster
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TransformerForecaster instance
    """
    # Convert dict to object if needed
    class ConfigObject:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, ConfigObject(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    if isinstance(config, dict):
        config = ConfigObject(config)
    
    return TransformerForecaster(config)


def demo_transformer_forecaster():
    """
    Demo function showing how to use TransformerForecaster
    """
    # Sample configuration
    config = {
        'system': {
            'debug': True
        },
        'forecasting': {
            'general': {
                'forecast_horizon': 24
            },
            'transformer': {
                'enabled': True,
                'architecture': {
                    'd_model': 128,
                    'n_heads': 8,
                    'n_layers': 4,
                    'd_ff': 512,
                    'dropout': 0.1,
                    'max_seq_length': 200
                },
                'training': {
                    'epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'warmup_steps': 1000
                }
            }
        }
    }
    
    # Create forecaster
    forecaster = create_transformer_forecaster(config)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    sequence_length = 100
    n_features = 10
    
    # Create synthetic time series data
    t = np.linspace(0, 100, n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        data[:, i] = (
            np.sin(t * (i + 1) * 0.1) + 
            np.cos(t * (i + 1) * 0.05) + 
            np.random.normal(0, 0.1, n_samples)
        )
    
    # Create sequences
    X, y = forecaster.create_sequences(data, sequence_length, 
                                       forecaster.forecast_horizon)
    
    print(f"Created {len(X)} sequences")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    # Build and display model
    forecaster.build_model(X.shape[1:], y.shape[1] * y.shape[2])
    forecaster.summary()
    
    # Train model (with reduced epochs for demo)
    forecaster.epochs = 5
    history = forecaster.train(X, y.reshape(len(y), -1))
    
    # Make predictions
    test_sequence = X[0:1]  # Take first sequence
    forecast = forecaster.forecast(test_sequence[0], n_steps=48)
    
    print(f"\nForecast shape: {forecast.shape}")
    print(f"Forecast for next 48 steps completed")
    
    # Evaluate
    metrics = forecaster.evaluate(X[:100], y[:100].reshape(100, -1))
    print(f"\nEvaluation metrics: {metrics}")
    
    # Save model
    forecaster.save('transformer_forecaster_demo.h5')
    print("\nModel saved successfully")


if __name__ == "__main__":
    # Run demo when script is executed directly
    print("Running TransformerForecaster demo...")
    demo_transformer_forecaster()
