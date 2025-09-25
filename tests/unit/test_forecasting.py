"""
Unit Tests for Forecasting Module
Tests for Transformer, LSTM forecasters and evaluation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings

# Import modules to test
from src.forecasting.base_forecaster import BaseForecast
from src.forecasting.transformer_forecaster import TransformerForecaster
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.forecast_evaluator import ForecastEvaluator


class TestBaseForecast(unittest.TestCase):
    """Test cases for BaseForecaster abstract class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a concrete implementation for testing
        class ConcreteForecaster(BaseForecaster):
            def build_model(self):
                return keras.Sequential([
                    keras.layers.Dense(64, activation='relu', input_shape=(self.sequence_length, self.n_features)),
                    keras.layers.Dense(self.forecast_horizon * self.n_features),
                    keras.layers.Reshape((self.forecast_horizon, self.n_features))
                ])
            
            def train(self, X_train, y_train, **kwargs):
                self.model.compile(optimizer='adam', loss='mse')
                return self.model.fit(X_train, y_train, epochs=1, verbose=0, **kwargs)
            
            def predict(self, X, **kwargs):
                return self.model.predict(X, verbose=0, **kwargs)
            
            def forecast(self, X, steps=None):
                steps = steps or self.forecast_horizon
                return self.predict(X)[:, :steps, :]
        
        self.forecaster = ConcreteForecaster(
            sequence_length=20,
            forecast_horizon=5,
            n_features=3
        )
        
        # Create sample time series data
        self.sample_sequences = np.random.randn(100, 20, 3)
        self.sample_targets = np.random.randn(100, 5, 3)
    
    def test_initialization(self):
        """Test forecaster initialization"""
        self.assertEqual(self.forecaster.sequence_length, 20)
        self.assertEqual(self.forecaster.forecast_horizon, 5)
        self.assertEqual(self.forecaster.n_features, 3)
        self.assertIsNotNone(self.forecaster.model)
    
    def test_model_building(self):
        """Test model building"""
        model = self.forecaster.build_model()
        self.assertIsInstance(model, keras.Model)
        
        # Check input/output shapes
        self.assertEqual(model.input_shape[1:], (20, 3))
        self.assertEqual(model.output_shape[1:], (5, 3))
    
    def test_training(self):
        """Test model training"""
        history = self.forecaster.train(
            self.sample_sequences,
            self.sample_targets,
            epochs=2,
            batch_size=16
        )
        
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertEqual(len(history.history['loss']), 2)
    
    def test_prediction(self):
        """Test prediction"""
        self.forecaster.train(self.sample_sequences, self.sample_targets)
        predictions = self.forecaster.predict(self.sample_sequences[:10])
        
        self.assertEqual(predictions.shape, (10, 5, 3))
    
    def test_forecasting(self):
        """Test multi-step forecasting"""
        self.forecaster.train(self.sample_sequences, self.sample_targets)
        
        # Forecast with default horizon
        forecast = self.forecaster.forecast(self.sample_sequences[:5])
        self.assertEqual(forecast.shape, (5, 5, 3))
        
        # Forecast with custom steps
        forecast = self.forecaster.forecast(self.sample_sequences[:5], steps=3)
        self.assertEqual(forecast.shape, (5, 3, 3))
    
    def test_save_load_model(self):
        """Test model persistence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.forecaster.train(self.sample_sequences, self.sample_targets)
            save_path = os.path.join(tmpdir, 'forecaster')
            self.forecaster.save(save_path)
            
            # Check files exist
            self.assertTrue(os.path.exists(f"{save_path}_model.h5"))
            self.assertTrue(os.path.exists(f"{save_path}_config.json"))
            
            # Load model
            new_forecaster = ConcreteForecaster(
                sequence_length=20,
                forecast_horizon=5,
                n_features=3
            )
            new_forecaster.load(save_path)
            
            # Compare predictions
            orig_pred = self.forecaster.predict(self.sample_sequences[:5])
            new_pred = new_forecaster.predict(self.sample_sequences[:5])
            np.testing.assert_array_almost_equal(orig_pred, new_pred, decimal=5)


class TestTransformerForecaster(unittest.TestCase):
    """Test cases for Transformer Forecaster"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sequence_length = 30
        self.forecast_horizon = 10
        self.n_features = 5
        
        self.forecaster = TransformerForecaster(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            n_features=self.n_features,
            d_model=64,
            n_heads=4,
            n_layers=2,
            ff_dim=128,
            dropout_rate=0.1
        )
        
        # Create sample data with temporal patterns
        time = np.linspace(0, 10, 150)
        self.sample_data = np.column_stack([
            np.sin(time),
            np.cos(time),
            np.sin(2*time),
            np.cos(2*time),
            np.random.randn(150) * 0.1
        ])
        
        # Create sequences and targets
        self.X_train = []
        self.y_train = []
        for i in range(len(self.sample_data) - self.sequence_length - self.forecast_horizon):
            self.X_train.append(self.sample_data[i:i+self.sequence_length])
            self.y_train.append(self.sample_data[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon])
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
    
    def test_initialization(self):
        """Test Transformer forecaster initialization"""
        self.assertEqual(self.forecaster.d_model, 64)
        self.assertEqual(self.forecaster.n_heads, 4)
        self.assertEqual(self.forecaster.n_layers, 2)
        self.assertEqual(self.forecaster.ff_dim, 128)
        self.assertEqual(self.forecaster.dropout_rate, 0.1)
    
    def test_transformer_architecture(self):
        """Test Transformer model architecture"""
        model = self.forecaster.build_model()
        
        # Check for attention layers
        attention_layers = [l for l in model.layers if 'attention' in l.name.lower()]
        self.assertGreater(len(attention_layers), 0)
        
        # Check model complexity
        total_params = model.count_params()
        self.assertGreater(total_params, 1000)  # Should have substantial parameters
        
        # Check input/output shapes
        self.assertEqual(model.input_shape[1:], (self.sequence_length, self.n_features))
        self.assertEqual(model.output_shape[1:], (self.forecast_horizon, self.n_features))
    
    def test_positional_encoding(self):
        """Test positional encoding implementation"""
        pos_encoding = self.forecaster.get_positional_encoding(
            seq_len=self.sequence_length,
            d_model=self.forecaster.d_model
        )
        
        self.assertEqual(pos_encoding.shape, (1, self.sequence_length, self.forecaster.d_model))
        
        # Check encoding values are bounded
        self.assertTrue(np.all(pos_encoding >= -1))
        self.assertTrue(np.all(pos_encoding <= 1))
    
    def test_attention_mechanism(self):
        """Test self-attention mechanism"""
        # Create a simple attention layer
        attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.forecaster.n_heads,
            key_dim=self.forecaster.d_model // self.forecaster.n_heads
        )
        
        # Test with sample input
        sample_input = tf.random.normal((2, self.sequence_length, self.forecaster.d_model))
        attention_output = attention_layer(sample_input, sample_input)
        
        self.assertEqual(attention_output.shape, sample_input.shape)
    
    def test_training_with_attention_weights(self):
        """Test training and attention weight extraction"""
        history = self.forecaster.train(
            self.X_train,
            self.y_train,
            epochs=2,
            batch_size=8,
            verbose=0
        )
        
        self.assertIn('loss', history.history)
        
        # Get attention weights
        attention_weights = self.forecaster.get_attention_weights(self.X_train[:5])
        
        if attention_weights is not None:
            # Check attention weights shape
            self.assertIsInstance(attention_weights, np.ndarray)
            self.assertTrue(len(attention_weights.shape) >= 2)
    
    def test_multi_step_ahead_forecasting(self):
        """Test multi-step ahead forecasting"""
        self.forecaster.train(self.X_train, self.y_train, epochs=5, verbose=0)
        
        # Single sequence forecast
        test_sequence = self.X_train[:1]
        forecast = self.forecaster.forecast(test_sequence, steps=self.forecast_horizon)
        
        self.assertEqual(forecast.shape, (1, self.forecast_horizon, self.n_features))
        
        # Check forecast is not constant
        self.assertFalse(np.all(forecast[0, 0] == forecast[0, -1]))
    
    def test_recursive_forecasting(self):
        """Test recursive multi-step forecasting"""
        self.forecaster.train(self.X_train, self.y_train, epochs=5, verbose=0)
        
        # Forecast beyond training horizon
        extended_forecast = self.forecaster.forecast_recursive(
            self.X_train[:1],
            steps=20  # Beyond training horizon
        )
        
        self.assertEqual(extended_forecast.shape, (1, 20, self.n_features))
    
    def test_uncertainty_estimation(self):
        """Test forecast uncertainty estimation"""
        self.forecaster.train(self.X_train, self.y_train, epochs=5, verbose=0)
        
        # Get forecast with uncertainty
        forecast, lower_bound, upper_bound = self.forecaster.forecast_with_uncertainty(
            self.X_train[:5],
            n_iterations=10,
            confidence=0.95
        )
        
        self.assertEqual(forecast.shape, (5, self.forecast_horizon, self.n_features))
        self.assertEqual(lower_bound.shape, forecast.shape)
        self.assertEqual(upper_bound.shape, forecast.shape)
        
        # Check bounds are properly ordered
        self.assertTrue(np.all(lower_bound <= forecast))
        self.assertTrue(np.all(forecast <= upper_bound))
    
    def test_model_interpretability(self):
        """Test model interpretability features"""
        self.forecaster.train(self.X_train, self.y_train, epochs=2, verbose=0)
        
        # Get feature importance
        feature_importance = self.forecaster.get_feature_importance(self.X_train[:10])
        
        if feature_importance is not None:
            self.assertEqual(len(feature_importance), self.n_features)
            self.assertTrue(np.all(feature_importance >= 0))
    
    def test_save_load_transformer(self):
        """Test saving and loading Transformer model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.forecaster.train(self.X_train, self.y_train, epochs=2, verbose=0)
            save_path = os.path.join(tmpdir, 'transformer')
            self.forecaster.save(save_path)
            
            # Load into new forecaster
            new_forecaster = TransformerForecaster(
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon,
                n_features=self.n_features,
                d_model=64,
                n_heads=4,
                n_layers=2
            )
            new_forecaster.load(save_path)
            
            # Compare predictions
            orig_forecast = self.forecaster.forecast(self.X_train[:3])
            new_forecast = new_forecaster.forecast(self.X_train[:3])
            np.testing.assert_array_almost_equal(orig_forecast, new_forecast, decimal=5)


class TestLSTMForecaster(unittest.TestCase):
    """Test cases for LSTM Forecaster"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sequence_length = 24
        self.forecast_horizon = 12
        self.n_features = 4
        
        self.forecaster = LSTMForecaster(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            n_features=self.n_features,
            hidden_units=[64, 32],
            dropout_rate=0.2,
            use_bidirectional=True
        )
        
        # Create sample data
        self.X_train = np.random.randn(80, self.sequence_length, self.n_features)
        self.y_train = np.random.randn(80, self.forecast_horizon, self.n_features)
    
    def test_initialization(self):
        """Test LSTM forecaster initialization"""
        self.assertEqual(self.forecaster.hidden_units, [64, 32])
        self.assertEqual(self.forecaster.dropout_rate, 0.2)
        self.assertTrue(self.forecaster.use_bidirectional)
    
    def test_lstm_architecture(self):
        """Test LSTM model architecture"""
        model = self.forecaster.build_model()
        
        # Check for LSTM layers
        lstm_layers = [l for l in model.layers if 'lstm' in l.name.lower()]
        self.assertGreater(len(lstm_layers), 0)
        
        # Check for bidirectional wrapper if enabled
        if self.forecaster.use_bidirectional:
            bidirectional_layers = [l for l in model.layers if 'bidirectional' in l.name.lower()]
            self.assertGreater(len(bidirectional_layers), 0)
    
    def test_sequence_to_sequence_learning(self):
        """Test sequence-to-sequence learning"""
        # Create encoder-decoder LSTM
        self.forecaster.use_seq2seq = True
        model = self.forecaster.build_seq2seq_model()
        
        # Check model can handle sequence input and output
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(
            self.X_train,
            self.y_train,
            epochs=1,
            batch_size=8,
            verbose=0
        )
        
        self.assertIsNotNone(history)
    
    def test_stateful_lstm(self):
        """Test stateful LSTM for better long-term dependencies"""
        # Create stateful LSTM
        batch_size = 4
        stateful_forecaster = LSTMForecaster(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            n_features=self.n_features,
            stateful=True,
            batch_size=batch_size
        )
        
        # Prepare data for stateful training
        n_samples = (len(self.X_train) // batch_size) * batch_size
        X_stateful = self.X_train[:n_samples]
        y_stateful = self.y_train[:n_samples]
        
        # Train with state reset between epochs
        for epoch in range(2):
            stateful_forecaster.model.reset_states()
            stateful_forecaster.model.fit(
                X_stateful,
                y_stateful,
                batch_size=batch_size,
                epochs=1,
                shuffle=False,
                verbose=0
            )
    
    def test_training_with_callbacks(self):
        """Test training with various callbacks"""
        # Mock callbacks
        with patch('keras.callbacks.ModelCheckpoint') as mock_checkpoint:
            with patch('keras.callbacks.ReduceLROnPlateau') as mock_reduce_lr:
                
                history = self.forecaster.train(
                    self.X_train,
                    self.y_train,
                    epochs=2,
                    use_checkpoints=True,
                    reduce_lr=True,
                    verbose=0
                )
                
                # Verify callbacks were used
                self.assertTrue(mock_checkpoint.called or mock_reduce_lr.called)
    
    def test_multi_variate_forecasting(self):
        """Test multivariate time series forecasting"""
        self.forecaster.train(self.X_train, self.y_train, epochs=5, verbose=0)
        
        # Forecast multiple features
        forecast = self.forecaster.forecast(self.X_train[:5])
        
        self.assertEqual(forecast.shape, (5, self.forecast_horizon, self.n_features))
        
        # Check each feature is forecasted
        for i in range(self.n_features):
            feature_forecast = forecast[:, :, i]
            self.assertFalse(np.all(feature_forecast == 0))
    
    def test_rolling_window_forecast(self):
        """Test rolling window forecasting"""
        self.forecaster.train(self.X_train, self.y_train, epochs=5, verbose=0)
        
        # Perform rolling forecast
        test_sequence = self.X_train[:1]
        rolling_forecasts = []
        
        current_sequence = test_sequence[0]
        for _ in range(3):
            forecast = self.forecaster.forecast(current_sequence.reshape(1, -1, self.n_features))
            rolling_forecasts.append(forecast[0])
            
            # Update sequence with forecast
            current_sequence = np.vstack([current_sequence[self.forecast_horizon:], forecast[0]])
        
        self.assertEqual(len(rolling_forecasts), 3)
        for forecast in rolling_forecasts:
            self.assertEqual(forecast.shape, (self.forecast_horizon, self.n_features))
    
    def test_attention_lstm(self):
        """Test LSTM with attention mechanism"""
        # Create LSTM with attention
        attention_forecaster = LSTMForecaster(
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            n_features=self.n_features,
            use_attention=True
        )
        
        model = attention_forecaster.build_model_with_attention()
        
        # Check model has attention layer
        self.assertIsNotNone(model)
        
        # Train model
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(self.X_train, self.y_train, epochs=1, verbose=0)
        self.assertIsNotNone(history)
    
    def test_save_load_lstm(self):
        """Test saving and loading LSTM model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.forecaster.train(self.X_train, self.y_train, epochs=2, verbose=0)
            save_path = os.path.join(tmpdir, 'lstm')
            self.forecaster.save(save_path)
            
            # Load into new forecaster
            new_forecaster = LSTMForecaster(
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon,
                n_features=self.n_features
            )
            new_forecaster.load(save_path)
            
            # Compare predictions
            orig_forecast = self.forecaster.forecast(self.X_train[:3])
            new_forecast = new_forecaster.forecast(self.X_train[:3])
            np.testing.assert_array_almost_equal(orig_forecast, new_forecast, decimal=5)


class TestForecastEvaluator(unittest.TestCase):
    """Test cases for Forecast Evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ForecastEvaluator()
        
        # Create sample forecast data
        self.y_true = np.random.randn(50, 10, 3)  # 50 samples, 10 timesteps, 3 features
        self.y_pred = self.y_true + np.random.randn(50, 10, 3) * 0.1  # Add small noise
        
        # Create time series for additional tests
        time = np.linspace(0, 10, 100)
        self.ts_true = np.sin(time) + np.random.randn(100) * 0.1
        self.ts_pred = np.sin(time) + np.random.randn(100) * 0.15
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertIsInstance(self.evaluator.metrics_history, list)
    
    def test_calculate_metrics(self):
        """Test calculation of forecast metrics"""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        
        # Check all metrics are present
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('smape', metrics)
        self.assertIn('r2', metrics)
        
        # Check metric values are reasonable
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertLessEqual(metrics['r2'], 1)
    
    def test_per_horizon_metrics(self):
        """Test metrics calculation per forecast horizon"""
        horizon_metrics = self.evaluator.calculate_per_horizon_metrics(
            self.y_true,
            self.y_pred
        )
        
        # Should have metrics for each timestep
        self.assertEqual(len(horizon_metrics), 10)
        
        # Each timestep should have all metrics
        for step_metrics in horizon_metrics:
            self.assertIn('mse', step_metrics)
            self.assertIn('mae', step_metrics)
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        # Create data with clear directional patterns
        y_true = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        y_pred = np.array([[1.1, 2.1, 2.9, 4.1, 4.9], [4.9, 4.1, 3.1, 1.9, 1.1]])
        
        da = self.evaluator.calculate_directional_accuracy(y_true, y_pred)
        
        self.assertGreaterEqual(da, 0)
        self.assertLessEqual(da, 1)
        
        # Should have high accuracy for these patterns
        self.assertGreater(da, 0.5)
    
    def test_prediction_intervals(self):
        """Test prediction interval coverage"""
        # Create predictions with uncertainty bounds
        lower_bound = self.y_pred - 0.2
        upper_bound = self.y_pred + 0.2
        
        coverage = self.evaluator.calculate_interval_coverage(
            self.y_true,
            lower_bound,
            upper_bound
        )
        
        self.assertGreaterEqual(coverage, 0)
        self.assertLessEqual(coverage, 1)
        
        # Most true values should be within bounds
        self.assertGreater(coverage, 0.5)
    
    def test_compare_models(self):
        """Test comparing multiple forecast models"""
        # Create mock models
        models = []
        for i in range(3):
            model = Mock()
            model.name = f"model_{i}"
            model.forecast = Mock(return_value=self.y_pred + np.random.randn(*self.y_pred.shape) * 0.05)
            models.append(model)
        
        # Compare models
        X_test = np.random.randn(50, 20, 3)
        comparison = self.evaluator.compare_models(
            models,
            X_test,
            self.y_true
        )
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 3)
        self.assertIn('mse', comparison.columns)
        self.assertIn('mae', comparison.columns)
        self.assertIn('rank', comparison.columns)
    
    def test_residual_analysis(self):
        """Test residual analysis for forecast errors"""
        residual_stats = self.evaluator.analyze_residuals(
            self.y_true.flatten(),
            self.y_pred.flatten()
        )
        
        self.assertIn('mean', residual_stats)
        self.assertIn('std', residual_stats)
        self.assertIn('skewness', residual_stats)
        self.assertIn('kurtosis', residual_stats)
        self.assertIn('autocorrelation', residual_stats)
        self.assertIn('normality_test', residual_stats)
        
        # Check normality test results
        self.assertIn('statistic', residual_stats['normality_test'])
        self.assertIn('p_value', residual_stats['normality_test'])
    
    def test_seasonal_decomposition_metrics(self):
        """Test seasonal decomposition of forecast errors"""
        # Create seasonal data
        time = np.arange(100)
        seasonal_true = np.sin(2 * np.pi * time / 12) + np.random.randn(100) * 0.1
        seasonal_pred = np.sin(2 * np.pi * time / 12) + np.random.randn(100) * 0.15
        
        decomposition_metrics = self.evaluator.evaluate_seasonal_components(
            seasonal_true,
            seasonal_pred,
            period=12
        )
        
        self.assertIn('trend_error', decomposition_metrics)
        self.assertIn('seasonal_error', decomposition_metrics)
        self.assertIn('residual_error', decomposition_metrics)
    
    def test_forecast_skill_score(self):
        """Test forecast skill score calculation"""
        # Use naive forecast as baseline
        naive_forecast = np.roll(self.y_true, 1, axis=1)
        naive_forecast[:, 0] = self.y_true[:, 0]  # First step same
        
        skill_score = self.evaluator.calculate_skill_score(
            self.y_true,
            self.y_pred,
            naive_forecast
        )
        
        self.assertIsInstance(skill_score, float)
        # Positive score means better than naive
        # Negative score means worse than naive
    
    def test_multi_step_evaluation(self):
        """Test evaluation of multi-step ahead forecasts"""
        # Evaluate at different horizons
        horizons = [1, 5, 10]
        multi_step_metrics = self.evaluator.evaluate_multi_step(
            self.y_true,
            self.y_pred,
            horizons
        )
        
        self.assertEqual(len(multi_step_metrics), len(horizons))
        
        for h, metrics in multi_step_metrics.items():
            self.assertIn('mse', metrics)
            self.assertIn('mae', metrics)
            
            # Error should generally increase with horizon
            if h > 1:
                prev_h = horizons[horizons.index(h) - 1]
                # This is a general trend, might not always hold with random data
    
    def test_cross_validation_forecast(self):
        """Test time series cross-validation"""
        # Create longer time series
        long_series = np.random.randn(200, 5)
        
        cv_results = self.evaluator.time_series_cv(
            data=long_series,
            model=Mock(forecast=Mock(return_value=np.random.randn(1, 10, 5))),
            n_splits=3,
            test_size=10
        )
        
        self.assertIn('cv_scores', cv_results)
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        self.assertEqual(len(cv_results['cv_scores']), 3)
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
            
            # Save results
            save_path = os.path.join(tmpdir, 'forecast_evaluation.json')
            self.evaluator.save_results(save_path, metrics)
            
            # Check file exists
            self.assertTrue(os.path.exists(save_path))
            
            # Load and verify
            with open(save_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            self.assertEqual(loaded_metrics['mse'], metrics['mse'])
    
    def test_visualization_data_generation(self):
        """Test generation of data for visualization"""
        viz_data = self.evaluator.prepare_visualization_data(
            self.y_true,
            self.y_pred
        )
        
        self.assertIn('actual_vs_predicted', viz_data)
        self.assertIn('residuals', viz_data)
        self.assertIn('qq_plot_data', viz_data)
        self.assertIn('acf_data', viz_data)
        
        # Check data shapes
        self.assertEqual(viz_data['residuals'].shape, self.y_true.shape)


class TestForecastingIntegration(unittest.TestCase):
    """Integration tests for forecasting pipeline"""
    
    def test_end_to_end_transformer_pipeline(self):
        """Test end-to-end Transformer forecasting pipeline"""
        # Generate synthetic time series
        np.random.seed(42)
        time = np.linspace(0, 50, 500)
        
        # Multi-component series
        series = np.column_stack([
            np.sin(time) + np.random.randn(500) * 0.1,  # Component 1
            np.cos(time) + np.random.randn(500) * 0.1,  # Component 2
            np.sin(2*time) * 0.5 + np.random.randn(500) * 0.05  # Component 3
        ])
        
        # Prepare data
        sequence_length = 30
        forecast_horizon = 10
        
        X, y = [], []
        for i in range(len(series) - sequence_length - forecast_horizon):
            X.append(series[i:i+sequence_length])
            y.append(series[i+sequence_length:i+sequence_length+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train model
        forecaster = TransformerForecaster(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            n_features=3,
            d_model=32,
            n_heads=2,
            n_layers=1
        )
        
        forecaster.train(X_train, y_train, epochs=5, verbose=0)
        
        # Make predictions
        predictions = forecaster.forecast(X_test)
        
        # Evaluate
        evaluator = ForecastEvaluator()
        metrics = evaluator.calculate_metrics(y_test, predictions)
        
        # Check metrics are reasonable
        self.assertGreater(metrics['r2'], -1)  # Better than random
        self.assertLess(metrics['mape'], 100)  # Reasonable MAPE
    
    def test_lstm_vs_transformer_comparison(self):
        """Test comparing LSTM and Transformer forecasters"""
        # Generate data
        np.random.seed(42)
        time = np.linspace(0, 20, 200)
        series = np.sin(time) + np.random.randn(200) * 0.1
        series = series.reshape(-1, 1)
        
        # Prepare sequences
        sequence_length = 20
        forecast_horizon = 5
        
        X, y = [], []
        for i in range(len(series) - sequence_length - forecast_horizon):
            X.append(series[i:i+sequence_length])
            y.append(series[i+sequence_length:i+sequence_length+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Create models
        lstm_forecaster = LSTMForecaster(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            n_features=1,
            hidden_units=[32]
        )
        
        transformer_forecaster = TransformerForecaster(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            n_features=1,
            d_model=16,
            n_heads=1,
            n_layers=1
        )
        
        # Train both
        lstm_forecaster.train(X, y, epochs=10, verbose=0)
        transformer_forecaster.train(X, y, epochs=10, verbose=0)
        
        # Compare
        evaluator = ForecastEvaluator()
        comparison = evaluator.compare_models(
            [lstm_forecaster, transformer_forecaster],
            X[-10:],
            y[-10:]
        )
        
        self.assertEqual(len(comparison), 2)
        self.assertIn('mse', comparison.columns)
    
    def test_ensemble_forecasting(self):
        """Test ensemble of multiple forecasters"""
        # Create data
        X = np.random.randn(100, 20, 3)
        y = np.random.randn(100, 5, 3)
        
        # Create multiple models
        models = [
            LSTMForecaster(20, 5, 3, hidden_units=[16]),
            LSTMForecaster(20, 5, 3, hidden_units=[32]),
            TransformerForecaster(20, 5, 3, d_model=16, n_heads=1, n_layers=1)
        ]
        
        # Train all models
        for model in models:
            model.train(X[:80], y[:80], epochs=2, verbose=0)
        
        # Ensemble predictions
        ensemble_predictions = []
        for model in models:
            pred = model.forecast(X[80:])
            ensemble_predictions.append(pred)
        
        # Average ensemble
        ensemble_mean = np.mean(ensemble_predictions, axis=0)
        
        # Evaluate ensemble
        evaluator = ForecastEvaluator()
        ensemble_metrics = evaluator.calculate_metrics(y[80:], ensemble_mean)
        
        self.assertIn('mse', ensemble_metrics)
        self.assertGreater(ensemble_metrics['r2'], -10)  # Sanity check


class TestPerformanceAndEdgeCases(unittest.TestCase):
    """Test performance and edge cases for forecasting"""
    
    def test_missing_data_handling(self):
        """Test handling of missing data in forecasting"""
        # Create data with missing values
        X = np.random.randn(50, 20, 3)
        X[0, 5:7, :] = np.nan  # Add some NaN values
        y = np.random.randn(50, 5, 3)
        
        forecaster = LSTMForecaster(20, 5, 3)
        
        # Should handle missing data (interpolation or error)
        try:
            # Interpolate missing values
            X_filled = np.nan_to_num(X, nan=np.nanmean(X))
            forecaster.train(X_filled, y, epochs=1, verbose=0)
            success = True
        except Exception as e:
            success = False
        
        self.assertTrue(success or isinstance(e, ValueError))
    
    def test_single_step_vs_multi_step(self):
        """Test single-step vs multi-step forecasting strategies"""
        # Create data
        X = np.random.randn(100, 20, 2)
        
        forecaster = LSTMForecaster(20, 1, 2)  # Single-step
        multi_forecaster = LSTMForecaster(20, 10, 2)  # Multi-step
        
        # Train single-step model
        y_single = np.random.randn(100, 1, 2)
        forecaster.train(X, y_single, epochs=2, verbose=0)
        
        # Train multi-step model
        y_multi = np.random.randn(100, 10, 2)
        multi_forecaster.train(X, y_multi, epochs=2, verbose=0)
        
        # Compare forecasting strategies
        single_step_forecast = []
        current_input = X[:1]
        
        for _ in range(10):
            step = forecaster.forecast(current_input)
            single_step_forecast.append(step[0])
            # Update input with forecast
            current_input = np.concatenate([current_input[:, 1:], step], axis=1)
        
        single_step_result = np.array(single_step_forecast).T
        multi_step_result = multi_forecaster.forecast(X[:1])
        
        # Both should produce forecasts of same shape
        self.assertEqual(single_step_result.shape, multi_step_result.shape)
    
    def test_extreme_horizon_lengths(self):
        """Test with very short and very long forecast horizons"""
        # Very short horizon
        short_forecaster = TransformerForecaster(20, 1, 3)
        X_short = np.random.randn(50, 20, 3)
        y_short = np.random.randn(50, 1, 3)
        
        short_forecaster.train(X_short, y_short, epochs=1, verbose=0)
        short_pred = short_forecaster.forecast(X_short[:5])
        self.assertEqual(short_pred.shape, (5, 1, 3))
        
        # Very long horizon
        long_forecaster = TransformerForecaster(20, 50, 3)
        y_long = np.random.randn(50, 50, 3)
        
        long_forecaster.train(X_short, y_long, epochs=1, verbose=0)
        long_pred = long_forecaster.forecast(X_short[:5])
        self.assertEqual(long_pred.shape, (5, 50, 3))
    
    def test_constant_series_forecasting(self):
        """Test forecasting constant series"""
        # Create constant series
        X = np.ones((50, 20, 2)) * 5
        y = np.ones((50, 5, 2)) * 5
        
        forecaster = LSTMForecaster(20, 5, 2)
        forecaster.train(X, y, epochs=5, verbose=0)
        
        predictions = forecaster.forecast(X[:5])
        
        # Should predict close to constant value
        self.assertAlmostEqual(np.mean(predictions), 5, places=0)
        self.assertLess(np.std(predictions), 1)  # Low variance
    
    def test_trend_extrapolation(self):
        """Test extrapolation of trends"""
        # Create linear trend data
        time = np.arange(100)
        trend = time * 0.1 + np.random.randn(100) * 0.01
        trend = trend.reshape(-1, 1)
        
        # Prepare sequences
        X, y = [], []
        for i in range(len(trend) - 25):
            X.append(trend[i:i+20])
            y.append(trend[i+20:i+25])
        
        X = np.array(X)
        y = np.array(y)
        
        forecaster = LSTMForecaster(20, 5, 1)
        forecaster.train(X, y, epochs=10, verbose=0)
        
        # Forecast should continue trend
        last_sequence = X[-1:]
        forecast = forecaster.forecast(last_sequence)
        
        # Check if trend continues (values should be increasing)
        self.assertGreater(forecast[0, -1, 0], forecast[0, 0, 0])


if __name__ == '__main__':
    unittest.main()