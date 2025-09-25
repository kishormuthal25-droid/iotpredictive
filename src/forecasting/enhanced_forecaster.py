"""
Enhanced Forecasting Module with Advanced Uncertainty Quantification
Implements quantile regression, conformal prediction, and bootstrap methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from abc import ABC, abstractmethod

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import norm

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.forecasting.base_forecaster import BaseForecast, ForecastResult, ForecastMetrics
from src.forecasting.transformer_forecaster import TransformerForecaster
from src.forecasting.lstm_forecaster import LSTMForecaster
from config.settings import settings, get_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedForecastResult(ForecastResult):
    """Enhanced forecast result with additional uncertainty measures"""
    quantile_predictions: Optional[Dict[float, np.ndarray]] = None  # Multiple quantiles
    prediction_std: Optional[np.ndarray] = None  # Standard deviation
    coverage_probabilities: Optional[Dict[float, float]] = None  # Coverage for different levels
    epistemic_uncertainty: Optional[np.ndarray] = None  # Model uncertainty
    aleatoric_uncertainty: Optional[np.ndarray] = None  # Data uncertainty
    ensemble_predictions: Optional[List[np.ndarray]] = None  # Individual ensemble predictions

    @property
    def total_uncertainty(self) -> Optional[np.ndarray]:
        """Get total uncertainty (epistemic + aleatoric)"""
        if self.epistemic_uncertainty is not None and self.aleatoric_uncertainty is not None:
            return np.sqrt(self.epistemic_uncertainty**2 + self.aleatoric_uncertainty**2)
        return self.prediction_std


class QuantileLoss:
    """Custom quantile loss for neural networks"""

    def __init__(self, quantile: float):
        self.quantile = quantile

    def __call__(self, y_true, y_pred):
        """Quantile loss function"""
        error = y_true - y_pred
        return tf.keras.backend.mean(
            tf.keras.backend.maximum(
                self.quantile * error,
                (self.quantile - 1) * error
            )
        )


class QuantileForecaster:
    """Neural network-based quantile forecasting"""

    def __init__(self,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 architecture: Dict = None):
        """
        Initialize quantile forecaster

        Args:
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
            architecture: Neural network architecture parameters
        """
        self.quantiles = sorted(quantiles)
        self.architecture = architecture or {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu'
        }
        self.models = {}  # One model per quantile
        self.scaler = StandardScaler()
        self.is_fitted = False

    def build_quantile_model(self, input_shape: Tuple, quantile: float) -> keras.Model:
        """Build neural network for specific quantile"""
        inputs = Input(shape=input_shape)
        x = inputs

        # Hidden layers
        for units in self.architecture['hidden_layers']:
            x = Dense(units, activation=self.architecture['activation'])(x)
            x = Dropout(self.architecture['dropout_rate'])(x)

        # Output layer
        outputs = Dense(1, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile with quantile loss
        model.compile(
            optimizer='adam',
            loss=QuantileLoss(quantile),
            metrics=['mae']
        )

        return model

    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_split: float = 0.2,
            epochs: int = 100) -> 'QuantileForecaster':
        """Train quantile models"""

        # Prepare data
        X_scaled = self.scaler.fit_transform(X)

        # Train model for each quantile
        for quantile in self.quantiles:
            logger.info(f"Training quantile model for {quantile}")

            model = self.build_quantile_model(X_scaled.shape[1:], quantile)

            # Early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            # Train model
            model.fit(
                X_scaled, y,
                validation_split=validation_split,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=0
            )

            self.models[quantile] = model

        self.is_fitted = True
        return self

    def predict_quantiles(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """Predict multiple quantiles"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X_scaled, verbose=0).flatten()

        return predictions


class ConformalPredictor:
    """Conformal prediction for distribution-free prediction intervals"""

    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor

        Args:
            alpha: Miscoverage level (1-alpha is coverage level)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile_threshold = None

    def calibrate(self,
                  residuals: np.ndarray,
                  prediction_errors: Optional[np.ndarray] = None):
        """
        Calibrate conformal predictor using residuals

        Args:
            residuals: Calibration residuals
            prediction_errors: Optional prediction error estimates
        """
        if prediction_errors is not None:
            # Normalized residuals
            self.calibration_scores = np.abs(residuals) / (prediction_errors + 1e-8)
        else:
            # Absolute residuals
            self.calibration_scores = np.abs(residuals)

        # Calculate quantile threshold
        n = len(self.calibration_scores)
        quantile_level = (1 - self.alpha) * (1 + 1/n)
        self.quantile_threshold = np.quantile(self.calibration_scores, quantile_level)

    def predict_interval(self,
                        point_predictions: np.ndarray,
                        prediction_errors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate conformal prediction intervals

        Args:
            point_predictions: Point forecasts
            prediction_errors: Optional prediction error estimates

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self.quantile_threshold is None:
            raise ValueError("Must calibrate first")

        if prediction_errors is not None:
            # Adaptive intervals
            interval_width = self.quantile_threshold * prediction_errors
        else:
            # Fixed-width intervals
            interval_width = self.quantile_threshold

        lower_bound = point_predictions - interval_width
        upper_bound = point_predictions + interval_width

        return lower_bound, upper_bound


class EnsembleForecaster:
    """Ensemble forecasting with uncertainty quantification"""

    def __init__(self,
                 base_forecaster_class,
                 n_models: int = 5,
                 bootstrap_ratio: float = 0.8,
                 **forecaster_kwargs):
        """
        Initialize ensemble forecaster

        Args:
            base_forecaster_class: Base forecasting class
            n_models: Number of models in ensemble
            bootstrap_ratio: Ratio of data to use for each model
            **forecaster_kwargs: Arguments for base forecaster
        """
        self.base_forecaster_class = base_forecaster_class
        self.n_models = n_models
        self.bootstrap_ratio = bootstrap_ratio
        self.forecaster_kwargs = forecaster_kwargs
        self.models = []
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'EnsembleForecaster':
        """Train ensemble of models"""

        n_samples = len(X)
        sample_size = int(n_samples * self.bootstrap_ratio)

        for i in range(self.n_models):
            logger.info(f"Training ensemble model {i+1}/{self.n_models}")

            # Bootstrap sampling
            indices = np.random.choice(n_samples, sample_size, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices] if y is not None else None

            # Create and train model
            model = self.base_forecaster_class(**self.forecaster_kwargs)
            model.fit(X_bootstrap, y_bootstrap)

            self.models.append(model)

        self.is_fitted = True
        return self

    def predict_ensemble(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate ensemble predictions with uncertainty

        Returns:
            Dictionary with predictions, std, and individual forecasts
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")

        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'individual': predictions,
            'quantiles': {
                0.1: np.percentile(predictions, 10, axis=0),
                0.25: np.percentile(predictions, 25, axis=0),
                0.5: np.percentile(predictions, 50, axis=0),
                0.75: np.percentile(predictions, 75, axis=0),
                0.9: np.percentile(predictions, 90, axis=0)
            }
        }


class EnhancedForecaster(BaseForecast):
    """
    Enhanced forecaster with advanced uncertainty quantification
    Combines multiple uncertainty estimation methods
    """

    def __init__(self,
                 base_model: str = "transformer",  # "transformer" or "lstm"
                 uncertainty_methods: List[str] = ["quantile", "ensemble", "conformal"],
                 confidence_levels: List[float] = [0.8, 0.9, 0.95],
                 ensemble_size: int = 5,
                 **kwargs):
        """
        Initialize enhanced forecaster

        Args:
            base_model: Base forecasting model type
            uncertainty_methods: List of uncertainty estimation methods
            confidence_levels: Confidence levels for prediction intervals
            ensemble_size: Number of models in ensemble
            **kwargs: Additional arguments for base forecaster
        """
        super().__init__(**kwargs)

        self.base_model = base_model
        self.uncertainty_methods = uncertainty_methods
        self.confidence_levels = confidence_levels
        self.ensemble_size = ensemble_size

        # Initialize components
        self.main_forecaster = None
        self.quantile_forecaster = None
        self.ensemble_forecaster = None
        self.conformal_predictor = None

        # Uncertainty estimates
        self.uncertainty_estimates = {}

    def _build_model(self):
        """Build the underlying forecasting models"""

        # Main forecaster
        if self.base_model == "transformer":
            self.main_forecaster = TransformerForecaster(get_config())
        elif self.base_model == "lstm":
            from src.forecasting.lstm_forecaster import LSTMForecasterConfig
            config = LSTMForecasterConfig()
            self.main_forecaster = LSTMForecaster(config)
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

        # Quantile forecaster
        if "quantile" in self.uncertainty_methods:
            quantiles = []
            for conf_level in self.confidence_levels:
                alpha = 1 - conf_level
                quantiles.extend([alpha/2, 1 - alpha/2])
            quantiles.extend([0.5])  # Always include median
            quantiles = sorted(list(set(quantiles)))

            self.quantile_forecaster = QuantileForecaster(quantiles=quantiles)

        # Ensemble forecaster
        if "ensemble" in self.uncertainty_methods:
            if self.base_model == "transformer":
                self.ensemble_forecaster = EnsembleForecaster(
                    TransformerForecaster,
                    n_models=self.ensemble_size,
                    config=get_config()
                )
            else:
                from src.forecasting.lstm_forecaster import LSTMForecasterConfig
                self.ensemble_forecaster = EnsembleForecaster(
                    LSTMForecaster,
                    n_models=self.ensemble_size,
                    config=LSTMForecasterConfig()
                )

        # Conformal predictor
        if "conformal" in self.uncertainty_methods:
            self.conformal_predictor = ConformalPredictor()

    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Internal fit implementation"""

        # Split data for conformal prediction calibration
        if "conformal" in self.uncertainty_methods:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
        else:
            X_train, y_train = X, y

        # Fit main forecaster
        logger.info("Training main forecaster...")
        self.main_forecaster.fit(X_train, y_train)

        # Fit quantile forecaster
        if self.quantile_forecaster is not None:
            logger.info("Training quantile forecaster...")

            # Prepare data for quantile regression
            if y_train is None:
                # Create sequences for autoregressive forecasting
                X_seq, y_seq = self._create_sequences(X_train)
            else:
                X_seq, y_seq = X_train, y_train

            if len(X_seq.shape) > 2:
                X_seq = X_seq.reshape(X_seq.shape[0], -1)  # Flatten for quantile regression

            self.quantile_forecaster.fit(X_seq, y_seq.flatten())

        # Fit ensemble forecaster
        if self.ensemble_forecaster is not None:
            logger.info("Training ensemble forecaster...")
            self.ensemble_forecaster.fit(X_train, y_train)

        # Calibrate conformal predictor
        if self.conformal_predictor is not None and "conformal" in self.uncertainty_methods:
            logger.info("Calibrating conformal predictor...")

            # Get calibration predictions
            cal_predictions = self.main_forecaster.predict(X_cal)
            cal_residuals = y_cal - cal_predictions

            self.conformal_predictor.calibrate(cal_residuals.flatten())

    def _predict(self, X: np.ndarray, return_confidence: bool = False) -> Union[np.ndarray, Tuple]:
        """Internal prediction implementation"""

        # Main predictions
        main_predictions = self.main_forecaster.predict(X)

        if not return_confidence:
            return main_predictions

        # Initialize uncertainty estimates
        uncertainty_estimates = {}

        # Quantile predictions
        if self.quantile_forecaster is not None:
            # Prepare data
            if len(X.shape) > 2:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X

            quantile_preds = self.quantile_forecaster.predict_quantiles(X_flat)
            uncertainty_estimates['quantiles'] = quantile_preds

        # Ensemble predictions
        if self.ensemble_forecaster is not None:
            ensemble_results = self.ensemble_forecaster.predict_ensemble(X)
            uncertainty_estimates['ensemble'] = ensemble_results

        # Conformal intervals
        if self.conformal_predictor is not None:
            for conf_level in self.confidence_levels:
                self.conformal_predictor.alpha = 1 - conf_level
                lower, upper = self.conformal_predictor.predict_interval(main_predictions.flatten())
                uncertainty_estimates[f'conformal_{conf_level}'] = (lower, upper)

        # Combine uncertainty estimates
        combined_lower, combined_upper = self._combine_uncertainty_estimates(
            main_predictions, uncertainty_estimates
        )

        return main_predictions, combined_lower, combined_upper

    def _combine_uncertainty_estimates(self,
                                     predictions: np.ndarray,
                                     uncertainty_estimates: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multiple uncertainty estimates"""

        lower_bounds = []
        upper_bounds = []

        # Use the most conservative bounds from all methods
        for conf_level in self.confidence_levels:
            bounds_for_level = []

            # Quantile bounds
            if 'quantiles' in uncertainty_estimates:
                alpha = 1 - conf_level
                lower_quantile = alpha / 2
                upper_quantile = 1 - alpha / 2

                quantiles = uncertainty_estimates['quantiles']
                if lower_quantile in quantiles and upper_quantile in quantiles:
                    bounds_for_level.append((
                        quantiles[lower_quantile],
                        quantiles[upper_quantile]
                    ))

            # Ensemble bounds
            if 'ensemble' in uncertainty_estimates:
                ensemble = uncertainty_estimates['ensemble']
                mean_pred = ensemble['mean']
                std_pred = ensemble['std']

                z_score = norm.ppf(1 - (1 - conf_level) / 2)
                bounds_for_level.append((
                    mean_pred - z_score * std_pred,
                    mean_pred + z_score * std_pred
                ))

            # Conformal bounds
            conf_key = f'conformal_{conf_level}'
            if conf_key in uncertainty_estimates:
                bounds_for_level.append(uncertainty_estimates[conf_key])

            # Combine bounds (use most conservative)
            if bounds_for_level:
                all_lower = np.array([b[0] for b in bounds_for_level])
                all_upper = np.array([b[1] for b in bounds_for_level])

                final_lower = np.min(all_lower, axis=0)
                final_upper = np.max(all_upper, axis=0)

                lower_bounds.append(final_lower)
                upper_bounds.append(final_upper)

        # Return bounds for default confidence level
        if lower_bounds and upper_bounds:
            return lower_bounds[0], upper_bounds[0]
        else:
            # Fallback to simple standard deviation bounds
            std_estimate = np.std(predictions) if len(predictions) > 1 else 0.1
            return (predictions - 2 * std_estimate,
                   predictions + 2 * std_estimate)

    def predict_enhanced(self, X: np.ndarray) -> EnhancedForecastResult:
        """Generate enhanced forecast with comprehensive uncertainty quantification"""

        # Get main predictions with confidence intervals
        predictions, lower, upper = self._predict(X, return_confidence=True)

        # Get detailed uncertainty estimates
        uncertainty_estimates = {}

        # Quantile predictions
        quantile_predictions = None
        if self.quantile_forecaster is not None:
            X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
            quantile_predictions = self.quantile_forecaster.predict_quantiles(X_flat)

        # Ensemble predictions
        ensemble_predictions = None
        epistemic_uncertainty = None
        if self.ensemble_forecaster is not None:
            ensemble_results = self.ensemble_forecaster.predict_ensemble(X)
            ensemble_predictions = ensemble_results['individual']
            epistemic_uncertainty = ensemble_results['std']

        # Prediction standard deviation
        prediction_std = epistemic_uncertainty if epistemic_uncertainty is not None else np.std(predictions)

        # Generate timestamps
        timestamps = self._generate_timestamps(len(predictions))

        # Create enhanced result
        result = EnhancedForecastResult(
            predictions=predictions,
            timestamps=timestamps,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence_level=self.confidence_levels[0] if self.confidence_levels else 0.95,
            horizon=self.horizon,
            quantile_predictions=quantile_predictions,
            prediction_std=prediction_std,
            epistemic_uncertainty=epistemic_uncertainty,
            ensemble_predictions=ensemble_predictions,
            metadata={
                'model': f"Enhanced{self.base_model.title()}Forecaster",
                'uncertainty_methods': self.uncertainty_methods,
                'confidence_levels': self.confidence_levels,
                'generated_at': datetime.now()
            }
        )

        return result

    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for saving"""
        return {
            'base_model': self.base_model,
            'uncertainty_methods': self.uncertainty_methods,
            'confidence_levels': self.confidence_levels,
            'ensemble_size': self.ensemble_size
        }

    def _set_model_state(self, state: Dict[str, Any]):
        """Set model-specific state for loading"""
        self.base_model = state.get('base_model', 'transformer')
        self.uncertainty_methods = state.get('uncertainty_methods', ['quantile', 'ensemble'])
        self.confidence_levels = state.get('confidence_levels', [0.8, 0.9, 0.95])
        self.ensemble_size = state.get('ensemble_size', 5)


if __name__ == "__main__":
    # Demo and testing
    print("\n" + "="*60)
    print("Testing Enhanced Forecaster")
    print("="*60)

    # Create synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    t = np.linspace(0, 100, n_samples)

    # Multi-variate time series with trend and seasonality
    data = np.zeros((n_samples, 3))
    for i in range(3):
        data[:, i] = (
            0.1 * t +  # Trend
            2 * np.sin(2 * np.pi * 0.1 * t * (i + 1)) +  # Seasonality
            np.random.normal(0, 0.5, n_samples)  # Noise
        )

    # Test enhanced forecaster
    print("\n1. Testing Enhanced Forecaster...")

    forecaster = EnhancedForecaster(
        base_model="transformer",
        uncertainty_methods=["quantile", "ensemble", "conformal"],
        confidence_levels=[0.8, 0.9, 0.95],
        ensemble_size=3,  # Reduced for demo
        horizon=10,
        lookback=50
    )

    # Fit forecaster
    train_data = data[:800]
    test_data = data[800:]

    print("  Building models...")
    forecaster._build_model()

    print("  Training models...")
    forecaster.fit(train_data)

    # Generate enhanced predictions
    print("  Generating enhanced predictions...")
    enhanced_result = forecaster.predict_enhanced(train_data[-100:])

    print(f"\nResults:")
    print(f"  Predictions shape: {enhanced_result.predictions.shape}")
    print(f"  Confidence intervals available: {enhanced_result.confidence_lower is not None}")
    print(f"  Quantile predictions available: {enhanced_result.quantile_predictions is not None}")
    print(f"  Epistemic uncertainty available: {enhanced_result.epistemic_uncertainty is not None}")

    if enhanced_result.quantile_predictions:
        print(f"  Available quantiles: {list(enhanced_result.quantile_predictions.keys())}")

    print("\n" + "="*60)
    print("Enhanced forecaster test complete")
    print("="*60)