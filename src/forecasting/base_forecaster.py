"""
Base Forecaster Module for Time Series Prediction
Abstract base class and common functionality for all forecasting models
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import pickle
import joblib
import json
from collections import defaultdict, deque
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    r2_score, explained_variance_score, median_absolute_error
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecasting results"""
    predictions: np.ndarray  # Predicted values
    timestamps: Optional[np.ndarray] = None  # Future timestamps
    confidence_lower: Optional[np.ndarray] = None  # Lower confidence bound
    confidence_upper: Optional[np.ndarray] = None  # Upper confidence bound
    confidence_level: float = 0.95  # Confidence level (e.g., 95%)
    horizon: int = 1  # Forecast horizon
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def point_forecast(self) -> np.ndarray:
        """Get point forecasts"""
        return self.predictions
    
    @property
    def prediction_intervals(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get prediction intervals"""
        if self.confidence_lower is not None and self.confidence_upper is not None:
            return (self.confidence_lower, self.confidence_upper)
        return None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        df = pd.DataFrame({'forecast': self.predictions.flatten()})
        
        if self.timestamps is not None:
            df['timestamp'] = self.timestamps
            df.set_index('timestamp', inplace=True)
        
        if self.confidence_lower is not None:
            df['lower_bound'] = self.confidence_lower.flatten()
        
        if self.confidence_upper is not None:
            df['upper_bound'] = self.confidence_upper.flatten()
        
        return df


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics"""
    mae: float = 0.0  # Mean Absolute Error
    mse: float = 0.0  # Mean Squared Error
    rmse: float = 0.0  # Root Mean Squared Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    smape: float = 0.0  # Symmetric MAPE
    r2: float = 0.0  # R-squared
    explained_variance: float = 0.0
    median_ae: float = 0.0  # Median Absolute Error
    
    # Direction accuracy (for classification of up/down)
    direction_accuracy: float = 0.0
    
    # Coverage probability for prediction intervals
    coverage_probability: float = 0.0
    interval_width: float = 0.0
    
    # Horizon-specific metrics
    horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'mape': self.mape,
            'smape': self.smape,
            'r2': self.r2,
            'explained_variance': self.explained_variance,
            'median_ae': self.median_ae,
            'direction_accuracy': self.direction_accuracy,
            'coverage_probability': self.coverage_probability,
            'interval_width': self.interval_width,
            'horizon_metrics': self.horizon_metrics
        }


class BaseForecast(ABC):
    """
    Abstract base class for all time series forecasting models
    """
    
    def __init__(self,
                 name: str = "BaseForecaster",
                 horizon: int = 1,
                 lookback: int = 100,
                 confidence_level: float = 0.95,
                 seasonality: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize base forecaster
        
        Args:
            name: Forecaster name
            horizon: Number of steps to forecast
            lookback: Number of historical steps to use
            confidence_level: Confidence level for prediction intervals
            seasonality: Seasonal period (if applicable)
            random_state: Random seed
        """
        self.name = name
        self.horizon = horizon
        self.lookback = lookback
        self.confidence_level = confidence_level
        self.seasonality = seasonality
        self.random_state = random_state
        
        # Model state
        self.is_fitted = False
        self.model = None
        
        # Data properties
        self.n_features = None
        self.feature_names: Optional[List[str]] = None
        self.frequency: Optional[str] = None  # Time series frequency
        
        # Scaling
        self.scaler: Optional[StandardScaler] = None
        self.scale_data = True
        
        # Training history
        self.training_history: Dict[str, List] = defaultdict(list)
        
        # Metrics
        self.train_metrics: Optional[ForecastMetrics] = None
        self.test_metrics: Optional[ForecastMetrics] = None
        
        # Cache for predictions
        self.prediction_cache: Dict[int, ForecastResult] = {}
        
        logger.info(f"{self.name} initialized with horizon={horizon}, lookback={lookback}")
    
    @abstractmethod
    def _build_model(self):
        """Build the underlying model architecture"""
        pass
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Internal fit implementation"""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray, return_confidence: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Internal prediction implementation
        
        Returns:
            Predictions or tuple of (predictions, lower_bound, upper_bound)
        """
        pass
    
    def fit(self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            validation_split: float = 0.2,
            verbose: int = 1) -> 'BaseForecast':
        """
        Fit the forecasting model
        
        Args:
            X: Historical time series data
            y: Target values (for supervised learning)
            validation_split: Validation data ratio
            verbose: Verbosity level
            
        Returns:
            Self
        """
        # Validate input
        X = self._validate_input(X, training=True)
        
        # Store data properties
        if X.ndim == 2:
            self.n_features = X.shape[1]
        elif X.ndim == 3:
            self.n_features = X.shape[2]
        else:
            self.n_features = 1
        
        # Scale data if needed
        if self.scale_data:
            if self.scaler is None:
                self.scaler = StandardScaler()
            X = self._scale_data(X, fit=True)
        
        # Create sequences for training
        X_seq, y_seq = self._create_sequences(X, y)
        
        # Split data for validation
        if validation_split > 0:
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        else:
            X_train, X_val = X_seq, None
            y_train, y_val = y_seq, None
        
        # Build model if needed
        if self.model is None:
            self._build_model()
        
        # Fit model
        self._fit(X_train, y_train)
        
        # Evaluate on training data
        if verbose > 0:
            train_pred = self._predict(X_train)
            self.train_metrics = self._evaluate(y_train, train_pred)
            logger.info(f"Training RMSE: {self.train_metrics.rmse:.4f}")
        
        # Evaluate on validation data
        if X_val is not None and y_val is not None:
            val_pred = self._predict(X_val)
            self.test_metrics = self._evaluate(y_val, val_pred)
            if verbose > 0:
                logger.info(f"Validation RMSE: {self.test_metrics.rmse:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self,
                X: Optional[np.ndarray] = None,
                n_steps: Optional[int] = None,
                return_confidence: bool = False) -> ForecastResult:
        """
        Generate forecasts
        
        Args:
            X: Historical data for prediction base
            n_steps: Number of steps to forecast (overrides horizon)
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Forecast results
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted")
        
        # Determine forecast horizon
        forecast_horizon = n_steps or self.horizon
        
        # Validate input
        if X is not None:
            X = self._validate_input(X)
            if self.scale_data:
                X = self._scale_data(X, fit=False)
        else:
            # Use last available data from training
            X = self._get_last_sequence()
        
        # Generate predictions
        if return_confidence:
            predictions, lower, upper = self._predict(X, return_confidence=True)
        else:
            predictions = self._predict(X, return_confidence=False)
            lower, upper = None, None
        
        # Inverse scale if needed
        if self.scale_data and self.scaler is not None:
            predictions = self._inverse_scale(predictions)
            if lower is not None:
                lower = self._inverse_scale(lower)
                upper = self._inverse_scale(upper)
        
        # Generate timestamps
        timestamps = self._generate_timestamps(len(predictions))
        
        # Create result
        result = ForecastResult(
            predictions=predictions,
            timestamps=timestamps,
            confidence_lower=lower,
            confidence_upper=upper,
            confidence_level=self.confidence_level,
            horizon=forecast_horizon,
            metadata={
                'model': self.name,
                'generated_at': datetime.now()
            }
        )
        
        return result
    
    def predict_recursive(self,
                         X: np.ndarray,
                         n_steps: int,
                         return_confidence: bool = False) -> ForecastResult:
        """
        Generate multi-step forecasts recursively
        
        Args:
            X: Initial historical data
            n_steps: Number of steps to forecast
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Forecast results
        """
        X = self._validate_input(X)
        if self.scale_data:
            X = self._scale_data(X, fit=False)
        
        predictions = []
        lower_bounds = [] if return_confidence else None
        upper_bounds = [] if return_confidence else None
        
        current_input = X.copy()
        
        for step in range(n_steps):
            # Predict next step
            if return_confidence:
                pred, lower, upper = self._predict(current_input, return_confidence=True)
                lower_bounds.append(lower[0])
                upper_bounds.append(upper[0])
            else:
                pred = self._predict(current_input, return_confidence=False)
            
            predictions.append(pred[0])
            
            # Update input for next prediction
            current_input = self._update_input(current_input, pred[0])
        
        predictions = np.array(predictions)
        
        # Inverse scale
        if self.scale_data:
            predictions = self._inverse_scale(predictions)
            if return_confidence:
                lower_bounds = self._inverse_scale(np.array(lower_bounds))
                upper_bounds = self._inverse_scale(np.array(upper_bounds))
        
        return ForecastResult(
            predictions=predictions,
            confidence_lower=lower_bounds,
            confidence_upper=upper_bounds,
            confidence_level=self.confidence_level,
            horizon=n_steps
        )
    
    def _create_sequences(self, 
                         X: np.ndarray,
                         y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for training
        
        Args:
            X: Time series data
            y: Target values
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if y is not None:
            # Supervised: X and y are separate
            return X, y
        
        # Self-supervised: create sequences from X
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - self.lookback - self.horizon + 1):
            sequences_X.append(X[i:i + self.lookback])
            sequences_y.append(X[i + self.lookback:i + self.lookback + self.horizon])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def _scale_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale data using StandardScaler"""
        original_shape = X.shape
        
        # Reshape for scaling
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Scale
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Restore shape
        if len(original_shape) == 3:
            X_scaled = X_scaled.reshape(original_shape)
        elif len(original_shape) == 1:
            X_scaled = X_scaled.flatten()
        
        return X_scaled
    
    def _inverse_scale(self, X: np.ndarray) -> np.ndarray:
        """Inverse scale data"""
        if self.scaler is None:
            return X
        
        original_shape = X.shape
        
        # Reshape for scaling
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Inverse scale
        X_original = self.scaler.inverse_transform(X)
        
        # Restore shape
        if len(original_shape) == 3:
            X_original = X_original.reshape(original_shape)
        elif len(original_shape) == 1:
            X_original = X_original.flatten()
        
        return X_original
    
    def _validate_input(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Validate and prepare input data"""
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, pd.Series):
            X = X.values
        X = np.asarray(X)
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Input contains NaN/Inf values, replacing with 0")
            X = np.nan_to_num(X, 0)
        
        # Ensure correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X
    
    def _evaluate(self, 
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 confidence_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> ForecastMetrics:
        """
        Evaluate forecast performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_bounds: Optional (lower, upper) confidence bounds
            
        Returns:
            Forecast metrics
        """
        # Flatten arrays for metrics calculation
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        metrics = ForecastMetrics()
        
        # Point forecast metrics
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.median_ae = median_absolute_error(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        if np.any(mask):
            metrics.mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # SMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if np.any(mask):
            metrics.smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        
        # R-squared and explained variance
        metrics.r2 = r2_score(y_true, y_pred)
        metrics.explained_variance = explained_variance_score(y_true, y_pred)
        
        # Direction accuracy (for time series)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            metrics.direction_accuracy = np.mean(true_direction == pred_direction)
        
        # Prediction interval metrics
        if confidence_bounds is not None:
            lower, upper = confidence_bounds
            lower = lower.flatten()
            upper = upper.flatten()
            
            # Coverage probability
            coverage = np.mean((y_true >= lower) & (y_true <= upper))
            metrics.coverage_probability = coverage
            
            # Average interval width
            metrics.interval_width = np.mean(upper - lower)
        
        # Horizon-specific metrics (if multi-step)
        if y_true.ndim > 1 or (len(y_true) > 1 and self.horizon > 1):
            # Calculate metrics for each horizon step
            for h in range(min(self.horizon, len(y_true))):
                h_true = y_true[h::self.horizon] if len(y_true) > self.horizon else y_true[h:h+1]
                h_pred = y_pred[h::self.horizon] if len(y_pred) > self.horizon else y_pred[h:h+1]
                
                if len(h_true) > 0 and len(h_pred) > 0:
                    metrics.horizon_metrics[h+1] = {
                        'mae': mean_absolute_error(h_true, h_pred),
                        'rmse': np.sqrt(mean_squared_error(h_true, h_pred))
                    }
        
        return metrics
    
    def _update_input(self, X: np.ndarray, new_value: np.ndarray) -> np.ndarray:
        """Update input sequence with new prediction for recursive forecasting"""
        # Shift sequence and add new value
        if X.ndim == 1:
            return np.append(X[1:], new_value)
        elif X.ndim == 2:
            return np.vstack([X[1:], new_value.reshape(1, -1)])
        else:
            # For 3D input (batch, sequence, features)
            updated = X.copy()
            updated[:, :-1] = X[:, 1:]
            updated[:, -1] = new_value
            return updated
    
    def _get_last_sequence(self) -> np.ndarray:
        """Get last sequence from training data (placeholder)"""
        # This should be overridden by specific implementations
        # Return dummy data for now
        if self.n_features:
            return np.zeros((self.lookback, self.n_features))
        return np.zeros(self.lookback)
    
    def _generate_timestamps(self, n_steps: int) -> np.ndarray:
        """Generate future timestamps"""
        # Generate hourly timestamps by default
        start = datetime.now()
        timestamps = pd.date_range(
            start=start,
            periods=n_steps,
            freq=self.frequency or 'H'
        )
        return timestamps.values
    
    def cross_validate(self,
                      X: np.ndarray,
                      n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            X: Time series data
            n_splits: Number of splits
            
        Returns:
            Cross-validation scores
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = defaultdict(list)
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            
            # Create sequences
            X_train_seq, y_train_seq = self._create_sequences(X_train)
            X_test_seq, y_test_seq = self._create_sequences(X_test)
            
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                continue
            
            # Fit model
            self._fit(X_train_seq, y_train_seq)
            
            # Predict
            predictions = self._predict(X_test_seq)
            
            # Evaluate
            metrics = self._evaluate(y_test_seq, predictions)
            cv_scores['mae'].append(metrics.mae)
            cv_scores['rmse'].append(metrics.rmse)
            cv_scores['mape'].append(metrics.mape)
            cv_scores['r2'].append(metrics.r2)
        
        return dict(cv_scores)
    
    def plot_forecast(self,
                     historical: np.ndarray,
                     forecast: ForecastResult,
                     actual: Optional[np.ndarray] = None,
                     save_path: Optional[Path] = None):
        """
        Plot forecast with historical data
        
        Args:
            historical: Historical time series
            forecast: Forecast results
            actual: Actual future values (if available)
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        hist_len = len(historical)
        ax.plot(range(hist_len), historical, 'b-', label='Historical', alpha=0.7)
        
        # Plot forecast
        forecast_range = range(hist_len, hist_len + len(forecast.predictions))
        ax.plot(forecast_range, forecast.predictions, 'r-', label='Forecast', linewidth=2)
        
        # Plot confidence intervals if available
        if forecast.confidence_lower is not None and forecast.confidence_upper is not None:
            ax.fill_between(
                forecast_range,
                forecast.confidence_lower.flatten(),
                forecast.confidence_upper.flatten(),
                color='red', alpha=0.2,
                label=f'{int(forecast.confidence_level*100)}% CI'
            )
        
        # Plot actual values if available
        if actual is not None:
            actual_range = range(hist_len, hist_len + len(actual))
            ax.plot(actual_range, actual, 'g-', label='Actual', alpha=0.7)
        
        ax.axvline(x=hist_len, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'{self.name} - Forecast (Horizon: {forecast.horizon})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      save_path: Optional[Path] = None):
        """
        Plot residual analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save plot
        """
        residuals = y_true.flatten() - y_pred.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals over time
        axes[0, 0].plot(residuals, 'b-', alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals histogram
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals vs Fitted
        axes[1, 1].scatter(y_pred.flatten(), residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals vs Fitted')
        axes[1, 1].set_xlabel('Fitted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Residual Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def save(self, filepath: Path):
        """Save forecaster state"""
        state = {
            'name': self.name,
            'horizon': self.horizon,
            'lookback': self.lookback,
            'confidence_level': self.confidence_level,
            'seasonality': self.seasonality,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'model_state': self._get_model_state()
        }
        
        with open(filepath, 'wb') as f:
            joblib.dump(state, f)
        
        logger.info(f"Saved {self.name} to {filepath}")
    
    def load(self, filepath: Path):
        """Load forecaster state"""
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
        
        self.name = state['name']
        self.horizon = state['horizon']
        self.lookback = state['lookback']
        self.confidence_level = state['confidence_level']
        self.seasonality = state['seasonality']
        self.is_fitted = state['is_fitted']
        self.n_features = state['n_features']
        self.feature_names = state['feature_names']
        self.scaler = state['scaler']
        self.train_metrics = state['train_metrics']
        self.test_metrics = state['test_metrics']
        
        # Rebuild model and load state
        if self.model is None:
            self._build_model()
        self._set_model_state(state['model_state'])
        
        logger.info(f"Loaded {self.name} from {filepath}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for saving"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model-specific state for loading"""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get forecaster parameters"""
        return {
            'name': self.name,
            'horizon': self.horizon,
            'lookback': self.lookback,
            'confidence_level': self.confidence_level,
            'seasonality': self.seasonality,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(horizon={self.horizon}, lookback={self.lookback}, fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()


class OnlineForecaster(BaseForecast):
    """
    Base class for online/streaming forecasters
    """
    
    def __init__(self,
                 name: str = "OnlineForecaster",
                 buffer_size: int = 1000,
                 update_interval: int = 100,
                 **kwargs):
        """
        Initialize online forecaster
        
        Args:
            name: Forecaster name
            buffer_size: Size of data buffer
            update_interval: Model update interval
            **kwargs: Additional arguments for base class
        """
        super().__init__(name=name, **kwargs)
        
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Streaming buffers
        self.data_buffer = deque(maxlen=buffer_size)
        self.forecast_buffer = deque(maxlen=buffer_size)
        
        # Streaming statistics
        self.samples_seen = 0
        self.last_update = 0
        self.forecast_errors = deque(maxlen=100)
    
    def update(self, new_value: float) -> ForecastResult:
        """
        Update with new value and generate forecast
        
        Args:
            new_value: New observation
            
        Returns:
            Forecast for next step(s)
        """
        # Add to buffer
        self.data_buffer.append(new_value)
        self.samples_seen += 1
        
        # Check if model needs update
        if self.samples_seen - self.last_update >= self.update_interval:
            self._update_model()
            self.last_update = self.samples_seen
        
        # Generate forecast if enough data
        if len(self.data_buffer) >= self.lookback:
            X = np.array(list(self.data_buffer)[-self.lookback:])
            forecast = self.predict(X.reshape(-1, 1))
            self.forecast_buffer.append(forecast.predictions[0])
            return forecast
        
        return ForecastResult(
            predictions=np.array([new_value]),  # Return same value if not enough data
            horizon=1
        )
    
    def _update_model(self):
        """Update model with buffered data"""
        if len(self.data_buffer) < self.lookback + self.horizon:
            return
        
        # Convert buffer to array
        X = np.array(self.data_buffer)
        
        # Update model
        if not self.is_fitted:
            self.fit(X.reshape(-1, 1))
        else:
            # Incremental update if supported
            if hasattr(self, 'partial_fit'):
                self.partial_fit(X.reshape(-1, 1))
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'samples_seen': self.samples_seen,
            'buffer_size': len(self.data_buffer),
            'last_update': self.last_update,
            'avg_forecast_error': np.mean(self.forecast_errors) if self.forecast_errors else 0
        }


if __name__ == "__main__":
    # Test base forecaster functionality
    print("\n" + "="*60)
    print("Testing Base Forecaster")
    print("="*60)
    
    # Create a simple concrete implementation for testing
    class SimpleForecaster(BaseForecast):
        """Simple forecaster for testing"""
        
        def _build_model(self):
            """Simple moving average model"""
            self.model = {'type': 'simple_ma'}
        
        def _fit(self, X, y=None):
            """Fit by calculating statistics"""
            self.mean = np.mean(X)
            self.std = np.std(X)
            
        def _predict(self, X, return_confidence=False):
            """Predict using moving average"""
            if not hasattr(self, 'mean'):
                predictions = np.zeros((len(X), self.horizon))
            else:
                # Simple prediction: last value + noise
                predictions = np.tile(np.mean(X, axis=1).reshape(-1, 1), (1, self.horizon))
                predictions += np.random.normal(0, self.std * 0.1, predictions.shape)
            
            if return_confidence:
                lower = predictions - 1.96 * self.std
                upper = predictions + 1.96 * self.std
                return predictions, lower, upper
            
            return predictions
        
        def _get_model_state(self):
            return {'mean': self.mean, 'std': self.std} if hasattr(self, 'mean') else {}
        
        def _set_model_state(self, state):
            if 'mean' in state:
                self.mean = state['mean']
                self.std = state['std']
    
    # Create test time series
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    ts_data = np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(len(t))
    
    # Test forecaster
    print("\n1. Testing Simple Forecaster...")
    forecaster = SimpleForecaster(
        name="TestForecaster",
        horizon=10,
        lookback=50
    )
    
    # Fit forecaster
    train_data = ts_data[:800]
    test_data = ts_data[800:]
    
    forecaster.fit(train_data.reshape(-1, 1), validation_split=0.2)
    
    # Get metrics
    if forecaster.test_metrics:
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {forecaster.test_metrics.rmse:.4f}")
        print(f"  MAE: {forecaster.test_metrics.mae:.4f}")
        print(f"  R2: {forecaster.test_metrics.r2:.4f}")
    
    # Generate forecast
    print("\n2. Testing Forecast Generation...")
    forecast = forecaster.predict(
        train_data[-50:].reshape(-1, 1),
        return_confidence=True
    )
    
    print(f"Forecast shape: {forecast.predictions.shape}")
    print(f"Confidence intervals available: {forecast.prediction_intervals is not None}")
    
    # Test recursive forecasting
    print("\n3. Testing Recursive Forecasting...")
    recursive_forecast = forecaster.predict_recursive(
        train_data[-50:].reshape(-1, 1),
        n_steps=20,
        return_confidence=True
    )
    print(f"Recursive forecast shape: {recursive_forecast.predictions.shape}")
    
    # Plot forecast
    print("\n4. Plotting Forecast...")
    forecaster.plot_forecast(
        train_data[-100:],
        forecast,
        actual=test_data[:10]
    )
    
    # Test online forecaster
    print("\n5. Testing Online Forecaster...")
    
    class SimpleOnlineForecaster(OnlineForecaster, SimpleForecaster):
        """Online version of simple forecaster"""
        pass
    
    online_forecaster = SimpleOnlineForecaster(
        name="OnlineForecaster",
        horizon=1,
        lookback=20,
        buffer_size=100,
        update_interval=50
    )
    
    # Simulate streaming
    for i, value in enumerate(train_data[:200]):
        forecast_result = online_forecaster.update(value)
        
        if i == 100:
            stats = online_forecaster.get_streaming_stats()
            print(f"  Streaming stats at sample 100: {stats}")
    
    print("\n" + "="*60)
    print("Base forecaster test complete")
    print("="*60)
