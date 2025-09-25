"""
Normalizer Module for IoT Telemetry Data
Handles various normalization techniques for consistent data scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import pickle
import joblib
from collections import defaultdict
from scipy import stats
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, 
    MaxAbsScaler, Normalizer as SklearnNormalizer,
    PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Statistics for normalization tracking"""
    method: str
    fitted: bool = False
    n_samples_seen: int = 0
    n_features: int = 0
    fit_time: Optional[datetime] = None
    transform_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_range: Optional[Tuple[float, float]] = None
    mean_values: Optional[np.ndarray] = None
    std_values: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method,
            'fitted': self.fitted,
            'n_samples_seen': self.n_samples_seen,
            'n_features': self.n_features,
            'fit_time': self.fit_time.isoformat() if self.fit_time else None,
            'transform_count': self.transform_count,
            'parameters': self.parameters,
            'data_range': self.data_range,
            'has_statistics': self.mean_values is not None
        }


class AdaptiveNormalizer(BaseEstimator, TransformerMixin):
    """
    Adaptive normalizer that adjusts to data distribution
    """
    
    def __init__(self, 
                 method: str = 'auto',
                 alpha: float = 0.01,
                 window_size: int = 1000):
        """
        Initialize adaptive normalizer
        
        Args:
            method: Normalization method or 'auto' for automatic selection
            alpha: Learning rate for adaptive updates
            window_size: Window size for statistics calculation
        """
        self.method = method
        self.alpha = alpha
        self.window_size = window_size
        
        # Running statistics
        self.running_mean = None
        self.running_var = None
        self.running_min = None
        self.running_max = None
        self.n_samples_seen = 0
        
        # Selected normalizer
        self.normalizer = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y=None):
        """
        Fit the adaptive normalizer
        
        Args:
            X: Training data
            y: Not used
        """
        X = np.asarray(X)
        
        if self.method == 'auto':
            # Automatically select best method based on data characteristics
            self.normalizer = self._select_best_normalizer(X)
        else:
            self.normalizer = self._create_normalizer(self.method)
        
        # Fit the normalizer
        self.normalizer.fit(X)
        
        # Initialize running statistics
        self.running_mean = np.mean(X, axis=0)
        self.running_var = np.var(X, axis=0)
        self.running_min = np.min(X, axis=0)
        self.running_max = np.max(X, axis=0)
        self.n_samples_seen = len(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data with optional adaptation
        
        Args:
            X: Data to transform
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")
        
        X = np.asarray(X)
        
        # Update running statistics
        self._update_statistics(X)
        
        # Transform using selected normalizer
        return self.normalizer.transform(X)
    
    def partial_fit(self, X: np.ndarray, y=None):
        """
        Incrementally fit the normalizer
        
        Args:
            X: New batch of data
            y: Not used
        """
        X = np.asarray(X)
        
        if not self.is_fitted:
            return self.fit(X)
        
        # Update running statistics
        self._update_statistics(X)
        
        # Adapt normalizer parameters if supported
        if hasattr(self.normalizer, 'partial_fit'):
            self.normalizer.partial_fit(X)
        elif isinstance(self.normalizer, (MinMaxScaler, StandardScaler)):
            # Manual adaptation for common scalers
            self._adapt_normalizer(X)
        
        return self
    
    def _update_statistics(self, X: np.ndarray):
        """Update running statistics"""
        batch_mean = np.mean(X, axis=0)
        batch_var = np.var(X, axis=0)
        batch_min = np.min(X, axis=0)
        batch_max = np.max(X, axis=0)
        
        # Exponential moving average update
        if self.running_mean is not None:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * batch_mean
            self.running_var = (1 - self.alpha) * self.running_var + self.alpha * batch_var
            self.running_min = np.minimum(self.running_min, batch_min)
            self.running_max = np.maximum(self.running_max, batch_max)
        else:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.running_min = batch_min
            self.running_max = batch_max
        
        self.n_samples_seen += len(X)
    
    def _adapt_normalizer(self, X: np.ndarray):
        """Adapt normalizer parameters based on new data"""
        if isinstance(self.normalizer, MinMaxScaler):
            # Update min-max bounds
            self.normalizer.data_min_ = self.running_min
            self.normalizer.data_max_ = self.running_max
            self.normalizer.data_range_ = self.running_max - self.running_min
            
        elif isinstance(self.normalizer, StandardScaler):
            # Update mean and variance
            self.normalizer.mean_ = self.running_mean
            self.normalizer.var_ = self.running_var
            self.normalizer.scale_ = np.sqrt(self.running_var)
    
    def _select_best_normalizer(self, X: np.ndarray):
        """Automatically select best normalizer based on data characteristics"""
        # Check for outliers
        z_scores = np.abs(stats.zscore(X.flatten()))
        has_outliers = np.any(z_scores > 3)
        
        # Check for skewness
        skewness = stats.skew(X.flatten())
        is_skewed = abs(skewness) > 1
        
        # Check for heavy tails
        kurtosis = stats.kurtosis(X.flatten())
        has_heavy_tails = kurtosis > 3
        
        # Select normalizer based on characteristics
        if has_outliers or has_heavy_tails:
            logger.info("Detected outliers/heavy tails - using RobustScaler")
            return RobustScaler()
        elif is_skewed:
            logger.info("Detected skewed distribution - using PowerTransformer")
            return PowerTransformer(method='yeo-johnson')
        else:
            logger.info("Normal distribution detected - using StandardScaler")
            return StandardScaler()
    
    def _create_normalizer(self, method: str):
        """Create normalizer based on method"""
        if method == 'minmax':
            return MinMaxScaler()
        elif method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'maxabs':
            return MaxAbsScaler()
        elif method == 'l2':
            return SklearnNormalizer(norm='l2')
        elif method == 'power':
            return PowerTransformer()
        elif method == 'quantile':
            return QuantileTransformer()
        else:
            return StandardScaler()


class StreamingNormalizer:
    """
    Normalizer for streaming data with online updates
    """
    
    def __init__(self,
                 method: str = 'standard',
                 buffer_size: int = 1000,
                 update_interval: int = 100):
        """
        Initialize streaming normalizer
        
        Args:
            method: Normalization method
            buffer_size: Size of data buffer for statistics
            update_interval: Interval for updating normalization parameters
        """
        self.method = method
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Data buffers for each channel
        self.buffers: Dict[str, list] = defaultdict(list)
        
        # Normalization parameters for each channel
        self.params: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Sample counters
        self.sample_counts: Dict[str, int] = defaultdict(int)
        
        # Fitted flag for each channel
        self.fitted_channels: set = set()
    
    def update(self, channel: str, value: float) -> float:
        """
        Update with new value and return normalized value
        
        Args:
            channel: Channel name
            value: New value
            
        Returns:
            Normalized value
        """
        # Add to buffer
        self.buffers[channel].append(value)
        if len(self.buffers[channel]) > self.buffer_size:
            self.buffers[channel].pop(0)
        
        self.sample_counts[channel] += 1
        
        # Check if we need to update parameters
        if self.sample_counts[channel] % self.update_interval == 0:
            self._update_params(channel)
        
        # Normalize value
        if channel in self.fitted_channels:
            return self._normalize_value(channel, value)
        elif len(self.buffers[channel]) >= 10:  # Minimal data for initial fit
            self._update_params(channel)
            return self._normalize_value(channel, value)
        else:
            return value  # Return unnormalized if not enough data
    
    def update_batch(self, channel: str, values: np.ndarray) -> np.ndarray:
        """
        Update with batch of values
        
        Args:
            channel: Channel name
            values: Array of values
            
        Returns:
            Normalized values
        """
        # Add to buffer
        self.buffers[channel].extend(values.tolist())
        if len(self.buffers[channel]) > self.buffer_size:
            self.buffers[channel] = self.buffers[channel][-self.buffer_size:]
        
        self.sample_counts[channel] += len(values)
        
        # Update parameters
        self._update_params(channel)
        
        # Normalize batch
        if channel in self.fitted_channels:
            return np.array([self._normalize_value(channel, v) for v in values])
        else:
            return values
    
    def _update_params(self, channel: str):
        """Update normalization parameters for channel"""
        if len(self.buffers[channel]) < 2:
            return
        
        data = np.array(self.buffers[channel])
        
        if self.method == 'minmax':
            self.params[channel]['min'] = np.min(data)
            self.params[channel]['max'] = np.max(data)
            self.params[channel]['range'] = self.params[channel]['max'] - self.params[channel]['min']
            
        elif self.method == 'standard':
            self.params[channel]['mean'] = np.mean(data)
            self.params[channel]['std'] = np.std(data)
            
        elif self.method == 'robust':
            self.params[channel]['median'] = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            self.params[channel]['iqr'] = q3 - q1
            
        elif self.method == 'maxabs':
            self.params[channel]['max_abs'] = np.max(np.abs(data))
        
        self.fitted_channels.add(channel)
    
    def _normalize_value(self, channel: str, value: float) -> float:
        """Normalize single value"""
        params = self.params[channel]
        
        if self.method == 'minmax':
            if params['range'] > 0:
                return (value - params['min']) / params['range']
            return 0.5
            
        elif self.method == 'standard':
            if params['std'] > 0:
                return (value - params['mean']) / params['std']
            return 0
            
        elif self.method == 'robust':
            if params['iqr'] > 0:
                return (value - params['median']) / params['iqr']
            return 0
            
        elif self.method == 'maxabs':
            if params['max_abs'] > 0:
                return value / params['max_abs']
            return 0
        
        return value
    
    def get_params(self, channel: str) -> Dict[str, Any]:
        """Get normalization parameters for channel"""
        return dict(self.params.get(channel, {}))


class MultiChannelNormalizer:
    """
    Normalizer for multi-channel time series data
    """
    
    def __init__(self,
                 method: str = 'standard',
                 per_channel: bool = True,
                 feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize multi-channel normalizer
        
        Args:
            method: Normalization method
            per_channel: Whether to normalize each channel independently
            feature_range: Range for MinMaxScaler
        """
        self.method = method
        self.per_channel = per_channel
        self.feature_range = feature_range
        
        # Normalizers for each channel
        self.normalizers: Dict[str, Any] = {}
        
        # Global normalizer (if not per-channel)
        self.global_normalizer = None
        
        # Statistics
        self.stats: Dict[str, NormalizationStats] = {}
        
        # Channel names
        self.channels: List[str] = []
        
        logger.info(f"MultiChannelNormalizer initialized: method={method}, per_channel={per_channel}")
    
    def fit(self, 
            X: np.ndarray,
            channel_names: Optional[List[str]] = None) -> 'MultiChannelNormalizer':
        """
        Fit normalizer on multi-channel data
        
        Args:
            X: Data array (n_samples, n_channels) or (n_samples, n_timesteps, n_channels)
            channel_names: Names for each channel
            
        Returns:
            Self
        """
        # Handle different input shapes
        if X.ndim == 2:
            n_samples, n_channels = X.shape
            X_reshaped = X
        elif X.ndim == 3:
            n_samples, n_timesteps, n_channels = X.shape
            X_reshaped = X.reshape(-1, n_channels)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
        
        # Set channel names
        if channel_names:
            self.channels = channel_names
        else:
            self.channels = [f'channel_{i}' for i in range(n_channels)]
        
        # Fit normalizers
        if self.per_channel:
            # Fit separate normalizer for each channel
            for i, channel in enumerate(self.channels):
                channel_data = X_reshaped[:, i].reshape(-1, 1)
                
                # Create and fit normalizer
                normalizer = self._create_normalizer()
                normalizer.fit(channel_data)
                self.normalizers[channel] = normalizer
                
                # Store statistics
                self.stats[channel] = NormalizationStats(
                    method=self.method,
                    fitted=True,
                    n_samples_seen=len(channel_data),
                    n_features=1,
                    fit_time=datetime.now(),
                    mean_values=np.mean(channel_data),
                    std_values=np.std(channel_data),
                    data_range=(np.min(channel_data), np.max(channel_data))
                )
        else:
            # Fit single normalizer for all channels
            self.global_normalizer = self._create_normalizer()
            self.global_normalizer.fit(X_reshaped)
            
            # Store global statistics
            self.stats['global'] = NormalizationStats(
                method=self.method,
                fitted=True,
                n_samples_seen=len(X_reshaped),
                n_features=n_channels,
                fit_time=datetime.now(),
                mean_values=np.mean(X_reshaped, axis=0),
                std_values=np.std(X_reshaped, axis=0),
                data_range=(np.min(X_reshaped), np.max(X_reshaped))
            )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform multi-channel data
        
        Args:
            X: Data array
            
        Returns:
            Normalized data
        """
        original_shape = X.shape
        
        # Handle different input shapes
        if X.ndim == 2:
            n_samples, n_channels = X.shape
            X_reshaped = X
        elif X.ndim == 3:
            n_samples, n_timesteps, n_channels = X.shape
            X_reshaped = X.reshape(-1, n_channels)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
        
        # Transform data
        if self.per_channel:
            # Transform each channel separately
            X_normalized = np.zeros_like(X_reshaped)
            
            for i, channel in enumerate(self.channels[:n_channels]):
                if channel in self.normalizers:
                    channel_data = X_reshaped[:, i].reshape(-1, 1)
                    X_normalized[:, i] = self.normalizers[channel].transform(channel_data).flatten()
                    
                    # Update statistics
                    if channel in self.stats:
                        self.stats[channel].transform_count += 1
                else:
                    # Channel not fitted, use identity
                    X_normalized[:, i] = X_reshaped[:, i]
        else:
            # Transform all channels together
            if self.global_normalizer:
                X_normalized = self.global_normalizer.transform(X_reshaped)
                
                # Update statistics
                if 'global' in self.stats:
                    self.stats['global'].transform_count += 1
            else:
                X_normalized = X_reshaped
        
        # Restore original shape
        if len(original_shape) == 3:
            X_normalized = X_normalized.reshape(original_shape)
        
        return X_normalized
    
    def fit_transform(self, 
                     X: np.ndarray,
                     channel_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step
        
        Args:
            X: Data array
            channel_names: Channel names
            
        Returns:
            Normalized data
        """
        self.fit(X, channel_names)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            X: Normalized data
            
        Returns:
            Original scale data
        """
        original_shape = X.shape
        
        # Handle different input shapes
        if X.ndim == 2:
            n_samples, n_channels = X.shape
            X_reshaped = X
        elif X.ndim == 3:
            n_samples, n_timesteps, n_channels = X.shape
            X_reshaped = X.reshape(-1, n_channels)
        else:
            raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")
        
        # Inverse transform
        if self.per_channel:
            X_original = np.zeros_like(X_reshaped)
            
            for i, channel in enumerate(self.channels[:n_channels]):
                if channel in self.normalizers:
                    channel_data = X_reshaped[:, i].reshape(-1, 1)
                    X_original[:, i] = self.normalizers[channel].inverse_transform(channel_data).flatten()
                else:
                    X_original[:, i] = X_reshaped[:, i]
        else:
            if self.global_normalizer:
                X_original = self.global_normalizer.inverse_transform(X_reshaped)
            else:
                X_original = X_reshaped
        
        # Restore original shape
        if len(original_shape) == 3:
            X_original = X_original.reshape(original_shape)
        
        return X_original
    
    def _create_normalizer(self):
        """Create normalizer based on method"""
        if self.method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'standard':
            return StandardScaler()
        elif self.method == 'robust':
            return RobustScaler()
        elif self.method == 'maxabs':
            return MaxAbsScaler()
        elif self.method == 'l2':
            return SklearnNormalizer(norm='l2')
        elif self.method == 'power':
            return PowerTransformer(method='yeo-johnson')
        elif self.method == 'quantile':
            return QuantileTransformer(output_distribution='normal')
        else:
            return StandardScaler()
    
    def get_statistics(self) -> Dict[str, NormalizationStats]:
        """Get normalization statistics"""
        return dict(self.stats)
    
    def save(self, filepath: Path):
        """
        Save normalizer state
        
        Args:
            filepath: Path to save file
        """
        state = {
            'method': self.method,
            'per_channel': self.per_channel,
            'feature_range': self.feature_range,
            'normalizers': self.normalizers,
            'global_normalizer': self.global_normalizer,
            'channels': self.channels,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            joblib.dump(state, f)
        
        logger.info(f"Saved normalizer to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load normalizer state
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
        
        self.method = state['method']
        self.per_channel = state['per_channel']
        self.feature_range = state['feature_range']
        self.normalizers = state['normalizers']
        self.global_normalizer = state['global_normalizer']
        self.channels = state['channels']
        self.stats = state['stats']
        
        logger.info(f"Loaded normalizer from {filepath}")


class RangeNormalizer:
    """
    Custom normalizer with configurable ranges and clipping
    """
    
    def __init__(self,
                 input_range: Optional[Tuple[float, float]] = None,
                 output_range: Tuple[float, float] = (0, 1),
                 clip: bool = True):
        """
        Initialize range normalizer
        
        Args:
            input_range: Expected input range (min, max)
            output_range: Desired output range
            clip: Whether to clip values outside range
        """
        self.input_range = input_range
        self.output_range = output_range
        self.clip = clip
        
        # Computed parameters
        self.input_min = None
        self.input_max = None
        self.scale = None
        self.offset = None
    
    def fit(self, X: np.ndarray) -> 'RangeNormalizer':
        """
        Fit normalizer to data
        
        Args:
            X: Training data
            
        Returns:
            Self
        """
        if self.input_range is None:
            # Learn range from data
            self.input_min = np.min(X)
            self.input_max = np.max(X)
        else:
            self.input_min = self.input_range[0]
            self.input_max = self.input_range[1]
        
        # Calculate scaling parameters
        input_range = self.input_max - self.input_min
        output_range = self.output_range[1] - self.output_range[0]
        
        if input_range > 0:
            self.scale = output_range / input_range
            self.offset = self.output_range[0] - self.scale * self.input_min
        else:
            self.scale = 1
            self.offset = 0
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to output range
        
        Args:
            X: Data to transform
            
        Returns:
            Normalized data
        """
        if self.scale is None:
            raise ValueError("Normalizer not fitted")
        
        # Apply transformation
        X_normalized = X * self.scale + self.offset
        
        # Clip if requested
        if self.clip:
            X_normalized = np.clip(X_normalized, self.output_range[0], self.output_range[1])
        
        return X_normalized
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform from output range to input range
        
        Args:
            X: Normalized data
            
        Returns:
            Original scale data
        """
        if self.scale is None or self.scale == 0:
            raise ValueError("Cannot inverse transform")
        
        return (X - self.offset) / self.scale
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)


def create_normalizer(method: str = 'standard', **kwargs) -> Any:
    """
    Factory function to create normalizer
    
    Args:
        method: Normalization method
        **kwargs: Additional arguments for normalizer
        
    Returns:
        Normalizer instance
    """
    if method == 'adaptive':
        return AdaptiveNormalizer(**kwargs)
    elif method == 'streaming':
        return StreamingNormalizer(method=kwargs.get('base_method', 'standard'), **kwargs)
    elif method == 'multichannel':
        return MultiChannelNormalizer(**kwargs)
    elif method == 'range':
        return RangeNormalizer(**kwargs)
    elif method == 'minmax':
        return MinMaxScaler(**kwargs)
    elif method == 'standard':
        return StandardScaler(**kwargs)
    elif method == 'robust':
        return RobustScaler(**kwargs)
    elif method == 'maxabs':
        return MaxAbsScaler(**kwargs)
    elif method == 'power':
        return PowerTransformer(**kwargs)
    elif method == 'quantile':
        return QuantileTransformer(**kwargs)
    else:
        logger.warning(f"Unknown method {method}, using StandardScaler")
        return StandardScaler()


if __name__ == "__main__":
    # Test normalizers
    print("\n" + "="*60)
    print("Testing Normalizer Module")
    print("="*60)
    
    # Create test data
    np.random.seed(42)
    
    # Single channel data with outliers
    single_channel = np.random.randn(1000) * 2 + 5
    single_channel[100:110] = 20  # Add outliers
    
    # Multi-channel data
    multi_channel = np.random.randn(1000, 3)
    multi_channel[:, 0] *= 2  # Different scales
    multi_channel[:, 1] += 10
    multi_channel[:, 2] *= 0.1
    
    # Test AdaptiveNormalizer
    print("\n1. Testing AdaptiveNormalizer...")
    adaptive_norm = AdaptiveNormalizer(method='auto')
    adaptive_norm.fit(single_channel.reshape(-1, 1))
    normalized = adaptive_norm.transform(single_channel.reshape(-1, 1))
    
    print(f"   Original range: [{single_channel.min():.2f}, {single_channel.max():.2f}]")
    print(f"   Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"   Selected method: {adaptive_norm.normalizer.__class__.__name__}")
    
    # Test StreamingNormalizer
    print("\n2. Testing StreamingNormalizer...")
    stream_norm = StreamingNormalizer(method='standard')
    
    normalized_stream = []
    for i, value in enumerate(single_channel[:100]):
        norm_value = stream_norm.update('channel_1', value)
        normalized_stream.append(norm_value)
        if i == 50:
            params = stream_norm.get_params('channel_1')
            print(f"   Parameters at sample 50: {params}")
    
    normalized_stream = np.array(normalized_stream)
    print(f"   Stream normalized range: [{normalized_stream.min():.2f}, {normalized_stream.max():.2f}]")
    
    # Test MultiChannelNormalizer
    print("\n3. Testing MultiChannelNormalizer...")
    
    # Per-channel normalization
    multi_norm_per = MultiChannelNormalizer(method='minmax', per_channel=True)
    multi_norm_per.fit(multi_channel, channel_names=['sensor_1', 'sensor_2', 'sensor_3'])
    normalized_per = multi_norm_per.transform(multi_channel)
    
    print("   Per-channel normalization:")
    for i in range(3):
        print(f"     Channel {i}: [{normalized_per[:, i].min():.2f}, {normalized_per[:, i].max():.2f}]")
    
    # Global normalization
    multi_norm_global = MultiChannelNormalizer(method='minmax', per_channel=False)
    multi_norm_global.fit(multi_channel)
    normalized_global = multi_norm_global.transform(multi_channel)
    
    print("   Global normalization:")
    print(f"     All channels: [{normalized_global.min():.2f}, {normalized_global.max():.2f}]")
    
    # Test inverse transform
    print("\n4. Testing inverse transform...")
    recovered = multi_norm_per.inverse_transform(normalized_per)
    error = np.mean(np.abs(recovered - multi_channel))
    print(f"   Reconstruction error: {error:.6f}")
    
    # Test RangeNormalizer
    print("\n5. Testing RangeNormalizer...")
    range_norm = RangeNormalizer(input_range=(-10, 10), output_range=(-1, 1), clip=True)
    range_norm.fit(single_channel)
    
    test_values = np.array([-15, -10, 0, 10, 15])  # Include out-of-range values
    normalized_range = range_norm.transform(test_values)
    
    print("   Input values:", test_values)
    print("   Normalized values:", normalized_range)
    
    # Test 3D data (for LSTM input)
    print("\n6. Testing 3D data normalization...")
    data_3d = multi_channel.reshape(100, 10, 3)  # (batch, timesteps, features)
    
    multi_norm_3d = MultiChannelNormalizer(method='standard', per_channel=True)
    multi_norm_3d.fit(data_3d)
    normalized_3d = multi_norm_3d.transform(data_3d)
    
    print(f"   Input shape: {data_3d.shape}")
    print(f"   Output shape: {normalized_3d.shape}")
    print(f"   Mean per channel: {normalized_3d.reshape(-1, 3).mean(axis=0)}")
    print(f"   Std per channel: {normalized_3d.reshape(-1, 3).std(axis=0)}")
    
    # Get statistics
    print("\n7. Normalization Statistics:")
    stats = multi_norm_per.get_statistics()
    for channel, stat in stats.items():
        print(f"   {channel}: {stat.to_dict()}")
    
    print("\n" + "="*60)
    print("Normalizer test complete")
    print("="*60)
