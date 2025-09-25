"""
Data Preprocessor Module for IoT Telemetry Data
Handles data cleaning, normalization, and feature engineering for anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
from scipy import signal, stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, 
    PowerTransformer, QuantileTransformer
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from collections import defaultdict
import pickle
import joblib
from tqdm import tqdm

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.data_ingestion.data_loader import TelemetryData

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    # Normalization
    normalization_method: str = "minmax"  # minmax, standard, robust, power, quantile
    feature_range: Tuple[float, float] = (0, 1)
    
    # Window settings
    window_size: int = 100
    stride: int = 10
    
    # Cleaning
    remove_outliers: bool = True
    outlier_method: str = "zscore"  # zscore, iqr, isolation_forest
    outlier_threshold: float = 3.0
    interpolate_missing: bool = True
    interpolation_method: str = "linear"  # linear, polynomial, spline
    max_missing_ratio: float = 0.3  # Maximum ratio of missing values
    
    # Feature engineering
    use_rolling_stats: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [10, 30, 60])
    use_fft_features: bool = False
    fft_components: int = 10
    use_wavelet_features: bool = False
    wavelet_type: str = "db4"
    use_statistical_features: bool = True
    use_pca: bool = False
    pca_components: int = 10
    
    # Validation
    validate_data: bool = True
    min_variance: float = 1e-6
    
    # Caching
    cache_preprocessed: bool = True
    cache_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'normalization_method': self.normalization_method,
            'window_size': self.window_size,
            'stride': self.stride,
            'remove_outliers': self.remove_outliers,
            'use_rolling_stats': self.use_rolling_stats,
            'use_fft_features': self.use_fft_features,
            'use_statistical_features': self.use_statistical_features
        }


@dataclass
class PreprocessedData:
    """Container for preprocessed data"""
    data: np.ndarray
    timestamps: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scaler: Optional[Any] = None
    feature_names: Optional[List[str]] = None
    
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[0]
    
    @property
    def n_features(self) -> int:
        return self.features.shape[1] if self.features is not None else self.data.shape[1]


class DataPreprocessor:
    """
    Main data preprocessor for telemetry data
    Handles cleaning, normalization, and feature engineering
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize data preprocessor
        
        Args:
            config: Preprocessing configuration
        """
        # Load configuration
        if config is None:
            self.config = self._load_config_from_settings()
        else:
            self.config = config
        
        # Initialize scalers dictionary
        self.scalers: Dict[str, Any] = {}
        
        # Feature engineering components
        self.pca_models: Dict[str, PCA] = {}
        self.feature_selectors: Dict[str, SelectKBest] = {}
        
        # Cache directory
        if self.config.cache_dir is None:
            self.config.cache_dir = get_data_path('processed') / 'preprocessor_cache'
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.preprocessing_stats: Dict[str, Any] = defaultdict(dict)
        
        logger.info(f"DataPreprocessor initialized with config: {self.config.normalization_method}")
    
    def _load_config_from_settings(self) -> PreprocessingConfig:
        """Load configuration from settings file"""
        preprocessing_config = get_config('preprocessing', {})
        
        return PreprocessingConfig(
            normalization_method=preprocessing_config.get('normalization', {}).get('method', 'minmax'),
            feature_range=tuple(preprocessing_config.get('normalization', {}).get('feature_range', [0, 1])),
            window_size=preprocessing_config.get('window', {}).get('size', 100),
            stride=preprocessing_config.get('window', {}).get('stride', 10),
            remove_outliers=preprocessing_config.get('cleaning', {}).get('remove_outliers', True),
            outlier_threshold=preprocessing_config.get('cleaning', {}).get('outlier_threshold', 3),
            interpolate_missing=preprocessing_config.get('cleaning', {}).get('interpolate_missing', True),
            interpolation_method=preprocessing_config.get('cleaning', {}).get('interpolation_method', 'linear'),
            use_rolling_stats=preprocessing_config.get('features', {}).get('use_rolling_stats', True),
            rolling_windows=preprocessing_config.get('features', {}).get('rolling_window_sizes', [10, 30, 60]),
            use_fft_features=preprocessing_config.get('features', {}).get('use_fft_features', False),
            use_wavelet_features=preprocessing_config.get('features', {}).get('use_wavelet_features', False)
        )
    
    def preprocess_telemetry(self, 
                            telemetry: TelemetryData,
                            fit_scaler: bool = True) -> PreprocessedData:
        """
        Preprocess single telemetry channel
        
        Args:
            telemetry: Telemetry data object
            fit_scaler: Whether to fit new scaler or use existing
            
        Returns:
            Preprocessed data object
        """
        channel_name = telemetry.channel_names[0] if telemetry.channel_names else 'unknown'
        logger.info(f"Preprocessing channel: {channel_name}")
        
        # Get raw data
        data = telemetry.data.copy()
        
        # Ensure 2D shape
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Step 1: Clean data
        data = self._clean_data(data, channel_name)
        
        # Step 2: Normalize data
        data, scaler = self._normalize_data(data, channel_name, fit_scaler)
        
        # Step 3: Extract features
        features = self._extract_features(data, channel_name)
        
        # Step 4: Create windows
        if self.config.window_size > 1:
            windowed_data = self._create_windows(data)
            windowed_features = self._create_windows(features) if features is not None else None
        else:
            windowed_data = data
            windowed_features = features
        
        # Create preprocessed data object
        preprocessed = PreprocessedData(
            data=windowed_data,
            timestamps=telemetry.timestamps,
            features=windowed_features,
            labels=self._create_labels(telemetry),
            metadata={
                'channel': channel_name,
                'spacecraft': telemetry.spacecraft,
                'original_shape': telemetry.shape,
                'preprocessing_config': self.config.to_dict()
            },
            scaler=scaler,
            feature_names=self._get_feature_names(channel_name)
        )
        
        # Cache if configured
        if self.config.cache_preprocessed:
            self._cache_preprocessed(preprocessed, channel_name)
        
        return preprocessed
    
    def _clean_data(self, data: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Clean data by handling missing values and outliers
        
        Args:
            data: Input data
            channel_name: Channel identifier
            
        Returns:
            Cleaned data
        """
        original_shape = data.shape
        
        # Handle missing values (NaN, Inf)
        nan_mask = np.isnan(data) | np.isinf(data)
        nan_ratio = np.sum(nan_mask) / data.size
        
        if nan_ratio > self.config.max_missing_ratio:
            logger.warning(f"Channel {channel_name} has {nan_ratio:.2%} missing values (exceeds threshold)")
        
        if self.config.interpolate_missing and nan_ratio > 0:
            data = self._interpolate_missing(data, nan_mask)
        
        # Remove outliers
        if self.config.remove_outliers:
            data = self._remove_outliers(data)
        
        # Store statistics
        self.preprocessing_stats[channel_name]['nan_ratio'] = nan_ratio
        self.preprocessing_stats[channel_name]['cleaned_shape'] = data.shape
        
        return data
    
    def _interpolate_missing(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Interpolate missing values
        
        Args:
            data: Data with missing values
            mask: Boolean mask of missing values
            
        Returns:
            Interpolated data
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            mask = mask.reshape(-1, 1)
        
        for col in range(data.shape[1]):
            col_mask = mask[:, col]
            if np.any(col_mask):
                # Get valid indices and values
                valid_idx = np.where(~col_mask)[0]
                valid_values = data[valid_idx, col]
                
                if len(valid_idx) > 1:
                    # Create interpolation function
                    if self.config.interpolation_method == 'linear':
                        f = interp1d(valid_idx, valid_values, kind='linear', 
                                    fill_value='extrapolate', bounds_error=False)
                    elif self.config.interpolation_method == 'polynomial':
                        # Use polynomial of degree 3
                        z = np.polyfit(valid_idx, valid_values, min(3, len(valid_idx)-1))
                        f = np.poly1d(z)
                    else:  # spline
                        f = interp1d(valid_idx, valid_values, kind='cubic', 
                                    fill_value='extrapolate', bounds_error=False)
                    
                    # Interpolate missing values
                    missing_idx = np.where(col_mask)[0]
                    data[missing_idx, col] = f(missing_idx)
                else:
                    # Fill with mean if not enough valid points
                    data[col_mask, col] = np.mean(valid_values) if len(valid_values) > 0 else 0
        
        return data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Remove outliers from data
        
        Args:
            data: Input data
            
        Returns:
            Data with outliers removed/clipped
        """
        if self.config.outlier_method == 'zscore':
            # Z-score based outlier removal
            z_scores = np.abs(stats.zscore(data, axis=0))
            outlier_mask = z_scores > self.config.outlier_threshold
            
            # Clip outliers to threshold
            for col in range(data.shape[1]):
                col_outliers = outlier_mask[:, col] if data.ndim > 1 else outlier_mask
                if np.any(col_outliers):
                    col_data = data[:, col] if data.ndim > 1 else data
                    mean = np.mean(col_data[~col_outliers])
                    std = np.std(col_data[~col_outliers])
                    
                    # Clip values
                    lower_bound = mean - self.config.outlier_threshold * std
                    upper_bound = mean + self.config.outlier_threshold * std
                    
                    if data.ndim > 1:
                        data[:, col] = np.clip(data[:, col], lower_bound, upper_bound)
                    else:
                        data = np.clip(data, lower_bound, upper_bound)
        
        elif self.config.outlier_method == 'iqr':
            # IQR based outlier removal
            Q1 = np.percentile(data, 25, axis=0)
            Q3 = np.percentile(data, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data = np.clip(data, lower_bound, upper_bound)
        
        return data
    
    def _normalize_data(self, 
                       data: np.ndarray,
                       channel_name: str,
                       fit_scaler: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Normalize data using configured method
        
        Args:
            data: Input data
            channel_name: Channel identifier
            fit_scaler: Whether to fit new scaler
            
        Returns:
            Tuple of (normalized data, scaler)
        """
        # Get or create scaler
        if channel_name in self.scalers and not fit_scaler:
            scaler = self.scalers[channel_name]
        else:
            # Create new scaler
            if self.config.normalization_method == 'minmax':
                scaler = MinMaxScaler(feature_range=self.config.feature_range)
            elif self.config.normalization_method == 'standard':
                scaler = StandardScaler()
            elif self.config.normalization_method == 'robust':
                scaler = RobustScaler()
            elif self.config.normalization_method == 'power':
                scaler = PowerTransformer(method='yeo-johnson')
            elif self.config.normalization_method == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                scaler = MinMaxScaler()
            
            # Fit scaler
            scaler.fit(data)
            self.scalers[channel_name] = scaler
        
        # Transform data
        normalized_data = scaler.transform(data)
        
        return normalized_data, scaler
    
    def _extract_features(self, data: np.ndarray, channel_name: str) -> Optional[np.ndarray]:
        """
        Extract features from data
        
        Args:
            data: Input data
            channel_name: Channel identifier
            
        Returns:
            Feature array or None
        """
        features_list = []
        
        # Rolling statistics
        if self.config.use_rolling_stats:
            rolling_features = self._compute_rolling_stats(data)
            features_list.append(rolling_features)
        
        # FFT features
        if self.config.use_fft_features:
            fft_features = self._compute_fft_features(data)
            features_list.append(fft_features)
        
        # Wavelet features
        if self.config.use_wavelet_features:
            wavelet_features = self._compute_wavelet_features(data)
            features_list.append(wavelet_features)
        
        # Statistical features
        if self.config.use_statistical_features:
            stat_features = self._compute_statistical_features(data)
            features_list.append(stat_features)
        
        if not features_list:
            return None
        
        # Concatenate all features
        features = np.hstack(features_list)
        
        # Apply PCA if configured
        if self.config.use_pca and features.shape[1] > self.config.pca_components:
            features = self._apply_pca(features, channel_name)
        
        return features
    
    def _compute_rolling_stats(self, data: np.ndarray) -> np.ndarray:
        """
        Compute rolling statistics features
        
        Args:
            data: Input data
            
        Returns:
            Rolling statistics features
        """
        features = []
        
        for window_size in self.config.rolling_windows:
            if window_size >= len(data):
                continue
            
            # Rolling mean
            rolling_mean = pd.Series(data.flatten()).rolling(
                window=window_size, min_periods=1
            ).mean().values.reshape(-1, 1)
            features.append(rolling_mean)
            
            # Rolling std
            rolling_std = pd.Series(data.flatten()).rolling(
                window=window_size, min_periods=1
            ).std().fillna(0).values.reshape(-1, 1)
            features.append(rolling_std)
            
            # Rolling min/max
            rolling_min = pd.Series(data.flatten()).rolling(
                window=window_size, min_periods=1
            ).min().values.reshape(-1, 1)
            rolling_max = pd.Series(data.flatten()).rolling(
                window=window_size, min_periods=1
            ).max().values.reshape(-1, 1)
            features.extend([rolling_min, rolling_max])
        
        if features:
            return np.hstack(features)
        else:
            return np.zeros((len(data), 1))
    
    def _compute_fft_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute FFT features
        
        Args:
            data: Input data
            
        Returns:
            FFT features
        """
        # Compute FFT
        fft_values = np.fft.fft(data.flatten())
        
        # Get magnitude of first N components
        fft_magnitude = np.abs(fft_values[:self.config.fft_components])
        
        # Repeat for all samples (simplified approach)
        features = np.tile(fft_magnitude, (len(data), 1))
        
        return features
    
    def _compute_wavelet_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute wavelet features
        
        Args:
            data: Input data
            
        Returns:
            Wavelet features
        """
        try:
            import pywt
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(data.flatten(), self.config.wavelet_type, level=4)
            
            # Extract features from coefficients
            features = []
            for coeff in coeffs:
                features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(np.abs(coeff))
                ])
            
            # Repeat for all samples
            features = np.tile(features, (len(data), 1))
            
            return features
            
        except ImportError:
            logger.warning("PyWavelets not installed, skipping wavelet features")
            return np.zeros((len(data), 1))
    
    def _compute_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """
        Compute statistical features
        
        Args:
            data: Input data
            
        Returns:
            Statistical features
        """
        features = []
        
        # Use sliding window for local statistics
        window = min(50, len(data) // 4)
        
        for i in range(len(data)):
            start = max(0, i - window)
            end = min(len(data), i + window + 1)
            window_data = data[start:end].flatten()
            
            # Compute statistics
            feat = [
                np.mean(window_data),
                np.std(window_data),
                np.median(window_data),
                stats.skew(window_data),
                stats.kurtosis(window_data),
                np.percentile(window_data, 25),
                np.percentile(window_data, 75)
            ]
            features.append(feat)
        
        return np.array(features)
    
    def _apply_pca(self, features: np.ndarray, channel_name: str) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            features: Input features
            channel_name: Channel identifier
            
        Returns:
            Reduced features
        """
        if channel_name not in self.pca_models:
            # Fit new PCA model
            pca = PCA(n_components=min(self.config.pca_components, features.shape[1]))
            pca.fit(features)
            self.pca_models[channel_name] = pca
        else:
            pca = self.pca_models[channel_name]
        
        # Transform features
        reduced_features = pca.transform(features)
        
        return reduced_features
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding windows from data
        
        Args:
            data: Input data
            
        Returns:
            Windowed data
        """
        windows = []
        
        for i in range(0, len(data) - self.config.window_size + 1, self.config.stride):
            window = data[i:i + self.config.window_size]
            windows.append(window)
        
        if not windows:
            # If data is too short, pad and create single window
            padded = np.pad(data, ((0, self.config.window_size - len(data)), (0, 0)), 'constant')
            windows.append(padded)
        
        return np.array(windows)
    
    def _create_labels(self, telemetry: TelemetryData) -> Optional[np.ndarray]:
        """
        Create labels from anomaly sequences
        
        Args:
            telemetry: Telemetry data object
            
        Returns:
            Label array or None
        """
        if not telemetry.anomaly_sequences:
            return None
        
        # Create binary labels
        labels = np.zeros(len(telemetry.data), dtype=np.int32)
        
        for start, end in telemetry.anomaly_sequences:
            labels[start:end] = 1
        
        # Window the labels if needed
        if self.config.window_size > 1:
            windowed_labels = []
            for i in range(0, len(labels) - self.config.window_size + 1, self.config.stride):
                # Label window as anomaly if any point in window is anomaly
                window_label = 1 if np.any(labels[i:i + self.config.window_size]) else 0
                windowed_labels.append(window_label)
            
            return np.array(windowed_labels)
        
        return labels
    
    def _get_feature_names(self, channel_name: str) -> List[str]:
        """
        Generate feature names
        
        Args:
            channel_name: Channel identifier
            
        Returns:
            List of feature names
        """
        names = []
        
        if self.config.use_rolling_stats:
            for window in self.config.rolling_windows:
                names.extend([
                    f'rolling_mean_{window}',
                    f'rolling_std_{window}',
                    f'rolling_min_{window}',
                    f'rolling_max_{window}'
                ])
        
        if self.config.use_fft_features:
            names.extend([f'fft_{i}' for i in range(self.config.fft_components)])
        
        if self.config.use_statistical_features:
            names.extend(['mean', 'std', 'median', 'skew', 'kurtosis', 'q25', 'q75'])
        
        return names
    
    def _cache_preprocessed(self, data: PreprocessedData, channel_name: str):
        """
        Cache preprocessed data
        
        Args:
            data: Preprocessed data
            channel_name: Channel identifier
        """
        cache_file = self.config.cache_dir / f"{channel_name}_preprocessed.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached preprocessed data for {channel_name}")
        except Exception as e:
            logger.warning(f"Failed to cache preprocessed data: {e}")
    
    def load_cached(self, channel_name: str) -> Optional[PreprocessedData]:
        """
        Load cached preprocessed data
        
        Args:
            channel_name: Channel identifier
            
        Returns:
            Preprocessed data or None
        """
        cache_file = self.config.cache_dir / f"{channel_name}_preprocessed.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded cached preprocessed data for {channel_name}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        return None
    
    def create_sequences(self,
                        data: np.ndarray,
                        sequence_length: int,
                        prediction_length: int = 1,
                        overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for sequence models
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            prediction_length: Length of prediction
            overlap: Overlap ratio between sequences
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        stride = int(sequence_length * (1 - overlap))
        stride = max(1, stride)
        
        for i in range(0, len(data) - sequence_length - prediction_length + 1, stride):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + prediction_length])
        
        return np.array(X), np.array(y)
    
    def prepare_for_lstm(self,
                        data: np.ndarray,
                        lookback: int = 100,
                        horizon: int = 1,
                        train_split: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Prepare data specifically for LSTM models
        
        Args:
            data: Input data
            lookback: Number of time steps to look back
            horizon: Number of time steps to predict
            train_split: Train/test split ratio
            
        Returns:
            Dictionary with train/test X/y arrays
        """
        # Create sequences
        X, y = self.create_sequences(data, lookback, horizon)
        
        # Split into train/test
        split_idx = int(len(X) * train_split)
        
        return {
            'X_train': X[:split_idx],
            'y_train': y[:split_idx],
            'X_test': X[split_idx:],
            'y_test': y[split_idx:]
        }
    
    def inverse_transform(self, 
                         data: np.ndarray,
                         channel_name: str) -> np.ndarray:
        """
        Inverse transform normalized data
        
        Args:
            data: Normalized data
            channel_name: Channel identifier
            
        Returns:
            Original scale data
        """
        if channel_name not in self.scalers:
            logger.warning(f"No scaler found for {channel_name}")
            return data
        
        scaler = self.scalers[channel_name]
        
        # Reshape if needed
        original_shape = data.shape
        if data.ndim == 3:
            # Flatten for inverse transform
            data = data.reshape(-1, data.shape[-1])
        
        # Inverse transform
        data = scaler.inverse_transform(data)
        
        # Restore shape
        if len(original_shape) == 3:
            data = data.reshape(original_shape)
        
        return data
    
    def save_preprocessor(self, filepath: Path):
        """
        Save preprocessor state
        
        Args:
            filepath: Path to save file
        """
        state = {
            'config': self.config,
            'scalers': self.scalers,
            'pca_models': self.pca_models,
            'feature_selectors': self.feature_selectors,
            'preprocessing_stats': dict(self.preprocessing_stats)
        }
        
        with open(filepath, 'wb') as f:
            joblib.dump(state, f)
        
        logger.info(f"Saved preprocessor to {filepath}")
    
    def load_preprocessor(self, filepath: Path):
        """
        Load preprocessor state
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
        
        self.config = state['config']
        self.scalers = state['scalers']
        self.pca_models = state['pca_models']
        self.feature_selectors = state['feature_selectors']
        self.preprocessing_stats = defaultdict(dict, state['preprocessing_stats'])
        
        logger.info(f"Loaded preprocessor from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return dict(self.preprocessing_stats)


class StreamPreprocessor:
    """Real-time preprocessor for streaming data"""
    
    def __init__(self, preprocessor: DataPreprocessor):
        """
        Initialize stream preprocessor
        
        Args:
            preprocessor: Trained data preprocessor
        """
        self.preprocessor = preprocessor
        self.buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def process_stream_event(self, event_data: np.ndarray, channel: str) -> np.ndarray:
        """
        Process single streaming event
        
        Args:
            event_data: Event data
            channel: Channel name
            
        Returns:
            Preprocessed data
        """
        # Add to buffer
        self.buffer[channel].append(event_data)
        
        # Get recent context
        context = np.array(self.buffer[channel])
        
        # Clean
        context = self.preprocessor._clean_data(context, channel)
        
        # Normalize using existing scaler
        if channel in self.preprocessor.scalers:
            context, _ = self.preprocessor._normalize_data(context, channel, fit_scaler=False)
        
        # Return latest processed point
        return context[-1]


if __name__ == "__main__":
    # Test data preprocessor
    print("\n" + "="*60)
    print("Testing Data Preprocessor")
    print("="*60)
    
    # Create test configuration
    config = PreprocessingConfig(
        normalization_method="minmax",
        window_size=100,
        stride=10,
        use_rolling_stats=True,
        use_fft_features=True,
        use_statistical_features=True
    )
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Create test data
    print("\nCreating test telemetry data...")
    test_data = np.sin(np.linspace(0, 10*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    
    # Add some anomalies
    test_data[400:450] += 2.0  # Spike anomaly
    test_data[700:750] *= 0.1  # Dip anomaly
    
    # Create telemetry object
    telemetry = TelemetryData(
        data=test_data,
        timestamps=pd.date_range(start='2024-01-01', periods=1000, freq='1min').values,
        channel_names=['test_channel'],
        spacecraft='test',
        anomaly_sequences=[(400, 450), (700, 750)]
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessed = preprocessor.preprocess_telemetry(telemetry)
    
    print(f"\nPreprocessing Results:")
    print(f"  Original shape: {telemetry.shape}")
    print(f"  Preprocessed shape: {preprocessed.shape}")
    print(f"  Number of windows: {preprocessed.n_samples}")
    print(f"  Features per window: {preprocessed.n_features if preprocessed.features is not None else 'N/A'}")
    print(f"  Has labels: {preprocessed.labels is not None}")
    
    if preprocessed.labels is not None:
        anomaly_ratio = np.mean(preprocessed.labels)
        print(f"  Anomaly ratio: {anomaly_ratio:.2%}")
    
    # Test LSTM preparation
    print("\nPreparing for LSTM...")
    lstm_data = preprocessor.prepare_for_lstm(test_data.reshape(-1, 1), lookback=50, horizon=10)
    
    print(f"LSTM Data Shapes:")
    print(f"  X_train: {lstm_data['X_train'].shape}")
    print(f"  y_train: {lstm_data['y_train'].shape}")
    print(f"  X_test: {lstm_data['X_test'].shape}")
    print(f"  y_test: {lstm_data['y_test'].shape}")
    
    # Get statistics
    print("\nPreprocessing Statistics:")
    stats = preprocessor.get_statistics()
    for channel, channel_stats in stats.items():
        print(f"  {channel}:")
        for key, value in channel_stats.items():
            print(f"    {key}: {value}")
    
    print("\n" + "="*60)
    print("Data preprocessor test complete")
    print("="*60)
