"""
Feature Engineering Module for IoT Telemetry Data
Advanced feature extraction and selection for anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import signal, stats, fft
from scipy.stats import entropy, kurtosis, skew
from scipy.spatial.distance import euclidean
from sklearn.feature_selection import (
    mutual_info_regression, SelectKBest, f_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import PolynomialFeatures
import warnings
from collections import defaultdict
from tqdm import tqdm

# Try to import optional dependencies
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("PyWavelets not installed. Wavelet features will be limited.")

try:
    from statsmodels.tsa.stattools import acf, pacf, adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not installed. Some time series features will be limited.")

try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    logging.warning("TSFresh not installed. Automated feature extraction will be limited.")

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features"""
    features: np.ndarray
    feature_names: List[str]
    timestamps: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_scores: Optional[np.ndarray] = None
    
    @property
    def n_features(self) -> int:
        return self.features.shape[1] if self.features.ndim > 1 else 1
    
    @property
    def n_samples(self) -> int:
        return self.features.shape[0]
    
    def get_top_features(self, n: int = 10) -> Tuple[np.ndarray, List[str]]:
        """Get top n important features"""
        if self.importance_scores is not None:
            top_indices = np.argsort(self.importance_scores)[-n:][::-1]
            return self.features[:, top_indices], [self.feature_names[i] for i in top_indices]
        return self.features[:, :n], self.feature_names[:n]


class TimeDomainFeatures:
    """Extract time-domain features from signals"""
    
    @staticmethod
    def basic_statistics(x: np.ndarray) -> Dict[str, float]:
        """Basic statistical features"""
        return {
            'mean': np.mean(x),
            'std': np.std(x),
            'var': np.var(x),
            'median': np.median(x),
            'min': np.min(x),
            'max': np.max(x),
            'range': np.ptp(x),
            'rms': np.sqrt(np.mean(x**2))
        }
    
    @staticmethod
    def higher_order_statistics(x: np.ndarray) -> Dict[str, float]:
        """Higher order statistical features"""
        return {
            'skewness': skew(x),
            'kurtosis': kurtosis(x),
            'q1': np.percentile(x, 25),
            'q3': np.percentile(x, 75),
            'iqr': np.percentile(x, 75) - np.percentile(x, 25),
            'percentile_10': np.percentile(x, 10),
            'percentile_90': np.percentile(x, 90)
        }
    
    @staticmethod
    def shape_features(x: np.ndarray) -> Dict[str, float]:
        """Shape-based features"""
        diff1 = np.diff(x)
        diff2 = np.diff(diff1)
        
        return {
            'crest_factor': np.max(np.abs(x)) / np.sqrt(np.mean(x**2)) if np.mean(x**2) > 0 else 0,
            'shape_factor': np.sqrt(np.mean(x**2)) / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
            'impulse_factor': np.max(np.abs(x)) / np.mean(np.abs(x)) if np.mean(np.abs(x)) > 0 else 0,
            'clearance_factor': np.max(np.abs(x)) / (np.mean(np.sqrt(np.abs(x)))**2) if np.mean(np.sqrt(np.abs(x))) > 0 else 0,
            'peak_to_peak': np.ptp(x),
            'mean_abs_deviation': np.mean(np.abs(x - np.mean(x))),
            'mean_diff': np.mean(diff1) if len(diff1) > 0 else 0,
            'mean_abs_diff': np.mean(np.abs(diff1)) if len(diff1) > 0 else 0,
            'mean_second_diff': np.mean(diff2) if len(diff2) > 0 else 0
        }
    
    @staticmethod
    def trend_features(x: np.ndarray) -> Dict[str, float]:
        """Trend and change detection features"""
        n = len(x)
        t = np.arange(n)
        
        # Linear trend
        if n > 1:
            slope, intercept = np.polyfit(t, x, 1)
            detrended = x - (slope * t + intercept)
            trend_strength = 1 - np.var(detrended) / np.var(x) if np.var(x) > 0 else 0
        else:
            slope = 0
            trend_strength = 0
        
        # Number of peaks
        peaks, _ = signal.find_peaks(x)
        valleys, _ = signal.find_peaks(-x)
        
        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(x - np.mean(x))) != 0)
        
        return {
            'trend_slope': slope,
            'trend_strength': trend_strength,
            'n_peaks': len(peaks),
            'n_valleys': len(valleys),
            'zero_crossings': zero_crossings,
            'mean_change': np.mean(np.diff(x)) if n > 1 else 0,
            'abs_energy': np.sum(x**2),
            'abs_sum_changes': np.sum(np.abs(np.diff(x))) if n > 1 else 0
        }
    
    @staticmethod
    def complexity_features(x: np.ndarray) -> Dict[str, float]:
        """Complexity and entropy measures"""
        # Approximate entropy
        def approx_entropy(U, m, r):
            def _maxdist(xi, xj, m):
                return max([abs(float(U[xi+k]) - float(U[xj+k])) for k in range(m)])
            
            def _phi(m):
                patterns = np.array([[U[i:i+m] for i in range(len(U)-m+1)]])
                C = np.zeros(len(patterns[0]))
                
                for i in range(len(patterns[0])):
                    dist_list = np.array([_maxdist(i, j, m) for j in range(len(patterns[0]))])
                    C[i] = len(np.where(dist_list <= r)[0]) / len(patterns[0])
                
                phi = np.sum(np.log(C)) / len(C)
                return phi
            
            try:
                return _phi(m) - _phi(m+1)
            except:
                return 0
        
        # Sample entropy (simplified)
        def sample_entropy(U, m, r):
            N = len(U)
            B = 0.0
            A = 0.0
            
            # Split into all possible m-length sequences
            xmi = [[U[i+j] for j in range(m)] for i in range(N-m)]
            xmj = [[U[i+j] for j in range(m+1)] for i in range(N-m)]
            
            # Count similar patterns
            for i in range(N-m):
                for j in range(N-m):
                    if i != j and max([abs(ua - va) for ua, va in zip(xmi[i], xmi[j])]) <= r:
                        B += 1
                    if i != j and max([abs(ua - va) for ua, va in zip(xmj[i][:m], xmj[j][:m])]) <= r:
                        if max([abs(xmj[i][m] - xmj[j][m])]) <= r:
                            A += 1
            
            return -np.log(A/B) if B > 0 and A > 0 else 0
        
        # Shannon entropy
        hist, _ = np.histogram(x, bins=10)
        hist = hist / np.sum(hist)
        shannon_entropy = entropy(hist + 1e-10)
        
        return {
            'approx_entropy': approx_entropy(x, 2, 0.2 * np.std(x)) if len(x) > 10 else 0,
            'sample_entropy': sample_entropy(x, 2, 0.2 * np.std(x)) if len(x) > 10 else 0,
            'shannon_entropy': shannon_entropy,
            'permutation_entropy': 0  # Placeholder, requires special implementation
        }


class FrequencyDomainFeatures:
    """Extract frequency-domain features from signals"""
    
    @staticmethod
    def fft_features(x: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
        """FFT-based features"""
        # Compute FFT
        fft_vals = fft.fft(x)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
        freqs = fft.fftfreq(len(x), 1/sampling_rate)[:len(fft_vals)//2]
        
        # Power spectral density
        psd = fft_mag ** 2
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(fft_mag)[0]
        if len(peak_indices) > 0:
            dominant_freq = freqs[peak_indices[np.argmax(fft_mag[peak_indices])]]
            max_magnitude = np.max(fft_mag[peak_indices])
        else:
            dominant_freq = 0
            max_magnitude = np.max(fft_mag) if len(fft_mag) > 0 else 0
        
        # Spectral features
        total_power = np.sum(psd)
        if total_power > 0:
            spectral_centroid = np.sum(freqs * psd) / total_power
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
            spectral_entropy = entropy(psd / total_power + 1e-10)
        else:
            spectral_centroid = 0
            spectral_spread = 0
            spectral_entropy = 0
        
        # Band powers
        low_band = np.sum(psd[freqs < 0.1 * sampling_rate])
        mid_band = np.sum(psd[(freqs >= 0.1 * sampling_rate) & (freqs < 0.3 * sampling_rate)])
        high_band = np.sum(psd[freqs >= 0.3 * sampling_rate])
        
        return {
            'dominant_frequency': dominant_freq,
            'max_magnitude': max_magnitude,
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_entropy': spectral_entropy,
            'spectral_energy': total_power,
            'spectral_kurtosis': kurtosis(fft_mag),
            'spectral_skewness': skew(fft_mag),
            'low_band_power': low_band,
            'mid_band_power': mid_band,
            'high_band_power': high_band,
            'band_power_ratio': low_band / (high_band + 1e-10),
            'n_peaks_fft': len(peak_indices)
        }
    
    @staticmethod
    def welch_features(x: np.ndarray, sampling_rate: float = 1.0) -> Dict[str, float]:
        """Welch's method for spectral analysis"""
        try:
            freqs, psd = signal.welch(x, fs=sampling_rate, nperseg=min(256, len(x)))
            
            # Find peak frequency
            peak_freq = freqs[np.argmax(psd)]
            
            # Bandwidth (frequency range containing 90% of power)
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            if total_power > 0:
                idx_low = np.where(cumsum_psd >= 0.05 * total_power)[0][0]
                idx_high = np.where(cumsum_psd >= 0.95 * total_power)[0][0]
                bandwidth = freqs[idx_high] - freqs[idx_low]
            else:
                bandwidth = 0
            
            return {
                'welch_peak_freq': peak_freq,
                'welch_bandwidth': bandwidth,
                'welch_total_power': total_power
            }
        except:
            return {
                'welch_peak_freq': 0,
                'welch_bandwidth': 0,
                'welch_total_power': 0
            }


class WaveletFeatures:
    """Extract wavelet-based features"""
    
    @staticmethod
    def dwt_features(x: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
        """Discrete Wavelet Transform features"""
        if not PYWT_AVAILABLE:
            return {}
        
        try:
            # Perform DWT
            coeffs = pywt.wavedec(x, wavelet, level=min(level, pywt.dwt_max_level(len(x), wavelet)))
            
            features = {}
            
            # Extract features from each level
            for i, coeff in enumerate(coeffs):
                level_name = 'cA' if i == 0 else f'cD{i}'
                features.update({
                    f'{level_name}_mean': np.mean(coeff),
                    f'{level_name}_std': np.std(coeff),
                    f'{level_name}_energy': np.sum(coeff**2),
                    f'{level_name}_entropy': entropy(np.abs(coeff) + 1e-10)
                })
            
            # Wavelet packet energy
            total_energy = sum([np.sum(c**2) for c in coeffs])
            relative_energies = [np.sum(c**2) / total_energy if total_energy > 0 else 0 for c in coeffs]
            
            for i, energy in enumerate(relative_energies):
                features[f'relative_energy_level_{i}'] = energy
            
            return features
            
        except Exception as e:
            logger.debug(f"Error computing wavelet features: {e}")
            return {}
    
    @staticmethod
    def cwt_features(x: np.ndarray, scales: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Continuous Wavelet Transform features"""
        if not PYWT_AVAILABLE:
            return {}
        
        try:
            if scales is None:
                scales = np.arange(1, min(128, len(x)//2))
            
            # Perform CWT
            coeffs, freqs = pywt.cwt(x, scales, 'morl')
            
            # Extract features
            features = {
                'cwt_mean_coeff': np.mean(np.abs(coeffs)),
                'cwt_std_coeff': np.std(np.abs(coeffs)),
                'cwt_max_coeff': np.max(np.abs(coeffs)),
                'cwt_dominant_scale': scales[np.unravel_index(np.argmax(np.abs(coeffs)), coeffs.shape)[0]]
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"Error computing CWT features: {e}")
            return {}


class TimeSeriesFeatures:
    """Extract time series specific features"""
    
    @staticmethod
    def autocorrelation_features(x: np.ndarray, max_lag: int = 20) -> Dict[str, float]:
        """Autocorrelation and partial autocorrelation features"""
        features = {}
        
        if STATSMODELS_AVAILABLE and len(x) > max_lag:
            try:
                # Autocorrelation
                acf_values = acf(x, nlags=min(max_lag, len(x)//4))
                features['acf_1'] = acf_values[1] if len(acf_values) > 1 else 0
                features['acf_5'] = acf_values[5] if len(acf_values) > 5 else 0
                features['acf_10'] = acf_values[10] if len(acf_values) > 10 else 0
                
                # First zero crossing of ACF
                zero_crossing = np.where(np.diff(np.sign(acf_values)))[0]
                features['acf_first_zero'] = zero_crossing[0] if len(zero_crossing) > 0 else max_lag
                
                # Partial autocorrelation
                try:
                    pacf_values = pacf(x, nlags=min(max_lag//2, len(x)//4))
                    features['pacf_1'] = pacf_values[1] if len(pacf_values) > 1 else 0
                    features['pacf_5'] = pacf_values[5] if len(pacf_values) > 5 else 0
                except:
                    features['pacf_1'] = 0
                    features['pacf_5'] = 0
                
            except Exception as e:
                logger.debug(f"Error computing autocorrelation: {e}")
        
        # Manual autocorrelation if statsmodels not available
        else:
            for lag in [1, 5, 10]:
                if lag < len(x):
                    correlation = np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
                    features[f'acf_{lag}'] = correlation if not np.isnan(correlation) else 0
                else:
                    features[f'acf_{lag}'] = 0
        
        return features
    
    @staticmethod
    def stationarity_features(x: np.ndarray) -> Dict[str, float]:
        """Test for stationarity"""
        features = {}
        
        if STATSMODELS_AVAILABLE and len(x) > 20:
            try:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(x, autolag='AIC')
                features['adf_statistic'] = adf_result[0]
                features['adf_pvalue'] = adf_result[1]
                features['is_stationary'] = 1 if adf_result[1] < 0.05 else 0
                
                # Ljung-Box test for autocorrelation
                lb_result = acorr_ljungbox(x, lags=min(10, len(x)//4), return_df=False)
                features['ljungbox_pvalue'] = lb_result[1][0] if len(lb_result[1]) > 0 else 1.0
                
            except Exception as e:
                logger.debug(f"Error computing stationarity tests: {e}")
                features['is_stationary'] = 0
        else:
            # Simple stationarity check
            mid = len(x) // 2
            first_half_mean = np.mean(x[:mid])
            second_half_mean = np.mean(x[mid:])
            first_half_std = np.std(x[:mid])
            second_half_std = np.std(x[mid:])
            
            mean_change = abs(first_half_mean - second_half_mean) / (abs(first_half_mean) + 1e-10)
            std_change = abs(first_half_std - second_half_std) / (first_half_std + 1e-10)
            
            features['mean_change_ratio'] = mean_change
            features['std_change_ratio'] = std_change
            features['is_stationary'] = 1 if mean_change < 0.1 and std_change < 0.1 else 0
        
        return features
    
    @staticmethod
    def lag_features(x: np.ndarray, lags: List[int] = None) -> Dict[str, float]:
        """Lag-based features"""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        features = {}
        
        for lag in lags:
            if lag < len(x):
                # Lagged values
                features[f'lag_{lag}_corr'] = np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
                features[f'lag_{lag}_diff_mean'] = np.mean(x[lag:] - x[:-lag])
                features[f'lag_{lag}_diff_std'] = np.std(x[lag:] - x[:-lag])
        
        return features


class AdvancedFeatureEngineering:
    """Main feature engineering class with advanced techniques"""
    
    def __init__(self):
        """Initialize feature engineering components"""
        self.time_domain = TimeDomainFeatures()
        self.freq_domain = FrequencyDomainFeatures()
        self.wavelet = WaveletFeatures()
        self.time_series = TimeSeriesFeatures()
        
        # Feature selection models
        self.feature_selector = None
        self.pca_model = None
        self.ica_model = None
        
        # Feature importance tracker
        self.feature_importance: Dict[str, float] = {}
        
        logger.info("AdvancedFeatureEngineering initialized")
    
    def extract_all_features(self, 
                            x: np.ndarray,
                            feature_groups: List[str] = None,
                            window_size: Optional[int] = None) -> FeatureSet:
        """
        Extract comprehensive feature set
        
        Args:
            x: Input signal
            feature_groups: List of feature groups to extract
            window_size: Window size for windowed features
            
        Returns:
            FeatureSet object
        """
        if feature_groups is None:
            feature_groups = ['time', 'frequency', 'wavelet', 'timeseries']
        
        all_features = {}
        
        # Time domain features
        if 'time' in feature_groups:
            all_features.update(self.time_domain.basic_statistics(x))
            all_features.update(self.time_domain.higher_order_statistics(x))
            all_features.update(self.time_domain.shape_features(x))
            all_features.update(self.time_domain.trend_features(x))
            all_features.update(self.time_domain.complexity_features(x))
        
        # Frequency domain features
        if 'frequency' in feature_groups:
            all_features.update(self.freq_domain.fft_features(x))
            all_features.update(self.freq_domain.welch_features(x))
        
        # Wavelet features
        if 'wavelet' in feature_groups and PYWT_AVAILABLE:
            all_features.update(self.wavelet.dwt_features(x))
            all_features.update(self.wavelet.cwt_features(x))
        
        # Time series features
        if 'timeseries' in feature_groups:
            all_features.update(self.time_series.autocorrelation_features(x))
            all_features.update(self.time_series.stationarity_features(x))
            all_features.update(self.time_series.lag_features(x))
        
        # Convert to arrays
        feature_values = np.array(list(all_features.values()))
        feature_names = list(all_features.keys())
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, 0)
        
        # Create feature set
        feature_set = FeatureSet(
            features=feature_values.reshape(1, -1),
            feature_names=feature_names,
            metadata={'n_features': len(feature_names)}
        )
        
        return feature_set
    
    def extract_windowed_features(self,
                                 x: np.ndarray,
                                 window_size: int,
                                 stride: int,
                                 feature_groups: List[str] = None) -> FeatureSet:
        """
        Extract features from sliding windows
        
        Args:
            x: Input signal
            window_size: Size of sliding window
            stride: Step size
            feature_groups: Feature groups to extract
            
        Returns:
            FeatureSet with windowed features
        """
        all_window_features = []
        
        # Generate windows
        for i in range(0, len(x) - window_size + 1, stride):
            window = x[i:i + window_size]
            
            # Extract features for this window
            window_features = self.extract_all_features(window, feature_groups)
            all_window_features.append(window_features.features.flatten())
        
        if not all_window_features:
            return FeatureSet(features=np.array([]), feature_names=[])
        
        # Stack all window features
        features_array = np.vstack(all_window_features)
        
        # Get feature names from first extraction
        first_extraction = self.extract_all_features(x[:window_size], feature_groups)
        
        return FeatureSet(
            features=features_array,
            feature_names=first_extraction.feature_names,
            metadata={'window_size': window_size, 'stride': stride}
        )
    
    def extract_multivariate_features(self,
                                     X: np.ndarray,
                                     method: str = 'correlation') -> FeatureSet:
        """
        Extract features from multivariate time series
        
        Args:
            X: Multivariate time series (n_samples, n_channels)
            method: Method for multivariate features
            
        Returns:
            FeatureSet with multivariate features
        """
        n_samples, n_channels = X.shape
        features = []
        feature_names = []
        
        if method == 'correlation':
            # Cross-correlation between channels
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                    features.append(corr)
                    feature_names.append(f'corr_ch{i}_ch{j}')
                    
                    # Lagged correlations
                    for lag in [1, 5, 10]:
                        if lag < n_samples:
                            lagged_corr = np.corrcoef(X[:-lag, i], X[lag:, j])[0, 1]
                            features.append(lagged_corr)
                            feature_names.append(f'lagged_corr_ch{i}_ch{j}_lag{lag}')
        
        elif method == 'mutual_information':
            # Mutual information between channels
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    mi = mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])[0]
                    features.append(mi)
                    feature_names.append(f'mi_ch{i}_ch{j}')
        
        elif method == 'coherence':
            # Spectral coherence between channels
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    f, Cxy = signal.coherence(X[:, i], X[:, j])
                    features.extend([
                        np.mean(Cxy),
                        np.max(Cxy),
                        f[np.argmax(Cxy)]
                    ])
                    feature_names.extend([
                        f'coherence_mean_ch{i}_ch{j}',
                        f'coherence_max_ch{i}_ch{j}',
                        f'coherence_peak_freq_ch{i}_ch{j}'
                    ])
        
        return FeatureSet(
            features=np.array(features).reshape(1, -1),
            feature_names=feature_names,
            metadata={'method': method, 'n_channels': n_channels}
        )
    
    def select_features(self,
                       X: np.ndarray,
                       y: Optional[np.ndarray] = None,
                       method: str = 'variance',
                       n_features: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select most important features
        
        Args:
            X: Feature matrix
            y: Target values (for supervised methods)
            method: Selection method
            n_features: Number of features to select
            
        Returns:
            Selected features and indices
        """
        n_features = min(n_features, X.shape[1])
        
        if method == 'variance':
            # Remove low variance features
            selector = VarianceThreshold(threshold=0.01)
            X_selected = selector.fit_transform(X)
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'mutual_info' and y is not None:
            # Select based on mutual information
            selector = SelectKBest(
                score_func=lambda X, y: mutual_info_regression(X, y),
                k=n_features
            )
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'f_classif' and y is not None:
            # ANOVA F-test
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            
        elif method == 'rfe' and y is not None:
            # Recursive Feature Elimination
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_indices = np.where(selector.support_)[0]
            
        elif method == 'random_forest' and y is not None:
            # Random Forest importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            selected_indices = np.argsort(importances)[-n_features:][::-1]
            X_selected = X[:, selected_indices]
            
            # Store importance scores
            self.feature_importance = dict(enumerate(importances))
            
        else:
            # Default: select first n features
            X_selected = X[:, :n_features]
            selected_indices = np.arange(n_features)
        
        return X_selected, selected_indices
    
    def reduce_dimensions(self,
                         X: np.ndarray,
                         method: str = 'pca',
                         n_components: int = 10) -> np.ndarray:
        """
        Reduce feature dimensions
        
        Args:
            X: Feature matrix
            method: Reduction method (pca, ica, nmf)
            n_components: Number of components
            
        Returns:
            Reduced feature matrix
        """
        n_components = min(n_components, min(X.shape))
        
        if method == 'pca':
            if self.pca_model is None:
                self.pca_model = PCA(n_components=n_components)
                X_reduced = self.pca_model.fit_transform(X)
            else:
                X_reduced = self.pca_model.transform(X)
                
        elif method == 'ica':
            if self.ica_model is None:
                self.ica_model = FastICA(n_components=n_components, random_state=42)
                X_reduced = self.ica_model.fit_transform(X)
            else:
                X_reduced = self.ica_model.transform(X)
                
        elif method == 'nmf':
            # Ensure non-negative values for NMF
            X_positive = X - X.min() + 1e-10
            nmf = NMF(n_components=n_components, random_state=42)
            X_reduced = nmf.fit_transform(X_positive)
            
        else:
            X_reduced = X[:, :n_components]
        
        return X_reduced
    
    def create_polynomial_features(self,
                                  X: np.ndarray,
                                  degree: int = 2,
                                  interaction_only: bool = False) -> np.ndarray:
        """
        Create polynomial and interaction features
        
        Args:
            X: Feature matrix
            degree: Polynomial degree
            interaction_only: Only interaction features
            
        Returns:
            Extended feature matrix
        """
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only)
        X_poly = poly.fit_transform(X)
        
        return X_poly
    
    def extract_tsfresh_features(self, 
                                data: pd.DataFrame,
                                column_id: str = 'id',
                                column_sort: str = 'time') -> pd.DataFrame:
        """
        Extract features using TSFresh library
        
        Args:
            data: DataFrame with time series
            column_id: ID column name
            column_sort: Time column name
            
        Returns:
            DataFrame with extracted features
        """
        if not TSFRESH_AVAILABLE:
            logger.warning("TSFresh not available")
            return pd.DataFrame()
        
        try:
            # Extract features
            extracted_features = extract_features(
                data, 
                column_id=column_id,
                column_sort=column_sort,
                disable_progressbar=True
            )
            
            # Impute missing values
            imputed_features = impute(extracted_features)
            
            return imputed_features
            
        except Exception as e:
            logger.error(f"Error extracting TSFresh features: {e}")
            return pd.DataFrame()


class OnlineFeatureExtractor:
    """Online feature extraction for streaming data"""
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 update_interval: int = 10):
        """
        Initialize online feature extractor
        
        Args:
            buffer_size: Size of data buffer
            update_interval: Interval for feature update
        """
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Data buffers
        self.buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        
        # Feature extractors
        self.feature_eng = AdvancedFeatureEngineering()
        
        # Incremental statistics
        self.running_mean: Dict[str, float] = defaultdict(float)
        self.running_var: Dict[str, float] = defaultdict(float)
        self.sample_count: Dict[str, int] = defaultdict(int)
        
    def update(self, channel: str, value: float) -> Optional[np.ndarray]:
        """
        Update with new value and extract features if ready
        
        Args:
            channel: Channel name
            value: New value
            
        Returns:
            Features if update interval reached
        """
        # Add to buffer
        self.buffers[channel].append(value)
        
        # Update running statistics
        self.sample_count[channel] += 1
        n = self.sample_count[channel]
        
        # Welford's online algorithm for mean and variance
        delta = value - self.running_mean[channel]
        self.running_mean[channel] += delta / n
        delta2 = value - self.running_mean[channel]
        self.running_var[channel] += delta * delta2
        
        # Check if time to extract features
        if n % self.update_interval == 0 and len(self.buffers[channel]) >= 100:
            return self._extract_online_features(channel)
        
        return None
    
    def _extract_online_features(self, channel: str) -> np.ndarray:
        """Extract features from buffer"""
        data = np.array(self.buffers[channel])
        
        # Extract subset of features suitable for online processing
        features = {}
        
        # Running statistics
        features['online_mean'] = self.running_mean[channel]
        features['online_std'] = np.sqrt(self.running_var[channel] / self.sample_count[channel])
        
        # Recent window statistics
        recent = data[-100:]
        features.update(self.feature_eng.time_domain.basic_statistics(recent))
        features.update(self.feature_eng.time_domain.trend_features(recent))
        
        # Simple frequency features
        if len(recent) >= 50:
            features.update(self.feature_eng.freq_domain.fft_features(recent))
        
        return np.array(list(features.values()))


if __name__ == "__main__":
    # Test feature engineering
    print("\n" + "="*60)
    print("Testing Advanced Feature Engineering")
    print("="*60)
    
    # Create test signal
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Complex test signal with multiple components
    signal_clean = (
        np.sin(2 * np.pi * 2 * t) +  # 2 Hz component
        0.5 * np.sin(2 * np.pi * 5 * t) +  # 5 Hz component
        0.3 * np.sin(2 * np.pi * 10 * t)  # 10 Hz component
    )
    
    # Add noise and anomalies
    noise = np.random.normal(0, 0.1, len(t))
    test_signal = signal_clean + noise
    
    # Add anomaly
    test_signal[400:450] += 2.0  # Amplitude anomaly
    test_signal[700:750] = np.random.normal(0, 2, 50)  # Noise burst
    
    # Initialize feature engineering
    print("\nInitializing feature engineering...")
    feature_eng = AdvancedFeatureEngineering()
    
    # Extract all features
    print("\nExtracting comprehensive features...")
    feature_set = feature_eng.extract_all_features(test_signal)
    
    print(f"\nExtracted Features:")
    print(f"  Number of features: {feature_set.n_features}")
    print(f"  Feature shape: {feature_set.features.shape}")
    
    # Show sample features
    print("\nSample features (first 10):")
    for i, (name, value) in enumerate(zip(feature_set.feature_names[:10], 
                                          feature_set.features.flatten()[:10])):
        print(f"  {name}: {value:.4f}")
    
    # Test windowed feature extraction
    print("\nExtracting windowed features...")
    windowed_features = feature_eng.extract_windowed_features(
        test_signal, 
        window_size=100, 
        stride=50
    )
    
    print(f"\nWindowed Features:")
    print(f"  Number of windows: {windowed_features.n_samples}")
    print(f"  Features per window: {windowed_features.n_features}")
    print(f"  Total shape: {windowed_features.features.shape}")
    
    # Test multivariate features
    print("\nTesting multivariate features...")
    multivariate_signal = np.column_stack([test_signal, signal_clean, noise])
    multi_features = feature_eng.extract_multivariate_features(
        multivariate_signal,
        method='correlation'
    )
    
    print(f"\nMultivariate Features:")
    print(f"  Number of features: {multi_features.n_features}")
    print(f"  Cross-channel correlations extracted")
    
    # Test feature selection
    if windowed_features.n_samples > 10:
        print("\nTesting feature selection...")
        
        # Create dummy labels for supervised selection
        labels = np.zeros(windowed_features.n_samples)
        labels[8:12] = 1  # Mark some windows as anomalies
        
        selected_features, indices = feature_eng.select_features(
            windowed_features.features,
            y=labels,
            method='variance',
            n_features=20
        )
        
        print(f"\nFeature Selection Results:")
        print(f"  Original features: {windowed_features.n_features}")
        print(f"  Selected features: {selected_features.shape[1]}")
        print(f"  Selected indices: {indices[:10]}...")
    
    # Test online feature extraction
    print("\nTesting online feature extraction...")
    online_extractor = OnlineFeatureExtractor(buffer_size=200, update_interval=10)
    
    for i, value in enumerate(test_signal[:100]):
        features = online_extractor.update('test_channel', value)
        if features is not None:
            print(f"  Features extracted at sample {i}: shape={features.shape}")
    
    print("\n" + "="*60)
    print("Feature engineering test complete")
    print("="*60)
