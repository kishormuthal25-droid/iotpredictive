"""
Helper Functions Module
Provides utility functions for the IoT Anomaly Detection System
"""

import os
import json
import yaml
import pickle
import hashlib
import uuid
import re
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from functools import wraps, lru_cache
import threading
import time
from collections import defaultdict, deque
import gzip
import base64
from urllib.parse import urlparse
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ========================================
# Data Manipulation Helpers
# ========================================

def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide two numbers or arrays, handling division by zero
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value for division by zero
    
    Returns:
        Result of division or default value
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = default
        else:
            result = default if not np.isfinite(result) else result
    return result

def normalize_data(data: np.ndarray, method: str = 'standard',
                  feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Any]:
    """Normalize data using specified method
    
    Args:
        data: Input data
        method: Normalization method ('standard', 'minmax', 'robust')
        feature_range: Range for minmax scaling
    
    Returns:
        Tuple of (normalized_data, scaler)
    """
    try:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Reshape if necessary
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        normalized = scaler.fit_transform(data)
        return normalized, scaler
        
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        raise

def create_sliding_windows(data: np.ndarray, window_size: int, 
                          stride: int = 1) -> np.ndarray:
    """Create sliding windows from time series data
    
    Args:
        data: Input time series data
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        Array of sliding windows
    """
    try:
        if len(data) < window_size:
            raise ValueError(f"Data length {len(data)} is less than window size {window_size}")
        
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i + window_size])
        
        return np.array(windows)
        
    except Exception as e:
        logger.error(f"Error creating sliding windows: {str(e)}")
        raise

def interpolate_missing_values(data: pd.DataFrame, method: str = 'linear',
                              limit: Optional[int] = None) -> pd.DataFrame:
    """Interpolate missing values in DataFrame
    
    Args:
        data: Input DataFrame
        method: Interpolation method
        limit: Maximum number of consecutive NaNs to fill
    
    Returns:
        DataFrame with interpolated values
    """
    try:
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_copy = data.copy()
        
        for col in numeric_columns:
            if data_copy[col].isna().any():
                data_copy[col] = data_copy[col].interpolate(method=method, limit=limit)
        
        return data_copy
        
    except Exception as e:
        logger.error(f"Error interpolating values: {str(e)}")
        raise

def remove_outliers(data: np.ndarray, method: str = 'iqr',
                    threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers from data
    
    Args:
        data: Input data
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Tuple of (clean_data, outlier_mask)
    """
    try:
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outlier_mask = z_scores > threshold
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        clean_data = data[~outlier_mask]
        return clean_data, outlier_mask
        
    except Exception as e:
        logger.error(f"Error removing outliers: {str(e)}")
        raise

# ========================================
# Time Series Helpers
# ========================================

def resample_timeseries(df: pd.DataFrame, freq: str,
                       agg_func: Union[str, Dict] = 'mean') -> pd.DataFrame:
    """Resample time series data to different frequency
    
    Args:
        df: DataFrame with DatetimeIndex
        freq: Target frequency ('1H', '1D', etc.)
        agg_func: Aggregation function or dict of functions per column
    
    Returns:
        Resampled DataFrame
    """
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
        
        return df.resample(freq).agg(agg_func)
        
    except Exception as e:
        logger.error(f"Error resampling time series: {str(e)}")
        raise

def detect_seasonality(data: np.ndarray, max_lag: int = 100) -> Optional[int]:
    """Detect seasonality period in time series data
    
    Args:
        data: Time series data
        max_lag: Maximum lag to check
    
    Returns:
        Detected period or None
    """
    try:
        from scipy.signal import find_peaks
        
        # Calculate autocorrelation
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr[:max_lag], height=0.3)
        
        if len(peaks) > 0:
            # Return the first significant peak as period
            return int(peaks[0])
        
        return None
        
    except Exception as e:
        logger.error(f"Error detecting seasonality: {str(e)}")
        return None

def create_time_features(df: pd.DataFrame, 
                        datetime_col: str = 'timestamp') -> pd.DataFrame:
    """Create time-based features from datetime column
    
    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column
    
    Returns:
        DataFrame with additional time features
    """
    try:
        df = df.copy()
        
        # Ensure datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Extract time features
        df['hour'] = df[datetime_col].dt.hour
        df['day'] = df[datetime_col].dt.day
        df['dayofweek'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month
        df['quarter'] = df[datetime_col].dt.quarter
        df['year'] = df[datetime_col].dt.year
        df['dayofyear'] = df[datetime_col].dt.dayofyear
        df['weekofyear'] = df[datetime_col].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating time features: {str(e)}")
        raise

# ========================================
# File I/O Helpers
# ========================================

def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {str(e)}")
        raise

def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2):
    """Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise

def load_yaml(filepath: Union[str, Path]) -> Dict:
    """Load YAML file
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML from {filepath}: {str(e)}")
        raise

def save_yaml(data: Dict, filepath: Union[str, Path]):
    """Save data to YAML file
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving YAML to {filepath}: {str(e)}")
        raise

def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded object
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {str(e)}")
        raise

def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Save object to pickle file
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {str(e)}")
        raise

def compress_file(filepath: Union[str, Path], delete_original: bool = False) -> str:
    """Compress file using gzip
    
    Args:
        filepath: Path to file
        delete_original: Whether to delete original file
    
    Returns:
        Path to compressed file
    """
    try:
        compressed_path = f"{filepath}.gz"
        
        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        if delete_original:
            os.remove(filepath)
        
        logger.info(f"Compressed {filepath} to {compressed_path}")
        return compressed_path
        
    except Exception as e:
        logger.error(f"Error compressing file: {str(e)}")
        raise

# ========================================
# Validation Helpers
# ========================================

def validate_dataframe(df: pd.DataFrame, required_columns: List[str],
                       dtype_mapping: Optional[Dict[str, type]] = None) -> bool:
    """Validate DataFrame structure and types
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        dtype_mapping: Optional mapping of column names to expected types
    
    Returns:
        True if valid, raises exception otherwise
    """
    try:
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if dtype_mapping:
            for col, expected_type in dtype_mapping.items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    if not np.issubdtype(actual_type, expected_type):
                        raise TypeError(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        # Check for empty DataFrame
        if df.empty:
            warnings.warn("DataFrame is empty")
        
        return True
        
    except Exception as e:
        logger.error(f"DataFrame validation failed: {str(e)}")
        raise

def validate_time_series(data: Union[np.ndarray, pd.Series],
                        min_length: int = 10,
                        check_stationarity: bool = False) -> bool:
    """Validate time series data
    
    Args:
        data: Time series data
        min_length: Minimum required length
        check_stationarity: Whether to check for stationarity
    
    Returns:
        True if valid
    """
    try:
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        
        # Check length
        if len(data) < min_length:
            raise ValueError(f"Time series too short: {len(data)} < {min_length}")
        
        # Check for NaN values
        if np.isnan(data).any():
            raise ValueError("Time series contains NaN values")
        
        # Check for constant values
        if np.std(data) == 0:
            raise ValueError("Time series is constant")
        
        # Check stationarity
        if check_stationarity:
            from statsmodels.stats.stattools import adfuller
            result = adfuller(data)
            if result[1] > 0.05:
                warnings.warn(f"Time series may be non-stationary (p-value: {result[1]:.4f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Time series validation failed: {str(e)}")
        raise

def validate_config(config: Dict, schema: Dict) -> bool:
    """Validate configuration against schema
    
    Args:
        config: Configuration dictionary
        schema: Schema dictionary defining required fields and types
    
    Returns:
        True if valid
    """
    try:
        def check_nested(cfg, sch, path=""):
            for key, value_type in sch.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in cfg:
                    raise ValueError(f"Missing required config key: {current_path}")
                
                if isinstance(value_type, dict):
                    if not isinstance(cfg[key], dict):
                        raise TypeError(f"Config key {current_path} should be dict")
                    check_nested(cfg[key], value_type, current_path)
                elif value_type is not None:
                    if not isinstance(cfg[key], value_type):
                        raise TypeError(f"Config key {current_path} should be {value_type.__name__}")
        
        check_nested(config, schema)
        return True
        
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise

# ========================================
# Conversion Helpers
# ========================================

def convert_to_numpy(data: Union[list, pd.DataFrame, pd.Series, np.ndarray]) -> np.ndarray:
    """Convert various data types to numpy array
    
    Args:
        data: Input data
    
    Returns:
        Numpy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to numpy array")

def convert_to_dataframe(data: Union[np.ndarray, list, dict],
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert various data types to DataFrame
    
    Args:
        data: Input data
        columns: Optional column names
    
    Returns:
        DataFrame
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data, columns=columns)
    elif isinstance(data, list):
        return pd.DataFrame(data, columns=columns)
    elif isinstance(data, dict):
        return pd.DataFrame(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to DataFrame")

def serialize_numpy(obj: Any) -> Any:
    """Serialize numpy arrays for JSON
    
    Args:
        obj: Object to serialize
    
    Returns:
        Serialized object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_numpy(item) for item in obj]
    else:
        return obj

# ========================================
# Mathematical Helpers
# ========================================

def calculate_moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Calculate moving average
    
    Args:
        data: Input data
        window: Window size
    
    Returns:
        Moving average
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')

def calculate_exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Calculate exponential moving average
    
    Args:
        data: Input data
        alpha: Smoothing factor (0-1)
    
    Returns:
        Exponential moving average
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema

def calculate_rolling_stats(data: pd.Series, window: int) -> pd.DataFrame:
    """Calculate rolling statistics
    
    Args:
        data: Input series
        window: Window size
    
    Returns:
        DataFrame with rolling statistics
    """
    return pd.DataFrame({
        'mean': data.rolling(window).mean(),
        'std': data.rolling(window).std(),
        'min': data.rolling(window).min(),
        'max': data.rolling(window).max(),
        'median': data.rolling(window).median()
    })

# ========================================
# String Helpers
# ========================================

def sanitize_string(text: str, allowed_chars: str = None) -> str:
    """Sanitize string by removing special characters
    
    Args:
        text: Input text
        allowed_chars: Additional allowed characters
    
    Returns:
        Sanitized string
    """
    if allowed_chars is None:
        allowed_chars = ''
    
    pattern = f'[^a-zA-Z0-9\\s{re.escape(allowed_chars)}]'
    return re.sub(pattern, '', text)

def generate_id(prefix: str = '') -> str:
    """Generate unique ID
    
    Args:
        prefix: Optional prefix
    
    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}{unique_id}" if prefix else unique_id

def hash_string(text: str, algorithm: str = 'sha256') -> str:
    """Hash string using specified algorithm
    
    Args:
        text: Input text
        algorithm: Hash algorithm
    
    Returns:
        Hash string
    """
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()

# ========================================
# Caching Helpers
# ========================================

class Cache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, ttl: int = 3600):
        """Initialize cache
        
        Args:
            ttl: Time to live in seconds
        """
        self.cache = {}
        self.ttl = ttl
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
    
    def cleanup(self):
        """Remove expired entries"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, (_, timestamp) in self.cache.items()
                if current_time - timestamp >= self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]

# Global cache instance
_cache = Cache()

def cached(ttl: int = 3600):
    """Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            result = _cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            _cache.set(cache_key, result)
            logger.debug(f"Cache miss for {func.__name__}")
            return result
        
        return wrapper
    return decorator

# ========================================
# Database Helpers
# ========================================

def build_insert_query(table: str, data: Dict[str, Any]) -> Tuple[str, List]:
    """Build INSERT SQL query
    
    Args:
        table: Table name
        data: Data to insert
    
    Returns:
        Tuple of (query, values)
    """
    columns = list(data.keys())
    values = list(data.values())
    placeholders = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)
    
    query = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"
    return query, values

def build_update_query(table: str, data: Dict[str, Any],
                      where: Dict[str, Any]) -> Tuple[str, List]:
    """Build UPDATE SQL query
    
    Args:
        table: Table name
        data: Data to update
        where: WHERE conditions
    
    Returns:
        Tuple of (query, values)
    """
    set_clause = ', '.join([f"{k} = %s" for k in data.keys()])
    where_clause = ' AND '.join([f"{k} = %s" for k in where.keys()])
    
    query = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
    values = list(data.values()) + list(where.values())
    
    return query, values

def paginate_query(query: str, page: int = 1, 
                  page_size: int = 100) -> str:
    """Add pagination to SQL query
    
    Args:
        query: Base query
        page: Page number (1-indexed)
        page_size: Items per page
    
    Returns:
        Query with pagination
    """
    offset = (page - 1) * page_size
    return f"{query} LIMIT {page_size} OFFSET {offset}"

# ========================================
# Retry and Error Handling
# ========================================

def retry(max_attempts: int = 3, delay: float = 1.0,
         backoff: float = 2.0, exceptions: Tuple = (Exception,)):
    """Decorator for retrying function on failure
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

# ========================================
# Parallel Processing Helpers
# ========================================

def chunk_data(data: Union[List, np.ndarray], chunk_size: int) -> List:
    """Split data into chunks
    
    Args:
        data: Data to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks

def parallel_apply(func: Callable, data: List, n_workers: int = 4) -> List:
    """Apply function to data in parallel
    
    Args:
        func: Function to apply
        data: List of data items
        n_workers: Number of worker threads
    
    Returns:
        List of results
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, data))
    
    return results

# ========================================
# System Helpers
# ========================================

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage
    
    Returns:
        Dictionary with memory statistics
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

def get_disk_usage(path: str = '/') -> Dict[str, float]:
    """Get disk usage statistics
    
    Args:
        path: Path to check
    
    Returns:
        Dictionary with disk statistics
    """
    import shutil
    
    total, used, free = shutil.disk_usage(path)
    
    return {
        'total_gb': total / (1024**3),
        'used_gb': used / (1024**3),
        'free_gb': free / (1024**3),
        'percent': (used / total) * 100
    }

def ensure_directory(path: Union[str, Path]):
    """Ensure directory exists
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_old_files(directory: Union[str, Path], days: int = 30,
                   pattern: str = '*'):
    """Clean old files from directory
    
    Args:
        directory: Directory path
        days: Age threshold in days
        pattern: File pattern to match
    """
    directory = Path(directory)
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                logger.info(f"Deleted old file: {file_path}")

# ========================================
# URL and Network Helpers
# ========================================

def validate_url(url: str) -> bool:
    """Validate URL format
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

@retry(max_attempts=3, delay=1.0)
def download_file(url: str, destination: Union[str, Path],
                 chunk_size: int = 8192) -> bool:
    """Download file from URL
    
    Args:
        url: Source URL
        destination: Destination path
        chunk_size: Download chunk size
    
    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded {url} to {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise

# ========================================
# Encoding Helpers
# ========================================

def encode_base64(data: bytes) -> str:
    """Encode data to base64
    
    Args:
        data: Binary data
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(data).decode('utf-8')

def decode_base64(encoded: str) -> bytes:
    """Decode base64 string
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Decoded binary data
    """
    return base64.b64decode(encoded)

# ========================================
# Date/Time Helpers
# ========================================

def parse_datetime(date_str: str, format: Optional[str] = None) -> datetime:
    """Parse datetime string
    
    Args:
        date_str: Date string
        format: Optional format string
    
    Returns:
        Datetime object
    """
    if format:
        return datetime.strptime(date_str, format)
    else:
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse datetime: {date_str}")

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def get_date_range(start_date: Union[str, datetime],
                  end_date: Union[str, datetime],
                  freq: str = 'D') -> pd.DatetimeIndex:
    """Generate date range
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency string
    
    Returns:
        DatetimeIndex
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    return pd.date_range(start=start_date, end=end_date, freq=freq)