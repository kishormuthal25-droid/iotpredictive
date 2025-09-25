"""
Data Loader Module for NASA SMAP and MSL Telemetry Data
Handles loading of .npy and .h5 files for unsupervised anomaly detection
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Generator
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pickle
from tqdm import tqdm
import warnings

# Import settings
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Container for telemetry data"""
    data: np.ndarray
    timestamps: Optional[np.ndarray] = None
    channel_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    spacecraft: str = "unknown"
    anomaly_sequences: Optional[List[Tuple[int, int]]] = None
    
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[0]
    
    @property
    def n_channels(self) -> int:
        return self.data.shape[1] if len(self.data.shape) > 1 else 1
    
    def __len__(self) -> int:
        return self.n_samples


class DataLoader:
    """
    Main data loader class for SMAP and MSL telemetry data
    Handles .npy and .h5 files with support for streaming and batching
    """
    
    def __init__(self, 
                 spacecraft: str = "smap",
                 data_dir: Optional[Path] = None,
                 normalize: bool = True,
                 scaler_type: str = "minmax",
                 cache_processed: bool = True,
                 verbose: bool = True):
        """
        Initialize data loader
        
        Args:
            spacecraft: 'smap' or 'msl'
            data_dir: Path to data directory
            normalize: Whether to normalize data
            scaler_type: Type of scaler ('minmax', 'standard', 'robust')
            cache_processed: Cache processed data for faster loading
            verbose: Print loading progress
        """
        self.spacecraft = spacecraft.lower()
        assert self.spacecraft in ['smap', 'msl'], "Spacecraft must be 'smap' or 'msl'"
        
        # Set data directory
        if data_dir is None:
            self.data_dir = get_data_path(f'{self.spacecraft}_data')
        else:
            self.data_dir = Path(data_dir)
        
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.cache_processed = cache_processed
        self.verbose = verbose
        
        # Initialize containers
        self.telemetry_data: Dict[str, TelemetryData] = {}
        self.scalers: Dict[str, Any] = {}
        self.channel_info: Dict[str, Dict] = {}
        
        # File patterns
        self.npy_pattern = "*.npy"
        self.h5_pattern = "*.h5"
        self.hdf_pattern = "*.hdf5"
        
        # Cache directory
        self.cache_dir = get_data_path('processed') / f'{self.spacecraft}_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataLoader initialized for {self.spacecraft.upper()} spacecraft")
        logger.info(f"Data directory: {self.data_dir}")
    
    def load_all_data(self) -> Dict[str, TelemetryData]:
        """
        Load all available data files
        
        Returns:
            Dictionary of telemetry data objects
        """
        if self.telemetry_data:
            logger.info("Data already loaded, returning cached data")
            return self.telemetry_data
        
        # Check cache first
        if self.cache_processed:
            cached_data = self._load_from_cache()
            if cached_data:
                self.telemetry_data = cached_data
                return self.telemetry_data
        
        # Load NPY files
        npy_files = self._find_files(self.npy_pattern)
        if npy_files:
            logger.info(f"Found {len(npy_files)} .npy files")
            self._load_npy_files(npy_files)
        
        # Load H5/HDF5 files
        h5_files = self._find_files(self.h5_pattern) + self._find_files(self.hdf_pattern)
        if h5_files:
            logger.info(f"Found {len(h5_files)} .h5/.hdf5 files")
            self._load_h5_files(h5_files)
        
        # Load labeled anomaly data if available
        self._load_anomaly_labels()
        
        # Normalize data if requested
        if self.normalize and self.telemetry_data:
            self._normalize_data()
        
        # Cache processed data
        if self.cache_processed and self.telemetry_data:
            self._save_to_cache()
        
        logger.info(f"Loaded {len(self.telemetry_data)} telemetry channels")
        return self.telemetry_data
    
    def _find_files(self, pattern: str) -> List[Path]:
        """Find all files matching pattern in data directory"""
        files = list(self.data_dir.glob(pattern))
        # Also check subdirectories
        files.extend(list(self.data_dir.glob(f"**/{pattern}")))
        return sorted(set(files))
    
    def _load_npy_files(self, files: List[Path]):
        """Load data from NPY files"""
        for file_path in tqdm(files, desc="Loading NPY files", disable=not self.verbose):
            try:
                # Load numpy array
                data = np.load(file_path, allow_pickle=True)
                
                # Get channel name from filename
                channel_name = file_path.stem
                
                # Handle different data formats
                if isinstance(data, np.ndarray):
                    if data.dtype == object:
                        # Try to extract numeric data
                        data = self._extract_numeric_data(data)
                    
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    
                    # Create telemetry object
                    telemetry = TelemetryData(
                        data=data,
                        timestamps=self._generate_timestamps(len(data)),
                        channel_names=[channel_name],
                        spacecraft=self.spacecraft,
                        metadata={'source_file': str(file_path)}
                    )
                    
                    self.telemetry_data[channel_name] = telemetry
                    logger.debug(f"Loaded {channel_name}: shape={data.shape}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    def _load_h5_files(self, files: List[Path]):
        """Load data from H5/HDF5 files"""
        for file_path in tqdm(files, desc="Loading H5 files", disable=not self.verbose):
            try:
                with h5py.File(file_path, 'r') as f:
                    # Explore the structure
                    self._explore_h5_structure(f, file_path)
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    def _explore_h5_structure(self, h5_file: h5py.File, file_path: Path, prefix: str = ""):
        """Recursively explore and load data from H5 file"""
        for key in h5_file.keys():
            item = h5_file[key]
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(item, h5py.Dataset):
                # This is actual data
                try:
                    data = item[:]
                    
                    # Convert to numpy array if needed
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    
                    # Ensure 2D shape
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    elif data.ndim > 2:
                        # Flatten extra dimensions
                        data = data.reshape(data.shape[0], -1)
                    
                    # Create telemetry object
                    telemetry = TelemetryData(
                        data=data,
                        timestamps=self._generate_timestamps(len(data)),
                        channel_names=[full_key],
                        spacecraft=self.spacecraft,
                        metadata={
                            'source_file': str(file_path),
                            'h5_path': full_key,
                            'original_shape': item.shape,
                            'dtype': str(item.dtype)
                        }
                    )
                    
                    # Use cleaned key for storage
                    clean_key = full_key.replace('/', '_')
                    self.telemetry_data[clean_key] = telemetry
                    logger.debug(f"Loaded {clean_key}: shape={data.shape}")
                    
                except Exception as e:
                    logger.warning(f"Could not load dataset {full_key}: {e}")
                    
            elif isinstance(item, h5py.Group):
                # Recursively explore groups
                self._explore_h5_structure(item, file_path, f"{full_key}/")
    
    def _extract_numeric_data(self, data: np.ndarray) -> np.ndarray:
        """Extract numeric data from object arrays"""
        if data.dtype == object:
            # Try to convert to float
            try:
                if len(data.shape) == 1:
                    # Check if it contains dictionaries or complex objects
                    if isinstance(data[0], dict):
                        # Extract values from dictionaries
                        keys = data[0].keys()
                        extracted = []
                        for item in data:
                            if isinstance(item, dict):
                                extracted.append(list(item.values()))
                        return np.array(extracted, dtype=np.float32)
                    else:
                        # Try direct conversion
                        return np.array(data, dtype=np.float32)
                else:
                    return np.array(data, dtype=np.float32)
            except:
                logger.warning("Could not extract numeric data from object array")
                return np.array([])
        return data
    
    def _load_anomaly_labels(self):
        """Load anomaly labels if available"""
        # Look for label files (CSV or JSON)
        label_files = list(self.data_dir.glob("*label*.csv")) + \
                     list(self.data_dir.glob("*label*.json")) + \
                     list(self.data_dir.glob("*anomaly*.csv")) + \
                     list(self.data_dir.glob("*anomaly*.json"))
        
        for label_file in label_files:
            try:
                if label_file.suffix == '.csv':
                    labels_df = pd.read_csv(label_file)
                    self._process_label_dataframe(labels_df)
                elif label_file.suffix == '.json':
                    with open(label_file, 'r') as f:
                        labels = json.load(f)
                    self._process_label_dict(labels)
                    
                logger.info(f"Loaded anomaly labels from {label_file}")
                
            except Exception as e:
                logger.warning(f"Could not load labels from {label_file}: {e}")
    
    def _process_label_dataframe(self, df: pd.DataFrame):
        """Process anomaly labels from dataframe"""
        # Expected columns: channel, start, end
        if all(col in df.columns for col in ['chan_id', 'anomaly_sequences']):
            for _, row in df.iterrows():
                channel = row['chan_id']
                if channel in self.telemetry_data:
                    # Parse anomaly sequences
                    sequences = eval(row['anomaly_sequences']) if isinstance(row['anomaly_sequences'], str) else row['anomaly_sequences']
                    self.telemetry_data[channel].anomaly_sequences = sequences
    
    def _process_label_dict(self, labels: Dict):
        """Process anomaly labels from dictionary"""
        for channel, anomalies in labels.items():
            if channel in self.telemetry_data:
                self.telemetry_data[channel].anomaly_sequences = anomalies
    
    def _generate_timestamps(self, n_samples: int) -> np.ndarray:
        """Generate synthetic timestamps for data"""
        # Assume 1-minute sampling interval
        start_time = datetime.now() - timedelta(minutes=n_samples)
        timestamps = pd.date_range(start=start_time, periods=n_samples, freq='1min')
        return timestamps.values
    
    def _normalize_data(self):
        """Normalize telemetry data using specified scaler"""
        logger.info(f"Normalizing data using {self.scaler_type} scaler")
        
        for channel_name, telemetry in tqdm(self.telemetry_data.items(), 
                                           desc="Normalizing channels",
                                           disable=not self.verbose):
            # Select scaler
            if self.scaler_type == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaler type: {self.scaler_type}, using MinMaxScaler")
                scaler = MinMaxScaler()
            
            # Fit and transform data
            original_shape = telemetry.data.shape
            if telemetry.data.ndim == 1:
                telemetry.data = telemetry.data.reshape(-1, 1)
            
            telemetry.data = scaler.fit_transform(telemetry.data)
            
            # Store scaler for inverse transform
            self.scalers[channel_name] = scaler
            
            # Restore original shape if needed
            if original_shape != telemetry.data.shape and len(original_shape) == 1:
                telemetry.data = telemetry.data.flatten()
    
    def inverse_transform(self, channel_name: str, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data"""
        if channel_name in self.scalers:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return self.scalers[channel_name].inverse_transform(data)
        return data
    
    def get_channel_data(self, channel_name: str) -> Optional[TelemetryData]:
        """Get data for specific channel"""
        return self.telemetry_data.get(channel_name)
    
    def get_windowed_data(self, 
                         channel_name: str,
                         window_size: int = 100,
                         stride: int = 10) -> Generator[np.ndarray, None, None]:
        """
        Generate windowed data for time series processing
        
        Args:
            channel_name: Channel to process
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Yields:
            Window of data
        """
        if channel_name not in self.telemetry_data:
            logger.error(f"Channel {channel_name} not found")
            return
        
        data = self.telemetry_data[channel_name].data
        
        for i in range(0, len(data) - window_size + 1, stride):
            yield data[i:i + window_size]
    
    def get_train_test_split(self, 
                            channel_name: str,
                            test_size: float = 0.2,
                            sequential: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train and test sets
        
        Args:
            channel_name: Channel to split
            test_size: Proportion of data for testing
            sequential: If True, split sequentially; if False, random split
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if channel_name not in self.telemetry_data:
            raise ValueError(f"Channel {channel_name} not found")
        
        data = self.telemetry_data[channel_name].data
        n_samples = len(data)
        
        if sequential:
            # Sequential split (common for time series)
            split_idx = int(n_samples * (1 - test_size))
            train_data = data[:split_idx]
            test_data = data[split_idx:]
        else:
            # Random split
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * (1 - test_size))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            train_data = data[train_indices]
            test_data = data[test_indices]
        
        return train_data, test_data
    
    def create_sequences(self, 
                        data: np.ndarray,
                        sequence_length: int,
                        prediction_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            prediction_length: Length of prediction sequences
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_length + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length:i + sequence_length + prediction_length])
        
        return np.array(X), np.array(y)
    
    def _save_to_cache(self):
        """Save processed data to cache"""
        cache_file = self.cache_dir / f"{self.spacecraft}_telemetry.pkl"
        scaler_file = self.cache_dir / f"{self.spacecraft}_scalers.pkl"
        
        try:
            # Save telemetry data
            with open(cache_file, 'wb') as f:
                pickle.dump(self.telemetry_data, f)
            
            # Save scalers
            if self.scalers:
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers, f)
            
            logger.info(f"Cached processed data to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _load_from_cache(self) -> Optional[Dict[str, TelemetryData]]:
        """Load processed data from cache"""
        cache_file = self.cache_dir / f"{self.spacecraft}_telemetry.pkl"
        scaler_file = self.cache_dir / f"{self.spacecraft}_scalers.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    telemetry_data = pickle.load(f)
                
                # Load scalers if they exist
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers = pickle.load(f)
                
                logger.info(f"Loaded cached data from {cache_file}")
                return telemetry_data
                
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
                return None
        
        return None
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all loaded channels"""
        stats = {}
        
        for channel_name, telemetry in self.telemetry_data.items():
            data = telemetry.data
            stats[channel_name] = {
                'shape': data.shape,
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'n_samples': len(data),
                'n_channels': telemetry.n_channels,
                'has_anomalies': telemetry.anomaly_sequences is not None,
                'n_anomaly_sequences': len(telemetry.anomaly_sequences) if telemetry.anomaly_sequences else 0
            }
        
        return stats
    
    def print_summary(self):
        """Print summary of loaded data"""
        print("\n" + "="*60)
        print(f"Data Summary for {self.spacecraft.upper()}")
        print("="*60)
        
        if not self.telemetry_data:
            print("No data loaded yet. Call load_all_data() first.")
            return
        
        print(f"Total channels loaded: {len(self.telemetry_data)}")
        print(f"Normalization: {self.normalize} ({self.scaler_type if self.normalize else 'None'})")
        
        stats = self.get_statistics()
        
        # Create summary table
        from tabulate import tabulate
        
        table_data = []
        for channel, stat in stats.items():
            table_data.append([
                channel[:30],  # Truncate long names
                stat['shape'],
                f"{stat['mean']:.4f}",
                f"{stat['std']:.4f}",
                stat['n_anomaly_sequences']
            ])
        
        headers = ["Channel", "Shape", "Mean", "Std", "Anomalies"]
        print("\n" + tabulate(table_data[:10], headers=headers, tablefmt="grid"))
        
        if len(table_data) > 10:
            print(f"... and {len(table_data) - 10} more channels")
        
        print("\n" + "="*60)


# Convenience function for quick data loading
def load_telemetry_data(spacecraft: str = "smap", 
                       normalize: bool = True,
                       verbose: bool = True) -> Dict[str, TelemetryData]:
    """
    Quick function to load telemetry data
    
    Args:
        spacecraft: 'smap' or 'msl'
        normalize: Whether to normalize data
        verbose: Print progress
        
    Returns:
        Dictionary of telemetry data
    """
    loader = DataLoader(spacecraft=spacecraft, normalize=normalize, verbose=verbose)
    return loader.load_all_data()


if __name__ == "__main__":
    # Test data loading
    print("\n" + "="*60)
    print("Testing Data Loader Module")
    print("="*60)
    
    # Test SMAP data loading
    print("\nLoading SMAP data...")
    smap_loader = DataLoader(spacecraft="smap", normalize=True)
    smap_data = smap_loader.load_all_data()
    smap_loader.print_summary()
    
    # Test MSL data loading
    print("\nLoading MSL data...")
    msl_loader = DataLoader(spacecraft="msl", normalize=True)
    msl_data = msl_loader.load_all_data()
    msl_loader.print_summary()
    
    # Test windowing
    if smap_data:
        channel = list(smap_data.keys())[0]
        print(f"\nTesting windowing for channel: {channel}")
        windows = list(smap_loader.get_windowed_data(channel, window_size=100, stride=50))
        print(f"Generated {len(windows)} windows")
        if windows:
            print(f"Window shape: {windows[0].shape}")
