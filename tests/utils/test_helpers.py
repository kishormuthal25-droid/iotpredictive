"""
Test utilities and helper functions for IoT Predictive Maintenance System tests
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import json
from pathlib import Path

class TestDataGenerator:
    """Generate test data for various components"""

    @staticmethod
    def generate_sensor_timeseries(
        sensor_id: str,
        duration_hours: int = 24,
        frequency: str = '1min',
        anomaly_probability: float = 0.05,
        base_value: float = 50.0,
        noise_std: float = 5.0
    ) -> pd.DataFrame:
        """Generate synthetic sensor time series data"""

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        timestamps = pd.date_range(start_time, end_time, freq=frequency)

        # Generate base signal with trend and seasonality
        n_points = len(timestamps)
        trend = np.linspace(0, 2, n_points)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 60))  # Daily cycle
        noise = np.random.normal(0, noise_std, n_points)

        values = base_value + trend + seasonal + noise

        # Add anomalies
        anomaly_mask = np.random.random(n_points) < anomaly_probability
        anomaly_values = values.copy()
        anomaly_values[anomaly_mask] += np.random.normal(20, 10, np.sum(anomaly_mask))

        return pd.DataFrame({
            'timestamp': timestamps,
            'sensor_id': sensor_id,
            'value': anomaly_values,
            'is_anomaly': anomaly_mask
        })

    @staticmethod
    def generate_nasa_telemetry_batch(
        spacecraft: str = 'MSL',
        n_sensors: int = 55,
        n_timestamps: int = 1000
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate NASA telemetry batch data"""

        if spacecraft == 'MSL':
            sensor_ids = [f'MSL_{i}' for i in range(25, 25 + n_sensors)]
        else:  # SMAP
            sensor_ids = [f'SMAP_{i:02d}' for i in range(n_sensors)]

        # Generate correlated sensor data
        base_data = np.random.randn(n_timestamps, n_sensors)

        # Add correlations between sensors
        correlation_matrix = np.random.uniform(0.1, 0.9, (n_sensors, n_sensors))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)

        L = np.linalg.cholesky(correlation_matrix)
        correlated_data = base_data @ L.T

        # Normalize to reasonable ranges
        normalized_data = []
        for i in range(n_sensors):
            sensor_data = correlated_data[:, i]
            # Different sensors have different typical ranges
            if 'temperature' in sensor_ids[i].lower() or 'temp' in sensor_ids[i].lower():
                sensor_data = 20 + sensor_data * 15  # Temperature range
            elif 'voltage' in sensor_ids[i].lower() or 'volt' in sensor_ids[i].lower():
                sensor_data = 28 + sensor_data * 2   # Voltage range
            elif 'current' in sensor_ids[i].lower():
                sensor_data = 8 + np.abs(sensor_data) * 4  # Current range
            else:
                sensor_data = 50 + sensor_data * 20  # General range

            normalized_data.append(sensor_data)

        return np.array(normalized_data).T, sensor_ids

class TestFileManager:
    """Manage temporary test files and directories"""

    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []

    def create_temp_file(self, content: str, suffix: str = '.txt') -> str:
        """Create a temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name

    def create_temp_dir(self) -> str:
        """Create a temporary directory"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up all temporary files and directories"""
        import shutil

        for file_path in self.temp_files:
            try:
                Path(file_path).unlink()
            except FileNotFoundError:
                pass

        for dir_path in self.temp_dirs:
            try:
                shutil.rmtree(dir_path)
            except FileNotFoundError:
                pass

        self.temp_files.clear()
        self.temp_dirs.clear()

class MockResponseBuilder:
    """Build mock responses for testing API endpoints"""

    @staticmethod
    def build_sensor_response(
        sensor_id: str,
        status: str = 'normal',
        value: float = 50.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Build a mock sensor response"""
        if timestamp is None:
            timestamp = datetime.now()

        return {
            'sensor_id': sensor_id,
            'timestamp': timestamp.isoformat(),
            'value': value,
            'status': status,
            'metadata': {
                'equipment_id': f"{sensor_id.split('_')[0]}-PWR-001",
                'subsystem': 'POWER',
                'location': 'Mars Surface' if 'MSL' in sensor_id else 'Satellite'
            }
        }

    @staticmethod
    def build_anomaly_response(
        sensor_id: str,
        anomaly_score: float = 0.95,
        threshold: float = 0.8,
        model_name: str = 'LSTM_Autoencoder'
    ) -> Dict[str, Any]:
        """Build a mock anomaly detection response"""
        return {
            'sensor_id': sensor_id,
            'timestamp': datetime.now().isoformat(),
            'anomaly_score': anomaly_score,
            'threshold': threshold,
            'is_anomaly': anomaly_score > threshold,
            'model_name': model_name,
            'confidence': min(anomaly_score * 1.1, 1.0),
            'severity': 'HIGH' if anomaly_score > 0.9 else 'MEDIUM'
        }

def assert_dataframe_structure(
    df: pd.DataFrame,
    expected_columns: List[str],
    min_rows: int = 1
):
    """Assert that a DataFrame has the expected structure"""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    missing_columns = set(expected_columns) - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"

def assert_model_output_shape(
    output: np.ndarray,
    expected_shape: Tuple[int, ...],
    allow_batch_dimension: bool = True
):
    """Assert that model output has the expected shape"""
    assert isinstance(output, np.ndarray), "Expected numpy array"

    if allow_batch_dimension and len(expected_shape) == len(output.shape) - 1:
        # Allow for batch dimension
        assert output.shape[1:] == expected_shape, f"Expected shape {expected_shape}, got {output.shape[1:]}"
    else:
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

def compare_model_outputs(
    output1: np.ndarray,
    output2: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """Compare two model outputs within tolerance"""
    if output1.shape != output2.shape:
        return False

    return np.allclose(output1, output2, atol=tolerance)