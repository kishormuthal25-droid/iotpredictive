"""
Pytest configuration and shared fixtures for IoT Predictive Maintenance System tests
"""

import pytest
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path for tests"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_path(project_root_path):
    """Provide test data directory path"""
    return project_root_path / "tests" / "fixtures"

@pytest.fixture
def sample_sensor_data():
    """Generate sample sensor data for testing"""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=24),
        end=datetime.now(),
        freq='1min'
    )

    data = []
    for i, ts in enumerate(timestamps):
        data.append({
            'timestamp': ts,
            'sensor_id': f'MSL_{25 + (i % 55)}',
            'value': np.random.normal(50, 10),
            'equipment_id': f'MSL-PWR-{(i % 3) + 1:03d}',
            'subsystem': 'POWER',
            'location': 'Mars Surface'
        })

    return pd.DataFrame(data)

@pytest.fixture
def sample_anomaly_data():
    """Generate sample anomaly data for testing"""
    return {
        'sensor_id': 'MSL_25',
        'timestamp': datetime.now(),
        'value': 85.5,
        'anomaly_score': 0.95,
        'threshold': 0.8,
        'is_anomaly': True,
        'model_name': 'LSTM_Autoencoder'
    }

@pytest.fixture
def mock_model():
    """Create a mock ML model for testing"""
    model = Mock()
    model.predict.return_value = np.array([[0.1, 0.8, 0.05, 0.05]])
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    model.score.return_value = 0.95
    return model

@pytest.fixture
def mock_data_loader():
    """Create a mock data loader for testing"""
    loader = Mock()
    loader.load_data.return_value = (
        np.random.randn(1000, 50),  # Sample data
        np.random.randint(0, 2, 1000)  # Sample labels
    )
    return loader

@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
    return {
        'data': {
            'smap_path': 'data/raw/smap',
            'msl_path': 'data/raw/msl',
            'batch_size': 32
        },
        'models': {
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        },
        'dashboard': {
            'host': '127.0.0.1',
            'port': 8050,
            'debug': True
        }
    }

@pytest.fixture
def nasa_telemetry_sample():
    """Sample NASA telemetry data structure"""
    return {
        'MSL': {
            'sensors': [f'MSL_{i}' for i in range(25, 80)],
            'data': np.random.randn(100, 55),
            'timestamps': pd.date_range('2024-01-01', periods=100, freq='1min')
        },
        'SMAP': {
            'sensors': [f'SMAP_{i:02d}' for i in range(25)],
            'data': np.random.randn(100, 25),
            'timestamps': pd.date_range('2024-01-01', periods=100, freq='1min')
        }
    }

@pytest.fixture(autouse=True)
def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings during testing"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow