"""
NASA Telemanom Integration for Real-time Dashboard
Integrates trained NASA Telemanom models with the existing dashboard infrastructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import glob

# Import our NASA Telemanom implementation
from src.anomaly_detection.nasa_telemanom import NASATelemanom
from src.data_ingestion.equipment_mapper import EquipmentComponent

logger = logging.getLogger(__name__)


@dataclass
class TelemanoMResult:
    """Result from NASA Telemanom anomaly detection"""
    timestamp: datetime
    sensor_id: str
    equipment_id: str
    equipment_type: str
    subsystem: str

    # Anomaly metrics
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    threshold: float

    # Sensor details
    sensor_values: Dict[str, float]
    model_name: str = "NASA_Telemanom"


class TelemanoMIntegration:
    """
    Integration layer for NASA Telemanom models with existing dashboard
    Loads trained models and provides real-time anomaly detection
    """

    def __init__(self, models_dir: str = "data/models/telemanom", use_lazy_loading: bool = True):
        """Initialize Telemanom integration

        Args:
            models_dir: Directory containing trained Telemanom models
            use_lazy_loading: If True, use lazy loading for fast startup
        """
        self.models_dir = Path(models_dir)
        self.use_lazy_loading = use_lazy_loading

        if use_lazy_loading:
            # NEW: Use lazy model manager for fast startup
            from src.model_registry.lazy_model_manager import get_lazy_model_manager
            self.lazy_manager = get_lazy_model_manager()
            self.trained_models: Dict[str, any] = {}  # Will contain lazy proxies
            self.model_info: Dict[str, Dict] = {}

            # Fast initialization - metadata only
            self._initialize_lazy_models()
            logger.info(f"TelemanoM Integration initialized with {len(self.trained_models)} models (lazy loading)")
        else:
            # OLD: Eager loading (will be slow)
            self.lazy_manager = None
            self.trained_models: Dict[str, NASATelemanom] = {}
            self.model_info: Dict[str, Dict] = {}

            # Load all available trained models (SLOW)
            self._load_trained_models()
            logger.info(f"TelemanoM Integration initialized with {len(self.trained_models)} models (eager loading)")

    def _load_trained_models(self):
        """Load all trained NASA Telemanom models"""
        try:
            # Find all trained model files
            model_files = list(self.models_dir.glob("*.pkl"))

            for model_file in model_files:
                try:
                    # Load the model
                    model = NASATelemanom.load_model(str(model_file))
                    sensor_id = model.sensor_id

                    # Store model and its info
                    self.trained_models[sensor_id] = model
                    self.model_info[sensor_id] = model.get_model_info()

                    logger.info(f"Loaded model for {sensor_id}, threshold: {model.error_threshold:.4f}")

                except Exception as e:
                    logger.error(f"Failed to load model from {model_file}: {e}")

            if len(self.trained_models) == 0:
                logger.warning("No trained Telemanom models found!")
            else:
                logger.info(f"Successfully loaded {len(self.trained_models)} Telemanom models")

        except Exception as e:
            logger.error(f"Error loading trained models: {e}")

    def _initialize_lazy_models(self):
        """Initialize lazy models (fast - metadata only)"""
        try:
            available_sensors = self.lazy_manager.get_available_sensors()

            for sensor_id in available_sensors:
                # Create lazy proxy (no actual model loading)
                proxy = self.lazy_manager.get_model(sensor_id)
                if proxy:
                    self.trained_models[sensor_id] = proxy
                    # Create lightweight model info
                    self.model_info[sensor_id] = {
                        'sensor_id': sensor_id,
                        'model_path': str(proxy.metadata.model_path),
                        'file_size': proxy.metadata.file_size,
                        'lazy_loaded': True
                    }

            logger.info(f"Initialized {len(self.trained_models)} lazy model proxies")

        except Exception as e:
            logger.error(f"Error initializing lazy models: {e}")
            # Fallback to eager loading if lazy loading fails
            logger.info("Falling back to eager model loading")
            self.use_lazy_loading = False
            self._load_trained_models()

    def detect_anomalies(self, sensor_data: Dict[str, np.ndarray],
                        timestamp: Optional[datetime] = None) -> List[TelemanoMResult]:
        """Detect anomalies using trained NASA Telemanom models

        Args:
            sensor_data: Dictionary mapping sensor IDs to data arrays
            timestamp: Timestamp for the detection

        Returns:
            List of anomaly detection results
        """
        if timestamp is None:
            timestamp = datetime.now()

        results = []

        for sensor_id, data in sensor_data.items():
            if sensor_id in self.trained_models:
                try:
                    result = self._detect_sensor_anomaly(sensor_id, data, timestamp)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {sensor_id}: {e}")

        return results

    def _detect_sensor_anomaly(self, sensor_id: str, data: np.ndarray,
                             timestamp: datetime) -> Optional[TelemanoMResult]:
        """Detect anomaly for a specific sensor

        Args:
            sensor_id: Sensor identifier
            data: Sensor data array
            timestamp: Timestamp for detection

        Returns:
            Anomaly detection result or None if error
        """
        try:
            if self.use_lazy_loading:
                # Use lazy loading manager
                anomaly_results = self.lazy_manager.predict_anomalies(sensor_id, data)
                if anomaly_results is None:
                    logger.warning(f"No model available for sensor {sensor_id}")
                    return None
            else:
                # Use eager loaded model
                model = self.trained_models[sensor_id]
                anomaly_results = model.predict_anomalies(data)

            # Get the latest anomaly information
            anomalies = anomaly_results['anomalies']
            scores = anomaly_results['scores']
            threshold = anomaly_results['threshold']

            if len(scores) > 0:
                # Get the maximum score as overall anomaly score
                max_score = float(np.max(scores))
                has_anomaly = bool(np.any(anomalies))

                # Calculate confidence (how far above/below threshold)
                confidence = min(max_score / threshold, 2.0) if threshold > 0 else 1.0

                # Extract equipment info from sensor ID
                equipment_info = self._parse_sensor_id(sensor_id)

                # Create sensor values dict (use last values)
                sensor_values = {sensor_id: float(data[-1]) if len(data) > 0 else 0.0}

                return TelemanoMResult(
                    timestamp=timestamp,
                    sensor_id=sensor_id,
                    equipment_id=equipment_info['equipment_id'],
                    equipment_type=equipment_info['equipment_type'],
                    subsystem=equipment_info['subsystem'],
                    anomaly_score=max_score,
                    is_anomaly=has_anomaly,
                    confidence=confidence,
                    threshold=float(threshold),
                    sensor_values=sensor_values,
                    model_name="NASA_Telemanom"
                )

        except Exception as e:
            logger.error(f"Error in anomaly detection for {sensor_id}: {e}")
            return None

    def _parse_sensor_id(self, sensor_id: str) -> Dict[str, str]:
        """Parse sensor ID to extract equipment information

        Args:
            sensor_id: Sensor ID like 'SMAP_00_Solar_Panel_Voltage'

        Returns:
            Dictionary with equipment information
        """
        try:
            parts = sensor_id.split('_')
            if len(parts) >= 3:
                spacecraft = parts[0]  # SMAP or MSL
                sensor_num = parts[1]  # 00, 01, etc.
                sensor_name = '_'.join(parts[2:])  # Solar_Panel_Voltage

                # Map to equipment based on sensor naming convention
                if 'Power' in sensor_name or 'Solar' in sensor_name or 'Battery' in sensor_name or 'Bus_Voltage' in sensor_name:
                    subsystem = "POWER"
                    equipment_id = f"{spacecraft}-PWR-001"
                elif 'Communication' in sensor_name or 'Antenna' in sensor_name:
                    subsystem = "COMMUNICATION"
                    equipment_id = f"{spacecraft}-COM-001"
                elif 'Temperature' in sensor_name or 'Thermal' in sensor_name:
                    subsystem = "THERMAL"
                    equipment_id = f"{spacecraft}-THM-001"
                elif 'Attitude' in sensor_name or 'Gyro' in sensor_name:
                    subsystem = "ATTITUDE"
                    equipment_id = f"{spacecraft}-ATT-001"
                elif 'Mobility' in sensor_name or 'Wheel' in sensor_name:
                    subsystem = "MOBILITY"
                    equipment_id = f"{spacecraft}-MOB-001"
                else:
                    subsystem = "UNKNOWN"
                    equipment_id = f"{spacecraft}-UNK-001"

                return {
                    'equipment_id': equipment_id,
                    'equipment_type': f"{spacecraft} {subsystem.title()} System",
                    'subsystem': subsystem
                }

        except Exception as e:
            logger.error(f"Error parsing sensor ID {sensor_id}: {e}")

        # Fallback
        return {
            'equipment_id': 'UNKNOWN-001',
            'equipment_type': 'Unknown System',
            'subsystem': 'UNKNOWN'
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models

        Returns:
            Dictionary with model status information
        """
        return {
            'total_models': len(self.trained_models),
            'loaded_models': list(self.trained_models.keys()),
            'model_info': self.model_info,
            'models_dir': str(self.models_dir)
        }

    def get_available_sensors(self) -> List[str]:
        """Get list of sensors with trained models

        Returns:
            List of sensor IDs that have trained models
        """
        return list(self.trained_models.keys())

    def has_model_for_sensor(self, sensor_id: str) -> bool:
        """Check if a trained model exists for a sensor

        Args:
            sensor_id: Sensor identifier

        Returns:
            True if model exists, False otherwise
        """
        return sensor_id in self.trained_models

    def get_sensor_model_info(self, sensor_id: str) -> Optional[Dict]:
        """Get model information for a specific sensor

        Args:
            sensor_id: Sensor identifier

        Returns:
            Model information dictionary or None if not found
        """
        return self.model_info.get(sensor_id)


# Global instance for dashboard integration
telemanom_integration = TelemanoMIntegration()