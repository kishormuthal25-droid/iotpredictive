"""
Unified Data Controller
Central data management layer for the IoT Predictive Maintenance System
Provides consistent data access across all dashboard modules
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_data_path
from src.data_ingestion.nasa_data_ingestion_service import NASADataIngestionService
from src.data_ingestion.database_manager import DatabaseManager, TelemetryData
from src.data_ingestion.equipment_mapper import equipment_mapper
# Removed MLFlow model_manager import from critical startup path
# Will be imported lazily when needed

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class SensorData:
    """Data structure for individual sensor information"""
    sensor_id: str
    equipment_id: str
    values: np.ndarray
    timestamps: np.ndarray
    anomaly_scores: np.ndarray = None
    status: str = "OPERATIONAL"
    last_updated: datetime = None

@dataclass
class EquipmentStatus:
    """Data structure for equipment-level status"""
    equipment_id: str
    name: str
    category: str
    sensors: List[str]
    overall_health: float
    status: str
    active_anomalies: int
    last_updated: datetime

class UnifiedDataController:
    """
    Central data controller providing unified access to all sensor and equipment data
    Ensures data consistency across all dashboard modules
    """

    def __init__(self):
        """Initialize the unified data controller"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Unified Data Controller...")

        # Core data structures
        self.sensor_data: Dict[str, SensorData] = {}
        self.equipment_status: Dict[str, EquipmentStatus] = {}
        self.equipment_mapping = self._load_equipment_mapping()

        # Initialize core services (fast, no MLFlow)
        self.db_manager = DatabaseManager()
        self.nasa_ingestion = NASADataIngestionService(self.db_manager)

        # MLFlow model manager - lazy initialization (not loaded at startup)
        self._model_manager = None
        self.anomaly_engine = None  # Deprecated - using model_manager instead

        # Data update tracking
        self.last_data_update = None
        self.update_lock = threading.Lock()
        self.cache_duration = timedelta(minutes=5)  # Cache data for 5 minutes

        # Background update thread
        self.update_thread = None
        self.stop_updating = False

        self.logger.info("Unified Data Controller initialized successfully (MLFlow-free startup)")

    @property
    def model_manager(self):
        """Lazy loading property for model manager (only loads when actually needed)"""
        if self._model_manager is None:
            try:
                # Import only when needed to avoid startup delays
                from src.model_registry.model_manager import get_model_manager
                self.logger.info("Loading model manager on-demand...")
                self._model_manager = get_model_manager()
                self.logger.info("Model manager loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model manager: {e}")
                # Return None - calling code should handle gracefully
                return None
        return self._model_manager

    def _load_equipment_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Load the 82-sensor to equipment mapping configuration"""
        mapping = {
            # Power Systems - A series sensors
            "SMAP-PWR-001": {
                "name": "SMAP Power System",
                "category": "POWER",
                "sensors": [f"A-{i}" for i in range(1, 10)],  # A-1 to A-9
                "description": "Satellite power management and battery systems"
            },
            "MSL-PWR-001": {
                "name": "MSL Power System",
                "category": "POWER",
                "sensors": [f"A-{i}" for i in range(10, 16)],  # A-10 to A-15 (if they exist)
                "description": "Mars rover power and energy systems"
            },

            # Communication Systems - B series sensors
            "SMAP-COM-001": {
                "name": "SMAP Communication",
                "category": "COMMUNICATION",
                "sensors": ["B-1"],
                "description": "Satellite communication and data transmission"
            },
            "MSL-COM-001": {
                "name": "MSL Communication",
                "category": "COMMUNICATION",
                "sensors": [f"B-{i}" for i in range(2, 6)],  # B-2 to B-5 (if they exist)
                "description": "Mars rover communication systems"
            },

            # Environmental/Thermal - C series sensors
            "SMAP-THM-001": {
                "name": "SMAP Thermal Control",
                "category": "THERMAL",
                "sensors": ["C-1"],
                "description": "Satellite thermal management systems"
            },
            "MSL-ENV-001": {
                "name": "MSL Environmental",
                "category": "ENVIRONMENTAL",
                "sensors": ["C-2"],
                "description": "Mars environmental monitoring systems"
            },

            # Science/Navigation - D series sensors
            "MSL-SCI-001": {
                "name": "MSL Science Instruments",
                "category": "SCIENCE",
                "sensors": [f"D-{i}" for i in range(1, 11)],  # D-1 to D-10
                "description": "Mars science and analysis instruments"
            },
            "MSL-NAV-001": {
                "name": "MSL Navigation",
                "category": "NAVIGATION",
                "sensors": [f"D-{i}" for i in range(11, 17)],  # D-11 to D-16
                "description": "Mars rover navigation and mobility"
            },

            # Mobility - Additional equipment category
            "MSL-MOB-001": {
                "name": "MSL Mobility Primary",
                "category": "MOBILITY",
                "sensors": [f"D-{i}" for i in [2, 3, 4, 5]],  # Selected D sensors for mobility
                "description": "Mars rover primary mobility systems"
            },
            "MSL-MOB-002": {
                "name": "MSL Mobility Secondary",
                "category": "MOBILITY",
                "sensors": [f"D-{i}" for i in [12, 13, 14, 15]],  # Selected D sensors for mobility
                "description": "Mars rover secondary mobility systems"
            },

            # Attitude Control - SMAP specific
            "SMAP-ATT-001": {
                "name": "SMAP Attitude Control",
                "category": "ATTITUDE",
                "sensors": [f"A-{i}" for i in [7, 8, 9]],  # Selected A sensors for attitude
                "description": "Satellite attitude control and orientation"
            },

            # Payload - SMAP specific
            "SMAP-PAY-001": {
                "name": "SMAP Payload",
                "category": "PAYLOAD",
                "sensors": [f"A-{i}" for i in [1, 2, 3]],  # Selected A sensors for payload
                "description": "Satellite payload and mission-specific instruments"
            }
        }

        return mapping

    def get_available_sensors(self) -> List[str]:
        """Get list of all available sensor IDs from the comprehensive dataset"""
        try:
            data_path = Path("data/raw/data/data/2018-05-19_15.00.10/smoothed_errors")
            if not data_path.exists():
                self.logger.warning(f"Comprehensive dataset path not found: {data_path}")
                return []

            sensor_files = list(data_path.glob("*.npy"))
            sensors = [f.stem for f in sensor_files]  # Remove .npy extension
            sensors.sort()  # Sort alphabetically: A-1, A-2, ..., D-16

            self.logger.info(f"Found {len(sensors)} sensors in comprehensive dataset")
            return sensors

        except Exception as e:
            self.logger.error(f"Error getting available sensors: {e}")
            return []

    def load_sensor_data(self, sensor_id: str) -> Optional[SensorData]:
        """Load data for a specific sensor from the comprehensive dataset"""
        try:
            data_path = Path(f"data/raw/data/data/2018-05-19_15.00.10/smoothed_errors/{sensor_id}.npy")

            if not data_path.exists():
                self.logger.warning(f"Sensor data file not found: {data_path}")
                return None

            # Load sensor values
            values = np.load(data_path)

            # Create timestamps (assuming 1 minute intervals)
            timestamps_pd = pd.date_range(
                end=datetime.now(),
                periods=len(values),
                freq='1min'
            )
            # Convert to Python datetime objects to avoid numpy.datetime64 isoformat issues
            timestamps = [ts.to_pydatetime() for ts in timestamps_pd]

            # Determine equipment assignment
            equipment_id = self._get_equipment_for_sensor(sensor_id)

            sensor_data = SensorData(
                sensor_id=sensor_id,
                equipment_id=equipment_id,
                values=values,
                timestamps=timestamps,
                last_updated=datetime.now()
            )

            return sensor_data

        except Exception as e:
            self.logger.error(f"Error loading sensor data for {sensor_id}: {e}")
            return None

    def detect_anomalies_for_sensor(self, sensor_id: str, sensor_data: Optional[SensorData] = None) -> np.ndarray:
        """
        Perform lazy anomaly detection for a specific sensor using MLFlow models

        Args:
            sensor_id: ID of the sensor
            sensor_data: Optional sensor data, if not provided will load it

        Returns:
            Array of anomaly scores
        """
        try:
            # Get sensor data if not provided
            if sensor_data is None:
                sensor_data = self.load_sensor_data(sensor_id)
                if sensor_data is None:
                    self.logger.warning(f"Could not load data for sensor {sensor_id}")
                    return np.array([])

            # Determine appropriate model name for this sensor
            equipment_id = sensor_data.equipment_id
            model_name = self._get_model_name_for_equipment(equipment_id, sensor_id)

            # Load model lazily via MLFlow model manager (if available)
            if self.model_manager is None:
                self.logger.info(f"Model manager not available, using fallback for sensor {sensor_id}")
                # Return simulated anomaly scores as fallback
                return np.random.uniform(0.1, 0.3, len(sensor_data.values))

            model = self.model_manager.get_model(model_name, timeout=30.0)
            if model is None:
                self.logger.warning(f"Could not load model {model_name} for sensor {sensor_id}")
                # Return low anomaly scores as fallback
                return np.random.uniform(0.1, 0.3, len(sensor_data.values))

            # Prepare data for model prediction
            # Most models expect windowed data - create simple sliding windows
            window_size = 50
            values = sensor_data.values

            if len(values) < window_size:
                # Pad with zeros if not enough data
                padded_values = np.pad(values, (window_size - len(values), 0), mode='constant')
                values = padded_values

            # Create windowed data for prediction
            anomaly_scores = []
            for i in range(len(values) - window_size + 1):
                window_data = values[i:i + window_size].reshape(1, -1)

                try:
                    # Different model types have different prediction interfaces
                    if hasattr(model, 'predict'):
                        if str(type(model)).find('tensorflow') != -1 or str(type(model)).find('keras') != -1:
                            # TensorFlow/Keras model
                            pred = model.predict(window_data.reshape(1, window_size, 1), verbose=0)
                            # For autoencoders, calculate reconstruction error
                            if pred.shape == window_data.shape:
                                score = np.mean(np.square(window_data - pred.flatten()))
                            else:
                                score = float(pred.flatten()[0])
                        else:
                            # Scikit-learn model
                            score = model.predict(window_data)[0]
                    else:
                        # MLFlow pyfunc model
                        pred = model.predict(pd.DataFrame(window_data))
                        score = float(pred.iloc[0] if hasattr(pred, 'iloc') else pred[0])

                    anomaly_scores.append(abs(float(score)))

                except Exception as e:
                    self.logger.debug(f"Prediction error for window {i}: {e}")
                    anomaly_scores.append(0.0)

            # Pad scores to match original data length
            if len(anomaly_scores) < len(sensor_data.values):
                padding_size = len(sensor_data.values) - len(anomaly_scores)
                anomaly_scores = [0.0] * padding_size + anomaly_scores

            return np.array(anomaly_scores)

        except Exception as e:
            self.logger.error(f"Error detecting anomalies for sensor {sensor_id}: {e}")
            return np.zeros(len(sensor_data.values) if sensor_data else 0)

    def _get_model_name_for_equipment(self, equipment_id: str, sensor_id: str) -> str:
        """
        Determine the best model name for a given equipment/sensor combination

        Args:
            equipment_id: Equipment identifier
            sensor_id: Sensor identifier

        Returns:
            Model name to use for anomaly detection
        """
        # Try equipment-specific model first
        if "MSL" in equipment_id:
            if "COM" in equipment_id:
                return "MSL-COM-001_anomaly_detector_best"
            elif "ENV" in equipment_id:
                return "MSL-ENV-001_anomaly_detector_best"
            elif "MOB" in equipment_id:
                return "MSL-MOB-001_anomaly_detector_best"
            elif "NAV" in equipment_id:
                return "MSL-NAV-001_anomaly_detector_best"
            elif "PWR" in equipment_id:
                return "MSL-PWR-001_anomaly_detector_best"
            else:
                return "MSL-COM-001_anomaly_detector_best"  # Default MSL
        elif "SMAP" in equipment_id:
            return f"SMAP_{sensor_id.replace('-', '_')}_anomaly_detector_best"
        else:
            # Fallback to generic model
            return "MSL-COM-001_anomaly_detector_best"

    def _get_equipment_for_sensor(self, sensor_id: str) -> str:
        """Determine which equipment a sensor belongs to"""
        for equipment_id, config in self.equipment_mapping.items():
            if sensor_id in config["sensors"]:
                return equipment_id

        # Default assignment based on sensor prefix
        if sensor_id.startswith("A-"):
            return "SMAP-PWR-001"
        elif sensor_id.startswith("B-"):
            return "SMAP-COM-001"
        elif sensor_id.startswith("C-"):
            return "SMAP-THM-001"
        elif sensor_id.startswith("D-"):
            return "MSL-SCI-001"
        else:
            return "UNKNOWN"

    def refresh_all_data(self, force_refresh: bool = False) -> bool:
        """Refresh all sensor and equipment data"""
        try:
            with self.update_lock:
                # Check if we need to refresh based on cache duration
                if not force_refresh and self.last_data_update:
                    time_since_update = datetime.now() - self.last_data_update
                    if time_since_update < self.cache_duration:
                        self.logger.debug("Using cached data, skipping refresh")
                        return True

                self.logger.info("Refreshing all sensor and equipment data...")

                # Get all available sensors
                available_sensors = self.get_available_sensors()

                # Load data for each sensor
                loaded_sensors = 0
                for sensor_id in available_sensors:
                    sensor_data = self.load_sensor_data(sensor_id)
                    if sensor_data:
                        self.sensor_data[sensor_id] = sensor_data
                        loaded_sensors += 1

                # Update equipment status
                self._update_equipment_status()

                # Update timestamp
                self.last_data_update = datetime.now()

                self.logger.info(f"Data refresh completed: {loaded_sensors}/{len(available_sensors)} sensors loaded")
                return True

        except Exception as e:
            self.logger.error(f"Error refreshing data: {e}")
            return False

    def _update_equipment_status(self):
        """Update equipment-level status based on sensor data"""
        try:
            for equipment_id, config in self.equipment_mapping.items():
                equipment_sensors = []
                total_health = 0
                sensor_count = 0
                active_anomalies = 0

                # Collect data from all sensors for this equipment
                for sensor_id in config["sensors"]:
                    if sensor_id in self.sensor_data:
                        sensor = self.sensor_data[sensor_id]
                        equipment_sensors.append(sensor_id)

                        # Calculate health score (simplified - based on data variance)
                        recent_values = sensor.values[-100:] if len(sensor.values) > 100 else sensor.values
                        health_score = max(0, 100 - (np.std(recent_values) * 10))
                        total_health += health_score
                        sensor_count += 1

                        # Count anomalies (simplified)
                        if sensor.anomaly_scores is not None:
                            active_anomalies += np.sum(sensor.anomaly_scores > 0.7)

                # Calculate overall equipment health
                overall_health = total_health / sensor_count if sensor_count > 0 else 0

                # Determine status
                if overall_health > 80:
                    status = "OPERATIONAL"
                elif overall_health > 60:
                    status = "WARNING"
                elif overall_health > 40:
                    status = "DEGRADED"
                else:
                    status = "CRITICAL"

                # Update equipment status
                self.equipment_status[equipment_id] = EquipmentStatus(
                    equipment_id=equipment_id,
                    name=config["name"],
                    category=config["category"],
                    sensors=equipment_sensors,
                    overall_health=overall_health,
                    status=status,
                    active_anomalies=active_anomalies,
                    last_updated=datetime.now()
                )

        except Exception as e:
            self.logger.error(f"Error updating equipment status: {e}")

    def get_sensor_data(self, sensor_id: str, time_range: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get data for a specific sensor"""
        try:
            if sensor_id not in self.sensor_data:
                # Try to load the sensor data
                sensor_data = self.load_sensor_data(sensor_id)
                if sensor_data:
                    self.sensor_data[sensor_id] = sensor_data
                else:
                    return None

            sensor = self.sensor_data[sensor_id]

            # Apply time range filter if specified
            if time_range:
                end_time = datetime.now()
                if time_range == "1hour":
                    start_time = end_time - timedelta(hours=1)
                elif time_range == "24hour":
                    start_time = end_time - timedelta(hours=24)
                elif time_range == "7day":
                    start_time = end_time - timedelta(days=7)
                else:
                    start_time = end_time - timedelta(hours=1)  # Default

                # Filter data by time range
                mask = (sensor.timestamps >= start_time) & (sensor.timestamps <= end_time)
                filtered_values = sensor.values[mask]
                filtered_timestamps = sensor.timestamps[mask]
            else:
                filtered_values = sensor.values
                filtered_timestamps = sensor.timestamps

            return {
                "sensor_id": sensor.sensor_id,
                "equipment_id": sensor.equipment_id,
                "values": filtered_values.tolist() if len(filtered_values) > 0 else [],
                "timestamps": [ts.isoformat() for ts in filtered_timestamps],
                "status": sensor.status,
                "last_updated": sensor.last_updated.isoformat() if sensor.last_updated else None,
                "summary": {
                    "count": len(filtered_values),
                    "mean": float(np.mean(filtered_values)) if len(filtered_values) > 0 else 0,
                    "std": float(np.std(filtered_values)) if len(filtered_values) > 0 else 0,
                    "min": float(np.min(filtered_values)) if len(filtered_values) > 0 else 0,
                    "max": float(np.max(filtered_values)) if len(filtered_values) > 0 else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting sensor data for {sensor_id}: {e}")
            return None

    def get_equipment_data(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive data for a specific equipment"""
        try:
            if equipment_id not in self.equipment_status:
                self.logger.warning(f"Equipment {equipment_id} not found")
                return None

            equipment = self.equipment_status[equipment_id]
            config = self.equipment_mapping[equipment_id]

            # Get data for all sensors in this equipment
            sensor_data = {}
            for sensor_id in equipment.sensors:
                data = self.get_sensor_data(sensor_id)
                if data:
                    sensor_data[sensor_id] = data

            return {
                "equipment_id": equipment.equipment_id,
                "name": equipment.name,
                "category": equipment.category,
                "description": config.get("description", ""),
                "overall_health": equipment.overall_health,
                "status": equipment.status,
                "active_anomalies": equipment.active_anomalies,
                "sensor_count": len(equipment.sensors),
                "sensors": equipment.sensors,
                "sensor_data": sensor_data,
                "last_updated": equipment.last_updated.isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting equipment data for {equipment_id}: {e}")
            return None

    def get_all_equipment_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all equipment"""
        try:
            # Ensure data is fresh
            self.refresh_all_data()

            equipment_data = {}
            for equipment_id in self.equipment_status:
                equipment_data[equipment_id] = self.get_equipment_data(equipment_id)

            return equipment_data

        except Exception as e:
            self.logger.error(f"Error getting all equipment status: {e}")
            return {}

    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide overview data"""
        try:
            # Ensure data is fresh
            self.refresh_all_data()

            # Calculate system-wide metrics
            total_sensors = len(self.sensor_data)
            total_equipment = len(self.equipment_status)

            operational_equipment = sum(1 for eq in self.equipment_status.values() if eq.status == "OPERATIONAL")
            warning_equipment = sum(1 for eq in self.equipment_status.values() if eq.status == "WARNING")
            degraded_equipment = sum(1 for eq in self.equipment_status.values() if eq.status == "DEGRADED")
            critical_equipment = sum(1 for eq in self.equipment_status.values() if eq.status == "CRITICAL")

            total_anomalies = sum(eq.active_anomalies for eq in self.equipment_status.values())

            # Calculate overall system health
            if total_equipment > 0:
                avg_health = sum(eq.overall_health for eq in self.equipment_status.values()) / total_equipment
            else:
                avg_health = 0

            return {
                "total_sensors": total_sensors,
                "total_equipment": total_equipment,
                "system_health": avg_health,
                "equipment_status": {
                    "operational": operational_equipment,
                    "warning": warning_equipment,
                    "degraded": degraded_equipment,
                    "critical": critical_equipment
                },
                "total_anomalies": total_anomalies,
                "last_updated": datetime.now().isoformat(),
                "data_source": "comprehensive_dataset",
                "dataset_info": {
                    "path": "data/raw/data/data/2018-05-19_15.00.10/",
                    "sensors_available": len(self.get_available_sensors()),
                    "cache_duration_minutes": int(self.cache_duration.total_seconds() / 60)
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting system overview: {e}")
            return {}

    def start_background_updates(self, update_interval: int = 300):
        """Start background data updates (every 5 minutes by default)"""
        if self.update_thread and self.update_thread.is_alive():
            self.logger.warning("Background updates already running")
            return

        def update_loop():
            while not self.stop_updating:
                try:
                    self.refresh_all_data()
                    time.sleep(update_interval)
                except Exception as e:
                    self.logger.error(f"Error in background update: {e}")
                    time.sleep(update_interval)

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        self.logger.info(f"Started background data updates (interval: {update_interval}s)")

    def stop_background_updates(self):
        """Stop background data updates"""
        self.stop_updating = True
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.logger.info("Stopped background data updates")

    def __del__(self):
        """Cleanup when controller is destroyed"""
        try:
            self.stop_background_updates()
        except:
            pass


# Global instance for easy access
_unified_controller = None

def get_unified_controller() -> UnifiedDataController:
    """Get global instance of unified data controller"""
    global _unified_controller
    if _unified_controller is None:
        _unified_controller = UnifiedDataController()
    return _unified_controller