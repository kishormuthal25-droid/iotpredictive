"""
MLFlow-Integrated NASA Telemanom Integration
Drop-in replacement for telemanom_integration.py with lazy model loading
Solves the 97-model startup hang issue
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Import original classes and model manager
from src.anomaly_detection.telemanom_integration import TelemanoMResult
from src.model_registry.model_manager import get_model_manager
from src.data_ingestion.equipment_mapper import EquipmentComponent

logger = logging.getLogger(__name__)


class MLFlowTelemanoMIntegration:
    """
    MLFlow-integrated NASA Telemanom integration with lazy model loading
    Drop-in replacement for TelemanoMIntegration that eliminates startup hang
    """

    def __init__(self, models_dir: str = "data/models/telemanom"):
        """Initialize MLFlow Telemanom integration

        Args:
            models_dir: Directory containing trained models (for fallback)
        """
        self.models_dir = Path(models_dir)

        # Get MLFlow model manager (shared instance)
        self.model_manager = get_model_manager()

        # Cache for equipment information
        self.equipment_cache: Dict[str, EquipmentComponent] = {}
        self._equipment_mapper = self._init_equipment_mapper()

        # Statistics
        self.detection_stats = {
            "total_detections": 0,
            "anomalies_detected": 0,
            "models_used": set(),
            "last_detection": None
        }

        logger.info("MLFlow Telemanom Integration initialized with lazy loading")
        logger.info(f"Available models will be loaded on-demand")

    def _init_equipment_mapper(self):
        """Initialize equipment mapper for sensor metadata"""
        try:
            from src.data_ingestion.equipment_mapper import IoTEquipmentMapper
            return IoTEquipmentMapper()
        except Exception as e:
            logger.warning(f"Could not initialize equipment mapper: {e}")
            return None

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models from MLFlow registry"""
        return self.model_manager.get_available_models()

    def preload_models(self, model_names: List[str] = None) -> Dict[str, bool]:
        """
        Preload specific models (optional warmup)

        Args:
            model_names: List of model names to preload, or None for default warmup models

        Returns:
            Dictionary mapping model names to success status
        """
        if model_names is None:
            # Use model manager's warmup functionality
            return self.model_manager.warmup_models()

        # Custom model preloading
        results = {}
        for model_name in model_names:
            try:
                model = self.model_manager.load_model(model_name)
                results[model_name] = model is not None
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
                results[model_name] = False

        return results

    def detect_anomalies(self, sensor_data: Dict[str, np.ndarray],
                        timestamp: Optional[datetime] = None) -> List[TelemanoMResult]:
        """Detect anomalies using trained NASA Telemanom models with lazy loading

        Args:
            sensor_data: Dictionary mapping sensor IDs to data arrays
            timestamp: Timestamp for the detection

        Returns:
            List of anomaly detection results
        """
        if timestamp is None:
            timestamp = datetime.now()

        results = []
        available_models = self.get_available_models()

        for sensor_id, data in sensor_data.items():
            if sensor_id in available_models:
                try:
                    result = self._detect_sensor_anomaly_lazy(sensor_id, data, timestamp)
                    if result:
                        results.append(result)
                        self.detection_stats["anomalies_detected"] += 1

                    self.detection_stats["models_used"].add(sensor_id)
                    self.detection_stats["total_detections"] += 1

                except Exception as e:
                    logger.error(f"Error detecting anomalies for {sensor_id}: {e}")
            else:
                logger.debug(f"No trained model available for sensor {sensor_id}")

        self.detection_stats["last_detection"] = timestamp
        return results

    def _detect_sensor_anomaly_lazy(self, sensor_id: str, data: np.ndarray,
                                   timestamp: datetime) -> Optional[TelemanoMResult]:
        """Detect anomaly for single sensor using lazy-loaded model"""

        # Lazy load the model
        model = self.model_manager.get_model(sensor_id, timeout=10.0)

        if model is None:
            logger.warning(f"Could not load model for sensor {sensor_id}")
            return None

        try:
            # Get equipment information (cached)
            equipment = self._get_equipment_info(sensor_id)

            # Prepare data for model
            if data.ndim == 1:
                # Add batch dimension if needed
                input_data = data.reshape(1, -1)
            else:
                input_data = data

            # Perform anomaly detection
            if hasattr(model, 'detect_anomalies'):
                # New API
                result = model.detect_anomalies(input_data)
                is_anomaly = result.get('is_anomaly', False)
                anomaly_score = result.get('score', 0.0)
                confidence = result.get('confidence', 0.0)
                threshold = result.get('threshold', model.error_threshold if hasattr(model, 'error_threshold') else 0.0)
            elif hasattr(model, 'predict'):
                # Standard prediction API
                predictions = model.predict(input_data)
                errors = np.abs(data - predictions.flatten()) if predictions is not None else np.zeros_like(data)
                threshold = model.error_threshold if hasattr(model, 'error_threshold') else np.percentile(errors, 95)
                anomaly_score = np.max(errors)
                is_anomaly = anomaly_score > threshold
                confidence = min(anomaly_score / threshold, 2.0) if threshold > 0 else 0.0
            else:
                logger.error(f"Model for {sensor_id} has no compatible prediction method")
                return None

            # Create result
            result = TelemanoMResult(
                timestamp=timestamp,
                sensor_id=sensor_id,
                equipment_id=equipment.equipment_id if equipment else "unknown",
                equipment_type=equipment.equipment_type if equipment else "unknown",
                subsystem=equipment.subsystem if equipment else "unknown",
                anomaly_score=float(anomaly_score),
                is_anomaly=bool(is_anomaly),
                confidence=float(confidence),
                threshold=float(threshold),
                sensor_values={sensor_id: float(np.mean(data))},  # Store mean value
                model_name=f"NASA_Telemanom_{sensor_id}"
            )

            if is_anomaly:
                logger.info(f"Anomaly detected for {sensor_id}: score={anomaly_score:.4f}, threshold={threshold:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error in anomaly detection for {sensor_id}: {e}")
            return None

    def _get_equipment_info(self, sensor_id: str) -> Optional[EquipmentComponent]:
        """Get equipment information with caching"""
        if sensor_id in self.equipment_cache:
            return self.equipment_cache[sensor_id]

        equipment = None
        if self._equipment_mapper:
            try:
                equipment = self._equipment_mapper.get_sensor_equipment(sensor_id)
            except Exception as e:
                logger.debug(f"Could not get equipment info for {sensor_id}: {e}")

        # Create default equipment info if mapper failed
        if equipment is None:
            if sensor_id.startswith("MSL_"):
                equipment = EquipmentComponent(
                    equipment_id=sensor_id,
                    equipment_type="Mars_Rover",
                    subsystem="MSL_Science_Lab",
                    sensors=[sensor_id],
                    criticality="HIGH"
                )
            elif sensor_id.startswith("SMAP_"):
                equipment = EquipmentComponent(
                    equipment_id=sensor_id,
                    equipment_type="Satellite",
                    subsystem="SMAP_Observatory",
                    sensors=[sensor_id],
                    criticality="HIGH"
                )
            else:
                equipment = EquipmentComponent(
                    equipment_id=sensor_id,
                    equipment_type="Unknown",
                    subsystem="Generic",
                    sensors=[sensor_id],
                    criticality="MEDIUM"
                )

        self.equipment_cache[sensor_id] = equipment
        return equipment

    def get_model_info(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """Get model information for a specific sensor"""
        available_models = self.get_available_models()
        if sensor_id in available_models:
            return available_models[sensor_id]
        return None

    def get_trained_sensors(self) -> List[str]:
        """Get list of sensors with trained models"""
        available_models = self.get_available_models()
        return list(available_models.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        stats = self.detection_stats.copy()
        stats["models_used"] = list(stats["models_used"])
        stats["available_models"] = len(self.get_available_models())
        stats["model_manager_stats"] = self.model_manager.get_stats()
        return stats

    def batch_detect_anomalies(self, sensor_data_batch: List[Dict[str, np.ndarray]],
                             timestamps: Optional[List[datetime]] = None) -> List[List[TelemanoMResult]]:
        """
        Batch anomaly detection for multiple time points

        Args:
            sensor_data_batch: List of sensor data dictionaries
            timestamps: Optional timestamps for each batch item

        Returns:
            List of detection results for each batch item
        """
        if timestamps is None:
            timestamps = [datetime.now() for _ in sensor_data_batch]

        results = []
        for i, sensor_data in enumerate(sensor_data_batch):
            timestamp = timestamps[i] if i < len(timestamps) else datetime.now()
            batch_results = self.detect_anomalies(sensor_data, timestamp)
            results.append(batch_results)

        return results

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up MLFlow Telemanom Integration...")
        self.equipment_cache.clear()
        # Model manager cleanup is handled globally


# Factory function to create the appropriate integration
def create_telemanom_integration(use_mlflow: bool = True) -> Any:
    """
    Factory function to create telemanom integration

    Args:
        use_mlflow: Whether to use MLFlow integration (recommended)

    Returns:
        Telemanom integration instance
    """
    if use_mlflow:
        logger.info("Creating MLFlow Telemanom Integration (lazy loading)")
        return MLFlowTelemanoMIntegration()
    else:
        logger.info("Creating original Telemanom Integration (loads all models)")
        from src.anomaly_detection.telemanom_integration import TelemanoMIntegration
        return TelemanoMIntegration()


# For backward compatibility
TelemanoMIntegration = MLFlowTelemanoMIntegration