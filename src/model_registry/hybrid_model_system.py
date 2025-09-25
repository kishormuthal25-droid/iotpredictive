"""
Hybrid Model System - Best of Both Worlds

Combines:
1. LazyModelManager (Primary) - Always works, fast, reliable
2. OptionalMLFlowService (Enhancement) - Advanced features when available
3. Direct Pickle Loading (Fallback) - Emergency backup

Architecture:
- Dashboard always works (via LazyModelManager)
- MLFlow enhances capabilities when available
- Automatic fallback hierarchy for maximum reliability
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading
import time
from enum import Enum

from src.model_registry.lazy_model_manager import get_lazy_model_manager
from src.model_registry.optional_mlflow_service import get_optional_mlflow_service

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    """Model loading source priority"""
    MLFLOW = "mlflow"          # Enhanced features
    LAZY_MANAGER = "lazy"      # Primary reliable system
    DIRECT_PICKLE = "pickle"   # Emergency fallback


class HybridModelSystem:
    """
    Hybrid model system that provides the best available model loading capability

    Features:
    - Primary: LazyModelManager (always works)
    - Enhanced: MLFlow (when available)
    - Fallback: Direct pickle loading (emergency)
    - Smart routing based on availability
    - Performance monitoring and optimization
    """

    def __init__(self,
                 enable_mlflow: bool = True,
                 mlflow_timeout: float = 5.0,
                 prefer_mlflow: bool = False):
        """
        Initialize hybrid model system

        Args:
            enable_mlflow: Enable MLFlow integration
            mlflow_timeout: Timeout for MLFlow operations
            prefer_mlflow: Prefer MLFlow over LazyManager when available
        """
        self.enable_mlflow = enable_mlflow
        self.mlflow_timeout = mlflow_timeout
        self.prefer_mlflow = prefer_mlflow

        # Initialize components
        self.lazy_manager = get_lazy_model_manager()
        self.mlflow_service = get_optional_mlflow_service() if enable_mlflow else None

        # Performance tracking
        self.performance_stats = {
            'predictions_total': 0,
            'predictions_by_source': {
                ModelSource.MLFLOW.value: 0,
                ModelSource.LAZY_MANAGER.value: 0,
                ModelSource.DIRECT_PICKLE.value: 0
            },
            'response_times': {
                ModelSource.MLFLOW.value: [],
                ModelSource.LAZY_MANAGER.value: [],
                ModelSource.DIRECT_PICKLE.value: []
            },
            'failures_by_source': {
                ModelSource.MLFLOW.value: 0,
                ModelSource.LAZY_MANAGER.value: 0,
                ModelSource.DIRECT_PICKLE.value: 0
            }
        }

        self.lock = threading.RLock()

        logger.info(f"HybridModelSystem initialized - MLFlow: {'enabled' if enable_mlflow else 'disabled'}")

    def predict_anomalies(self, sensor_id: str, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict anomalies using the best available model source

        Args:
            sensor_id: Sensor identifier
            data: Input sensor data

        Returns:
            Anomaly prediction results with source information
        """
        start_time = time.time()
        sources_tried = []
        last_error = None

        with self.lock:
            self.performance_stats['predictions_total'] += 1

        # Determine source priority order
        source_order = self._get_source_priority()

        for source in source_order:
            try:
                sources_tried.append(source)
                result = self._predict_with_source(sensor_id, data, source)

                if result is not None:
                    # Add metadata about which source was used
                    result['model_source'] = source.value
                    result['sources_tried'] = [s.value for s in sources_tried]

                    # Update performance stats
                    response_time = time.time() - start_time
                    with self.lock:
                        self.performance_stats['predictions_by_source'][source.value] += 1
                        self.performance_stats['response_times'][source.value].append(response_time)

                    logger.debug(f"Prediction successful via {source.value} for {sensor_id} in {response_time:.3f}s")
                    return result

            except Exception as e:
                last_error = e
                with self.lock:
                    self.performance_stats['failures_by_source'][source.value] += 1

                logger.warning(f"Prediction failed via {source.value} for {sensor_id}: {e}")
                continue

        # All sources failed
        logger.error(f"All prediction sources failed for {sensor_id}. Last error: {last_error}")
        return None

    def _get_source_priority(self) -> List[ModelSource]:
        """Determine source priority order based on availability and preference"""
        if self.prefer_mlflow and self.mlflow_service and self.mlflow_service.is_available():
            return [ModelSource.MLFLOW, ModelSource.LAZY_MANAGER, ModelSource.DIRECT_PICKLE]
        else:
            # Default: LazyManager first (most reliable), MLFlow as enhancement
            if self.mlflow_service and self.mlflow_service.is_available():
                return [ModelSource.LAZY_MANAGER, ModelSource.MLFLOW, ModelSource.DIRECT_PICKLE]
            else:
                return [ModelSource.LAZY_MANAGER, ModelSource.DIRECT_PICKLE]

    def _predict_with_source(self, sensor_id: str, data: np.ndarray, source: ModelSource) -> Optional[Dict[str, Any]]:
        """Predict using specific model source"""

        if source == ModelSource.MLFLOW:
            return self._predict_with_mlflow(sensor_id, data)
        elif source == ModelSource.LAZY_MANAGER:
            return self._predict_with_lazy_manager(sensor_id, data)
        elif source == ModelSource.DIRECT_PICKLE:
            return self._predict_with_direct_pickle(sensor_id, data)
        else:
            raise ValueError(f"Unknown model source: {source}")

    def _predict_with_mlflow(self, sensor_id: str, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Predict using MLFlow service"""
        if not self.mlflow_service or not self.mlflow_service.is_available():
            return None

        return self.mlflow_service.predict_anomalies(sensor_id, data)

    def _predict_with_lazy_manager(self, sensor_id: str, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Predict using LazyModelManager"""
        return self.lazy_manager.predict_anomalies(sensor_id, data)

    def _predict_with_direct_pickle(self, sensor_id: str, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Direct pickle loading as emergency fallback"""
        try:
            import pickle
            from pathlib import Path

            # Try to load model directly
            model_path = Path("data/models/telemanom") / f"{sensor_id}.pkl"

            if not model_path.exists():
                logger.warning(f"Direct pickle fallback: Model file not found for {sensor_id}")
                return None

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Simple prediction (this is sensor-specific)
            if hasattr(model, 'predict_anomalies'):
                return model.predict_anomalies(data)
            elif hasattr(model, 'predict'):
                # Generic prediction - convert to anomaly format
                predictions = model.predict(data.reshape(1, -1) if len(data.shape) == 1 else data)
                return {
                    'anomalies': predictions > 0.5,  # Simple threshold
                    'scores': predictions.flatten(),
                    'threshold': 0.5
                }
            else:
                logger.warning(f"Direct pickle: Model for {sensor_id} doesn't have prediction method")
                return None

        except Exception as e:
            logger.error(f"Direct pickle fallback failed for {sensor_id}: {e}")
            return None

    def get_available_sensors(self) -> List[str]:
        """Get list of available sensors from all sources"""
        sensors = set()

        # Always get from LazyManager (primary)
        sensors.update(self.lazy_manager.get_available_sensors())

        # Add from MLFlow if available
        if self.mlflow_service and self.mlflow_service.is_available():
            mlflow_manager = self.mlflow_service.get_model_manager()
            if mlflow_manager and hasattr(mlflow_manager, 'get_available_sensors'):
                sensors.update(mlflow_manager.get_available_sensors())

        return sorted(list(sensors))

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'primary_system': 'LazyModelManager',
            'lazy_manager_status': self.lazy_manager.get_stats(),
            'mlflow_enabled': self.enable_mlflow,
            'mlflow_status': None,
            'performance_stats': self._get_performance_summary(),
            'available_sensors': len(self.get_available_sensors()),
            'total_predictions': self.performance_stats['predictions_total']
        }

        if self.mlflow_service:
            status['mlflow_status'] = self.mlflow_service.get_status()

        return status

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {}

        for source, times in self.performance_stats['response_times'].items():
            if times:
                summary[source] = {
                    'predictions': self.performance_stats['predictions_by_source'][source],
                    'failures': self.performance_stats['failures_by_source'][source],
                    'avg_response_time': sum(times) / len(times),
                    'success_rate': (
                        self.performance_stats['predictions_by_source'][source] /
                        max(1, self.performance_stats['predictions_by_source'][source] +
                            self.performance_stats['failures_by_source'][source])
                    ) * 100
                }

        return summary

    def force_mlflow_reconnection(self) -> bool:
        """Force MLFlow reconnection"""
        if self.mlflow_service:
            return self.mlflow_service.force_reconnect()
        return False

    def enable_mlflow_integration(self):
        """Enable MLFlow integration"""
        if not self.enable_mlflow:
            self.enable_mlflow = True
            self.mlflow_service = get_optional_mlflow_service()
            logger.info("MLFlow integration enabled")

    def disable_mlflow_integration(self):
        """Disable MLFlow integration"""
        if self.enable_mlflow:
            self.enable_mlflow = False
            if self.mlflow_service:
                self.mlflow_service.shutdown()
                self.mlflow_service = None
            logger.info("MLFlow integration disabled")


# Global singleton instance
_hybrid_model_system = None


def get_hybrid_model_system(
    enable_mlflow: bool = True,
    mlflow_timeout: float = 5.0,
    prefer_mlflow: bool = False
) -> HybridModelSystem:
    """
    Get or create global HybridModelSystem instance

    Args:
        enable_mlflow: Enable MLFlow integration
        mlflow_timeout: Timeout for MLFlow operations
        prefer_mlflow: Prefer MLFlow over LazyManager when available

    Returns:
        HybridModelSystem instance
    """
    global _hybrid_model_system

    if _hybrid_model_system is None:
        _hybrid_model_system = HybridModelSystem(
            enable_mlflow=enable_mlflow,
            mlflow_timeout=mlflow_timeout,
            prefer_mlflow=prefer_mlflow
        )

    return _hybrid_model_system


def reset_hybrid_model_system():
    """Reset the global instance (useful for testing)"""
    global _hybrid_model_system

    if _hybrid_model_system:
        if _hybrid_model_system.mlflow_service:
            _hybrid_model_system.mlflow_service.shutdown()

    _hybrid_model_system = None