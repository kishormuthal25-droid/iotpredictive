"""
MLFlow-Enhanced Dashboard Model Manager
Provides lazy loading interface for dashboard components while preserving all original functionality
Drop-in replacement for PretrainedModelManager with MLFlow integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.model_registry.model_manager import get_model_manager
from src.data_ingestion.equipment_mapper import equipment_mapper
from config.settings import get_data_path

logger = logging.getLogger(__name__)


class MLFlowDashboardModelManager:
    """
    MLFlow-aware model manager for dashboard components
    Provides same interface as PretrainedModelManager but with lazy loading
    Integrates with the MLFlow model registry for optimal memory usage
    """

    def __init__(self):
        """Initialize the MLFlow dashboard model manager with lazy loading"""
        self.models_dir = get_data_path('models')

        # Get reference to global MLFlow model manager
        self.mlflow_manager = get_model_manager()

        # Dashboard-specific caches and state
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.inference_stats = {
            'total_inferences': 0,
            'model_usage': {},
            'avg_inference_time': 0.0
        }

        # Lazy loading state - models are discovered but not loaded
        self._available_models = {}
        self._metadata_cache = {}

        # Discover available models without loading them
        self._discover_models()

        logger.info(f"MLFlow Dashboard Model Manager initialized")
        logger.info(f"Available models: {len(self._available_models)} (lazy loading enabled)")

    def _discover_models(self):
        """Discover available models without loading them"""
        try:
            # Get available models from MLFlow registry
            mlflow_models = self.mlflow_manager.get_available_models()
            self._available_models.update(mlflow_models)

            # Also discover traditional models for compatibility
            self._discover_traditional_models()

            logger.info(f"Discovered {len(self._available_models)} models for lazy loading")

        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            # Fallback to traditional discovery
            self._discover_traditional_models()

    def _discover_traditional_models(self):
        """Discover traditional model files without loading them"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return

        # Look for .h5 model files and corresponding metadata
        model_files = list(self.models_dir.glob("*_anomaly_detector_best.h5"))
        quick_model_files = list(self.models_dir.glob("*_quick_anomaly_detector.h5"))

        # Look for NASA Telemanom models (.pkl files in telemanom/ subdirectory)
        telemanom_dir = self.models_dir / "telemanom"
        telemanom_files = []
        if telemanom_dir.exists():
            telemanom_files = list(telemanom_dir.glob("*.pkl"))

        logger.debug(f"Found {len(model_files)} best models, {len(quick_model_files)} quick models, and {len(telemanom_files)} NASA Telemanom models")

        # Register NASA Telemanom models (highest priority)
        for model_path in telemanom_files:
            model_id = model_path.stem
            self._available_models[model_id] = {
                'name': model_id,
                'version': 'local',
                'stage': 'Production',
                'model_path': str(model_path),
                'source': 'local_telemanom',
                'file_type': '.pkl',
                'model_type': 'nasa_telemanom'
            }

        # Register best models first
        for model_path in model_files:
            equipment_id = model_path.stem.replace("_anomaly_detector_best", "")
            if equipment_id not in self._available_models:
                self._available_models[equipment_id] = {
                    'name': equipment_id,
                    'version': 'local',
                    'stage': 'Production',
                    'model_path': str(model_path),
                    'source': 'local_best',
                    'file_type': '.h5',
                    'model_type': 'lstm_autoencoder'
                }

        # Register quick models as fallback
        for model_path in quick_model_files:
            equipment_id = model_path.stem.replace("_quick_anomaly_detector", "")
            if equipment_id not in self._available_models:
                self._available_models[equipment_id] = {
                    'name': equipment_id,
                    'version': 'local',
                    'stage': 'Production',
                    'model_path': str(model_path),
                    'source': 'local_quick',
                    'file_type': '.h5',
                    'model_type': 'lstm_autoencoder_quick'
                }

    def get_model_for_equipment(self, equipment_id: str) -> Optional[Any]:
        """
        Get model for equipment with lazy loading
        This is the main method used by dashboard components

        Args:
            equipment_id: Equipment identifier

        Returns:
            Loaded model or None if not available
        """
        # Check if already loaded in dashboard cache
        if equipment_id in self.loaded_models:
            return self.loaded_models[equipment_id]['model']

        # Check if model is available
        if equipment_id not in self._available_models:
            logger.debug(f"No model available for equipment {equipment_id}")
            return None

        # Lazy load the model
        return self._load_model_lazy(equipment_id)

    def _load_model_lazy(self, equipment_id: str) -> Optional[Any]:
        """Lazy load a single model"""
        start_time = time.time()

        try:
            model_info = self._available_models[equipment_id]

            # Try MLFlow loading first
            if model_info['source'].startswith('local'):
                # Load traditional model
                model = self._load_traditional_model(model_info)
            else:
                # Use MLFlow manager
                model = self.mlflow_manager.get_model(equipment_id)

            if model is not None:
                # Cache in dashboard manager
                loading_time = time.time() - start_time
                self.loaded_models[equipment_id] = {
                    'model': model,
                    'metadata': self._get_model_metadata(equipment_id),
                    'loaded_at': datetime.now(),
                    'loading_time': loading_time,
                    'inference_count': 0
                }

                # Update stats
                self.inference_stats['model_usage'][equipment_id] = {
                    'loading_time': loading_time,
                    'inference_count': 0,
                    'last_used': datetime.now()
                }

                logger.info(f"Lazy loaded model for {equipment_id} in {loading_time:.2f}s")

            return model

        except Exception as e:
            logger.error(f"Failed to lazy load model for {equipment_id}: {e}")
            return None

    def _load_traditional_model(self, model_info: Dict[str, Any]) -> Optional[Any]:
        """Load traditional model file"""
        model_path = Path(model_info['model_path'])
        file_type = model_info['file_type']
        model_type = model_info['model_type']

        try:
            if file_type == '.pkl' and model_type == 'nasa_telemanom':
                # NASA Telemanom model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Create NASA Telemanom wrapper
                from src.anomaly_detection.nasa_telemanom import NASATelemanom
                model = NASATelemanom(model_path.stem)
                model.model = model_data.get('model')
                model.error_threshold = model_data.get('error_threshold', 3.0)

                return model

            elif file_type == '.h5':
                # TensorFlow/Keras model
                model = load_model(str(model_path), compile=False)
                return model

            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to load traditional model {model_path}: {e}")
            return None

    def _get_model_metadata(self, equipment_id: str) -> Dict[str, Any]:
        """Get metadata for a model (cached)"""
        if equipment_id in self._metadata_cache:
            return self._metadata_cache[equipment_id]

        # Try to load from file
        metadata_path = self.models_dir / f"{equipment_id}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self._metadata_cache[equipment_id] = metadata
                    return metadata
            except Exception as e:
                logger.debug(f"Could not load metadata for {equipment_id}: {e}")

        # Create default metadata
        equipment_info = equipment_mapper.get_equipment_by_id(equipment_id)
        metadata = {
            'equipment_id': equipment_id,
            'equipment_type': equipment_info.equipment_type if equipment_info else 'Unknown',
            'subsystem': equipment_info.subsystem if equipment_info else 'Unknown',
            'sensors': equipment_info.sensors if equipment_info else [equipment_id],
            'model_type': self._available_models[equipment_id]['model_type'],
            'threshold': 3.0,
            'window_size': 100,
            'features': len(equipment_info.sensors) if equipment_info else 1
        }

        self._metadata_cache[equipment_id] = metadata
        return metadata

    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self._available_models.keys())

    def get_model_metadata(self, equipment_id: str) -> Dict[str, Any]:
        """Get model metadata without loading the model"""
        return self._get_model_metadata(equipment_id)

    def get_model_info(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get model info for compatibility with existing dashboard code"""
        if equipment_id in self._available_models:
            model_info = self._available_models[equipment_id].copy()
            model_info.update(self._get_model_metadata(equipment_id))
            return model_info
        return None

    def predict_anomaly(self, equipment_id: str, sensor_data: np.ndarray) -> Dict[str, Any]:
        """
        Predict anomaly for equipment data
        Compatible with original PretrainedModelManager interface

        Args:
            equipment_id: Equipment identifier
            sensor_data: Sensor data array

        Returns:
            Dictionary with anomaly prediction results
        """
        start_time = time.time()

        # Get model (lazy loading)
        model = self.get_model_for_equipment(equipment_id)
        if model is None:
            return {
                'equipment_id': equipment_id,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'threshold': 3.0,
                'error': 'Model not available'
            }

        try:
            # Get metadata for prediction parameters
            metadata = self._get_model_metadata(equipment_id)

            # Prepare data
            if sensor_data.ndim == 1:
                # Ensure we have the right window size
                window_size = metadata.get('window_size', 100)
                if len(sensor_data) < window_size:
                    # Pad with zeros if needed
                    padded_data = np.zeros(window_size)
                    padded_data[-len(sensor_data):] = sensor_data
                    sensor_data = padded_data
                elif len(sensor_data) > window_size:
                    # Take last window_size points
                    sensor_data = sensor_data[-window_size:]

                # Reshape for model input
                input_data = sensor_data.reshape(1, window_size, 1)
            else:
                input_data = sensor_data

            # Make prediction based on model type
            model_type = metadata.get('model_type', 'unknown')

            if model_type == 'nasa_telemanom':
                # NASA Telemanom prediction
                predictions = model.predict(input_data.flatten())
                if predictions is not None:
                    errors = np.abs(input_data.flatten() - predictions)
                    anomaly_score = np.mean(errors)
                    threshold = getattr(model, 'error_threshold', 3.0)
                    is_anomaly = anomaly_score > threshold
                    confidence = min(anomaly_score / threshold, 2.0) if threshold > 0 else 0.0
                else:
                    anomaly_score = 0.0
                    is_anomaly = False
                    confidence = 0.0
                    threshold = 3.0
            else:
                # LSTM Autoencoder prediction
                reconstructed = model.predict(input_data, verbose=0)
                mse = np.mean(np.square(input_data - reconstructed))
                threshold = metadata.get('threshold', 3.0)
                anomaly_score = float(mse)
                is_anomaly = mse > threshold
                confidence = min(mse / threshold, 2.0) if threshold > 0 else 0.0

            # Update statistics
            inference_time = time.time() - start_time
            self.inference_stats['total_inferences'] += 1

            if equipment_id in self.loaded_models:
                self.loaded_models[equipment_id]['inference_count'] += 1

            if equipment_id in self.inference_stats['model_usage']:
                usage_stats = self.inference_stats['model_usage'][equipment_id]
                usage_stats['inference_count'] += 1
                usage_stats['last_used'] = datetime.now()

                # Update average inference time
                total_time = usage_stats.get('total_inference_time', 0.0) + inference_time
                total_count = usage_stats['inference_count']
                usage_stats['avg_inference_time'] = total_time / total_count
                usage_stats['total_inference_time'] = total_time

            return {
                'equipment_id': equipment_id,
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'threshold': float(threshold),
                'inference_time': inference_time,
                'model_type': model_type
            }

        except Exception as e:
            logger.error(f"Error in anomaly prediction for {equipment_id}: {e}")
            return {
                'equipment_id': equipment_id,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'threshold': 3.0,
                'error': str(e)
            }

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        stats = self.inference_stats.copy()

        # Add MLFlow manager stats
        mlflow_stats = self.mlflow_manager.get_stats()
        stats['mlflow'] = mlflow_stats

        # Add loaded model count
        stats['loaded_models_count'] = len(self.loaded_models)
        stats['available_models_count'] = len(self._available_models)

        return stats

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models"""
        return {
            equipment_id: {
                'loaded_at': info['loaded_at'],
                'loading_time': info['loading_time'],
                'inference_count': info['inference_count']
            }
            for equipment_id, info in self.loaded_models.items()
        }

    def clear_model_cache(self):
        """Clear loaded model cache to free memory"""
        logger.info(f"Clearing model cache ({len(self.loaded_models)} models)")
        self.loaded_models.clear()

        # Also clear MLFlow manager cache
        self.mlflow_manager.cache.clear()

    def warmup_models(self, equipment_ids: List[str] = None) -> Dict[str, bool]:
        """
        Warmup specific models by loading them in advance

        Args:
            equipment_ids: List of equipment IDs to warmup, or None for default set

        Returns:
            Dictionary mapping equipment IDs to success status
        """
        if equipment_ids is None:
            # Default warmup set - high priority equipment
            equipment_ids = [
                'MSL-COM-001', 'MSL-ENV-001', 'MSL-MOB-001', 'MSL-NAV-001', 'MSL-POW-001',
                'SMAP-PWR-001', 'SMAP-COM-001', 'SMAP-ENV-001'
            ]
            # Filter to available models
            equipment_ids = [eid for eid in equipment_ids if eid in self._available_models]

        logger.info(f"Warming up {len(equipment_ids)} models...")
        results = {}

        for equipment_id in equipment_ids:
            try:
                model = self.get_model_for_equipment(equipment_id)
                results[equipment_id] = model is not None
            except Exception as e:
                logger.error(f"Failed to warmup model {equipment_id}: {e}")
                results[equipment_id] = False

        successful = sum(results.values())
        logger.info(f"Warmed up {successful}/{len(equipment_ids)} models")

        return results


# Global instance (singleton pattern)
_mlflow_dashboard_manager = None

def get_mlflow_dashboard_model_manager() -> MLFlowDashboardModelManager:
    """Get or create the global MLFlow dashboard model manager"""
    global _mlflow_dashboard_manager
    if _mlflow_dashboard_manager is None:
        _mlflow_dashboard_manager = MLFlowDashboardModelManager()
    return _mlflow_dashboard_manager

# For backward compatibility with existing dashboard code
pretrained_model_manager = get_mlflow_dashboard_model_manager()