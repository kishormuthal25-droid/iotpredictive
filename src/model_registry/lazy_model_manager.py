"""
Lazy Model Manager - True lazy loading for NASA Telemanom models
Solves the 30+ second startup hang by loading models only when needed
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
import time
import weakref
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Lightweight model metadata without loading the actual model"""
    sensor_id: str
    model_path: Path
    file_size: int
    last_modified: datetime
    threshold: Optional[float] = None
    equipment_type: Optional[str] = None
    subsystem: Optional[str] = None

    @property
    def cache_key(self) -> str:
        """Generate unique cache key for this model"""
        return f"{self.sensor_id}_{self.file_size}_{self.last_modified.timestamp()}"


class LazyModelProxy:
    """
    Proxy for a NASA Telemanom model that loads only when needed
    """

    def __init__(self, metadata: ModelMetadata, manager: 'LazyModelManager'):
        self.metadata = metadata
        self.manager = manager
        self._model = None
        self._last_used = None
        self._loading = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded in memory"""
        return self._model is not None

    @property
    def sensor_id(self) -> str:
        return self.metadata.sensor_id

    def predict_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Predict anomalies using the model (loads on first use)

        Args:
            data: Input sensor data

        Returns:
            Dictionary with anomaly results
        """
        self._last_used = datetime.now()

        # Load model if not already loaded
        if self._model is None:
            self._load_model()

        # Use the loaded model
        try:
            return self._model.predict_anomalies(data)
        except Exception as e:
            logger.error(f"Error predicting with model {self.sensor_id}: {e}")
            # Try reloading the model once
            self._model = None
            self._load_model()
            return self._model.predict_anomalies(data)

    def _load_model(self):
        """Load the actual model (thread-safe)"""
        with self._loading:
            if self._model is not None:
                return  # Already loaded by another thread

            try:
                logger.info(f"Loading model for sensor {self.sensor_id}")
                start_time = time.time()

                # Import here to avoid circular imports
                from src.anomaly_detection.nasa_telemanom import NASATelemanom

                # Load the model
                self._model = NASATelemanom.load_model(str(self.metadata.model_path))

                load_time = time.time() - start_time
                logger.info(f"Loaded model {self.sensor_id} in {load_time:.3f}s")

                # Register with manager for lifecycle management
                self.manager._register_loaded_model(self)

            except Exception as e:
                logger.error(f"Failed to load model {self.sensor_id}: {e}")
                raise

    def unload(self):
        """Unload the model from memory"""
        if self._model is not None:
            logger.debug(f"Unloading model {self.sensor_id}")
            self._model = None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (loads model if needed)"""
        if self._model is None:
            self._load_model()
        return self._model.get_model_info()


class LazyModelManager:
    """
    Manages lazy loading of NASA Telemanom models
    Features:
    - Metadata-only scanning (fast startup)
    - On-demand model loading
    - Automatic model unloading after timeout
    - Usage tracking and statistics
    """

    def __init__(self, models_dir: str = "data/models/telemanom",
                 max_loaded_models: int = 10,
                 unload_timeout_minutes: int = 15):
        """
        Initialize lazy model manager

        Args:
            models_dir: Directory containing model files
            max_loaded_models: Maximum number of models to keep loaded
            unload_timeout_minutes: Unload models after this many minutes of inactivity
        """
        self.models_dir = Path(models_dir)
        self.max_loaded_models = max_loaded_models
        self.unload_timeout = timedelta(minutes=unload_timeout_minutes)

        # Model registry (lightweight metadata only)
        self.model_metadata: Dict[str, ModelMetadata] = {}
        self.model_proxies: Dict[str, LazyModelProxy] = {}

        # Loaded model tracking
        self._loaded_models: Dict[str, LazyModelProxy] = {}
        self._model_usage: Dict[str, int] = defaultdict(int)
        self._cleanup_lock = threading.Lock()

        # Statistics
        self.stats = {
            'models_discovered': 0,
            'models_loaded': 0,
            'total_predictions': 0,
            'cache_hits': 0,
            'load_times': [],
        }

        # Initialize
        self._discover_models()
        self._start_cleanup_thread()

        logger.info(f"LazyModelManager initialized with {len(self.model_metadata)} models")

    def _discover_models(self):
        """
        Discover available models (metadata only - FAST)
        This replaces the slow model loading during startup
        """
        try:
            model_files = list(self.models_dir.glob("*.pkl"))

            for model_file in model_files:
                try:
                    # Extract sensor ID from filename
                    sensor_id = model_file.stem

                    # Get file metadata (no loading)
                    stat = model_file.stat()

                    metadata = ModelMetadata(
                        sensor_id=sensor_id,
                        model_path=model_file,
                        file_size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime)
                    )

                    self.model_metadata[sensor_id] = metadata
                    self.model_proxies[sensor_id] = LazyModelProxy(metadata, self)

                except Exception as e:
                    logger.error(f"Error processing model file {model_file}: {e}")

            self.stats['models_discovered'] = len(self.model_metadata)
            logger.info(f"Discovered {len(self.model_metadata)} model files (no loading)")

        except Exception as e:
            logger.error(f"Error discovering models: {e}")

    def get_model(self, sensor_id: str) -> Optional[LazyModelProxy]:
        """
        Get a model proxy (creates on first access)

        Args:
            sensor_id: Sensor identifier

        Returns:
            LazyModelProxy or None if not found
        """
        if sensor_id in self.model_proxies:
            self._model_usage[sensor_id] += 1
            return self.model_proxies[sensor_id]
        return None

    def predict_anomalies(self, sensor_id: str, data: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Predict anomalies for a sensor (loads model if needed)

        Args:
            sensor_id: Sensor identifier
            data: Input sensor data

        Returns:
            Anomaly prediction results or None
        """
        proxy = self.get_model(sensor_id)
        if proxy is None:
            logger.warning(f"No model found for sensor {sensor_id}")
            return None

        try:
            self.stats['total_predictions'] += 1
            if proxy.is_loaded:
                self.stats['cache_hits'] += 1

            return proxy.predict_anomalies(data)

        except Exception as e:
            logger.error(f"Error predicting anomalies for {sensor_id}: {e}")
            return None

    def _register_loaded_model(self, proxy: LazyModelProxy):
        """Register a newly loaded model for lifecycle management"""
        with self._cleanup_lock:
            self._loaded_models[proxy.sensor_id] = proxy
            self.stats['models_loaded'] = len(self._loaded_models)

            # Enforce max loaded models limit
            if len(self._loaded_models) > self.max_loaded_models:
                self._evict_least_used_model()

    def _evict_least_used_model(self):
        """Evict the least recently used model"""
        if not self._loaded_models:
            return

        # Find least recently used model
        lru_sensor = min(
            self._loaded_models.keys(),
            key=lambda s: self._loaded_models[s]._last_used or datetime.min
        )

        logger.info(f"Evicting least used model: {lru_sensor}")
        self._loaded_models[lru_sensor].unload()
        del self._loaded_models[lru_sensor]
        self.stats['models_loaded'] = len(self._loaded_models)

    def _start_cleanup_thread(self):
        """Start background thread for model cleanup"""
        def cleanup_loop():
            while True:
                try:
                    self._cleanup_inactive_models()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    time.sleep(60)

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.debug("Model cleanup thread started")

    def _cleanup_inactive_models(self):
        """Unload models that haven't been used recently"""
        with self._cleanup_lock:
            current_time = datetime.now()
            to_unload = []

            for sensor_id, proxy in self._loaded_models.items():
                if proxy._last_used and (current_time - proxy._last_used) > self.unload_timeout:
                    to_unload.append(sensor_id)

            for sensor_id in to_unload:
                logger.info(f"Unloading inactive model: {sensor_id}")
                self._loaded_models[sensor_id].unload()
                del self._loaded_models[sensor_id]

            if to_unload:
                self.stats['models_loaded'] = len(self._loaded_models)

    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor IDs"""
        return list(self.model_metadata.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self.stats.copy()
        stats['models_loaded'] = len(self._loaded_models)
        stats['cache_hit_rate'] = (
            self.stats['cache_hits'] / max(1, self.stats['total_predictions']) * 100
        )
        stats['top_used_sensors'] = sorted(
            self._model_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        return stats

    def preload_models(self, sensor_ids: List[str]):
        """
        Preload specific models (useful for dashboard initialization)

        Args:
            sensor_ids: List of sensor IDs to preload
        """
        for sensor_id in sensor_ids:
            proxy = self.get_model(sensor_id)
            if proxy and not proxy.is_loaded:
                try:
                    proxy._load_model()
                    logger.info(f"Preloaded model for {sensor_id}")
                except Exception as e:
                    logger.error(f"Failed to preload model {sensor_id}: {e}")


# Global singleton instance
_lazy_model_manager = None


def get_lazy_model_manager() -> LazyModelManager:
    """Get the global lazy model manager instance"""
    global _lazy_model_manager
    if _lazy_model_manager is None:
        _lazy_model_manager = LazyModelManager()
    return _lazy_model_manager


def reset_lazy_model_manager():
    """Reset the global instance (useful for testing)"""
    global _lazy_model_manager
    _lazy_model_manager = None