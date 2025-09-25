"""
MLFlow-based Model Manager
Provides lazy loading and caching for NASA Telemanom models
"""

import os
import sys
import logging
import pickle
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

# MLFlow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLFlow not available, falling back to local model loading")

from src.model_registry.mlflow_config import get_mlflow_config

logger = logging.getLogger(__name__)


class ModelCache:
    """LRU Cache for loaded models"""

    def __init__(self, max_size: int = 20):
        """
        Initialize model cache

        Args:
            max_size: Maximum number of models to keep in memory
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()

    def get(self, model_id: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if model_id in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(model_id)
                self.access_order.append(model_id)
                return self.cache[model_id]
            return None

    def put(self, model_id: str, model: Any):
        """Put model in cache"""
        with self.lock:
            if model_id in self.cache:
                # Update existing
                self.cache[model_id] = model
                self.access_order.remove(model_id)
                self.access_order.append(model_id)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    lru_id = self.access_order.pop(0)
                    del self.cache[lru_id]
                    logger.debug(f"Evicted model {lru_id} from cache")

                self.cache[model_id] = model
                self.access_order.append(model_id)

        logger.debug(f"Cached model {model_id} (cache size: {len(self.cache)})")

    def clear(self):
        """Clear all cached models"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "models": list(self.cache.keys()),
                "access_order": self.access_order.copy()
            }


class MLFlowModelManager:
    """MLFlow-based model manager with lazy loading and caching"""

    def __init__(self,
                 cache_size: int = 20,
                 warmup_models: Optional[List[str]] = None):
        """
        Initialize MLFlow model manager

        Args:
            cache_size: Maximum models to keep in memory
            warmup_models: Models to preload during startup
        """
        self.cache = ModelCache(cache_size)
        self.warmup_models = warmup_models or []
        self.mlflow_config = get_mlflow_config()
        self.client: Optional[MlflowClient] = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ModelLoader")

        # Model loading statistics
        self.loading_stats = {
            "models_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "loading_times": {},
            "last_loaded": None
        }

        # Lock for thread safety
        self.lock = threading.RLock()

        # Track available models
        self._available_models: Dict[str, Dict[str, Any]] = {}
        self._models_discovered = False

        # Initialize MLFlow client
        self._setup_client()

    def _setup_client(self):
        """Setup MLFlow client"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLFlow not available, using fallback mode")
            return

        try:
            # Ensure MLFlow server is running
            if not self.mlflow_config.is_running:
                logger.info("Starting MLFlow server for model management...")
                self.mlflow_config.start_server(wait_for_startup=True)

            self.client = self.mlflow_config.setup_client()
            logger.info(f"MLFlow model manager connected to {self.mlflow_config.tracking_uri}")

            # Discover available models
            self._discover_models()

        except Exception as e:
            logger.error(f"Failed to setup MLFlow client: {e}")
            logger.info("Falling back to local model loading")

    def _discover_models(self):
        """Discover available models in MLFlow registry"""
        if not self.client:
            return

        try:
            # Get all registered models
            registered_models = self.client.search_registered_models()

            for rm in registered_models:
                model_name = rm.name

                # Get latest version
                latest_versions = self.client.get_latest_versions(model_name)
                if latest_versions:
                    latest_version = latest_versions[0]
                    self._available_models[model_name] = {
                        "name": model_name,
                        "version": latest_version.version,
                        "stage": latest_version.current_stage,
                        "model_uri": f"models:/{model_name}/{latest_version.version}",
                        "source": "mlflow"
                    }

            logger.info(f"Discovered {len(self._available_models)} models in MLFlow registry")
            self._models_discovered = True

        except Exception as e:
            logger.error(f"Failed to discover MLFlow models: {e}")

    def _fallback_discover_models(self):
        """Discover models from local filesystem (fallback) - Enhanced for 308 models"""
        models_dir = Path("data/models")
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return

        # Discover all model files (.h5, .pkl, .joblib)
        model_patterns = ["*.h5", "*.pkl", "*.joblib"]
        for pattern in model_patterns:
            for model_file in models_dir.glob(f"**/{pattern}"):
                model_name = model_file.stem

                # Skip duplicate model names (prefer _best.h5 over others)
                if model_name in self._available_models:
                    if "_best" in model_file.name:
                        # Replace with best version
                        pass
                    else:
                        # Skip if we already have this model
                        continue

                self._available_models[model_name] = {
                    "name": model_name,
                    "version": "local",
                    "stage": "Production",
                    "model_path": str(model_file),
                    "source": "local",
                    "file_type": model_file.suffix,
                    "size_mb": model_file.stat().st_size / 1024 / 1024 if model_file.exists() else 0
                }

        logger.info(f"Discovered {len(self._available_models)} local models (H5: {sum(1 for m in self._available_models.values() if m['file_type'] == '.h5')}, PKL: {sum(1 for m in self._available_models.values() if m['file_type'] == '.pkl')}, JOBLIB: {sum(1 for m in self._available_models.values() if m['file_type'] == '.joblib')})")

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models"""
        if not self._models_discovered:
            if self.client:
                self._discover_models()
            else:
                self._fallback_discover_models()

        return self._available_models.copy()

    def load_model_async(self, model_name: str) -> Future:
        """Load model asynchronously"""
        return self.executor.submit(self.load_model, model_name)

    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load model with caching and fallback

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model or None if failed
        """
        start_time = time.time()

        # Check cache first
        cached_model = self.cache.get(model_name)
        if cached_model is not None:
            with self.lock:
                self.loading_stats["cache_hits"] += 1
            logger.debug(f"Model {model_name} loaded from cache")
            return cached_model

        with self.lock:
            self.loading_stats["cache_misses"] += 1

        # Load model
        model = None
        try:
            # Try MLFlow first
            if self.client and model_name in self._available_models:
                model_info = self._available_models[model_name]
                if model_info["source"] == "mlflow":
                    model = self._load_from_mlflow(model_name, model_info)
                else:
                    model = self._load_from_local(model_name, model_info)
            else:
                # Fallback to local loading
                model = self._load_from_local_fallback(model_name)

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

        if model is not None:
            # Cache the model
            self.cache.put(model_name, model)

            # Update statistics
            loading_time = time.time() - start_time
            with self.lock:
                self.loading_stats["models_loaded"] += 1
                self.loading_stats["loading_times"][model_name] = loading_time
                self.loading_stats["last_loaded"] = datetime.now()

            logger.info(f"Model {model_name} loaded successfully in {loading_time:.2f}s")
        else:
            logger.error(f"Failed to load model {model_name}")

        return model

    def _load_from_mlflow(self, model_name: str, model_info: Dict[str, Any]) -> Any:
        """Load model from MLFlow registry"""
        model_uri = model_info["model_uri"]
        logger.debug(f"Loading model {model_name} from MLFlow: {model_uri}")

        # Use MLFlow's model loading
        model = mlflow.pyfunc.load_model(model_uri)
        return model

    def _load_from_local(self, model_name: str, model_info: Dict[str, Any]) -> Any:
        """Load model from local file - Enhanced for multiple formats"""
        model_path = model_info["model_path"]
        file_type = model_info.get("file_type", ".pkl")

        logger.debug(f"Loading model {model_name} from local file: {model_path} (type: {file_type})")

        try:
            if file_type == ".h5":
                # TensorFlow/Keras model
                try:
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path, compile=False)
                    logger.debug(f"Loaded TensorFlow model {model_name}")
                    return model
                except ImportError:
                    logger.error("TensorFlow not available for H5 model loading")
                    return None

            elif file_type in [".pkl", ".joblib"]:
                # Pickle or Joblib model
                if file_type == ".joblib":
                    try:
                        import joblib
                        model = joblib.load(model_path)
                        logger.debug(f"Loaded Joblib model {model_name}")
                        return model
                    except ImportError:
                        logger.warning("Joblib not available, trying pickle")

                # Fallback to pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.debug(f"Loaded Pickle model {model_name}")
                return model
            else:
                logger.error(f"Unsupported model file type: {file_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
            return None

    def _load_from_local_fallback(self, model_name: str) -> Optional[Any]:
        """Fallback: Load model from expected local path - Multiple file types"""
        # Try different file types in order of preference
        search_patterns = [
            f"data/models/**/{model_name}_best.h5",
            f"data/models/**/{model_name}.h5",
            f"data/models/**/{model_name}.pkl",
            f"data/models/**/{model_name}.joblib"
        ]

        for pattern in search_patterns:
            model_files = list(Path(".").glob(pattern))
            if model_files:
                model_path = model_files[0]  # Take first match
                file_type = model_path.suffix

                logger.debug(f"Fallback loading model {model_name} from: {model_path}")

                # Use same loading logic as _load_from_local
                model_info = {
                    "model_path": str(model_path),
                    "file_type": file_type
                }
                return self._load_from_local(model_name, model_info)

        logger.warning(f"Model {model_name} not found in any supported format")
        return None

    def warmup_models(self) -> Dict[str, bool]:
        """Preload warmup models"""
        if not self.warmup_models:
            return {}

        logger.info(f"Warming up {len(self.warmup_models)} models...")
        results = {}

        # Load models in parallel
        futures = {
            model_name: self.load_model_async(model_name)
            for model_name in self.warmup_models
        }

        for model_name, future in futures.items():
            try:
                model = future.result(timeout=30)  # 30 second timeout per model
                results[model_name] = model is not None
            except Exception as e:
                logger.error(f"Failed to warmup model {model_name}: {e}")
                results[model_name] = False

        successful = sum(results.values())
        logger.info(f"Warmed up {successful}/{len(self.warmup_models)} models")
        return results

    def get_model(self, model_name: str, timeout: float = 30.0) -> Optional[Any]:
        """
        Get model with timeout (convenience method)

        Args:
            model_name: Name of model to get
            timeout: Maximum time to wait for model loading

        Returns:
            Loaded model or None
        """
        try:
            future = self.load_model_async(model_name)
            return future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Failed to get model {model_name} within {timeout}s: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self.lock:
            stats = self.loading_stats.copy()

        stats.update({
            "available_models": len(self._available_models),
            "cache_stats": self.cache.get_stats(),
            "mlflow_connected": self.client is not None,
            "mlflow_server_running": self.mlflow_config.is_running
        })

        return stats

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up MLFlow model manager...")
        self.cache.clear()
        self.executor.shutdown(wait=True)

        if self.mlflow_config.is_running:
            self.mlflow_config.stop_server()


# Global model manager instance
_model_manager: Optional[MLFlowModelManager] = None

def get_model_manager() -> MLFlowModelManager:
    """Get or create global model manager instance"""
    global _model_manager
    if _model_manager is None:
        # Default warmup models based on your actual model naming pattern
        warmup_models = [
            "MSL-COM-001_anomaly_detector_best",
            "MSL-ENV-001_anomaly_detector_best",
            "MSL-MOB-001_anomaly_detector_best",
            "MSL-NAV-001_anomaly_detector_best",
            "MSL-POW-001_anomaly_detector_best"
        ]
        _model_manager = MLFlowModelManager(cache_size=25, warmup_models=warmup_models)
    return _model_manager