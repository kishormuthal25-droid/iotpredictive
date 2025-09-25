"""
Optional MLFlow Service
Provides MLFlow integration as an optional enhancement that doesn't break the system if unavailable
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class OptionalMLFlowService:
    """
    MLFlow service that works as an optional enhancement

    Key Features:
    - Non-blocking initialization
    - Background connection attempts
    - Graceful fallback when MLFlow unavailable
    - Auto-reconnection when MLFlow becomes available
    """

    def __init__(self,
                 connection_timeout: float = 5.0,
                 retry_attempts: int = 3,
                 retry_delay: float = 2.0,
                 background_sync: bool = True):
        """
        Initialize optional MLFlow service

        Args:
            connection_timeout: Max time to wait for MLFlow connection
            retry_attempts: Number of connection retry attempts
            retry_delay: Delay between retry attempts
            background_sync: Enable background connection monitoring
        """
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.background_sync = background_sync

        # Connection state
        self.is_connected = False
        self.is_connecting = False
        self.last_connection_attempt = None
        self.connection_error = None

        # MLFlow components (loaded on-demand)
        self._mlflow_client = None
        self._mlflow_config = None
        self._model_manager = None

        # Background monitoring
        self._background_thread = None
        self._stop_background = False
        self._connection_lock = threading.RLock()

        # Statistics
        self.stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'last_successful_connection': None,
            'uptime': timedelta()
        }

        logger.info("OptionalMLFlowService initialized (background mode)")

        if self.background_sync:
            self._start_background_monitoring()

    def _start_background_monitoring(self):
        """Start background thread for connection monitoring"""
        if self._background_thread is not None:
            return

        def monitoring_loop():
            """Background monitoring loop"""
            while not self._stop_background:
                try:
                    if not self.is_connected:
                        self._attempt_connection(blocking=False)

                    # Check connection health every 30 seconds
                    time.sleep(30)

                    if self.is_connected:
                        self._check_connection_health()

                except Exception as e:
                    logger.error(f"Error in MLFlow background monitoring: {e}")
                    time.sleep(10)

        self._background_thread = threading.Thread(
            target=monitoring_loop,
            daemon=True,
            name="MLFlowMonitor"
        )
        self._background_thread.start()
        logger.info("MLFlow background monitoring started")

    def _attempt_connection(self, blocking: bool = False) -> bool:
        """
        Attempt to connect to MLFlow

        Args:
            blocking: If True, wait for connection. If False, return immediately

        Returns:
            True if connection successful or already connected
        """
        with self._connection_lock:
            if self.is_connected:
                return True

            if self.is_connecting and not blocking:
                return False

            self.is_connecting = True
            self.stats['connection_attempts'] += 1

        try:
            logger.info("Attempting MLFlow connection...")
            self.last_connection_attempt = datetime.now()

            # Try to import and initialize MLFlow
            success = self._initialize_mlflow_components()

            if success:
                self.is_connected = True
                self.is_connecting = False
                self.connection_error = None
                self.stats['successful_connections'] += 1
                self.stats['last_successful_connection'] = datetime.now()

                logger.info("✅ MLFlow connection successful - enhanced features available")
                return True
            else:
                self.is_connecting = False
                self.stats['failed_connections'] += 1
                logger.info("❌ MLFlow connection failed - using fallback system")
                return False

        except Exception as e:
            self.is_connected = False
            self.is_connecting = False
            self.connection_error = str(e)
            self.stats['failed_connections'] += 1

            logger.info(f"MLFlow connection error: {e} - dashboard continues with basic features")
            return False

    def _initialize_mlflow_components(self) -> bool:
        """
        Initialize MLFlow components with timeout

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import MLFlow modules (with timeout)
            import mlflow
            from mlflow.tracking import MlflowClient
            from src.model_registry.mlflow_config import get_mlflow_config
            from src.model_registry.model_manager import get_model_manager

            # Test MLFlow server availability with timeout
            config = get_mlflow_config()

            # Quick health check without starting server
            if not config.is_server_running():
                # Try to start server in background
                if config.start_server(wait_for_startup=False):
                    # Give it a moment to start
                    time.sleep(1)
                    if not config.is_server_running():
                        logger.info("MLFlow server not ready yet - will retry in background")
                        return False
                else:
                    logger.info("Could not start MLFlow server - will retry later")
                    return False

            # Initialize components
            self._mlflow_config = config
            self._mlflow_client = MlflowClient()
            self._model_manager = get_model_manager()

            # Test basic functionality
            self._mlflow_client.list_experiments(max_results=1)

            logger.info("MLFlow components initialized successfully")
            return True

        except Exception as e:
            logger.info(f"MLFlow initialization failed: {e}")
            return False

    def _check_connection_health(self):
        """Check if MLFlow connection is still healthy"""
        try:
            if self._mlflow_client:
                # Simple health check
                self._mlflow_client.list_experiments(max_results=1)
                return True
        except Exception as e:
            logger.warning(f"MLFlow connection lost: {e}")
            self.is_connected = False
            self._mlflow_client = None
            self._model_manager = None
            return False

    def get_model_manager(self) -> Optional[Any]:
        """
        Get MLFlow model manager if available

        Returns:
            Model manager instance or None if MLFlow not available
        """
        if not self.is_connected and not self.is_connecting:
            # Try connection if not attempted recently
            if (self.last_connection_attempt is None or
                datetime.now() - self.last_connection_attempt > timedelta(minutes=5)):

                self._attempt_connection(blocking=False)

        if self.is_connected and self._model_manager:
            return self._model_manager

        return None

    def predict_anomalies(self, sensor_id: str, data) -> Optional[Dict[str, Any]]:
        """
        Predict anomalies using MLFlow models if available

        Args:
            sensor_id: Sensor identifier
            data: Input data for prediction

        Returns:
            Prediction results or None if MLFlow not available
        """
        model_manager = self.get_model_manager()

        if model_manager is None:
            return None

        try:
            return model_manager.predict_anomalies(sensor_id, data)
        except Exception as e:
            logger.error(f"MLFlow prediction failed for {sensor_id}: {e}")
            return None

    def is_available(self) -> bool:
        """Check if MLFlow is currently available"""
        return self.is_connected

    def get_status(self) -> Dict[str, Any]:
        """Get detailed MLFlow service status"""
        return {
            'connected': self.is_connected,
            'connecting': self.is_connecting,
            'last_connection_attempt': self.last_connection_attempt,
            'connection_error': self.connection_error,
            'background_monitoring': self.background_sync and self._background_thread is not None,
            'stats': self.stats.copy()
        }

    def force_reconnect(self):
        """Force a reconnection attempt"""
        logger.info("Forcing MLFlow reconnection...")
        self.is_connected = False
        self._mlflow_client = None
        self._model_manager = None
        return self._attempt_connection(blocking=True)

    def shutdown(self):
        """Shutdown the service gracefully"""
        logger.info("Shutting down OptionalMLFlowService...")
        self._stop_background = True

        if self._background_thread:
            self._background_thread.join(timeout=5)

        self.is_connected = False
        self._mlflow_client = None
        self._model_manager = None


# Global singleton instance
_optional_mlflow_service = None


def get_optional_mlflow_service(
    connection_timeout: float = 5.0,
    retry_attempts: int = 3,
    background_sync: bool = True
) -> OptionalMLFlowService:
    """
    Get or create global OptionalMLFlowService instance

    Args:
        connection_timeout: Max time to wait for connections
        retry_attempts: Number of retry attempts
        background_sync: Enable background monitoring

    Returns:
        OptionalMLFlowService instance
    """
    global _optional_mlflow_service

    if _optional_mlflow_service is None:
        _optional_mlflow_service = OptionalMLFlowService(
            connection_timeout=connection_timeout,
            retry_attempts=retry_attempts,
            background_sync=background_sync
        )

    return _optional_mlflow_service


def reset_optional_mlflow_service():
    """Reset the global service instance (useful for testing)"""
    global _optional_mlflow_service

    if _optional_mlflow_service:
        _optional_mlflow_service.shutdown()
        _optional_mlflow_service = None