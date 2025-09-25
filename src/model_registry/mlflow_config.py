"""
MLFlow Configuration and Setup
Handles MLFlow tracking server configuration and model registry setup
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import socket

logger = logging.getLogger(__name__)


class MLFlowConfig:
    """MLFlow configuration and server management"""

    def __init__(self,
                 tracking_uri: str = "http://localhost:5000",
                 backend_store_uri: Optional[str] = None,
                 artifact_store_uri: Optional[str] = None,
                 registry_store_uri: Optional[str] = None):
        """
        Initialize MLFlow configuration

        Args:
            tracking_uri: MLFlow tracking server URI
            backend_store_uri: Backend store for experiment metadata
            artifact_store_uri: Artifact store for model files
            registry_store_uri: Model registry database URI
        """
        self.tracking_uri = tracking_uri
        self.host, self.port = self._parse_uri(tracking_uri)

        # Setup default paths if not provided
        project_root = Path(__file__).parent.parent.parent
        mlflow_dir = project_root / "mlflow_data"
        mlflow_dir.mkdir(exist_ok=True)

        self.backend_store_uri = backend_store_uri or f"sqlite:///{mlflow_dir}/mlflow.db"
        self.artifact_store_uri = artifact_store_uri or str(mlflow_dir / "artifacts")
        self.registry_store_uri = registry_store_uri or self.backend_store_uri

        self.server_process: Optional[subprocess.Popen] = None
        self.is_running = False

        # Set MLFlow environment variables
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri

    def _parse_uri(self, uri: str) -> tuple[str, int]:
        """Parse tracking URI to extract host and port"""
        if uri.startswith("http://"):
            uri = uri[7:]
        elif uri.startswith("https://"):
            uri = uri[8:]

        if ":" in uri:
            host, port_str = uri.split(":")
            port = int(port_str)
        else:
            host = uri
            port = 5000

        return host, port

    def is_server_running(self) -> bool:
        """Check if MLFlow server is already running"""
        try:
            import mlflow
            # Try to connect to the tracking server
            mlflow.set_tracking_uri(self.tracking_uri)
            # This will raise an exception if server is not reachable
            mlflow.get_experiment_by_name("Default")
            return True
        except Exception:
            return False

    def check_port_available(self) -> bool:
        """Check if the MLFlow port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((self.host, self.port))
                return True
            except OSError:
                return False

    def start_server(self, wait_for_startup: bool = True) -> bool:
        """
        Start MLFlow tracking server

        Args:
            wait_for_startup: Whether to wait for server to be ready

        Returns:
            True if server started successfully
        """
        if self.is_server_running():
            logger.info(f"MLFlow server already running at {self.tracking_uri}")
            self.is_running = True
            return True

        if not self.check_port_available():
            logger.error(f"Port {self.port} is already in use")
            return False

        try:
            # Prepare MLFlow server command
            cmd = [
                sys.executable, "-m", "mlflow", "server",
                "--backend-store-uri", self.backend_store_uri,
                "--default-artifact-root", self.artifact_store_uri,
                "--host", self.host,
                "--port", str(self.port),
                "--serve-artifacts"  # Enable artifact serving
            ]

            logger.info(f"Starting MLFlow server: {' '.join(cmd)}")

            # Start server process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if wait_for_startup:
                # Wait for server to be ready
                max_attempts = 30
                for attempt in range(max_attempts):
                    time.sleep(1)
                    if self.is_server_running():
                        logger.info(f"MLFlow server started successfully at {self.tracking_uri}")
                        self.is_running = True
                        return True

                logger.error("MLFlow server failed to start within timeout")
                self.stop_server()
                return False
            else:
                # Start monitoring thread
                monitor_thread = threading.Thread(target=self._monitor_server, daemon=True)
                monitor_thread.start()
                self.is_running = True
                return True

        except Exception as e:
            logger.error(f"Failed to start MLFlow server: {e}")
            return False

    def _monitor_server(self):
        """Monitor server process and update status"""
        while self.server_process and self.server_process.poll() is None:
            time.sleep(5)

        # Server process ended
        self.is_running = False
        if self.server_process:
            logger.warning(f"MLFlow server process ended with code: {self.server_process.returncode}")

    def stop_server(self):
        """Stop MLFlow tracking server"""
        if self.server_process:
            logger.info("Stopping MLFlow server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("MLFlow server didn't stop gracefully, forcing...")
                self.server_process.kill()
                self.server_process.wait()

            self.server_process = None
            self.is_running = False
            logger.info("MLFlow server stopped")

    def setup_client(self) -> 'mlflow.MlflowClient':
        """Setup and return MLFlow client"""
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)

        # Set registry URI for model registry operations
        mlflow.set_registry_uri(self.registry_store_uri)

        client = mlflow.MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_store_uri
        )

        return client

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "tracking_uri": self.tracking_uri,
            "backend_store_uri": self.backend_store_uri,
            "artifact_store_uri": self.artifact_store_uri,
            "registry_store_uri": self.registry_store_uri,
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running
        }


# Global MLFlow configuration instance
_mlflow_config: Optional[MLFlowConfig] = None

def get_mlflow_config() -> MLFlowConfig:
    """Get or create global MLFlow configuration instance"""
    global _mlflow_config
    if _mlflow_config is None:
        _mlflow_config = MLFlowConfig()
    return _mlflow_config

def setup_mlflow() -> MLFlowConfig:
    """Setup MLFlow with default configuration"""
    config = get_mlflow_config()

    logger.info("Setting up MLFlow infrastructure...")
    logger.info(f"Tracking URI: {config.tracking_uri}")
    logger.info(f"Backend Store: {config.backend_store_uri}")
    logger.info(f"Artifact Store: {config.artifact_store_uri}")

    # Start server if not running
    if not config.is_running:
        success = config.start_server(wait_for_startup=False)
        if success:
            logger.info("MLFlow server started successfully")
        else:
            logger.warning("Failed to start MLFlow server, using existing instance")

    return config