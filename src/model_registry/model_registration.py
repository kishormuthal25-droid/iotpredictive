"""
Model Registration Script - Migrate NASA Telemanom Models to MLFlow
Registers all 97 existing NASA Telemanom models in MLFlow Model Registry
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

# MLFlow imports
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pyfunc
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: MLFlow not available. Install with: pip install mlflow")

from src.model_registry.mlflow_config import setup_mlflow, get_mlflow_config
from src.anomaly_detection.nasa_telemanom import NASATelemanom

logger = logging.getLogger(__name__)


class TelemanoMModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow wrapper for NASA Telemanom models"""

    def load_context(self, context):
        """Load model context"""
        import pickle
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        """Make predictions using the wrapped model"""
        # Handle both pandas DataFrame and numpy array inputs
        if hasattr(model_input, 'values'):
            # DataFrame input
            input_data = model_input.values
        else:
            # Array input
            input_data = model_input

        # Use the model's predict method
        if hasattr(self.model, 'predict'):
            return self.model.predict(input_data)
        elif hasattr(self.model, 'detect_anomalies'):
            return self.model.detect_anomalies(input_data)
        else:
            # Generic prediction
            return self.model(input_data)


class ModelRegistrar:
    """Handles registration of NASA Telemanom models to MLFlow"""

    def __init__(self, models_dir: str = "data/models/telemanom"):
        """
        Initialize model registrar

        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.mlflow_config = None
        self.client: Optional[MlflowClient] = None
        self.experiment_name = "NASA_Telemanom_Models"

        # Statistics
        self.registration_stats = {
            "total_models": 0,
            "successful_registrations": 0,
            "failed_registrations": 0,
            "skipped_registrations": 0,
            "start_time": None,
            "end_time": None
        }

    def setup_mlflow(self) -> bool:
        """Setup MLFlow infrastructure"""
        if not MLFLOW_AVAILABLE:
            logger.error("MLFlow not available. Please install MLFlow first.")
            return False

        try:
            # Setup MLFlow configuration
            self.mlflow_config = setup_mlflow()
            self.client = self.mlflow_config.setup_client()

            # Create experiment if it doesn't exist
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created MLFlow experiment: {self.experiment_name} (ID: {experiment_id})")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")

                mlflow.set_experiment(self.experiment_name)

            except Exception as e:
                logger.error(f"Failed to setup experiment: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to setup MLFlow: {e}")
            return False

    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover all NASA Telemanom models"""
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return []

        models = []
        for model_file in self.models_dir.glob("*.pkl"):
            model_info = {
                "name": model_file.stem,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime)
            }

            # Try to extract additional metadata
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'config'):
                        model_info["config"] = model.config.__dict__ if hasattr(model.config, '__dict__') else str(model.config)
                    if hasattr(model, 'threshold'):
                        model_info["threshold"] = model.threshold
                    if hasattr(model, 'sensor_id'):
                        model_info["sensor_id"] = model.sensor_id

            except Exception as e:
                logger.warning(f"Could not extract metadata from {model_file}: {e}")

            models.append(model_info)

        logger.info(f"Discovered {len(models)} models in {self.models_dir}")
        return models

    def register_single_model(self, model_info: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Register a single model to MLFlow

        Args:
            model_info: Model information dictionary
            overwrite: Whether to overwrite existing registered models

        Returns:
            True if registration successful
        """
        model_name = model_info["name"]
        model_path = model_info["path"]

        try:
            # Check if model already registered
            try:
                registered_model = self.client.get_registered_model(model_name)
                if not overwrite:
                    logger.info(f"Model {model_name} already registered, skipping")
                    self.registration_stats["skipped_registrations"] += 1
                    return True
                else:
                    logger.info(f"Model {model_name} already registered, overwriting")
            except:
                # Model doesn't exist yet
                pass

            # Load model for signature inference
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Create sample input for signature inference
            sample_input = self._create_sample_input(model_info.get("sensor_id", model_name))

            # Create MLFlow run for this model
            with mlflow.start_run(run_name=f"register_{model_name}") as run:
                # Log model parameters
                if "config" in model_info:
                    if isinstance(model_info["config"], dict):
                        mlflow.log_params(model_info["config"])
                    else:
                        mlflow.log_param("config", str(model_info["config"]))

                # Log model metadata
                mlflow.log_param("sensor_id", model_info.get("sensor_id", model_name))
                mlflow.log_param("model_file", model_path)
                mlflow.log_param("file_size", model_info["size"])
                mlflow.log_param("modified_date", model_info["modified"].isoformat())

                if "threshold" in model_info:
                    mlflow.log_param("anomaly_threshold", model_info["threshold"])

                # Log model artifacts
                artifacts = {"model": model_path}

                # Infer model signature
                try:
                    signature = infer_signature(sample_input, model.predict(sample_input) if hasattr(model, 'predict') else None)
                except Exception as e:
                    logger.warning(f"Could not infer signature for {model_name}: {e}")
                    signature = None

                # Register model using the wrapper
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=TelemanoMModelWrapper(),
                    artifacts=artifacts,
                    signature=signature,
                    registered_model_name=model_name,
                    input_example=sample_input
                )

                logger.info(f"Successfully registered model: {model_name}")
                self.registration_stats["successful_registrations"] += 1
                return True

        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            self.registration_stats["failed_registrations"] += 1
            return False

    def _create_sample_input(self, sensor_id: str) -> np.ndarray:
        """Create sample input for signature inference"""
        # Create sample time series data (250 timesteps as per NASA Telemanom default)
        sample_length = 250
        if "MSL" in sensor_id:
            # Mars rover data - typically engineering parameters
            sample_data = np.random.normal(0.5, 0.1, (1, sample_length))
        elif "SMAP" in sensor_id:
            # Satellite data - typically sensor readings
            sample_data = np.random.normal(0.0, 1.0, (1, sample_length))
        else:
            # Generic time series
            sample_data = np.random.normal(0.0, 1.0, (1, sample_length))

        return sample_data

    def register_all_models(self, overwrite: bool = False, batch_size: int = 10) -> Dict[str, Any]:
        """
        Register all discovered models

        Args:
            overwrite: Whether to overwrite existing models
            batch_size: Number of models to process before logging progress

        Returns:
            Registration summary
        """
        self.registration_stats["start_time"] = datetime.now()

        # Discover models
        models = self.discover_models()
        self.registration_stats["total_models"] = len(models)

        if not models:
            logger.warning("No models found to register")
            return self.registration_stats

        logger.info(f"Starting registration of {len(models)} models...")

        # Register models
        for i, model_info in enumerate(models, 1):
            model_name = model_info["name"]
            logger.info(f"Registering model {i}/{len(models)}: {model_name}")

            success = self.register_single_model(model_info, overwrite=overwrite)

            # Log progress
            if i % batch_size == 0:
                logger.info(f"Progress: {i}/{len(models)} models processed")
                logger.info(f"Success: {self.registration_stats['successful_registrations']}, "
                           f"Failed: {self.registration_stats['failed_registrations']}, "
                           f"Skipped: {self.registration_stats['skipped_registrations']}")

        self.registration_stats["end_time"] = datetime.now()
        duration = self.registration_stats["end_time"] - self.registration_stats["start_time"]

        logger.info("Model registration completed!")
        logger.info(f"Total models: {self.registration_stats['total_models']}")
        logger.info(f"Successful: {self.registration_stats['successful_registrations']}")
        logger.info(f"Failed: {self.registration_stats['failed_registrations']}")
        logger.info(f"Skipped: {self.registration_stats['skipped_registrations']}")
        logger.info(f"Duration: {duration}")

        return self.registration_stats

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """List all models in MLFlow registry"""
        if not self.client:
            logger.error("MLFlow client not initialized")
            return []

        try:
            registered_models = self.client.search_registered_models()
            models_info = []

            for rm in registered_models:
                # Get latest version info
                latest_versions = self.client.get_latest_versions(rm.name)
                latest_version = latest_versions[0] if latest_versions else None

                model_info = {
                    "name": rm.name,
                    "creation_timestamp": rm.creation_timestamp,
                    "last_updated_timestamp": rm.last_updated_timestamp,
                    "description": rm.description,
                    "latest_version": latest_version.version if latest_version else None,
                    "stage": latest_version.current_stage if latest_version else None,
                }
                models_info.append(model_info)

            return models_info

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            return []


def main():
    """Main registration script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    registrar = ModelRegistrar()

    # Setup MLFlow
    if not registrar.setup_mlflow():
        print("Failed to setup MLFlow. Exiting.")
        return 1

    # Register all models
    print("Starting NASA Telemanom model registration...")
    results = registrar.register_all_models(overwrite=False)

    # Print summary
    print("\n" + "="*60)
    print("REGISTRATION SUMMARY")
    print("="*60)
    print(f"Total models discovered: {results['total_models']}")
    print(f"Successfully registered: {results['successful_registrations']}")
    print(f"Failed to register: {results['failed_registrations']}")
    print(f"Skipped (already exist): {results['skipped_registrations']}")
    print(f"Duration: {results['end_time'] - results['start_time']}")

    # List registered models
    print("\nRegistered models in MLFlow:")
    models = registrar.list_registered_models()
    for model in models:
        print(f"- {model['name']} (v{model['latest_version']}, {model['stage']})")

    return 0 if results['failed_registrations'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())