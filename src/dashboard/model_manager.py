"""
Pre-trained Model Manager for Dashboard - MLFlow Enhanced
MLFlow-aware lazy loading model manager that preserves original interface
Drop-in replacement that solves 308-model memory issue
"""

# Import the MLFlow-enhanced manager as the main implementation
from src.dashboard.mlflow_model_manager import (
    MLFlowDashboardModelManager,
    get_mlflow_dashboard_model_manager
)

import logging

logger = logging.getLogger(__name__)


class PretrainedModelManager(MLFlowDashboardModelManager):
    """
    MLFlow-enhanced pre-trained model manager
    Provides exact same interface as original but with lazy loading
    """

    def __init__(self):
        """Initialize with MLFlow lazy loading"""
        super().__init__()
        logger.info(f"PretrainedModelManager (MLFlow Enhanced) initialized with lazy loading")


# Global instance for compatibility with existing code
pretrained_model_manager = get_mlflow_dashboard_model_manager()