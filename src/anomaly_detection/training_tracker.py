"""
Training Progress Tracker for NASA Anomaly Detection Models
Provides real-time training progress monitoring and status management
"""

import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training status enumeration"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    LOADING = "loading"


@dataclass
class ModelProgress:
    """Progress information for a single model"""
    equipment_id: str
    status: TrainingStatus = TrainingStatus.NOT_STARTED
    current_epoch: int = 0
    total_epochs: int = 0
    loss: float = 0.0
    val_loss: float = 0.0
    accuracy: float = 0.0
    progress_percent: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: str = ""
    training_samples: int = 0

    @property
    def duration(self) -> float:
        """Training duration in seconds"""
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return 0.0

    @property
    def eta_seconds(self) -> float:
        """Estimated time to completion in seconds"""
        if self.progress_percent > 0 and self.duration > 0:
            total_estimated = self.duration / (self.progress_percent / 100.0)
            return max(0, total_estimated - self.duration)
        return 0.0


class TrainingProgressTracker:
    """Thread-safe training progress tracker"""

    def __init__(self):
        """Initialize training tracker"""
        self._models: Dict[str, ModelProgress] = {}
        self._lock = threading.Lock()
        self._observers: List[Callable[[Dict[str, ModelProgress]], None]] = []

        # Global training status
        self._global_status = TrainingStatus.NOT_STARTED
        self._total_models = 0
        self._completed_models = 0

        logger.info("Training progress tracker initialized")

    def register_observer(self, callback: Callable[[Dict[str, ModelProgress]], None]):
        """Register observer for progress updates

        Args:
            callback: Function to call when progress updates
        """
        with self._lock:
            self._observers.append(callback)

    def unregister_observer(self, callback: Callable[[Dict[str, ModelProgress]], None]):
        """Unregister observer

        Args:
            callback: Function to remove from observers
        """
        with self._lock:
            if callback in self._observers:
                self._observers.remove(callback)

    def start_training(self, equipment_ids: List[str], total_epochs: int = 10):
        """Start training for multiple models

        Args:
            equipment_ids: List of equipment IDs to train
            total_epochs: Total epochs for training
        """
        with self._lock:
            self._total_models = len(equipment_ids)
            self._completed_models = 0
            self._global_status = TrainingStatus.INITIALIZING

            # Initialize progress for each model
            for equipment_id in equipment_ids:
                self._models[equipment_id] = ModelProgress(
                    equipment_id=equipment_id,
                    status=TrainingStatus.NOT_STARTED,
                    total_epochs=total_epochs
                )

            logger.info(f"Started training tracking for {len(equipment_ids)} models")
            self._notify_observers()

    def start_model_training(self, equipment_id: str, training_samples: int):
        """Mark model training as started

        Args:
            equipment_id: Equipment identifier
            training_samples: Number of training samples
        """
        with self._lock:
            if equipment_id in self._models:
                progress = self._models[equipment_id]
                progress.status = TrainingStatus.TRAINING
                progress.start_time = datetime.now()
                progress.training_samples = training_samples
                progress.progress_percent = 0.0

                logger.info(f"Started training for {equipment_id} with {training_samples} samples")
                self._notify_observers()

    def update_epoch(self, equipment_id: str, epoch: int, loss: float,
                     val_loss: float = 0.0, accuracy: float = 0.0):
        """Update training progress for an epoch

        Args:
            equipment_id: Equipment identifier
            epoch: Current epoch number
            loss: Training loss
            val_loss: Validation loss
            accuracy: Training accuracy
        """
        with self._lock:
            if equipment_id in self._models:
                progress = self._models[equipment_id]
                progress.current_epoch = epoch
                progress.loss = loss
                progress.val_loss = val_loss
                progress.accuracy = accuracy

                # Calculate progress percentage
                if progress.total_epochs > 0:
                    progress.progress_percent = (epoch / progress.total_epochs) * 100.0

                logger.debug(f"Epoch {epoch} for {equipment_id}: loss={loss:.4f}, val_loss={val_loss:.4f}")
                self._notify_observers()

    def complete_model_training(self, equipment_id: str, success: bool = True,
                               error_message: str = ""):
        """Mark model training as completed

        Args:
            equipment_id: Equipment identifier
            success: Whether training was successful
            error_message: Error message if failed
        """
        with self._lock:
            if equipment_id in self._models:
                progress = self._models[equipment_id]
                progress.end_time = datetime.now()
                progress.error_message = error_message

                if success:
                    progress.status = TrainingStatus.COMPLETED
                    progress.progress_percent = 100.0
                    self._completed_models += 1
                    logger.info(f"Completed training for {equipment_id} in {progress.duration:.1f}s")
                else:
                    progress.status = TrainingStatus.FAILED
                    logger.error(f"Failed training for {equipment_id}: {error_message}")

                # Update global status
                if self._completed_models >= self._total_models:
                    self._global_status = TrainingStatus.COMPLETED

                self._notify_observers()

    def set_model_loading(self, equipment_id: str):
        """Mark model as loading from disk

        Args:
            equipment_id: Equipment identifier
        """
        with self._lock:
            if equipment_id not in self._models:
                self._models[equipment_id] = ModelProgress(equipment_id=equipment_id)

            self._models[equipment_id].status = TrainingStatus.LOADING
            logger.debug(f"Loading model for {equipment_id}")
            self._notify_observers()

    def get_model_progress(self, equipment_id: str) -> Optional[ModelProgress]:
        """Get progress for specific model

        Args:
            equipment_id: Equipment identifier

        Returns:
            Model progress or None if not found
        """
        with self._lock:
            return self._models.get(equipment_id)

    def get_all_progress(self) -> Dict[str, ModelProgress]:
        """Get progress for all models

        Returns:
            Dictionary of equipment_id -> ModelProgress
        """
        with self._lock:
            return dict(self._models)

    def get_global_status(self) -> Dict[str, Any]:
        """Get global training status

        Returns:
            Global status information
        """
        with self._lock:
            total_progress = 0.0
            if self._total_models > 0:
                total_progress = (self._completed_models / self._total_models) * 100.0

            return {
                'status': self._global_status.value,
                'total_models': self._total_models,
                'completed_models': self._completed_models,
                'progress_percent': total_progress,
                'models': len(self._models)
            }

    def _notify_observers(self):
        """Notify all observers of progress update"""
        progress_copy = dict(self._models)
        for observer in self._observers:
            try:
                observer(progress_copy)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")

    def reset(self):
        """Reset all progress tracking"""
        with self._lock:
            self._models.clear()
            self._global_status = TrainingStatus.NOT_STARTED
            self._total_models = 0
            self._completed_models = 0
            logger.info("Training progress tracker reset")


# Global instance
training_tracker = TrainingProgressTracker()