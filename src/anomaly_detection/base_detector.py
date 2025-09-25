"""
Base Detector Module for Anomaly Detection
Abstract base class and common functionality for all anomaly detectors
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import pickle
import joblib
import json
from collections import defaultdict, deque
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Container for anomaly detection results"""
    is_anomaly: np.ndarray  # Binary predictions
    anomaly_scores: np.ndarray  # Continuous scores
    threshold: float  # Decision threshold
    timestamps: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_anomalies(self) -> int:
        """Number of detected anomalies"""
        return np.sum(self.is_anomaly)
    
    @property
    def anomaly_ratio(self) -> float:
        """Ratio of anomalies in results"""
        return self.n_anomalies / len(self.is_anomaly) if len(self.is_anomaly) > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        df = pd.DataFrame({
            'is_anomaly': self.is_anomaly,
            'anomaly_score': self.anomaly_scores
        })
        
        if self.timestamps is not None:
            df['timestamp'] = self.timestamps
            
        return df


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    average_precision: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    threshold: float = 0.5
    
    @property
    def specificity(self) -> float:
        """Calculate specificity (true negative rate)"""
        total_negatives = self.true_negatives + self.false_positives
        return self.true_negatives / total_negatives if total_negatives > 0 else 0
    
    @property
    def balanced_accuracy(self) -> float:
        """Calculate balanced accuracy"""
        return (self.recall + self.specificity) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'average_precision': self.average_precision,
            'specificity': self.specificity,
            'balanced_accuracy': self.balanced_accuracy,
            'confusion_matrix': {
                'TP': self.true_positives,
                'FP': self.false_positives,
                'TN': self.true_negatives,
                'FN': self.false_negatives
            },
            'threshold': self.threshold
        }


class ThresholdStrategy(ABC):
    """Abstract base class for threshold determination strategies"""
    
    @abstractmethod
    def compute_threshold(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        """Compute threshold from scores"""
        pass


class PercentileThreshold(ThresholdStrategy):
    """Percentile-based threshold"""
    
    def __init__(self, percentile: float = 95):
        self.percentile = percentile
    
    def compute_threshold(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        return np.percentile(scores, self.percentile)


class StatisticalThreshold(ThresholdStrategy):
    """Statistical threshold (mean + k*std)"""
    
    def __init__(self, k: float = 3):
        self.k = k
    
    def compute_threshold(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        return np.mean(scores) + self.k * np.std(scores)


class OptimalThreshold(ThresholdStrategy):
    """Optimal threshold based on F1 score (requires labels)"""
    
    def compute_threshold(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        if labels is None:
            raise ValueError("Labels required for optimal threshold")
        
        # Try different thresholds
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = scores > threshold
            f1 = f1_score(labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detection models
    """
    
    def __init__(self,
                 name: str = "BaseDetector",
                 threshold_strategy: Optional[ThresholdStrategy] = None,
                 contamination: float = 0.1,
                 window_size: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize base anomaly detector
        
        Args:
            name: Detector name
            threshold_strategy: Strategy for threshold determination
            contamination: Expected proportion of anomalies
            window_size: Window size for sequence models
            random_state: Random seed
        """
        self.name = name
        self.threshold_strategy = threshold_strategy or PercentileThreshold(100 * (1 - contamination))
        self.contamination = contamination
        self.window_size = window_size
        self.random_state = random_state
        
        # Model state
        self.is_fitted = False
        self.threshold = None
        self.model = None
        
        # Training history
        self.training_history: Dict[str, List] = defaultdict(list)
        
        # Metrics
        self.train_metrics: Optional[ModelMetrics] = None
        self.test_metrics: Optional[ModelMetrics] = None
        
        # Feature information
        self.n_features = None
        self.feature_names: Optional[List[str]] = None
        
        # Cache for predictions
        self.prediction_cache: Dict[int, DetectionResult] = {}
        
        logger.info(f"{self.name} initialized with contamination={contamination}")
    
    @abstractmethod
    def _build_model(self):
        """Build the underlying model architecture"""
        pass
    
    @abstractmethod
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Internal fit implementation"""
        pass
    
    @abstractmethod
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Internal score prediction"""
        pass
    
    def fit(self, 
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            validation_split: float = 0.2,
            verbose: int = 1) -> 'BaseAnomalyDetector':
        """
        Fit the anomaly detector
        
        Args:
            X: Training data
            y: Optional labels for semi-supervised learning
            validation_split: Validation data ratio
            verbose: Verbosity level
            
        Returns:
            Self
        """
        # Validate input
        X = self._validate_input(X, training=True)
        
        # Store feature information
        if X.ndim == 2:
            self.n_features = X.shape[1]
        elif X.ndim == 3:
            self.n_features = X.shape[2]
        
        # Split data if labels provided
        if y is not None and validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None
        
        # Build and fit model
        if self.model is None:
            self._build_model()
        
        # Fit model
        self._fit(X_train, y_train)
        
        # Compute threshold
        train_scores = self._predict_scores(X_train)
        self.threshold = self.threshold_strategy.compute_threshold(train_scores, y_train)
        
        # Evaluate on training data
        if y_train is not None:
            self.train_metrics = self._evaluate(X_train, y_train)
            if verbose > 0:
                logger.info(f"Training metrics: F1={self.train_metrics.f1:.3f}, "
                          f"AUC={self.train_metrics.roc_auc:.3f}")
        
        # Evaluate on validation data
        if X_val is not None and y_val is not None:
            self.test_metrics = self._evaluate(X_val, y_val)
            if verbose > 0:
                logger.info(f"Validation metrics: F1={self.test_metrics.f1:.3f}, "
                          f"AUC={self.test_metrics.roc_auc:.3f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, 
                X: np.ndarray,
                threshold: Optional[float] = None,
                return_scores: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict anomalies
        
        Args:
            X: Data to predict
            threshold: Override threshold
            return_scores: Whether to return anomaly scores
            
        Returns:
            Binary predictions and optionally scores
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted")
        
        # Validate input
        X = self._validate_input(X)
        
        # Get scores
        scores = self._predict_scores(X)
        
        # Apply threshold
        threshold = threshold or self.threshold
        predictions = (scores > threshold).astype(int)
        
        if return_scores:
            return predictions, scores
        return predictions
    
    def detect(self, 
               X: np.ndarray,
               timestamps: Optional[np.ndarray] = None,
               threshold: Optional[float] = None) -> DetectionResult:
        """
        Detect anomalies and return detailed results
        
        Args:
            X: Data to analyze
            timestamps: Optional timestamps
            threshold: Override threshold
            
        Returns:
            Detection results
        """
        # Get predictions and scores
        predictions, scores = self.predict(X, threshold=threshold, return_scores=True)
        
        # Create result object
        result = DetectionResult(
            is_anomaly=predictions,
            anomaly_scores=scores,
            threshold=threshold or self.threshold,
            timestamps=timestamps,
            metadata={
                'detector': self.name,
                'n_samples': len(X),
                'detection_time': datetime.now()
            }
        )
        
        return result
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} not fitted")
        
        X = self._validate_input(X)
        return self._predict_scores(X)
    
    def _validate_input(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Validate and prepare input data
        
        Args:
            X: Input data
            training: Whether this is training data
            
        Returns:
            Validated data
        """
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        
        # Check for NaN/Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Input contains NaN/Inf values, replacing with 0")
            X = np.nan_to_num(X, 0)
        
        # Ensure correct shape
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check feature consistency
        if not training and self.n_features is not None:
            if X.ndim == 2 and X.shape[1] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
            elif X.ndim == 3 and X.shape[2] != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {X.shape[2]}")
        
        return X
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Model metrics
        """
        # Get predictions
        predictions, scores = self.predict(X, return_scores=True)
        
        # Calculate metrics
        metrics = ModelMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(y, predictions)
        metrics.precision = precision_score(y, predictions, zero_division=0)
        metrics.recall = recall_score(y, predictions, zero_division=0)
        metrics.f1 = f1_score(y, predictions, zero_division=0)
        
        # ROC AUC
        try:
            metrics.roc_auc = roc_auc_score(y, scores)
        except:
            metrics.roc_auc = 0.5
        
        # Average precision
        try:
            metrics.average_precision = average_precision_score(y, scores)
        except:
            metrics.average_precision = 0.0
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        metrics.true_positives = tp
        metrics.false_positives = fp
        metrics.true_negatives = tn
        metrics.false_negatives = fn
        metrics.threshold = self.threshold
        
        return metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        Public evaluation method
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        return self._evaluate(X, y)
    
    def optimize_threshold(self, 
                          X: np.ndarray,
                          y: np.ndarray,
                          metric: str = 'f1') -> float:
        """
        Optimize detection threshold
        
        Args:
            X: Validation data
            y: True labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        scores = self.score_samples(X)
        
        # Try different thresholds
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_score = 0
        best_threshold = self.threshold
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y, predictions, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y, predictions, zero_division=0)
            else:
                score = f1_score(y, predictions, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimized threshold: {best_threshold:.4f} (best {metric}: {best_score:.4f})")
        
        # Update threshold
        self.threshold = best_threshold
        return best_threshold
    
    def plot_metrics(self, 
                    X: np.ndarray,
                    y: np.ndarray,
                    save_path: Optional[Path] = None):
        """
        Plot evaluation metrics
        
        Args:
            X: Test data
            y: True labels
            save_path: Path to save plot
        """
        # Get predictions and scores
        predictions, scores = self.predict(X, return_scores=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, scores)
        avg_precision = average_precision_score(y, scores)
        axes[0, 2].plot(recall, precision, label=f'AP={avg_precision:.3f}')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Score Distribution
        axes[1, 0].hist(scores[y == 0], bins=50, alpha=0.5, label='Normal', color='blue')
        axes[1, 0].hist(scores[y == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
        axes[1, 0].axvline(self.threshold, color='green', linestyle='--', label=f'Threshold={self.threshold:.3f}')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        
        # 5. Threshold Impact
        thresholds = np.linspace(np.min(scores), np.max(scores), 50)
        f1_scores = []
        precisions = []
        recalls = []
        
        for t in thresholds:
            preds = (scores > t).astype(int)
            f1_scores.append(f1_score(y, preds, zero_division=0))
            precisions.append(precision_score(y, preds, zero_division=0))
            recalls.append(recall_score(y, preds, zero_division=0))
        
        axes[1, 1].plot(thresholds, f1_scores, label='F1', color='green')
        axes[1, 1].plot(thresholds, precisions, label='Precision', color='blue')
        axes[1, 1].plot(thresholds, recalls, label='Recall', color='red')
        axes[1, 1].axvline(self.threshold, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Threshold Impact on Metrics')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Detection Timeline (if we have enough samples)
        if len(predictions) > 100:
            sample_indices = np.arange(min(500, len(predictions)))
            axes[1, 2].scatter(sample_indices[predictions[:500] == 0], 
                             scores[:500][predictions[:500] == 0],
                             c='blue', alpha=0.5, s=10, label='Normal')
            axes[1, 2].scatter(sample_indices[predictions[:500] == 1],
                             scores[:500][predictions[:500] == 1],
                             c='red', alpha=0.7, s=20, label='Anomaly')
            axes[1, 2].axhline(self.threshold, color='green', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Sample Index')
            axes[1, 2].set_ylabel('Anomaly Score')
            axes[1, 2].set_title('Detection Timeline (First 500 samples)')
            axes[1, 2].legend()
        
        plt.suptitle(f'{self.name} - Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def save(self, filepath: Path):
        """
        Save detector state
        
        Args:
            filepath: Path to save file
        """
        state = {
            'name': self.name,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'model_state': self._get_model_state()
        }
        
        with open(filepath, 'wb') as f:
            joblib.dump(state, f)
        
        logger.info(f"Saved {self.name} to {filepath}")
    
    def load(self, filepath: Path):
        """
        Load detector state
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
        
        self.name = state['name']
        self.threshold = state['threshold']
        self.contamination = state['contamination']
        self.window_size = state['window_size']
        self.is_fitted = state['is_fitted']
        self.n_features = state['n_features']
        self.feature_names = state['feature_names']
        self.train_metrics = state['train_metrics']
        self.test_metrics = state['test_metrics']
        
        # Rebuild model and load state
        if self.model is None:
            self._build_model()
        self._set_model_state(state['model_state'])
        
        logger.info(f"Loaded {self.name} from {filepath}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model-specific state for saving"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model-specific state for loading"""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get detector parameters"""
        return {
            'name': self.name,
            'threshold': self.threshold,
            'contamination': self.contamination,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name}(fitted={self.is_fitted}, threshold={self.threshold:.4f if self.threshold else None})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()


class StreamingAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for streaming anomaly detection
    """
    
    def __init__(self,
                 name: str = "StreamingDetector",
                 buffer_size: int = 1000,
                 update_interval: int = 100,
                 **kwargs):
        """
        Initialize streaming detector
        
        Args:
            name: Detector name
            buffer_size: Size of data buffer
            update_interval: Model update interval
            **kwargs: Additional arguments for base class
        """
        super().__init__(name=name, **kwargs)
        
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Streaming buffers
        self.data_buffer = deque(maxlen=buffer_size)
        self.score_buffer = deque(maxlen=buffer_size)
        
        # Streaming statistics
        self.samples_seen = 0
        self.anomalies_detected = 0
        self.last_update = 0
    
    def update(self, sample: np.ndarray) -> Tuple[bool, float]:
        """
        Update with new sample and detect anomaly
        
        Args:
            sample: New data sample
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        # Add to buffer
        self.data_buffer.append(sample)
        self.samples_seen += 1
        
        # Check if model needs update
        if self.samples_seen - self.last_update >= self.update_interval:
            self._update_model()
            self.last_update = self.samples_seen
        
        # Detect anomaly
        if self.is_fitted:
            score = self._predict_scores(sample.reshape(1, -1))[0]
            is_anomaly = score > self.threshold
            
            # Update statistics
            if is_anomaly:
                self.anomalies_detected += 1
            
            # Add to score buffer
            self.score_buffer.append(score)
            
            return is_anomaly, score
        
        return False, 0.0
    
    def _update_model(self):
        """Update model with buffered data"""
        if len(self.data_buffer) < 10:
            return
        
        # Convert buffer to array
        X = np.array(self.data_buffer)
        
        # Update model (implementation specific)
        if not self.is_fitted:
            self.fit(X)
        else:
            # Incremental update if supported
            if hasattr(self, 'partial_fit'):
                self.partial_fit(X)
            else:
                # Update threshold based on recent scores
                if len(self.score_buffer) > 0:
                    recent_scores = np.array(self.score_buffer)
                    self.threshold = self.threshold_strategy.compute_threshold(recent_scores)
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'samples_seen': self.samples_seen,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(1, self.samples_seen),
            'buffer_size': len(self.data_buffer),
            'last_update': self.last_update
        }


if __name__ == "__main__":
    # Test base detector functionality
    print("\n" + "="*60)
    print("Testing Base Anomaly Detector")
    print("="*60)
    
    # Create a simple concrete implementation for testing
    class SimpleDetector(BaseAnomalyDetector):
        """Simple detector for testing"""
        
        def _build_model(self):
            """Simple threshold model"""
            self.model = {'type': 'simple'}
        
        def _fit(self, X, y=None):
            """Fit by calculating statistics"""
            self.mean = np.mean(X)
            self.std = np.std(X)
        
        def _predict_scores(self, X):
            """Score based on deviation from mean"""
            if not hasattr(self, 'mean'):
                return np.zeros(len(X))
            return np.abs(X.flatten() - self.mean) / (self.std + 1e-10)
        
        def _get_model_state(self):
            return {'mean': self.mean, 'std': self.std} if hasattr(self, 'mean') else {}
        
        def _set_model_state(self, state):
            if 'mean' in state:
                self.mean = state['mean']
                self.std = state['std']
    
    # Create test data
    np.random.seed(42)
    normal_data = np.random.randn(1000) * 2 + 5
    anomaly_data = np.random.randn(100) * 5 + 15
    
    X = np.concatenate([normal_data, anomaly_data])
    y = np.concatenate([np.zeros(1000), np.ones(100)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Test detector
    print("\n1. Testing Simple Detector...")
    detector = SimpleDetector(name="TestDetector", contamination=0.1)
    
    # Fit detector
    detector.fit(X.reshape(-1, 1), y, validation_split=0.2)
    
    # Get metrics
    if detector.test_metrics:
        print(f"\nValidation Metrics:")
        for key, value in detector.test_metrics.to_dict().items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
    
    # Test prediction
    test_samples = np.array([5, 10, 20])  # Normal, borderline, anomaly
    predictions = detector.predict(test_samples.reshape(-1, 1))
    scores = detector.score_samples(test_samples.reshape(-1, 1))
    
    print(f"\nTest Predictions:")
    for val, pred, score in zip(test_samples, predictions, scores):
        print(f"  Value: {val:.1f} -> Anomaly: {bool(pred)} (score: {score:.3f})")
    
    # Test threshold optimization
    print("\n2. Testing Threshold Optimization...")
    original_threshold = detector.threshold
    optimal_threshold = detector.optimize_threshold(X.reshape(-1, 1), y, metric='f1')
    print(f"  Original threshold: {original_threshold:.3f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    
    # Test detection result
    print("\n3. Testing Detection Results...")
    result = detector.detect(X[:100].reshape(-1, 1))
    print(f"  Detected {result.n_anomalies} anomalies out of 100 samples")
    print(f"  Anomaly ratio: {result.anomaly_ratio:.2%}")
    
    # Test streaming detector
    print("\n4. Testing Streaming Detector...")
    
    class SimpleStreamingDetector(StreamingAnomalyDetector, SimpleDetector):
        """Streaming version of simple detector"""
        pass
    
    stream_detector = SimpleStreamingDetector(
        name="StreamDetector",
        buffer_size=100,
        update_interval=50
    )
    
    # Simulate streaming
    streaming_results = []
    for i, value in enumerate(X[:200]):
        is_anomaly, score = stream_detector.update(np.array([value]))
        streaming_results.append((is_anomaly, score))
        
        if i == 100:
            stats = stream_detector.get_streaming_stats()
            print(f"  Streaming stats at sample 100: {stats}")
    
    # Plot metrics (if we have labels)
    print("\n5. Generating evaluation plots...")
    detector.plot_metrics(X.reshape(-1, 1), y)
    
    print("\n" + "="*60)
    print("Base detector test complete")
    print("="*60)
