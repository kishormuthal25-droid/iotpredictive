"""
Model Evaluator Module for Anomaly Detection
Evaluates, compares, and selects best performing anomaly detection models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import json
import pickle
from collections import defaultdict
import warnings
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, 
    classification_report, silhouette_score
)
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.anomaly_detection.base_detector import BaseAnomalyDetector, ModelMetrics
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for model evaluation results"""
    model_name: str
    metrics: ModelMetrics
    training_time: float
    inference_time: float
    model_size_mb: float
    cross_val_scores: Optional[Dict[str, List[float]]] = None
    confusion_matrix: Optional[np.ndarray] = None
    roc_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    pr_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'metrics': self.metrics.to_dict(),
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'model_size_mb': self.model_size_mb,
            'cross_val_scores': self.cross_val_scores,
            'metadata': self.metadata
        }


@dataclass
class ComparisonResult:
    """Container for model comparison results"""
    best_model: str
    ranking: List[Tuple[str, float]]
    results: Dict[str, EvaluationResult]
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    ensemble_weights: Optional[Dict[str, float]] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy visualization"""
        data = []
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': result.metrics.accuracy,
                'Precision': result.metrics.precision,
                'Recall': result.metrics.recall,
                'F1': result.metrics.f1,
                'ROC-AUC': result.metrics.roc_auc,
                'Training Time (s)': result.training_time,
                'Inference Time (ms)': result.inference_time * 1000,
                'Model Size (MB)': result.model_size_mb
            }
            data.append(row)
        return pd.DataFrame(data)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison framework
    """
    
    def __init__(self,
                 models: Optional[List[BaseAnomalyDetector]] = None,
                 metrics: Optional[List[str]] = None,
                 cv_strategy: str = 'stratified',
                 n_splits: int = 5,
                 random_state: int = 42):
        """
        Initialize model evaluator
        
        Args:
            models: List of models to evaluate
            metrics: Metrics to compute
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            random_state: Random seed
        """
        self.models = models or []
        self.metrics = metrics or ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        
        # Results storage
        self.evaluation_results: Dict[str, EvaluationResult] = {}
        self.comparison_result: Optional[ComparisonResult] = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        
        logger.info(f"ModelEvaluator initialized with {len(self.models)} models")
    
    def add_model(self, model: BaseAnomalyDetector):
        """Add model to evaluation"""
        self.models.append(model)
        logger.info(f"Added model: {model.name}")
    
    def evaluate_model(self,
                      model: BaseAnomalyDetector,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      measure_time: bool = True) -> EvaluationResult:
        """
        Evaluate single model
        
        Args:
            model: Model to evaluate
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            measure_time: Whether to measure timing
            
        Returns:
            Evaluation result
        """
        import time
        import os
        
        logger.info(f"Evaluating {model.name}...")
        
        # Training
        if measure_time:
            start_time = time.time()
        
        model.fit(X_train, y_train, verbose=0)
        
        if measure_time:
            training_time = time.time() - start_time
        else:
            training_time = 0
        
        # Inference
        if measure_time:
            start_time = time.time()
            predictions = model.predict(X_test)
            inference_time = (time.time() - start_time) / len(X_test)
        else:
            predictions = model.predict(X_test)
            inference_time = 0
        
        # Get scores for probability-based metrics
        scores = model.score_samples(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, predictions, scores)
        
        # Calculate model size
        model_size = self._estimate_model_size(model)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Get ROC curve data
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_data = (fpr, tpr)
        
        # Get PR curve data
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_data = (precision, recall)
        
        # Create result
        result = EvaluationResult(
            model_name=model.name,
            metrics=metrics,
            training_time=training_time,
            inference_time=inference_time,
            model_size_mb=model_size,
            confusion_matrix=cm,
            roc_data=roc_data,
            pr_data=pr_data,
            metadata={'threshold': model.threshold}
        )
        
        # Store result
        self.evaluation_results[model.name] = result
        
        # Track performance
        self.performance_history[model.name].append({
            'timestamp': datetime.now(),
            'metrics': metrics.to_dict(),
            'data_size': len(X_train) + len(X_test)
        })
        
        return result
    
    def cross_validate_model(self,
                           model: BaseAnomalyDetector,
                           X: np.ndarray,
                           y: np.ndarray) -> Dict[str, List[float]]:
        """
        Perform cross-validation on model
        
        Args:
            model: Model to evaluate
            X: Data
            y: Labels
            
        Returns:
            Cross-validation scores
        """
        # Select CV strategy
        if self.cv_strategy == 'stratified':
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.n_splits)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        cv_scores = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model for this fold
            fold_model = self._clone_model(model)
            
            # Train
            fold_model.fit(X_train, y_train, verbose=0)
            
            # Predict
            predictions = fold_model.predict(X_val)
            scores = fold_model.score_samples(X_val)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, predictions))
            cv_scores['precision'].append(precision_score(y_val, predictions, zero_division=0))
            cv_scores['recall'].append(recall_score(y_val, predictions, zero_division=0))
            cv_scores['f1'].append(f1_score(y_val, predictions, zero_division=0))
            
            try:
                cv_scores['roc_auc'].append(roc_auc_score(y_val, scores))
            except:
                cv_scores['roc_auc'].append(0.5)
        
        return dict(cv_scores)
    
    def compare_models(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      ranking_metric: str = 'f1') -> ComparisonResult:
        """
        Compare all models
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            ranking_metric: Metric for ranking
            
        Returns:
            Comparison result
        """
        logger.info(f"Comparing {len(self.models)} models...")
        
        # Evaluate each model
        for model in self.models:
            result = self.evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Add cross-validation scores
            cv_scores = self.cross_validate_model(model, X_train, y_train)
            result.cross_val_scores = cv_scores
        
        # Rank models
        ranking = self._rank_models(ranking_metric)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(X_test, y_test)
        
        # Calculate ensemble weights
        ensemble_weights = self._calculate_ensemble_weights()
        
        # Create comparison result
        self.comparison_result = ComparisonResult(
            best_model=ranking[0][0],
            ranking=ranking,
            results=self.evaluation_results,
            statistical_tests=statistical_tests,
            ensemble_weights=ensemble_weights
        )
        
        return self.comparison_result
    
    def _calculate_metrics(self, 
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         scores: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive metrics"""
        metrics = ModelMetrics()
        
        # Basic metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = precision_score(y_true, y_pred, zero_division=0)
        metrics.recall = recall_score(y_true, y_pred, zero_division=0)
        metrics.f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC
        try:
            metrics.roc_auc = roc_auc_score(y_true, scores)
        except:
            metrics.roc_auc = 0.5
        
        # Average precision
        try:
            metrics.average_precision = average_precision_score(y_true, scores)
        except:
            metrics.average_precision = 0.0
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.true_positives = tp
        metrics.false_positives = fp
        metrics.true_negatives = tn
        metrics.false_negatives = fn
        
        return metrics
    
    def _estimate_model_size(self, model: BaseAnomalyDetector) -> float:
        """Estimate model size in MB"""
        try:
            # Try to get actual model size
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=True) as tmp:
                model.save(Path(tmp.name))
                size_bytes = Path(tmp.name).stat().st_size
                return size_bytes / (1024 * 1024)
        except:
            # Fallback: estimate based on parameters
            if hasattr(model, 'model') and model.model is not None:
                if hasattr(model.model, 'count_params'):
                    # TensorFlow/Keras model
                    n_params = model.model.count_params()
                    # Assume 4 bytes per parameter
                    return (n_params * 4) / (1024 * 1024)
            return 0.0
    
    def _clone_model(self, model: BaseAnomalyDetector) -> BaseAnomalyDetector:
        """Create a clone of the model"""
        # Simple approach: create new instance with same parameters
        model_class = type(model)
        cloned = model_class(
            name=f"{model.name}_clone",
            threshold_strategy=model.threshold_strategy,
            contamination=model.contamination,
            window_size=model.window_size,
            random_state=model.random_state
        )
        
        # Copy additional attributes if needed
        if hasattr(model, 'config'):
            cloned.config = model.config
        
        return cloned
    
    def _rank_models(self, metric: str = 'f1') -> List[Tuple[str, float]]:
        """Rank models by specified metric"""
        ranking = []
        
        for model_name, result in self.evaluation_results.items():
            if metric == 'accuracy':
                score = result.metrics.accuracy
            elif metric == 'precision':
                score = result.metrics.precision
            elif metric == 'recall':
                score = result.metrics.recall
            elif metric == 'f1':
                score = result.metrics.f1
            elif metric == 'roc_auc':
                score = result.metrics.roc_auc
            elif metric == 'balanced_accuracy':
                score = result.metrics.balanced_accuracy
            else:
                score = result.metrics.f1
            
            ranking.append((model_name, score))
        
        # Sort by score (descending)
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def _perform_statistical_tests(self,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests between models"""
        tests = {}
        
        if len(self.models) < 2:
            return tests
        
        # Get predictions from all models
        predictions = {}
        for model in self.models:
            predictions[model.name] = model.predict(X_test)
        
        # Pairwise McNemar's test
        tests['mcnemar'] = {}
        model_names = list(predictions.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Create contingency table
                pred1 = predictions[model1]
                pred2 = predictions[model2]
                
                # Count disagreements
                n01 = np.sum((pred1 == y_test) & (pred2 != y_test))
                n10 = np.sum((pred1 != y_test) & (pred2 == y_test))
                
                # McNemar's test
                if n01 + n10 > 0:
                    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                    p_value = 1 - stats.chi2.cdf(statistic, df=1)
                else:
                    p_value = 1.0
                
                tests['mcnemar'][f"{model1}_vs_{model2}"] = {
                    'statistic': statistic if n01 + n10 > 0 else 0,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Friedman test for multiple models
        if len(self.models) > 2:
            # Collect scores
            scores = []
            for model in self.models:
                model_scores = model.score_samples(X_test)
                scores.append(model_scores)
            
            scores = np.array(scores).T
            
            # Friedman test
            statistic, p_value = stats.friedmanchisquare(*scores.T)
            tests['friedman'] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return tests
    
    def _calculate_ensemble_weights(self) -> Dict[str, float]:
        """Calculate optimal ensemble weights based on performance"""
        weights = {}
        
        # Use F1 scores as basis for weights
        total_f1 = sum(result.metrics.f1 for result in self.evaluation_results.values())
        
        if total_f1 > 0:
            for model_name, result in self.evaluation_results.items():
                weights[model_name] = result.metrics.f1 / total_f1
        else:
            # Equal weights if all models failed
            n_models = len(self.evaluation_results)
            for model_name in self.evaluation_results:
                weights[model_name] = 1.0 / n_models
        
        return weights
    
    def create_ensemble(self,
                       models: Optional[List[BaseAnomalyDetector]] = None,
                       weights: Optional[Dict[str, float]] = None,
                       voting: str = 'soft') -> 'EnsembleDetector':
        """
        Create ensemble detector
        
        Args:
            models: Models to ensemble (uses self.models if None)
            weights: Model weights (uses calculated if None)
            voting: Voting strategy ('soft' or 'hard')
            
        Returns:
            Ensemble detector
        """
        if models is None:
            models = self.models
        
        if weights is None:
            weights = self._calculate_ensemble_weights()
        
        return EnsembleDetector(models, weights, voting)
    
    def plot_comparison(self, save_path: Optional[Path] = None):
        """Plot comprehensive model comparison"""
        if not self.comparison_result:
            logger.warning("No comparison results to plot")
            return
        
        # Create subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Metrics comparison bar plot
        ax1 = plt.subplot(2, 3, 1)
        df = self.comparison_result.to_dataframe()
        metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        df[metrics_cols].plot(kind='bar', ax=ax1)
        ax1.set_xticklabels(df['Model'].values, rotation=45)
        ax1.set_title('Model Performance Metrics')
        ax1.set_ylabel('Score')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC curves
        ax2 = plt.subplot(2, 3, 2)
        for model_name, result in self.comparison_result.results.items():
            if result.roc_data:
                fpr, tpr = result.roc_data
                auc_score = result.metrics.roc_auc
                ax2.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall curves
        ax3 = plt.subplot(2, 3, 3)
        for model_name, result in self.comparison_result.results.items():
            if result.pr_data:
                precision, recall = result.pr_data
                ap = result.metrics.average_precision
                ax3.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Training and inference time
        ax4 = plt.subplot(2, 3, 4)
        models = list(self.comparison_result.results.keys())
        train_times = [r.training_time for r in self.comparison_result.results.values()]
        inference_times = [r.inference_time * 1000 for r in self.comparison_result.results.values()]
        
        x = np.arange(len(models))
        width = 0.35
        ax4.bar(x - width/2, train_times, width, label='Training (s)')
        ax4.bar(x + width/2, inference_times, width, label='Inference (ms)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.set_ylabel('Time')
        ax4.set_title('Computational Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cross-validation scores
        ax5 = plt.subplot(2, 3, 5)
        cv_data = []
        for model_name, result in self.comparison_result.results.items():
            if result.cross_val_scores:
                for metric, scores in result.cross_val_scores.items():
                    if metric == 'f1':
                        cv_data.append(scores)
        
        if cv_data:
            ax5.boxplot(cv_data, labels=models)
            ax5.set_xticklabels(models, rotation=45)
            ax5.set_ylabel('F1 Score')
            ax5.set_title('Cross-Validation F1 Scores')
            ax5.grid(True, alpha=0.3)
        
        # 6. Model ranking
        ax6 = plt.subplot(2, 3, 6)
        ranking_df = pd.DataFrame(self.comparison_result.ranking, columns=['Model', 'Score'])
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ranking_df)))
        ax6.barh(ranking_df['Model'], ranking_df['Score'], color=colors)
        ax6.set_xlabel('F1 Score')
        ax6.set_title('Model Ranking')
        ax6.grid(True, alpha=0.3)
        
        # Add best model annotation
        best_model = self.comparison_result.best_model
        ax6.text(0.02, 0.98, f"Best Model: {best_model}", 
                transform=ax6.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top')
        
        plt.suptitle('Model Comparison Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path: Optional[Path] = None) -> str:
        """Generate comprehensive evaluation report"""
        if not self.comparison_result:
            return "No evaluation results available"
        
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Best model
        report.append(f"BEST MODEL: {self.comparison_result.best_model}")
        report.append("")
        
        # Model ranking
        report.append("MODEL RANKING:")
        for i, (model, score) in enumerate(self.comparison_result.ranking, 1):
            report.append(f"  {i}. {model}: {score:.4f}")
        report.append("")
        
        # Detailed metrics
        report.append("DETAILED METRICS:")
        df = self.comparison_result.to_dataframe()
        report.append(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f'))
        report.append("")
        
        # Cross-validation results
        report.append("CROSS-VALIDATION RESULTS:")
        for model_name, result in self.comparison_result.results.items():
            if result.cross_val_scores:
                report.append(f"\n{model_name}:")
                for metric, scores in result.cross_val_scores.items():
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    report.append(f"  {metric}: {mean_score:.4f} (+/- {std_score:.4f})")
        report.append("")
        
        # Statistical tests
        if self.comparison_result.statistical_tests:
            report.append("STATISTICAL TESTS:")
            
            if 'mcnemar' in self.comparison_result.statistical_tests:
                report.append("\nMcNemar's Test (pairwise):")
                for comparison, test in self.comparison_result.statistical_tests['mcnemar'].items():
                    report.append(f"  {comparison}:")
                    report.append(f"    p-value: {test['p_value']:.4f}")
                    report.append(f"    Significant: {test['significant']}")
            
            if 'friedman' in self.comparison_result.statistical_tests:
                report.append("\nFriedman Test:")
                test = self.comparison_result.statistical_tests['friedman']
                report.append(f"  Statistic: {test['statistic']:.4f}")
                report.append(f"  p-value: {test['p_value']:.4f}")
                report.append(f"  Significant: {test['significant']}")
        report.append("")
        
        # Ensemble weights
        if self.comparison_result.ensemble_weights:
            report.append("ENSEMBLE WEIGHTS:")
            for model, weight in self.comparison_result.ensemble_weights.items():
                report.append(f"  {model}: {weight:.4f}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        best_model_result = self.comparison_result.results[self.comparison_result.best_model]
        
        if best_model_result.metrics.f1 > 0.8:
            report.append("  ✓ The best model shows excellent performance")
        elif best_model_result.metrics.f1 > 0.6:
            report.append("  ⚠ The best model shows moderate performance")
        else:
            report.append("  ✗ Model performance is poor, consider:")
            report.append("    - Feature engineering")
            report.append("    - Hyperparameter tuning")
            report.append("    - Different algorithms")
        
        if best_model_result.inference_time < 0.01:
            report.append("  ✓ Inference time is suitable for real-time applications")
        else:
            report.append("  ⚠ Consider optimization for real-time deployment")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def save_results(self, save_dir: Path):
        """Save all evaluation results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison result
        if self.comparison_result:
            with open(save_dir / 'comparison_result.pkl', 'wb') as f:
                pickle.dump(self.comparison_result, f)
            
            # Save as JSON for readability
            json_data = {
                'best_model': self.comparison_result.best_model,
                'ranking': self.comparison_result.ranking,
                'results': {
                    name: result.to_dict() 
                    for name, result in self.comparison_result.results.items()
                }
            }
            with open(save_dir / 'comparison_result.json', 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        
        # Save performance history
        with open(save_dir / 'performance_history.json', 'w') as f:
            json.dump(dict(self.performance_history), f, indent=2, default=str)
        
        logger.info(f"Results saved to {save_dir}")


class EnsembleDetector(BaseAnomalyDetector):
    """Ensemble of multiple anomaly detectors"""
    
    def __init__(self,
                 models: List[BaseAnomalyDetector],
                 weights: Optional[Dict[str, float]] = None,
                 voting: str = 'soft',
                 name: str = "EnsembleDetector"):
        """
        Initialize ensemble detector
        
        Args:
            models: List of base models
            weights: Model weights
            voting: Voting strategy ('soft' or 'hard')
            name: Ensemble name
        """
        super().__init__(name=name)
        
        self.models = models
        self.weights = weights or {model.name: 1.0/len(models) for model in models}
        self.voting = voting
        
    def _build_model(self):
        """Build is handled by individual models"""
        pass
    
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit all models"""
        for model in self.models:
            if hasattr(model, 'config'):
                # Handle models with config
                model.n_features = X.shape[-1] if X.ndim > 1 else 1
                model._build_model()
            model._fit(X, y)
    
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Combine predictions from all models"""
        if self.voting == 'soft':
            # Weighted average of scores
            combined_scores = np.zeros(len(X))
            
            for model in self.models:
                scores = model._predict_scores(X)
                weight = self.weights.get(model.name, 1.0)
                combined_scores += scores * weight
            
            return combined_scores
        else:
            # Hard voting (majority vote)
            predictions = []
            
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            # Weighted majority vote
            weighted_votes = np.zeros(len(X))
            
            for i, model in enumerate(self.models):
                weight = self.weights.get(model.name, 1.0)
                weighted_votes += predictions[i] * weight
            
            threshold = sum(self.weights.values()) / 2
            return weighted_votes
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get ensemble state"""
        return {
            'models': self.models,
            'weights': self.weights,
            'voting': self.voting
        }
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set ensemble state"""
        self.models = state['models']
        self.weights = state['weights']
        self.voting = state['voting']


if __name__ == "__main__":
    # Test model evaluator
    print("\n" + "="*60)
    print("Testing Model Evaluator")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Normal data
    X_normal = np.random.randn(n_samples, n_features)
    y_normal = np.zeros(n_samples)
    
    # Anomaly data
    X_anomaly = np.random.randn(100, n_features) * 3 + 2
    y_anomaly = np.ones(100)
    
    # Combine
    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([y_normal, y_anomaly])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split data
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Data shape: {X.shape}")
    print(f"Anomaly ratio: {np.mean(y):.2%}")
    
    # Create mock models for testing
    from src.anomaly_detection.lstm_detector import LSTMDetector, LSTMConfig
    from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig
    
    # Simple models for testing
    models = []
    
    # Model 1: LSTM
    config1 = LSTMConfig(lstm_units=[32, 16], epochs=5, sequence_length=10)
    model1 = LSTMDetector(name="LSTM", config=config1)
    models.append(model1)
    
    # Model 2: Autoencoder
    config2 = LSTMAutoencoderConfig(encoder_units=[32, 16], epochs=5, sequence_length=10)
    model2 = LSTMAutoencoder(name="Autoencoder", config=config2)
    models.append(model2)
    
    # Initialize evaluator
    print("\n1. Initializing Model Evaluator...")
    evaluator = ModelEvaluator(models=models, cv_strategy='stratified', n_splits=3)
    
    # Compare models
    print("\n2. Comparing Models...")
    comparison = evaluator.compare_models(X_train, y_train, X_test, y_test)
    
    print(f"\nBest Model: {comparison.best_model}")
    print("\nModel Ranking:")
    for rank, (model, score) in enumerate(comparison.ranking, 1):
        print(f"  {rank}. {model}: {score:.4f}")
    
    # Generate report
    print("\n3. Generating Report...")
    report = evaluator.generate_report()
    print("\n" + report[:500] + "...")  # Print first 500 chars
    
    # Plot comparison
    print("\n4. Plotting Comparison...")
    evaluator.plot_comparison()
    
    # Create ensemble
    print("\n5. Creating Ensemble...")
    ensemble = evaluator.create_ensemble(voting='soft')
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_score = f1_score(y_test, ensemble_pred)
    print(f"Ensemble F1 Score: {ensemble_score:.4f}")
    
    # Save results
    print("\n6. Saving Results...")
    save_dir = get_data_path('evaluation_results')
    evaluator.save_results(save_dir)
    print(f"Results saved to {save_dir}")
    
    print("\n" + "="*60)
    print("Model evaluator test complete")
    print("="*60)
