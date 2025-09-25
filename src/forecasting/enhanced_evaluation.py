"""
Enhanced Evaluation Framework
Comprehensive evaluation metrics for the enhanced forecasting and predictions module
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
from abc import ABC, abstractmethod

# Statistical and evaluation imports
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    r2_score, precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.forecasting.enhanced_forecaster import EnhancedForecastResult
from src.forecasting.failure_probability import FailurePrediction
from src.forecasting.scenario_analysis import ScenarioResult
from src.forecasting.risk_matrix import RiskAssessment
from config.settings import settings, get_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for comprehensive evaluation metrics"""

    # Forecast accuracy metrics
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    smape: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0

    # Uncertainty quantification metrics
    coverage_probability: float = 0.0
    interval_width: float = 0.0
    interval_score: float = 0.0  # Winkler score
    miscoverage_rate: float = 0.0
    sharpness: float = 0.0  # Average interval width
    calibration_slope: float = 0.0
    calibration_intercept: float = 0.0

    # Failure prediction metrics
    failure_precision: float = 0.0
    failure_recall: float = 0.0
    failure_f1: float = 0.0
    failure_auc: float = 0.0
    time_to_failure_mae: float = 0.0
    time_to_failure_accuracy: float = 0.0

    # Risk assessment metrics
    risk_correlation: float = 0.0
    risk_ranking_accuracy: float = 0.0
    false_alarm_rate: float = 0.0
    missed_detection_rate: float = 0.0

    # Scenario analysis metrics
    scenario_accuracy: float = 0.0
    cost_prediction_error: float = 0.0
    availability_prediction_error: float = 0.0

    # System-level metrics
    overall_performance_score: float = 0.0
    computational_efficiency: float = 0.0
    model_stability: float = 0.0

    # Metadata
    evaluation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_quality_score: float = 0.8
    confidence_score: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items()}


class BaseEvaluator(ABC):
    """Base class for evaluation components"""

    def __init__(self, name: str):
        self.name = name
        self.results = {}

    @abstractmethod
    def evaluate(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate predictions against ground truth"""
        pass

    def save_results(self, filepath: Path):
        """Save evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)


class ForecastAccuracyEvaluator(BaseEvaluator):
    """Evaluator for forecast accuracy metrics"""

    def __init__(self):
        super().__init__("ForecastAccuracy")

    def evaluate(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate forecast accuracy"""

        # Flatten arrays if needed
        pred_flat = predictions.flatten()
        truth_flat = ground_truth[:len(pred_flat)]

        # Basic accuracy metrics
        mae = mean_absolute_error(truth_flat, pred_flat)
        mse = mean_squared_error(truth_flat, pred_flat)
        rmse = np.sqrt(mse)

        # Percentage errors
        mape = mean_absolute_percentage_error(truth_flat, pred_flat) * 100
        smape = self._calculate_smape(truth_flat, pred_flat)

        # R-squared
        r2 = r2_score(truth_flat, pred_flat)

        # Directional accuracy
        directional_acc = self._calculate_directional_accuracy(truth_flat, pred_flat)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'smape': smape,
            'r2': r2,
            'directional_accuracy': directional_acc
        }

    def _calculate_smape(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(truth) + np.abs(pred)) / 2.0
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs(truth[mask] - pred[mask]) / denominator[mask]) * 100

    def _calculate_directional_accuracy(self, truth: np.ndarray, pred: np.ndarray) -> float:
        """Calculate directional accuracy"""
        if len(truth) < 2:
            return 0.0

        truth_diff = np.diff(truth)
        pred_diff = np.diff(pred)

        truth_direction = truth_diff > 0
        pred_direction = pred_diff > 0

        return np.mean(truth_direction == pred_direction) * 100


class UncertaintyEvaluator(BaseEvaluator):
    """Evaluator for uncertainty quantification metrics"""

    def __init__(self):
        super().__init__("UncertaintyQuantification")

    def evaluate(self, forecast_result: EnhancedForecastResult,
                ground_truth: np.ndarray) -> Dict[str, float]:
        """Evaluate uncertainty quantification"""

        predictions = forecast_result.predictions.flatten()
        truth = ground_truth[:len(predictions)]

        metrics = {}

        # Coverage probability and interval metrics
        if (forecast_result.confidence_lower is not None and
            forecast_result.confidence_upper is not None):

            lower = forecast_result.confidence_lower.flatten()
            upper = forecast_result.confidence_upper.flatten()

            # Coverage probability
            coverage = np.mean((truth >= lower) & (truth <= upper))
            metrics['coverage_probability'] = coverage

            # Interval width
            interval_width = np.mean(upper - lower)
            metrics['interval_width'] = interval_width
            metrics['sharpness'] = interval_width

            # Miscoverage rate
            metrics['miscoverage_rate'] = 1 - coverage

            # Interval score (Winkler score)
            alpha = 1 - forecast_result.confidence_level
            interval_score = self._calculate_interval_score(
                truth, lower, upper, alpha
            )
            metrics['interval_score'] = interval_score

        # Calibration metrics for quantile predictions
        if hasattr(forecast_result, 'quantile_predictions') and forecast_result.quantile_predictions:
            cal_slope, cal_intercept = self._evaluate_calibration(
                forecast_result.quantile_predictions, truth
            )
            metrics['calibration_slope'] = cal_slope
            metrics['calibration_intercept'] = cal_intercept

        return metrics

    def _calculate_interval_score(self, truth: np.ndarray, lower: np.ndarray,
                                upper: np.ndarray, alpha: float) -> float:
        """Calculate interval score (Winkler score)"""

        width = upper - lower
        penalty_lower = (2.0 / alpha) * (lower - truth) * (truth < lower)
        penalty_upper = (2.0 / alpha) * (truth - upper) * (truth > upper)

        score = width + penalty_lower + penalty_upper
        return np.mean(score)

    def _evaluate_calibration(self, quantile_predictions: Dict[float, np.ndarray],
                            truth: np.ndarray) -> Tuple[float, float]:
        """Evaluate calibration of quantile predictions"""

        quantiles = sorted(quantile_predictions.keys())
        observed_frequencies = []
        expected_frequencies = []

        for q in quantiles:
            pred_quantile = quantile_predictions[q]
            observed_freq = np.mean(truth <= pred_quantile)
            observed_frequencies.append(observed_freq)
            expected_frequencies.append(q)

        # Linear regression to get slope and intercept
        slope, intercept = np.polyfit(expected_frequencies, observed_frequencies, 1)

        return slope, intercept


class FailurePredictionEvaluator(BaseEvaluator):
    """Evaluator for failure prediction metrics"""

    def __init__(self):
        super().__init__("FailurePrediction")

    def evaluate(self, predictions: List[FailurePrediction],
                ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate failure predictions"""

        # Extract failure probabilities and true failures
        failure_probs = [p.failure_probability for p in predictions]
        failure_times = [p.time_to_failure for p in predictions if p.time_to_failure]

        true_failures = ground_truth.get('failures', [])
        true_failure_times = ground_truth.get('failure_times', [])

        metrics = {}

        if true_failures:
            # Binary classification metrics
            pred_binary = [1 if p > 0.5 else 0 for p in failure_probs]

            metrics['failure_precision'] = precision_score(true_failures, pred_binary)
            metrics['failure_recall'] = recall_score(true_failures, pred_binary)
            metrics['failure_f1'] = f1_score(true_failures, pred_binary)

            # AUC score
            if len(set(true_failures)) > 1:  # Need both classes for AUC
                metrics['failure_auc'] = roc_auc_score(true_failures, failure_probs)

        # Time-to-failure accuracy
        if failure_times and true_failure_times:
            min_len = min(len(failure_times), len(true_failure_times))
            ttf_mae = mean_absolute_error(
                true_failure_times[:min_len],
                failure_times[:min_len]
            )
            metrics['time_to_failure_mae'] = ttf_mae

            # Time-to-failure accuracy within tolerance
            tolerance_hours = 24  # 24-hour tolerance
            accurate_predictions = np.abs(
                np.array(true_failure_times[:min_len]) -
                np.array(failure_times[:min_len])
            ) <= tolerance_hours

            metrics['time_to_failure_accuracy'] = np.mean(accurate_predictions) * 100

        return metrics


class RiskAssessmentEvaluator(BaseEvaluator):
    """Evaluator for risk assessment metrics"""

    def __init__(self):
        super().__init__("RiskAssessment")

    def evaluate(self, risk_assessments: List[RiskAssessment],
                ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate risk assessments"""

        risk_scores = [r.overall_risk_score for r in risk_assessments]
        true_incidents = ground_truth.get('incidents', [])

        metrics = {}

        if true_incidents and len(true_incidents) == len(risk_scores):
            # Correlation between risk scores and actual incidents
            correlation = np.corrcoef(risk_scores, true_incidents)[0, 1]
            metrics['risk_correlation'] = correlation if not np.isnan(correlation) else 0.0

            # Ranking accuracy (top-k accuracy)
            risk_ranking = np.argsort(risk_scores)[::-1]  # Descending order
            incident_ranking = np.argsort(true_incidents)[::-1]

            # Top-10% ranking accuracy
            top_k = max(1, len(risk_scores) // 10)
            top_risk_components = set(risk_ranking[:top_k])
            top_incident_components = set(incident_ranking[:top_k])

            ranking_accuracy = len(top_risk_components & top_incident_components) / top_k
            metrics['risk_ranking_accuracy'] = ranking_accuracy * 100

            # False alarm and missed detection rates
            high_risk_threshold = 0.7
            high_risk_pred = [1 if r > high_risk_threshold else 0 for r in risk_scores]
            high_incident_true = [1 if i > 0 else 0 for i in true_incidents]

            cm = confusion_matrix(high_incident_true, high_risk_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['false_alarm_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['missed_detection_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        return metrics


class ScenarioAnalysisEvaluator(BaseEvaluator):
    """Evaluator for scenario analysis metrics"""

    def __init__(self):
        super().__init__("ScenarioAnalysis")

    def evaluate(self, scenario_results: List[ScenarioResult],
                ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate scenario analysis results"""

        metrics = {}

        if not scenario_results:
            return metrics

        # Cost prediction accuracy
        predicted_costs = [r.total_cost for r in scenario_results]
        true_costs = ground_truth.get('actual_costs', [])

        if true_costs and len(true_costs) == len(predicted_costs):
            cost_error = mean_absolute_percentage_error(true_costs, predicted_costs) * 100
            metrics['cost_prediction_error'] = cost_error

        # Availability prediction accuracy
        predicted_availability = [r.system_availability for r in scenario_results]
        true_availability = ground_truth.get('actual_availability', [])

        if true_availability and len(true_availability) == len(predicted_availability):
            avail_error = mean_absolute_error(true_availability, predicted_availability) * 100
            metrics['availability_prediction_error'] = avail_error

        # Scenario ranking accuracy
        if len(scenario_results) > 1:
            cost_ranking = np.argsort([r.total_cost for r in scenario_results])
            if true_costs:
                true_cost_ranking = np.argsort(true_costs)

                # Spearman rank correlation
                scenario_accuracy = stats.spearmanr(cost_ranking, true_cost_ranking)[0]
                metrics['scenario_accuracy'] = scenario_accuracy if not np.isnan(scenario_accuracy) else 0.0

        return metrics


class EnhancedEvaluationFramework:
    """Comprehensive evaluation framework for enhanced forecasting module"""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize evaluation framework"""
        self.config = config or self._get_default_config()

        # Initialize evaluators
        self.forecast_evaluator = ForecastAccuracyEvaluator()
        self.uncertainty_evaluator = UncertaintyEvaluator()
        self.failure_evaluator = FailurePredictionEvaluator()
        self.risk_evaluator = RiskAssessmentEvaluator()
        self.scenario_evaluator = ScenarioAnalysisEvaluator()

        # Results storage
        self.evaluation_results = {}
        self.benchmark_results = {}

    def _get_default_config(self) -> Dict:
        """Get default evaluation configuration"""
        return {
            'evaluation_metrics': [
                'mae', 'rmse', 'mape', 'r2', 'coverage_probability',
                'failure_f1', 'risk_correlation', 'scenario_accuracy'
            ],
            'confidence_levels': [0.8, 0.9, 0.95],
            'tolerance_hours': 24,
            'benchmark_models': ['naive', 'linear_trend', 'seasonal_naive'],
            'cross_validation_folds': 5,
            'bootstrap_samples': 1000
        }

    def comprehensive_evaluation(self,
                               forecast_results: List[EnhancedForecastResult],
                               failure_predictions: List[FailurePrediction],
                               risk_assessments: List[RiskAssessment],
                               scenario_results: List[ScenarioResult],
                               ground_truth: Dict[str, Any]) -> EvaluationMetrics:
        """Perform comprehensive evaluation of all components"""

        logger.info("Starting comprehensive evaluation...")

        combined_metrics = EvaluationMetrics()

        # Evaluate forecasting accuracy
        if forecast_results and 'actual_values' in ground_truth:
            logger.info("Evaluating forecast accuracy...")

            predictions = np.concatenate([fr.predictions for fr in forecast_results])
            actual_values = ground_truth['actual_values']

            forecast_metrics = self.forecast_evaluator.evaluate(predictions, actual_values)

            combined_metrics.mae = forecast_metrics.get('mae', 0)
            combined_metrics.rmse = forecast_metrics.get('rmse', 0)
            combined_metrics.mape = forecast_metrics.get('mape', 0)
            combined_metrics.r2 = forecast_metrics.get('r2', 0)
            combined_metrics.directional_accuracy = forecast_metrics.get('directional_accuracy', 0)

        # Evaluate uncertainty quantification
        if forecast_results:
            logger.info("Evaluating uncertainty quantification...")

            uncertainty_metrics = []
            for fr in forecast_results:
                if 'actual_values' in ground_truth:
                    metrics = self.uncertainty_evaluator.evaluate(fr, ground_truth['actual_values'])
                    uncertainty_metrics.append(metrics)

            if uncertainty_metrics:
                combined_metrics.coverage_probability = np.mean([
                    m.get('coverage_probability', 0) for m in uncertainty_metrics
                ])
                combined_metrics.interval_width = np.mean([
                    m.get('interval_width', 0) for m in uncertainty_metrics
                ])
                combined_metrics.interval_score = np.mean([
                    m.get('interval_score', 0) for m in uncertainty_metrics
                ])

        # Evaluate failure predictions
        if failure_predictions:
            logger.info("Evaluating failure predictions...")

            failure_metrics = self.failure_evaluator.evaluate(
                failure_predictions, ground_truth
            )

            combined_metrics.failure_precision = failure_metrics.get('failure_precision', 0)
            combined_metrics.failure_recall = failure_metrics.get('failure_recall', 0)
            combined_metrics.failure_f1 = failure_metrics.get('failure_f1', 0)
            combined_metrics.failure_auc = failure_metrics.get('failure_auc', 0)
            combined_metrics.time_to_failure_mae = failure_metrics.get('time_to_failure_mae', 0)

        # Evaluate risk assessments
        if risk_assessments:
            logger.info("Evaluating risk assessments...")

            risk_metrics = self.risk_evaluator.evaluate(risk_assessments, ground_truth)

            combined_metrics.risk_correlation = risk_metrics.get('risk_correlation', 0)
            combined_metrics.risk_ranking_accuracy = risk_metrics.get('risk_ranking_accuracy', 0)
            combined_metrics.false_alarm_rate = risk_metrics.get('false_alarm_rate', 0)

        # Evaluate scenario analysis
        if scenario_results:
            logger.info("Evaluating scenario analysis...")

            scenario_metrics = self.scenario_evaluator.evaluate(scenario_results, ground_truth)

            combined_metrics.scenario_accuracy = scenario_metrics.get('scenario_accuracy', 0)
            combined_metrics.cost_prediction_error = scenario_metrics.get('cost_prediction_error', 0)
            combined_metrics.availability_prediction_error = scenario_metrics.get('availability_prediction_error', 0)

        # Calculate overall performance score
        combined_metrics.overall_performance_score = self._calculate_overall_score(combined_metrics)

        # Store results
        self.evaluation_results['comprehensive'] = combined_metrics

        logger.info("Comprehensive evaluation completed")
        return combined_metrics

    def _calculate_overall_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall performance score"""

        # Weights for different metric categories
        weights = {
            'forecast_accuracy': 0.3,
            'uncertainty_quality': 0.2,
            'failure_prediction': 0.25,
            'risk_assessment': 0.15,
            'scenario_analysis': 0.1
        }

        # Normalize metrics to 0-1 scale (higher is better)
        forecast_score = max(0, 1 - metrics.mape / 100)  # Lower MAPE is better
        uncertainty_score = metrics.coverage_probability  # Higher coverage is better
        failure_score = metrics.failure_f1  # Higher F1 is better
        risk_score = abs(metrics.risk_correlation)  # Higher correlation is better
        scenario_score = max(0, 1 - metrics.cost_prediction_error / 100)  # Lower error is better

        # Weighted average
        overall_score = (
            weights['forecast_accuracy'] * forecast_score +
            weights['uncertainty_quality'] * uncertainty_score +
            weights['failure_prediction'] * failure_score +
            weights['risk_assessment'] * risk_score +
            weights['scenario_analysis'] * scenario_score
        )

        return overall_score

    def benchmark_comparison(self,
                           enhanced_results: EvaluationMetrics,
                           baseline_results: Dict[str, EvaluationMetrics]) -> Dict[str, float]:
        """Compare enhanced forecasting with baseline methods"""

        comparison = {}

        for baseline_name, baseline_metrics in baseline_results.items():
            improvements = {}

            # Compare key metrics
            if baseline_metrics.rmse > 0:
                improvements['rmse_improvement'] = (
                    (baseline_metrics.rmse - enhanced_results.rmse) / baseline_metrics.rmse * 100
                )

            if baseline_metrics.mape > 0:
                improvements['mape_improvement'] = (
                    (baseline_metrics.mape - enhanced_results.mape) / baseline_metrics.mape * 100
                )

            improvements['r2_improvement'] = enhanced_results.r2 - baseline_metrics.r2
            improvements['coverage_improvement'] = (
                enhanced_results.coverage_probability - baseline_metrics.coverage_probability
            )

            comparison[baseline_name] = improvements

        return comparison

    def generate_evaluation_report(self, output_path: Path) -> str:
        """Generate comprehensive evaluation report"""

        if not self.evaluation_results:
            return "No evaluation results available"

        report_content = []
        report_content.append("# Enhanced Forecasting & Predictions Module - Evaluation Report")
        report_content.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("\n## Executive Summary")

        if 'comprehensive' in self.evaluation_results:
            metrics = self.evaluation_results['comprehensive']

            report_content.append(f"- **Overall Performance Score**: {metrics.overall_performance_score:.3f}")
            report_content.append(f"- **Forecast RMSE**: {metrics.rmse:.4f}")
            report_content.append(f"- **Forecast R²**: {metrics.r2:.4f}")
            report_content.append(f"- **Coverage Probability**: {metrics.coverage_probability:.3f}")
            report_content.append(f"- **Failure Prediction F1**: {metrics.failure_f1:.3f}")
            report_content.append(f"- **Risk Correlation**: {metrics.risk_correlation:.3f}")

        report_content.append("\n## Detailed Metrics")

        # Add detailed tables and analyses here...

        report_text = "\n".join(report_content)

        # Save report
        with open(output_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Evaluation report saved to {output_path}")
        return report_text

    def save_evaluation_results(self, filepath: Path):
        """Save all evaluation results to file"""

        results_data = {
            'evaluation_results': {
                name: metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
                for name, metrics in self.evaluation_results.items()
            },
            'benchmark_results': {
                name: metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
                for name, metrics in self.benchmark_results.items()
            },
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to {filepath}")


if __name__ == "__main__":
    # Demo and testing
    print("\n" + "="*60)
    print("Testing Enhanced Evaluation Framework")
    print("="*60)

    # Create evaluation framework
    evaluator = EnhancedEvaluationFramework()

    # Create mock data for testing
    print("\n1. Creating mock evaluation data...")

    # Mock forecast results
    mock_forecast_results = []
    for i in range(3):
        from src.forecasting.enhanced_forecaster import EnhancedForecastResult
        result = EnhancedForecastResult(
            predictions=np.random.normal(50, 10, 24),
            confidence_lower=np.random.normal(40, 8, 24),
            confidence_upper=np.random.normal(60, 12, 24),
            confidence_level=0.95,
            horizon=24
        )
        mock_forecast_results.append(result)

    # Mock ground truth
    mock_ground_truth = {
        'actual_values': np.random.normal(52, 8, 72),
        'failures': [0, 1, 0, 1, 0],
        'failure_times': [48, 72, None, 96, None],
        'incidents': [0.1, 0.8, 0.3, 0.9, 0.2],
        'actual_costs': [45000, 38000, 52000],
        'actual_availability': [0.92, 0.88, 0.95]
    }

    # Mock failure predictions
    from src.forecasting.failure_probability import FailurePrediction, SeverityLevel
    mock_failure_predictions = [
        FailurePrediction(
            equipment_id='SMAP',
            component_id='power_system',
            failure_probability=0.7,
            time_to_failure=48,
            severity=SeverityLevel.HIGH
        ),
        FailurePrediction(
            equipment_id='MSL',
            component_id='mobility',
            failure_probability=0.3,
            time_to_failure=96,
            severity=SeverityLevel.MEDIUM
        )
    ]

    # Run comprehensive evaluation
    print("\n2. Running comprehensive evaluation...")

    metrics = evaluator.comprehensive_evaluation(
        forecast_results=mock_forecast_results,
        failure_predictions=mock_failure_predictions,
        risk_assessments=[],  # Empty for demo
        scenario_results=[],  # Empty for demo
        ground_truth=mock_ground_truth
    )

    print(f"\nEvaluation Results:")
    print(f"  Overall Performance Score: {metrics.overall_performance_score:.3f}")
    print(f"  Forecast RMSE: {metrics.rmse:.4f}")
    print(f"  Forecast R²: {metrics.r2:.4f}")
    print(f"  Coverage Probability: {metrics.coverage_probability:.3f}")
    print(f"  Failure F1 Score: {metrics.failure_f1:.3f}")

    # Generate evaluation report
    print("\n3. Generating evaluation report...")

    report_path = Path("evaluation_report_demo.md")
    report = evaluator.generate_evaluation_report(report_path)

    print(f"Report generated and saved to {report_path}")

    print("\n" + "="*60)
    print("Enhanced evaluation framework test complete")
    print("="*60)