"""
Forecast Evaluator Module for Time Series Anomaly Detection
Comprehensive evaluation and analysis of forecasting models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest, kstest, shapiro
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    r2_score, explained_variance_score, median_absolute_error,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, accuracy_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Container for forecast evaluation metrics"""
    # Regression metrics
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    smape: float = 0.0  # Symmetric MAPE
    r2: float = 0.0
    explained_variance: float = 0.0
    median_ae: float = 0.0
    
    # Direction accuracy
    directional_accuracy: float = 0.0
    
    # Anomaly detection metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    specificity: float = 0.0
    
    # Statistical metrics
    bias: float = 0.0
    variance: float = 0.0
    max_error: float = 0.0
    
    # Time-based metrics
    lag_correlation: Dict[int, float] = field(default_factory=dict)
    autocorrelation: Dict[int, float] = field(default_factory=dict)
    
    # Confidence metrics
    coverage_probability: float = 0.0  # For prediction intervals
    interval_width: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'mape': self.mape,
            'smape': self.smape,
            'r2': self.r2,
            'explained_variance': self.explained_variance,
            'median_ae': self.median_ae,
            'directional_accuracy': self.directional_accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'specificity': self.specificity,
            'bias': self.bias,
            'variance': self.variance,
            'max_error': self.max_error,
            'lag_correlation': self.lag_correlation,
            'autocorrelation': self.autocorrelation,
            'coverage_probability': self.coverage_probability,
            'interval_width': self.interval_width
        }


class ForecastEvaluator:
    """Comprehensive evaluator for time series forecasting models"""
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 anomaly_threshold: float = 3.0):
        """Initialize Forecast Evaluator
        
        Args:
            confidence_level: Confidence level for prediction intervals
            anomaly_threshold: Threshold for anomaly detection
        """
        self.confidence_level = confidence_level
        self.anomaly_threshold = anomaly_threshold
        self.evaluation_results = {}
        self.comparison_results = {}
        
        logger.info(f"Initialized Forecast Evaluator with confidence level: {confidence_level}")
        
    def evaluate_forecast(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_anomaly_true: Optional[np.ndarray] = None,
                          y_anomaly_pred: Optional[np.ndarray] = None,
                          lower_bound: Optional[np.ndarray] = None,
                          upper_bound: Optional[np.ndarray] = None,
                          model_name: str = "model") -> ForecastMetrics:
        """Evaluate forecast performance comprehensively
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_anomaly_true: True anomaly labels
            y_anomaly_pred: Predicted anomaly labels
            lower_bound: Lower prediction interval
            upper_bound: Upper prediction interval
            model_name: Name of the model being evaluated
            
        Returns:
            ForecastMetrics object with all metrics
        """
        metrics = ForecastMetrics()
        
        # Ensure arrays are same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Flatten if needed
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
            
        # Calculate regression metrics
        metrics.mae = mean_absolute_error(y_true, y_pred)
        metrics.mse = mean_squared_error(y_true, y_pred)
        metrics.rmse = np.sqrt(metrics.mse)
        
        # MAPE with zero handling
        mask = y_true != 0
        if np.any(mask):
            metrics.mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics.mape = np.inf
            
        # SMAPE (Symmetric MAPE)
        metrics.smape = self._calculate_smape(y_true, y_pred)
        
        # R2 and explained variance
        metrics.r2 = r2_score(y_true, y_pred)
        metrics.explained_variance = explained_variance_score(y_true, y_pred)
        metrics.median_ae = median_absolute_error(y_true, y_pred)
        
        # Directional accuracy
        metrics.directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)
        
        # Statistical metrics
        errors = y_true - y_pred
        metrics.bias = np.mean(errors)
        metrics.variance = np.var(errors)
        metrics.max_error = np.max(np.abs(errors))
        
        # Lag correlations
        metrics.lag_correlation = self._calculate_lag_correlations(y_true, y_pred)
        
        # Autocorrelation of residuals
        metrics.autocorrelation = self._calculate_autocorrelation(errors)
        
        # Anomaly detection metrics if provided
        if y_anomaly_true is not None and y_anomaly_pred is not None:
            metrics = self._calculate_anomaly_metrics(
                metrics, y_anomaly_true, y_anomaly_pred
            )
            
        # Prediction interval metrics if provided
        if lower_bound is not None and upper_bound is not None:
            metrics = self._calculate_interval_metrics(
                metrics, y_true, lower_bound, upper_bound
            )
            
        # Store results
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Evaluation completed for {model_name}: RMSE={metrics.rmse:.4f}, MAE={metrics.mae:.4f}")
        
        return metrics
        
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denominator != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (correct trend prediction)"""
        if len(y_true) < 2:
            return 0.0
            
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction) * 100
        
    def _calculate_lag_correlations(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   max_lag: int = 10) -> Dict[int, float]:
        """Calculate correlation at different lags"""
        correlations = {}
        for lag in range(0, min(max_lag, len(y_true) // 4)):
            if lag == 0:
                correlations[lag] = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                correlations[lag] = np.corrcoef(y_true[:-lag], y_pred[lag:])[0, 1]
        return correlations
        
    def _calculate_autocorrelation(self, residuals: np.ndarray,
                                  max_lag: int = 20) -> Dict[int, float]:
        """Calculate autocorrelation of residuals"""
        autocorr = {}
        for lag in range(1, min(max_lag, len(residuals) // 4)):
            autocorr[lag] = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        return autocorr
        
    def _calculate_anomaly_metrics(self, metrics: ForecastMetrics,
                                  y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> ForecastMetrics:
        """Calculate anomaly detection metrics"""
        # Ensure binary labels
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        metrics.accuracy = accuracy_score(y_true, y_pred)
        metrics.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics.f1_score = f1_score(y_true, y_pred)
        metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
        
    def _calculate_interval_metrics(self, metrics: ForecastMetrics,
                                   y_true: np.ndarray,
                                   lower_bound: np.ndarray,
                                   upper_bound: np.ndarray) -> ForecastMetrics:
        """Calculate prediction interval metrics"""
        # Coverage probability
        within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        metrics.coverage_probability = np.mean(within_interval)
        
        # Average interval width
        metrics.interval_width = np.mean(upper_bound - lower_bound)
        
        return metrics
        
    def evaluate_multi_step_forecast(self,
                                    y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    horizon: int,
                                    model_name: str = "model") -> Dict[int, ForecastMetrics]:
        """Evaluate multi-step forecast at each horizon
        
        Args:
            y_true: True values (shape: [samples, horizon, features])
            y_pred: Predicted values (shape: [samples, horizon, features])
            horizon: Forecast horizon
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics for each step
        """
        step_metrics = {}
        
        for step in range(horizon):
            if len(y_true.shape) == 3:
                step_true = y_true[:, step, :]
                step_pred = y_pred[:, step, :]
            else:
                step_true = y_true[:, step]
                step_pred = y_pred[:, step]
                
            step_metrics[step + 1] = self.evaluate_forecast(
                step_true, step_pred,
                model_name=f"{model_name}_step_{step + 1}"
            )
            
        return step_metrics
        
    def compare_models(self, 
                      models_results: Dict[str, ForecastMetrics],
                      metric_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Compare multiple models based on their metrics
        
        Args:
            models_results: Dictionary of model names to metrics
            metric_weights: Weights for different metrics in scoring
            
        Returns:
            DataFrame with model comparison
        """
        if metric_weights is None:
            metric_weights = {
                'rmse': 0.25,
                'mae': 0.20,
                'mape': 0.15,
                'r2': 0.20,
                'f1_score': 0.20
            }
            
        comparison_data = []
        
        for model_name, metrics in models_results.items():
            row = {
                'Model': model_name,
                'RMSE': metrics.rmse,
                'MAE': metrics.mae,
                'MAPE': metrics.mape,
                'R2': metrics.r2,
                'F1-Score': metrics.f1_score,
                'Directional Accuracy': metrics.directional_accuracy
            }
            
            # Calculate weighted score (normalize first)
            score = 0
            for metric, weight in metric_weights.items():
                if hasattr(metrics, metric):
                    value = getattr(metrics, metric)
                    # Inverse for error metrics (lower is better)
                    if metric in ['rmse', 'mae', 'mse', 'mape']:
                        score -= weight * value
                    else:
                        score += weight * value
            row['Weighted Score'] = score
            
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Weighted Score', ascending=False)
        
        self.comparison_results = df
        return df
        
    def statistical_tests(self,
                         residuals1: np.ndarray,
                         residuals2: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform statistical tests on forecast residuals
        
        Args:
            residuals1: Residuals from first model
            residuals2: Optional residuals from second model for comparison
            
        Returns:
            Dictionary with test results
        """
        tests = {}
        
        # Normality tests for residuals1
        _, p_normal = normaltest(residuals1)
        tests['normality_test'] = {
            'p_value': p_normal,
            'is_normal': p_normal > 0.05
        }
        
        # Shapiro-Wilk test (for smaller samples)
        if len(residuals1) < 5000:
            _, p_shapiro = shapiro(residuals1)
            tests['shapiro_test'] = {
                'p_value': p_shapiro,
                'is_normal': p_shapiro > 0.05
            }
            
        # Ljung-Box test for autocorrelation
        tests['ljung_box'] = self._ljung_box_test(residuals1)
        
        # If comparing two models
        if residuals2 is not None:
            # Diebold-Mariano test
            tests['diebold_mariano'] = self._diebold_mariano_test(residuals1, residuals2)
            
            # Paired t-test
            _, p_ttest = stats.ttest_rel(np.abs(residuals1), np.abs(residuals2))
            tests['paired_ttest'] = {
                'p_value': p_ttest,
                'significant_difference': p_ttest < 0.05
            }
            
        return tests
        
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """Perform Ljung-Box test for autocorrelation"""
        n = len(residuals)
        acf = [np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, lags + 1)]
        
        # Calculate Ljung-Box statistic
        lb_statistic = n * (n + 2) * sum([(acf[i]**2) / (n - i - 1) for i in range(lags)])
        
        # Chi-square test
        p_value = 1 - stats.chi2.cdf(lb_statistic, lags)
        
        return {
            'statistic': lb_statistic,
            'p_value': p_value,
            'has_autocorrelation': p_value < 0.05
        }
        
    def _diebold_mariano_test(self, residuals1: np.ndarray, 
                             residuals2: np.ndarray) -> Dict[str, Any]:
        """Perform Diebold-Mariano test for forecast comparison"""
        # Calculate loss differential
        d = residuals1**2 - residuals2**2
        
        # Calculate test statistic
        mean_d = np.mean(d)
        var_d = np.var(d)
        n = len(d)
        
        dm_statistic = mean_d / np.sqrt(var_d / n)
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_statistic)))
        
        return {
            'statistic': dm_statistic,
            'p_value': p_value,
            'model1_better': dm_statistic < 0 and p_value < 0.05,
            'model2_better': dm_statistic > 0 and p_value < 0.05,
            'no_difference': p_value >= 0.05
        }
        
    def plot_forecast_results(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            title: str = "Forecast vs Actual",
                            lower_bound: Optional[np.ndarray] = None,
                            upper_bound: Optional[np.ndarray] = None,
                            anomalies: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None) -> go.Figure:
        """Create interactive plot of forecast results
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            lower_bound: Lower prediction interval
            upper_bound: Upper prediction interval
            anomalies: Anomaly flags
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Time axis
        time_index = np.arange(len(y_true))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=time_index,
            y=y_true.flatten(),
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=time_index,
            y=y_pred.flatten(),
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add prediction intervals if provided
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_index, time_index[::-1]]),
                y=np.concatenate([upper_bound.flatten(), lower_bound.flatten()[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='rgba(255, 0, 0, 0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Prediction Interval'
            ))
            
        # Mark anomalies if provided
        if anomalies is not None:
            anomaly_indices = np.where(anomalies)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_indices,
                    y=y_true.flatten()[anomaly_indices],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='x'
                    )
                ))
                
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def plot_residual_analysis(self,
                              residuals: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive residual analysis plots
        
        Args:
            residuals: Forecast residuals
            model_name: Name of the model
            save_path: Path to save the plot
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Residuals Over Time', 'Residual Distribution',
                          'Q-Q Plot', 'ACF of Residuals'],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        time_index = np.arange(len(residuals))
        
        # 1. Residuals over time
        fig.add_trace(
            go.Scatter(x=time_index, y=residuals, mode='lines', name='Residuals'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Residual distribution
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Distribution'),
            row=1, col=2
        )
        
        # 3. Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                      mode='markers', name='Q-Q'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=theoretical_quantiles,
                      mode='lines', line=dict(color='red', dash='dash'),
                      name='Normal'),
            row=2, col=1
        )
        
        # 4. ACF plot
        max_lag = min(20, len(residuals) // 4)
        lags = range(1, max_lag)
        acf_values = [np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1] for lag in lags]
        
        fig.add_trace(
            go.Bar(x=list(lags), y=acf_values, name='ACF'),
            row=2, col=2
        )
        
        # Add confidence bands for ACF
        confidence = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=confidence, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_hline(y=-confidence, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=f"Residual Analysis - {model_name}",
            showlegend=False,
            height=700,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Residual", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=2)
        fig.update_yaxes(title_text="Correlation", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def plot_metrics_comparison(self,
                               save_path: Optional[str] = None) -> go.Figure:
        """Create bar plot comparing metrics across models
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        if not self.comparison_results.empty:
            df = self.comparison_results
        else:
            # Create from evaluation results
            data = []
            for model_name, metrics in self.evaluation_results.items():
                data.append({
                    'Model': model_name,
                    'RMSE': metrics.rmse,
                    'MAE': metrics.mae,
                    'MAPE': metrics.mape,
                    'R2': metrics.r2,
                    'F1-Score': metrics.f1_score
                })
            df = pd.DataFrame(data)
            
        # Create subplot for different metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['RMSE', 'MAE', 'MAPE (%)', 'R2 Score', 'F1-Score', 'Weighted Score'],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        metrics_info = [
            ('RMSE', 1, 1, 'lower'),
            ('MAE', 1, 2, 'lower'),
            ('MAPE', 1, 3, 'lower'),
            ('R2', 2, 1, 'higher'),
            ('F1-Score', 2, 2, 'higher'),
            ('Weighted Score', 2, 3, 'higher')
        ]
        
        for metric, row, col, better in metrics_info:
            if metric in df.columns:
                colors = ['green' if better == 'higher' else 'red' 
                         if v == df[metric].max() 
                         else 'red' if better == 'higher' else 'green' 
                         if v == df[metric].min() 
                         else 'blue' 
                         for v in df[metric]]
                
                fig.add_trace(
                    go.Bar(x=df['Model'], y=df[metric], 
                          marker_color=colors,
                          showlegend=False),
                    row=row, col=col
                )
                
        # Update layout
        fig.update_layout(
            title="Model Performance Comparison",
            height=600,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def plot_error_distribution(self,
                               errors: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> go.Figure:
        """Plot error distributions for multiple models
        
        Args:
            errors: Dictionary of model names to error arrays
            save_path: Path to save the plot
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for model_name, error_array in errors.items():
            fig.add_trace(go.Violin(
                y=error_array.flatten(),
                name=model_name,
                box_visible=True,
                meanline_visible=True
            ))
            
        fig.update_layout(
            title="Error Distribution Comparison",
            yaxis_title="Error",
            xaxis_title="Model",
            template='plotly_white',
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
        
    def generate_report(self, 
                       output_path: str = "forecast_evaluation_report.html",
                       include_plots: bool = True) -> str:
        """Generate comprehensive evaluation report
        
        Args:
            output_path: Path to save the report
            include_plots: Whether to include plots in the report
            
        Returns:
            HTML report as string
        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forecast Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric-good { color: green; font-weight: bold; }
                .metric-bad { color: red; font-weight: bold; }
                .summary-box { background-color: #f9f9f9; padding: 15px; 
                             border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>Forecast Model Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
        """
        
        html_content = html_content.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Add summary section
        html_content += """
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <p>Evaluated {num_models} forecasting models with comprehensive metrics.</p>
            </div>
        """.format(num_models=len(self.evaluation_results))
        
        # Add detailed metrics table
        html_content += "<h2>Detailed Metrics</h2>"
        html_content += "<table>"
        html_content += """
            <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>MAPE (%)</th>
                <th>R²</th>
                <th>F1-Score</th>
                <th>Direction Acc (%)</th>
            </tr>
        """
        
        for model_name, metrics in self.evaluation_results.items():
            html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{metrics.rmse:.4f}</td>
                    <td>{metrics.mae:.4f}</td>
                    <td>{metrics.mape:.2f}</td>
                    <td class="{'metric-good' if metrics.r2 > 0.8 else 'metric-bad' if metrics.r2 < 0.5 else ''}">
                        {metrics.r2:.4f}
                    </td>
                    <td class="{'metric-good' if metrics.f1_score > 0.8 else 'metric-bad' if metrics.f1_score < 0.5 else ''}">
                        {metrics.f1_score:.4f}
                    </td>
                    <td>{metrics.directional_accuracy:.2f}</td>
                </tr>
            """
            
        html_content += "</table>"
        
        # Add model comparison if available
        if not self.comparison_results.empty:
            html_content += "<h2>Model Ranking</h2>"
            html_content += self.comparison_results.to_html(classes='table', index=False)
            
        # Add recommendations
        html_content += """
            <div class="summary-box">
                <h2>Recommendations</h2>
                <ul>
        """
        
        if self.evaluation_results:
            best_model = min(self.evaluation_results.items(), 
                           key=lambda x: x[1].rmse)
            html_content += f"<li>Best performing model: <strong>{best_model[0]}</strong> (RMSE: {best_model[1].rmse:.4f})</li>"
            
            # Check for overfitting
            for model_name, metrics in self.evaluation_results.items():
                if metrics.r2 > 0.95:
                    html_content += f"<li>Warning: {model_name} may be overfitting (R² = {metrics.r2:.4f})</li>"
                    
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Evaluation report saved to {output_path}")
        
        return html_content
        
    def save_results(self, filepath: str):
        """Save evaluation results to file
        
        Args:
            filepath: Path to save results
        """
        results = {
            'evaluation_results': {
                name: metrics.to_dict() 
                for name, metrics in self.evaluation_results.items()
            },
            'comparison_results': self.comparison_results.to_dict() if not self.comparison_results.empty else {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        logger.info(f"Evaluation results saved to {filepath}")
        
    def load_results(self, filepath: str):
        """Load evaluation results from file
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        # Reconstruct metrics objects
        self.evaluation_results = {}
        for name, metrics_dict in results['evaluation_results'].items():
            metrics = ForecastMetrics()
            for key, value in metrics_dict.items():
                setattr(metrics, key, value)
            self.evaluation_results[name] = metrics
            
        if results['comparison_results']:
            self.comparison_results = pd.DataFrame(results['comparison_results'])
            
        logger.info(f"Evaluation results loaded from {filepath}")