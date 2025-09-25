"""
Metrics Utility Module
Provides comprehensive metrics calculation for the IoT Anomaly Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, asdict
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class AnomalyMetrics:
    """Data class for anomaly detection metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    fpr: float  # False Positive Rate
    fnr: float  # False Negative Rate
    auc_roc: float
    auc_pr: float  # Area Under Precision-Recall Curve
    confusion_matrix: np.ndarray
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['confusion_matrix'] = data['confusion_matrix'].tolist()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ForecastMetrics:
    """Data class for forecasting metrics"""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    explained_variance: float
    max_error: float
    directional_accuracy: float
    forecast_bias: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class MaintenanceMetrics:
    """Data class for maintenance scheduling metrics"""
    schedule_efficiency: float
    resource_utilization: float
    mean_time_to_repair: float
    mean_time_between_failures: float
    preventive_maintenance_rate: float
    downtime_percentage: float
    work_order_completion_rate: float
    average_response_time: float
    cost_savings: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

class AnomalyDetectionMetrics:
    """Metrics calculator for anomaly detection models"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_scores: Optional[np.ndarray] = None,
                         threshold: float = 0.5) -> AnomalyMetrics:
        """Calculate comprehensive anomaly detection metrics
        
        Args:
            y_true: True labels (0: normal, 1: anomaly)
            y_pred: Predicted labels
            y_scores: Anomaly scores (probabilities)
            threshold: Decision threshold
        
        Returns:
            AnomalyMetrics object
        """
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # AUC metrics
            if y_scores is not None:
                auc_roc = roc_auc_score(y_true, y_scores)
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
                auc_pr = np.trapz(recall_vals, precision_vals)
            else:
                auc_roc = 0.0
                auc_pr = 0.0
            
            return AnomalyMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                specificity=specificity,
                fpr=fpr,
                fnr=fnr,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                confusion_matrix=cm,
                threshold=threshold
            )
            
        except Exception as e:
            logger.error(f"Error calculating anomaly metrics: {str(e)}")
            raise
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray,
                              metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal threshold for anomaly detection
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
        
        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        try:
            thresholds = np.unique(y_scores)
            best_threshold = 0.5
            best_metric = 0
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                
                if metric == 'f1':
                    metric_value = f1_score(y_true, y_pred, zero_division=0)
                elif metric == 'precision':
                    metric_value = precision_score(y_true, y_pred, zero_division=0)
                elif metric == 'recall':
                    metric_value = recall_score(y_true, y_pred, zero_division=0)
                elif metric == 'youden':
                    # Youden's J statistic = sensitivity + specificity - 1
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        metric_value = sensitivity + specificity - 1
                    else:
                        metric_value = 0
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_threshold = threshold
            
            logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (value: {best_metric:.3f})")
            return best_threshold, best_metric
            
        except Exception as e:
            logger.error(f"Error finding optimal threshold: {str(e)}")
            raise
    
    @staticmethod
    def calculate_anomaly_score_distribution(scores: np.ndarray,
                                            labels: np.ndarray) -> Dict[str, Any]:
        """Calculate distribution statistics for anomaly scores
        
        Args:
            scores: Anomaly scores
            labels: True labels (0: normal, 1: anomaly)
        
        Returns:
            Dictionary with distribution statistics
        """
        try:
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]
            
            stats_dict = {
                'normal': {
                    'mean': float(np.mean(normal_scores)) if len(normal_scores) > 0 else 0,
                    'std': float(np.std(normal_scores)) if len(normal_scores) > 0 else 0,
                    'median': float(np.median(normal_scores)) if len(normal_scores) > 0 else 0,
                    'min': float(np.min(normal_scores)) if len(normal_scores) > 0 else 0,
                    'max': float(np.max(normal_scores)) if len(normal_scores) > 0 else 0,
                    'q25': float(np.percentile(normal_scores, 25)) if len(normal_scores) > 0 else 0,
                    'q75': float(np.percentile(normal_scores, 75)) if len(normal_scores) > 0 else 0,
                    'count': len(normal_scores)
                },
                'anomaly': {
                    'mean': float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
                    'std': float(np.std(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
                    'median': float(np.median(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
                    'min': float(np.min(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
                    'max': float(np.max(anomaly_scores)) if len(anomaly_scores) > 0 else 0,
                    'q25': float(np.percentile(anomaly_scores, 25)) if len(anomaly_scores) > 0 else 0,
                    'q75': float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) > 0 else 0,
                    'count': len(anomaly_scores)
                },
                'separation': {
                    'overlap_ratio': 0,
                    'ks_statistic': 0,
                    'ks_pvalue': 0
                }
            }
            
            # Calculate separation metrics
            if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                # Overlap ratio
                normal_range = (np.min(normal_scores), np.max(normal_scores))
                anomaly_range = (np.min(anomaly_scores), np.max(anomaly_scores))
                overlap = (min(normal_range[1], anomaly_range[1]) - 
                          max(normal_range[0], anomaly_range[0]))
                total_range = max(normal_range[1], anomaly_range[1]) - min(normal_range[0], anomaly_range[0])
                stats_dict['separation']['overlap_ratio'] = float(max(0, overlap / total_range))
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = stats.ks_2samp(normal_scores, anomaly_scores)
                stats_dict['separation']['ks_statistic'] = float(ks_stat)
                stats_dict['separation']['ks_pvalue'] = float(ks_pval)
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating score distribution: {str(e)}")
            raise

class ForecastingMetrics:
    """Metrics calculator for time series forecasting models"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         multivariate: bool = False) -> ForecastMetrics:
        """Calculate comprehensive forecasting metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            multivariate: Whether data is multivariate
        
        Returns:
            ForecastMetrics object
        """
        try:
            # Ensure arrays are the same shape
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            if y_true.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
            
            # Basic metrics
            mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average' if multivariate else 'uniform_average')
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average' if multivariate else 'uniform_average')
            
            # MAPE - handle zero values
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                if np.any(y_true == 0):
                    # Use symmetric MAPE when zeros present
                    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                    mape = np.mean(np.abs((y_true - y_pred) / np.where(denominator != 0, denominator, 1))) * 100
                else:
                    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            
            # R² score
            r2 = r2_score(y_true, y_pred, multioutput='uniform_average' if multivariate else 'uniform_average')
            
            # Explained variance
            explained_var = explained_variance_score(y_true, y_pred, multioutput='uniform_average' if multivariate else 'uniform_average')
            
            # Max error
            max_err = np.max(np.abs(y_true - y_pred))
            
            # Directional accuracy (for time series)
            if len(y_true) > 1:
                true_direction = np.diff(y_true.flatten())
                pred_direction = np.diff(y_pred.flatten())
                directional_acc = np.mean(np.sign(true_direction) == np.sign(pred_direction))
            else:
                directional_acc = 0
            
            # Forecast bias
            bias = np.mean(y_pred - y_true)
            
            return ForecastMetrics(
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                mape=float(mape),
                r2=float(r2),
                explained_variance=float(explained_var),
                max_error=float(max_err),
                directional_accuracy=float(directional_acc),
                forecast_bias=float(bias)
            )
            
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {str(e)}")
            raise
    
    @staticmethod
    def calculate_prediction_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                      confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals for forecasts
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence level (0-1)
        
        Returns:
            Dictionary with lower and upper bounds
        """
        try:
            residuals = y_true - y_pred
            std_residuals = np.std(residuals)
            
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence) / 2)
            
            # Calculate intervals
            lower_bound = y_pred - z_score * std_residuals
            upper_bound = y_pred + z_score * std_residuals
            
            # Calculate coverage
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            
            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': upper_bound - lower_bound,
                'coverage': float(coverage),
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating prediction intervals: {str(e)}")
            raise
    
    @staticmethod
    def calculate_seasonal_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  season_length: int) -> Dict[str, float]:
        """Calculate seasonal decomposition metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            season_length: Length of seasonal period
        
        Returns:
            Dictionary with seasonal metrics
        """
        try:
            n_seasons = len(y_true) // season_length
            seasonal_errors = []
            
            for i in range(n_seasons):
                start_idx = i * season_length
                end_idx = (i + 1) * season_length
                
                if end_idx <= len(y_true):
                    season_true = y_true[start_idx:end_idx]
                    season_pred = y_pred[start_idx:end_idx]
                    season_mae = mean_absolute_error(season_true, season_pred)
                    seasonal_errors.append(season_mae)
            
            return {
                'mean_seasonal_mae': float(np.mean(seasonal_errors)) if seasonal_errors else 0,
                'std_seasonal_mae': float(np.std(seasonal_errors)) if seasonal_errors else 0,
                'max_seasonal_mae': float(np.max(seasonal_errors)) if seasonal_errors else 0,
                'min_seasonal_mae': float(np.min(seasonal_errors)) if seasonal_errors else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating seasonal metrics: {str(e)}")
            raise

class MaintenanceSchedulingMetrics:
    """Metrics calculator for maintenance scheduling optimization"""
    
    @staticmethod
    def calculate_metrics(work_orders_df: pd.DataFrame,
                         equipment_df: pd.DataFrame,
                         start_date: datetime,
                         end_date: datetime) -> MaintenanceMetrics:
        """Calculate comprehensive maintenance metrics
        
        Args:
            work_orders_df: DataFrame with work order data
            equipment_df: DataFrame with equipment data
            start_date: Period start date
            end_date: Period end date
        
        Returns:
            MaintenanceMetrics object
        """
        try:
            # Filter data for the period
            period_orders = work_orders_df[
                (work_orders_df['created_at'] >= start_date) &
                (work_orders_df['created_at'] <= end_date)
            ].copy()
            
            if period_orders.empty:
                return MaintenanceMetrics(
                    schedule_efficiency=0,
                    resource_utilization=0,
                    mean_time_to_repair=0,
                    mean_time_between_failures=0,
                    preventive_maintenance_rate=0,
                    downtime_percentage=0,
                    work_order_completion_rate=0,
                    average_response_time=0,
                    cost_savings=0
                )
            
            # Schedule efficiency
            scheduled_duration = period_orders['estimated_duration'].sum()
            actual_duration = period_orders['actual_duration'].sum()
            schedule_efficiency = (scheduled_duration / actual_duration * 100) if actual_duration > 0 else 0
            
            # Resource utilization
            total_available_hours = (end_date - start_date).total_seconds() / 3600
            total_work_hours = actual_duration / 60  # Convert minutes to hours
            resource_utilization = (total_work_hours / total_available_hours * 100) if total_available_hours > 0 else 0
            
            # Mean Time To Repair (MTTR)
            completed_orders = period_orders[period_orders['status'] == 'COMPLETED']
            mttr = completed_orders['actual_duration'].mean() if not completed_orders.empty else 0
            
            # Mean Time Between Failures (MTBF)
            failures_by_equipment = period_orders.groupby('equipment_id').size()
            if len(failures_by_equipment) > 1:
                time_between_failures = []
                for equipment_id in failures_by_equipment.index:
                    equipment_orders = period_orders[
                        period_orders['equipment_id'] == equipment_id
                    ].sort_values('created_at')
                    
                    if len(equipment_orders) > 1:
                        time_diffs = equipment_orders['created_at'].diff().dropna()
                        time_between_failures.extend(time_diffs.dt.total_seconds() / 3600)
                
                mtbf = np.mean(time_between_failures) if time_between_failures else 0
            else:
                mtbf = 0
            
            # Preventive vs Reactive maintenance rate
            preventive_count = len(period_orders[period_orders['maintenance_type'] == 'PREVENTIVE']) if 'maintenance_type' in period_orders.columns else 0
            total_count = len(period_orders)
            preventive_rate = (preventive_count / total_count * 100) if total_count > 0 else 0
            
            # Downtime percentage
            total_downtime = period_orders['actual_duration'].sum() / 60  # Convert to hours
            total_operational_hours = total_available_hours
            downtime_percentage = (total_downtime / total_operational_hours * 100) if total_operational_hours > 0 else 0
            
            # Work order completion rate
            completed_count = len(period_orders[period_orders['status'] == 'COMPLETED'])
            completion_rate = (completed_count / total_count * 100) if total_count > 0 else 0
            
            # Average response time
            period_orders['response_time'] = (
                pd.to_datetime(period_orders['assigned_at']) - 
                pd.to_datetime(period_orders['created_at'])
            ).dt.total_seconds() / 3600  # Convert to hours
            avg_response_time = period_orders['response_time'].mean() if 'assigned_at' in period_orders.columns else 0
            
            # Cost savings (placeholder - would need actual cost data)
            cost_savings = preventive_count * 1000  # Example: $1000 saved per preventive maintenance
            
            return MaintenanceMetrics(
                schedule_efficiency=float(schedule_efficiency),
                resource_utilization=float(resource_utilization),
                mean_time_to_repair=float(mttr),
                mean_time_between_failures=float(mtbf),
                preventive_maintenance_rate=float(preventive_rate),
                downtime_percentage=float(downtime_percentage),
                work_order_completion_rate=float(completion_rate),
                average_response_time=float(avg_response_time),
                cost_savings=float(cost_savings)
            )
            
        except Exception as e:
            logger.error(f"Error calculating maintenance metrics: {str(e)}")
            raise
    
    @staticmethod
    def calculate_technician_performance(work_orders_df: pd.DataFrame,
                                        technician: str) -> Dict[str, float]:
        """Calculate performance metrics for a specific technician
        
        Args:
            work_orders_df: DataFrame with work order data
            technician: Technician identifier
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            tech_orders = work_orders_df[
                work_orders_df['assigned_technician'] == technician
            ].copy()
            
            if tech_orders.empty:
                return {
                    'total_orders': 0,
                    'completion_rate': 0,
                    'avg_completion_time': 0,
                    'efficiency_score': 0,
                    'on_time_rate': 0
                }
            
            completed = tech_orders[tech_orders['status'] == 'COMPLETED']
            
            # Calculate metrics
            total_orders = len(tech_orders)
            completion_rate = (len(completed) / total_orders * 100) if total_orders > 0 else 0
            
            avg_completion_time = completed['actual_duration'].mean() if not completed.empty else 0
            
            # Efficiency score (actual vs estimated)
            if not completed.empty and 'estimated_duration' in completed.columns:
                efficiency_scores = completed['estimated_duration'] / completed['actual_duration']
                efficiency_score = efficiency_scores.mean() * 100
            else:
                efficiency_score = 0
            
            # On-time rate
            if 'due_date' in completed.columns and 'completed_at' in completed.columns:
                on_time = completed[
                    pd.to_datetime(completed['completed_at']) <= 
                    pd.to_datetime(completed['due_date'])
                ]
                on_time_rate = (len(on_time) / len(completed) * 100) if not completed.empty else 0
            else:
                on_time_rate = 0
            
            return {
                'total_orders': int(total_orders),
                'completion_rate': float(completion_rate),
                'avg_completion_time': float(avg_completion_time),
                'efficiency_score': float(efficiency_score),
                'on_time_rate': float(on_time_rate)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technician performance: {str(e)}")
            raise
    
    @staticmethod
    def calculate_equipment_reliability(equipment_df: pd.DataFrame,
                                       work_orders_df: pd.DataFrame,
                                       anomalies_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reliability metrics for equipment
        
        Args:
            equipment_df: DataFrame with equipment data
            work_orders_df: DataFrame with work order data
            anomalies_df: DataFrame with anomaly data
        
        Returns:
            DataFrame with reliability metrics per equipment
        """
        try:
            reliability_metrics = []
            
            for _, equipment in equipment_df.iterrows():
                equipment_id = equipment['equipment_id']
                
                # Get related data
                equipment_orders = work_orders_df[
                    work_orders_df['equipment_id'] == equipment_id
                ]
                equipment_anomalies = anomalies_df[
                    anomalies_df['equipment_id'] == equipment_id
                ]
                
                # Calculate reliability score
                failure_count = len(equipment_orders[
                    equipment_orders['maintenance_type'] == 'CORRECTIVE'
                ]) if 'maintenance_type' in equipment_orders.columns else len(equipment_orders)
                
                anomaly_count = len(equipment_anomalies)
                
                # Simple reliability score (inverse of failure rate)
                days_in_service = 365  # Placeholder
                reliability_score = 100 * np.exp(-failure_count / days_in_service)
                
                # Health score based on recent anomalies
                recent_anomalies = equipment_anomalies[
                    pd.to_datetime(equipment_anomalies['created_at']) > 
                    datetime.now() - timedelta(days=30)
                ] if not equipment_anomalies.empty else pd.DataFrame()
                
                if not recent_anomalies.empty:
                    avg_anomaly_score = recent_anomalies['anomaly_score'].mean()
                    health_score = 100 * (1 - avg_anomaly_score)
                else:
                    health_score = 100
                
                reliability_metrics.append({
                    'equipment_id': equipment_id,
                    'equipment_name': equipment.get('equipment_name', equipment_id),
                    'reliability_score': reliability_score,
                    'health_score': health_score,
                    'failure_count': failure_count,
                    'anomaly_count': anomaly_count,
                    'criticality': equipment.get('criticality', 'MEDIUM')
                })
            
            return pd.DataFrame(reliability_metrics)
            
        except Exception as e:
            logger.error(f"Error calculating equipment reliability: {str(e)}")
            raise

class ModelComparisonMetrics:
    """Metrics for comparing multiple models"""
    
    @staticmethod
    def compare_models(models_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models based on their metrics
        
        Args:
            models_results: Dictionary with model names as keys and metrics as values
        
        Returns:
            DataFrame with model comparison
        """
        try:
            comparison_data = []
            
            for model_name, metrics in models_results.items():
                row = {'model': model_name}
                
                # Add metrics based on type
                if 'accuracy' in metrics:
                    # Classification metrics
                    row.update({
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0),
                        'auc_roc': metrics.get('auc_roc', 0)
                    })
                elif 'mse' in metrics:
                    # Regression metrics
                    row.update({
                        'mse': metrics.get('mse', 0),
                        'rmse': metrics.get('rmse', 0),
                        'mae': metrics.get('mae', 0),
                        'r2': metrics.get('r2', 0),
                        'mape': metrics.get('mape', 0)
                    })
                
                comparison_data.append(row)
            
            df = pd.DataFrame(comparison_data)
            
            # Add ranking
            if 'accuracy' in df.columns:
                df['rank'] = df['f1_score'].rank(ascending=False, method='min').astype(int)
            elif 'rmse' in df.columns:
                df['rank'] = df['rmse'].rank(ascending=True, method='min').astype(int)
            
            return df.sort_values('rank')
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    @staticmethod
    def calculate_model_stability(predictions_over_time: List[np.ndarray]) -> Dict[str, float]:
        """Calculate stability metrics for model predictions over time
        
        Args:
            predictions_over_time: List of prediction arrays from different time periods
        
        Returns:
            Dictionary with stability metrics
        """
        try:
            if len(predictions_over_time) < 2:
                return {'stability_score': 1.0, 'drift_detected': False}
            
            # Calculate prediction drift
            drifts = []
            for i in range(1, len(predictions_over_time)):
                prev_pred = predictions_over_time[i-1]
                curr_pred = predictions_over_time[i]
                
                # KS test for distribution drift
                ks_stat, ks_pval = stats.ks_2samp(prev_pred.flatten(), curr_pred.flatten())
                drifts.append(ks_stat)
            
            # Stability score (inverse of average drift)
            avg_drift = np.mean(drifts)
            stability_score = 1 / (1 + avg_drift)
            
            # Detect significant drift
            drift_detected = any(d > 0.1 for d in drifts)  # Threshold for KS statistic
            
            return {
                'stability_score': float(stability_score),
                'average_drift': float(avg_drift),
                'max_drift': float(np.max(drifts)),
                'drift_detected': drift_detected,
                'num_periods': len(predictions_over_time)
            }
            
        except Exception as e:
            logger.error(f"Error calculating model stability: {str(e)}")
            raise

class StreamingMetrics:
    """Metrics calculator for streaming/online learning scenarios"""
    
    def __init__(self, window_size: int = 1000):
        """Initialize streaming metrics calculator
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.y_true_buffer = []
        self.y_pred_buffer = []
        self.timestamps = []
        
    def update(self, y_true: Union[float, np.ndarray], 
               y_pred: Union[float, np.ndarray],
               timestamp: Optional[datetime] = None):
        """Update metrics with new data point(s)
        
        Args:
            y_true: True value(s)
            y_pred: Predicted value(s)
            timestamp: Optional timestamp
        """
        if isinstance(y_true, (float, int)):
            y_true = [y_true]
        if isinstance(y_pred, (float, int)):
            y_pred = [y_pred]
        
        self.y_true_buffer.extend(y_true)
        self.y_pred_buffer.extend(y_pred)
        
        if timestamp:
            self.timestamps.append(timestamp)
        
        # Maintain window size
        if len(self.y_true_buffer) > self.window_size:
            excess = len(self.y_true_buffer) - self.window_size
            self.y_true_buffer = self.y_true_buffer[excess:]
            self.y_pred_buffer = self.y_pred_buffer[excess:]
            if self.timestamps:
                self.timestamps = self.timestamps[excess:]
    
    def get_current_metrics(self, metric_type: str = 'regression') -> Dict[str, float]:
        """Get current metrics for the sliding window
        
        Args:
            metric_type: Type of metrics ('regression' or 'classification')
        
        Returns:
            Dictionary with current metrics
        """
        if not self.y_true_buffer:
            return {}
        
        y_true = np.array(self.y_true_buffer)
        y_pred = np.array(self.y_pred_buffer)
        
        if metric_type == 'regression':
            return {
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'window_size': len(self.y_true_buffer)
            }
        elif metric_type == 'classification':
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                'window_size': len(self.y_true_buffer)
            }
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def get_trend(self, metric: str = 'mae', periods: int = 10) -> Dict[str, Any]:
        """Calculate trend in metrics over recent periods
        
        Args:
            metric: Metric to track
            periods: Number of periods to analyze
        
        Returns:
            Dictionary with trend information
        """
        if len(self.y_true_buffer) < periods * 2:
            return {'trend': 'insufficient_data'}
        
        period_size = len(self.y_true_buffer) // periods
        period_metrics = []
        
        for i in range(periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size
            
            y_true_period = np.array(self.y_true_buffer[start_idx:end_idx])
            y_pred_period = np.array(self.y_pred_buffer[start_idx:end_idx])
            
            if metric == 'mae':
                value = mean_absolute_error(y_true_period, y_pred_period)
            elif metric == 'accuracy':
                value = accuracy_score(y_true_period, y_pred_period)
            else:
                value = 0
            
            period_metrics.append(value)
        
        # Calculate trend
        x = np.arange(len(period_metrics))
        slope, intercept = np.polyfit(x, period_metrics, 1)
        
        return {
            'trend': 'improving' if slope < 0 else 'degrading',
            'slope': float(slope),
            'current_value': float(period_metrics[-1]),
            'change_rate': float((period_metrics[-1] - period_metrics[0]) / period_metrics[0] * 100)
        }

# Utility functions
def calculate_business_impact(metrics: Dict[str, Any], 
                             cost_parameters: Dict[str, float]) -> Dict[str, float]:
    """Calculate business impact of model performance
    
    Args:
        metrics: Model performance metrics
        cost_parameters: Cost parameters (false_positive_cost, false_negative_cost, etc.)
    
    Returns:
        Dictionary with business impact metrics
    """
    try:
        impact = {}
        
        # Cost of errors
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                
                fp_cost = fp * cost_parameters.get('false_positive_cost', 100)
                fn_cost = fn * cost_parameters.get('false_negative_cost', 1000)
                
                impact['total_cost'] = fp_cost + fn_cost
                impact['false_positive_cost'] = fp_cost
                impact['false_negative_cost'] = fn_cost
                
                # Savings from true positives
                impact['savings'] = tp * cost_parameters.get('true_positive_savings', 500)
                
                # Net impact
                impact['net_impact'] = impact['savings'] - impact['total_cost']
        
        # Downtime cost
        if 'downtime_percentage' in metrics:
            hourly_cost = cost_parameters.get('hourly_downtime_cost', 5000)
            impact['downtime_cost'] = metrics['downtime_percentage'] * hourly_cost * 24 * 30 / 100
        
        # ROI calculation
        if 'net_impact' in impact:
            investment = cost_parameters.get('system_cost', 100000)
            impact['roi'] = (impact['net_impact'] / investment) * 100
        
        return impact
        
    except Exception as e:
        logger.error(f"Error calculating business impact: {str(e)}")
        return {}

def generate_metrics_report(anomaly_metrics: Optional[AnomalyMetrics] = None,
                           forecast_metrics: Optional[ForecastMetrics] = None,
                           maintenance_metrics: Optional[MaintenanceMetrics] = None,
                           output_format: str = 'dict') -> Union[Dict, str]:
    """Generate comprehensive metrics report
    
    Args:
        anomaly_metrics: Anomaly detection metrics
        forecast_metrics: Forecasting metrics
        maintenance_metrics: Maintenance metrics
        output_format: Output format ('dict' or 'json')
    
    Returns:
        Metrics report in specified format
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }
    
    if anomaly_metrics:
        report['metrics']['anomaly_detection'] = anomaly_metrics.to_dict()
    
    if forecast_metrics:
        report['metrics']['forecasting'] = forecast_metrics.to_dict()
    
    if maintenance_metrics:
        report['metrics']['maintenance'] = maintenance_metrics.to_dict()
    
    if output_format == 'json':
        return json.dumps(report, indent=2)
    
    return report