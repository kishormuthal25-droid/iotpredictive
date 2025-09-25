"""
Failure Probability Estimation Engine
Equipment-specific failure probability modeling using survival analysis and degradation models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from enum import Enum
import json

# Statistical and ML imports
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma, gammaln
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.forecasting.enhanced_forecaster import EnhancedForecaster, EnhancedForecastResult
from config.settings import settings, get_config

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Equipment failure modes"""
    GRADUAL_DEGRADATION = "gradual_degradation"
    SUDDEN_FAILURE = "sudden_failure"
    WEAR_OUT = "wear_out"
    RANDOM_FAILURE = "random_failure"
    CASCADING_FAILURE = "cascading_failure"


class SeverityLevel(Enum):
    """Failure severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FailurePrediction:
    """Container for failure probability predictions"""
    equipment_id: str
    component_id: str
    failure_probability: float  # Probability of failure in next time horizon
    time_to_failure: Optional[float] = None  # Expected time to failure (hours)
    confidence_interval: Optional[Tuple[float, float]] = None  # CI for time to failure
    severity: SeverityLevel = SeverityLevel.MEDIUM
    failure_mode: FailureMode = FailureMode.GRADUAL_DEGRADATION
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    historical_mtbf: Optional[float] = None  # Mean time between failures
    degradation_rate: Optional[float] = None  # Rate of degradation
    remaining_useful_life: Optional[float] = None  # RUL estimate
    uncertainty: float = 0.0  # Prediction uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'equipment_id': self.equipment_id,
            'component_id': self.component_id,
            'failure_probability': self.failure_probability,
            'time_to_failure': self.time_to_failure,
            'confidence_interval': self.confidence_interval,
            'severity': self.severity.value,
            'failure_mode': self.failure_mode.value,
            'contributing_factors': self.contributing_factors,
            'historical_mtbf': self.historical_mtbf,
            'degradation_rate': self.degradation_rate,
            'remaining_useful_life': self.remaining_useful_life,
            'uncertainty': self.uncertainty,
            'metadata': self.metadata
        }


class WeibullModel:
    """Weibull distribution for reliability modeling"""

    def __init__(self):
        self.shape = None  # Beta parameter
        self.scale = None  # Eta parameter
        self.is_fitted = False

    def fit(self, times_to_failure: np.ndarray, censored: Optional[np.ndarray] = None):
        """
        Fit Weibull distribution parameters

        Args:
            times_to_failure: Observed failure times
            censored: Boolean array indicating censored observations
        """
        if censored is None:
            censored = np.zeros(len(times_to_failure), dtype=bool)

        # Use maximum likelihood estimation
        def neg_log_likelihood(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return np.inf

            likelihood = 0
            for i, t in enumerate(times_to_failure):
                if censored[i]:
                    # Survival function for censored data
                    likelihood += np.log(1 - stats.weibull_min.cdf(t, shape, scale=scale))
                else:
                    # PDF for observed failures
                    likelihood += np.log(stats.weibull_min.pdf(t, shape, scale=scale))

            return -likelihood

        # Initial guess
        mean_time = np.mean(times_to_failure[~censored]) if np.any(~censored) else np.mean(times_to_failure)
        initial_guess = [2.0, mean_time]

        # Optimize
        result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B',
                         bounds=[(0.1, 10), (0.1, None)])

        if result.success:
            self.shape, self.scale = result.x
            self.is_fitted = True
        else:
            logger.warning("Weibull fitting failed, using default parameters")
            self.shape, self.scale = 2.0, mean_time
            self.is_fitted = True

    def survival_function(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate survival probability at time t"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return np.exp(-((t / self.scale) ** self.shape))

    def failure_probability(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate failure probability by time t"""
        return 1 - self.survival_function(t)

    def hazard_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate instantaneous hazard rate at time t"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return (self.shape / self.scale) * ((t / self.scale) ** (self.shape - 1))

    def mean_time_to_failure(self) -> float:
        """Calculate mean time to failure"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.scale * gamma(1 + 1/self.shape)


class DegradationModel:
    """Degradation modeling for gradual failure prediction"""

    def __init__(self, model_type: str = "linear"):
        """
        Initialize degradation model

        Args:
            model_type: Type of degradation model ("linear", "exponential", "power", "ml")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.failure_threshold = None

    def fit(self,
            time_points: np.ndarray,
            degradation_values: np.ndarray,
            failure_threshold: float,
            features: Optional[np.ndarray] = None):
        """
        Fit degradation model

        Args:
            time_points: Time points for observations
            degradation_values: Degradation measurements
            failure_threshold: Threshold value indicating failure
            features: Additional features (temperature, load, etc.)
        """
        self.failure_threshold = failure_threshold

        if self.model_type == "linear":
            self.model = LinearRegression()
            X = time_points.reshape(-1, 1)
            self.model.fit(X, degradation_values)

        elif self.model_type == "exponential":
            # Fit y = a * exp(b * t)
            log_y = np.log(np.maximum(degradation_values, 1e-10))
            self.model = LinearRegression()
            X = time_points.reshape(-1, 1)
            self.model.fit(X, log_y)

        elif self.model_type == "power":
            # Fit y = a * t^b
            log_t = np.log(np.maximum(time_points, 1e-10))
            log_y = np.log(np.maximum(degradation_values, 1e-10))
            self.model = LinearRegression()
            self.model.fit(log_t.reshape(-1, 1), log_y)

        elif self.model_type == "ml":
            # Machine learning approach with additional features
            if features is not None:
                X = np.column_stack([time_points, features])
            else:
                X = time_points.reshape(-1, 1)

            X_scaled = self.scaler.fit_transform(X)
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, degradation_values)

        self.is_fitted = True

    def predict_degradation(self,
                          time_points: np.ndarray,
                          features: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict degradation at future time points"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        if self.model_type == "linear":
            X = time_points.reshape(-1, 1)
            return self.model.predict(X)

        elif self.model_type == "exponential":
            X = time_points.reshape(-1, 1)
            log_pred = self.model.predict(X)
            return np.exp(log_pred)

        elif self.model_type == "power":
            log_t = np.log(np.maximum(time_points, 1e-10))
            log_pred = self.model.predict(log_t.reshape(-1, 1))
            return np.exp(log_pred)

        elif self.model_type == "ml":
            if features is not None:
                X = np.column_stack([time_points, features])
            else:
                X = time_points.reshape(-1, 1)

            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)

    def time_to_failure(self, current_degradation: float, current_time: float) -> Optional[float]:
        """Estimate time to reach failure threshold"""
        if not self.is_fitted or self.failure_threshold is None:
            return None

        if current_degradation >= self.failure_threshold:
            return 0.0

        # Binary search for time when degradation reaches threshold
        max_time = current_time + 10000  # Search up to 10000 time units ahead
        min_time = current_time

        for _ in range(100):  # Max iterations
            mid_time = (min_time + max_time) / 2
            predicted_degradation = self.predict_degradation(np.array([mid_time]))[0]

            if abs(predicted_degradation - self.failure_threshold) < 0.01:
                return mid_time - current_time
            elif predicted_degradation < self.failure_threshold:
                min_time = mid_time
            else:
                max_time = mid_time

        return max_time - current_time if max_time < current_time + 10000 else None


class FailureProbabilityEstimator:
    """Main failure probability estimation engine"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize failure probability estimator

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.equipment_models = {}  # Equipment-specific models
        self.historical_data = {}  # Historical failure data
        self.degradation_models = {}  # Degradation models per component
        self.survival_models = {}  # Survival models per component
        self.anomaly_weights = {}  # Weights for anomaly impact

        # Equipment configuration from CLAUDE.md
        self.equipment_config = {
            'smap': {
                'components': {
                    'power_system': {'sensors': 6, 'criticality': 'critical'},
                    'communication': {'sensors': 5, 'criticality': 'high'},
                    'attitude_control': {'sensors': 6, 'criticality': 'critical'},
                    'thermal_control': {'sensors': 4, 'criticality': 'high'},
                    'payload_sensors': {'sensors': 4, 'criticality': 'high'}
                }
            },
            'msl': {
                'components': {
                    'mobility_front': {'sensors': 12, 'criticality': 'critical'},
                    'mobility_rear': {'sensors': 6, 'criticality': 'critical'},
                    'power_system': {'sensors': 8, 'criticality': 'critical'},
                    'environmental': {'sensors': 12, 'criticality': 'medium'},
                    'scientific': {'sensors': 10, 'criticality': 'high'},
                    'communication': {'sensors': 6, 'criticality': 'high'},
                    'navigation': {'sensors': 1, 'criticality': 'critical'}
                }
            }
        }

        logger.info("Initialized Failure Probability Estimator")

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'time_horizon': 168,  # Hours (1 week)
            'failure_thresholds': {
                'degradation_rate': 0.1,
                'anomaly_rate': 0.05,
                'sensor_deviation': 3.0
            },
            'survival_models': {
                'default_shape': 2.0,
                'default_scale': 8760  # 1 year in hours
            },
            'degradation_models': {
                'default_type': 'linear',
                'failure_threshold': 0.8
            },
            'uncertainty_factors': {
                'model_uncertainty': 0.1,
                'data_uncertainty': 0.05,
                'environmental_uncertainty': 0.02
            }
        }

    def add_historical_failure_data(self,
                                   equipment_id: str,
                                   component_id: str,
                                   failure_times: List[float],
                                   censored: Optional[List[bool]] = None):
        """
        Add historical failure data for a component

        Args:
            equipment_id: Equipment identifier
            component_id: Component identifier
            failure_times: List of failure times
            censored: List indicating censored observations
        """
        key = f"{equipment_id}_{component_id}"
        self.historical_data[key] = {
            'failure_times': np.array(failure_times),
            'censored': np.array(censored) if censored else np.zeros(len(failure_times), dtype=bool)
        }

        # Fit Weibull model
        weibull_model = WeibullModel()
        weibull_model.fit(self.historical_data[key]['failure_times'],
                         self.historical_data[key]['censored'])
        self.survival_models[key] = weibull_model

        logger.info(f"Added historical failure data for {key}")

    def add_degradation_data(self,
                           equipment_id: str,
                           component_id: str,
                           time_points: np.ndarray,
                           degradation_values: np.ndarray,
                           failure_threshold: float,
                           model_type: str = "linear"):
        """
        Add degradation data and fit degradation model

        Args:
            equipment_id: Equipment identifier
            component_id: Component identifier
            time_points: Time points for observations
            degradation_values: Degradation measurements
            failure_threshold: Threshold indicating failure
            model_type: Type of degradation model
        """
        key = f"{equipment_id}_{component_id}"

        degradation_model = DegradationModel(model_type)
        degradation_model.fit(time_points, degradation_values, failure_threshold)
        self.degradation_models[key] = degradation_model

        logger.info(f"Added degradation model for {key}")

    def estimate_failure_probability(self,
                                   equipment_id: str,
                                   component_id: str,
                                   current_sensor_data: np.ndarray,
                                   forecast_result: EnhancedForecastResult,
                                   anomaly_scores: Optional[np.ndarray] = None,
                                   operational_context: Optional[Dict] = None) -> FailurePrediction:
        """
        Estimate failure probability for a specific component

        Args:
            equipment_id: Equipment identifier
            component_id: Component identifier
            current_sensor_data: Current sensor readings
            forecast_result: Enhanced forecast results
            anomaly_scores: Anomaly detection scores
            operational_context: Operational context (temperature, load, etc.)

        Returns:
            FailurePrediction object
        """
        key = f"{equipment_id}_{component_id}"

        # Get equipment configuration
        equipment_type = equipment_id.lower()
        if equipment_type in self.equipment_config:
            component_config = self.equipment_config[equipment_type]['components'].get(
                component_id, {'criticality': 'medium'}
            )
            criticality = component_config['criticality']
        else:
            criticality = 'medium'

        # Initialize prediction
        prediction = FailurePrediction(
            equipment_id=equipment_id,
            component_id=component_id,
            failure_probability=0.0,
            severity=self._get_severity_from_criticality(criticality)
        )

        # Factors contributing to failure probability
        contributing_factors = {}

        # 1. Survival model based prediction
        survival_probability = 1.0
        if key in self.survival_models:
            survival_model = self.survival_models[key]
            time_horizon = self.config['time_horizon']
            survival_probability = survival_model.survival_function(time_horizon)
            contributing_factors['historical_reliability'] = 1 - survival_probability

        # 2. Degradation model based prediction
        degradation_probability = 0.0
        if key in self.degradation_models:
            degradation_model = self.degradation_models[key]

            # Estimate current degradation level
            current_degradation = self._estimate_current_degradation(current_sensor_data)

            # Predict future degradation
            future_times = np.arange(1, self.config['time_horizon'] + 1)
            future_degradation = degradation_model.predict_degradation(future_times)

            # Check if degradation exceeds threshold
            threshold = degradation_model.failure_threshold
            if threshold and np.any(future_degradation >= threshold):
                degradation_probability = 1.0
                first_failure_time = future_times[future_degradation >= threshold][0]
                prediction.time_to_failure = float(first_failure_time)
            else:
                # Estimate based on degradation rate
                degradation_rate = self._calculate_degradation_rate(future_degradation)
                prediction.degradation_rate = degradation_rate
                degradation_probability = min(degradation_rate * 10, 1.0)  # Scale factor

                # Estimate time to failure
                ttf = degradation_model.time_to_failure(current_degradation, 0)
                if ttf:
                    prediction.time_to_failure = ttf

            contributing_factors['degradation_trend'] = degradation_probability

        # 3. Forecast trend analysis
        forecast_probability = self._analyze_forecast_trends(forecast_result)
        contributing_factors['forecast_trends'] = forecast_probability

        # 4. Anomaly impact
        anomaly_probability = 0.0
        if anomaly_scores is not None:
            anomaly_probability = self._calculate_anomaly_impact(anomaly_scores)
            contributing_factors['anomaly_impact'] = anomaly_probability

        # 5. Operational stress factors
        stress_probability = 0.0
        if operational_context:
            stress_probability = self._calculate_operational_stress(operational_context, criticality)
            contributing_factors['operational_stress'] = stress_probability

        # Combine probabilities using weighted approach
        weights = {
            'historical_reliability': 0.3,
            'degradation_trend': 0.3,
            'forecast_trends': 0.2,
            'anomaly_impact': 0.15,
            'operational_stress': 0.05
        }

        total_probability = 0.0
        for factor, prob in contributing_factors.items():
            weight = weights.get(factor, 0.1)
            total_probability += weight * prob

        # Apply criticality multiplier
        criticality_multipliers = {
            'critical': 1.5,
            'high': 1.2,
            'medium': 1.0,
            'low': 0.8
        }
        total_probability *= criticality_multipliers.get(criticality, 1.0)

        # Ensure probability is between 0 and 1
        prediction.failure_probability = min(max(total_probability, 0.0), 1.0)
        prediction.contributing_factors = contributing_factors

        # Calculate uncertainty
        prediction.uncertainty = self._calculate_prediction_uncertainty(
            forecast_result, len(contributing_factors)
        )

        # Estimate remaining useful life
        if prediction.time_to_failure:
            prediction.remaining_useful_life = prediction.time_to_failure
        elif key in self.survival_models:
            # Use survival model to estimate RUL
            survival_model = self.survival_models[key]
            prediction.remaining_useful_life = survival_model.mean_time_to_failure()

        # Add metadata
        prediction.metadata = {
            'estimation_time': datetime.now().isoformat(),
            'model_versions': {
                'survival': key in self.survival_models,
                'degradation': key in self.degradation_models
            },
            'data_quality': self._assess_data_quality(current_sensor_data, forecast_result)
        }

        return prediction

    def estimate_cascade_failure_probability(self,
                                           primary_failure: FailurePrediction,
                                           dependent_components: List[str]) -> Dict[str, float]:
        """
        Estimate cascade failure probabilities

        Args:
            primary_failure: Primary component failure prediction
            dependent_components: List of dependent component IDs

        Returns:
            Dictionary of cascade failure probabilities
        """
        cascade_probabilities = {}

        # Define dependency relationships
        dependency_strengths = {
            ('power_system', 'communication'): 0.8,
            ('power_system', 'attitude_control'): 0.9,
            ('power_system', 'thermal_control'): 0.7,
            ('mobility_front', 'mobility_rear'): 0.6,
            ('navigation', 'mobility_front'): 0.5,
            ('navigation', 'mobility_rear'): 0.5
        }

        primary_component = primary_failure.component_id
        base_probability = primary_failure.failure_probability

        for dependent_component in dependent_components:
            # Get dependency strength
            dependency_key = (primary_component, dependent_component)
            reverse_key = (dependent_component, primary_component)

            strength = dependency_strengths.get(dependency_key,
                      dependency_strengths.get(reverse_key, 0.1))  # Default weak dependency

            # Calculate cascade probability
            # P(cascade) = P(primary) * dependency_strength * (1 - system_resilience)
            system_resilience = 0.3  # Assume some system resilience
            cascade_prob = base_probability * strength * (1 - system_resilience)

            cascade_probabilities[dependent_component] = cascade_prob

        return cascade_probabilities

    def _estimate_current_degradation(self, sensor_data: np.ndarray) -> float:
        """Estimate current degradation level from sensor data"""
        # Simple approach: use deviation from normal operation
        if len(sensor_data) == 0:
            return 0.0

        # Normalize sensor data and calculate degradation score
        normalized_data = (sensor_data - np.mean(sensor_data)) / (np.std(sensor_data) + 1e-8)
        degradation_score = np.mean(np.abs(normalized_data))

        return min(degradation_score / 3.0, 1.0)  # Scale to [0, 1]

    def _calculate_degradation_rate(self, degradation_values: np.ndarray) -> float:
        """Calculate rate of degradation"""
        if len(degradation_values) < 2:
            return 0.0

        # Linear regression to find slope
        x = np.arange(len(degradation_values))
        slope, _ = np.polyfit(x, degradation_values, 1)

        return max(slope, 0.0)  # Only positive degradation rates

    def _analyze_forecast_trends(self, forecast_result: EnhancedForecastResult) -> float:
        """Analyze forecast trends for failure indicators"""
        predictions = forecast_result.predictions

        if len(predictions) < 2:
            return 0.0

        # Check for concerning trends
        trend_indicators = []

        # 1. Increasing trend (degradation)
        slope = np.polyfit(range(len(predictions)), predictions.flatten(), 1)[0]
        if slope > 0:
            trend_indicators.append(min(slope * 10, 1.0))  # Scale factor

        # 2. High variability (instability)
        variability = np.std(predictions) / (np.mean(predictions) + 1e-8)
        if variability > 0.2:
            trend_indicators.append(min(variability, 1.0))

        # 3. Forecast uncertainty
        if hasattr(forecast_result, 'prediction_std') and forecast_result.prediction_std is not None:
            uncertainty = np.mean(forecast_result.prediction_std)
            if uncertainty > 0.1:
                trend_indicators.append(min(uncertainty * 5, 1.0))

        return np.mean(trend_indicators) if trend_indicators else 0.0

    def _calculate_anomaly_impact(self, anomaly_scores: np.ndarray) -> float:
        """Calculate failure probability impact from anomaly scores"""
        if len(anomaly_scores) == 0:
            return 0.0

        # Recent anomalies have higher impact
        weights = np.exp(-np.arange(len(anomaly_scores)) * 0.1)
        weighted_scores = anomaly_scores * weights

        impact = np.mean(weighted_scores)
        return min(impact, 1.0)

    def _calculate_operational_stress(self, operational_context: Dict, criticality: str) -> float:
        """Calculate failure probability from operational stress factors"""
        stress_factors = []

        # Temperature stress
        if 'temperature' in operational_context:
            temp = operational_context['temperature']
            normal_temp_range = (20, 25)  # Celsius
            if temp < normal_temp_range[0] or temp > normal_temp_range[1]:
                temp_stress = min(abs(temp - np.mean(normal_temp_range)) / 50, 1.0)
                stress_factors.append(temp_stress)

        # Load stress
        if 'load_factor' in operational_context:
            load = operational_context['load_factor']
            if load > 0.8:  # High load
                load_stress = (load - 0.8) / 0.2
                stress_factors.append(load_stress)

        # Operating hours stress
        if 'operating_hours' in operational_context:
            hours = operational_context['operating_hours']
            if hours > 8760:  # More than 1 year
                age_stress = min((hours - 8760) / 8760, 1.0)
                stress_factors.append(age_stress)

        base_stress = np.mean(stress_factors) if stress_factors else 0.0

        # Apply criticality factor
        criticality_factors = {
            'critical': 1.2,
            'high': 1.1,
            'medium': 1.0,
            'low': 0.9
        }

        return base_stress * criticality_factors.get(criticality, 1.0)

    def _calculate_prediction_uncertainty(self,
                                        forecast_result: EnhancedForecastResult,
                                        num_factors: int) -> float:
        """Calculate overall prediction uncertainty"""
        uncertainty_components = []

        # Model uncertainty from forecast
        if hasattr(forecast_result, 'epistemic_uncertainty') and forecast_result.epistemic_uncertainty is not None:
            model_uncertainty = np.mean(forecast_result.epistemic_uncertainty)
            uncertainty_components.append(model_uncertainty)

        # Data uncertainty
        data_uncertainty = self.config['uncertainty_factors']['data_uncertainty']
        uncertainty_components.append(data_uncertainty)

        # Factor uncertainty (more factors = more uncertainty)
        factor_uncertainty = min(num_factors * 0.02, 0.2)
        uncertainty_components.append(factor_uncertainty)

        return np.sqrt(np.sum(np.array(uncertainty_components) ** 2))

    def _assess_data_quality(self, sensor_data: np.ndarray, forecast_result: EnhancedForecastResult) -> float:
        """Assess data quality for prediction confidence"""
        quality_factors = []

        # Sensor data completeness
        if len(sensor_data) > 0:
            missing_ratio = np.isnan(sensor_data).sum() / len(sensor_data)
            quality_factors.append(1 - missing_ratio)

        # Forecast confidence
        if hasattr(forecast_result, 'confidence_level'):
            quality_factors.append(forecast_result.confidence_level)

        return np.mean(quality_factors) if quality_factors else 0.5

    def _get_severity_from_criticality(self, criticality: str) -> SeverityLevel:
        """Convert criticality to severity level"""
        mapping = {
            'critical': SeverityLevel.CRITICAL,
            'high': SeverityLevel.HIGH,
            'medium': SeverityLevel.MEDIUM,
            'low': SeverityLevel.LOW
        }
        return mapping.get(criticality, SeverityLevel.MEDIUM)

    def get_equipment_risk_summary(self, equipment_id: str) -> Dict[str, Any]:
        """Get risk summary for entire equipment"""
        equipment_type = equipment_id.lower()
        if equipment_type not in self.equipment_config:
            return {}

        components = self.equipment_config[equipment_type]['components']
        risk_summary = {
            'equipment_id': equipment_id,
            'total_components': len(components),
            'high_risk_components': 0,
            'medium_risk_components': 0,
            'low_risk_components': 0,
            'overall_risk_score': 0.0,
            'most_critical_component': None,
            'cascade_risk': 0.0
        }

        # This would be called after running predictions for all components
        # Implementation depends on having actual predictions stored

        return risk_summary

    def save_models(self, filepath: Path):
        """Save all trained models"""
        models_data = {
            'survival_models': {},
            'degradation_models': {},
            'historical_data': self.historical_data,
            'config': self.config
        }

        # Save survival models
        for key, model in self.survival_models.items():
            models_data['survival_models'][key] = {
                'shape': model.shape,
                'scale': model.scale,
                'is_fitted': model.is_fitted
            }

        # Save degradation models (simplified)
        for key, model in self.degradation_models.items():
            models_data['degradation_models'][key] = {
                'model_type': model.model_type,
                'is_fitted': model.is_fitted,
                'failure_threshold': model.failure_threshold
            }

        with open(filepath, 'w') as f:
            json.dump(models_data, f, indent=2, default=str)

        logger.info(f"Failure prediction models saved to {filepath}")


if __name__ == "__main__":
    # Demo and testing
    print("\n" + "="*60)
    print("Testing Failure Probability Estimator")
    print("="*60)

    # Create estimator
    estimator = FailureProbabilityEstimator()

    # Add some synthetic historical data
    print("\n1. Adding historical failure data...")
    failure_times = [1000, 2500, 4200, 5800, 7200]  # Hours
    estimator.add_historical_failure_data('SMAP', 'power_system', failure_times)

    # Add degradation data
    print("2. Adding degradation data...")
    time_points = np.linspace(0, 1000, 100)
    degradation_values = 0.1 + 0.0005 * time_points + 0.05 * np.random.randn(100)
    estimator.add_degradation_data('SMAP', 'power_system', time_points,
                                 degradation_values, failure_threshold=0.8)

    # Create synthetic sensor data and forecast
    print("3. Creating synthetic predictions...")
    sensor_data = np.random.normal(0, 1, 50)

    # Mock enhanced forecast result
    from src.forecasting.enhanced_forecaster import EnhancedForecastResult
    forecast = EnhancedForecastResult(
        predictions=np.random.normal(0, 0.1, 24),
        confidence_lower=np.random.normal(-0.2, 0.05, 24),
        confidence_upper=np.random.normal(0.2, 0.05, 24),
        prediction_std=np.random.uniform(0.05, 0.15, 24),
        epistemic_uncertainty=np.random.uniform(0.02, 0.08, 24)
    )

    # Generate failure prediction
    print("4. Estimating failure probability...")
    prediction = estimator.estimate_failure_probability(
        equipment_id='SMAP',
        component_id='power_system',
        current_sensor_data=sensor_data,
        forecast_result=forecast,
        anomaly_scores=np.random.uniform(0, 0.1, 10),
        operational_context={
            'temperature': 35,  # High temperature
            'load_factor': 0.9,  # High load
            'operating_hours': 10000
        }
    )

    print(f"\nFailure Prediction Results:")
    print(f"  Equipment: {prediction.equipment_id}")
    print(f"  Component: {prediction.component_id}")
    print(f"  Failure Probability: {prediction.failure_probability:.4f}")
    print(f"  Time to Failure: {prediction.time_to_failure:.1f} hours" if prediction.time_to_failure else "  Time to Failure: Not estimated")
    print(f"  Severity: {prediction.severity.value}")
    print(f"  Uncertainty: {prediction.uncertainty:.4f}")
    print(f"  Contributing Factors:")
    for factor, value in prediction.contributing_factors.items():
        print(f"    {factor}: {value:.4f}")

    # Test cascade failure
    print("\n5. Testing cascade failure estimation...")
    cascade_probs = estimator.estimate_cascade_failure_probability(
        prediction,
        ['communication', 'attitude_control', 'thermal_control']
    )

    print(f"Cascade Failure Probabilities:")
    for component, prob in cascade_probs.items():
        print(f"  {component}: {prob:.4f}")

    print("\n" + "="*60)
    print("Failure probability estimator test complete")
    print("="*60)