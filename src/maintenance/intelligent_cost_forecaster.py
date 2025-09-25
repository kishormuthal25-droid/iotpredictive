"""
Intelligent Cost Forecasting Module for Phase 3.1 IoT Predictive Maintenance
ML-based cost prediction, budget optimization, and financial analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import json
import pickle
import warnings
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Time series forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class CostComponent:
    """Individual cost component for detailed tracking"""
    component_id: str
    name: str
    category: str  # 'labor', 'parts', 'overhead', 'equipment', 'external'
    base_cost: float
    variable_factors: Dict[str, float]  # factor_name -> multiplier
    seasonal_pattern: Optional[List[float]] = None  # 12-month seasonal multipliers
    inflation_rate: float = 0.03  # Annual inflation rate
    cost_volatility: float = 0.1  # Standard deviation of cost variations


@dataclass
class CostForecast:
    """Cost forecast result with confidence intervals"""
    forecast_date: datetime
    predicted_cost: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    component_breakdown: Dict[str, float]
    forecast_horizon_days: int
    model_confidence: float
    risk_factors: List[str]


@dataclass
class BudgetOptimization:
    """Budget optimization recommendation"""
    optimization_id: str
    current_budget: float
    recommended_budget: float
    cost_savings: float
    risk_level: str  # 'low', 'medium', 'high'
    recommendations: List[Dict[str, Any]]
    confidence_score: float


class IntelligentCostForecaster:
    """Advanced cost forecasting with ML predictions and optimization"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Intelligent Cost Forecaster

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.historical_data = pd.DataFrame()
        self.cost_components = {}

        # Model configuration
        self.model_types = ['random_forest', 'gradient_boosting', 'arima', 'lstm']
        self.ensemble_weights = {'random_forest': 0.3, 'gradient_boosting': 0.3, 'arima': 0.2, 'lstm': 0.2}

        # Feature engineering parameters
        self.feature_columns = []
        self.target_column = 'total_cost'

        # Forecast parameters
        self.default_horizon_days = 30
        self.confidence_level = 0.95

        # Initialize components
        self._initialize_cost_components()
        self._initialize_models()

        logger.info("Initialized Intelligent Cost Forecaster")

    def _initialize_cost_components(self):
        """Initialize cost component definitions"""
        self.cost_components = {
            'labor_electrical': CostComponent(
                'labor_electrical', 'Electrical Labor', 'labor', 75.0,
                {'overtime': 1.5, 'weekend': 1.3, 'holiday': 2.0, 'emergency': 1.8},
                [1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0],  # Seasonal
                0.04, 0.15
            ),
            'labor_mechanical': CostComponent(
                'labor_mechanical', 'Mechanical Labor', 'labor', 70.0,
                {'overtime': 1.5, 'weekend': 1.3, 'holiday': 2.0, 'emergency': 1.8},
                [1.0, 1.0, 1.0, 1.1, 1.2, 1.2, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0],
                0.038, 0.12
            ),
            'parts_bearings': CostComponent(
                'parts_bearings', 'Bearings and Components', 'parts', 125.0,
                {'supply_shortage': 1.4, 'rush_order': 1.6, 'quality_premium': 1.2},
                [1.0, 1.1, 1.0, 1.0, 1.1, 1.2, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1],
                0.05, 0.25
            ),
            'parts_electronics': CostComponent(
                'parts_electronics', 'Electronic Components', 'parts', 85.0,
                {'supply_shortage': 1.6, 'rush_order': 1.8, 'technology_upgrade': 0.9},
                [1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0],
                0.02, 0.35  # High volatility due to tech market
            ),
            'overhead_facility': CostComponent(
                'overhead_facility', 'Facility Overhead', 'overhead', 50.0,
                {'utilization': 1.2, 'energy_cost': 1.1},
                [1.2, 1.1, 1.0, 1.0, 1.1, 1.3, 1.3, 1.2, 1.0, 1.0, 1.1, 1.2],
                0.035, 0.08
            ),
            'equipment_rental': CostComponent(
                'equipment_rental', 'Equipment Rental', 'equipment', 200.0,
                {'demand_surge': 1.3, 'availability': 1.5},
                [1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0],
                0.04, 0.20
            )
        }

    def _initialize_models(self):
        """Initialize ML models for cost forecasting"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=42, max_depth=10
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            'linear_regression': Ridge(alpha=1.0)
        }

        # Initialize scalers
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler()
        }

        # Initialize LSTM model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.models['lstm'] = self._create_lstm_model()

    def _create_lstm_model(self, sequence_length: int = 30, features: int = 10) -> Optional[tf.keras.Model]:
        """Create LSTM model for time series forecasting

        Args:
            sequence_length: Length of input sequences
            features: Number of input features

        Returns:
            Compiled LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            return None

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def load_historical_data(self, data_source: Union[str, pd.DataFrame]) -> bool:
        """Load historical cost data for training

        Args:
            data_source: File path or DataFrame with historical data

        Returns:
            True if data loaded successfully
        """
        try:
            if isinstance(data_source, str):
                # Load from file
                if data_source.endswith('.csv'):
                    self.historical_data = pd.read_csv(data_source)
                elif data_source.endswith('.json'):
                    self.historical_data = pd.read_json(data_source)
                else:
                    logger.error(f"Unsupported file format: {data_source}")
                    return False
            elif isinstance(data_source, pd.DataFrame):
                self.historical_data = data_source.copy()
            else:
                logger.error("Invalid data source type")
                return False

            # Ensure required columns exist
            if 'date' not in self.historical_data.columns:
                logger.error("Historical data must contain 'date' column")
                return False

            # Convert date column
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.historical_data = self.historical_data.sort_values('date')

            # Generate features if not present
            self._generate_features()

            logger.info(f"Loaded {len(self.historical_data)} historical records")
            return True

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False

    def generate_synthetic_data(self, start_date: datetime, end_date: datetime,
                               frequency: str = 'D') -> pd.DataFrame:
        """Generate synthetic historical data for demonstration

        Args:
            start_date: Start date for synthetic data
            end_date: End date for synthetic data
            frequency: Date frequency ('D' for daily, 'W' for weekly)

        Returns:
            Synthetic historical data DataFrame
        """
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)

        synthetic_data = []
        base_cost = 2000  # Base daily maintenance cost

        for i, date in enumerate(date_range):
            # Generate synthetic cost components
            cost_record = {
                'date': date,
                'equipment_count': np.random.randint(5, 15),
                'technician_count': np.random.randint(2, 8),
                'emergency_repairs': np.random.poisson(0.3),
                'scheduled_maintenance': np.random.poisson(2.5),
                'parts_orders': np.random.poisson(1.8),
                'overtime_hours': max(0, np.random.normal(2, 4)),
                'season': (date.month - 1) // 3 + 1,  # 1-4 for quarters
                'day_of_week': date.weekday(),
                'is_weekend': date.weekday() >= 5,
                'is_holiday': self._is_holiday(date),
            }

            # Calculate component costs
            labor_cost = self._calculate_synthetic_labor_cost(cost_record, date)
            parts_cost = self._calculate_synthetic_parts_cost(cost_record, date)
            overhead_cost = self._calculate_synthetic_overhead_cost(cost_record, date)
            equipment_cost = self._calculate_synthetic_equipment_cost(cost_record, date)

            cost_record.update({
                'labor_cost': labor_cost,
                'parts_cost': parts_cost,
                'overhead_cost': overhead_cost,
                'equipment_cost': equipment_cost,
                'total_cost': labor_cost + parts_cost + overhead_cost + equipment_cost
            })

            # Add noise and trends
            trend_factor = 1 + 0.02 * (i / len(date_range))  # 2% annual increase
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25)
            noise_factor = 1 + np.random.normal(0, 0.1)

            cost_record['total_cost'] *= trend_factor * seasonal_factor * noise_factor

            synthetic_data.append(cost_record)

        df = pd.DataFrame(synthetic_data)

        # Load as historical data
        self.historical_data = df
        self._generate_features()

        logger.info(f"Generated {len(df)} synthetic records from {start_date} to {end_date}")
        return df

    def _calculate_synthetic_labor_cost(self, record: Dict[str, Any], date: datetime) -> float:
        """Calculate synthetic labor cost component"""
        base_hourly = 75.0
        regular_hours = record['technician_count'] * 8
        overtime_hours = record['overtime_hours']

        # Apply multipliers
        weekend_multiplier = 1.3 if record['is_weekend'] else 1.0
        holiday_multiplier = 2.0 if record['is_holiday'] else 1.0
        emergency_multiplier = 1.0 + (record['emergency_repairs'] * 0.2)

        regular_cost = regular_hours * base_hourly * weekend_multiplier * holiday_multiplier
        overtime_cost = overtime_hours * base_hourly * 1.5 * weekend_multiplier * holiday_multiplier
        emergency_cost = record['emergency_repairs'] * 200 * emergency_multiplier

        return regular_cost + overtime_cost + emergency_cost

    def _calculate_synthetic_parts_cost(self, record: Dict[str, Any], date: datetime) -> float:
        """Calculate synthetic parts cost component"""
        base_parts_cost = record['parts_orders'] * 150
        emergency_parts = record['emergency_repairs'] * 300

        # Seasonal adjustment (higher costs in winter/summer)
        seasonal_multiplier = 1.0 + 0.2 * abs(np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25))

        # Supply chain volatility
        supply_volatility = np.random.lognormal(0, 0.15)

        return (base_parts_cost + emergency_parts) * seasonal_multiplier * supply_volatility

    def _calculate_synthetic_overhead_cost(self, record: Dict[str, Any], date: datetime) -> float:
        """Calculate synthetic overhead cost component"""
        base_overhead = 200  # Base daily overhead
        equipment_factor = record['equipment_count'] * 10
        utilization_factor = (record['scheduled_maintenance'] + record['emergency_repairs']) * 25

        # Energy cost seasonality (higher in summer/winter)
        energy_multiplier = 1.0 + 0.15 * abs(np.sin(2 * np.pi * (date.timetuple().tm_yday - 80) / 365.25))

        return (base_overhead + equipment_factor + utilization_factor) * energy_multiplier

    def _calculate_synthetic_equipment_cost(self, record: Dict[str, Any], date: datetime) -> float:
        """Calculate synthetic equipment cost component"""
        base_equipment = record['emergency_repairs'] * 500  # Equipment rental for emergencies
        specialized_equipment = record['scheduled_maintenance'] * 100  # Tools and equipment

        # Market demand seasonality
        demand_multiplier = 1.0 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25)

        return (base_equipment + specialized_equipment) * demand_multiplier

    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # New Year's Day, Independence Day, Christmas Day (simplified)
        holidays = [(1, 1), (7, 4), (12, 25)]
        return (date.month, date.day) in holidays

    def _generate_features(self):
        """Generate features for ML models from historical data"""
        if self.historical_data.empty:
            return

        df = self.historical_data.copy()

        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter

        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Lag features
        for lag in [1, 7, 30]:
            df[f'total_cost_lag_{lag}'] = df['total_cost'].shift(lag)

        # Rolling statistics
        for window in [7, 30]:
            df[f'total_cost_rolling_mean_{window}'] = df['total_cost'].rolling(window=window).mean()
            df[f'total_cost_rolling_std_{window}'] = df['total_cost'].rolling(window=window).std()

        # Component ratios
        if 'labor_cost' in df.columns and 'total_cost' in df.columns:
            df['labor_ratio'] = df['labor_cost'] / (df['total_cost'] + 1e-6)
            df['parts_ratio'] = df.get('parts_cost', 0) / (df['total_cost'] + 1e-6)
            df['overhead_ratio'] = df.get('overhead_cost', 0) / (df['total_cost'] + 1e-6)

        # External factors (if available)
        if 'equipment_count' in df.columns:
            df['cost_per_equipment'] = df['total_cost'] / (df['equipment_count'] + 1e-6)

        # Set feature columns
        self.feature_columns = [col for col in df.columns
                               if col not in ['date', 'total_cost'] and not col.startswith('cost_')]

        self.historical_data = df
        logger.info(f"Generated {len(self.feature_columns)} features for ML models")

    def train_models(self, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """Train all forecasting models

        Args:
            test_size: Fraction of data to use for testing

        Returns:
            Model performance metrics
        """
        if self.historical_data.empty:
            logger.error("No historical data available for training")
            return {}

        # Prepare data
        df = self.historical_data.dropna()
        if len(df) < 50:
            logger.warning("Insufficient data for robust model training")

        X = df[self.feature_columns]
        y = df[self.target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        # Scale features
        X_train_scaled = self.scalers['features'].fit_transform(X_train)
        X_test_scaled = self.scalers['features'].transform(X_test)

        performance_metrics = {}

        # Train traditional ML models
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                continue  # Handle LSTM separately

            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = model.predict(X_test_scaled)

                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
                }

                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()

                performance_metrics[model_name] = metrics
                logger.info(f"Trained {model_name}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")

        # Train LSTM model if available
        if TENSORFLOW_AVAILABLE and 'lstm' in self.models:
            try:
                lstm_metrics = self._train_lstm_model(X_train, X_test, y_train, y_test)
                performance_metrics['lstm'] = lstm_metrics
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}")

        # Train ARIMA model if available
        if STATSMODELS_AVAILABLE:
            try:
                arima_metrics = self._train_arima_model(df)
                performance_metrics['arima'] = arima_metrics
            except Exception as e:
                logger.error(f"Error training ARIMA model: {e}")

        return performance_metrics

    def _train_lstm_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
        """Train LSTM model for time series forecasting

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets

        Returns:
            Performance metrics
        """
        sequence_length = 30

        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)

        if len(X_train_seq) == 0:
            logger.warning("Insufficient data for LSTM sequence creation")
            return {}

        # Scale data
        X_train_seq_scaled = self.scalers['features'].fit_transform(
            X_train_seq.reshape(-1, X_train_seq.shape[-1])
        ).reshape(X_train_seq.shape)

        X_test_seq_scaled = self.scalers['features'].transform(
            X_test_seq.reshape(-1, X_test_seq.shape[-1])
        ).reshape(X_test_seq.shape)

        # Create and train LSTM model
        lstm_model = self._create_lstm_model(sequence_length, X_train_seq.shape[-1])

        # Train with early stopping
        history = lstm_model.fit(
            X_train_seq_scaled, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )

        # Evaluate
        y_pred = lstm_model.predict(X_test_seq_scaled)

        metrics = {
            'mae': mean_absolute_error(y_test_seq, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_seq, y_pred)),
            'r2': r2_score(y_test_seq, y_pred),
            'mape': np.mean(np.abs((y_test_seq - y_pred.flatten()) / (y_test_seq + 1e-6))) * 100
        }

        # Update model in storage
        self.models['lstm'] = lstm_model

        logger.info(f"Trained LSTM: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")
        return metrics

    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training

        Args:
            X: Features
            y: Targets
            sequence_length: Length of sequences

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []

        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])

        return np.array(X_seq), np.array(y_seq)

    def _train_arima_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train ARIMA model for time series forecasting

        Args:
            df: Historical data DataFrame

        Returns:
            Performance metrics
        """
        # Prepare time series
        ts = df.set_index('date')[self.target_column].resample('D').mean().fillna(method='ffill')

        # Check stationarity
        adf_result = adfuller(ts.dropna())
        is_stationary = adf_result[1] < 0.05

        # Determine differencing order
        d = 0 if is_stationary else 1

        # Simple ARIMA(1,d,1) model
        try:
            model = ARIMA(ts, order=(1, d, 1))
            fitted_model = model.fit()

            # Forecast on test set (last 20% of data)
            test_size = int(len(ts) * 0.2)
            train_ts = ts[:-test_size]
            test_ts = ts[-test_size:]

            # Refit on training data
            train_model = ARIMA(train_ts, order=(1, d, 1))
            fitted_train_model = train_model.fit()

            # Forecast
            forecast = fitted_train_model.forecast(steps=test_size)

            metrics = {
                'mae': mean_absolute_error(test_ts, forecast),
                'rmse': np.sqrt(mean_squared_error(test_ts, forecast)),
                'r2': r2_score(test_ts, forecast),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }

            # Store fitted model
            self.models['arima'] = fitted_model

            logger.info(f"Trained ARIMA: R² = {metrics['r2']:.3f}, AIC = {metrics['aic']:.2f}")
            return metrics

        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {}

    def forecast_costs(self, forecast_date: datetime,
                      horizon_days: int = None) -> CostForecast:
        """Generate cost forecast using ensemble of models

        Args:
            forecast_date: Date to forecast from
            horizon_days: Forecast horizon in days

        Returns:
            Cost forecast with confidence intervals
        """
        if horizon_days is None:
            horizon_days = self.default_horizon_days

        # Generate future features
        future_data = self._generate_future_features(forecast_date, horizon_days)

        # Get predictions from all models
        predictions = {}

        for model_name, model in self.models.items():
            if model_name == 'lstm':
                pred = self._predict_lstm(future_data)
            elif model_name == 'arima':
                pred = self._predict_arima(forecast_date, horizon_days)
            else:
                X_future = future_data[self.feature_columns]
                X_scaled = self.scalers['features'].transform(X_future)
                pred = model.predict(X_scaled)

            if pred is not None:
                predictions[model_name] = pred

        # Ensemble prediction
        if predictions:
            # Weighted average of predictions
            total_weight = sum(self.ensemble_weights.get(name, 0.25) for name in predictions.keys())

            ensemble_pred = sum(
                pred * self.ensemble_weights.get(name, 0.25) / total_weight
                for name, pred in predictions.items()
            )

            # Calculate confidence intervals (using prediction variance)
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values, axis=0)
                confidence_factor = 1.96  # 95% confidence interval

                if isinstance(ensemble_pred, np.ndarray):
                    predicted_cost = ensemble_pred[-1]  # Last prediction
                    ci_lower = predicted_cost - confidence_factor * pred_std[-1]
                    ci_upper = predicted_cost + confidence_factor * pred_std[-1]
                else:
                    predicted_cost = float(ensemble_pred)
                    ci_lower = predicted_cost - confidence_factor * np.mean(pred_std)
                    ci_upper = predicted_cost + confidence_factor * np.mean(pred_std)
            else:
                predicted_cost = float(ensemble_pred)
                ci_lower = predicted_cost * 0.9
                ci_upper = predicted_cost * 1.1

            # Generate component breakdown
            component_breakdown = self._generate_component_breakdown(predicted_cost, forecast_date)

            # Assess risk factors
            risk_factors = self._assess_risk_factors(forecast_date, horizon_days)

            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(predictions)

            forecast = CostForecast(
                forecast_date=forecast_date + timedelta(days=horizon_days),
                predicted_cost=predicted_cost,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                component_breakdown=component_breakdown,
                forecast_horizon_days=horizon_days,
                model_confidence=model_confidence,
                risk_factors=risk_factors
            )

            logger.info(f"Generated cost forecast: ${predicted_cost:.2f} ± ${(ci_upper-ci_lower)/2:.2f}")
            return forecast

        else:
            logger.error("No models available for forecasting")
            return None

    def _generate_future_features(self, start_date: datetime, horizon_days: int) -> pd.DataFrame:
        """Generate features for future dates

        Args:
            start_date: Start date for forecast
            horizon_days: Number of days to forecast

        Returns:
            DataFrame with future features
        """
        future_dates = pd.date_range(start=start_date, periods=horizon_days, freq='D')
        future_data = []

        for date in future_dates:
            # Basic time features
            row = {
                'date': date,
                'year': date.year,
                'month': date.month,
                'day_of_year': date.timetuple().tm_yday,
                'day_of_week': date.weekday(),
                'is_weekend': int(date.weekday() >= 5),
                'quarter': (date.month - 1) // 3 + 1,

                # Cyclical features
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'day_sin': np.sin(2 * np.pi * date.weekday() / 7),
                'day_cos': np.cos(2 * np.pi * date.weekday() / 7),
            }

            # Estimate other features based on patterns
            if not self.historical_data.empty:
                # Use seasonal averages for missing features
                same_month_data = self.historical_data[self.historical_data['month'] == date.month]

                for col in self.feature_columns:
                    if col not in row:
                        if len(same_month_data) > 0 and col in same_month_data.columns:
                            row[col] = same_month_data[col].mean()
                        else:
                            row[col] = 0  # Default value

            future_data.append(row)

        return pd.DataFrame(future_data)

    def _predict_lstm(self, future_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate LSTM predictions

        Args:
            future_data: Future features DataFrame

        Returns:
            LSTM predictions
        """
        if 'lstm' not in self.models or not TENSORFLOW_AVAILABLE:
            return None

        sequence_length = 30

        # Use last sequence_length points from historical data
        if len(self.historical_data) < sequence_length:
            return None

        last_sequence = self.historical_data[self.feature_columns].tail(sequence_length)

        # Scale the sequence
        last_sequence_scaled = self.scalers['features'].transform(last_sequence)

        # Reshape for LSTM
        X_pred = last_sequence_scaled.reshape(1, sequence_length, -1)

        # Predict
        prediction = self.models['lstm'].predict(X_pred, verbose=0)

        return prediction.flatten()

    def _predict_arima(self, forecast_date: datetime, horizon_days: int) -> Optional[np.ndarray]:
        """Generate ARIMA predictions

        Args:
            forecast_date: Forecast start date
            horizon_days: Forecast horizon

        Returns:
            ARIMA predictions
        """
        if 'arima' not in self.models or not STATSMODELS_AVAILABLE:
            return None

        try:
            forecast = self.models['arima'].forecast(steps=horizon_days)
            return np.array(forecast)
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            return None

    def _generate_component_breakdown(self, total_cost: float, forecast_date: datetime) -> Dict[str, float]:
        """Generate cost component breakdown

        Args:
            total_cost: Total predicted cost
            forecast_date: Forecast date

        Returns:
            Component breakdown dictionary
        """
        breakdown = {}

        # Use historical ratios or defaults
        if not self.historical_data.empty:
            labor_ratio = self.historical_data['labor_ratio'].mean() if 'labor_ratio' in self.historical_data.columns else 0.6
            parts_ratio = self.historical_data['parts_ratio'].mean() if 'parts_ratio' in self.historical_data.columns else 0.25
            overhead_ratio = self.historical_data['overhead_ratio'].mean() if 'overhead_ratio' in self.historical_data.columns else 0.15
        else:
            labor_ratio, parts_ratio, overhead_ratio = 0.6, 0.25, 0.15

        # Adjust for seasonality and other factors
        month = forecast_date.month
        seasonal_labor_factor = 1.0 + 0.1 * np.sin(2 * np.pi * month / 12)
        seasonal_parts_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (month - 3) / 12)

        breakdown['labor'] = total_cost * labor_ratio * seasonal_labor_factor
        breakdown['parts'] = total_cost * parts_ratio * seasonal_parts_factor
        breakdown['overhead'] = total_cost * overhead_ratio
        breakdown['equipment'] = total_cost - breakdown['labor'] - breakdown['parts'] - breakdown['overhead']

        return breakdown

    def _assess_risk_factors(self, forecast_date: datetime, horizon_days: int) -> List[str]:
        """Assess risk factors for cost forecast

        Args:
            forecast_date: Forecast date
            horizon_days: Forecast horizon

        Returns:
            List of risk factors
        """
        risk_factors = []

        # Seasonal risks
        month = forecast_date.month
        if month in [12, 1, 2]:  # Winter
            risk_factors.append("Winter season: Higher heating costs and equipment stress")
        elif month in [6, 7, 8]:  # Summer
            risk_factors.append("Summer season: Higher cooling costs and equipment demand")

        # Weekend/holiday risks
        if forecast_date.weekday() >= 5:
            risk_factors.append("Weekend work: Premium labor rates")

        # Long-term horizon risks
        if horizon_days > 60:
            risk_factors.append("Long forecast horizon: Increased uncertainty")

        # Data quality risks
        if len(self.historical_data) < 365:
            risk_factors.append("Limited historical data: Model uncertainty")

        # Market volatility (simulated)
        if np.random.random() < 0.3:  # 30% chance
            risk_factors.append("Supply chain volatility: Parts cost fluctuation")

        return risk_factors

    def _calculate_model_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall model confidence

        Args:
            predictions: Model predictions dictionary

        Returns:
            Confidence score (0-1)
        """
        if not predictions:
            return 0.0

        # Base confidence on model agreement
        pred_values = [float(p) if not isinstance(p, np.ndarray) else float(p[-1])
                      for p in predictions.values()]

        if len(pred_values) == 1:
            return 0.7  # Single model confidence

        # Calculate coefficient of variation
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        cv = std_pred / (mean_pred + 1e-6)

        # Convert to confidence (inverse relationship)
        confidence = max(0.0, min(1.0, 1.0 - cv))

        return confidence

    def optimize_budget(self, current_budget: float, forecast_period_days: int = 90) -> BudgetOptimization:
        """Optimize budget allocation using ML insights

        Args:
            current_budget: Current budget amount
            forecast_period_days: Period for optimization

        Returns:
            Budget optimization recommendations
        """
        forecast_date = datetime.now()
        forecast = self.forecast_costs(forecast_date, forecast_period_days)

        if not forecast:
            logger.error("Cannot optimize budget without cost forecast")
            return None

        # Analyze budget vs forecast
        predicted_total = forecast.predicted_cost
        budget_variance = (current_budget - predicted_total) / current_budget

        # Determine risk level
        if budget_variance < -0.15:  # Over budget by >15%
            risk_level = 'high'
        elif budget_variance < -0.05:  # Over budget by >5%
            risk_level = 'medium'
        else:
            risk_level = 'low'

        # Generate recommendations
        recommendations = []

        if budget_variance < 0:  # Over budget
            recommendations.extend(self._generate_cost_reduction_recommendations(forecast))
        else:  # Under budget
            recommendations.extend(self._generate_investment_recommendations(forecast))

        # Calculate optimized budget
        recommended_budget = self._calculate_optimized_budget(current_budget, forecast)
        cost_savings = max(0, current_budget - recommended_budget)

        optimization = BudgetOptimization(
            optimization_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            current_budget=current_budget,
            recommended_budget=recommended_budget,
            cost_savings=cost_savings,
            risk_level=risk_level,
            recommendations=recommendations,
            confidence_score=forecast.model_confidence
        )

        logger.info(f"Budget optimization: ${current_budget:.2f} → ${recommended_budget:.2f} (savings: ${cost_savings:.2f})")
        return optimization

    def _generate_cost_reduction_recommendations(self, forecast: CostForecast) -> List[Dict[str, Any]]:
        """Generate cost reduction recommendations

        Args:
            forecast: Cost forecast

        Returns:
            List of recommendations
        """
        recommendations = []

        # Analyze component breakdown for reduction opportunities
        breakdown = forecast.component_breakdown

        if breakdown.get('labor', 0) > breakdown.get('parts', 0) * 2:
            recommendations.append({
                'type': 'labor_optimization',
                'title': 'Optimize Labor Allocation',
                'description': 'Labor costs are disproportionately high. Consider cross-training and workload balancing.',
                'potential_savings': breakdown['labor'] * 0.15,
                'implementation_cost': 5000,
                'priority': 'high'
            })

        if breakdown.get('parts', 0) > forecast.predicted_cost * 0.4:
            recommendations.append({
                'type': 'parts_optimization',
                'title': 'Parts Inventory Optimization',
                'description': 'Parts costs are high. Consider bulk purchasing and alternative suppliers.',
                'potential_savings': breakdown['parts'] * 0.12,
                'implementation_cost': 2000,
                'priority': 'medium'
            })

        # Generic recommendations
        recommendations.append({
            'type': 'preventive_maintenance',
            'title': 'Increase Preventive Maintenance',
            'description': 'Shift from reactive to preventive maintenance to reduce emergency costs.',
            'potential_savings': forecast.predicted_cost * 0.20,
            'implementation_cost': 8000,
            'priority': 'high'
        })

        return recommendations

    def _generate_investment_recommendations(self, forecast: CostForecast) -> List[Dict[str, Any]]:
        """Generate investment recommendations when under budget

        Args:
            forecast: Cost forecast

        Returns:
            List of recommendations
        """
        recommendations = []

        recommendations.append({
            'type': 'equipment_upgrade',
            'title': 'Equipment Technology Upgrade',
            'description': 'Invest in more efficient equipment to reduce long-term costs.',
            'potential_savings': forecast.predicted_cost * 0.25,
            'implementation_cost': 50000,
            'priority': 'medium'
        })

        recommendations.append({
            'type': 'training_investment',
            'title': 'Technician Training Program',
            'description': 'Invest in advanced training to improve efficiency and reduce errors.',
            'potential_savings': forecast.predicted_cost * 0.10,
            'implementation_cost': 15000,
            'priority': 'high'
        })

        return recommendations

    def _calculate_optimized_budget(self, current_budget: float, forecast: CostForecast) -> float:
        """Calculate optimized budget amount

        Args:
            current_budget: Current budget
            forecast: Cost forecast

        Returns:
            Optimized budget amount
        """
        # Use upper confidence interval plus buffer
        safety_margin = 0.1  # 10% safety margin
        optimized_budget = forecast.confidence_interval_upper * (1 + safety_margin)

        # Don't recommend more than 20% change from current budget
        max_change = current_budget * 0.2

        if optimized_budget > current_budget + max_change:
            optimized_budget = current_budget + max_change
        elif optimized_budget < current_budget - max_change:
            optimized_budget = current_budget - max_change

        return optimized_budget

    def save_models(self, model_dir: str):
        """Save trained models to disk

        Args:
            model_dir: Directory to save models
        """
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)

        # Save scikit-learn models
        for name, model in self.models.items():
            if name not in ['lstm', 'arima']:
                joblib.dump(model, model_path / f"{name}_model.pkl")

        # Save scalers
        joblib.dump(self.scalers, model_path / "scalers.pkl")

        # Save TensorFlow model
        if 'lstm' in self.models and TENSORFLOW_AVAILABLE:
            self.models['lstm'].save(model_path / "lstm_model.h5")

        # Save ARIMA model
        if 'arima' in self.models and STATSMODELS_AVAILABLE:
            with open(model_path / "arima_model.pkl", 'wb') as f:
                pickle.dump(self.models['arima'], f)

        logger.info(f"Models saved to {model_dir}")

    def load_models(self, model_dir: str):
        """Load trained models from disk

        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)

        # Load scikit-learn models
        for model_file in model_path.glob("*_model.pkl"):
            if "arima" not in model_file.name:
                model_name = model_file.stem.replace("_model", "")
                self.models[model_name] = joblib.load(model_file)

        # Load scalers
        scaler_file = model_path / "scalers.pkl"
        if scaler_file.exists():
            self.scalers = joblib.load(scaler_file)

        # Load TensorFlow model
        lstm_file = model_path / "lstm_model.h5"
        if lstm_file.exists() and TENSORFLOW_AVAILABLE:
            self.models['lstm'] = tf.keras.models.load_model(lstm_file)

        # Load ARIMA model
        arima_file = model_path / "arima_model.pkl"
        if arima_file.exists():
            with open(arima_file, 'rb') as f:
                self.models['arima'] = pickle.load(f)

        logger.info(f"Models loaded from {model_dir}")


# Create demo instance
def create_demo_cost_forecaster() -> IntelligentCostForecaster:
    """Create demo cost forecaster with synthetic data

    Returns:
        Configured cost forecaster
    """
    forecaster = IntelligentCostForecaster()

    # Generate synthetic historical data
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now() - timedelta(days=1)

    forecaster.generate_synthetic_data(start_date, end_date)

    # Train models
    forecaster.train_models()

    return forecaster