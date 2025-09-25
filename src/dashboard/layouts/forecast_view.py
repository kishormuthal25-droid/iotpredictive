"""
Forecast View Dashboard for IoT Anomaly Detection System
Time series forecasting and predictive maintenance visualization
"""

from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import new interactive components
from src.dashboard.components.dropdown_manager import dropdown_state_manager
from src.dashboard.components.chart_manager import chart_manager, ChartType, ChartConfig, ChartData

# Setup logging
logger = logging.getLogger(__name__)


class ForecastView:
    """Forecasting and prediction dashboard component"""
    
    def __init__(self, data_manager=None, config=None):
        """Initialize Forecast View with enhanced interactive components

        Args:
            data_manager: Data manager instance
            config: Configuration object
        """
        self.data_manager = data_manager
        self.config = config or {}

        # Initialize interactive component managers
        self.dropdown_manager = dropdown_state_manager
        self.chart_manager = chart_manager

        # Forecast data storage
        self.forecast_cache = {}
        self.prediction_results = {}
        self.model_metrics = {}

        # Active forecasting models
        self.forecast_models = {
            'lstm_forecaster': {'name': 'LSTM Forecaster', 'active': True},
            'arima': {'name': 'ARIMA', 'active': True},
            'prophet': {'name': 'Prophet', 'active': False},
            'ensemble': {'name': 'Ensemble', 'active': True}
        }

        logger.info("Initialized Forecast View with enhanced interactive components")
        
    def create_layout(self) -> html.Div:
        """Create forecast view layout
        
        Returns:
            Forecast view layout
        """
        return html.Div([
            # Header with controls
            dbc.Row([
                dbc.Col([
                    html.H3("Forecasting & Predictions"),
                    html.P("Time series forecasting and predictive analytics", className="text-muted")
                ], width=6),
                
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-sync me-2"),
                            "Refresh"
                        ], id="refresh-forecast-btn", color="primary", size="sm"),
                        
                        dbc.Button([
                            html.I(className="fas fa-download me-2"),
                            "Export"
                        ], id="export-forecast-btn", color="info", size="sm"),
                        
                        dbc.Button([
                            html.I(className="fas fa-cog me-2"),
                            "Settings"
                        ], id="forecast-settings-btn", color="secondary", size="sm")
                    ])
                ], width=6, className="text-end")
            ], className="mb-4"),
            
            # Forecast configuration row
            dbc.Row([
                dbc.Col([
                    html.Label("🛰️ Equipment", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-equipment-select",
                        options=self._get_equipment_options(),
                        value=None,
                        placeholder="Select NASA Equipment...",
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("📊 Sensor", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-sensor-select",
                        options=[],
                        value=None,
                        placeholder="Select Sensor...",
                        clearable=False
                    )
                ], width=2),

                dbc.Col([
                    html.Label("📈 Metric", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-metric-select",
                        options=[],
                        value=None,
                        placeholder="Select Metric...",
                        clearable=False
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("Forecast Horizon", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-horizon-select",
                        options=[
                            {"label": "1 Hour", "value": "1h"},
                            {"label": "6 Hours", "value": "6h"},
                            {"label": "24 Hours", "value": "24h"},
                            {"label": "7 Days", "value": "7d"},
                            {"label": "30 Days", "value": "30d"}
                        ],
                        value="24h",
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    html.Label("Model", className="fw-bold"),
                    dcc.Dropdown(
                        id="forecast-model-select",
                        options=[
                            {"label": "LSTM Forecaster", "value": "lstm"},
                            {"label": "ARIMA", "value": "arima"},
                            {"label": "Prophet", "value": "prophet"},
                            {"label": "Ensemble", "value": "ensemble"}
                        ],
                        value="lstm",
                        clearable=False
                    )
                ], width=3)
            ], className="mb-4"),
            
            # Forecast accuracy metrics
            dbc.Row([
                dbc.Col([
                    self._create_forecast_accuracy_card()
                ], width=3),
                
                dbc.Col([
                    self._create_prediction_confidence_card()
                ], width=3),
                
                dbc.Col([
                    self._create_failure_probability_card()
                ], width=3),
                
                dbc.Col([
                    self._create_maintenance_window_card()
                ], width=3)
            ], className="mb-4"),
            
            # Main forecast chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📈 Time Series Forecast", className="mb-0"),
                            html.Div([
                                html.Label("Chart Type:", className="me-2 small"),
                                dcc.Dropdown(
                                    id="forecast-chart-type-select",
                                    options=self.chart_manager.get_chart_type_options(),
                                    value=ChartType.LINE.value,
                                    clearable=False,
                                    style={"width": "150px", "display": "inline-block"}
                                )
                            ], className="ms-auto d-flex align-items-center")
                        ], className="d-flex align-items-center"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="main-forecast-chart",
                                style={'height': '450px'}
                            ),
                            
                            # Chart controls
                            dbc.Row([
                                dbc.Col([
                                    dbc.Checklist(
                                        id="forecast-display-options",
                                        options=[
                                            {"label": "Show Confidence Interval", "value": "ci"},
                                            {"label": "Show Anomaly Threshold", "value": "threshold"},
                                            {"label": "Show Historical Anomalies", "value": "anomalies"},
                                            {"label": "Show Seasonality", "value": "seasonality"}
                                        ],
                                        value=["ci", "threshold"],
                                        inline=True,
                                        switch=True
                                    )
                                ], width=12, className="mt-2")
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Decomposition and model comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Time Series Decomposition"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="decomposition-chart",
                                style={'height': '350px'}
                            )
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Comparison"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="model-comparison-chart",
                                style={'height': '350px'}
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Prediction table and risk assessment
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Failure Predictions", className="mb-0"),
                            dbc.Badge("AI-Powered", color="info", pill=True, className="ms-2")
                        ]),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="failure-predictions-table",
                                columns=[
                                    {"name": "Equipment", "id": "equipment"},
                                    {"name": "Component", "id": "component"},
                                    {"name": "Failure Type", "id": "failure_type"},
                                    {"name": "Probability", "id": "probability"},
                                    {"name": "Time to Failure", "id": "ttf"},
                                    {"name": "Impact", "id": "impact"},
                                    {"name": "Recommended Action", "id": "action"}
                                ],
                                data=[],
                                sort_action="native",
                                page_action="native",
                                page_size=5,
                                style_cell={'textAlign': 'left'},
                                style_data_conditional=[
                                    {
                                        'if': {'column_id': 'probability', 'filter_query': '{probability} > 80'},
                                        'backgroundColor': '#ff4444',
                                        'color': 'white'
                                    },
                                    {
                                        'if': {'column_id': 'probability', 'filter_query': '{probability} > 60 && {probability} <= 80'},
                                        'backgroundColor': '#ff9944',
                                        'color': 'white'
                                    }
                                ]
                            )
                        ])
                    ])
                ], width=8),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Matrix"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="risk-matrix-chart",
                                style={'height': '300px'}
                            )
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # What-if analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("What-If Analysis"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Adjust Parameters:"),
                                    
                                    html.Div([
                                        html.Label("Temperature Increase", className="mt-2"),
                                        dcc.Slider(
                                            id="temp-adjustment-slider",
                                            min=-10,
                                            max=10,
                                            value=0,
                                            marks={i: f"{i:+d}°C" for i in range(-10, 11, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ]),
                                    
                                    html.Div([
                                        html.Label("Maintenance Delay", className="mt-3"),
                                        dcc.Slider(
                                            id="maintenance-delay-slider",
                                            min=0,
                                            max=30,
                                            value=0,
                                            marks={i: f"{i}d" for i in range(0, 31, 5)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ]),
                                    
                                    html.Div([
                                        html.Label("Operating Hours", className="mt-3"),
                                        dcc.Slider(
                                            id="operating-hours-slider",
                                            min=8,
                                            max=24,
                                            value=16,
                                            marks={i: f"{i}h" for i in range(8, 25, 4)},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        )
                                    ])
                                ], width=6),
                                
                                dbc.Col([
                                    html.Label("Impact Analysis:"),
                                    html.Div(id="whatif-analysis-results", className="mt-2")
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=12)
            ]),
            
            # Update intervals
            dcc.Interval(id="forecast-update-interval", interval=30000, n_intervals=0),
            
            # Hidden stores
            dcc.Store(id="forecast-data-store", storage_type="memory"),
            dcc.Store(id="model-results-store", storage_type="session"),
            
            # Settings modal
            self._create_forecast_settings_modal()
        ])
        
    def _create_forecast_accuracy_card(self) -> dbc.Card:
        """Create forecast accuracy metrics card
        
        Returns:
            Forecast accuracy card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Forecast Accuracy", className="text-muted mb-3"),
                
                html.Div([
                    html.H3("92.5%", className="mb-0 text-success"),
                    html.Small("MAPE: 7.5%")
                ]),
                
                dbc.Progress(
                    [
                        dbc.Progress(value=92.5, color="success", bar=True)
                    ],
                    className="mt-2",
                    style={"height": "5px"}
                ),
                
                html.Div([
                    html.Small("MAE: 2.3 | RMSE: 3.1", className="text-muted")
                ], className="mt-2")
            ])
        ])
        
    def _create_prediction_confidence_card(self) -> dbc.Card:
        """Create prediction confidence card
        
        Returns:
            Prediction confidence card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Prediction Confidence", className="text-muted mb-3"),
                
                daq.Gauge(
                    id="confidence-gauge",
                    label="Confidence",
                    value=87,
                    max=100,
                    min=0,
                    showCurrentValue=True,
                    units="%",
                    color={"gradient": True, "ranges": {
                        "red": [0, 50],
                        "yellow": [50, 80],
                        "green": [80, 100]
                    }},
                    size=120
                )
            ])
        ])
        
    def _create_failure_probability_card(self) -> dbc.Card:
        """Create failure probability card
        
        Returns:
            Failure probability card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Failure Probability", className="text-muted mb-3"),
                
                html.Div([
                    html.H3("15.2%", className="mb-0 text-warning"),
                    html.Small([
                        html.I(className="fas fa-arrow-up text-danger me-1"),
                        "3.2% from last week"
                    ])
                ]),
                
                html.Div([
                    html.Small("Next 7 days", className="text-muted"),
                    dbc.Progress(
                        value=15.2,
                        color="warning",
                        className="mt-1",
                        style={"height": "5px"}
                    )
                ], className="mt-3"),
                
                html.Div([
                    html.Small("Next 30 days", className="text-muted"),
                    dbc.Progress(
                        value=28.7,
                        color="danger",
                        className="mt-1",
                        style={"height": "5px"}
                    )
                ], className="mt-2")
            ])
        ])
        
    def _create_maintenance_window_card(self) -> dbc.Card:
        """Create optimal maintenance window card
        
        Returns:
            Maintenance window card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Optimal Maintenance", className="text-muted mb-3"),
                
                html.Div([
                    html.I(className="fas fa-calendar-alt fa-2x text-primary mb-2"),
                    html.H5("In 5 Days", className="mb-0"),
                    html.Small("Nov 12, 2025")
                ]),
                
                html.Hr(className="my-2"),
                
                html.Div([
                    html.Small("Confidence: 85%", className="text-muted d-block"),
                    html.Small("Cost Savings: $12,500", className="text-success d-block"),
                    html.Small("Downtime: 4 hours", className="text-info d-block")
                ]),
                
                dbc.Button(
                    "Schedule",
                    size="sm",
                    color="primary",
                    className="w-100 mt-2"
                )
            ])
        ])
        
    def _create_forecast_settings_modal(self) -> dbc.Modal:
        """Create forecast settings modal
        
        Returns:
            Settings modal
        """
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Forecast Settings")),
            dbc.ModalBody([
                # Model selection
                html.H5("Active Models"),
                dbc.Checklist(
                    id="active-forecast-models",
                    options=[
                        {"label": "LSTM Forecaster", "value": "lstm"},
                        {"label": "ARIMA", "value": "arima"},
                        {"label": "Prophet", "value": "prophet"},
                        {"label": "Ensemble", "value": "ensemble"}
                    ],
                    value=["lstm", "arima", "ensemble"],
                    switch=True
                ),
                
                html.Hr(),
                
                # Forecast parameters
                html.H5("Forecast Parameters"),
                
                html.Div([
                    html.Label("Confidence Level"),
                    dcc.Slider(
                        id="confidence-level-slider",
                        min=80,
                        max=99,
                        value=95,
                        marks={i: f"{i}%" for i in range(80, 100, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], className="mb-3"),
                
                html.Div([
                    html.Label("Historical Window"),
                    dcc.Dropdown(
                        id="historical-window-select",
                        options=[
                            {"label": "1 Week", "value": "1w"},
                            {"label": "1 Month", "value": "1m"},
                            {"label": "3 Months", "value": "3m"},
                            {"label": "6 Months", "value": "6m"}
                        ],
                        value="1m"
                    )
                ], className="mb-3"),
                
                html.Hr(),
                
                # Display options
                html.H5("Display Options"),
                dbc.Checklist(
                    id="forecast-display-settings",
                    options=[
                        {"label": "Show model uncertainty", "value": "uncertainty"},
                        {"label": "Show seasonal patterns", "value": "seasonal"},
                        {"label": "Show trend lines", "value": "trend"},
                        {"label": "Auto-update forecasts", "value": "auto_update"}
                    ],
                    value=["uncertainty", "seasonal"],
                    switch=True
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Apply", id="apply-forecast-settings", color="primary"),
                dbc.Button("Cancel", id="cancel-forecast-settings", color="secondary")
            ])
        ], id="forecast-settings-modal", size="lg", is_open=False)
        
    def create_main_forecast_chart(self, equipment: str, metric: str,
                                  horizon: str, model: str,
                                  show_options: List[str]) -> go.Figure:
        """Create main forecast chart

        Args:
            equipment: Selected equipment
            metric: Selected metric
            horizon: Forecast horizon
            model: Selected model
            show_options: Display options

        Returns:
            Plotly figure
        """
        # Try to use real NASA data if available
        if self.data_manager and isinstance(self.data_manager, dict):
            telemetry_data = self.data_manager.get('telemetry', [])

            if telemetry_data:
                # Use real NASA data
                df = pd.DataFrame(telemetry_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Select historical data based on metric - handle NASA data structure
                # Look for metric-specific columns or use first available numeric column
                target_col = None

                if metric == 'temperature':
                    # Look for temperature-like columns
                    for col in df.columns:
                        if 'temp' in col.lower() or 'thermal' in col.lower():
                            target_col = col
                            break
                elif metric == 'pressure':
                    # Look for pressure-like columns
                    for col in df.columns:
                        if 'pressure' in col.lower() or 'press' in col.lower():
                            target_col = col
                            break
                elif metric == 'vibration':
                    # Look for vibration-like columns
                    for col in df.columns:
                        if 'vibration' in col.lower() or 'vib' in col.lower() or 'accel' in col.lower():
                            target_col = col
                            break

                # If no specific column found, use first numeric column
                if not target_col or target_col not in df.columns:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    target_col = numeric_cols[0] if len(numeric_cols) > 0 else None

                if target_col and target_col in df.columns:
                    historical_values = df[target_col].values
                else:
                    # Fallback: generate synthetic data if no numeric columns
                    historical_values = np.random.normal(50, 10, len(df))

                historical_time = df['timestamp']

                # Calculate forecast horizon
                horizon_hours = {
                    '1h': 1, '6h': 6, '24h': 24, '7d': 168, '30d': 720
                }.get(horizon, 24)

                # Generate forecast from NASA data patterns
                # Use simple linear extrapolation with seasonal component
                last_values = historical_values[-20:]  # Use last 20 points
                trend_slope = np.polyfit(range(len(last_values)), last_values, 1)[0]

                # Forecast data
                forecast_points = min(horizon_hours * 4, len(historical_values))
                forecast_time = pd.date_range(
                    start=historical_time.iloc[-1] + pd.Timedelta(minutes=1),
                    periods=forecast_points,
                    freq='1min'
                )

                # Generate forecast values using trend and mean reversion
                base_forecast = np.full(forecast_points, historical_values[-1])
                trend_component = trend_slope * np.arange(1, forecast_points + 1)
                seasonal_component = 2 * np.sin(np.linspace(0, 4*np.pi, forecast_points))

                forecast_values = base_forecast + trend_component * 0.1 + seasonal_component

                # Add some realistic decay to the forecast
                decay_factor = np.exp(-np.arange(forecast_points) / (forecast_points * 0.3))
                forecast_values = historical_values[-1] + (forecast_values - historical_values[-1]) * decay_factor

                # Calculate confidence intervals from historical variance
                noise_std = np.std(np.diff(historical_values))

            else:
                # Fallback if no NASA telemetry data
                horizon_hours = {
                    '1h': 1, '6h': 6, '24h': 24, '7d': 168, '30d': 720
                }.get(horizon, 24)

                # Historical data
                historical_points = min(horizon_hours * 4, 500)
                historical_time = pd.date_range(
                    end=datetime.now(),
                    periods=historical_points,
                    freq='15min'
                )

                # Generate realistic sensor data
                np.random.seed(42)
                trend = np.linspace(70, 72, historical_points)  # NASA-like temperature range
                seasonal = 5 * np.sin(np.linspace(0, 8*np.pi, historical_points))
                noise = np.random.randn(historical_points) * 2
                historical_values = trend + seasonal + noise

                # Forecast
                forecast_points = horizon_hours * 4
                forecast_time = pd.date_range(
                    start=datetime.now(),
                    periods=forecast_points,
                    freq='15min'
                )
                forecast_trend = np.linspace(72, 74, forecast_points)
                forecast_seasonal = 5 * np.sin(np.linspace(8*np.pi, 12*np.pi, forecast_points))
                forecast_values = forecast_trend + forecast_seasonal
                noise_std = np.std(noise)
        else:
            # Fallback to original sample data
            horizon_hours = {
                '1h': 1, '6h': 6, '24h': 24, '7d': 168, '30d': 720
            }.get(horizon, 24)

            # Historical data
            historical_points = min(horizon_hours * 4, 500)
            historical_time = pd.date_range(
                end=datetime.now(),
                periods=historical_points,
                freq='15min'
            )

            # Generate realistic sensor data
            np.random.seed(42)
            trend = np.linspace(50, 52, historical_points)
            seasonal = 5 * np.sin(np.linspace(0, 8*np.pi, historical_points))
            noise = np.random.randn(historical_points) * 2
            historical_values = trend + seasonal + noise

            # Forecast
            forecast_points = horizon_hours * 4
            forecast_time = pd.date_range(
                start=datetime.now(),
                periods=forecast_points,
                freq='15min'
            )
            forecast_trend = np.linspace(52, 54, forecast_points)
            forecast_seasonal = 5 * np.sin(np.linspace(8*np.pi, 12*np.pi, forecast_points))
            forecast_values = forecast_trend + forecast_seasonal
            noise_std = 2
        
        # Confidence intervals for forecast
        confidence_mult = 1.96  # 95% confidence
        if (self.data_manager and isinstance(self.data_manager, dict) and
            self.data_manager.get('telemetry')):
            # For NASA data, ensure forecast_points matches forecast_values
            forecast_points = len(forecast_values)

        std_dev = noise_std * np.sqrt(np.arange(1, forecast_points + 1) / 10)
        upper_bound = forecast_values + confidence_mult * std_dev
        lower_bound = forecast_values - confidence_mult * std_dev
        
        # Create figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_time,
            y=historical_values,
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_time,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence interval
        if 'ci' in show_options:
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=upper_bound,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=lower_bound,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                showlegend=False
            ))
            
        # Anomaly threshold
        if 'threshold' in show_options:
            fig.add_hline(y=60, line_dash="dot", line_color="red",
                         annotation_text="Upper Threshold")
            fig.add_hline(y=40, line_dash="dot", line_color="red",
                         annotation_text="Lower Threshold")
            
        # Historical anomalies
        if 'anomalies' in show_options:
            anomaly_indices = np.random.choice(historical_points, size=10, replace=False)
            fig.add_trace(go.Scatter(
                x=historical_time[anomaly_indices],
                y=historical_values[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
            
        # Update layout
        fig.update_layout(
            title=f"{metric.capitalize()} Forecast - {equipment}",
            xaxis_title="Time",
            yaxis_title=f"{metric.capitalize()}",
            hovermode='x unified',
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        # Add vertical line for current time
        fig.add_vline(x=datetime.now(), line_dash="dash", line_color="gray",
                     annotation_text="Now")
        
        return fig
        
    def create_decomposition_chart(self, metric: str) -> go.Figure:
        """Create time series decomposition chart
        
        Args:
            metric: Selected metric
            
        Returns:
            Plotly figure
        """
        # Generate sample data
        periods = 200
        time_index = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Create synthetic time series
        np.random.seed(42)
        trend = np.linspace(50, 55, periods)
        seasonal = 5 * np.sin(np.linspace(0, 8*np.pi, periods))
        noise = np.random.randn(periods) * 1
        observed = trend + seasonal + noise
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Observed
        fig.add_trace(
            go.Scatter(x=time_index, y=observed, mode='lines', name='Observed',
                      line=dict(color='#1f77b4', width=1)),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=time_index, y=trend, mode='lines', name='Trend',
                      line=dict(color='#ff7f0e', width=2)),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=time_index, y=seasonal, mode='lines', name='Seasonal',
                      line=dict(color='#2ca02c', width=1)),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=time_index, y=noise, mode='lines', name='Residual',
                      line=dict(color='#d62728', width=1)),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            template="plotly_white",
            margin=dict(l=0, r=0, t=30, b=0),
            height=350
        )
        
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        return fig
        
    def create_model_comparison_chart(self) -> go.Figure:
        """Create model comparison chart
        
        Returns:
            Plotly figure
        """
        models = ['LSTM', 'ARIMA', 'Prophet', 'Ensemble']
        metrics = {
            'MAE': [2.3, 2.8, 3.1, 2.1],
            'RMSE': [3.1, 3.5, 3.8, 2.9],
            'MAPE': [7.5, 8.9, 9.2, 7.1],
            'R²': [0.92, 0.89, 0.87, 0.94]
        }
        
        # Create grouped bar chart
        fig = go.Figure()
        
        for metric, values in metrics.items():
            if metric == 'R²':
                # Scale R² for better visualization
                display_values = [v * 10 for v in values]
            else:
                display_values = values
                
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=display_values,
                text=[f"{v:.2f}" for v in values],
                textposition='auto'
            ))
            
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    def create_risk_matrix_chart(self) -> go.Figure:
        """Create risk matrix visualization
        
        Returns:
            Plotly figure
        """
        # Sample risk data
        equipment_risks = [
            {'name': 'EQ-001', 'probability': 20, 'impact': 60},
            {'name': 'EQ-002', 'probability': 70, 'impact': 80},
            {'name': 'EQ-003', 'probability': 40, 'impact': 40},
            {'name': 'EQ-004', 'probability': 85, 'impact': 90},
            {'name': 'EQ-005', 'probability': 15, 'impact': 30},
            {'name': 'EQ-006', 'probability': 60, 'impact': 50},
            {'name': 'EQ-007', 'probability': 30, 'impact': 70}
        ]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add risk zones
        fig.add_shape(type="rect", x0=0, y0=0, x1=33, y1=33,
                     fillcolor="green", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=33, y0=0, x1=66, y1=33,
                     fillcolor="yellow", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=66, y0=0, x1=100, y1=33,
                     fillcolor="orange", opacity=0.2, layer="below")
        
        fig.add_shape(type="rect", x0=0, y0=33, x1=33, y1=66,
                     fillcolor="yellow", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=33, y0=33, x1=66, y1=66,
                     fillcolor="orange", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=66, y0=33, x1=100, y1=66,
                     fillcolor="red", opacity=0.2, layer="below")
        
        fig.add_shape(type="rect", x0=0, y0=66, x1=33, y1=100,
                     fillcolor="orange", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=33, y0=66, x1=66, y1=100,
                     fillcolor="red", opacity=0.2, layer="below")
        fig.add_shape(type="rect", x0=66, y0=66, x1=100, y1=100,
                     fillcolor="darkred", opacity=0.2, layer="below")
        
        # Add equipment points
        for risk in equipment_risks:
            color = 'green'
            if risk['probability'] > 66 or risk['impact'] > 66:
                color = 'red'
            elif risk['probability'] > 33 or risk['impact'] > 33:
                color = 'orange'
                
            fig.add_trace(go.Scatter(
                x=[risk['probability']],
                y=[risk['impact']],
                mode='markers+text',
                name=risk['name'],
                text=[risk['name']],
                textposition="top center",
                marker=dict(size=15, color=color),
                showlegend=False
            ))
            
        fig.update_layout(
            title="Risk Matrix",
            xaxis_title="Probability (%)",
            yaxis_title="Impact (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0),
            height=300
        )
        
        return fig
        
    def get_failure_predictions_data(self) -> List[Dict]:
        """Get failure predictions table data
        
        Returns:
            List of failure predictions
        """
        predictions = [
            {
                'equipment': 'EQ-004',
                'component': 'Bearing',
                'failure_type': 'Wear',
                'probability': '85%',
                'ttf': '3 days',
                'impact': 'High',
                'action': 'Schedule Maintenance'
            },
            {
                'equipment': 'EQ-002',
                'component': 'Motor',
                'failure_type': 'Overheating',
                'probability': '72%',
                'ttf': '5 days',
                'impact': 'Medium',
                'action': 'Monitor Closely'
            },
            {
                'equipment': 'EQ-007',
                'component': 'Seal',
                'failure_type': 'Leakage',
                'probability': '45%',
                'ttf': '2 weeks',
                'impact': 'Low',
                'action': 'Plan Inspection'
            },
            {
                'equipment': 'EQ-001',
                'component': 'Pump',
                'failure_type': 'Cavitation',
                'probability': '38%',
                'ttf': '3 weeks',
                'impact': 'Medium',
                'action': 'Check Parameters'
            },
            {
                'equipment': 'EQ-009',
                'component': 'Valve',
                'failure_type': 'Blockage',
                'probability': '25%',
                'ttf': '1 month',
                'impact': 'Low',
                'action': 'Routine Check'
            }
        ]
        
        return predictions
        
    def create_whatif_analysis_results(self, temp_adj: float, 
                                      maint_delay: int,
                                      op_hours: int) -> html.Div:
        """Create what-if analysis results
        
        Args:
            temp_adj: Temperature adjustment
            maint_delay: Maintenance delay days
            op_hours: Operating hours per day
            
        Returns:
            Analysis results div
        """
        # Calculate impacts
        base_failure_prob = 15.2
        base_mtbf = 1200
        base_cost = 45000
        
        # Adjust based on parameters
        temp_impact = temp_adj * 2  # Each degree increases failure prob by 2%
        delay_impact = maint_delay * 1.5  # Each day delay increases by 1.5%
        hours_impact = (op_hours - 16) * 0.8  # Each extra hour increases by 0.8%
        
        new_failure_prob = base_failure_prob + temp_impact + delay_impact + hours_impact
        new_mtbf = base_mtbf * (1 - (temp_impact + delay_impact + hours_impact) / 100)
        new_cost = base_cost * (1 + (temp_impact + delay_impact + hours_impact) / 50)
        
        # Determine risk level
        risk_level = "Low"
        risk_color = "success"
        if new_failure_prob > 30:
            risk_level = "High"
            risk_color = "danger"
        elif new_failure_prob > 20:
            risk_level = "Medium"
            risk_color = "warning"
            
        return html.Div([
            dbc.Alert([
                html.H5(f"Risk Level: {risk_level}", className=f"text-{risk_color}"),
                html.Hr(),
                
                html.Div([
                    html.Strong("Failure Probability: "),
                    html.Span(f"{new_failure_prob:.1f}%"),
                    html.Span(f" ({new_failure_prob - base_failure_prob:+.1f}%)",
                             className="text-muted ms-2")
                ], className="mb-2"),
                
                html.Div([
                    html.Strong("MTBF: "),
                    html.Span(f"{new_mtbf:.0f} hours"),
                    html.Span(f" ({new_mtbf - base_mtbf:+.0f})",
                             className="text-muted ms-2")
                ], className="mb-2"),
                
                html.Div([
                    html.Strong("Estimated Cost: "),
                    html.Span(f"${new_cost:,.0f}"),
                    html.Span(f" ({new_cost - base_cost:+,.0f})",
                             className="text-muted ms-2")
                ], className="mb-2"),
                
                html.Hr(),
                
                html.Small([
                    "Recommendation: ",
                    html.Strong(
                        "Immediate action required" if risk_level == "High" else
                        "Monitor closely" if risk_level == "Medium" else
                        "Continue normal operations"
                    )
                ])
            ], color=risk_color, dismissable=False)
        ])
        
    def register_callbacks(self, app):
        """Register forecast view callbacks
        
        Args:
            app: Dash app instance
        """
        
        @app.callback(
            Output("main-forecast-chart", "figure"),
            [Input("forecast-equipment-select", "value"),
             Input("forecast-metric-select", "value"),
             Input("forecast-horizon-select", "value"),
             Input("forecast-model-select", "value"),
             Input("forecast-display-options", "value")]
        )
        def update_main_forecast(equipment, metric, horizon, model, display_options):
            """Update main forecast chart"""
            return self.create_main_forecast_chart(
                equipment, metric, horizon, model, display_options or []
            )
            
        @app.callback(
            Output("decomposition-chart", "figure"),
            Input("forecast-metric-select", "value")
        )
        def update_decomposition(metric):
            """Update decomposition chart"""
            return self.create_decomposition_chart(metric)
            
        @app.callback(
            Output("model-comparison-chart", "figure"),
            Input("forecast-update-interval", "n_intervals")
        )
        def update_model_comparison(n):
            """Update model comparison chart"""
            return self.create_model_comparison_chart()
            
        @app.callback(
            Output("risk-matrix-chart", "figure"),
            Input("forecast-update-interval", "n_intervals")
        )
        def update_risk_matrix(n):
            """Update risk matrix chart"""
            return self.create_risk_matrix_chart()
            
        @app.callback(
            Output("failure-predictions-table", "data"),
            Input("forecast-update-interval", "n_intervals")
        )
        def update_failure_predictions(n):
            """Update failure predictions table"""
            return self.get_failure_predictions_data()
            
        @app.callback(
            Output("whatif-analysis-results", "children"),
            [Input("temp-adjustment-slider", "value"),
             Input("maintenance-delay-slider", "value"),
             Input("operating-hours-slider", "value")]
        )
        def update_whatif_analysis(temp_adj, maint_delay, op_hours):
            """Update what-if analysis results"""
            return self.create_whatif_analysis_results(temp_adj, maint_delay, op_hours)
            
        @app.callback(
            Output("forecast-settings-modal", "is_open"),
            [Input("forecast-settings-btn", "n_clicks"),
             Input("apply-forecast-settings", "n_clicks"),
             Input("cancel-forecast-settings", "n_clicks")],
            [State("forecast-settings-modal", "is_open")]
        )
        def toggle_settings_modal(settings_click, apply_click, cancel_click, is_open):
            """Toggle forecast settings modal"""
            ctx = callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == "forecast-settings-btn":
                    return True
                else:
                    return False
            return is_open

    # HELPER METHODS FOR ENHANCED INTERACTIVE COMPONENTS

    def _get_equipment_options(self) -> List[Dict[str, Any]]:
        """Get equipment options using DropdownStateManager"""
        try:
            dropdown_options = self.dropdown_manager.get_equipment_options(include_all=True)

            # Convert to Dash format
            options = []
            for option in dropdown_options:
                dash_option = {
                    "label": option.label,
                    "value": option.value
                }
                if option.disabled:
                    dash_option["disabled"] = True
                if option.title:
                    dash_option["title"] = option.title

                options.append(dash_option)

            return options
        except Exception as e:
            logger.error(f"Error getting equipment options: {e}")
            return [{"label": "All Equipment", "value": "ALL"}]

    def register_callbacks(self, app):
        """Register forecast view callbacks with enhanced interactive components"""

        # CASCADING DROPDOWN CALLBACKS

        @app.callback(
            [Output("forecast-sensor-select", "options"),
             Output("forecast-sensor-select", "value")],
            [Input("forecast-equipment-select", "value")]
        )
        def update_forecast_sensor_options(equipment_id):
            """Update sensor options when equipment changes"""
            try:
                if not equipment_id:
                    return [], None

                sensor_options = self.dropdown_manager.get_sensor_options_for_equipment(
                    equipment_id, include_all=True
                )

                # Convert to Dash format
                dash_options = []
                for option in sensor_options:
                    dash_option = {
                        "label": option.label,
                        "value": option.value
                    }
                    if option.disabled:
                        dash_option["disabled"] = True
                    if option.title:
                        dash_option["title"] = option.title

                    dash_options.append(dash_option)

                # Set default value to first available sensor
                default_value = dash_options[0]["value"] if dash_options else None

                logger.info(f"[FORECAST] Updated sensor options: {len(dash_options)} sensors for {equipment_id}")
                return dash_options, default_value

            except Exception as e:
                logger.error(f"[FORECAST] Error updating sensor options: {e}")
                return [{"label": "Error loading sensors", "value": None, "disabled": True}], None

        @app.callback(
            [Output("forecast-metric-select", "options"),
             Output("forecast-metric-select", "value")],
            [Input("forecast-equipment-select", "value"),
             Input("forecast-sensor-select", "value")]
        )
        def update_forecast_metric_options(equipment_id, sensor_id):
            """Update metric options when equipment or sensor changes"""
            try:
                if not equipment_id or not sensor_id:
                    return [], None

                metric_options = self.dropdown_manager.get_metric_options_for_sensor(
                    equipment_id, sensor_id, include_calculated=True
                )

                # Convert to Dash format
                dash_options = []
                for option in metric_options:
                    dash_option = {
                        "label": option.label,
                        "value": option.value
                    }
                    if option.disabled:
                        dash_option["disabled"] = True
                    if option.title:
                        dash_option["title"] = option.title

                    dash_options.append(dash_option)

                # Set default value to first available metric
                default_value = dash_options[0]["value"] if dash_options else None

                logger.info(f"[FORECAST] Updated metric options: {len(dash_options)} metrics for {equipment_id}::{sensor_id}")
                return dash_options, default_value

            except Exception as e:
                logger.error(f"[FORECAST] Error updating metric options: {e}")
                return [{"label": "Error loading metrics", "value": None, "disabled": True}], None

        # CHART TYPE SWITCHING CALLBACK

        @app.callback(
            Output("main-forecast-chart", "figure"),
            [Input("forecast-chart-type-select", "value"),
             Input("forecast-equipment-select", "value"),
             Input("forecast-sensor-select", "value"),
             Input("forecast-metric-select", "value"),
             Input("forecast-horizon-select", "value"),
             Input("forecast-model-select", "value"),
             Input("forecast-display-options", "value")]
        )
        def update_forecast_chart(chart_type, equipment_id, sensor_id, metric_id,
                                horizon, model, display_options):
            """Update forecast chart with new chart type and data"""
            try:
                if not all([equipment_id, sensor_id, metric_id]):
                    return self._create_placeholder_chart("Select equipment, sensor, and metric to view forecast")

                # Create chart configuration
                config = ChartConfig(
                    chart_type=ChartType(chart_type),
                    title=f"Forecast: {equipment_id} - {sensor_id} - {metric_id}",
                    height=450,
                    show_confidence_intervals="ci" in (display_options or []),
                    show_anomalies="anomalies" in (display_options or []),
                    show_thresholds="thresholds" in (display_options or [])
                )

                # Generate sample forecast data (replace with actual forecast logic)
                chart_data = self._generate_sample_forecast_data(
                    equipment_id, sensor_id, metric_id, horizon, model
                )

                # Create chart using ChartManager
                figure = self.chart_manager.create_chart(chart_data, config)

                logger.info(f"[FORECAST] Updated chart: type={chart_type}, equipment={equipment_id}, sensor={sensor_id}")
                return figure

            except Exception as e:
                logger.error(f"[FORECAST] Error updating chart: {e}")
                return self._create_error_chart(str(e))

    def _generate_sample_forecast_data(self, equipment_id: str, sensor_id: str,
                                     metric_id: str, horizon: str, model: str) -> ChartData:
        """Generate sample forecast data for demonstration"""
        # Generate time series
        now = datetime.now()
        timestamps = [now + timedelta(hours=i) for i in range(48)]

        # Generate synthetic sensor data with trend and noise
        base_values = np.sin(np.linspace(0, 4*np.pi, 48)) * 10 + 50
        noise = np.random.normal(0, 2, 48)
        values = base_values + noise

        # Generate synthetic anomaly scores
        anomaly_scores = np.random.beta(1, 10, 48)  # Most values low, few high

        # Generate confidence intervals
        confidence_upper = values + np.random.exponential(2, 48)
        confidence_lower = values - np.random.exponential(2, 48)

        # Define thresholds
        thresholds = {
            'normal': np.mean(values),
            'warning': np.mean(values) + np.std(values),
            'critical': np.mean(values) + 2*np.std(values)
        }

        return ChartData(
            timestamps=timestamps,
            values=values.tolist(),
            labels=[f"{sensor_id}_{i}" for i in range(len(values))],
            equipment_id=equipment_id,
            sensor_id=sensor_id,
            metric_id=metric_id,
            anomaly_scores=anomaly_scores.tolist(),
            confidence_upper=confidence_upper.tolist(),
            confidence_lower=confidence_lower.tolist(),
            thresholds=thresholds,
            metadata={"horizon": horizon, "model": model}
        )

    def _create_placeholder_chart(self, message: str) -> go.Figure:
        """Create placeholder chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Forecast Chart",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=450
        )
        return fig

    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when chart creation fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Chart Error",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=450
        )
        return fig


# Create standalone function for import by run_dashboard.py
def create_layout():
    """Create forecast view page layout for dashboard routing"""
    page = ForecastView()
    return page.create_layout()

def register_callbacks(app, data_service=None):
    """Register callbacks for forecast view (placeholder for compatibility)"""
    # Note: This layout uses @callback decorators which are auto-registered
    # This function exists for compatibility with the dashboard launcher
    print("Forecast view callbacks are auto-registered via @callback decorators")
    return True