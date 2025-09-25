"""
Enhanced Forecasting Dashboard
Integrates advanced forecasting capabilities with improved uncertainty quantification
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
import logging
import json

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.forecasting.enhanced_forecaster import EnhancedForecaster, EnhancedForecastResult
from src.forecasting.failure_probability import FailureProbabilityEstimator, FailurePrediction
from src.forecasting.scenario_analysis import WhatIfAnalyzer, MaintenanceStrategy
from src.forecasting.risk_matrix import RiskMatrixSystem
from src.dashboard.components.dropdown_manager import dropdown_state_manager
from src.dashboard.components.chart_manager import chart_manager
from config.settings import settings, get_config

logger = logging.getLogger(__name__)


class EnhancedForecastingDashboard:
    """Enhanced forecasting dashboard with advanced uncertainty quantification"""

    def __init__(self, data_manager=None):
        """Initialize enhanced forecasting dashboard"""
        self.data_manager = data_manager
        self.config = get_config()

        # Initialize forecasting components
        self.enhanced_forecaster = None
        self.failure_estimator = None
        self.whatif_analyzer = None
        self.risk_matrix_system = None

        # Dashboard state
        self.current_equipment = "SMAP"
        self.current_component = "power_system"
        self.forecast_horizon = 24
        self.confidence_levels = [0.8, 0.9, 0.95]

        # Cache for forecasting results
        self.forecast_cache = {}
        self.failure_predictions_cache = {}
        self.risk_assessments_cache = {}

    def create_layout(self) -> html.Div:
        """Create the enhanced forecasting dashboard layout"""

        return html.Div([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H2("Enhanced Forecasting & Predictive Analytics",
                           className="text-primary mb-4"),
                    html.P("Advanced time series forecasting with uncertainty quantification, "
                          "failure probability estimation, and risk assessment.",
                          className="text-muted")
                ])
            ], className="mb-4"),

            # Control Panel
            dbc.Card([
                dbc.CardHeader([
                    html.H5("Forecasting Controls", className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Equipment", className="form-label"),
                            dcc.Dropdown(
                                id="enhanced-forecast-equipment-dropdown",
                                options=[
                                    {"label": "SMAP Satellite", "value": "SMAP"},
                                    {"label": "MSL Mars Rover", "value": "MSL"}
                                ],
                                value="SMAP",
                                className="mb-2"
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Component", className="form-label"),
                            dcc.Dropdown(
                                id="enhanced-forecast-component-dropdown",
                                options=[],
                                value="power_system",
                                className="mb-2"
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("Forecast Horizon (hours)", className="form-label"),
                            dcc.Slider(
                                id="enhanced-forecast-horizon-slider",
                                min=6,
                                max=168,
                                step=6,
                                value=24,
                                marks={6: "6h", 24: "24h", 72: "3d", 168: "1w"},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Model Type", className="form-label"),
                            dcc.Dropdown(
                                id="enhanced-forecast-model-dropdown",
                                options=[
                                    {"label": "Transformer (Recommended)", "value": "transformer"},
                                    {"label": "LSTM", "value": "lstm"}
                                ],
                                value="transformer",
                                className="mb-2"
                            )
                        ], width=2)
                    ])
                ])
            ], className="mb-4"),

            # Forecast Results Tabs
            dbc.Tabs([
                # Advanced Forecast Tab
                dbc.Tab(label="Advanced Forecast", tab_id="advanced-forecast", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="enhanced-forecast-main-chart")
                            ], width=8),
                            dbc.Col([
                                # Forecast Statistics Card
                                dbc.Card([
                                    dbc.CardHeader("Forecast Statistics"),
                                    dbc.CardBody([
                                        html.Div(id="forecast-statistics-content")
                                    ])
                                ]),
                                html.Br(),
                                # Uncertainty Analysis Card
                                dbc.Card([
                                    dbc.CardHeader("Uncertainty Analysis"),
                                    dbc.CardBody([
                                        html.Div(id="uncertainty-analysis-content")
                                    ])
                                ])
                            ], width=4)
                        ], className="mb-4"),

                        # Uncertainty Decomposition Charts
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="uncertainty-decomposition-chart")
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="quantile-forecast-chart")
                            ], width=6)
                        ])
                    ], className="mt-3")
                ]),

                # Failure Probability Tab
                dbc.Tab(label="Failure Probability", tab_id="failure-probability", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="failure-probability-chart")
                            ], width=8),
                            dbc.Col([
                                # Failure Risk Card
                                dbc.Card([
                                    dbc.CardHeader("Failure Risk Assessment"),
                                    dbc.CardBody([
                                        html.Div(id="failure-risk-content")
                                    ])
                                ]),
                                html.Br(),
                                # Contributing Factors Card
                                dbc.Card([
                                    dbc.CardHeader("Contributing Factors"),
                                    dbc.CardBody([
                                        html.Div(id="contributing-factors-content")
                                    ])
                                ])
                            ], width=4)
                        ], className="mb-4"),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="failure-timeline-chart")
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="cascade-failure-chart")
                            ], width=6)
                        ])
                    ], className="mt-3")
                ]),

                # What-If Analysis Tab
                dbc.Tab(label="What-If Analysis", tab_id="whatif-analysis", children=[
                    html.Div([
                        # Scenario Controls
                        dbc.Card([
                            dbc.CardHeader("Scenario Configuration"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Maintenance Strategy"),
                                        dcc.Dropdown(
                                            id="scenario-strategy-dropdown",
                                            options=[
                                                {"label": "Reactive", "value": "reactive"},
                                                {"label": "Preventive", "value": "preventive"},
                                                {"label": "Predictive", "value": "predictive"},
                                                {"label": "Hybrid", "value": "hybrid"}
                                            ],
                                            value="predictive"
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Budget Limit ($)"),
                                        dcc.Input(
                                            id="scenario-budget-input",
                                            type="number",
                                            value=50000,
                                            className="form-control"
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        html.Label("Time Horizon (hours)"),
                                        dcc.Input(
                                            id="scenario-horizon-input",
                                            type="number",
                                            value=168,
                                            className="form-control"
                                        )
                                    ], width=3),
                                    dbc.Col([
                                        dbc.Button(
                                            "Run Analysis",
                                            id="run-scenario-button",
                                            color="primary",
                                            className="mt-4"
                                        )
                                    ], width=3)
                                ])
                            ])
                        ], className="mb-4"),

                        # Scenario Results
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="scenario-comparison-chart")
                            ], width=8),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Scenario Results"),
                                    dbc.CardBody([
                                        html.Div(id="scenario-results-content")
                                    ])
                                ])
                            ], width=4)
                        ])
                    ], className="mt-3")
                ]),

                # Risk Matrix Tab
                dbc.Tab(label="Risk Matrix", tab_id="risk-matrix", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="risk-matrix-chart")
                            ], width=8),
                            dbc.Col([
                                # Risk Summary Card
                                dbc.Card([
                                    dbc.CardHeader("Risk Summary"),
                                    dbc.CardBody([
                                        html.Div(id="risk-summary-content")
                                    ])
                                ]),
                                html.Br(),
                                # Top Risks Card
                                dbc.Card([
                                    dbc.CardHeader("Top Risk Components"),
                                    dbc.CardBody([
                                        html.Div(id="top-risks-content")
                                    ])
                                ])
                            ], width=4)
                        ], className="mb-4"),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="risk-heatmap-chart")
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="risk-dashboard-chart")
                            ], width=6)
                        ])
                    ], className="mt-3")
                ])
            ], id="enhanced-forecast-tabs", active_tab="advanced-forecast"),

            # Data Storage Components
            dcc.Store(id="enhanced-forecast-data"),
            dcc.Store(id="failure-predictions-data"),
            dcc.Store(id="scenario-analysis-data"),
            dcc.Store(id="risk-assessments-data"),

            # Auto-refresh interval
            dcc.Interval(
                id="enhanced-forecast-interval",
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ])

    def register_callbacks(self, app):
        """Register all callbacks for enhanced forecasting dashboard"""

        # Equipment dropdown callback
        @app.callback(
            Output("enhanced-forecast-component-dropdown", "options"),
            Input("enhanced-forecast-equipment-dropdown", "value")
        )
        def update_component_dropdown(equipment):
            if equipment == "SMAP":
                return [
                    {"label": "Power System", "value": "power_system"},
                    {"label": "Communication", "value": "communication"},
                    {"label": "Attitude Control", "value": "attitude_control"},
                    {"label": "Thermal Control", "value": "thermal_control"},
                    {"label": "Payload Sensors", "value": "payload_sensors"}
                ]
            elif equipment == "MSL":
                return [
                    {"label": "Mobility Front", "value": "mobility_front"},
                    {"label": "Mobility Rear", "value": "mobility_rear"},
                    {"label": "Power System", "value": "power_system"},
                    {"label": "Environmental", "value": "environmental"},
                    {"label": "Scientific", "value": "scientific"},
                    {"label": "Communication", "value": "communication"},
                    {"label": "Navigation", "value": "navigation"}
                ]
            return []

        # Main forecast update callback
        @app.callback(
            [Output("enhanced-forecast-data", "data"),
             Output("failure-predictions-data", "data"),
             Output("risk-assessments-data", "data")],
            [Input("enhanced-forecast-interval", "n_intervals"),
             Input("enhanced-forecast-equipment-dropdown", "value"),
             Input("enhanced-forecast-component-dropdown", "value"),
             Input("enhanced-forecast-horizon-slider", "value"),
             Input("enhanced-forecast-model-dropdown", "value")]
        )
        def update_forecast_data(n_intervals, equipment, component, horizon, model_type):
            """Update forecast data based on current selections"""

            if not equipment or not component:
                return {}, {}, {}

            try:
                # Generate mock forecast data (in real application, use actual forecaster)
                forecast_data = self._generate_mock_forecast_data(
                    equipment, component, horizon, model_type
                )

                # Generate mock failure predictions
                failure_data = self._generate_mock_failure_predictions(
                    equipment, component
                )

                # Generate mock risk assessments
                risk_data = self._generate_mock_risk_assessments(
                    equipment, component
                )

                return forecast_data, failure_data, risk_data

            except Exception as e:
                logger.error(f"Error updating forecast data: {e}")
                return {}, {}, {}

        # Enhanced forecast chart callback
        @app.callback(
            Output("enhanced-forecast-main-chart", "figure"),
            Input("enhanced-forecast-data", "data")
        )
        def update_enhanced_forecast_chart(forecast_data):
            """Update the main enhanced forecast chart"""

            if not forecast_data:
                return go.Figure()

            fig = go.Figure()

            # Historical data
            if "historical" in forecast_data:
                fig.add_trace(go.Scatter(
                    x=forecast_data["historical"]["timestamps"],
                    y=forecast_data["historical"]["values"],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))

            # Forecast
            if "forecast" in forecast_data:
                fig.add_trace(go.Scatter(
                    x=forecast_data["forecast"]["timestamps"],
                    y=forecast_data["forecast"]["predictions"],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2)
                ))

                # Confidence intervals
                if "confidence_upper" in forecast_data["forecast"]:
                    fig.add_trace(go.Scatter(
                        x=forecast_data["forecast"]["timestamps"] +
                          forecast_data["forecast"]["timestamps"][::-1],
                        y=forecast_data["forecast"]["confidence_upper"] +
                          forecast_data["forecast"]["confidence_lower"][::-1],
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        name='95% Confidence',
                        hoverinfo="skip"
                    ))

            fig.update_layout(
                title="Enhanced Time Series Forecast with Uncertainty",
                xaxis_title="Time",
                yaxis_title="Sensor Value",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )

            return fig

        # Forecast statistics callback
        @app.callback(
            Output("forecast-statistics-content", "children"),
            Input("enhanced-forecast-data", "data")
        )
        def update_forecast_statistics(forecast_data):
            """Update forecast statistics display"""

            if not forecast_data or "forecast" not in forecast_data:
                return html.P("No forecast data available")

            stats = forecast_data.get("statistics", {})

            return html.Div([
                html.P([html.Strong("Forecast Horizon: "), f"{stats.get('horizon', 0)} hours"]),
                html.P([html.Strong("Model Type: "), stats.get('model_type', 'Unknown')]),
                html.P([html.Strong("RMSE: "), f"{stats.get('rmse', 0):.4f}"]),
                html.P([html.Strong("MAE: "), f"{stats.get('mae', 0):.4f}"]),
                html.P([html.Strong("R²: "), f"{stats.get('r2', 0):.4f}"]),
                html.P([html.Strong("Confidence Level: "), f"{stats.get('confidence_level', 0.95)*100:.0f}%"]),
            ])

        # Uncertainty analysis callback
        @app.callback(
            Output("uncertainty-analysis-content", "children"),
            Input("enhanced-forecast-data", "data")
        )
        def update_uncertainty_analysis(forecast_data):
            """Update uncertainty analysis display"""

            if not forecast_data or "uncertainty" not in forecast_data:
                return html.P("No uncertainty data available")

            uncertainty = forecast_data.get("uncertainty", {})

            return html.Div([
                html.P([html.Strong("Epistemic Uncertainty: "),
                       f"{uncertainty.get('epistemic', 0):.4f}"]),
                html.P([html.Strong("Aleatoric Uncertainty: "),
                       f"{uncertainty.get('aleatoric', 0):.4f}"]),
                html.P([html.Strong("Total Uncertainty: "),
                       f"{uncertainty.get('total', 0):.4f}"]),
                html.P([html.Strong("Prediction Std: "),
                       f"{uncertainty.get('prediction_std', 0):.4f}"]),
                html.P([html.Strong("Assessment Confidence: "),
                       f"{uncertainty.get('confidence', 0.8)*100:.0f}%"]),
            ])

        # Additional callbacks for other components would be added here...
        # (failure probability charts, what-if analysis, risk matrix, etc.)

    def _generate_mock_forecast_data(self, equipment, component, horizon, model_type):
        """Generate mock forecast data for demonstration"""

        # Generate historical data
        timestamps_hist = pd.date_range(
            end=datetime.now(),
            periods=100,
            freq='H'
        )

        # Simulate sensor data with trend and noise
        base_value = 50
        trend = np.linspace(0, 5, 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 24)  # Daily cycle
        noise = np.random.normal(0, 2, 100)

        historical_values = base_value + trend + seasonal + noise

        # Generate forecast data
        timestamps_forecast = pd.date_range(
            start=datetime.now(),
            periods=horizon,
            freq='H'
        )

        # Extend trend and seasonal pattern
        forecast_base = base_value + 5
        forecast_trend = np.linspace(0, 2, horizon)
        forecast_seasonal = 5 * np.sin(2 * np.pi * np.arange(horizon) / 24)
        forecast_noise = np.random.normal(0, 1, horizon)

        predictions = forecast_base + forecast_trend + forecast_seasonal + forecast_noise

        # Generate confidence intervals
        std_dev = 3.0
        confidence_lower = predictions - 1.96 * std_dev
        confidence_upper = predictions + 1.96 * std_dev

        return {
            "historical": {
                "timestamps": timestamps_hist.tolist(),
                "values": historical_values.tolist()
            },
            "forecast": {
                "timestamps": timestamps_forecast.tolist(),
                "predictions": predictions.tolist(),
                "confidence_lower": confidence_lower.tolist(),
                "confidence_upper": confidence_upper.tolist()
            },
            "statistics": {
                "horizon": horizon,
                "model_type": model_type,
                "rmse": 2.5,
                "mae": 1.8,
                "r2": 0.85,
                "confidence_level": 0.95
            },
            "uncertainty": {
                "epistemic": 0.15,
                "aleatoric": 0.08,
                "total": 0.17,
                "prediction_std": 2.2,
                "confidence": 0.82
            }
        }

    def _generate_mock_failure_predictions(self, equipment, component):
        """Generate mock failure predictions for demonstration"""

        return {
            "failure_probability": 0.3,
            "time_to_failure": 72.5,
            "severity": "high",
            "contributing_factors": {
                "degradation": 0.4,
                "anomalies": 0.2,
                "operational_stress": 0.1
            },
            "financial_impact": 25000,
            "operational_impact": 0.7,
            "uncertainty": 0.12
        }

    def _generate_mock_risk_assessments(self, equipment, component):
        """Generate mock risk assessments for demonstration"""

        return {
            "overall_risk_score": 0.65,
            "risk_level": "high",
            "probability_score": 0.3,
            "impact_score": 0.8,
            "timeline_score": 0.6,
            "assessment_confidence": 0.85,
            "recommendations": [
                "Schedule immediate inspection",
                "Consider preventive maintenance",
                "Monitor closely for next 48 hours"
            ]
        }


# Utility function to create the dashboard instance
def create_enhanced_forecasting_dashboard(data_manager=None):
    """Create enhanced forecasting dashboard instance"""
    return EnhancedForecastingDashboard(data_manager)


if __name__ == "__main__":
    # Demo - this would normally be integrated with the main dashboard app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    dashboard = create_enhanced_forecasting_dashboard()
    app.layout = dashboard.create_layout()
    dashboard.register_callbacks(app)

    if __name__ == "__main__":
        app.run_server(debug=True, port=8051)