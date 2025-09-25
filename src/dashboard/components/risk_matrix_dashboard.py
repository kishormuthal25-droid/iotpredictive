"""
Risk Matrix Dashboard Component
Interactive risk assessment and visualization for IoT equipment portfolio
"""

from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
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

from src.forecasting.risk_matrix import RiskMatrixSystem, RiskMatrixCalculator, RiskMatrixVisualizer, RiskLevel
from src.forecasting.failure_probability import FailurePrediction, SeverityLevel

logger = logging.getLogger(__name__)


class RiskMatrixDashboardComponent:
    """Interactive risk matrix dashboard component"""

    def __init__(self, data_manager=None):
        """Initialize risk matrix dashboard component"""
        self.data_manager = data_manager
        self.risk_matrix_system = RiskMatrixSystem()

        # Cache for risk data
        self.risk_cache = {}
        self.last_update = datetime.now()

    def create_risk_overview_card(self) -> dbc.Card:
        """Create risk overview summary card"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Risk Overview", className="mb-0"),
                dbc.Badge("Live", color="success", className="ms-2")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4("0", id="total-components-count", className="text-primary"),
                        html.Small("Total Components", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H4("0", id="critical-risk-count", className="text-danger"),
                        html.Small("Critical Risk", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H4("0", id="high-risk-count", className="text-warning"),
                        html.Small("High Risk", className="text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H4("$0", id="total-risk-exposure", className="text-info"),
                        html.Small("Risk Exposure", className="text-muted")
                    ], width=3)
                ])
            ])
        ], className="mb-4")

    def create_risk_matrix_card(self) -> dbc.Card:
        """Create risk matrix visualization card"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Risk Matrix", className="mb-0"),
                dbc.ButtonGroup([
                    dbc.Button("Refresh", id="risk-matrix-refresh-btn",
                              size="sm", outline=True, color="primary"),
                    dbc.Button("Export", id="risk-matrix-export-btn",
                              size="sm", outline=True, color="secondary")
                ], className="ms-auto")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="risk-matrix-plot",
                    style={"height": "500px"},
                    config={"displayModeBar": True, "displaylogo": False}
                )
            ])
        ])

    def create_risk_heatmap_card(self) -> dbc.Card:
        """Create risk heatmap visualization card"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Equipment Risk Heatmap", className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="risk-heatmap-plot",
                    style={"height": "400px"},
                    config={"displayModeBar": True, "displaylogo": False}
                )
            ])
        ])

    def create_top_risks_table(self) -> dbc.Card:
        """Create top risks table card"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Top Risk Components", className="mb-0"),
                dbc.Badge("Top 10", color="info", className="ms-2")
            ]),
            dbc.CardBody([
                dash_table.DataTable(
                    id="top-risks-table",
                    columns=[
                        {"name": "Equipment", "id": "equipment_id"},
                        {"name": "Component", "id": "component_id"},
                        {"name": "Risk Level", "id": "risk_level", "type": "text"},
                        {"name": "Risk Score", "id": "overall_risk_score", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Probability", "id": "probability_score", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Impact", "id": "impact_score", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Financial Impact", "id": "financial_impact", "type": "numeric", "format": {"specifier": ",.0f"}}
                    ],
                    data=[],
                    style_cell={'textAlign': 'left', 'fontSize': '12px'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{risk_level} = critical'},
                            'backgroundColor': '#ffebee',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{risk_level} = high'},
                            'backgroundColor': '#fff3e0',
                            'color': 'black',
                        }
                    ],
                    sort_action="native",
                    page_size=10,
                    style_table={'height': '400px', 'overflowY': 'auto'}
                )
            ])
        ])

    def create_risk_trends_card(self) -> dbc.Card:
        """Create risk trends visualization card"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Risk Trends", className="mb-0"),
                dcc.Dropdown(
                    id="risk-trends-timeframe",
                    options=[
                        {"label": "Last 24 Hours", "value": "24h"},
                        {"label": "Last 7 Days", "value": "7d"},
                        {"label": "Last 30 Days", "value": "30d"}
                    ],
                    value="7d",
                    className="ms-auto",
                    style={"width": "150px", "fontSize": "12px"}
                )
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="risk-trends-plot",
                    style={"height": "300px"},
                    config={"displayModeBar": False}
                )
            ])
        ])

    def create_risk_mitigation_panel(self) -> dbc.Card:
        """Create risk mitigation recommendations panel"""

        return dbc.Card([
            dbc.CardHeader([
                html.H5("Risk Mitigation Recommendations", className="mb-0"),
                dbc.Badge("AI Generated", color="info", className="ms-2")
            ]),
            dbc.CardBody([
                html.Div(id="risk-mitigation-content", children=[
                    dbc.Alert("Loading recommendations...", color="info")
                ])
            ])
        ])

    def create_full_layout(self) -> html.Div:
        """Create complete risk matrix dashboard layout"""

        return html.Div([
            # Risk Overview
            self.create_risk_overview_card(),

            # Main Risk Visualizations
            dbc.Row([
                dbc.Col([
                    self.create_risk_matrix_card()
                ], width=8),
                dbc.Col([
                    self.create_top_risks_table()
                ], width=4)
            ], className="mb-4"),

            # Secondary Risk Visualizations
            dbc.Row([
                dbc.Col([
                    self.create_risk_heatmap_card()
                ], width=6),
                dbc.Col([
                    self.create_risk_trends_card()
                ], width=6)
            ], className="mb-4"),

            # Risk Mitigation Panel
            self.create_risk_mitigation_panel(),

            # Data storage and refresh
            dcc.Store(id="risk-matrix-data"),
            dcc.Interval(
                id="risk-matrix-interval",
                interval=60*1000,  # 1 minute
                n_intervals=0
            )
        ])

    def register_callbacks(self, app):
        """Register callbacks for risk matrix dashboard"""

        @app.callback(
            Output("risk-matrix-data", "data"),
            [Input("risk-matrix-interval", "n_intervals"),
             Input("risk-matrix-refresh-btn", "n_clicks")]
        )
        def update_risk_data(n_intervals, refresh_clicks):
            """Update risk assessment data"""

            try:
                # In real application, get actual failure predictions
                mock_predictions = self._generate_mock_failure_predictions()

                # Update risk assessments
                risk_assessments = self.risk_matrix_system.update_risk_assessments(
                    mock_predictions,
                    operational_context={
                        'mission_critical': True,
                        'load_factor': 0.8,
                        'downtime_cost_per_hour': 1000
                    }
                )

                # Get risk summary
                risk_summary = self.risk_matrix_system.get_risk_summary()

                return {
                    'assessments': [assessment.to_dict() for assessment in risk_assessments],
                    'summary': risk_summary,
                    'last_updated': datetime.now().isoformat()
                }

            except Exception as e:
                logger.error(f"Error updating risk data: {e}")
                return {}

        @app.callback(
            [Output("total-components-count", "children"),
             Output("critical-risk-count", "children"),
             Output("high-risk-count", "children"),
             Output("total-risk-exposure", "children")],
            Input("risk-matrix-data", "data")
        )
        def update_risk_overview(risk_data):
            """Update risk overview metrics"""

            if not risk_data or 'summary' not in risk_data:
                return "0", "0", "0", "$0"

            summary = risk_data['summary']

            total_components = summary.get('total_components', 0)
            critical_count = summary.get('risk_level_distribution', {}).get('critical', 0)
            high_count = summary.get('risk_level_distribution', {}).get('high', 0)
            total_exposure = summary.get('total_financial_impact', 0)

            return (
                str(total_components),
                str(critical_count),
                str(high_count),
                f"${total_exposure:,.0f}"
            )

        @app.callback(
            Output("risk-matrix-plot", "figure"),
            Input("risk-matrix-data", "data")
        )
        def update_risk_matrix_plot(risk_data):
            """Update risk matrix visualization"""

            if not risk_data or 'assessments' not in risk_data:
                return go.Figure()

            assessments_data = risk_data['assessments']
            df = pd.DataFrame(assessments_data)

            if df.empty:
                return go.Figure()

            # Create risk matrix plot
            fig = go.Figure()

            # Color mapping for risk levels
            color_map = {
                'very_low': '#2E8B57',
                'low': '#32CD32',
                'medium': '#FFD700',
                'high': '#FF8C00',
                'very_high': '#FF4500',
                'critical': '#DC143C'
            }

            # Add risk level background regions
            self._add_risk_background_regions(fig)

            # Add scatter points for each component
            for risk_level in df['risk_level'].unique():
                level_data = df[df['risk_level'] == risk_level]

                fig.add_trace(go.Scatter(
                    x=level_data['probability_score'],
                    y=level_data['impact_score'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color_map.get(risk_level, '#888888'),
                        line=dict(width=2, color='white'),
                        symbol='circle'
                    ),
                    text=level_data['component_id'],
                    textposition='top center',
                    name=risk_level.replace('_', ' ').title(),
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        'Equipment: %{customdata[0]}<br>' +
                        'Probability: %{x:.3f}<br>' +
                        'Impact: %{y:.3f}<br>' +
                        'Risk Score: %{customdata[1]:.3f}<br>' +
                        'Financial Impact: $%{customdata[2]:,.0f}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=level_data[['equipment_id', 'overall_risk_score', 'financial_impact']].values
                ))

            fig.update_layout(
                title="Equipment Risk Matrix",
                xaxis_title="Failure Probability",
                yaxis_title="Impact Score",
                xaxis=dict(range=[0, 1], showgrid=True, gridcolor='lightgray'),
                yaxis=dict(range=[0, 1], showgrid=True, gridcolor='lightgray'),
                template='plotly_white',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            return fig

        @app.callback(
            Output("risk-heatmap-plot", "figure"),
            Input("risk-matrix-data", "data")
        )
        def update_risk_heatmap(risk_data):
            """Update risk heatmap visualization"""

            if not risk_data or 'assessments' not in risk_data:
                return go.Figure()

            assessments_data = risk_data['assessments']
            df = pd.DataFrame(assessments_data)

            if df.empty:
                return go.Figure()

            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values='overall_risk_score',
                index='component_id',
                columns='equipment_id',
                fill_value=0
            )

            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Risk Score"),
                hovetemplate=(
                    'Equipment: %{x}<br>' +
                    'Component: %{y}<br>' +
                    'Risk Score: %{z:.3f}<br>' +
                    '<extra></extra>'
                )
            ))

            fig.update_layout(
                title="Risk Heatmap by Equipment and Component",
                xaxis_title="Equipment",
                yaxis_title="Component",
                template='plotly_white',
                height=400
            )

            return fig

        @app.callback(
            Output("top-risks-table", "data"),
            Input("risk-matrix-data", "data")
        )
        def update_top_risks_table(risk_data):
            """Update top risks table"""

            if not risk_data or 'assessments' not in risk_data:
                return []

            assessments_data = risk_data['assessments']
            df = pd.DataFrame(assessments_data)

            if df.empty:
                return []

            # Sort by risk score and take top 10
            top_risks = df.nlargest(10, 'overall_risk_score')

            return top_risks.to_dict('records')

        @app.callback(
            Output("risk-trends-plot", "figure"),
            [Input("risk-matrix-data", "data"),
             Input("risk-trends-timeframe", "value")]
        )
        def update_risk_trends(risk_data, timeframe):
            """Update risk trends visualization"""

            # Generate mock trend data for demonstration
            if timeframe == "24h":
                periods = 24
                freq = 'H'
            elif timeframe == "7d":
                periods = 7
                freq = 'D'
            else:  # 30d
                periods = 30
                freq = 'D'

            timestamps = pd.date_range(
                end=datetime.now(),
                periods=periods,
                freq=freq
            )

            # Mock trend data
            avg_risk = 0.4 + 0.1 * np.sin(np.linspace(0, 2*np.pi, periods)) + \
                      0.05 * np.random.randn(periods)
            critical_count = np.random.poisson(2, periods)
            high_count = np.random.poisson(5, periods)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=avg_risk,
                mode='lines+markers',
                name='Average Risk Score',
                line=dict(color='blue', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=timestamps,
                y=critical_count,
                mode='lines+markers',
                name='Critical Components',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))

            fig.update_layout(
                title=f"Risk Trends - {timeframe.upper()}",
                xaxis_title="Time",
                yaxis=dict(title="Average Risk Score", side='left'),
                yaxis2=dict(title="Critical Components", side='right', overlaying='y'),
                template='plotly_white',
                height=300,
                showlegend=True
            )

            return fig

        @app.callback(
            Output("risk-mitigation-content", "children"),
            Input("risk-matrix-data", "data")
        )
        def update_risk_mitigation(risk_data):
            """Update risk mitigation recommendations"""

            if not risk_data or 'assessments' not in risk_data:
                return dbc.Alert("No risk data available", color="warning")

            # Generate recommendations based on risk data
            recommendations = self._generate_risk_recommendations(risk_data)

            return html.Div([
                dbc.Alert(
                    [html.I(className="fas fa-exclamation-triangle me-2"), rec],
                    color="warning" if "urgent" in rec.lower() else "info",
                    className="mb-2"
                )
                for rec in recommendations
            ])

    def _add_risk_background_regions(self, fig):
        """Add colored background regions for risk levels"""

        # Define risk regions with colors
        regions = [
            # Low risk (green)
            {'x': [0, 0.3, 0.3, 0], 'y': [0, 0, 0.3, 0.3], 'color': 'rgba(46, 139, 87, 0.1)'},
            {'x': [0, 0.2, 0.2, 0], 'y': [0.3, 0.3, 0.5, 0.5], 'color': 'rgba(46, 139, 87, 0.1)'},

            # Medium risk (yellow)
            {'x': [0.3, 0.6, 0.6, 0.3], 'y': [0, 0, 0.3, 0.3], 'color': 'rgba(255, 215, 0, 0.1)'},
            {'x': [0.2, 0.5, 0.5, 0.2], 'y': [0.3, 0.3, 0.6, 0.6], 'color': 'rgba(255, 215, 0, 0.1)'},

            # High risk (orange/red)
            {'x': [0.6, 1.0, 1.0, 0.6], 'y': [0, 0, 0.4, 0.4], 'color': 'rgba(255, 140, 0, 0.1)'},
            {'x': [0.5, 1.0, 1.0, 0.5], 'y': [0.4, 0.4, 1.0, 1.0], 'color': 'rgba(220, 20, 60, 0.1)'},
        ]

        for region in regions:
            fig.add_shape(
                type="path",
                path=f"M {region['x'][0]},{region['y'][0]} " +
                     f"L {region['x'][1]},{region['y'][1]} " +
                     f"L {region['x'][2]},{region['y'][2]} " +
                     f"L {region['x'][3]},{region['y'][3]} Z",
                fillcolor=region['color'],
                line=dict(width=0),
                layer="below"
            )

    def _generate_mock_failure_predictions(self) -> List[FailurePrediction]:
        """Generate mock failure predictions for testing"""

        return [
            FailurePrediction(
                equipment_id='SMAP',
                component_id='power_system',
                failure_probability=0.7,
                time_to_failure=48,
                severity=SeverityLevel.CRITICAL,
                contributing_factors={'degradation': 0.5, 'anomalies': 0.3}
            ),
            FailurePrediction(
                equipment_id='SMAP',
                component_id='communication',
                failure_probability=0.3,
                time_to_failure=168,
                severity=SeverityLevel.MEDIUM,
                contributing_factors={'forecast_trends': 0.2}
            ),
            FailurePrediction(
                equipment_id='MSL',
                component_id='mobility_front',
                failure_probability=0.9,
                time_to_failure=24,
                severity=SeverityLevel.CRITICAL,
                contributing_factors={'degradation': 0.7, 'operational_stress': 0.4}
            ),
            FailurePrediction(
                equipment_id='MSL',
                component_id='environmental',
                failure_probability=0.1,
                time_to_failure=720,
                severity=SeverityLevel.LOW,
                contributing_factors={'forecast_trends': 0.1}
            ),
            FailurePrediction(
                equipment_id='SMAP',
                component_id='attitude_control',
                failure_probability=0.5,
                time_to_failure=120,
                severity=SeverityLevel.HIGH,
                contributing_factors={'degradation': 0.3, 'anomalies': 0.2}
            )
        ]

    def _generate_risk_recommendations(self, risk_data) -> List[str]:
        """Generate risk mitigation recommendations"""

        assessments_data = risk_data['assessments']
        df = pd.DataFrame(assessments_data)

        recommendations = []

        # Critical risk recommendations
        critical_risks = df[df['risk_level'] == 'critical']
        if not critical_risks.empty:
            recommendations.append(
                f"URGENT: {len(critical_risks)} components at critical risk require immediate attention"
            )

        # High financial impact recommendations
        high_financial = df[df['financial_impact'] > 50000]
        if not high_financial.empty:
            recommendations.append(
                f"Consider priority maintenance for {len(high_financial)} high-cost components"
            )

        # Timeline recommendations
        urgent_timeline = df[df['timeline_score'] > 0.8]
        if not urgent_timeline.empty:
            recommendations.append(
                f"Schedule immediate inspections for {len(urgent_timeline)} time-critical components"
            )

        # Default recommendation
        if not recommendations:
            recommendations.append("System risk levels are within acceptable ranges")

        return recommendations


# Factory function
def create_risk_matrix_dashboard_component(data_manager=None):
    """Create risk matrix dashboard component instance"""
    return RiskMatrixDashboardComponent(data_manager)