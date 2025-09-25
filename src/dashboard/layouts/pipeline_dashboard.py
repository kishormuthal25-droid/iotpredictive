"""
Pipeline Dashboard Layout
Real-time pipeline monitoring dashboard for NASA telemetry processing
"""

from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Import the pipeline monitor, heatmap, and model status panel
from src.dashboard.components.pipeline_status_monitor import pipeline_status_monitor
from src.dashboard.components.anomaly_heatmap import anomaly_heatmap_manager
from src.dashboard.components.model_status_panel import model_status_manager

logger = logging.getLogger(__name__)


class PipelineDashboard:
    """Pipeline monitoring dashboard layout manager"""

    def __init__(self):
        """Initialize pipeline dashboard"""
        self.monitor = pipeline_status_monitor

    def create_layout(self) -> html.Div:
        """Create pipeline dashboard layout

        Returns:
            Pipeline dashboard layout
        """
        return html.Div([
            # Real-time status header
            self._create_status_header(),

            # Main metrics row
            dbc.Row([
                dbc.Col([
                    self._create_processing_rate_card()
                ], width=3),

                dbc.Col([
                    self._create_pipeline_health_card()
                ], width=3),

                dbc.Col([
                    self._create_throughput_card()
                ], width=3),

                dbc.Col([
                    self._create_queue_status_card()
                ], width=3)
            ], className="mb-4"),

            # Charts and data sources row
            dbc.Row([
                dbc.Col([
                    self._create_throughput_chart()
                ], width=8),

                dbc.Col([
                    self._create_data_sources_panel()
                ], width=4)
            ], className="mb-4"),

            # Equipment Anomaly Heatmap row
            dbc.Row([
                dbc.Col([
                    self._create_anomaly_heatmap_card()
                ], width=9),

                dbc.Col([
                    self._create_heatmap_summary_card()
                ], width=3)
            ], className="mb-4"),

            # Model Status Panel row
            dbc.Row([
                dbc.Col([
                    self._create_model_status_panel_card()
                ], width=9),

                dbc.Col([
                    self._create_model_summary_card()
                ], width=3)
            ], className="mb-4"),

            # Detailed metrics row
            dbc.Row([
                dbc.Col([
                    self._create_performance_metrics_card()
                ], width=6),

                dbc.Col([
                    self._create_system_status_card()
                ], width=6)
            ])
        ])

    def _create_status_header(self) -> html.Div:
        """Create real-time status header"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H2([
                        "🚀 Real-time NASA Telemetry Pipeline",
                        dbc.Badge(
                            id="pipeline-status-badge",
                            children="LIVE",
                            color="success",
                            className="ms-3",
                            pill=True
                        )
                    ]),
                    html.P(
                        id="pipeline-processing-rate",
                        children="Processing Rate: Connecting...",
                        className="text-muted mb-0"
                    )
                ], width=8),

                dbc.Col([
                    html.Div([
                        dbc.Button(
                            "Refresh",
                            id="pipeline-refresh-btn",
                            color="outline-primary",
                            size="sm",
                            className="me-2"
                        ),
                        dbc.Button(
                            "Reset Metrics",
                            id="pipeline-reset-btn",
                            color="outline-secondary",
                            size="sm"
                        )
                    ], className="text-end")
                ], width=4)
            ])
        ], className="mb-4")

    def _create_processing_rate_card(self) -> dbc.Card:
        """Create processing rate metrics card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("📊 Processing Rate", className="mb-0"),
                dbc.Badge(
                    id="processing-rate-trend",
                    children="↗️ +2.3%",
                    color="success",
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                html.H3(
                    id="processing-rate-value",
                    children="--",
                    className="text-primary mb-2"
                ),
                html.P(
                    id="processing-rate-subtitle",
                    children="records/second",
                    className="text-muted mb-2"
                ),
                dbc.Progress(
                    id="processing-rate-progress",
                    value=0,
                    max=100,
                    striped=True,
                    animated=True,
                    className="mb-2"
                ),
                html.Small(
                    id="processing-rate-details",
                    children="Target: 50+ rec/sec",
                    className="text-muted"
                )
            ])
        ])

    def _create_pipeline_health_card(self) -> dbc.Card:
        """Create pipeline health card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("💚 Pipeline Health", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div([
                    daq.Gauge(
                        id="pipeline-health-gauge",
                        value=95,
                        min=0,
                        max=100,
                        showCurrentValue=True,
                        units="Health Score",
                        size=120,
                        color={
                            "gradient": True,
                            "ranges": {
                                "red": [0, 50],
                                "yellow": [50, 80],
                                "green": [80, 100]
                            }
                        }
                    )
                ], className="text-center"),
                html.P(
                    id="pipeline-health-status",
                    children="EXCELLENT",
                    className="text-center text-success fw-bold mt-2 mb-0"
                )
            ])
        ])

    def _create_throughput_card(self) -> dbc.Card:
        """Create throughput metrics card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("⚡ Throughput", className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H4(
                            id="total-processed",
                            children="--",
                            className="text-info mb-1"
                        ),
                        html.Small("Total Processed", className="text-muted")
                    ], width=6),

                    dbc.Col([
                        html.H4(
                            id="anomalies-found",
                            children="--",
                            className="text-warning mb-1"
                        ),
                        html.Small("Anomalies Found", className="text-muted")
                    ], width=6)
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.P([
                            "Uptime: ",
                            html.Span(
                                id="pipeline-uptime",
                                children="--h",
                                className="fw-bold"
                            )
                        ], className="mb-1"),
                        html.P([
                            "Avg Rate: ",
                            html.Span(
                                id="avg-processing-rate",
                                children="--/sec",
                                className="fw-bold"
                            )
                        ], className="mb-0")
                    ])
                ])
            ])
        ])

    def _create_queue_status_card(self) -> dbc.Card:
        """Create processing queue status card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("📋 Queue Status", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="queue-status-list"),

                html.Hr(),

                html.P([
                    "Total Queued: ",
                    html.Span(
                        id="total-queued",
                        children="--",
                        className="fw-bold"
                    )
                ], className="mb-2"),

                dbc.Badge(
                    id="queue-health-badge",
                    children="GOOD",
                    color="success",
                    className="w-100"
                )
            ])
        ])

    def _create_throughput_chart(self) -> dbc.Card:
        """Create real-time throughput chart"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("📈 Real-time Throughput", className="mb-0"),
                dbc.Badge("Live Updates", color="danger", className="ms-2")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="throughput-chart",
                    config={'displayModeBar': False},
                    style={'height': '300px'}
                )
            ])
        ])

    def _create_data_sources_panel(self) -> dbc.Card:
        """Create data sources status panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🛰️ Data Sources", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="data-sources-status")
            ])
        ])

    def _create_performance_metrics_card(self) -> dbc.Card:
        """Create detailed performance metrics card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🔧 Performance Metrics", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="performance-metrics-details")
            ])
        ])

    def _create_system_status_card(self) -> dbc.Card:
        """Create system status card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("⚙️ System Status", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="system-status-details")
            ])
        ])

    def _create_anomaly_heatmap_card(self) -> dbc.Card:
        """Create equipment anomaly heatmap card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🛰️ Equipment Anomaly Heatmap", className="mb-0"),
                dbc.Badge("Real-time", color="danger", className="ms-2"),
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("🔄 Refresh", id="heatmap-refresh-btn", size="sm", color="outline-primary"),
                        dbc.Button("📊 Details", id="heatmap-details-btn", size="sm", color="outline-info"),
                    ], size="sm", className="ms-auto")
                ], className="d-flex align-items-center")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="equipment-anomaly-heatmap",
                    config={'displayModeBar': True, 'displaylogo': False},
                    style={'height': '500px'}
                )
            ], style={'padding': '10px'})
        ])

    def _create_heatmap_summary_card(self) -> dbc.Card:
        """Create heatmap summary statistics card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("📈 Equipment Summary", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div([
                    # Total equipment
                    html.Div([
                        html.H4(id="total-equipment-count", children="--", className="text-primary mb-1"),
                        html.Small("Total Equipment", className="text-muted")
                    ], className="text-center mb-3"),

                    html.Hr(),

                    # Mission breakdown
                    dbc.Row([
                        dbc.Col([
                            html.H5(id="smap-equipment-count", children="--", className="text-info mb-1"),
                            html.Small("SMAP", className="text-muted")
                        ], className="text-center", width=6),

                        dbc.Col([
                            html.H5(id="msl-equipment-count", children="--", className="text-warning mb-1"),
                            html.Small("MSL", className="text-muted")
                        ], className="text-center", width=6)
                    ], className="mb-3"),

                    html.Hr(),

                    # Status breakdown
                    html.Div([
                        html.P("Status Distribution:", className="fw-bold mb-2"),

                        html.Div([
                            dbc.Badge(id="normal-status-badge", children="Normal: --", color="success", className="mb-1 w-100"),
                            dbc.Badge(id="warning-status-badge", children="Warning: --", color="warning", className="mb-1 w-100"),
                            dbc.Badge(id="critical-status-badge", children="Critical: --", color="danger", className="mb-1 w-100")
                        ])
                    ], className="mb-3"),

                    html.Hr(),

                    # Key metrics
                    html.Div([
                        html.P([
                            "Anomaly Rate: ",
                            html.Span(id="anomaly-rate", children="--", className="fw-bold text-warning")
                        ], className="mb-2"),

                        html.P([
                            "Avg Uptime: ",
                            html.Span(id="avg-uptime", children="--", className="fw-bold text-success")
                        ], className="mb-2"),

                        html.P([
                            "Total Sensors: ",
                            html.Span(id="total-sensors-count", children="--", className="fw-bold text-info")
                        ], className="mb-0")
                    ])
                ])
            ])
        ])

    def _create_model_status_panel_card(self) -> dbc.Card:
        """Create model status panel card showing all 80 models"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("🤖 NASA Model Status - 80 Models Active", className="mb-0"),
                dbc.Badge("Live Updates", color="success", className="ms-2"),
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("SMAP", id="filter-smap-btn", size="sm", color="info", outline=True),
                        dbc.Button("MSL", id="filter-msl-btn", size="sm", color="warning", outline=True),
                        dbc.Button("All", id="filter-all-btn", size="sm", color="primary"),
                    ], size="sm", className="ms-auto")
                ], className="d-flex align-items-center")
            ]),
            dbc.CardBody([
                html.Div([
                    # Model grid display
                    html.Div(id="model-status-grid", className="mb-3"),

                    # Model performance chart
                    dcc.Graph(
                        id="model-performance-chart",
                        config={'displayModeBar': False},
                        style={'height': '200px'}
                    )
                ])
            ], style={'padding': '15px'})
        ])

    def _create_model_summary_card(self) -> dbc.Card:
        """Create model summary statistics card"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5("📊 Model Overview", className="mb-0")
            ]),
            dbc.CardBody([
                html.Div([
                    # Total models
                    html.Div([
                        html.H3(id="total-models-count", children="80", className="text-primary mb-1"),
                        html.Small("NASA Models", className="text-muted")
                    ], className="text-center mb-3"),

                    html.Hr(),

                    # Active/Inactive status
                    dbc.Row([
                        dbc.Col([
                            html.H4(id="active-models-count", children="--", className="text-success mb-1"),
                            html.Small("Active", className="text-muted")
                        ], className="text-center", width=6),

                        dbc.Col([
                            html.H4(id="processing-models-count", children="--", className="text-info mb-1"),
                            html.Small("Processing", className="text-muted")
                        ], className="text-center", width=6)
                    ], className="mb-3"),

                    html.Hr(),

                    # Mission breakdown
                    html.Div([
                        html.P("Mission Distribution:", className="fw-bold mb-2"),

                        dbc.Row([
                            dbc.Col([
                                html.H5(id="smap-models-count", children="25", className="text-info mb-1"),
                                html.Small("SMAP", className="text-muted")
                            ], className="text-center", width=6),

                            dbc.Col([
                                html.H5(id="msl-models-count", children="55", className="text-warning mb-1"),
                                html.Small("MSL", className="text-muted")
                            ], className="text-center", width=6)
                        ])
                    ], className="mb-3"),

                    html.Hr(),

                    # Performance metrics
                    html.Div([
                        html.P([
                            "Avg Health: ",
                            html.Span(id="avg-model-health", children="--", className="fw-bold text-success")
                        ], className="mb-2"),

                        html.P([
                            "Total Inferences: ",
                            html.Span(id="total-model-inferences", children="--", className="fw-bold text-primary")
                        ], className="mb-2"),

                        html.P([
                            "Recent Anomalies: ",
                            html.Span(id="recent-model-anomalies", children="--", className="fw-bold text-warning")
                        ], className="mb-0")
                    ])
                ])
            ])
        ])


# Register callbacks for real-time updates
@callback(
    [
        Output("pipeline-processing-rate", "children"),
        Output("pipeline-status-badge", "children"),
        Output("pipeline-status-badge", "color"),
        Output("processing-rate-value", "children"),
        Output("processing-rate-progress", "value"),
        Output("pipeline-health-gauge", "value"),
        Output("pipeline-health-status", "children"),
        Output("pipeline-health-status", "className"),
        Output("total-processed", "children"),
        Output("anomalies-found", "children"),
        Output("pipeline-uptime", "children"),
        Output("avg-processing-rate", "children")
    ],
    [Input("pipeline-interval", "n_intervals")]
)
def update_pipeline_metrics(n):
    """Update main pipeline metrics display"""
    try:
        # Get current metrics
        metrics = pipeline_status_monitor.get_current_metrics()
        health_status = pipeline_status_monitor.get_pipeline_health_status()
        processing_rate_display = pipeline_status_monitor.get_processing_rate_display()

        # Processing rate
        rate = metrics.processing_rate
        if rate >= 1000:
            rate_display = f"{rate/1000:.1f}K"
        else:
            rate_display = f"{rate:.1f}"

        # Progress bar (scale to 100 rec/sec max)
        progress_value = min((rate / 100) * 100, 100)

        # Health status
        health_score = health_status['health_score']
        health_text = health_status['status']
        health_color = f"text-center text-{health_status['color']} fw-bold mt-2 mb-0"

        # Status badge
        if health_score >= 90:
            badge_text = "LIVE"
            badge_color = "success"
        elif health_score >= 70:
            badge_text = "RUNNING"
            badge_color = "primary"
        else:
            badge_text = "WARNING"
            badge_color = "warning"

        # Format numbers
        total_processed = f"{metrics.total_processed:,}" if metrics.total_processed < 1000000 else f"{metrics.total_processed/1000000:.1f}M"
        anomalies_found = f"{metrics.anomalies_detected:,}"
        uptime_display = f"{metrics.pipeline_uptime:.1f}h"
        avg_rate_display = f"{rate:.1f}/sec"

        return (
            processing_rate_display,
            badge_text,
            badge_color,
            rate_display,
            progress_value,
            health_score,
            health_text,
            health_color,
            total_processed,
            anomalies_found,
            uptime_display,
            avg_rate_display
        )

    except Exception as e:
        logger.error(f"Error updating pipeline metrics: {e}")
        return (
            "Processing Rate: Error loading data",
            "ERROR",
            "danger",
            "--",
            0,
            0,
            "ERROR",
            "text-center text-danger fw-bold mt-2 mb-0",
            "--",
            "--",
            "--h",
            "--/sec"
        )


@callback(
    Output("throughput-chart", "figure"),
    [Input("pipeline-interval", "n_intervals")]
)
def update_throughput_chart(n):
    """Update real-time throughput chart"""
    try:
        throughput_data = pipeline_status_monitor.get_throughput_metrics()

        if not throughput_data['timestamps']:
            # Empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="Waiting for data...",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="gray")
            )
            fig.update_layout(
                title="Real-time Throughput",
                xaxis_title="Time",
                yaxis_title="Records/sec",
                height=280,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            return fig

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Processing rate line
        fig.add_trace(
            go.Scatter(
                x=throughput_data['timestamps'],
                y=throughput_data['processing_rates'],
                mode='lines+markers',
                name='Processing Rate',
                line=dict(color='#007bff', width=2),
                marker=dict(size=4)
            ),
            secondary_y=False
        )

        # Anomaly rate line
        fig.add_trace(
            go.Scatter(
                x=throughput_data['timestamps'],
                y=[rate * 100 for rate in throughput_data['anomaly_rates']],  # Convert to percentage
                mode='lines',
                name='Anomaly Rate %',
                line=dict(color='#dc3545', width=1, dash='dash'),
                opacity=0.7
            ),
            secondary_y=True
        )

        # Update layout
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Processing Rate (records/sec)", secondary_y=False)
        fig.update_yaxes(title_text="Anomaly Rate (%)", secondary_y=True)

        fig.update_layout(
            title="Real-time Processing Throughput",
            height=280,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating throughput chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig


@callback(
    Output("data-sources-status", "children"),
    [Input("pipeline-interval", "n_intervals")]
)
def update_data_sources_status(n):
    """Update data sources status display"""
    try:
        sources_status = pipeline_status_monitor.get_data_sources_status()

        sources_components = []
        for source in sources_status:
            sources_components.append(
                dbc.Row([
                    dbc.Col([
                        html.P([
                            dbc.Badge(
                                "●",
                                color=source['color'],
                                className="me-2"
                            ),
                            html.Strong(source['name'])
                        ], className="mb-1"),
                        html.Small([
                            f"{source['records_per_minute']:.0f} rec/min • ",
                            f"{source['last_data']}"
                        ], className="text-muted")
                    ])
                ], className="mb-2")
            )

        return sources_components

    except Exception as e:
        logger.error(f"Error updating data sources: {e}")
        return html.P("Error loading data sources", className="text-danger")


@callback(
    Output("equipment-anomaly-heatmap", "figure"),
    [Input("pipeline-interval", "n_intervals")]
)
def update_anomaly_heatmap(n):
    """Update equipment anomaly heatmap"""
    try:
        fig = anomaly_heatmap_manager.create_heatmap_figure()
        return fig

    except Exception as e:
        logger.error(f"Error updating anomaly heatmap: {e}")
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading heatmap: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Equipment Anomaly Heatmap - Error",
            height=500
        )
        return fig


@callback(
    [
        Output("total-equipment-count", "children"),
        Output("smap-equipment-count", "children"),
        Output("msl-equipment-count", "children"),
        Output("normal-status-badge", "children"),
        Output("warning-status-badge", "children"),
        Output("critical-status-badge", "children"),
        Output("anomaly-rate", "children"),
        Output("avg-uptime", "children"),
        Output("total-sensors-count", "children")
    ],
    [Input("pipeline-interval", "n_intervals")]
)
def update_heatmap_summary(n):
    """Update heatmap summary statistics"""
    try:
        stats = anomaly_heatmap_manager.get_summary_statistics()

        return (
            str(stats['total_equipment']),
            str(stats['smap_equipment']),
            str(stats['msl_equipment']),
            f"Normal: {stats['normal_count']}",
            f"Warning: {stats['warning_count']}",
            f"Critical: {stats['critical_count']}",
            f"{stats['anomaly_rate']:.1f}%",
            f"{stats['average_uptime']:.1f}%",
            str(stats['total_sensors'])
        )

    except Exception as e:
        logger.error(f"Error updating heatmap summary: {e}")
        return ("--", "--", "--", "Normal: --", "Warning: --", "Critical: --", "--", "--", "--")


@callback(
    Output("model-status-grid", "children"),
    [Input("pipeline-interval", "n_intervals")]
)
def update_model_status_grid(n):
    """Update model status grid display"""
    try:
        grid_data = model_status_manager.get_model_grid_data(rows=10, cols=8)

        if not grid_data['model_ids']:
            return html.P("No model data available", className="text-muted text-center")

        # Create grid of model status cards
        model_cards = []
        for i, model_id in enumerate(grid_data['model_ids'][:80]):  # Show up to 80 models
            status = grid_data['statuses'][i]
            health = grid_data['health_scores'][i]
            processing = grid_data['processing_states'][i]
            hover_text = grid_data['hover_texts'][i]
            color = grid_data['colors'][i]

            # Determine badge color based on status
            if status == "ACTIVE":
                if processing == "PROCESSING":
                    badge_color = "success"
                    badge_text = "🟢"
                else:
                    badge_color = "primary"
                    badge_text = "🔵"
            else:
                badge_color = "secondary"
                badge_text = "⚫"

            model_card = dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Small(model_id, className="fw-bold"),
                            dbc.Badge(
                                badge_text,
                                color=badge_color,
                                className="ms-1",
                                title=f"Health: {health:.1f}%"
                            )
                        ], className="text-center"),
                    ], style={'padding': '5px'})
                ], size="sm", style={'border-color': color, 'border-width': '2px'})
            ], width=1, className="mb-1")

            model_cards.append(model_card)

            # Break into rows of 10
            if (i + 1) % 10 == 0:
                model_cards.append(html.Br())

        return dbc.Row(model_cards)

    except Exception as e:
        logger.error(f"Error updating model status grid: {e}")
        return html.P(f"Error loading model grid: {str(e)}", className="text-danger")


@callback(
    Output("model-performance-chart", "figure"),
    [Input("pipeline-interval", "n_intervals")]
)
def update_model_performance_chart(n):
    """Update model performance chart"""
    try:
        stats = model_status_manager.get_summary_statistics()
        group_stats = stats.get('group_stats', {})

        # Create performance comparison chart
        spacecraft = ['SMAP', 'MSL']
        health_scores = []
        accuracies = []

        for sc in spacecraft:
            if sc in group_stats:
                health_scores.append(group_stats[sc].average_health_score)
                accuracies.append(group_stats[sc].average_accuracy * 100)
            else:
                health_scores.append(0)
                accuracies.append(0)

        fig = go.Figure()

        # Health scores
        fig.add_trace(go.Bar(
            name='Health Score',
            x=spacecraft,
            y=health_scores,
            marker_color='lightblue',
            yaxis='y'
        ))

        # Accuracy scores
        fig.add_trace(go.Bar(
            name='Accuracy (%)',
            x=spacecraft,
            y=accuracies,
            marker_color='lightgreen',
            yaxis='y'
        ))

        fig.update_layout(
            title="Model Performance by Mission",
            xaxis_title="Spacecraft",
            yaxis_title="Score",
            barmode='group',
            height=180,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    except Exception as e:
        logger.error(f"Error updating model performance chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig


@callback(
    [
        Output("active-models-count", "children"),
        Output("processing-models-count", "children"),
        Output("avg-model-health", "children"),
        Output("total-model-inferences", "children"),
        Output("recent-model-anomalies", "children")
    ],
    [Input("pipeline-interval", "n_intervals")]
)
def update_model_summary_stats(n):
    """Update model summary statistics"""
    try:
        stats = model_status_manager.get_summary_statistics()

        active_count = stats.get('active_models', 0)
        processing_count = stats.get('processing_models', 0)
        avg_health = stats.get('average_health', 0)
        total_inferences = stats.get('total_inferences', 0)
        recent_anomalies = stats.get('recent_anomalies', 0)

        # Format numbers
        if total_inferences >= 1000000:
            inferences_display = f"{total_inferences/1000000:.1f}M"
        elif total_inferences >= 1000:
            inferences_display = f"{total_inferences/1000:.1f}K"
        else:
            inferences_display = str(total_inferences)

        return (
            str(active_count),
            str(processing_count),
            f"{avg_health:.1f}%",
            inferences_display,
            str(recent_anomalies)
        )

    except Exception as e:
        logger.error(f"Error updating model summary stats: {e}")
        return ("--", "--", "--", "--", "--")


# Global instance
pipeline_dashboard = PipelineDashboard()