"""
Dashboard Layout Components
Main layout functions for the IoT dashboard
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
# Import the standalone layout functions
from .overview import create_layout as create_overview_layout
from .anomaly_monitor import create_layout as create_anomaly_layout
from .forecast_view import create_layout as create_forecast_layout
from .maintenance_scheduler import create_layout as create_maintenance_layout
from .work_orders import create_layout as create_work_orders_layout


def create_main_layout():
    """Create the main dashboard layout"""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        create_header(),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_sidebar()
                ], width=2),
                dbc.Col([
                    html.Div(id='page-content')
                ], width=10)
            ])
        ], fluid=True),
        create_footer()
    ])


def create_header():
    """Create dashboard header"""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Img(src="/assets/logo.png", height="40px"),
                    dbc.NavbarBrand("IoT Predictive Maintenance", className="ms-2")
                ]),
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact")),
                        dbc.NavItem(dbc.NavLink("Anomalies", href="/anomalies", active="exact")),
                        dbc.NavItem(dbc.NavLink("Forecasts", href="/forecasts", active="exact")),
                        dbc.NavItem(dbc.NavLink("Maintenance", href="/maintenance", active="exact")),
                        dbc.NavItem(dbc.NavLink("Work Orders", href="/work-orders", active="exact")),
                    ], navbar=True)
                ])
            ], className="w-100")
        ], fluid=True),
        color="primary",
        dark=True,
        className="mb-3"
    )


def create_sidebar():
    """Create dashboard sidebar"""
    return html.Div([
        html.H4("Navigation", className="text-center mb-3"),
        dbc.Nav([
            dbc.NavLink([
                html.I(className="fas fa-tachometer-alt me-2"),
                "Overview"
            ], href="/", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Anomaly Monitor"
            ], href="/anomalies", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Forecasting"
            ], href="/forecasts", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-wrench me-2"),
                "Maintenance"
            ], href="/maintenance", active="exact", className="mb-2"),
            dbc.NavLink([
                html.I(className="fas fa-clipboard-list me-2"),
                "Work Orders"
            ], href="/work-orders", active="exact", className="mb-2"),
        ], vertical=True, pills=True)
    ], className="bg-light p-3", style={"height": "100vh"})


def create_footer():
    """Create dashboard footer"""
    return dbc.Container([
        html.Hr(),
        html.P([
            "IoT Predictive Maintenance System � 2024 | ",
            html.A("Documentation", href="#", className="text-decoration-none"),
            " | ",
            html.A("Support", href="#", className="text-decoration-none")
        ], className="text-center text-muted")
    ], fluid=True)


# Layout functions are now imported directly from individual modules