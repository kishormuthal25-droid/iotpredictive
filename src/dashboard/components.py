"""
Dashboard Components
Reusable UI components for the IoT dashboard
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional


def create_metric_card(value: Any, title: str, subtitle: str = "",
                      icon: str = "", color: str = "primary"):
    """Create a metric display card"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas {icon} fa-2x text-{color}") if icon else None,
                html.Div([
                    html.H3(str(value), className="mb-0"),
                    html.P(title, className="text-muted mb-0"),
                    html.Small(subtitle, className="text-secondary") if subtitle else None
                ], className="ms-3")
            ], className="d-flex align-items-center")
        ])
    ], className="h-100")


def create_alert_table(alerts: List[Dict] = None):
    """Create alerts table component"""
    if not alerts:
        alerts = [
            {"id": 1, "message": "No alerts", "severity": "info", "timestamp": "N/A"}
        ]

    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Alert"),
                html.Th("Severity"),
                html.Th("Time")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(alert.get("message", "N/A")),
                html.Td(html.Span(
                    alert.get("severity", "info").upper(),
                    className=f"badge bg-{_get_severity_color(alert.get('severity', 'info'))}"
                )),
                html.Td(alert.get("timestamp", "N/A"))
            ]) for alert in alerts
        ])
    ], striped=True, hover=True)


def create_equipment_status(equipment_data: List[Dict] = None):
    """Create equipment status component"""
    if not equipment_data:
        equipment_data = [
            {"name": "Equipment 1", "status": "operational", "health": 85},
            {"name": "Equipment 2", "status": "warning", "health": 65},
            {"name": "Equipment 3", "status": "operational", "health": 92}
        ]

    return html.Div([
        html.H5("Equipment Status", className="mb-3"),
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H6(eq["name"], className="mb-2"),
                    html.Div([
                        html.Span(
                            eq["status"].upper(),
                            className=f"badge bg-{_get_status_color(eq['status'])}"
                        ),
                        html.Span(f"Health: {eq['health']}%", className="ms-2 text-muted")
                    ])
                ])
            ], className="mb-2") for eq in equipment_data
        ])
    ])


def create_work_order_list(work_orders: List[Dict] = None):
    """Create work order list component"""
    if not work_orders:
        work_orders = [
            {"id": "WO-001", "description": "Sample maintenance task", "priority": "high", "status": "pending"}
        ]

    return html.Div([
        html.H5("Work Orders", className="mb-3"),
        dbc.ListGroup([
            dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"#{wo['id']}"),
                    html.Span(
                        wo['priority'].upper(),
                        className=f"badge bg-{_get_priority_color(wo['priority'])} ms-2"
                    )
                ], className="d-flex justify-content-between"),
                html.P(wo['description'], className="mb-1"),
                html.Small(f"Status: {wo['status']}", className="text-muted")
            ]) for wo in work_orders
        ])
    ])


def _get_severity_color(severity: str) -> str:
    """Get color for alert severity"""
    colors = {
        "critical": "danger",
        "high": "warning",
        "medium": "primary",
        "low": "info",
        "info": "light"
    }
    return colors.get(severity.lower(), "secondary")


def _get_status_color(status: str) -> str:
    """Get color for equipment status"""
    colors = {
        "operational": "success",
        "warning": "warning",
        "error": "danger",
        "maintenance": "info"
    }
    return colors.get(status.lower(), "secondary")


def _get_priority_color(priority: str) -> str:
    """Get color for priority level"""
    colors = {
        "critical": "danger",
        "high": "warning",
        "medium": "primary",
        "low": "info"
    }
    return colors.get(priority.lower(), "secondary")