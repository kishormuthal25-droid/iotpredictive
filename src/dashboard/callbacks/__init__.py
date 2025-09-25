"""
Dashboard Callbacks Module
"""

from .dashboard_callbacks import DashboardCallbacks

def register_callbacks(app):
    """Register all dashboard callbacks"""
    return DashboardCallbacks(app)