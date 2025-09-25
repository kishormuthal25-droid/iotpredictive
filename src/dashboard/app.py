"""
Main Dashboard Application for IoT Anomaly Detection System
Core dashboard components and utilities
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from flask import Flask
import logging
from pathlib import Path
import sys
from collections import deque
import queue

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import dashboard utilities
from src.dashboard.utils import DataManager, WebSocketManager
from src.dashboard.components import (
    create_metric_card,
    create_alert_table,
    create_equipment_status,
    create_work_order_list
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data storage for real-time updates
real_time_data = {
    'alerts': deque(maxlen=100),
    'metrics': deque(maxlen=1000),
    'equipment_status': {},
    'work_orders': deque(maxlen=50),
    'system_health': {}
}

# Thread-safe queue for updates
update_queue = queue.Queue()

# Global utility instances
data_manager = DataManager()
ws_manager = WebSocketManager()


# Utility functions for dashboard components
def get_live_data():
    """Get current live data from buffers"""
    return {
        'alerts': list(real_time_data['alerts']),
        'metrics': list(real_time_data['metrics']),
        'equipment_status': real_time_data['equipment_status'],
        'work_orders': list(real_time_data['work_orders']),
        'system_health': real_time_data['system_health']
    }

def update_real_time_data(data_type, data):
    """Update real-time data buffers"""
    if data_type in real_time_data:
        if isinstance(real_time_data[data_type], deque):
            real_time_data[data_type].append(data)
        else:
            real_time_data[data_type].update(data)
