"""
Dashboard Utilities
Helper classes and functions for dashboard functionality with advanced caching
"""

import threading
import queue
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import advanced caching system
from src.utils.advanced_cache import (
    advanced_cache,
    DashboardCacheHelper,
    EquipmentCacheHelper,
    cache_result
)


class DataManager:
    """Manages data operations for the dashboard with advanced caching"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_helper = DashboardCacheHelper()
        self.equipment_cache = EquipmentCacheHelper()

    @cache_result(ttl=120, key_pattern="anomaly_data:{time_range}")
    def get_anomaly_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get anomaly detection data with caching"""
        # Check cache first
        cached_data = self.cache_helper.get_component("anomaly_data", {"time_range": time_range})
        if cached_data:
            return cached_data

        # Simulate data fetching (replace with actual database query)
        data = {
            "total_anomalies": 15,
            "anomalies_last_hour": 3,
            "trend": "increasing",
            "data": [],
            "timestamp": datetime.now().isoformat()
        }

        # Cache the result
        self.cache_helper.cache_component("anomaly_data", data, {"time_range": time_range}, ttl=120)
        return data

    @cache_result(ttl=180, key_pattern="equipment_data")
    def get_equipment_data(self) -> List[Dict]:
        """Get equipment status data with caching"""
        cached_data = self.cache_helper.get_component("equipment_data")
        if cached_data:
            return cached_data

        # Simulate equipment data fetching
        data = [
            {"name": "Pump A", "status": "operational", "health": 85, "last_update": datetime.now().isoformat()},
            {"name": "Motor B", "status": "warning", "health": 65, "last_update": datetime.now().isoformat()},
            {"name": "Sensor C", "status": "operational", "health": 92, "last_update": datetime.now().isoformat()}
        ]

        self.cache_helper.cache_component("equipment_data", data, ttl=180)
        return data

    @cache_result(ttl=60, key_pattern="alerts:{limit}")
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts with caching"""
        cached_data = self.cache_helper.get_component("alerts", {"limit": limit})
        if cached_data:
            return cached_data

        # Simulate alerts fetching
        data = [
            {
                "id": 1,
                "message": "Temperature sensor anomaly detected",
                "severity": "high",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]

        self.cache_helper.cache_component("alerts", data, {"limit": limit}, ttl=60)
        return data

    @cache_result(ttl=300, key_pattern="work_orders:{status}")
    def get_work_orders(self, status: str = "all") -> List[Dict]:
        """Get work orders with caching"""
        cached_data = self.cache_helper.get_component("work_orders", {"status": status})
        if cached_data:
            return cached_data

        # Simulate work orders fetching
        data = [
            {
                "id": "WO-001",
                "description": "Replace temperature sensor",
                "priority": "high",
                "status": "pending",
                "assigned_to": "Tech Team A",
                "last_update": datetime.now().isoformat()
            }
        ]

        self.cache_helper.cache_component("work_orders", data, {"status": status}, ttl=300)
        return data


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections = []

    def add_connection(self, connection):
        """Add new WebSocket connection"""
        self.connections.append(connection)

    def remove_connection(self, connection):
        """Remove WebSocket connection"""
        if connection in self.connections:
            self.connections.remove(connection)

    def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        for conn in self.connections:
            try:
                conn.send(message)
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")


class CacheManager:
    """
    Legacy CacheManager wrapper around AdvancedCacheManager
    Maintains backward compatibility while providing enhanced performance
    """

    def __init__(self, default_timeout: int = 300):
        self.timeout = default_timeout
        self.cache = advanced_cache  # Use the advanced cache backend

    def get(self, key: str) -> Any:
        """Get value from cache using advanced cache backend"""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache using advanced cache backend"""
        self.cache.set(key, value, self.timeout)

    def clear(self) -> None:
        """Clear all cache using advanced cache backend"""
        self.cache.clear_all()

    def get_metrics(self):
        """Get cache performance metrics"""
        return self.cache.get_metrics()


class AlertManager:
    """Manages alert generation and distribution with advanced caching"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_queue = queue.Queue()
        self.cache_helper = DashboardCacheHelper()

    def create_alert(self, message: str, severity: str = "info",
                    equipment_id: str = None) -> Dict[str, Any]:
        """Create new alert with cache invalidation"""
        alert = {
            "id": datetime.now().timestamp(),
            "message": message,
            "severity": severity,
            "equipment_id": equipment_id,
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False
        }
        self.alert_queue.put(alert)

        # Invalidate alert cache to ensure fresh data
        advanced_cache.invalidate_pattern("dashboard:alerts:*")

        # Cache recent alert for quick access
        self.cache_helper.cache_component(f"recent_alert_{alert['id']}", alert, ttl=300)

        return alert

    @cache_result(ttl=30, key_pattern="alerts_queue:{limit}")
    def get_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts from queue with caching"""
        # Try cache first for frequently requested alert lists
        cached_alerts = self.cache_helper.get_component("alert_queue", {"limit": limit})
        if cached_alerts:
            return cached_alerts

        alerts = []
        count = 0
        temp_alerts = []

        # Extract alerts from queue while preserving them
        while not self.alert_queue.empty() and count < limit:
            alert = self.alert_queue.get()
            alerts.append(alert)
            temp_alerts.append(alert)
            count += 1

        # Put alerts back in queue
        for alert in temp_alerts:
            self.alert_queue.put(alert)

        # Cache the result briefly
        self.cache_helper.cache_component("alert_queue", alerts, {"limit": limit}, ttl=30)
        return alerts

    def acknowledge_alert(self, alert_id: float) -> bool:
        """Acknowledge an alert and update cache"""
        # Invalidate cache since alert status changed
        advanced_cache.invalidate_pattern("dashboard:alerts:*")
        advanced_cache.invalidate_pattern(f"dashboard:recent_alert_{alert_id}:*")
        return True

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics with caching"""
        cached_stats = self.cache_helper.get_component("alert_statistics")
        if cached_stats:
            return cached_stats

        # Calculate statistics
        stats = {
            "total_alerts": self.alert_queue.qsize(),
            "severity_breakdown": {"info": 0, "warning": 0, "high": 0, "critical": 0},
            "last_updated": datetime.now().isoformat()
        }

        # Cache for 60 seconds
        self.cache_helper.cache_component("alert_statistics", stats, ttl=60)
        return stats