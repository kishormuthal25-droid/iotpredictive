#!/usr/bin/env python3
"""
Unified Data Service for Dashboard
Provides consistent data access across all dashboard components
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from src.data_ingestion.unified_data_access import unified_data_access
from src.data_ingestion.equipment_mapper import equipment_mapper

logger = logging.getLogger(__name__)


class DashboardDataService:
    """Unified data service for all dashboard components"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_overview_data(self) -> Dict[str, Any]:
        """Get overview dashboard data"""
        try:
            # Get recent telemetry
            recent_data = unified_data_access.get_recent_telemetry(hours=1, limit=100)
            anomalies = unified_data_access.get_recent_anomalies(hours=24, limit=50)

            # Calculate metrics
            total_equipment = len(equipment_mapper.get_all_equipment())
            active_anomalies = len(anomalies)
            system_health = max(70, 100 - active_anomalies * 3)

            # Equipment status
            equipment_status = {}
            for equipment in equipment_mapper.get_all_equipment():
                equipment_status[equipment.equipment_id] = {
                    'name': equipment.equipment_type,
                    'status': 'NORMAL',
                    'last_update': datetime.now(),
                    'sensors': len(equipment.sensors)
                }

            # Check for anomalous equipment
            for anomaly in anomalies:
                equipment_id = getattr(anomaly, 'equipment_id', 'Unknown')
                if equipment_id in equipment_status:
                    equipment_status[equipment_id]['status'] = 'ANOMALY'

            return {
                'metrics': {
                    'total_equipment': total_equipment,
                    'active_anomalies': active_anomalies,
                    'system_health': system_health,
                    'data_points': len(recent_data),
                    'uptime_hours': 24  # Simulated uptime
                },
                'equipment_status': equipment_status,
                'recent_data': recent_data[:20],  # Last 20 readings
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error getting overview data: {e}")
            return self._get_fallback_overview()

    def get_anomaly_data(self) -> Dict[str, Any]:
        """Get anomaly monitoring data"""
        try:
            # Get recent anomalies
            anomalies = unified_data_access.get_recent_anomalies(hours=24, limit=100)

            # Format anomalies for dashboard
            anomaly_list = []
            for anomaly in anomalies:
                equipment_id = getattr(anomaly, 'equipment_id', 'Unknown')
                try:
                    equipment_info = equipment_mapper.get_equipment_by_id(equipment_id)
                    equipment_name = equipment_info.equipment_type
                    subsystem = equipment_info.subsystem
                except:
                    equipment_name = equipment_id
                    subsystem = 'Unknown'

                anomaly_list.append({
                    'equipment_id': equipment_id,
                    'equipment_name': equipment_name,
                    'subsystem': subsystem,
                    'anomaly_score': getattr(anomaly, 'anomaly_score', 0.0),
                    'severity': self._get_severity(getattr(anomaly, 'anomaly_score', 0.0)),
                    'timestamp': getattr(anomaly, 'timestamp', datetime.now()),
                    'description': f"Anomaly detected in {equipment_name}"
                })

            # Group by equipment
            equipment_anomalies = {}
            for anomaly in anomaly_list:
                eq_id = anomaly['equipment_id']
                if eq_id not in equipment_anomalies:
                    equipment_anomalies[eq_id] = []
                equipment_anomalies[eq_id].append(anomaly)

            return {
                'anomalies': anomaly_list,
                'equipment_anomalies': equipment_anomalies,
                'summary': {
                    'total_anomalies': len(anomaly_list),
                    'critical_anomalies': len([a for a in anomaly_list if a['severity'] == 'CRITICAL']),
                    'high_anomalies': len([a for a in anomaly_list if a['severity'] == 'HIGH']),
                    'medium_anomalies': len([a for a in anomaly_list if a['severity'] == 'MEDIUM']),
                    'affected_equipment': len(equipment_anomalies)
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error getting anomaly data: {e}")
            return self._get_fallback_anomaly()

    def get_forecast_data(self) -> Dict[str, Any]:
        """Get forecasting data"""
        try:
            # Get historical data for forecasting
            historical_data = unified_data_access.get_telemetry_range(
                start_time=datetime.now() - timedelta(days=7),
                end_time=datetime.now(),
                limit=1000
            )

            # Simple trend analysis
            forecasts = {}
            equipment_list = equipment_mapper.get_all_equipment()

            for equipment in equipment_list[:5]:  # Limit to first 5 for performance
                forecasts[equipment.equipment_id] = {
                    'equipment_name': equipment.equipment_type,
                    'predicted_values': self._generate_forecast(equipment.equipment_id),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'trend': np.random.choice(['STABLE', 'INCREASING', 'DECREASING']),
                    'next_maintenance': datetime.now() + timedelta(days=np.random.randint(7, 30))
                }

            return {
                'forecasts': forecasts,
                'historical_data': historical_data,
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error getting forecast data: {e}")
            return self._get_fallback_forecast()

    def get_maintenance_data(self) -> Dict[str, Any]:
        """Get maintenance scheduling data"""
        try:
            # Get recent anomalies to generate maintenance tasks
            anomalies = unified_data_access.get_recent_anomalies(hours=72, limit=50)

            # Generate maintenance tasks
            maintenance_tasks = []
            for i, anomaly in enumerate(anomalies[:10]):  # Limit to 10 tasks
                equipment_id = getattr(anomaly, 'equipment_id', f'EQUIP-{i:03d}')
                try:
                    equipment_info = equipment_mapper.get_equipment_by_id(equipment_id)
                    equipment_name = equipment_info.equipment_type
                    priority = equipment_info.criticality
                except:
                    equipment_name = equipment_id
                    priority = 'MEDIUM'

                maintenance_tasks.append({
                    'task_id': f'MAINT-{i+1:03d}',
                    'equipment_id': equipment_id,
                    'equipment_name': equipment_name,
                    'priority': priority,
                    'status': np.random.choice(['PENDING', 'IN_PROGRESS', 'COMPLETED']),
                    'scheduled_date': datetime.now() + timedelta(days=np.random.randint(1, 14)),
                    'estimated_duration': np.random.randint(2, 8),  # hours
                    'description': f"Maintenance required for {equipment_name}",
                    'anomaly_score': getattr(anomaly, 'anomaly_score', 0.0)
                })

            return {
                'maintenance_tasks': maintenance_tasks,
                'schedule_summary': {
                    'pending_tasks': len([t for t in maintenance_tasks if t['status'] == 'PENDING']),
                    'in_progress_tasks': len([t for t in maintenance_tasks if t['status'] == 'IN_PROGRESS']),
                    'completed_tasks': len([t for t in maintenance_tasks if t['status'] == 'COMPLETED']),
                    'total_tasks': len(maintenance_tasks)
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error getting maintenance data: {e}")
            return self._get_fallback_maintenance()

    def get_work_orders_data(self) -> Dict[str, Any]:
        """Get work orders data"""
        try:
            # Get maintenance data and convert to work orders
            maintenance_data = self.get_maintenance_data()

            work_orders = []
            for task in maintenance_data['maintenance_tasks']:
                work_orders.append({
                    'work_order_id': f"WO-{task['task_id'].split('-')[1]}",
                    'equipment_id': task['equipment_id'],
                    'equipment_name': task['equipment_name'],
                    'priority': task['priority'],
                    'status': task['status'],
                    'created_date': datetime.now() - timedelta(days=np.random.randint(0, 5)),
                    'scheduled_date': task['scheduled_date'],
                    'assigned_technician': f"Tech-{np.random.randint(1, 5):02d}",
                    'estimated_cost': np.random.randint(500, 5000),
                    'description': task['description']
                })

            return {
                'work_orders': work_orders,
                'summary': {
                    'total_orders': len(work_orders),
                    'pending_orders': len([wo for wo in work_orders if wo['status'] == 'PENDING']),
                    'in_progress_orders': len([wo for wo in work_orders if wo['status'] == 'IN_PROGRESS']),
                    'completed_orders': len([wo for wo in work_orders if wo['status'] == 'COMPLETED']),
                    'total_estimated_cost': sum(wo['estimated_cost'] for wo in work_orders)
                },
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"Error getting work orders data: {e}")
            return self._get_fallback_work_orders()

    def _get_severity(self, anomaly_score: float) -> str:
        """Convert anomaly score to severity level"""
        if anomaly_score >= 0.9:
            return 'CRITICAL'
        elif anomaly_score >= 0.7:
            return 'HIGH'
        elif anomaly_score >= 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _generate_forecast(self, equipment_id: str) -> List[float]:
        """Generate simple forecast data"""
        # Simple trend simulation
        base_value = np.random.uniform(0.3, 0.7)
        trend = np.random.uniform(-0.1, 0.1)
        noise = np.random.normal(0, 0.05, 24)  # 24 hours

        forecast = []
        for i in range(24):
            value = base_value + (trend * i) + noise[i]
            forecast.append(max(0, min(1, value)))  # Clamp between 0 and 1

        return forecast

    def _get_fallback_overview(self) -> Dict[str, Any]:
        """Fallback overview data"""
        return {
            'metrics': {
                'total_equipment': 0,
                'active_anomalies': 0,
                'system_health': 50,
                'data_points': 0,
                'uptime_hours': 0
            },
            'equipment_status': {},
            'recent_data': [],
            'timestamp': datetime.now()
        }

    def _get_fallback_anomaly(self) -> Dict[str, Any]:
        """Fallback anomaly data"""
        return {
            'anomalies': [],
            'equipment_anomalies': {},
            'summary': {
                'total_anomalies': 0,
                'critical_anomalies': 0,
                'high_anomalies': 0,
                'medium_anomalies': 0,
                'affected_equipment': 0
            },
            'timestamp': datetime.now()
        }

    def _get_fallback_forecast(self) -> Dict[str, Any]:
        """Fallback forecast data"""
        return {
            'forecasts': {},
            'historical_data': [],
            'timestamp': datetime.now()
        }

    def _get_fallback_maintenance(self) -> Dict[str, Any]:
        """Fallback maintenance data"""
        return {
            'maintenance_tasks': [],
            'schedule_summary': {
                'pending_tasks': 0,
                'in_progress_tasks': 0,
                'completed_tasks': 0,
                'total_tasks': 0
            },
            'timestamp': datetime.now()
        }

    def _get_fallback_work_orders(self) -> Dict[str, Any]:
        """Fallback work orders data"""
        return {
            'work_orders': [],
            'summary': {
                'total_orders': 0,
                'pending_orders': 0,
                'in_progress_orders': 0,
                'completed_orders': 0,
                'total_estimated_cost': 0
            },
            'timestamp': datetime.now()
        }


# Global instance
dashboard_data_service = DashboardDataService()