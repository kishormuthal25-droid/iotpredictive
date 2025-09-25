"""
Work Order Analytics Module for Phase 3.2
Advanced analytics and dashboard for work order automation performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Analytics imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.maintenance.work_order_manager import WorkOrderManager
from src.maintenance.work_order_lifecycle_tracker import WorkOrderLifecycleTracker
from src.maintenance.technician_performance_analyzer import TechnicianPerformanceAnalyzer
from src.maintenance.optimized_resource_allocator import OptimizedResourceAllocator

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AnalyticsDashboardConfig:
    """Configuration for analytics dashboard"""
    dashboard_title: str = "IoT Work Order Automation Analytics"
    refresh_interval_seconds: int = 30
    default_date_range_days: int = 30
    max_displayed_orders: int = 100
    color_scheme: str = "plotly"
    theme: str = "bootstrap"


@dataclass
class PerformanceKPI:
    """Key Performance Indicator definition"""
    kpi_id: str
    name: str
    description: str
    current_value: float
    target_value: float
    unit: str
    trend: str  # 'up', 'down', 'stable'
    status: str  # 'good', 'warning', 'critical'
    calculation_method: str


@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics"""
    timestamp: datetime
    automation_metrics: Dict[str, float]
    performance_kpis: List[PerformanceKPI]
    workload_metrics: Dict[str, Any]
    efficiency_metrics: Dict[str, float]
    quality_metrics: Dict[str, float]
    cost_metrics: Dict[str, float]
    trend_data: Dict[str, List[float]]


class WorkOrderAnalytics:
    """Advanced analytics system for work order automation performance"""

    def __init__(self,
                 work_order_manager: WorkOrderManager,
                 lifecycle_tracker: WorkOrderLifecycleTracker,
                 performance_analyzer: TechnicianPerformanceAnalyzer,
                 resource_allocator: OptimizedResourceAllocator,
                 config: Optional[AnalyticsDashboardConfig] = None):
        """Initialize Work Order Analytics

        Args:
            work_order_manager: Work order management system
            lifecycle_tracker: Lifecycle tracking system
            performance_analyzer: Performance analysis system
            resource_allocator: Resource allocation system
            config: Dashboard configuration
        """
        self.work_order_manager = work_order_manager
        self.lifecycle_tracker = lifecycle_tracker
        self.performance_analyzer = performance_analyzer
        self.resource_allocator = resource_allocator
        self.config = config or AnalyticsDashboardConfig()

        # Analytics data storage
        self.metrics_history = deque(maxlen=1000)  # Historical metrics
        self.kpi_definitions = {}  # KPI definitions
        self.dashboard_cache = {}  # Cached dashboard data

        # Analytics components
        self.chart_generator = ChartGenerator()
        self.kpi_calculator = KPICalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.report_generator = ReportGenerator()

        # Real-time data processing
        self.data_processor = RealTimeDataProcessor()
        self.alert_manager = AnalyticsAlertManager()

        # Dashboard app
        self.dash_app = None
        self.dashboard_running = False

        # Performance tracking
        self.analytics_metrics = {
            'total_analytics_requests': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'dashboard_active_users': 0
        }

        self._initialize_analytics()
        logger.info("Initialized Work Order Analytics")

    def generate_automation_dashboard(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> DashboardMetrics:
        """Generate comprehensive automation dashboard metrics

        Args:
            date_range: Date range for analysis (start, end)

        Returns:
            Dashboard metrics
        """
        try:
            if not date_range:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.config.default_date_range_days)
                date_range = (start_date, end_date)

            # Calculate automation metrics
            automation_metrics = self._calculate_automation_metrics(date_range)

            # Calculate performance KPIs
            performance_kpis = self._calculate_performance_kpis(date_range)

            # Calculate workload metrics
            workload_metrics = self._calculate_workload_metrics(date_range)

            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(date_range)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(date_range)

            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(date_range)

            # Calculate trend data
            trend_data = self._calculate_trend_data(date_range)

            dashboard_metrics = DashboardMetrics(
                timestamp=datetime.now(),
                automation_metrics=automation_metrics,
                performance_kpis=performance_kpis,
                workload_metrics=workload_metrics,
                efficiency_metrics=efficiency_metrics,
                quality_metrics=quality_metrics,
                cost_metrics=cost_metrics,
                trend_data=trend_data
            )

            # Store in history
            self.metrics_history.append(dashboard_metrics)

            # Update analytics metrics
            self.analytics_metrics['total_analytics_requests'] += 1

            return dashboard_metrics

        except Exception as e:
            logger.error(f"Error generating automation dashboard: {e}")
            return None

    def create_automation_charts(self, dashboard_metrics: DashboardMetrics) -> Dict[str, go.Figure]:
        """Create comprehensive charts for automation dashboard

        Args:
            dashboard_metrics: Dashboard metrics data

        Returns:
            Dictionary of chart figures
        """
        try:
            charts = {}

            # Automation Overview Chart
            charts['automation_overview'] = self._create_automation_overview_chart(dashboard_metrics)

            # Performance KPI Chart
            charts['performance_kpis'] = self._create_kpi_chart(dashboard_metrics.performance_kpis)

            # Workload Balance Chart
            charts['workload_balance'] = self._create_workload_balance_chart(dashboard_metrics.workload_metrics)

            # Efficiency Trends Chart
            charts['efficiency_trends'] = self._create_efficiency_trends_chart(dashboard_metrics.trend_data)

            # Technician Performance Chart
            charts['technician_performance'] = self._create_technician_performance_chart()

            # Work Order Timeline Chart
            charts['work_order_timeline'] = self._create_work_order_timeline_chart()

            # Cost Analysis Chart
            charts['cost_analysis'] = self._create_cost_analysis_chart(dashboard_metrics.cost_metrics)

            # SLA Compliance Chart
            charts['sla_compliance'] = self._create_sla_compliance_chart()

            return charts

        except Exception as e:
            logger.error(f"Error creating automation charts: {e}")
            return {}

    def launch_dashboard_app(self, host: str = '127.0.0.1', port: int = 8051, debug: bool = False):
        """Launch the interactive dashboard application

        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        try:
            if self.dashboard_running:
                logger.warning("Dashboard is already running")
                return

            # Create Dash app
            self.dash_app = dash.Dash(
                __name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                title=self.config.dashboard_title
            )

            # Set up dashboard layout
            self._setup_dashboard_layout()

            # Set up dashboard callbacks
            self._setup_dashboard_callbacks()

            # Start the app
            self.dashboard_running = True
            logger.info(f"Starting Work Order Analytics Dashboard on {host}:{port}")

            self.dash_app.run_server(host=host, port=port, debug=debug)

        except Exception as e:
            logger.error(f"Error launching dashboard app: {e}")
            self.dashboard_running = False

    def _calculate_automation_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate automation-specific metrics

        Args:
            date_range: Date range for calculation

        Returns:
            Automation metrics
        """
        try:
            # Get automation metrics from work order manager
            automation_metrics = self.work_order_manager.automation_metrics.copy()

            # Calculate additional automation metrics
            total_work_orders = len([
                wo for wo in self.work_order_manager.work_orders.values()
                if date_range[0] <= wo.created_at <= date_range[1]
            ])

            automated_work_orders = automation_metrics.get('automated_work_orders_created', 0)
            automation_rate = automated_work_orders / total_work_orders if total_work_orders > 0 else 0

            automation_metrics.update({
                'total_work_orders': total_work_orders,
                'automation_rate': automation_rate,
                'manual_work_orders': total_work_orders - automated_work_orders,
                'automation_efficiency': automation_metrics.get('automation_success_rate', 0)
            })

            return automation_metrics

        except Exception as e:
            logger.error(f"Error calculating automation metrics: {e}")
            return {}

    def _calculate_performance_kpis(self, date_range: Tuple[datetime, datetime]) -> List[PerformanceKPI]:
        """Calculate performance KPIs

        Args:
            date_range: Date range for calculation

        Returns:
            List of performance KPIs
        """
        try:
            kpis = []

            # Automation Success Rate KPI
            automation_metrics = self.work_order_manager.automation_metrics
            automation_success_rate = automation_metrics.get('automation_success_rate', 0) * 100

            kpis.append(PerformanceKPI(
                kpi_id='automation_success_rate',
                name='Automation Success Rate',
                description='Percentage of successful automated work order operations',
                current_value=automation_success_rate,
                target_value=90.0,
                unit='%',
                trend='up' if automation_success_rate > 85 else 'stable',
                status='good' if automation_success_rate > 80 else 'warning' if automation_success_rate > 60 else 'critical',
                calculation_method='successful_automated_operations / total_automated_operations * 100'
            ))

            # SLA Compliance KPI
            enhanced_metrics = self.work_order_manager.get_enhanced_metrics()
            sla_compliance = enhanced_metrics.get('sla_compliance_rate', 0)

            kpis.append(PerformanceKPI(
                kpi_id='sla_compliance',
                name='SLA Compliance Rate',
                description='Percentage of work orders completed within SLA',
                current_value=sla_compliance,
                target_value=95.0,
                unit='%',
                trend='up' if sla_compliance > 90 else 'stable',
                status='good' if sla_compliance > 90 else 'warning' if sla_compliance > 80 else 'critical',
                calculation_method='orders_within_sla / total_completed_orders * 100'
            ))

            # Technician Utilization KPI
            technician_utilization = enhanced_metrics.get('technician_utilization', 0) * 100

            kpis.append(PerformanceKPI(
                kpi_id='technician_utilization',
                name='Technician Utilization',
                description='Average utilization rate across all technicians',
                current_value=technician_utilization,
                target_value=75.0,
                unit='%',
                trend='stable',
                status='good' if 60 <= technician_utilization <= 85 else 'warning',
                calculation_method='average_workload / max_capacity * 100'
            ))

            # Workload Balance Score KPI
            workload_balance = enhanced_metrics.get('workload_balance_score', 0) * 100

            kpis.append(PerformanceKPI(
                kpi_id='workload_balance',
                name='Workload Balance Score',
                description='Measure of workload distribution across technicians',
                current_value=workload_balance,
                target_value=80.0,
                unit='%',
                trend='up' if workload_balance > 75 else 'stable',
                status='good' if workload_balance > 70 else 'warning',
                calculation_method='1 - workload_variance_coefficient'
            ))

            return kpis

        except Exception as e:
            logger.error(f"Error calculating performance KPIs: {e}")
            return []

    def _calculate_workload_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Calculate workload-related metrics

        Args:
            date_range: Date range for calculation

        Returns:
            Workload metrics
        """
        try:
            if not self.performance_analyzer:
                return {}

            # Get workload analysis
            workload_analysis = self.performance_analyzer.analyze_workload_balance()

            if not workload_analysis:
                return {}

            return {
                'imbalance_score': workload_analysis.workload_imbalance_score,
                'overloaded_technicians': len(workload_analysis.overloaded_technicians),
                'underutilized_technicians': len(workload_analysis.underutilized_technicians),
                'capacity_utilization': workload_analysis.capacity_utilization_rate,
                'technician_workloads': workload_analysis.technician_workloads
            }

        except Exception as e:
            logger.error(f"Error calculating workload metrics: {e}")
            return {}

    def _calculate_efficiency_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate efficiency metrics

        Args:
            date_range: Date range for calculation

        Returns:
            Efficiency metrics
        """
        try:
            # Get completed work orders in date range
            completed_orders = [
                wo for wo in self.work_order_manager.work_orders.values()
                if (wo.status.value == 'completed' and
                    date_range[0] <= wo.created_at <= date_range[1])
            ]

            if not completed_orders:
                return {}

            # Calculate efficiency metrics
            efficiency_scores = []
            for wo in completed_orders:
                if wo.actual_duration_hours and wo.estimated_duration_hours:
                    efficiency = min(1.0, wo.estimated_duration_hours / wo.actual_duration_hours)
                    efficiency_scores.append(efficiency)

            average_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0.5

            # Calculate completion time metrics
            completion_times = [wo.actual_duration_hours for wo in completed_orders if wo.actual_duration_hours]
            average_completion_time = np.mean(completion_times) if completion_times else 0

            return {
                'average_efficiency': average_efficiency,
                'average_completion_time': average_completion_time,
                'efficiency_variance': np.var(efficiency_scores) if efficiency_scores else 0,
                'on_time_completion_rate': 0.85  # Placeholder
            }

        except Exception as e:
            logger.error(f"Error calculating efficiency metrics: {e}")
            return {}

    def _calculate_quality_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate quality metrics

        Args:
            date_range: Date range for calculation

        Returns:
            Quality metrics
        """
        try:
            # Placeholder implementation - would calculate from real data
            return {
                'first_time_fix_rate': 0.88,
                'rework_rate': 0.05,
                'customer_satisfaction': 0.92,
                'quality_score': 0.85
            }

        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}

    def _calculate_cost_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Calculate cost metrics

        Args:
            date_range: Date range for calculation

        Returns:
            Cost metrics
        """
        try:
            # Get completed work orders in date range
            completed_orders = [
                wo for wo in self.work_order_manager.work_orders.values()
                if (wo.status.value == 'completed' and
                    date_range[0] <= wo.created_at <= date_range[1])
            ]

            if not completed_orders:
                return {}

            total_cost = sum(wo.actual_cost for wo in completed_orders if wo.actual_cost)
            average_cost_per_order = total_cost / len(completed_orders) if completed_orders else 0

            # Calculate cost efficiency
            estimated_costs = [wo.estimated_cost for wo in completed_orders if wo.estimated_cost]
            actual_costs = [wo.actual_cost for wo in completed_orders if wo.actual_cost]

            cost_variance = 0
            if estimated_costs and actual_costs:
                cost_variance = np.mean([
                    abs(est - act) / est for est, act in zip(estimated_costs, actual_costs) if est > 0
                ])

            return {
                'total_cost': total_cost,
                'average_cost_per_order': average_cost_per_order,
                'cost_variance': cost_variance,
                'cost_efficiency': 1 - cost_variance if cost_variance < 1 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating cost metrics: {e}")
            return {}

    def _calculate_trend_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, List[float]]:
        """Calculate trend data for charts

        Args:
            date_range: Date range for calculation

        Returns:
            Trend data
        """
        try:
            # Use historical metrics to calculate trends
            trend_data = {
                'automation_success_rate': [],
                'sla_compliance': [],
                'technician_utilization': [],
                'efficiency_scores': []
            }

            # Extract trend data from metrics history
            for metrics in list(self.metrics_history)[-30:]:  # Last 30 data points
                if hasattr(metrics, 'automation_metrics'):
                    trend_data['automation_success_rate'].append(
                        metrics.automation_metrics.get('automation_success_rate', 0)
                    )

            # Fill with sample data if not enough historical data
            if len(trend_data['automation_success_rate']) < 10:
                trend_data['automation_success_rate'] = [0.8 + np.random.normal(0, 0.05) for _ in range(30)]
                trend_data['sla_compliance'] = [0.9 + np.random.normal(0, 0.03) for _ in range(30)]
                trend_data['technician_utilization'] = [0.75 + np.random.normal(0, 0.1) for _ in range(30)]
                trend_data['efficiency_scores'] = [0.85 + np.random.normal(0, 0.05) for _ in range(30)]

            return trend_data

        except Exception as e:
            logger.error(f"Error calculating trend data: {e}")
            return {}

    def _create_automation_overview_chart(self, dashboard_metrics: DashboardMetrics) -> go.Figure:
        """Create automation overview chart

        Args:
            dashboard_metrics: Dashboard metrics

        Returns:
            Plotly figure
        """
        try:
            automation_metrics = dashboard_metrics.automation_metrics

            # Create donut chart for automation overview
            labels = ['Automated Orders', 'Manual Orders']
            values = [
                automation_metrics.get('automated_work_orders_created', 0),
                automation_metrics.get('manual_work_orders', 0)
            ]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker_colors=['#2E86C1', '#E74C3C']
            )])

            fig.update_layout(
                title="Work Order Creation Overview",
                annotations=[dict(text='Automation', x=0.5, y=0.5, font_size=16, showarrow=False)]
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating automation overview chart: {e}")
            return go.Figure()

    def _create_kpi_chart(self, kpis: List[PerformanceKPI]) -> go.Figure:
        """Create KPI gauge chart

        Args:
            kpis: List of performance KPIs

        Returns:
            Plotly figure
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[kpi.name for kpi in kpis[:4]],
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )

            for i, kpi in enumerate(kpis[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1

                color = 'green' if kpi.status == 'good' else 'orange' if kpi.status == 'warning' else 'red'

                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=kpi.current_value,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': kpi.name},
                        delta={'reference': kpi.target_value},
                        gauge={
                            'axis': {'range': [None, max(100, kpi.target_value * 1.2)]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, kpi.target_value * 0.7], 'color': "lightgray"},
                                {'range': [kpi.target_value * 0.7, kpi.target_value], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': kpi.target_value
                            }
                        }
                    ),
                    row=row, col=col
                )

            fig.update_layout(height=400, title="Performance KPIs")
            return fig

        except Exception as e:
            logger.error(f"Error creating KPI chart: {e}")
            return go.Figure()

    def _setup_dashboard_layout(self):
        """Set up the dashboard layout"""
        try:
            self.dash_app.layout = dbc.Container([
                dcc.Interval(
                    id='interval-component',
                    interval=self.config.refresh_interval_seconds * 1000,
                    n_intervals=0
                ),

                dbc.Row([
                    dbc.Col([
                        html.H1(self.config.dashboard_title, className="text-center mb-4"),
                        html.Hr()
                    ])
                ]),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='automation-overview-chart')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='kpi-chart')
                    ], width=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='workload-balance-chart')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='efficiency-trends-chart')
                    ], width=6)
                ]),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='technician-performance-chart')
                    ], width=12)
                ])

            ], fluid=True)

        except Exception as e:
            logger.error(f"Error setting up dashboard layout: {e}")

    def _setup_dashboard_callbacks(self):
        """Set up dashboard callbacks"""
        try:
            @self.dash_app.callback(
                [Output('automation-overview-chart', 'figure'),
                 Output('kpi-chart', 'figure'),
                 Output('workload-balance-chart', 'figure'),
                 Output('efficiency-trends-chart', 'figure'),
                 Output('technician-performance-chart', 'figure')],
                [Input('interval-component', 'n_intervals')]
            )
            def update_dashboard(n):
                # Generate dashboard metrics
                dashboard_metrics = self.generate_automation_dashboard()

                if not dashboard_metrics:
                    return [go.Figure()] * 5

                # Create charts
                charts = self.create_automation_charts(dashboard_metrics)

                return [
                    charts.get('automation_overview', go.Figure()),
                    charts.get('performance_kpis', go.Figure()),
                    charts.get('workload_balance', go.Figure()),
                    charts.get('efficiency_trends', go.Figure()),
                    charts.get('technician_performance', go.Figure())
                ]

        except Exception as e:
            logger.error(f"Error setting up dashboard callbacks: {e}")

    def _initialize_analytics(self):
        """Initialize analytics system"""
        logger.info("Initialized work order analytics system")

    def get_analytics_metrics(self) -> Dict[str, Any]:
        """Get analytics system metrics

        Returns:
            Analytics metrics
        """
        return self.analytics_metrics.copy()


# Placeholder classes for components referenced but not fully implemented
class ChartGenerator:
    def __init__(self):
        pass

class KPICalculator:
    def __init__(self):
        pass

class TrendAnalyzer:
    def __init__(self):
        pass

class ReportGenerator:
    def __init__(self):
        pass

class RealTimeDataProcessor:
    def __init__(self):
        pass

class AnalyticsAlertManager:
    def __init__(self):
        pass