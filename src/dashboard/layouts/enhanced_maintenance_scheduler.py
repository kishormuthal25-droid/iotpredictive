"""
Enhanced Maintenance Scheduler for Phase 3.1 IoT Anomaly Detection System
Advanced calendar views, resource optimization, cost forecasting, and compliance tracking
"""

from dash import html, dcc, Input, Output, State, callback, dash_table, ALL, MATCH, clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
from collections import defaultdict
import logging
import json
import calendar
import uuid
from dataclasses import dataclass, asdict
import pulp

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class MaintenanceTask:
    """Enhanced maintenance task with resource requirements"""
    task_id: str
    equipment_id: str
    task_type: str
    priority: str
    start_date: datetime
    duration_hours: float
    assigned_technician: Optional[str] = None
    required_skills: List[str] = None
    required_parts: List[Dict[str, Any]] = None
    estimated_cost: float = 0.0
    compliance_requirements: List[str] = None
    dependencies: List[str] = None
    is_draggable: bool = True
    status: str = "scheduled"


@dataclass
class Technician:
    """Enhanced technician with skills and availability"""
    technician_id: str
    name: str
    skills: List[str]
    hourly_rate: float
    availability: Dict[str, List[Tuple[str, str]]]  # date -> [(start, end), ...]
    max_hours_per_day: int = 8
    current_workload: float = 0.0
    efficiency_rating: float = 1.0


@dataclass
class InventoryItem:
    """Parts inventory item"""
    part_id: str
    name: str
    current_stock: int
    min_stock_level: int
    max_stock_level: int
    unit_cost: float
    supplier: str
    lead_time_days: int
    last_order_date: Optional[datetime] = None


class EnhancedMaintenanceScheduler:
    """Enhanced maintenance scheduling with Phase 3.1 features"""

    def __init__(self, data_manager=None, config=None):
        """Initialize Enhanced Maintenance Scheduler

        Args:
            data_manager: Data manager instance
            config: Configuration object
        """
        self.data_manager = data_manager
        self.config = config or {}

        # Enhanced data structures
        self.maintenance_tasks = {}
        self.technicians = {}
        self.inventory = {}
        self.compliance_rules = {}
        self.cost_model = None

        # Calendar state
        self.current_view = "month"
        self.current_date = datetime.now()

        # Optimization parameters
        self.optimization_weights = {
            'cost': 0.3,
            'time': 0.3,
            'resource_utilization': 0.2,
            'priority': 0.2
        }

        # Initialize sample data
        self._initialize_sample_data()

        logger.info("Initialized Enhanced Maintenance Scheduler")

    def _initialize_sample_data(self):
        """Initialize sample data for demonstration"""
        # Sample technicians with skills
        self.technicians = {
            'tech_001': Technician(
                'tech_001', 'John Smith',
                ['electrical', 'mechanical', 'hydraulic'],
                75.0, {}, 8, 0.0, 1.2
            ),
            'tech_002': Technician(
                'tech_002', 'Jane Doe',
                ['electronics', 'programming', 'diagnostics'],
                85.0, {}, 8, 0.0, 1.1
            ),
            'tech_003': Technician(
                'tech_003', 'Bob Johnson',
                ['mechanical', 'welding', 'fabrication'],
                70.0, {}, 8, 0.0, 0.9
            ),
            'tech_004': Technician(
                'tech_004', 'Alice Brown',
                ['electrical', 'automation', 'sensors'],
                90.0, {}, 8, 0.0, 1.3
            )
        }

        # Sample inventory
        self.inventory = {
            'bearing_001': InventoryItem(
                'bearing_001', 'High-Performance Bearing',
                25, 10, 50, 125.50, 'BearingCorp', 5
            ),
            'filter_001': InventoryItem(
                'filter_001', 'Oil Filter Assembly',
                8, 15, 40, 45.75, 'FilterMax', 3
            ),
            'sensor_001': InventoryItem(
                'sensor_001', 'Temperature Sensor',
                12, 5, 30, 85.00, 'SensorTech', 7
            )
        }

        # Sample compliance rules
        self.compliance_rules = {
            'safety_inspection': {
                'frequency_days': 30,
                'required_certifications': ['safety_cert'],
                'documentation_required': True
            },
            'regulatory_audit': {
                'frequency_days': 365,
                'required_certifications': ['audit_cert'],
                'documentation_required': True
            }
        }

    def create_enhanced_layout(self) -> html.Div:
        """Create enhanced maintenance scheduler layout

        Returns:
            Enhanced maintenance scheduler layout
        """
        return html.Div([
            # Enhanced header with advanced controls
            dbc.Row([
                dbc.Col([
                    html.H3("Enhanced Maintenance Scheduler"),
                    html.P("AI-powered scheduling with resource optimization", className="text-muted")
                ], width=6),

                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-plus me-2"),
                            "New Task"
                        ], id="enhanced-new-task-btn", color="success", size="sm"),

                        dbc.Button([
                            html.I(className="fas fa-magic me-2"),
                            "Auto-Optimize"
                        ], id="auto-optimize-btn", color="warning", size="sm"),

                        dbc.Button([
                            html.I(className="fas fa-brain me-2"),
                            "AI Suggestions"
                        ], id="ai-suggestions-btn", color="info", size="sm"),

                        dbc.Button([
                            html.I(className="fas fa-download me-2"),
                            "Export"
                        ], id="enhanced-export-btn", color="secondary", size="sm")
                    ])
                ], width=6, className="text-end")
            ], className="mb-4"),

            # Enhanced overview cards with real-time metrics
            dbc.Row([
                dbc.Col([
                    self._create_enhanced_schedule_overview()
                ], width=3),

                dbc.Col([
                    self._create_ai_resource_optimization()
                ], width=3),

                dbc.Col([
                    self._create_cost_forecasting_card()
                ], width=3),

                dbc.Col([
                    self._create_compliance_dashboard()
                ], width=3)
            ], className="mb-4"),

            # Advanced filters and controls
            dbc.Row([
                dbc.Col([
                    self._create_advanced_filters()
                ], width=12)
            ], className="mb-4"),

            # Main enhanced calendar view
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col([
                                    dbc.ButtonGroup([
                                        dbc.Button("Month", id="month-view-btn", color="primary", size="sm", active=True),
                                        dbc.Button("Week", id="week-view-btn", color="primary", size="sm"),
                                        dbc.Button("Day", id="day-view-btn", color="primary", size="sm"),
                                        dbc.Button("Gantt", id="enhanced-gantt-btn", color="primary", size="sm"),
                                        dbc.Button("Timeline", id="timeline-view-btn", color="primary", size="sm")
                                    ])
                                ], width=8),

                                dbc.Col([
                                    html.Div(id="calendar-date-display", className="text-end fw-bold")
                                ], width=4)
                            ])
                        ]),

                        dbc.CardBody([
                            html.Div(id="enhanced-calendar-container", children=[
                                self._create_drag_drop_calendar()
                            ], style={'minHeight': '600px'})
                        ])
                    ])
                ], width=9),

                # Enhanced side panel with multiple tabs
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Tabs([
                                dbc.Tab(label="Task Details", tab_id="task-details-tab"),
                                dbc.Tab(label="Resources", tab_id="resources-tab"),
                                dbc.Tab(label="Inventory", tab_id="inventory-tab"),
                                dbc.Tab(label="Compliance", tab_id="compliance-tab")
                            ], id="side-panel-tabs", active_tab="task-details-tab")
                        ]),

                        dbc.CardBody([
                            html.Div(id="side-panel-content")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),

            # Enhanced bottom panels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("AI-Powered Optimization Suggestions"),
                        dbc.CardBody([
                            html.Div(id="ai-optimization-suggestions")
                        ])
                    ])
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Parts Inventory Manager"),
                        dbc.CardBody([
                            html.Div(id="inventory-manager")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Advanced analytics panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predictive Analytics Dashboard"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab(label="Cost Trends", tab_id="cost-trends"),
                                dbc.Tab(label="Resource Utilization", tab_id="resource-util"),
                                dbc.Tab(label="Compliance Metrics", tab_id="compliance-metrics"),
                                dbc.Tab(label="Performance KPIs", tab_id="performance-kpis")
                            ], id="analytics-tabs", active_tab="cost-trends"),

                            html.Div(id="analytics-content", className="mt-3")
                        ])
                    ])
                ], width=12)
            ]),

            # Hidden stores for enhanced functionality
            dcc.Store(id="enhanced-schedule-store", storage_type="memory"),
            dcc.Store(id="optimization-results-store", storage_type="session"),
            dcc.Store(id="drag-drop-state-store", storage_type="memory"),
            dcc.Store(id="ai-suggestions-store", storage_type="memory"),

            # Enhanced modals
            self._create_enhanced_task_modal(),
            self._create_optimization_modal(),
            self._create_inventory_order_modal(),
            self._create_compliance_audit_modal(),

            # Interval for real-time updates
            dcc.Interval(
                id="real-time-update-interval",
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ])

    def _create_enhanced_schedule_overview(self) -> dbc.Card:
        """Create enhanced schedule overview with AI insights

        Returns:
            Enhanced schedule overview card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("AI Schedule Overview", className="text-muted mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.H4("42", className="mb-0 text-primary", id="tasks-this-week"),
                        html.Small("This Week")
                    ], width=6),

                    dbc.Col([
                        html.H4("94%", className="mb-0 text-success", id="efficiency-score"),
                        html.Small("Efficiency")
                    ], width=6)
                ]),

                html.Hr(className="my-2"),

                # Real-time task status with AI predictions
                html.Div([
                    self._create_smart_stat("Optimized", 28, "success", "AI-optimized schedule"),
                    self._create_smart_stat("At Risk", 3, "warning", "Potential delays detected"),
                    self._create_smart_stat("Critical", 1, "danger", "Requires immediate attention")
                ]),

                # AI insights mini-panel
                html.Div([
                    html.Hr(className="my-2"),
                    html.Small([
                        html.I(className="fas fa-lightbulb text-warning me-1"),
                        "AI suggests rescheduling 2 tasks for 15% cost savings"
                    ], className="text-muted")
                ])
            ])
        ])

    def _create_smart_stat(self, label: str, value: int, color: str, tooltip: str) -> html.Div:
        """Create smart statistic with tooltip

        Args:
            label: Stat label
            value: Stat value
            color: Display color
            tooltip: Tooltip text

        Returns:
            Smart stat div with tooltip
        """
        return html.Div([
            html.Small(label, className="text-muted", title=tooltip),
            dbc.Badge([
                str(value),
                html.I(className="fas fa-info-circle ms-1", style={'fontSize': '8px'})
            ], color=color, pill=True, className="ms-auto")
        ], className="d-flex justify-content-between align-items-center mb-1")

    def _create_ai_resource_optimization(self) -> dbc.Card:
        """Create AI-powered resource optimization card

        Returns:
            AI resource optimization card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("AI Resource Optimizer", className="text-muted mb-3"),

                # Optimization score
                html.Div([
                    daq.Gauge(
                        id="optimization-score-gauge",
                        label="Optimization",
                        value=87,
                        max=100,
                        min=0,
                        showCurrentValue=True,
                        units="%",
                        color={"gradient": True, "ranges": {
                            "red": [0, 60],
                            "yellow": [60, 80],
                            "green": [80, 100]
                        }},
                        size=100
                    )
                ], className="text-center"),

                # Resource allocation efficiency
                html.Div([
                    html.Small("Resource Allocation", className="text-muted d-block"),
                    html.Div([
                        html.Span("Technicians: "),
                        dbc.Badge("Optimal", color="success", className="ms-1")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Equipment: "),
                        dbc.Badge("Good", color="info", className="ms-1")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Parts: "),
                        dbc.Badge("Needs Attention", color="warning", className="ms-1")
                    ])
                ], className="mt-2")
            ])
        ])

    def _create_cost_forecasting_card(self) -> dbc.Card:
        """Create intelligent cost forecasting card

        Returns:
            Cost forecasting card with ML predictions
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Cost Forecasting", className="text-muted mb-3"),

                html.Div([
                    html.H4("$24,500", className="mb-0", id="predicted-cost"),
                    html.Small([
                        html.I(className="fas fa-chart-line text-primary me-1"),
                        "ML Prediction for next 30 days"
                    ])
                ]),

                # Cost breakdown
                html.Div([
                    html.Hr(className="my-2"),
                    html.Div([
                        html.Small("Labor: $15,200 (62%)", className="d-block text-muted"),
                        html.Small("Parts: $6,800 (28%)", className="d-block text-muted"),
                        html.Small("Overhead: $2,500 (10%)", className="d-block text-muted")
                    ])
                ]),

                # Confidence and trend
                dcc.Graph(
                    id="cost-trend-mini",
                    figure=self._create_cost_trend_chart(),
                    config={'displayModeBar': False},
                    style={'height': '60px', 'marginTop': '10px'}
                ),

                html.Small([
                    html.I(className="fas fa-shield-alt text-success me-1"),
                    "95% confidence interval"
                ], className="text-muted")
            ])
        ])

    def _create_compliance_dashboard(self) -> dbc.Card:
        """Create regulatory compliance dashboard card

        Returns:
            Compliance dashboard card
        """
        return dbc.Card([
            dbc.CardBody([
                html.H6("Compliance Tracker", className="text-muted mb-3"),

                # Overall compliance score
                html.Div([
                    daq.Gauge(
                        id="compliance-score-gauge",
                        label="Compliance",
                        value=94,
                        max=100,
                        min=0,
                        showCurrentValue=True,
                        units="%",
                        color={"gradient": True, "ranges": {
                            "red": [0, 70],
                            "yellow": [70, 90],
                            "green": [90, 100]
                        }},
                        size=100
                    )
                ], className="text-center"),

                # Compliance alerts
                html.Div([
                    html.Hr(className="my-2"),
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                        html.Small("3 audits due this month")
                    ], className="mb-1"),
                    html.Div([
                        html.I(className="fas fa-clock text-info me-2"),
                        html.Small("5 certifications expiring")
                    ], className="mb-1"),
                    html.Div([
                        html.I(className="fas fa-check-circle text-success me-2"),
                        html.Small("12 compliant tasks completed")
                    ])
                ])
            ])
        ])

    def _create_advanced_filters(self) -> dbc.Card:
        """Create advanced filtering and control panel

        Returns:
            Advanced filters card
        """
        return dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Smart Date Range", className="fw-bold"),
                        dcc.DatePickerRange(
                            id="enhanced-date-range",
                            start_date=datetime.now(),
                            end_date=datetime.now() + timedelta(days=30),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"}
                        )
                    ], width=2),

                    dbc.Col([
                        html.Label("Equipment Filter", className="fw-bold"),
                        dcc.Dropdown(
                            id="enhanced-equipment-filter",
                            options=[
                                {"label": "All Equipment", "value": "all"},
                                {"label": "Critical Assets", "value": "critical"},
                                {"label": "High Priority", "value": "high_priority"},
                                {"label": "Zone A", "value": "zone_a"},
                                {"label": "Zone B", "value": "zone_b"}
                            ],
                            value="all",
                            clearable=False
                        )
                    ], width=2),

                    dbc.Col([
                        html.Label("Task Type", className="fw-bold"),
                        dcc.Dropdown(
                            id="enhanced-task-type-filter",
                            options=[
                                {"label": "All Types", "value": "all"},
                                {"label": "Preventive", "value": "preventive"},
                                {"label": "Corrective", "value": "corrective"},
                                {"label": "Predictive", "value": "predictive"},
                                {"label": "Emergency", "value": "emergency"}
                            ],
                            value="all",
                            clearable=False
                        )
                    ], width=2),

                    dbc.Col([
                        html.Label("Technician Skills", className="fw-bold"),
                        dcc.Dropdown(
                            id="technician-skills-filter",
                            options=[
                                {"label": "All Skills", "value": "all"},
                                {"label": "Electrical", "value": "electrical"},
                                {"label": "Mechanical", "value": "mechanical"},
                                {"label": "Electronics", "value": "electronics"},
                                {"label": "Hydraulic", "value": "hydraulic"}
                            ],
                            value="all",
                            multi=True
                        )
                    ], width=2),

                    dbc.Col([
                        html.Label("Cost Range", className="fw-bold"),
                        dcc.RangeSlider(
                            id="cost-range-filter",
                            min=0,
                            max=10000,
                            step=100,
                            value=[0, 5000],
                            marks={0: '$0', 2500: '$2.5K', 5000: '$5K', 7500: '$7.5K', 10000: '$10K'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=2),

                    dbc.Col([
                        html.Label("Actions", className="fw-bold"),
                        dbc.ButtonGroup([
                            dbc.Button([
                                html.I(className="fas fa-filter me-1"),
                                "Apply"
                            ], id="apply-filters-btn", color="primary", size="sm"),

                            dbc.Button([
                                html.I(className="fas fa-undo me-1"),
                                "Reset"
                            ], id="reset-filters-btn", color="secondary", size="sm")
                        ], vertical=False)
                    ], width=2)
                ])
            ])
        ])

    def _create_drag_drop_calendar(self) -> html.Div:
        """Create drag-and-drop enabled calendar

        Returns:
            Enhanced calendar with drag-drop functionality
        """
        return html.Div([
            # Calendar navigation
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("◄◄", id="prev-month-btn", size="sm", color="secondary"),
                        dbc.Button("◄", id="prev-week-btn", size="sm", color="secondary"),
                        dbc.Button("Today", id="today-btn", size="sm", color="info"),
                        dbc.Button("►", id="next-week-btn", size="sm", color="secondary"),
                        dbc.Button("►►", id="next-month-btn", size="sm", color="secondary")
                    ])
                ], width="auto"),

                dbc.Col([
                    html.H5(
                        datetime.now().strftime("%B %Y"),
                        id="calendar-title",
                        className="mb-0 text-center"
                    )
                ], width=True),

                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-sync-alt me-1"),
                            "Refresh"
                        ], id="refresh-calendar-btn", size="sm", color="success"),

                        dbc.Button([
                            html.I(className="fas fa-cog me-1"),
                            "Settings"
                        ], id="calendar-settings-btn", size="sm", color="secondary")
                    ])
                ], width="auto")
            ], className="mb-3"),

            # Enhanced calendar grid with drag-drop zones
            html.Div([
                html.Div(id="calendar-grid-container", children=[
                    self._create_enhanced_month_view()
                ], style={
                    'border': '1px solid #dee2e6',
                    'borderRadius': '0.375rem',
                    'backgroundColor': '#ffffff'
                })
            ]),

            # Drag and drop feedback
            html.Div(id="drag-drop-feedback", className="mt-2"),

            # Task creation zone
            html.Div([
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "Drag tasks between dates to reschedule. Click on empty dates to create new tasks."
                ], color="info", className="mb-0 mt-2", style={'fontSize': '0.875rem'})
            ])
        ])

    def _create_enhanced_month_view(self) -> html.Div:
        """Create enhanced month view with interactive features

        Returns:
            Enhanced month view div
        """
        current_date = datetime.now()
        cal = calendar.monthcalendar(current_date.year, current_date.month)

        # Days of week header with enhanced styling
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        header = dbc.Row([
            dbc.Col(
                html.Div(
                    day[:3],
                    className="text-center fw-bold py-2 border-bottom",
                    style={'backgroundColor': '#f8f9fa', 'color': '#495057'}
                ),
                width=True
            )
            for day in weekdays
        ], className="mb-0", style={'borderRadius': '0.375rem 0.375rem 0 0'})

        # Enhanced calendar weeks with drag-drop zones
        weeks = []
        for week_idx, week in enumerate(cal):
            week_cols = []
            for day in week:
                if day == 0:
                    # Empty day cell
                    week_cols.append(
                        dbc.Col(
                            html.Div("", className="calendar-day-cell empty-day"),
                            width=True
                        )
                    )
                else:
                    # Active day cell with tasks and drop zone
                    tasks = self._get_enhanced_tasks_for_day(current_date.year, current_date.month, day)
                    is_today = (day == current_date.day)

                    day_cell = html.Div([
                        # Day number header
                        html.Div([
                            html.Span(str(day), className="day-number"),
                            html.Div([
                                html.I(className="fas fa-plus", style={'fontSize': '10px'}),
                            ], className="add-task-btn",
                               id={"type": "add-task", "date": f"{current_date.year}-{current_date.month:02d}-{day:02d}"})
                        ], className="day-header"),

                        # Task container (drop zone)
                        html.Div(
                            tasks,
                            className="task-container",
                            id={"type": "drop-zone", "date": f"{current_date.year}-{current_date.month:02d}-{day:02d}"}
                        ),

                        # Workload indicator
                        html.Div([
                            html.Div(
                                className="workload-bar",
                                style={
                                    'width': f"{min(len(tasks) * 25, 100)}%",
                                    'backgroundColor': self._get_workload_color(len(tasks))
                                }
                            )
                        ], className="workload-indicator")

                    ], className=f"calendar-day-cell {'today' if is_today else ''} {'weekend' if calendar.weekday(current_date.year, current_date.month, day) >= 5 else ''}",
                       id={"type": "calendar-day", "date": f"{current_date.year}-{current_date.month:02d}-{day:02d}"})

                    week_cols.append(dbc.Col(day_cell, width=True))

            weeks.append(dbc.Row(week_cols, className="calendar-week-row", no_gutters=True))

        return html.Div([header] + weeks, className="enhanced-calendar-container")

    def _get_enhanced_tasks_for_day(self, year: int, month: int, day: int) -> List[html.Div]:
        """Get enhanced maintenance tasks for specific day with drag-drop capability

        Args:
            year: Year
            month: Month
            day: Day

        Returns:
            List of enhanced draggable task divs
        """
        # Enhanced sample tasks with more details
        sample_tasks = {
            5: [
                {'id': 'task_001', 'type': 'preventive', 'equipment': 'EQ-001', 'technician': 'John', 'priority': 'medium', 'duration': 2},
                {'id': 'task_002', 'type': 'inspection', 'equipment': 'EQ-003', 'technician': 'Jane', 'priority': 'low', 'duration': 1}
            ],
            10: [
                {'id': 'task_003', 'type': 'corrective', 'equipment': 'EQ-002', 'technician': 'Bob', 'priority': 'high', 'duration': 4}
            ],
            15: [
                {'id': 'task_004', 'type': 'predictive', 'equipment': 'EQ-004', 'technician': 'Alice', 'priority': 'critical', 'duration': 3},
                {'id': 'task_005', 'type': 'inspection', 'equipment': 'EQ-005', 'technician': 'John', 'priority': 'medium', 'duration': 1.5}
            ]
        }

        tasks = []
        if day in sample_tasks:
            for task in sample_tasks[day]:
                priority_colors = {
                    'critical': '#dc3545',
                    'high': '#fd7e14',
                    'medium': '#ffc107',
                    'low': '#28a745'
                }

                type_colors = {
                    'preventive': '#28a745',
                    'corrective': '#dc3545',
                    'predictive': '#ffc107',
                    'inspection': '#17a2b8',
                    'emergency': '#dc3545'
                }

                task_div = html.Div([
                    html.Div([
                        html.I(className="fas fa-grip-vertical me-1", style={'fontSize': '8px'}),
                        html.Span(task['equipment'], style={'fontSize': '10px', 'fontWeight': 'bold'}),
                        html.Span(f" ({task['duration']}h)", style={'fontSize': '9px', 'color': '#6c757d'})
                    ], className="task-header"),

                    html.Div([
                        html.Span(task['technician'], style={'fontSize': '9px'}),
                        html.Div(
                            className="priority-indicator",
                            style={
                                'backgroundColor': priority_colors.get(task['priority'], '#6c757d'),
                                'width': '8px',
                                'height': '8px',
                                'borderRadius': '50%',
                                'marginLeft': 'auto'
                            }
                        )
                    ], className="task-footer d-flex align-items-center")

                ], className="draggable-task",
                   draggable=True,
                   id={"type": "task", "task_id": task['id']},
                   style={
                       'backgroundColor': type_colors.get(task['type'], '#f8f9fa'),
                       'border': f"1px solid {type_colors.get(task['type'], '#dee2e6')}",
                       'borderRadius': '3px',
                       'padding': '2px 4px',
                       'margin': '1px 0',
                       'cursor': 'move',
                       'fontSize': '10px',
                       'minHeight': '24px'
                   })

                tasks.append(task_div)

        return tasks

    def _get_workload_color(self, task_count: int) -> str:
        """Get workload indicator color based on task count

        Args:
            task_count: Number of tasks

        Returns:
            Color hex code
        """
        if task_count == 0:
            return '#e9ecef'
        elif task_count <= 2:
            return '#28a745'
        elif task_count <= 4:
            return '#ffc107'
        else:
            return '#dc3545'

    def _create_cost_trend_chart(self) -> go.Figure:
        """Create cost trend mini-chart with ML predictions

        Returns:
            Plotly figure for cost trends
        """
        # Generate sample cost data with predictions
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=60, freq='D')
        historical_costs = np.random.normal(23000, 2000, 30)
        predicted_costs = np.random.normal(24500, 1500, 30)

        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=dates[:30],
            y=historical_costs,
            mode='lines',
            line=dict(color='#17a2b8', width=2),
            name='Historical',
            showlegend=False
        ))

        # Predicted data
        fig.add_trace(go.Scatter(
            x=dates[30:],
            y=predicted_costs,
            mode='lines',
            line=dict(color='#ffc107', width=2, dash='dash'),
            name='Predicted',
            showlegend=False
        ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode=False,
            height=60
        )

        return fig

    def _create_enhanced_task_modal(self) -> dbc.Modal:
        """Create enhanced task creation/editing modal

        Returns:
            Enhanced task modal with AI suggestions
        """
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Enhanced Task Scheduler")),
            dbc.ModalBody([
                dbc.Tabs([
                    dbc.Tab(label="Basic Info", tab_id="basic-info"),
                    dbc.Tab(label="Resources", tab_id="resources"),
                    dbc.Tab(label="Compliance", tab_id="compliance"),
                    dbc.Tab(label="AI Optimizer", tab_id="ai-optimizer")
                ], id="task-modal-tabs", active_tab="basic-info"),

                html.Div(id="task-modal-content", className="mt-3")
            ]),
            dbc.ModalFooter([
                dbc.Button("AI Optimize", id="ai-optimize-task", color="warning"),
                dbc.Button("Schedule", id="schedule-enhanced-task", color="primary"),
                dbc.Button("Cancel", id="cancel-enhanced-task", color="secondary")
            ])
        ], id="enhanced-task-modal", size="xl", is_open=False)

    def _create_optimization_modal(self) -> dbc.Modal:
        """Create optimization results modal

        Returns:
            Optimization modal
        """
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("AI Optimization Results")),
            dbc.ModalBody([
                html.Div(id="optimization-results-content")
            ]),
            dbc.ModalFooter([
                dbc.Button("Apply Optimization", id="apply-optimization", color="success"),
                dbc.Button("Close", id="close-optimization", color="secondary")
            ])
        ], id="optimization-modal", size="lg", is_open=False)

    def _create_inventory_order_modal(self) -> dbc.Modal:
        """Create inventory ordering modal

        Returns:
            Inventory order modal
        """
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Parts Inventory Manager")),
            dbc.ModalBody([
                html.Div(id="inventory-order-content")
            ]),
            dbc.ModalFooter([
                dbc.Button("Place Order", id="place-inventory-order", color="primary"),
                dbc.Button("Cancel", id="cancel-inventory-order", color="secondary")
            ])
        ], id="inventory-order-modal", size="lg", is_open=False)

    def _create_compliance_audit_modal(self) -> dbc.Modal:
        """Create compliance audit modal

        Returns:
            Compliance audit modal
        """
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Compliance Audit Manager")),
            dbc.ModalBody([
                html.Div(id="compliance-audit-content")
            ]),
            dbc.ModalFooter([
                dbc.Button("Generate Report", id="generate-compliance-report", color="info"),
                dbc.Button("Schedule Audit", id="schedule-compliance-audit", color="primary"),
                dbc.Button("Close", id="close-compliance-audit", color="secondary")
            ])
        ], id="compliance-audit-modal", size="lg", is_open=False)


# Enhanced callback functions for Phase 3.1 functionality
def register_enhanced_callbacks(app, data_service=None):
    """Register enhanced callbacks for Phase 3.1 features

    Args:
        app: Dash app instance
        data_service: Data service instance
    """

    @app.callback(
        Output("side-panel-content", "children"),
        Input("side-panel-tabs", "active_tab")
    )
    def update_side_panel(active_tab):
        """Update side panel content based on selected tab"""
        if active_tab == "task-details-tab":
            return html.Div([
                html.P("Select a task to view enhanced details", className="text-muted text-center"),
                html.I(className="fas fa-calendar-check fa-3x text-muted d-block text-center mt-4")
            ])
        elif active_tab == "resources-tab":
            return create_enhanced_resource_panel()
        elif active_tab == "inventory-tab":
            return create_inventory_status_panel()
        elif active_tab == "compliance-tab":
            return create_compliance_status_panel()

    @app.callback(
        Output("analytics-content", "children"),
        Input("analytics-tabs", "active_tab")
    )
    def update_analytics_content(active_tab):
        """Update analytics content based on selected tab"""
        if active_tab == "cost-trends":
            return create_cost_analytics_panel()
        elif active_tab == "resource-util":
            return create_resource_analytics_panel()
        elif active_tab == "compliance-metrics":
            return create_compliance_analytics_panel()
        elif active_tab == "performance-kpis":
            return create_performance_kpis_panel()


def create_enhanced_resource_panel():
    """Create enhanced resource management panel"""
    return html.Div([
        html.H6("AI Resource Allocation", className="mb-3"),

        # Technician optimization
        html.Div([
            html.Small("Optimized Technician Assignment", className="fw-bold d-block mb-2"),
            html.Div([
                html.Div([
                    html.Span("John Smith (Electrical)"),
                    dbc.Badge("95% Match", color="success", className="ms-auto")
                ], className="d-flex justify-content-between mb-1"),

                html.Div([
                    html.Span("Jane Doe (Electronics)"),
                    dbc.Badge("87% Match", color="info", className="ms-auto")
                ], className="d-flex justify-content-between mb-1"),

                html.Div([
                    html.Span("Bob Johnson (Mechanical)"),
                    dbc.Badge("72% Match", color="warning", className="ms-auto")
                ], className="d-flex justify-content-between mb-1")
            ])
        ]),

        html.Hr(),

        # Equipment availability
        html.Div([
            html.Small("Equipment Availability", className="fw-bold d-block mb-2"),
            html.Div([
                html.Div([
                    html.Span("EQ-001 (Critical)"),
                    daq.Indicator(value=True, color="green", size=10, className="ms-auto")
                ], className="d-flex justify-content-between align-items-center mb-1"),

                html.Div([
                    html.Span("EQ-002 (High)"),
                    daq.Indicator(value=False, color="red", size=10, className="ms-auto")
                ], className="d-flex justify-content-between align-items-center mb-1")
            ])
        ])
    ])


def create_inventory_status_panel():
    """Create inventory status panel"""
    return html.Div([
        html.H6("Smart Inventory Status", className="mb-3"),

        html.Div([
            html.Div([
                html.Span("High-Performance Bearing"),
                html.Div([
                    html.Small("25 units", className="me-2"),
                    dbc.Badge("In Stock", color="success")
                ])
            ], className="d-flex justify-content-between align-items-center mb-2"),

            html.Div([
                html.Span("Oil Filter Assembly"),
                html.Div([
                    html.Small("8 units", className="me-2"),
                    dbc.Badge("Low Stock", color="warning")
                ])
            ], className="d-flex justify-content-between align-items-center mb-2"),

            html.Div([
                html.Span("Temperature Sensor"),
                html.Div([
                    html.Small("12 units", className="me-2"),
                    dbc.Badge("Order Soon", color="info")
                ])
            ], className="d-flex justify-content-between align-items-center mb-2")
        ]),

        html.Hr(),

        dbc.Button([
            html.I(className="fas fa-shopping-cart me-2"),
            "AI-Suggested Orders"
        ], color="primary", size="sm", block=True)
    ])


def create_compliance_status_panel():
    """Create compliance status panel"""
    return html.Div([
        html.H6("Compliance Overview", className="mb-3"),

        html.Div([
            html.Div([
                html.Span("Safety Inspections"),
                dbc.Badge("3 Due", color="warning", className="ms-auto")
            ], className="d-flex justify-content-between align-items-center mb-2"),

            html.Div([
                html.Span("Regulatory Audits"),
                dbc.Badge("1 Overdue", color="danger", className="ms-auto")
            ], className="d-flex justify-content-between align-items-center mb-2"),

            html.Div([
                html.Span("Certifications"),
                dbc.Badge("5 Expiring", color="info", className="ms-auto")
            ], className="d-flex justify-content-between align-items-center mb-2")
        ]),

        html.Hr(),

        dbc.Button([
            html.I(className="fas fa-file-alt me-2"),
            "Generate Report"
        ], color="info", size="sm", block=True)
    ])


def create_cost_analytics_panel():
    """Create cost analytics panel"""
    return html.Div([
        dcc.Graph(
            figure=px.line(
                x=pd.date_range(start='2024-01-01', periods=12, freq='M'),
                y=np.random.uniform(20000, 30000, 12),
                title="Monthly Maintenance Costs with ML Predictions"
            )
        )
    ])


def create_resource_analytics_panel():
    """Create resource analytics panel"""
    return html.Div([
        dcc.Graph(
            figure=px.bar(
                x=['John', 'Jane', 'Bob', 'Alice'],
                y=[85, 92, 78, 88],
                title="Technician Utilization Rates"
            )
        )
    ])


def create_compliance_analytics_panel():
    """Create compliance analytics panel"""
    return html.Div([
        dcc.Graph(
            figure=px.pie(
                values=[94, 6],
                names=['Compliant', 'Non-Compliant'],
                title="Overall Compliance Status"
            )
        )
    ])


def create_performance_kpis_panel():
    """Create performance KPIs panel"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("94%", className="text-success"),
                html.Small("Schedule Efficiency")
            ], width=3),
            dbc.Col([
                html.H4("$2.4K", className="text-info"),
                html.Small("Cost Savings")
            ], width=3),
            dbc.Col([
                html.H4("87%", className="text-warning"),
                html.Small("Resource Utilization")
            ], width=3),
            dbc.Col([
                html.H4("99.2%", className="text-success"),
                html.Small("Compliance Rate")
            ], width=3)
        ])
    ])


# Create standalone function for import
def create_enhanced_layout():
    """Create enhanced maintenance scheduler page layout"""
    scheduler = EnhancedMaintenanceScheduler()
    return scheduler.create_enhanced_layout()