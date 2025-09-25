"""
Interactive Alert Action Manager
Provides comprehensive alert management with acknowledge, work order creation, and dismissal capabilities
"""

import numpy as np
import pandas as pd
from dash import html, dcc, callback, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import uuid
import json

# Import project modules
from src.alerts.alert_manager import AlertManager, Alert, AlertStatus, AlertSeverity, AlertType
from src.maintenance.work_order_manager import WorkOrderManager, WorkOrder, WorkOrderPriority, MaintenanceType
from src.anomaly_detection.nasa_anomaly_engine import AnomalyResult

logger = logging.getLogger(__name__)


@dataclass
class AlertAction:
    """Represents an action taken on an alert"""
    action_id: str
    alert_id: str
    action_type: str  # 'acknowledge', 'dismiss', 'escalate', 'create_work_order'
    user_id: str
    timestamp: datetime
    reason: Optional[str] = None
    notes: Optional[str] = None
    work_order_id: Optional[str] = None
    escalation_level: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertActionResult:
    """Result of an alert action"""
    success: bool
    message: str
    alert_id: str
    action_id: Optional[str] = None
    work_order_id: Optional[str] = None
    new_alert_status: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class InteractiveAlertActionManager:
    """
    Manages interactive alert actions including acknowledge, work order creation, and dismissal
    Integrates with existing AlertManager and WorkOrderManager systems
    """

    def __init__(self, alert_manager: AlertManager = None, work_order_manager: WorkOrderManager = None):
        """Initialize the Interactive Alert Action Manager"""
        # Initialize managers
        self.alert_manager = alert_manager or AlertManager()
        self.work_order_manager = work_order_manager or WorkOrderManager()

        # Action history tracking
        self.action_history = {}
        self.user_preferences = {}

        # Predefined dismissal reasons
        self.dismissal_reasons = [
            "False positive - sensor calibration issue",
            "Expected behavior during maintenance",
            "Environmental conditions causing temporary anomaly",
            "Software bug - to be fixed in next update",
            "Duplicate alert - already being handled",
            "Normal operational variance",
            "Equipment in testing phase",
            "Other (specify in notes)"
        ]

        # Work order templates for different equipment types
        self.work_order_templates = {
            'POWER': {
                'duration_hours': 4,
                'priority': WorkOrderPriority.HIGH,
                'required_skills': ['electrical', 'power_systems'],
                'estimated_cost': 500
            },
            'MOBILITY': {
                'duration_hours': 6,
                'priority': WorkOrderPriority.CRITICAL,
                'required_skills': ['mechanical', 'mobility_systems'],
                'estimated_cost': 800
            },
            'COMMUNICATION': {
                'duration_hours': 3,
                'priority': WorkOrderPriority.HIGH,
                'required_skills': ['rf_systems', 'communication'],
                'estimated_cost': 400
            },
            'DEFAULT': {
                'duration_hours': 2,
                'priority': WorkOrderPriority.MEDIUM,
                'required_skills': ['general_maintenance'],
                'estimated_cost': 300
            }
        }

        logger.info("Interactive Alert Action Manager initialized")

    def create_alert_action_interface(self, alert: Alert) -> html.Div:
        """
        Create interactive interface for alert actions

        Args:
            alert: Alert object to create interface for

        Returns:
            Dash HTML component with action interface
        """
        alert_id = alert.alert_id

        return html.Div([
            # Alert summary card
            self._create_alert_summary_card(alert),

            # Action buttons row
            self._create_action_buttons(alert_id),

            # Action modals
            self._create_acknowledge_modal(alert_id),
            self._create_dismiss_modal(alert_id),
            self._create_work_order_modal(alert_id, alert),
            self._create_escalate_modal(alert_id),

            # Action status area
            html.Div(id=f"action-status-{alert_id}", className="mt-3"),

            # Action history
            self._create_action_history_section(alert_id)
        ], id=f"alert-action-interface-{alert_id}")

    def _create_alert_summary_card(self, alert: Alert) -> dbc.Card:
        """Create alert summary card"""
        severity_color = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: 'info',
            AlertSeverity.LOW: 'light'
        }.get(alert.severity, 'secondary')

        status_color = {
            AlertStatus.NEW: 'primary',
            AlertStatus.ACKNOWLEDGED: 'warning',
            AlertStatus.IN_PROGRESS: 'info',
            AlertStatus.RESOLVED: 'success',
            AlertStatus.CLOSED: 'secondary'
        }.get(alert.status, 'light')

        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        alert.title
                    ], className="mb-0"),
                    html.Div([
                        dbc.Badge(alert.severity.name, color=severity_color, className="me-2"),
                        dbc.Badge(alert.status.value.upper(), color=status_color)
                    ])
                ], className="d-flex justify-content-between align-items-center")
            ]),
            dbc.CardBody([
                html.P(alert.description, className="mb-2"),
                html.Div([
                    html.Strong("Equipment: "), ", ".join(alert.affected_equipment), html.Br(),
                    html.Strong("Source: "), alert.source, html.Br(),
                    html.Strong("Created: "), alert.created_at.strftime("%Y-%m-%d %H:%M:%S"), html.Br(),
                    html.Strong("Alert ID: "), alert.alert_id
                ], className="text-muted small")
            ])
        ], className="mb-3")

    def _create_action_buttons(self, alert_id: str) -> dbc.ButtonGroup:
        """Create action buttons for alert"""
        return dbc.ButtonGroup([
            dbc.Button([
                html.I(className="fas fa-check me-2"),
                "Acknowledge"
            ], id=f"acknowledge-btn-{alert_id}", color="success", size="sm"),

            dbc.Button([
                html.I(className="fas fa-wrench me-2"),
                "Create Work Order"
            ], id=f"work-order-btn-{alert_id}", color="primary", size="sm"),

            dbc.Button([
                html.I(className="fas fa-times me-2"),
                "Dismiss"
            ], id=f"dismiss-btn-{alert_id}", color="warning", size="sm"),

            dbc.Button([
                html.I(className="fas fa-arrow-up me-2"),
                "Escalate"
            ], id=f"escalate-btn-{alert_id}", color="danger", size="sm")
        ], className="mb-3")

    def _create_acknowledge_modal(self, alert_id: str) -> dbc.Modal:
        """Create acknowledge modal"""
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Acknowledge Alert")),
            dbc.ModalBody([
                html.P("Please confirm acknowledgment of this alert:"),
                dbc.Label("Notes (optional):"),
                dbc.Textarea(
                    id=f"acknowledge-notes-{alert_id}",
                    placeholder="Add any notes about this acknowledgment...",
                    rows=3
                )
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel", id=f"acknowledge-cancel-{alert_id}",
                    color="secondary", className="me-2"
                ),
                dbc.Button(
                    "Acknowledge", id=f"acknowledge-confirm-{alert_id}",
                    color="success"
                )
            ])
        ], id=f"acknowledge-modal-{alert_id}", is_open=False)

    def _create_dismiss_modal(self, alert_id: str) -> dbc.Modal:
        """Create dismiss modal"""
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Dismiss Alert")),
            dbc.ModalBody([
                html.P("Please select a reason for dismissing this alert:"),
                dbc.Label("Dismissal Reason:"),
                dcc.Dropdown(
                    id=f"dismiss-reason-{alert_id}",
                    options=[{"label": reason, "value": reason} for reason in self.dismissal_reasons],
                    placeholder="Select a reason...",
                    className="mb-3"
                ),
                dbc.Label("Additional Notes:"),
                dbc.Textarea(
                    id=f"dismiss-notes-{alert_id}",
                    placeholder="Provide additional details...",
                    rows=3
                )
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel", id=f"dismiss-cancel-{alert_id}",
                    color="secondary", className="me-2"
                ),
                dbc.Button(
                    "Dismiss", id=f"dismiss-confirm-{alert_id}",
                    color="warning"
                )
            ])
        ], id=f"dismiss-modal-{alert_id}", is_open=False)

    def _create_work_order_modal(self, alert_id: str, alert: Alert) -> dbc.Modal:
        """Create work order creation modal"""
        # Get equipment subsystem for template selection
        equipment_subsystem = self._determine_equipment_subsystem(alert.affected_equipment)
        template = self.work_order_templates.get(equipment_subsystem, self.work_order_templates['DEFAULT'])

        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Create Work Order")),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Work Order Type:"),
                        dcc.Dropdown(
                            id=f"wo-type-{alert_id}",
                            options=[
                                {"label": "Corrective Maintenance", "value": "CORRECTIVE"},
                                {"label": "Emergency Repair", "value": "EMERGENCY"},
                                {"label": "Inspection", "value": "INSPECTION"},
                                {"label": "Preventive Maintenance", "value": "PREVENTIVE"}
                            ],
                            value="CORRECTIVE"
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Priority:"),
                        dcc.Dropdown(
                            id=f"wo-priority-{alert_id}",
                            options=[
                                {"label": "Critical", "value": "CRITICAL"},
                                {"label": "High", "value": "HIGH"},
                                {"label": "Medium", "value": "MEDIUM"},
                                {"label": "Low", "value": "LOW"}
                            ],
                            value=template['priority'].name
                        )
                    ], width=6)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Label("Estimated Duration (hours):"),
                        dbc.Input(
                            id=f"wo-duration-{alert_id}",
                            type="number",
                            value=template['duration_hours'],
                            min=0.5,
                            step=0.5
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Estimated Cost ($):"),
                        dbc.Input(
                            id=f"wo-cost-{alert_id}",
                            type="number",
                            value=template['estimated_cost'],
                            min=0,
                            step=50
                        )
                    ], width=6)
                ], className="mb-3"),

                dbc.Label("Description:"),
                dbc.Textarea(
                    id=f"wo-description-{alert_id}",
                    value=f"Work order created from alert: {alert.title}\n\nEquipment: {', '.join(alert.affected_equipment)}\nAlert Details: {alert.description}",
                    rows=4,
                    className="mb-3"
                ),

                dbc.Label("Required Skills:"),
                dcc.Dropdown(
                    id=f"wo-skills-{alert_id}",
                    options=[
                        {"label": "Electrical Systems", "value": "electrical"},
                        {"label": "Mechanical Systems", "value": "mechanical"},
                        {"label": "Power Systems", "value": "power_systems"},
                        {"label": "Mobility Systems", "value": "mobility_systems"},
                        {"label": "Communication Systems", "value": "communication"},
                        {"label": "RF Systems", "value": "rf_systems"},
                        {"label": "General Maintenance", "value": "general_maintenance"}
                    ],
                    value=template['required_skills'],
                    multi=True
                )
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel", id=f"wo-cancel-{alert_id}",
                    color="secondary", className="me-2"
                ),
                dbc.Button(
                    "Create Work Order", id=f"wo-confirm-{alert_id}",
                    color="primary"
                )
            ])
        ], id=f"work-order-modal-{alert_id}", is_open=False, size="lg")

    def _create_escalate_modal(self, alert_id: str) -> dbc.Modal:
        """Create escalate modal"""
        return dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Escalate Alert")),
            dbc.ModalBody([
                html.P("Escalate this alert to a higher level of attention:"),
                dbc.Label("Escalation Reason:"),
                dcc.Dropdown(
                    id=f"escalate-reason-{alert_id}",
                    options=[
                        {"label": "No response within SLA", "value": "sla_breach"},
                        {"label": "Severity increased", "value": "severity_increase"},
                        {"label": "Multiple related failures", "value": "pattern_detected"},
                        {"label": "Critical equipment impact", "value": "critical_impact"},
                        {"label": "Safety concern", "value": "safety_concern"},
                        {"label": "Other", "value": "other"}
                    ],
                    placeholder="Select escalation reason..."
                ),
                dbc.Label("Additional Details:", className="mt-3"),
                dbc.Textarea(
                    id=f"escalate-notes-{alert_id}",
                    placeholder="Provide details for escalation...",
                    rows=3
                )
            ]),
            dbc.ModalFooter([
                dbc.Button(
                    "Cancel", id=f"escalate-cancel-{alert_id}",
                    color="secondary", className="me-2"
                ),
                dbc.Button(
                    "Escalate", id=f"escalate-confirm-{alert_id}",
                    color="danger"
                )
            ])
        ], id=f"escalate-modal-{alert_id}", is_open=False)

    def _create_action_history_section(self, alert_id: str) -> dbc.Card:
        """Create action history section"""
        return dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="fas fa-history me-2"),
                    "Action History"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id=f"action-history-content-{alert_id}")
            ])
        ], className="mt-3")

    def acknowledge_alert(self, alert_id: str, user_id: str, notes: Optional[str] = None) -> AlertActionResult:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert identifier
            user_id: User performing the action
            notes: Optional notes

        Returns:
            Result of the acknowledgment action
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return AlertActionResult(
                    success=False,
                    message=f"Alert {alert_id} not found",
                    alert_id=alert_id,
                    errors=["Alert not found"]
                )

            # Check if alert can be acknowledged
            if alert.status == AlertStatus.ACKNOWLEDGED:
                return AlertActionResult(
                    success=False,
                    message="Alert is already acknowledged",
                    alert_id=alert_id,
                    errors=["Already acknowledged"]
                )

            # Create action record
            action = AlertAction(
                action_id=str(uuid.uuid4()),
                alert_id=alert_id,
                action_type='acknowledge',
                user_id=user_id,
                timestamp=datetime.now(),
                notes=notes
            )

            # Update alert status
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = user_id
            alert.acknowledged_at = datetime.now()
            alert.updated_at = datetime.now()

            # Save action to history
            self._save_action_to_history(action)

            # Update alert in manager
            self.alert_manager.update_alert(alert)

            logger.info(f"Alert {alert_id} acknowledged by user {user_id}")

            return AlertActionResult(
                success=True,
                message="Alert acknowledged successfully",
                alert_id=alert_id,
                action_id=action.action_id,
                new_alert_status=AlertStatus.ACKNOWLEDGED.value
            )

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return AlertActionResult(
                success=False,
                message=f"Error acknowledging alert: {str(e)}",
                alert_id=alert_id,
                errors=[str(e)]
            )

    def dismiss_alert(self, alert_id: str, user_id: str, reason: str,
                     notes: Optional[str] = None) -> AlertActionResult:
        """
        Dismiss an alert

        Args:
            alert_id: Alert identifier
            user_id: User performing the action
            reason: Reason for dismissal
            notes: Optional additional notes

        Returns:
            Result of the dismissal action
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return AlertActionResult(
                    success=False,
                    message=f"Alert {alert_id} not found",
                    alert_id=alert_id,
                    errors=["Alert not found"]
                )

            # Create action record
            action = AlertAction(
                action_id=str(uuid.uuid4()),
                alert_id=alert_id,
                action_type='dismiss',
                user_id=user_id,
                timestamp=datetime.now(),
                reason=reason,
                notes=notes
            )

            # Update alert status
            alert.status = AlertStatus.CLOSED
            alert.resolved_by = user_id
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()
            alert.suppressed = True
            alert.suppression_reason = reason

            # Save action to history
            self._save_action_to_history(action)

            # Update alert in manager
            self.alert_manager.update_alert(alert)

            logger.info(f"Alert {alert_id} dismissed by user {user_id} - Reason: {reason}")

            return AlertActionResult(
                success=True,
                message="Alert dismissed successfully",
                alert_id=alert_id,
                action_id=action.action_id,
                new_alert_status=AlertStatus.CLOSED.value
            )

        except Exception as e:
            logger.error(f"Error dismissing alert {alert_id}: {e}")
            return AlertActionResult(
                success=False,
                message=f"Error dismissing alert: {str(e)}",
                alert_id=alert_id,
                errors=[str(e)]
            )

    def create_work_order_from_alert(self, alert_id: str, user_id: str,
                                   work_order_params: Dict[str, Any]) -> AlertActionResult:
        """
        Create a work order from an alert

        Args:
            alert_id: Alert identifier
            user_id: User creating the work order
            work_order_params: Work order parameters

        Returns:
            Result of the work order creation
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return AlertActionResult(
                    success=False,
                    message=f"Alert {alert_id} not found",
                    alert_id=alert_id,
                    errors=["Alert not found"]
                )

            # Create work order
            work_order_id = str(uuid.uuid4())

            work_order = WorkOrder(
                order_id=work_order_id,
                equipment_id=alert.affected_equipment[0] if alert.affected_equipment else "UNKNOWN",
                anomaly_id=alert_id,
                type=MaintenanceType(work_order_params.get('type', 'CORRECTIVE')),
                priority=WorkOrderPriority[work_order_params.get('priority', 'MEDIUM')],
                status=WorkOrderStatus.CREATED,
                created_at=datetime.now(),
                description=work_order_params.get('description', ''),
                estimated_duration_hours=work_order_params.get('duration_hours', 2.0),
                estimated_cost=work_order_params.get('cost', 300.0),
                anomaly_details={
                    'alert_id': alert_id,
                    'alert_title': alert.title,
                    'alert_severity': alert.severity.name,
                    'affected_equipment': alert.affected_equipment
                }
            )

            # Add work order to manager
            self.work_order_manager.add_work_order(work_order)

            # Create action record
            action = AlertAction(
                action_id=str(uuid.uuid4()),
                alert_id=alert_id,
                action_type='create_work_order',
                user_id=user_id,
                timestamp=datetime.now(),
                work_order_id=work_order_id,
                metadata=work_order_params
            )

            # Update alert with work order reference
            alert.work_order_id = work_order_id
            alert.status = AlertStatus.IN_PROGRESS
            alert.updated_at = datetime.now()

            # Save action to history
            self._save_action_to_history(action)

            # Update alert in manager
            self.alert_manager.update_alert(alert)

            logger.info(f"Work order {work_order_id} created from alert {alert_id} by user {user_id}")

            return AlertActionResult(
                success=True,
                message="Work order created successfully",
                alert_id=alert_id,
                action_id=action.action_id,
                work_order_id=work_order_id,
                new_alert_status=AlertStatus.IN_PROGRESS.value
            )

        except Exception as e:
            logger.error(f"Error creating work order from alert {alert_id}: {e}")
            return AlertActionResult(
                success=False,
                message=f"Error creating work order: {str(e)}",
                alert_id=alert_id,
                errors=[str(e)]
            )

    def escalate_alert(self, alert_id: str, user_id: str, reason: str,
                      notes: Optional[str] = None) -> AlertActionResult:
        """
        Escalate an alert

        Args:
            alert_id: Alert identifier
            user_id: User performing the escalation
            reason: Reason for escalation
            notes: Optional additional notes

        Returns:
            Result of the escalation action
        """
        try:
            # Get the alert
            alert = self.alert_manager.get_alert(alert_id)
            if not alert:
                return AlertActionResult(
                    success=False,
                    message=f"Alert {alert_id} not found",
                    alert_id=alert_id,
                    errors=["Alert not found"]
                )

            # Increase escalation level
            new_escalation_level = alert.escalation_level + 1

            # Create action record
            action = AlertAction(
                action_id=str(uuid.uuid4()),
                alert_id=alert_id,
                action_type='escalate',
                user_id=user_id,
                timestamp=datetime.now(),
                reason=reason,
                notes=notes,
                escalation_level=new_escalation_level
            )

            # Update alert
            alert.escalation_level = new_escalation_level
            alert.status = AlertStatus.ESCALATED
            alert.updated_at = datetime.now()

            # Save action to history
            self._save_action_to_history(action)

            # Update alert in manager
            self.alert_manager.update_alert(alert)

            logger.info(f"Alert {alert_id} escalated to level {new_escalation_level} by user {user_id}")

            return AlertActionResult(
                success=True,
                message=f"Alert escalated to level {new_escalation_level}",
                alert_id=alert_id,
                action_id=action.action_id,
                new_alert_status=AlertStatus.ESCALATED.value
            )

        except Exception as e:
            logger.error(f"Error escalating alert {alert_id}: {e}")
            return AlertActionResult(
                success=False,
                message=f"Error escalating alert: {str(e)}",
                alert_id=alert_id,
                errors=[str(e)]
            )

    def get_action_history(self, alert_id: str) -> List[AlertAction]:
        """Get action history for an alert"""
        return self.action_history.get(alert_id, [])

    def create_action_history_display(self, alert_id: str) -> html.Div:
        """Create action history display"""
        actions = self.get_action_history(alert_id)

        if not actions:
            return html.P("No actions taken yet.", className="text-muted")

        action_items = []
        for action in sorted(actions, key=lambda x: x.timestamp, reverse=True):
            icon = {
                'acknowledge': 'fas fa-check text-success',
                'dismiss': 'fas fa-times text-warning',
                'create_work_order': 'fas fa-wrench text-primary',
                'escalate': 'fas fa-arrow-up text-danger'
            }.get(action.action_type, 'fas fa-info')

            action_items.append(
                html.Div([
                    html.I(className=f"{icon} me-2"),
                    html.Strong(action.action_type.replace('_', ' ').title()),
                    html.Small(f" by {action.user_id} at {action.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                              className="text-muted ms-2"),
                    html.Br(),
                    html.Small(action.reason if action.reason else action.notes, className="text-muted")
                ], className="mb-2")
            )

        return html.Div(action_items)

    # Helper methods
    def _save_action_to_history(self, action: AlertAction):
        """Save action to history"""
        if action.alert_id not in self.action_history:
            self.action_history[action.alert_id] = []
        self.action_history[action.alert_id].append(action)

    def _determine_equipment_subsystem(self, affected_equipment: List[str]) -> str:
        """Determine subsystem from affected equipment"""
        if not affected_equipment:
            return 'DEFAULT'

        equipment_id = affected_equipment[0]
        if 'PWR' in equipment_id:
            return 'POWER'
        elif 'MOB' in equipment_id:
            return 'MOBILITY'
        elif 'COM' in equipment_id:
            return 'COMMUNICATION'
        else:
            return 'DEFAULT'


# Global instance for dashboard integration
alert_action_manager = InteractiveAlertActionManager()