"""
Work Order Manager Module for IoT Anomaly Detection System
Automated work order creation, tracking, and management for maintenance tasks
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pulp
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    cp_model = None

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class WorkOrderPriority(Enum):
    """Work order priority levels"""
    CRITICAL = 1  # System down, safety risk
    HIGH = 2      # Major impact, needs urgent attention
    MEDIUM = 3    # Moderate impact, schedule soon
    LOW = 4       # Minor issue, can be scheduled
    PREVENTIVE = 5  # Routine maintenance


class WorkOrderStatus(Enum):
    """Work order status states"""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    VERIFIED = "verified"


class MaintenanceType(Enum):
    """Types of maintenance work"""
    CORRECTIVE = "corrective"      # Fix failures
    PREVENTIVE = "preventive"      # Scheduled maintenance
    PREDICTIVE = "predictive"      # Based on condition monitoring
    EMERGENCY = "emergency"        # Immediate action required
    INSPECTION = "inspection"      # Regular inspection


@dataclass
class Equipment:
    """Equipment information"""
    equipment_id: str
    name: str
    type: str
    location: str
    criticality: str  # 'critical', 'high', 'medium', 'low'
    installation_date: datetime
    last_maintenance: Optional[datetime] = None
    maintenance_schedule: Optional[Dict[str, Any]] = None
    specifications: Dict[str, Any] = field(default_factory=dict)
    sensors: List[str] = field(default_factory=list)
    
    def get_age_days(self) -> int:
        """Get equipment age in days"""
        return (datetime.now() - self.installation_date).days
        
    def is_maintenance_due(self) -> bool:
        """Check if maintenance is due"""
        if not self.maintenance_schedule or not self.last_maintenance:
            return False
            
        interval_days = self.maintenance_schedule.get('interval_days', 90)
        days_since_maintenance = (datetime.now() - self.last_maintenance).days
        
        return days_since_maintenance >= interval_days


@dataclass
class Technician:
    """Technician information"""
    technician_id: str
    name: str
    skills: List[str]
    availability: Dict[str, List[Tuple[datetime, datetime]]]  # Day -> time slots
    current_workload: int
    max_daily_orders: int = 8
    location: str = ""
    certification_level: int = 1  # 1-5, higher is more skilled
    hourly_rate: float = 50.0
    
    def is_available(self, start_time: datetime, duration_hours: int) -> bool:
        """Check if technician is available for given time"""
        end_time = start_time + timedelta(hours=duration_hours)
        day = start_time.strftime('%A')
        
        if day not in self.availability:
            return False
            
        for slot_start, slot_end in self.availability[day]:
            if slot_start <= start_time and slot_end >= end_time:
                return True
                
        return False
        
    def has_skill(self, required_skill: str) -> bool:
        """Check if technician has required skill"""
        return required_skill.lower() in [s.lower() for s in self.skills]


@dataclass
class WorkOrder:
    """Work order details"""
    order_id: str
    equipment_id: str
    anomaly_id: Optional[str]
    type: MaintenanceType
    priority: WorkOrderPriority
    status: WorkOrderStatus
    created_at: datetime
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    assigned_technician: Optional[str] = None
    description: str = ""
    root_cause: Optional[str] = None
    actions_taken: List[str] = field(default_factory=list)
    parts_used: List[Dict[str, Any]] = field(default_factory=list)
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_duration_hours: float = 2.0
    actual_duration_hours: Optional[float] = None
    sla_deadline: Optional[datetime] = None
    completion_notes: str = ""
    verification_status: Optional[str] = None
    anomaly_details: Optional[Dict[str, Any]] = None
    
    def is_overdue(self) -> bool:
        """Check if work order is overdue"""
        if self.sla_deadline and self.status not in [WorkOrderStatus.COMPLETED, WorkOrderStatus.CANCELLED]:
            return datetime.now() > self.sla_deadline
        return False
        
    def get_duration(self) -> float:
        """Get actual or estimated duration"""
        if self.actual_duration_hours:
            return self.actual_duration_hours
        return self.estimated_duration_hours
        
    def calculate_actual_cost(self) -> float:
        """Calculate actual cost based on labor and parts"""
        labor_cost = self.actual_duration_hours * 50 if self.actual_duration_hours else 0
        parts_cost = sum(part.get('cost', 0) for part in self.parts_used)
        return labor_cost + parts_cost


class WorkOrderManager:
    """Main work order management system with Phase 3.2 automation integration"""

    def __init__(self,
                 database_config: Optional[Dict] = None,
                 notification_config: Optional[Dict] = None,
                 phase32_config: Optional[Dict] = None):
        """Initialize Work Order Manager with Phase 3.2 automation

        Args:
            database_config: Database configuration
            notification_config: Notification system configuration
            phase32_config: Phase 3.2 automation configuration
        """
        self.work_orders = {}
        self.equipment_registry = {}
        self.technician_registry = {}
        self.order_queue = queue.PriorityQueue()
        self.assignment_engine = AssignmentEngine()
        self.scheduler = MaintenanceScheduler()
        self.cost_estimator = CostEstimator()
        self.notification_system = NotificationManager(notification_config)

        # Phase 3.2 Automation Components
        self.phase32_config = phase32_config or {}
        self.automation_enabled = self.phase32_config.get('automation_enabled', True)

        # Initialize Phase 3.2 components (will be set after initialization)
        self.automated_creator = None
        self.resource_allocator = None
        self.lifecycle_tracker = None
        self.performance_analyzer = None

        # Phase 3.1 Integration
        self.phase31_integration = None

        # Automation metrics
        self.automation_metrics = {
            'automated_work_orders_created': 0,
            'automated_assignments_made': 0,
            'automation_success_rate': 0.0,
            'manual_interventions': 0,
            'automation_time_savings': 0.0
        }

        # Metrics tracking (enhanced)
        self.metrics = {
            'total_orders': 0,
            'completed_orders': 0,
            'average_completion_time': 0,
            'sla_compliance_rate': 0,
            'total_cost': 0,
            'automation_efficiency': 0,
            'workload_balance_score': 0,
            'technician_utilization': 0
        }

        # Background executor for async tasks
        self.executor = ThreadPoolExecutor(max_workers=6)  # Increased for automation

        logger.info("Initialized Work Order Manager with Phase 3.2 automation support")

    def initialize_phase32_automation(self,
                                     automated_creator=None,
                                     resource_allocator=None,
                                     lifecycle_tracker=None,
                                     performance_analyzer=None,
                                     phase31_integration=None):
        """Initialize Phase 3.2 automation components

        Args:
            automated_creator: AutomatedWorkOrderCreator instance
            resource_allocator: OptimizedResourceAllocator instance
            lifecycle_tracker: WorkOrderLifecycleTracker instance
            performance_analyzer: TechnicianPerformanceAnalyzer instance
            phase31_integration: Phase 3.1 integration instance
        """
        self.automated_creator = automated_creator
        self.resource_allocator = resource_allocator
        self.lifecycle_tracker = lifecycle_tracker
        self.performance_analyzer = performance_analyzer
        self.phase31_integration = phase31_integration

        logger.info("Initialized Phase 3.2 automation components")

    def create_work_order_from_anomaly_automated(self,
                                                anomaly_data: Dict[str, Any],
                                                equipment_id: str,
                                                force_creation: bool = False) -> Optional[WorkOrder]:
        """Create work order from anomaly using Phase 3.2 automation

        Args:
            anomaly_data: Anomaly detection results
            equipment_id: Equipment identifier
            force_creation: Force creation even if automation criteria not met

        Returns:
            Created work order or None
        """
        try:
            if not self.automation_enabled or not self.automated_creator:
                # Fall back to manual creation
                return self.create_work_order_from_anomaly(anomaly_data, equipment_id, auto_assign=True)

            # Use automated creator
            work_order = self.automated_creator.process_anomaly_for_work_order(
                anomaly_data=anomaly_data,
                equipment_id=equipment_id,
                auto_assign=True,
                force_creation=force_creation
            )

            if work_order:
                # Start lifecycle tracking
                if self.lifecycle_tracker:
                    self.lifecycle_tracker.start_work_order_tracking(work_order)

                # Update automation metrics
                self.automation_metrics['automated_work_orders_created'] += 1

                logger.info(f"Automated work order creation successful: {work_order.order_id}")

            return work_order

        except Exception as e:
            logger.error(f"Error in automated work order creation: {e}")
            # Fall back to manual creation
            self.automation_metrics['manual_interventions'] += 1
            return self.create_work_order_from_anomaly(anomaly_data, equipment_id, auto_assign=True)

    def assign_technician_automated(self, order_id: str, algorithm: str = 'hybrid') -> Optional[str]:
        """Assign technician using Phase 3.2 automated resource allocation

        Args:
            order_id: Work order ID
            algorithm: Assignment algorithm to use

        Returns:
            Assigned technician ID or None
        """
        try:
            if not self.automation_enabled or not self.resource_allocator:
                # Fall back to manual assignment
                return self.assign_technician(order_id)

            work_order = self.work_orders.get(order_id)
            if not work_order:
                logger.error(f"Work order {order_id} not found")
                return None

            # Use automated resource allocator
            assignment_solution = self.resource_allocator.allocate_technician_to_work_order(
                work_order=work_order,
                algorithm=algorithm,
                consider_cost=True
            )

            if assignment_solution and order_id in assignment_solution.assignments:
                technician_id = assignment_solution.assignments[order_id]

                # Track lifecycle event
                if self.lifecycle_tracker:
                    self.lifecycle_tracker.track_work_order_event(
                        work_order_id=order_id,
                        event_type='assigned',
                        new_status=WorkOrderStatus.ASSIGNED,
                        performer='automation_system',
                        automated=True,
                        event_data={
                            'assignment_algorithm': algorithm,
                            'assignment_confidence': assignment_solution.confidence_level,
                            'skill_match_score': assignment_solution.skill_match_score
                        }
                    )

                # Update automation metrics
                self.automation_metrics['automated_assignments_made'] += 1

                logger.info(f"Automated technician assignment successful: {technician_id} -> {order_id}")
                return technician_id

            else:
                # Fall back to manual assignment
                self.automation_metrics['manual_interventions'] += 1
                return self.assign_technician(order_id)

        except Exception as e:
            logger.error(f"Error in automated technician assignment: {e}")
            # Fall back to manual assignment
            self.automation_metrics['manual_interventions'] += 1
            return self.assign_technician(order_id)

    def update_work_order_status_with_tracking(self,
                                             order_id: str,
                                             new_status: WorkOrderStatus,
                                             performer: Optional[str] = None,
                                             notes: str = "") -> bool:
        """Update work order status with lifecycle tracking

        Args:
            order_id: Work order ID
            new_status: New status
            performer: Who performed the action
            notes: Additional notes

        Returns:
            Success flag
        """
        try:
            # Update status using original method
            success = self.update_work_order_status(order_id, new_status, notes)

            if success and self.lifecycle_tracker:
                # Track lifecycle event
                event_type = new_status.value
                self.lifecycle_tracker.track_work_order_event(
                    work_order_id=order_id,
                    event_type=event_type,
                    new_status=new_status,
                    performer=performer,
                    automated=False,
                    notes=notes
                )

                # Complete tracking if work order is completed
                if new_status == WorkOrderStatus.COMPLETED:
                    self.lifecycle_tracker.complete_work_order_tracking(order_id)

                    # Update performance analytics
                    if self.performance_analyzer and performer:
                        # Trigger performance analysis update
                        self.performance_analyzer.analyze_technician_performance(performer)

            return success

        except Exception as e:
            logger.error(f"Error updating work order status with tracking: {e}")
            return False

    def get_workload_balance_analysis(self) -> Optional[Dict[str, Any]]:
        """Get current workload balance analysis

        Returns:
            Workload analysis or None
        """
        try:
            if not self.performance_analyzer:
                logger.warning("Performance analyzer not initialized")
                return None

            analysis = self.performance_analyzer.analyze_workload_balance()
            return analysis.__dict__ if analysis else None

        except Exception as e:
            logger.error(f"Error getting workload balance analysis: {e}")
            return None

    def get_technician_performance_report(self, technician_id: str) -> Optional[Dict[str, Any]]:
        """Get performance report for technician

        Args:
            technician_id: Technician identifier

        Returns:
            Performance report or None
        """
        try:
            if not self.performance_analyzer:
                logger.warning("Performance analyzer not initialized")
                return None

            metrics = self.performance_analyzer.analyze_technician_performance(technician_id)
            return metrics.__dict__ if metrics else None

        except Exception as e:
            logger.error(f"Error getting technician performance report: {e}")
            return None

    def get_automation_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive automation dashboard data

        Returns:
            Dashboard data
        """
        try:
            dashboard_data = {
                'automation_metrics': self.automation_metrics.copy(),
                'enhanced_metrics': self.get_enhanced_metrics(),
                'workload_analysis': self.get_workload_balance_analysis(),
                'recent_automated_orders': self._get_recent_automated_orders(),
                'automation_efficiency_trends': self._get_automation_efficiency_trends(),
                'system_health': self._get_automation_system_health()
            }

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting automation dashboard data: {e}")
            return {}

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including automation performance

        Returns:
            Enhanced metrics
        """
        try:
            # Calculate automation success rate
            total_automated = (self.automation_metrics['automated_work_orders_created'] +
                             self.automation_metrics['automated_assignments_made'])
            total_attempts = total_automated + self.automation_metrics['manual_interventions']

            if total_attempts > 0:
                self.automation_metrics['automation_success_rate'] = total_automated / total_attempts
            else:
                self.automation_metrics['automation_success_rate'] = 0.0

            # Update enhanced metrics
            enhanced_metrics = self.metrics.copy()

            # Add automation efficiency
            if self.automated_creator:
                creator_metrics = self.automated_creator.get_automation_metrics()
                enhanced_metrics['automation_efficiency'] = creator_metrics.get('automation_success_rate', 0)

            # Add workload balance score
            if self.performance_analyzer:
                analyzer_metrics = self.performance_analyzer.get_analyzer_metrics()
                enhanced_metrics['workload_balance_score'] = analyzer_metrics.get('workload_balance_score', 0)

            # Add technician utilization
            if self.technician_registry:
                total_utilization = sum(
                    tech.current_workload / tech.max_daily_orders
                    for tech in self.technician_registry.values()
                )
                enhanced_metrics['technician_utilization'] = total_utilization / len(self.technician_registry)

            return enhanced_metrics

        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {e}")
            return self.metrics.copy()

    def _get_recent_automated_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent automated work orders

        Args:
            limit: Maximum number of orders to return

        Returns:
            List of recent automated orders
        """
        try:
            automated_orders = []
            for order_id, work_order in self.work_orders.items():
                if hasattr(work_order, 'automation_metadata') and work_order.automation_metadata:
                    if work_order.automation_metadata.get('created_by_automation'):
                        automated_orders.append({
                            'order_id': order_id,
                            'equipment_id': work_order.equipment_id,
                            'priority': work_order.priority.name,
                            'status': work_order.status.value,
                            'created_at': work_order.created_at.isoformat(),
                            'automation_confidence': work_order.automation_metadata.get('automation_confidence', 0)
                        })

            # Sort by creation time (most recent first)
            automated_orders.sort(key=lambda x: x['created_at'], reverse=True)
            return automated_orders[:limit]

        except Exception as e:
            logger.error(f"Error getting recent automated orders: {e}")
            return []

    def _get_automation_efficiency_trends(self) -> Dict[str, Any]:
        """Get automation efficiency trends

        Returns:
            Efficiency trends data
        """
        try:
            # Placeholder implementation
            trends = {
                'efficiency_over_time': [],
                'success_rate_trend': 'stable',
                'cost_savings_trend': 'improving',
                'time_savings_trend': 'improving'
            }

            return trends

        except Exception as e:
            logger.error(f"Error calculating automation efficiency trends: {e}")
            return {}

    def _get_automation_system_health(self) -> Dict[str, Any]:
        """Get automation system health status

        Returns:
            System health data
        """
        try:
            health = {
                'automated_creator_status': 'operational' if self.automated_creator else 'not_initialized',
                'resource_allocator_status': 'operational' if self.resource_allocator else 'not_initialized',
                'lifecycle_tracker_status': 'operational' if self.lifecycle_tracker else 'not_initialized',
                'performance_analyzer_status': 'operational' if self.performance_analyzer else 'not_initialized',
                'phase31_integration_status': 'operational' if self.phase31_integration else 'not_initialized',
                'overall_health': 'healthy' if all([
                    self.automated_creator,
                    self.resource_allocator,
                    self.lifecycle_tracker,
                    self.performance_analyzer
                ]) else 'partial'
            }

            return health

        except Exception as e:
            logger.error(f"Error getting automation system health: {e}")
            return {'overall_health': 'error'}

    def create_work_order_from_anomaly(self,
                                      anomaly_data: Dict[str, Any],
                                      equipment_id: str,
                                      auto_assign: bool = True) -> WorkOrder:
        """Create work order from detected anomaly
        
        Args:
            anomaly_data: Anomaly detection results
            equipment_id: Equipment identifier
            auto_assign: Automatically assign technician
            
        Returns:
            Created work order
        """
        # Determine priority based on anomaly severity
        priority = self._determine_priority(anomaly_data)
        
        # Determine maintenance type
        maintenance_type = self._determine_maintenance_type(anomaly_data)
        
        # Create work order
        work_order = WorkOrder(
            order_id=self._generate_order_id(),
            equipment_id=equipment_id,
            anomaly_id=anomaly_data.get('anomaly_id'),
            type=maintenance_type,
            priority=priority,
            status=WorkOrderStatus.CREATED,
            created_at=datetime.now(),
            description=self._generate_description(anomaly_data),
            estimated_duration_hours=self._estimate_duration(anomaly_data),
            anomaly_details=anomaly_data
        )
        
        # Set SLA deadline based on priority
        work_order.sla_deadline = self._calculate_sla_deadline(priority)
        
        # Estimate cost
        work_order.estimated_cost = self.cost_estimator.estimate_cost(work_order)
        
        # Store work order
        self.work_orders[work_order.order_id] = work_order
        self.metrics['total_orders'] += 1
        
        # Add to queue
        self.order_queue.put((priority.value, work_order.order_id))
        
        # Auto-assign if requested
        if auto_assign:
            self.assign_technician(work_order.order_id)
            
        # Send notification
        self.notification_system.send_work_order_created(work_order)
        
        logger.info(f"Created work order {work_order.order_id} for equipment {equipment_id}")
        
        return work_order
        
    def _determine_priority(self, anomaly_data: Dict[str, Any]) -> WorkOrderPriority:
        """Determine work order priority from anomaly data"""
        severity = anomaly_data.get('severity', 'medium')
        confidence = anomaly_data.get('confidence', 0.5)
        
        if severity == 'critical' or confidence > 0.95:
            return WorkOrderPriority.CRITICAL
        elif severity == 'high' or confidence > 0.8:
            return WorkOrderPriority.HIGH
        elif severity == 'medium' or confidence > 0.6:
            return WorkOrderPriority.MEDIUM
        else:
            return WorkOrderPriority.LOW
            
    def _determine_maintenance_type(self, anomaly_data: Dict[str, Any]) -> MaintenanceType:
        """Determine maintenance type from anomaly data"""
        anomaly_type = anomaly_data.get('type', 'unknown')
        
        if anomaly_type == 'failure':
            return MaintenanceType.EMERGENCY
        elif anomaly_type == 'degradation':
            return MaintenanceType.PREDICTIVE
        elif anomaly_type == 'scheduled':
            return MaintenanceType.PREVENTIVE
        else:
            return MaintenanceType.CORRECTIVE
            
    def _generate_order_id(self) -> str:
        """Generate unique work order ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"WO-{timestamp}-{unique_id}"
        
    def _generate_description(self, anomaly_data: Dict[str, Any]) -> str:
        """Generate work order description from anomaly data"""
        anomaly_type = anomaly_data.get('type', 'Unknown anomaly')
        sensor = anomaly_data.get('sensor', 'Unknown sensor')
        value = anomaly_data.get('value', 'N/A')
        threshold = anomaly_data.get('threshold', 'N/A')
        
        description = f"Anomaly detected: {anomaly_type} on {sensor}. "
        description += f"Value: {value}, Threshold: {threshold}. "
        description += f"Timestamp: {anomaly_data.get('timestamp', 'Unknown')}"
        
        return description
        
    def _estimate_duration(self, anomaly_data: Dict[str, Any]) -> float:
        """Estimate work duration based on anomaly type"""
        severity = anomaly_data.get('severity', 'medium')
        
        duration_map = {
            'critical': 4.0,
            'high': 3.0,
            'medium': 2.0,
            'low': 1.0
        }
        
        return duration_map.get(severity, 2.0)
        
    def _calculate_sla_deadline(self, priority: WorkOrderPriority) -> datetime:
        """Calculate SLA deadline based on priority"""
        deadline_hours = {
            WorkOrderPriority.CRITICAL: 4,
            WorkOrderPriority.HIGH: 24,
            WorkOrderPriority.MEDIUM: 72,
            WorkOrderPriority.LOW: 168,  # 1 week
            WorkOrderPriority.PREVENTIVE: 336  # 2 weeks
        }
        
        hours = deadline_hours.get(priority, 72)
        return datetime.now() + timedelta(hours=hours)
        
    def assign_technician(self, order_id: str) -> Optional[str]:
        """Assign technician to work order
        
        Args:
            order_id: Work order ID
            
        Returns:
            Assigned technician ID or None
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return None
            
        work_order = self.work_orders[order_id]
        
        # Find best technician
        technician_id = self.assignment_engine.find_best_technician(
            work_order,
            list(self.technician_registry.values()),
            self.equipment_registry.get(work_order.equipment_id)
        )
        
        if technician_id:
            work_order.assigned_technician = technician_id
            work_order.status = WorkOrderStatus.ASSIGNED
            
            # Update technician workload
            if technician_id in self.technician_registry:
                self.technician_registry[technician_id].current_workload += 1
                
            # Schedule work
            scheduled_time = self.scheduler.schedule_work_order(
                work_order,
                self.technician_registry.get(technician_id)
            )
            
            work_order.scheduled_start = scheduled_time
            work_order.scheduled_end = scheduled_time + timedelta(hours=work_order.estimated_duration_hours)
            
            # Send notification
            self.notification_system.send_assignment_notification(work_order, technician_id)
            
            logger.info(f"Assigned technician {technician_id} to work order {order_id}")
            
        return technician_id
        
    def update_work_order_status(self,
                                order_id: str,
                                new_status: WorkOrderStatus,
                                notes: str = "") -> bool:
        """Update work order status
        
        Args:
            order_id: Work order ID
            new_status: New status
            notes: Additional notes
            
        Returns:
            Success flag
        """
        if order_id not in self.work_orders:
            logger.error(f"Work order {order_id} not found")
            return False
            
        work_order = self.work_orders[order_id]
        old_status = work_order.status
        work_order.status = new_status
        
        # Handle status-specific updates
        if new_status == WorkOrderStatus.IN_PROGRESS:
            work_order.actual_start = datetime.now()
        elif new_status == WorkOrderStatus.COMPLETED:
            work_order.actual_end = datetime.now()
            if work_order.actual_start:
                work_order.actual_duration_hours = (
                    work_order.actual_end - work_order.actual_start
                ).total_seconds() / 3600
            work_order.actual_cost = work_order.calculate_actual_cost()
            self.metrics['completed_orders'] += 1
            
            # Update technician workload
            if work_order.assigned_technician in self.technician_registry:
                self.technician_registry[work_order.assigned_technician].current_workload -= 1
                
        # Add notes
        if notes:
            work_order.completion_notes = notes
            
        logger.info(f"Updated work order {order_id} status from {old_status} to {new_status}")
        
        return True
        
    def add_parts_used(self,
                      order_id: str,
                      parts: List[Dict[str, Any]]) -> bool:
        """Add parts used to work order
        
        Args:
            order_id: Work order ID
            parts: List of parts with details
            
        Returns:
            Success flag
        """
        if order_id not in self.work_orders:
            return False
            
        work_order = self.work_orders[order_id]
        work_order.parts_used.extend(parts)
        
        # Update actual cost
        work_order.actual_cost = work_order.calculate_actual_cost()
        
        return True
        
    def get_pending_work_orders(self) -> List[WorkOrder]:
        """Get all pending work orders"""
        return [
            wo for wo in self.work_orders.values()
            if wo.status in [WorkOrderStatus.CREATED, WorkOrderStatus.ASSIGNED]
        ]
        
    def get_overdue_work_orders(self) -> List[WorkOrder]:
        """Get all overdue work orders"""
        return [
            wo for wo in self.work_orders.values()
            if wo.is_overdue()
        ]
        
    def register_equipment(self, equipment: Equipment):
        """Register equipment in the system
        
        Args:
            equipment: Equipment object
        """
        self.equipment_registry[equipment.equipment_id] = equipment
        logger.info(f"Registered equipment {equipment.equipment_id}")
        
    def register_technician(self, technician: Technician):
        """Register technician in the system
        
        Args:
            technician: Technician object
        """
        self.technician_registry[technician.technician_id] = technician
        logger.info(f"Registered technician {technician.technician_id}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get work order metrics"""
        # Calculate additional metrics
        completed_orders = [
            wo for wo in self.work_orders.values()
            if wo.status == WorkOrderStatus.COMPLETED
        ]
        
        if completed_orders:
            avg_completion_time = np.mean([
                wo.actual_duration_hours for wo in completed_orders
                if wo.actual_duration_hours
            ])
            
            on_time_orders = sum(
                1 for wo in completed_orders
                if wo.sla_deadline and wo.actual_end <= wo.sla_deadline
            )
            
            sla_compliance = on_time_orders / len(completed_orders) * 100
            
            total_cost = sum(wo.actual_cost for wo in completed_orders)
        else:
            avg_completion_time = 0
            sla_compliance = 100
            total_cost = 0
            
        self.metrics.update({
            'average_completion_time': avg_completion_time,
            'sla_compliance_rate': sla_compliance,
            'total_cost': total_cost,
            'pending_orders': len(self.get_pending_work_orders()),
            'overdue_orders': len(self.get_overdue_work_orders())
        })
        
        return self.metrics
        
    def generate_maintenance_report(self, 
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, Any]:
        """Generate maintenance report for date range
        
        Args:
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Report data
        """
        # Filter work orders by date range
        filtered_orders = [
            wo for wo in self.work_orders.values()
            if start_date <= wo.created_at <= end_date
        ]
        
        report = {
            'period': f"{start_date.date()} to {end_date.date()}",
            'total_orders': len(filtered_orders),
            'by_status': defaultdict(int),
            'by_priority': defaultdict(int),
            'by_type': defaultdict(int),
            'by_equipment': defaultdict(int),
            'by_technician': defaultdict(int),
            'total_cost': 0,
            'average_resolution_time': 0,
            'sla_compliance': 0
        }
        
        completed_times = []
        on_time_count = 0
        
        for wo in filtered_orders:
            report['by_status'][wo.status.value] += 1
            report['by_priority'][wo.priority.name] += 1
            report['by_type'][wo.type.value] += 1
            report['by_equipment'][wo.equipment_id] += 1
            
            if wo.assigned_technician:
                report['by_technician'][wo.assigned_technician] += 1
                
            if wo.status == WorkOrderStatus.COMPLETED:
                report['total_cost'] += wo.actual_cost
                if wo.actual_duration_hours:
                    completed_times.append(wo.actual_duration_hours)
                if wo.sla_deadline and wo.actual_end <= wo.sla_deadline:
                    on_time_count += 1
                    
        # Calculate averages
        if completed_times:
            report['average_resolution_time'] = np.mean(completed_times)
            
        completed_count = report['by_status'].get(WorkOrderStatus.COMPLETED.value, 0)
        if completed_count > 0:
            report['sla_compliance'] = (on_time_count / completed_count) * 100
            
        return report


class AssignmentEngine:
    """Engine for optimal technician assignment"""
    
    def __init__(self):
        """Initialize Assignment Engine"""
        self.assignment_history = []
        
    def find_best_technician(self,
                           work_order: WorkOrder,
                           technicians: List[Technician],
                           equipment: Optional[Equipment] = None) -> Optional[str]:
        """Find best technician for work order
        
        Args:
            work_order: Work order to assign
            technicians: Available technicians
            equipment: Equipment details
            
        Returns:
            Best technician ID or None
        """
        if not technicians:
            return None
            
        # Score each technician
        scores = []
        for tech in technicians:
            score = self._score_technician(tech, work_order, equipment)
            scores.append((tech.technician_id, score))
            
        # Sort by score (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best technician if score is positive
        if scores[0][1] > 0:
            return scores[0][0]
            
        return None
        
    def _score_technician(self,
                        technician: Technician,
                        work_order: WorkOrder,
                        equipment: Optional[Equipment]) -> float:
        """Score technician for work order assignment
        
        Args:
            technician: Technician to score
            work_order: Work order details
            equipment: Equipment details
            
        Returns:
            Score (higher is better)
        """
        score = 0.0
        
        # Check availability
        if technician.current_workload >= technician.max_daily_orders:
            return -1  # Not available
            
        # Skill match
        if equipment:
            required_skill = equipment.type
            if technician.has_skill(required_skill):
                score += 10
                
        # Certification level for priority
        if work_order.priority == WorkOrderPriority.CRITICAL:
            score += technician.certification_level * 3
        elif work_order.priority == WorkOrderPriority.HIGH:
            score += technician.certification_level * 2
        else:
            score += technician.certification_level
            
        # Workload balance (prefer less loaded technicians)
        score -= technician.current_workload * 2
        
        # Location proximity (if applicable)
        if equipment and technician.location:
            if technician.location == equipment.location:
                score += 5
                
        # Cost consideration
        score -= technician.hourly_rate * 0.01
        
        return score
        
    def optimize_assignments(self,
                           work_orders: List[WorkOrder],
                           technicians: List[Technician]) -> Dict[str, str]:
        """Optimize multiple work order assignments
        
        Args:
            work_orders: List of work orders
            technicians: Available technicians
            
        Returns:
            Mapping of order_id to technician_id
        """
        # Use OR-Tools for optimization
        model = cp_model.CpModel()
        
        # Variables
        assignments = {}
        for wo in work_orders:
            for tech in technicians:
                var_name = f"{wo.order_id}_{tech.technician_id}"
                assignments[var_name] = model.NewBoolVar(var_name)
                
        # Constraints
        # Each work order assigned to exactly one technician
        for wo in work_orders:
            model.Add(
                sum(assignments[f"{wo.order_id}_{tech.technician_id}"]
                    for tech in technicians) == 1
            )
            
        # Technician capacity constraints
        for tech in technicians:
            model.Add(
                sum(assignments[f"{wo.order_id}_{tech.technician_id}"]
                    for wo in work_orders) <= tech.max_daily_orders
            )
            
        # Objective: maximize total score
        objective = []
        for wo in work_orders:
            for tech in technicians:
                score = self._score_technician(tech, wo, None)
                if score > 0:
                    objective.append(
                        score * assignments[f"{wo.order_id}_{tech.technician_id}"]
                    )
                    
        model.Maximize(sum(objective))
        
        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # Extract solution
        result = {}
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for wo in work_orders:
                for tech in technicians:
                    if solver.Value(assignments[f"{wo.order_id}_{tech.technician_id}"]):
                        result[wo.order_id] = tech.technician_id
                        break
                        
        return result


class MaintenanceScheduler:
    """Schedule maintenance tasks optimally"""
    
    def __init__(self):
        """Initialize Maintenance Scheduler"""
        self.schedule = defaultdict(list)
        
    def schedule_work_order(self,
                          work_order: WorkOrder,
                          technician: Optional[Technician]) -> datetime:
        """Schedule work order for technician
        
        Args:
            work_order: Work order to schedule
            technician: Assigned technician
            
        Returns:
            Scheduled start time
        """
        # Determine earliest available slot
        if work_order.priority == WorkOrderPriority.CRITICAL:
            # Schedule immediately
            return datetime.now()
            
        if technician:
            # Find next available slot for technician
            return self._find_next_available_slot(
                technician,
                work_order.estimated_duration_hours
            )
        else:
            # Default scheduling based on priority
            hours_delay = {
                WorkOrderPriority.HIGH: 2,
                WorkOrderPriority.MEDIUM: 8,
                WorkOrderPriority.LOW: 24,
                WorkOrderPriority.PREVENTIVE: 48
            }
            
            delay = hours_delay.get(work_order.priority, 8)
            return datetime.now() + timedelta(hours=delay)
            
    def _find_next_available_slot(self,
                                 technician: Technician,
                                 duration_hours: float) -> datetime:
        """Find next available time slot for technician
        
        Args:
            technician: Technician to schedule
            duration_hours: Required duration
            
        Returns:
            Start time for slot
        """
        # Simple implementation - find next working hour
        current_time = datetime.now()
        
        # Round to next hour
        next_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        
        # Check technician availability
        for day_offset in range(7):  # Check next 7 days
            check_date = next_hour + timedelta(days=day_offset)
            if technician.is_available(check_date, duration_hours):
                return check_date
                
        # Default to next business day
        return next_hour + timedelta(days=1)
        
    def create_preventive_schedule(self,
                                  equipment_list: List[Equipment]) -> List[WorkOrder]:
        """Create preventive maintenance schedule
        
        Args:
            equipment_list: List of equipment
            
        Returns:
            List of preventive work orders
        """
        preventive_orders = []
        
        for equipment in equipment_list:
            if equipment.is_maintenance_due():
                # Create preventive work order
                work_order = WorkOrder(
                    order_id=f"PM-{equipment.equipment_id}-{datetime.now().strftime('%Y%m%d')}",
                    equipment_id=equipment.equipment_id,
                    anomaly_id=None,
                    type=MaintenanceType.PREVENTIVE,
                    priority=WorkOrderPriority.PREVENTIVE,
                    status=WorkOrderStatus.CREATED,
                    created_at=datetime.now(),
                    description=f"Scheduled preventive maintenance for {equipment.name}",
                    estimated_duration_hours=2.0
                )
                
                preventive_orders.append(work_order)
                
        return preventive_orders


class CostEstimator:
    """Estimate maintenance costs"""
    
    def __init__(self):
        """Initialize Cost Estimator"""
        self.cost_history = []
        self.part_costs = {}
        
    def estimate_cost(self, work_order: WorkOrder) -> float:
        """Estimate cost for work order
        
        Args:
            work_order: Work order to estimate
            
        Returns:
            Estimated cost
        """
        # Base labor cost
        labor_hours = work_order.estimated_duration_hours
        labor_rate = 50.0  # Default hourly rate
        
        # Adjust for priority
        if work_order.priority == WorkOrderPriority.CRITICAL:
            labor_rate *= 1.5  # Overtime rate
        elif work_order.priority == WorkOrderPriority.HIGH:
            labor_rate *= 1.2
            
        labor_cost = labor_hours * labor_rate
        
        # Estimate parts cost based on type
        parts_cost = self._estimate_parts_cost(work_order)
        
        # Add overhead
        overhead = (labor_cost + parts_cost) * 0.15
        
        total_cost = labor_cost + parts_cost + overhead
        
        return round(total_cost, 2)
        
    def _estimate_parts_cost(self, work_order: WorkOrder) -> float:
        """Estimate parts cost based on maintenance type
        
        Args:
            work_order: Work order details
            
        Returns:
            Estimated parts cost
        """
        # Base estimates by type
        base_costs = {
            MaintenanceType.EMERGENCY: 500,
            MaintenanceType.CORRECTIVE: 200,
            MaintenanceType.PREDICTIVE: 150,
            MaintenanceType.PREVENTIVE: 100,
            MaintenanceType.INSPECTION: 0
        }
        
        base_cost = base_costs.get(work_order.type, 100)
        
        # Adjust for priority
        if work_order.priority == WorkOrderPriority.CRITICAL:
            base_cost *= 1.5
            
        return base_cost


class NotificationManager:
    """Manage notifications for work orders"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Notification Manager
        
        Args:
            config: Notification configuration
        """
        self.config = config or {}
        self.notification_queue = queue.Queue()
        self.notification_history = deque(maxlen=1000)
        
    def send_work_order_created(self, work_order: WorkOrder):
        """Send notification for new work order
        
        Args:
            work_order: Created work order
        """
        message = f"New work order created: {work_order.order_id}\n"
        message += f"Equipment: {work_order.equipment_id}\n"
        message += f"Priority: {work_order.priority.name}\n"
        message += f"Description: {work_order.description}"
        
        self._send_notification(
            subject=f"New Work Order - {work_order.priority.name} Priority",
            message=message,
            recipients=self._get_recipients(work_order.priority)
        )
        
    def send_assignment_notification(self, work_order: WorkOrder, technician_id: str):
        """Send notification for work order assignment
        
        Args:
            work_order: Assigned work order
            technician_id: Assigned technician
        """
        message = f"Work order {work_order.order_id} assigned to technician {technician_id}\n"
        message += f"Scheduled: {work_order.scheduled_start}\n"
        message += f"Estimated duration: {work_order.estimated_duration_hours} hours"
        
        self._send_notification(
            subject=f"Work Order Assignment - {work_order.order_id}",
            message=message,
            recipients=[technician_id]
        )
        
    def _send_notification(self, subject: str, message: str, recipients: List[str]):
        """Send notification to recipients
        
        Args:
            subject: Notification subject
            message: Notification message
            recipients: List of recipient IDs
        """
        notification = {
            'timestamp': datetime.now(),
            'subject': subject,
            'message': message,
            'recipients': recipients
        }
        
        self.notification_queue.put(notification)
        self.notification_history.append(notification)
        
        # Log notification
        logger.info(f"Notification sent: {subject}")
        
    def _get_recipients(self, priority: WorkOrderPriority) -> List[str]:
        """Get notification recipients based on priority
        
        Args:
            priority: Work order priority
            
        Returns:
            List of recipient IDs
        """
        if priority == WorkOrderPriority.CRITICAL:
            return self.config.get('critical_recipients', ['manager', 'supervisor'])
        elif priority == WorkOrderPriority.HIGH:
            return self.config.get('high_recipients', ['supervisor'])
        else:
            return self.config.get('default_recipients', ['supervisor'])