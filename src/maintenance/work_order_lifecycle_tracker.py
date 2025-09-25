"""
Work Order Lifecycle Tracker Module for Phase 3.2
Comprehensive timeline tracking, SLA monitoring, and lifecycle management
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path
from src.maintenance.work_order_manager import (
    WorkOrder, WorkOrderPriority, WorkOrderStatus, MaintenanceType,
    Equipment, Technician, WorkOrderManager
)

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class LifecycleEvent:
    """Individual lifecycle event in work order progression"""
    event_id: str
    work_order_id: str
    event_type: str  # 'created', 'assigned', 'started', 'paused', 'resumed', 'completed', etc.
    timestamp: datetime
    previous_status: Optional[WorkOrderStatus]
    new_status: WorkOrderStatus
    performer: Optional[str]  # Who performed the action (technician_id, system, etc.)
    duration_from_previous: Optional[float]  # Hours since previous event
    event_data: Dict[str, Any] = field(default_factory=dict)
    automated: bool = False  # Whether event was triggered automatically
    notes: str = ""


@dataclass
class SLATarget:
    """SLA target configuration"""
    sla_id: str
    name: str
    priority: WorkOrderPriority
    target_response_hours: float  # Time to first response/assignment
    target_resolution_hours: float  # Time to completion
    escalation_thresholds: List[float]  # Escalation points (e.g., [50%, 75%, 90%])
    business_hours_only: bool = False
    penalty_per_hour_overdue: float = 0.0


@dataclass
class SLAMetrics:
    """SLA performance metrics"""
    work_order_id: str
    sla_target: SLATarget
    response_time_hours: Optional[float]
    resolution_time_hours: Optional[float]
    response_sla_met: bool
    resolution_sla_met: bool
    response_sla_breach_hours: float = 0.0
    resolution_sla_breach_hours: float = 0.0
    escalation_events: List[Dict[str, Any]] = field(default_factory=list)
    current_escalation_level: int = 0


@dataclass
class TimelineAnalysis:
    """Timeline analysis for work order"""
    work_order_id: str
    total_lifecycle_hours: float
    waiting_time_hours: float  # Time waiting for assignment/action
    active_work_time_hours: float  # Time actually being worked on
    delay_analysis: Dict[str, float]  # Breakdown of delays by cause
    efficiency_score: float  # 0.0 to 1.0
    bottlenecks: List[str]  # Identified bottlenecks
    improvement_recommendations: List[str]


@dataclass
class LifecyclePattern:
    """Pattern analysis for similar work orders"""
    pattern_id: str
    pattern_description: str
    equipment_type: str
    maintenance_type: MaintenanceType
    typical_duration_hours: float
    common_bottlenecks: List[str]
    success_factors: List[str]
    risk_factors: List[str]
    optimization_opportunities: List[str]


class WorkOrderLifecycleTracker:
    """Comprehensive work order lifecycle tracking and management system"""

    def __init__(self,
                 work_order_manager: WorkOrderManager,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize Work Order Lifecycle Tracker

        Args:
            work_order_manager: Work order management system
            config: Configuration dictionary
        """
        self.work_order_manager = work_order_manager
        self.config = config or {}

        # Lifecycle tracking
        self.lifecycle_events = defaultdict(list)  # work_order_id -> [LifecycleEvent]
        self.active_timelines = {}  # work_order_id -> current timeline data
        self.sla_targets = {}  # priority -> SLATarget
        self.sla_metrics = {}  # work_order_id -> SLAMetrics

        # Pattern recognition
        self.lifecycle_patterns = {}  # pattern_id -> LifecyclePattern
        self.pattern_matcher = LifecyclePatternMatcher()

        # Timeline analysis
        self.timeline_analyzer = TimelineAnalyzer()
        self.bottleneck_detector = BottleneckDetector()
        self.trend_analyzer = TrendAnalyzer()

        # Real-time monitoring
        self.monitoring_queue = queue.Queue()
        self.monitoring_executor = ThreadPoolExecutor(max_workers=2)
        self.real_time_monitoring = True

        # SLA management
        self.sla_monitor = SLAMonitor()
        self.escalation_manager = EscalationManager()

        # Performance metrics
        self.lifecycle_metrics = {
            'total_work_orders_tracked': 0,
            'average_lifecycle_duration': 0.0,
            'sla_compliance_rate': 0.0,
            'efficiency_improvement_rate': 0.0,
            'bottleneck_resolution_rate': 0.0
        }

        self._initialize_lifecycle_tracker()
        logger.info("Initialized Work Order Lifecycle Tracker")

    def track_work_order_event(self,
                              work_order_id: str,
                              event_type: str,
                              new_status: WorkOrderStatus,
                              performer: Optional[str] = None,
                              automated: bool = False,
                              event_data: Optional[Dict[str, Any]] = None,
                              notes: str = "") -> LifecycleEvent:
        """Track a lifecycle event for a work order

        Args:
            work_order_id: Work order identifier
            event_type: Type of event
            new_status: New status after event
            performer: Who performed the action
            automated: Whether event was automated
            event_data: Additional event data
            notes: Event notes

        Returns:
            Created lifecycle event
        """
        try:
            # Get current work order
            work_order = self.work_order_manager.work_orders.get(work_order_id)
            if not work_order:
                logger.error(f"Work order {work_order_id} not found")
                return None

            # Get previous status and calculate duration
            previous_events = self.lifecycle_events[work_order_id]
            previous_status = work_order.status if not previous_events else previous_events[-1].new_status

            duration_from_previous = None
            if previous_events:
                last_event = previous_events[-1]
                duration_from_previous = (datetime.now() - last_event.timestamp).total_seconds() / 3600

            # Create lifecycle event
            event = LifecycleEvent(
                event_id=f"EVENT-{work_order_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(previous_events)}",
                work_order_id=work_order_id,
                event_type=event_type,
                timestamp=datetime.now(),
                previous_status=previous_status,
                new_status=new_status,
                performer=performer,
                duration_from_previous=duration_from_previous,
                event_data=event_data or {},
                automated=automated,
                notes=notes
            )

            # Store event
            self.lifecycle_events[work_order_id].append(event)

            # Update work order status
            work_order.status = new_status

            # Update SLA metrics
            self._update_sla_metrics(work_order_id, event)

            # Real-time analysis
            if self.real_time_monitoring:
                self._perform_real_time_analysis(work_order_id, event)

            # Pattern matching
            self._update_pattern_matching(work_order_id, event)

            logger.info(f"Tracked lifecycle event: {event_type} for work order {work_order_id}")

            return event

        except Exception as e:
            logger.error(f"Error tracking lifecycle event: {e}")
            return None

    def start_work_order_tracking(self, work_order: WorkOrder):
        """Start comprehensive tracking for a work order

        Args:
            work_order: Work order to start tracking
        """
        # Initialize SLA metrics
        sla_target = self._get_sla_target(work_order.priority)
        if sla_target:
            sla_metrics = SLAMetrics(
                work_order_id=work_order.order_id,
                sla_target=sla_target,
                response_time_hours=None,
                resolution_time_hours=None,
                response_sla_met=False,
                resolution_sla_met=False
            )
            self.sla_metrics[work_order.order_id] = sla_metrics

        # Track initial creation event
        self.track_work_order_event(
            work_order.order_id,
            'created',
            WorkOrderStatus.CREATED,
            performer='system',
            automated=True,
            event_data={
                'priority': work_order.priority.name,
                'equipment_id': work_order.equipment_id,
                'estimated_duration': work_order.estimated_duration_hours
            }
        )

        # Initialize active timeline
        self.active_timelines[work_order.order_id] = {
            'start_time': datetime.now(),
            'current_phase': 'created',
            'phase_start_time': datetime.now(),
            'delays': [],
            'escalations': []
        }

        logger.info(f"Started lifecycle tracking for work order {work_order.order_id}")

    def complete_work_order_tracking(self, work_order_id: str):
        """Complete tracking for a work order and perform final analysis

        Args:
            work_order_id: Work order identifier
        """
        try:
            # Track completion event
            self.track_work_order_event(
                work_order_id,
                'completed',
                WorkOrderStatus.COMPLETED,
                automated=False
            )

            # Perform final timeline analysis
            timeline_analysis = self._perform_complete_timeline_analysis(work_order_id)

            # Update pattern recognition
            self._update_pattern_recognition(work_order_id, timeline_analysis)

            # Calculate final SLA metrics
            self._finalize_sla_metrics(work_order_id)

            # Update performance metrics
            self._update_lifecycle_metrics(work_order_id)

            # Clean up active timeline
            if work_order_id in self.active_timelines:
                del self.active_timelines[work_order_id]

            logger.info(f"Completed lifecycle tracking for work order {work_order_id}")

        except Exception as e:
            logger.error(f"Error completing work order tracking: {e}")

    def get_work_order_timeline(self, work_order_id: str) -> List[LifecycleEvent]:
        """Get complete timeline for a work order

        Args:
            work_order_id: Work order identifier

        Returns:
            List of lifecycle events
        """
        return self.lifecycle_events.get(work_order_id, [])

    def get_sla_status(self, work_order_id: str) -> Optional[SLAMetrics]:
        """Get SLA status for a work order

        Args:
            work_order_id: Work order identifier

        Returns:
            SLA metrics or None
        """
        return self.sla_metrics.get(work_order_id)

    def get_timeline_analysis(self, work_order_id: str) -> Optional[TimelineAnalysis]:
        """Get timeline analysis for a work order

        Args:
            work_order_id: Work order identifier

        Returns:
            Timeline analysis or None
        """
        if work_order_id not in self.lifecycle_events:
            return None

        return self._perform_complete_timeline_analysis(work_order_id)

    def get_bottleneck_analysis(self,
                              date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get bottleneck analysis across work orders

        Args:
            date_range: Date range to analyze (start, end)

        Returns:
            Bottleneck analysis
        """
        return self.bottleneck_detector.analyze_bottlenecks(
            self.lifecycle_events, date_range
        )

    def get_efficiency_trends(self,
                            date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get efficiency trend analysis

        Args:
            date_range: Date range to analyze

        Returns:
            Efficiency trends
        """
        return self.trend_analyzer.analyze_efficiency_trends(
            self.lifecycle_events, date_range
        )

    def _get_sla_target(self, priority: WorkOrderPriority) -> Optional[SLATarget]:
        """Get SLA target for priority level

        Args:
            priority: Work order priority

        Returns:
            SLA target or None
        """
        return self.sla_targets.get(priority)

    def _update_sla_metrics(self, work_order_id: str, event: LifecycleEvent):
        """Update SLA metrics based on lifecycle event

        Args:
            work_order_id: Work order identifier
            event: Lifecycle event
        """
        sla_metrics = self.sla_metrics.get(work_order_id)
        if not sla_metrics:
            return

        work_order = self.work_order_manager.work_orders[work_order_id]
        creation_time = work_order.created_at

        # Update response time
        if event.event_type in ['assigned'] and sla_metrics.response_time_hours is None:
            response_time = (event.timestamp - creation_time).total_seconds() / 3600
            sla_metrics.response_time_hours = response_time
            sla_metrics.response_sla_met = response_time <= sla_metrics.sla_target.target_response_hours

            if not sla_metrics.response_sla_met:
                sla_metrics.response_sla_breach_hours = response_time - sla_metrics.sla_target.target_response_hours

        # Update resolution time
        if event.event_type in ['completed'] and sla_metrics.resolution_time_hours is None:
            resolution_time = (event.timestamp - creation_time).total_seconds() / 3600
            sla_metrics.resolution_time_hours = resolution_time
            sla_metrics.resolution_sla_met = resolution_time <= sla_metrics.sla_target.target_resolution_hours

            if not sla_metrics.resolution_sla_met:
                sla_metrics.resolution_sla_breach_hours = resolution_time - sla_metrics.sla_target.target_resolution_hours

        # Check for escalation
        self._check_sla_escalation(work_order_id, sla_metrics)

    def _check_sla_escalation(self, work_order_id: str, sla_metrics: SLAMetrics):
        """Check if SLA escalation is needed

        Args:
            work_order_id: Work order identifier
            sla_metrics: Current SLA metrics
        """
        work_order = self.work_order_manager.work_orders[work_order_id]
        creation_time = work_order.created_at
        current_time = datetime.now()
        elapsed_hours = (current_time - creation_time).total_seconds() / 3600

        target_hours = sla_metrics.sla_target.target_resolution_hours
        elapsed_percentage = elapsed_hours / target_hours

        for i, threshold in enumerate(sla_metrics.sla_target.escalation_thresholds):
            if elapsed_percentage >= threshold and sla_metrics.current_escalation_level <= i:
                # Trigger escalation
                escalation_event = {
                    'escalation_level': i + 1,
                    'threshold_percentage': threshold,
                    'elapsed_hours': elapsed_hours,
                    'timestamp': current_time,
                    'triggered_by': 'sla_monitor'
                }

                sla_metrics.escalation_events.append(escalation_event)
                sla_metrics.current_escalation_level = i + 1

                # Trigger escalation action
                self.escalation_manager.trigger_escalation(work_order_id, escalation_event)

                logger.warning(f"SLA escalation level {i + 1} triggered for work order {work_order_id}")

    def _perform_real_time_analysis(self, work_order_id: str, event: LifecycleEvent):
        """Perform real-time analysis on lifecycle event

        Args:
            work_order_id: Work order identifier
            event: Lifecycle event
        """
        # Queue analysis task
        analysis_task = {
            'work_order_id': work_order_id,
            'event': event,
            'timestamp': datetime.now()
        }

        self.monitoring_queue.put(analysis_task)

    def _perform_complete_timeline_analysis(self, work_order_id: str) -> TimelineAnalysis:
        """Perform complete timeline analysis for work order

        Args:
            work_order_id: Work order identifier

        Returns:
            Timeline analysis
        """
        events = self.lifecycle_events[work_order_id]
        if not events:
            return None

        # Calculate timeline metrics
        start_time = events[0].timestamp
        end_time = events[-1].timestamp if events[-1].event_type == 'completed' else datetime.now()

        total_duration = (end_time - start_time).total_seconds() / 3600

        # Calculate waiting vs active time
        waiting_time = 0.0
        active_time = 0.0

        active_states = ['in_progress', 'assigned']
        waiting_states = ['created', 'on_hold']

        current_state = 'created'
        current_state_start = start_time

        for event in events[1:]:
            duration = (event.timestamp - current_state_start).total_seconds() / 3600

            if current_state in active_states:
                active_time += duration
            elif current_state in waiting_states:
                waiting_time += duration

            current_state = event.event_type
            current_state_start = event.timestamp

        # Calculate efficiency score
        estimated_duration = self.work_order_manager.work_orders[work_order_id].estimated_duration_hours
        efficiency_score = min(1.0, estimated_duration / total_duration) if total_duration > 0 else 0.0

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(events)

        # Generate recommendations
        recommendations = self._generate_improvement_recommendations(events, efficiency_score)

        return TimelineAnalysis(
            work_order_id=work_order_id,
            total_lifecycle_hours=total_duration,
            waiting_time_hours=waiting_time,
            active_work_time_hours=active_time,
            delay_analysis=self._analyze_delays(events),
            efficiency_score=efficiency_score,
            bottlenecks=bottlenecks,
            improvement_recommendations=recommendations
        )

    def _identify_bottlenecks(self, events: List[LifecycleEvent]) -> List[str]:
        """Identify bottlenecks in work order timeline

        Args:
            events: List of lifecycle events

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        # Check for long waiting periods
        for i in range(1, len(events)):
            if events[i].duration_from_previous and events[i].duration_from_previous > 8:
                if events[i-1].event_type == 'created':
                    bottlenecks.append('Long assignment delay')
                elif events[i-1].event_type == 'assigned':
                    bottlenecks.append('Delayed work start')
                elif events[i-1].event_type == 'on_hold':
                    bottlenecks.append('Extended hold period')

        return bottlenecks

    def _analyze_delays(self, events: List[LifecycleEvent]) -> Dict[str, float]:
        """Analyze delays in work order progression

        Args:
            events: List of lifecycle events

        Returns:
            Delay analysis breakdown
        """
        delays = {
            'assignment_delay': 0.0,
            'start_delay': 0.0,
            'execution_delay': 0.0,
            'completion_delay': 0.0
        }

        for i in range(1, len(events)):
            prev_event = events[i-1]
            curr_event = events[i]

            if curr_event.duration_from_previous:
                if prev_event.event_type == 'created' and curr_event.event_type == 'assigned':
                    delays['assignment_delay'] = curr_event.duration_from_previous
                elif prev_event.event_type == 'assigned' and curr_event.event_type == 'in_progress':
                    delays['start_delay'] = curr_event.duration_from_previous

        return delays

    def _generate_improvement_recommendations(self,
                                            events: List[LifecycleEvent],
                                            efficiency_score: float) -> List[str]:
        """Generate improvement recommendations based on timeline analysis

        Args:
            events: List of lifecycle events
            efficiency_score: Calculated efficiency score

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        if efficiency_score < 0.7:
            recommendations.append("Consider process optimization to improve efficiency")

        # Check for specific improvement areas
        bottlenecks = self._identify_bottlenecks(events)

        if 'Long assignment delay' in bottlenecks:
            recommendations.append("Implement automated technician assignment")

        if 'Delayed work start' in bottlenecks:
            recommendations.append("Improve technician notification system")

        if 'Extended hold period' in bottlenecks:
            recommendations.append("Review parts availability and procurement process")

        return recommendations

    def _initialize_lifecycle_tracker(self):
        """Initialize lifecycle tracker with default configurations"""
        # Initialize default SLA targets
        self.sla_targets = {
            WorkOrderPriority.CRITICAL: SLATarget(
                sla_id='sla_critical',
                name='Critical Priority SLA',
                priority=WorkOrderPriority.CRITICAL,
                target_response_hours=1.0,
                target_resolution_hours=4.0,
                escalation_thresholds=[0.5, 0.75, 0.9]
            ),
            WorkOrderPriority.HIGH: SLATarget(
                sla_id='sla_high',
                name='High Priority SLA',
                priority=WorkOrderPriority.HIGH,
                target_response_hours=4.0,
                target_resolution_hours=24.0,
                escalation_thresholds=[0.5, 0.75, 0.9]
            ),
            WorkOrderPriority.MEDIUM: SLATarget(
                sla_id='sla_medium',
                name='Medium Priority SLA',
                priority=WorkOrderPriority.MEDIUM,
                target_response_hours=8.0,
                target_resolution_hours=72.0,
                escalation_thresholds=[0.75, 0.9]
            ),
            WorkOrderPriority.LOW: SLATarget(
                sla_id='sla_low',
                name='Low Priority SLA',
                priority=WorkOrderPriority.LOW,
                target_response_hours=24.0,
                target_resolution_hours=168.0,
                escalation_thresholds=[0.9]
            )
        }

        logger.info("Initialized lifecycle tracker with default SLA targets")

    def _update_pattern_matching(self, work_order_id: str, event: LifecycleEvent):
        """Update pattern matching with new event

        Args:
            work_order_id: Work order identifier
            event: Lifecycle event
        """
        # Pattern matching implementation would go here
        pass

    def _update_pattern_recognition(self, work_order_id: str, timeline_analysis: TimelineAnalysis):
        """Update pattern recognition with completed timeline

        Args:
            work_order_id: Work order identifier
            timeline_analysis: Timeline analysis
        """
        # Pattern recognition implementation would go here
        pass

    def _finalize_sla_metrics(self, work_order_id: str):
        """Finalize SLA metrics for completed work order

        Args:
            work_order_id: Work order identifier
        """
        sla_metrics = self.sla_metrics.get(work_order_id)
        if sla_metrics and sla_metrics.resolution_time_hours is None:
            # Calculate final resolution time if not already set
            events = self.lifecycle_events[work_order_id]
            if events:
                work_order = self.work_order_manager.work_orders[work_order_id]
                creation_time = work_order.created_at
                completion_event = next((e for e in reversed(events) if e.event_type == 'completed'), None)

                if completion_event:
                    resolution_time = (completion_event.timestamp - creation_time).total_seconds() / 3600
                    sla_metrics.resolution_time_hours = resolution_time
                    sla_metrics.resolution_sla_met = resolution_time <= sla_metrics.sla_target.target_resolution_hours

    def _update_lifecycle_metrics(self, work_order_id: str):
        """Update overall lifecycle metrics

        Args:
            work_order_id: Work order identifier
        """
        self.lifecycle_metrics['total_work_orders_tracked'] += 1

        # Update running averages
        timeline_analysis = self._perform_complete_timeline_analysis(work_order_id)
        if timeline_analysis:
            n = self.lifecycle_metrics['total_work_orders_tracked']
            self.lifecycle_metrics['average_lifecycle_duration'] = (
                (self.lifecycle_metrics['average_lifecycle_duration'] * (n - 1) +
                 timeline_analysis.total_lifecycle_hours) / n
            )

        # Update SLA compliance rate
        sla_metrics = self.sla_metrics.get(work_order_id)
        if sla_metrics:
            compliant_orders = sum(
                1 for sla in self.sla_metrics.values()
                if sla.resolution_sla_met
            )
            self.lifecycle_metrics['sla_compliance_rate'] = (
                compliant_orders / len(self.sla_metrics) * 100
            )

    def get_lifecycle_metrics(self) -> Dict[str, Any]:
        """Get lifecycle performance metrics

        Returns:
            Lifecycle metrics
        """
        return self.lifecycle_metrics.copy()


# Placeholder classes for components referenced but not fully implemented
class LifecyclePatternMatcher:
    def __init__(self):
        pass

class TimelineAnalyzer:
    def __init__(self):
        pass

class BottleneckDetector:
    def __init__(self):
        pass

    def analyze_bottlenecks(self, lifecycle_events: Dict, date_range: Optional[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        return {'bottlenecks': [], 'analysis': 'Not implemented'}

class TrendAnalyzer:
    def __init__(self):
        pass

    def analyze_efficiency_trends(self, lifecycle_events: Dict, date_range: Optional[Tuple[datetime, datetime]]) -> Dict[str, Any]:
        return {'trends': [], 'analysis': 'Not implemented'}

class SLAMonitor:
    def __init__(self):
        pass

class EscalationManager:
    def __init__(self):
        pass

    def trigger_escalation(self, work_order_id: str, escalation_event: Dict[str, Any]):
        logger.info(f"Escalation triggered for work order {work_order_id}: Level {escalation_event['escalation_level']}")