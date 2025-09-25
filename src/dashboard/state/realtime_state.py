"""
Real-time State Manager
Coordinates and synchronizes all real-time components in the NASA IoT dashboard
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import time
import queue
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import uuid
from enum import Enum

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of dashboard components"""
    PIPELINE_MONITOR = "pipeline_monitor"
    ANOMALY_HEATMAP = "anomaly_heatmap"
    MODEL_STATUS = "model_status"
    ALERTS_PIPELINE = "alerts_pipeline"
    DATA_ORCHESTRATOR = "data_orchestrator"


class UpdatePriority(Enum):
    """Update priority levels"""
    CRITICAL = 1    # Immediate updates (alerts, critical anomalies)
    HIGH = 2        # High frequency updates (processing rates)
    NORMAL = 3      # Standard updates (most components)
    LOW = 4         # Background updates (statistics)


@dataclass
class ComponentState:
    """State information for a dashboard component"""
    component_id: str
    component_type: ComponentType
    is_active: bool
    last_update: datetime
    update_interval: float  # seconds
    priority: UpdatePriority
    error_count: int = 0
    last_error: Optional[str] = None
    update_callback: Optional[Callable] = None
    data_cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateUpdate:
    """Represents a state update request"""
    update_id: str
    component_id: str
    timestamp: datetime
    priority: UpdatePriority
    data: Dict[str, Any]
    callback: Optional[Callable] = None


class RealTimeStateManager:
    """
    Centralized real-time state manager for NASA IoT dashboard
    Coordinates updates across all pipeline components for optimal performance
    """

    def __init__(self):
        """Initialize real-time state manager"""
        # Component management
        self.components: Dict[str, ComponentState] = {}
        self.update_queue = queue.PriorityQueue()
        self.state_cache: Dict[str, Any] = {}

        # Update coordination
        self.update_scheduler = None
        self.background_executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        self._stop_event = threading.Event()

        # Performance tracking
        self.performance_metrics = {
            'total_updates': 0,
            'failed_updates': 0,
            'avg_update_time': 0.0,
            'updates_per_second': 0.0,
            'last_performance_check': datetime.now()
        }

        # Update history for analysis
        self.update_history = deque(maxlen=1000)

        # Component managers (will be injected)
        self.pipeline_monitor = None
        self.anomaly_heatmap = None
        self.model_status = None
        self.alerts_pipeline = None
        self.data_orchestrator = None

        # Initialize component managers
        self._initialize_components()

        logger.info("Real-time State Manager initialized")

    def _initialize_components(self):
        """Initialize all component managers"""
        try:
            from src.dashboard.components.pipeline_status_monitor import pipeline_status_monitor
            self.pipeline_monitor = pipeline_status_monitor
            self._register_component(
                "pipeline_monitor",
                ComponentType.PIPELINE_MONITOR,
                update_interval=2.0,
                priority=UpdatePriority.HIGH
            )
        except ImportError as e:
            logger.warning(f"Pipeline monitor not available: {e}")

        try:
            from src.dashboard.components.anomaly_heatmap import anomaly_heatmap_manager
            self.anomaly_heatmap = anomaly_heatmap_manager
            self._register_component(
                "anomaly_heatmap",
                ComponentType.ANOMALY_HEATMAP,
                update_interval=5.0,
                priority=UpdatePriority.NORMAL
            )
        except ImportError as e:
            logger.warning(f"Anomaly heatmap not available: {e}")

        try:
            from src.dashboard.components.model_status_panel import model_status_manager
            self.model_status = model_status_manager
            self._register_component(
                "model_status",
                ComponentType.MODEL_STATUS,
                update_interval=3.0,
                priority=UpdatePriority.NORMAL
            )
        except ImportError as e:
            logger.warning(f"Model status panel not available: {e}")

        try:
            from src.dashboard.components.alerts_pipeline import nasa_alerts_manager
            self.alerts_pipeline = nasa_alerts_manager
            self._register_component(
                "alerts_pipeline",
                ComponentType.ALERTS_PIPELINE,
                update_interval=1.0,
                priority=UpdatePriority.CRITICAL
            )
        except ImportError as e:
            logger.warning(f"Alerts pipeline not available: {e}")

        try:
            from src.dashboard.unified_data_orchestrator import unified_data_orchestrator
            self.data_orchestrator = unified_data_orchestrator
            self._register_component(
                "data_orchestrator",
                ComponentType.DATA_ORCHESTRATOR,
                update_interval=10.0,
                priority=UpdatePriority.LOW
            )
        except ImportError as e:
            logger.warning(f"Data orchestrator not available: {e}")

    def _register_component(self, component_id: str, component_type: ComponentType,
                          update_interval: float, priority: UpdatePriority):
        """Register a component with the state manager

        Args:
            component_id: Unique component identifier
            component_type: Type of component
            update_interval: Update interval in seconds
            priority: Update priority
        """
        self.components[component_id] = ComponentState(
            component_id=component_id,
            component_type=component_type,
            is_active=True,
            last_update=datetime.now(),
            update_interval=update_interval,
            priority=priority
        )

        logger.info(f"Registered component: {component_id} ({component_type.value})")

    def start_real_time_updates(self):
        """Start the real-time update system"""
        if self.is_running:
            logger.warning("Real-time updates already running")
            return

        self.is_running = True
        self._stop_event.clear()

        # Start update scheduler thread
        self.update_scheduler = threading.Thread(target=self._update_scheduler_loop, daemon=True)
        self.update_scheduler.start()

        # Start component update threads
        for component_id, component in self.components.items():
            if component.is_active:
                self.background_executor.submit(self._component_update_loop, component_id)

        logger.info("Real-time update system started")

    def stop_real_time_updates(self):
        """Stop the real-time update system"""
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()

        # Wait for scheduler to stop
        if self.update_scheduler:
            self.update_scheduler.join(timeout=5)

        # Shutdown executor
        self.background_executor.shutdown(wait=True)

        logger.info("Real-time update system stopped")

    def _update_scheduler_loop(self):
        """Main update scheduler loop"""
        while not self._stop_event.is_set():
            try:
                # Process high-priority updates immediately
                self._process_priority_updates()

                # Check for component updates
                self._check_component_updates()

                # Performance monitoring
                self._update_performance_metrics()

                # Sleep briefly
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in update scheduler: {e}")
                time.sleep(1.0)

    def _component_update_loop(self, component_id: str):
        """Update loop for individual component

        Args:
            component_id: Component to update
        """
        while not self._stop_event.is_set():
            try:
                component = self.components.get(component_id)
                if not component or not component.is_active:
                    break

                # Check if update is due
                time_since_update = (datetime.now() - component.last_update).total_seconds()
                if time_since_update >= component.update_interval:
                    self._update_component(component_id)

                # Sleep based on priority
                sleep_time = self._get_sleep_time(component.priority)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in component update loop for {component_id}: {e}")
                time.sleep(5.0)

    def _update_component(self, component_id: str):
        """Update a specific component

        Args:
            component_id: Component to update
        """
        try:
            component = self.components.get(component_id)
            if not component:
                return

            start_time = datetime.now()

            # Update component based on type
            update_data = self._get_component_update_data(component_id, component.component_type)

            if update_data:
                # Cache the update data
                self.state_cache[component_id] = update_data

                # Update component state
                component.last_update = datetime.now()
                component.error_count = 0
                component.last_error = None

                # Record performance
                update_time = (datetime.now() - start_time).total_seconds()
                self._record_update_performance(component_id, update_time, success=True)

                logger.debug(f"Updated component {component_id} in {update_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to update component {component_id}: {e}")
            component = self.components.get(component_id)
            if component:
                component.error_count += 1
                component.last_error = str(e)
                self._record_update_performance(component_id, 0, success=False)

    def _get_component_update_data(self, component_id: str, component_type: ComponentType) -> Optional[Dict[str, Any]]:
        """Get update data for a component

        Args:
            component_id: Component identifier
            component_type: Component type

        Returns:
            Update data dictionary
        """
        try:
            if component_type == ComponentType.PIPELINE_MONITOR and self.pipeline_monitor:
                return {
                    'metrics': self.pipeline_monitor.get_current_metrics(),
                    'health_status': self.pipeline_monitor.get_pipeline_health_status(),
                    'throughput_metrics': self.pipeline_monitor.get_throughput_metrics(),
                    'data_sources_status': self.pipeline_monitor.get_data_sources_status(),
                    'queue_status': self.pipeline_monitor.get_queue_status()
                }

            elif component_type == ComponentType.ANOMALY_HEATMAP and self.anomaly_heatmap:
                return {
                    'heatmap_data': self.anomaly_heatmap.get_heatmap_data(),
                    'summary_statistics': self.anomaly_heatmap.get_summary_statistics()
                }

            elif component_type == ComponentType.MODEL_STATUS and self.model_status:
                return {
                    'summary_statistics': self.model_status.get_summary_statistics(),
                    'active_models_count': self.model_status.get_active_models_count(),
                    'grid_data': self.model_status.get_model_grid_data()
                }

            elif component_type == ComponentType.ALERTS_PIPELINE and self.alerts_pipeline:
                # Generate new alerts
                new_alerts = self.alerts_pipeline.generate_real_time_alerts()

                return {
                    'live_alerts_stream': self.alerts_pipeline.get_live_alerts_stream(),
                    'active_alerts': self.alerts_pipeline.get_active_alerts(),
                    'alert_statistics': self.alerts_pipeline.get_alert_statistics(),
                    'new_alerts_count': len(new_alerts)
                }

            elif component_type == ComponentType.DATA_ORCHESTRATOR and self.data_orchestrator:
                return {
                    'system_overview': self.data_orchestrator.get_system_overview(),
                    'equipment_status': self.data_orchestrator.get_all_equipment_status()
                }

        except Exception as e:
            logger.error(f"Error getting update data for {component_id}: {e}")

        return None

    def _process_priority_updates(self):
        """Process high-priority updates from the queue"""
        try:
            while not self.update_queue.empty():
                try:
                    priority, update = self.update_queue.get_nowait()
                    self._execute_state_update(update)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error processing priority updates: {e}")

    def _check_component_updates(self):
        """Check which components need updates"""
        current_time = datetime.now()

        for component_id, component in self.components.items():
            if not component.is_active:
                continue

            time_since_update = (current_time - component.last_update).total_seconds()

            # Check if urgent update is needed
            if component.priority == UpdatePriority.CRITICAL and time_since_update > 0.5:
                self._schedule_urgent_update(component_id)
            elif component.priority == UpdatePriority.HIGH and time_since_update > 2.0:
                self._schedule_urgent_update(component_id)

    def _schedule_urgent_update(self, component_id: str):
        """Schedule an urgent update for a component

        Args:
            component_id: Component needing urgent update
        """
        component = self.components.get(component_id)
        if not component:
            return

        update = StateUpdate(
            update_id=str(uuid.uuid4()),
            component_id=component_id,
            timestamp=datetime.now(),
            priority=component.priority,
            data={}
        )

        # Add to priority queue
        self.update_queue.put((component.priority.value, update))

    def _execute_state_update(self, update: StateUpdate):
        """Execute a state update

        Args:
            update: State update to execute
        """
        try:
            self._update_component(update.component_id)

            if update.callback:
                update.callback(update)

        except Exception as e:
            logger.error(f"Error executing state update {update.update_id}: {e}")

    def _get_sleep_time(self, priority: UpdatePriority) -> float:
        """Get sleep time based on priority

        Args:
            priority: Update priority

        Returns:
            Sleep time in seconds
        """
        sleep_times = {
            UpdatePriority.CRITICAL: 0.5,
            UpdatePriority.HIGH: 1.0,
            UpdatePriority.NORMAL: 2.0,
            UpdatePriority.LOW: 5.0
        }
        return sleep_times.get(priority, 2.0)

    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            current_time = datetime.now()
            time_since_check = (current_time - self.performance_metrics['last_performance_check']).total_seconds()

            if time_since_check >= 10.0:  # Update every 10 seconds
                # Calculate updates per second
                recent_updates = [u for u in self.update_history
                                if (current_time - u['timestamp']).total_seconds() < 60]

                if recent_updates:
                    self.performance_metrics['updates_per_second'] = len(recent_updates) / 60.0

                    # Calculate average update time
                    successful_updates = [u for u in recent_updates if u['success']]
                    if successful_updates:
                        avg_time = sum([u['duration'] for u in successful_updates]) / len(successful_updates)
                        self.performance_metrics['avg_update_time'] = avg_time

                self.performance_metrics['last_performance_check'] = current_time

        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    def _record_update_performance(self, component_id: str, duration: float, success: bool):
        """Record performance metrics for an update

        Args:
            component_id: Component that was updated
            duration: Update duration in seconds
            success: Whether update was successful
        """
        self.update_history.append({
            'component_id': component_id,
            'timestamp': datetime.now(),
            'duration': duration,
            'success': success
        })

        self.performance_metrics['total_updates'] += 1
        if not success:
            self.performance_metrics['failed_updates'] += 1

    def get_component_data(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a component

        Args:
            component_id: Component identifier

        Returns:
            Cached component data
        """
        return self.state_cache.get(component_id)

    def get_all_component_data(self) -> Dict[str, Any]:
        """Get all cached component data

        Returns:
            All component data
        """
        return self.state_cache.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status

        Returns:
            System status information
        """
        active_components = sum(1 for c in self.components.values() if c.is_active)
        error_components = sum(1 for c in self.components.values() if c.error_count > 0)

        return {
            'is_running': self.is_running,
            'total_components': len(self.components),
            'active_components': active_components,
            'error_components': error_components,
            'performance_metrics': self.performance_metrics,
            'last_update': max([c.last_update for c in self.components.values()]) if self.components else datetime.now()
        }

    def get_component_status(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific component

        Args:
            component_id: Component identifier

        Returns:
            Component status information
        """
        component = self.components.get(component_id)
        if not component:
            return None

        return {
            'component_id': component.component_id,
            'component_type': component.component_type.value,
            'is_active': component.is_active,
            'last_update': component.last_update.isoformat(),
            'update_interval': component.update_interval,
            'priority': component.priority.value,
            'error_count': component.error_count,
            'last_error': component.last_error,
            'has_cached_data': component_id in self.state_cache
        }

    def activate_component(self, component_id: str):
        """Activate a component

        Args:
            component_id: Component to activate
        """
        component = self.components.get(component_id)
        if component:
            component.is_active = True
            logger.info(f"Activated component: {component_id}")

    def deactivate_component(self, component_id: str):
        """Deactivate a component

        Args:
            component_id: Component to deactivate
        """
        component = self.components.get(component_id)
        if component:
            component.is_active = False
            logger.info(f"Deactivated component: {component_id}")

    def force_update_component(self, component_id: str):
        """Force an immediate update of a component

        Args:
            component_id: Component to update
        """
        if component_id in self.components:
            self.background_executor.submit(self._update_component, component_id)

    def force_update_all_components(self):
        """Force an immediate update of all active components"""
        for component_id, component in self.components.items():
            if component.is_active:
                self.background_executor.submit(self._update_component, component_id)


# Global instance for dashboard integration
realtime_state_manager = RealTimeStateManager()