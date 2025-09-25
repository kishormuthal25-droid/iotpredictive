"""
Shared State Manager
Global state management for dashboard components and cross-page persistence
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StateUpdate:
    """State update event"""
    component_id: str
    state_key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)


class SharedStateManager:
    """
    Global state manager for dashboard components
    Handles cross-component state sharing and persistence
    """

    def __init__(self):
        """Initialize shared state manager"""
        self._state = {}
        self._subscribers = defaultdict(list)  # state_key -> [callback_functions]
        self._lock = threading.Lock()
        self._update_history = []
        self._max_history = 100

        # Initialize default state structure
        self._initialize_default_state()

        logger.info("SharedStateManager initialized")

    def _initialize_default_state(self):
        """Initialize default state structure"""
        self._state = {
            # Global UI state
            'ui': {
                'current_page': '/',
                'sidebar_collapsed': False,
                'theme': 'light',
                'notification_count': 0
            },

            # Equipment and sensor selections
            'selections': {
                'equipment_id': None,
                'sensor_id': None,
                'metric_id': None,
                'subsystem': None,
                'equipment_type': None
            },

            # Time and filtering state
            'filters': {
                'time_range': 'last_1h',
                'start_time': None,
                'end_time': None,
                'is_realtime': False,
                'auto_refresh': True,
                'refresh_interval': 60000,  # milliseconds
                'chart_type': 'line',
                'show_anomalies': True,
                'show_thresholds': True,
                'show_confidence': False
            },

            # Dashboard-specific state
            'anomaly_monitor': {
                'selected_sensors': [],
                'detection_status': 'active',
                'alert_count': 0,
                'last_anomaly': None
            },

            'forecast': {
                'horizon': '24h',
                'model': 'lstm',
                'confidence_level': 0.95,
                'forecast_data': None
            },

            'maintenance': {
                'schedule_view': 'calendar',
                'filter_status': 'all',
                'selected_technician': None,
                'optimization_mode': 'cost'
            },

            'work_orders': {
                'view_mode': 'list',
                'filter_priority': 'all',
                'filter_status': 'active',
                'selected_order': None
            },

            # NASA mission state
            'nasa': {
                'active_mission': 'both',  # 'smap', 'msl', 'both'
                'mission_time': 'earth',   # 'earth', 'mars', 'mission'
                'spacecraft_status': {
                    'smap': 'nominal',
                    'msl': 'nominal'
                }
            }
        }

    def get_state(self, key_path: str) -> Any:
        """
        Get state value by key path

        Args:
            key_path: Dot-separated key path (e.g., 'selections.equipment_id')

        Returns:
            State value or None if not found
        """
        with self._lock:
            try:
                keys = key_path.split('.')
                value = self._state

                for key in keys:
                    value = value[key]

                return value

            except (KeyError, TypeError):
                logger.warning(f"State key not found: {key_path}")
                return None

    def set_state(self, key_path: str, value: Any, component_id: str = "unknown") -> bool:
        """
        Set state value by key path

        Args:
            key_path: Dot-separated key path
            value: Value to set
            component_id: ID of component making the change

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                keys = key_path.split('.')
                current = self._state
                old_value = self.get_state(key_path)

                # Navigate to parent of target key
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # Set the value
                current[keys[-1]] = value

                # Record update
                update = StateUpdate(
                    component_id=component_id,
                    state_key=key_path,
                    old_value=old_value,
                    new_value=value
                )
                self._update_history.append(update)

                # Trim history if needed
                if len(self._update_history) > self._max_history:
                    self._update_history = self._update_history[-self._max_history:]

                # Notify subscribers
                self._notify_subscribers(key_path, old_value, value)

                logger.debug(f"State updated: {key_path} = {value} by {component_id}")
                return True

            except Exception as e:
                logger.error(f"Error setting state {key_path}: {e}")
                return False

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state dictionary"""
        with self._lock:
            return self._state.copy()

    def update_multiple(self, updates: Dict[str, Any], component_id: str = "unknown") -> bool:
        """
        Update multiple state values atomically

        Args:
            updates: Dictionary of key_path -> value
            component_id: ID of component making changes

        Returns:
            True if all updates successful
        """
        success = True
        for key_path, value in updates.items():
            if not self.set_state(key_path, value, component_id):
                success = False

        return success

    def subscribe(self, key_path: str, callback: Callable[[str, Any, Any], None],
                 component_id: str = "unknown"):
        """
        Subscribe to state changes

        Args:
            key_path: Key path to monitor
            callback: Function(key_path, old_value, new_value)
            component_id: Component identifier
        """
        with self._lock:
            self._subscribers[key_path].append({
                'callback': callback,
                'component_id': component_id
            })

        logger.info(f"Component {component_id} subscribed to {key_path}")

    def unsubscribe(self, key_path: str, component_id: str):
        """
        Unsubscribe from state changes

        Args:
            key_path: Key path to stop monitoring
            component_id: Component identifier
        """
        with self._lock:
            if key_path in self._subscribers:
                self._subscribers[key_path] = [
                    sub for sub in self._subscribers[key_path]
                    if sub['component_id'] != component_id
                ]

        logger.info(f"Component {component_id} unsubscribed from {key_path}")

    def _notify_subscribers(self, key_path: str, old_value: Any, new_value: Any):
        """Notify subscribers of state changes"""
        try:
            # Notify exact match subscribers
            if key_path in self._subscribers:
                for subscriber in self._subscribers[key_path]:
                    try:
                        subscriber['callback'](key_path, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber {subscriber['component_id']}: {e}")

            # Notify wildcard subscribers (parent paths)
            parts = key_path.split('.')
            for i in range(len(parts)):
                parent_path = '.'.join(parts[:i+1]) + '.*'
                if parent_path in self._subscribers:
                    for subscriber in self._subscribers[parent_path]:
                        try:
                            subscriber['callback'](key_path, old_value, new_value)
                        except Exception as e:
                            logger.error(f"Error notifying wildcard subscriber {subscriber['component_id']}: {e}")

        except Exception as e:
            logger.error(f"Error notifying subscribers for {key_path}: {e}")

    def get_update_history(self, component_id: Optional[str] = None,
                          key_path: Optional[str] = None,
                          limit: int = 20) -> List[StateUpdate]:
        """
        Get state update history

        Args:
            component_id: Filter by component ID
            key_path: Filter by key path
            limit: Maximum number of updates to return

        Returns:
            List of state updates
        """
        with self._lock:
            history = self._update_history.copy()

            # Apply filters
            if component_id:
                history = [u for u in history if u.component_id == component_id]

            if key_path:
                history = [u for u in history if u.state_key == key_path]

            # Return most recent updates
            return history[-limit:]

    def reset_state(self, preserve_keys: Optional[List[str]] = None):
        """
        Reset state to default values

        Args:
            preserve_keys: List of key paths to preserve
        """
        with self._lock:
            preserved_values = {}

            if preserve_keys:
                for key_path in preserve_keys:
                    value = self.get_state(key_path)
                    if value is not None:
                        preserved_values[key_path] = value

            # Reset to defaults
            self._initialize_default_state()

            # Restore preserved values
            for key_path, value in preserved_values.items():
                self.set_state(key_path, value, "system_reset")

        logger.info("State reset to defaults")

    def export_state(self) -> str:
        """Export state as JSON string"""
        try:
            with self._lock:
                return json.dumps(self._state, default=str, indent=2)
        except Exception as e:
            logger.error(f"Error exporting state: {e}")
            return "{}"

    def import_state(self, state_json: str, component_id: str = "import") -> bool:
        """
        Import state from JSON string

        Args:
            state_json: JSON string containing state
            component_id: Component performing import

        Returns:
            True if successful
        """
        try:
            imported_state = json.loads(state_json)

            with self._lock:
                old_state = self._state.copy()
                self._state.update(imported_state)

                # Create update record
                update = StateUpdate(
                    component_id=component_id,
                    state_key="*",
                    old_value=old_state,
                    new_value=self._state.copy()
                )
                self._update_history.append(update)

            logger.info(f"State imported by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error importing state: {e}")
            return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        with self._lock:
            return {
                'total_keys': len(str(self._state).split(':')),
                'subscriber_count': sum(len(subs) for subs in self._subscribers.values()),
                'update_count': len(self._update_history),
                'last_update': self._update_history[-1].timestamp if self._update_history else None,
                'active_subscriptions': list(self._subscribers.keys())
            }


# Global instance for use across the dashboard
shared_state_manager = SharedStateManager()