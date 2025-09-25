"""
Time State Manager
Specialized state management for time controls and real-time updates
Integrates TimeControlManager with SharedStateManager
"""

import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
import threading
import time
from dataclasses import dataclass

from .shared_state import shared_state_manager
from ..components.time_controls import time_control_manager, TimeRange

logger = logging.getLogger(__name__)


@dataclass
class RealTimeUpdate:
    """Real-time update event"""
    component_id: str
    data_type: str
    update_count: int
    last_update: datetime
    error_count: int = 0


class TimeStateManager:
    """
    Specialized time state manager that integrates TimeControlManager
    with SharedStateManager for global time coordination
    """

    def __init__(self):
        """Initialize time state manager"""
        self.shared_state = shared_state_manager
        self.time_control = time_control_manager

        # Real-time update management
        self._update_threads = {}
        self._active_components = {}
        self._update_stats = {}
        self._is_running = False

        # Subscribe to shared state time changes
        self.shared_state.subscribe(
            'filters.time_range',
            self._on_time_range_change,
            'time_state_manager'
        )
        self.shared_state.subscribe(
            'filters.is_realtime',
            self._on_realtime_toggle,
            'time_state_manager'
        )
        self.shared_state.subscribe(
            'filters.refresh_interval',
            self._on_refresh_interval_change,
            'time_state_manager'
        )

        # Initialize time state in shared state
        self._initialize_time_state()

        logger.info("TimeStateManager initialized with real-time update coordination")

    def _initialize_time_state(self):
        """Initialize time-related state in shared state manager"""
        current_window = self.time_control.get_current_window()

        time_state = {
            'time_range': current_window.range_type.value,
            'start_time': current_window.start_time.isoformat(),
            'end_time': current_window.end_time.isoformat(),
            'is_realtime': current_window.is_realtime,
            'refresh_interval': current_window.refresh_interval or 60000,
            'auto_refresh': True,
            'last_update': datetime.now().isoformat(),
            'nasa_mission_time': 'earth',
            'time_sync_enabled': True
        }

        for key, value in time_state.items():
            self.shared_state.set_state(f'filters.{key}', value, 'time_state_manager')

    def set_time_range(self, range_type: str, component_id: str = "unknown",
                      custom_start: Optional[datetime] = None,
                      custom_end: Optional[datetime] = None) -> bool:
        """
        Set global time range and notify all components

        Args:
            range_type: Time range type
            component_id: Component making the change
            custom_start: Custom start time for custom ranges
            custom_end: Custom end time for custom ranges

        Returns:
            True if successful
        """
        try:
            # Update time control manager
            window = self.time_control.set_time_range(range_type, custom_start, custom_end)

            # Update shared state
            updates = {
                'filters.time_range': range_type,
                'filters.start_time': window.start_time.isoformat(),
                'filters.end_time': window.end_time.isoformat(),
                'filters.is_realtime': window.is_realtime,
                'filters.refresh_interval': window.refresh_interval or 60000,
                'filters.last_update': datetime.now().isoformat()
            }

            success = self.shared_state.update_multiple(updates, component_id)

            if success:
                # Update real-time processing if needed
                self._update_realtime_processing()
                logger.info(f"Time range set to {range_type} by {component_id}")

            return success

        except Exception as e:
            logger.error(f"Error setting time range: {e}")
            return False

    def get_current_time_window(self) -> Dict[str, Any]:
        """Get current time window from shared state"""
        try:
            return {
                'range_type': self.shared_state.get_state('filters.time_range'),
                'start_time': datetime.fromisoformat(self.shared_state.get_state('filters.start_time')),
                'end_time': datetime.fromisoformat(self.shared_state.get_state('filters.end_time')),
                'is_realtime': self.shared_state.get_state('filters.is_realtime'),
                'refresh_interval': self.shared_state.get_state('filters.refresh_interval'),
                'auto_refresh': self.shared_state.get_state('filters.auto_refresh')
            }
        except Exception as e:
            logger.error(f"Error getting time window: {e}")
            return {}

    def register_realtime_component(self, component_id: str,
                                   update_callback: Callable,
                                   data_type: str = "telemetry") -> bool:
        """
        Register component for real-time updates

        Args:
            component_id: Unique component identifier
            update_callback: Function to call for updates
            data_type: Type of data being updated

        Returns:
            True if registration successful
        """
        try:
            self._active_components[component_id] = {
                'callback': update_callback,
                'data_type': data_type,
                'registered_at': datetime.now(),
                'update_count': 0,
                'error_count': 0
            }

            self._update_stats[component_id] = RealTimeUpdate(
                component_id=component_id,
                data_type=data_type,
                update_count=0,
                last_update=datetime.now()
            )

            # Start real-time updates if enabled
            if self.shared_state.get_state('filters.is_realtime'):
                self._start_component_updates(component_id)

            logger.info(f"Component {component_id} registered for real-time {data_type} updates")
            return True

        except Exception as e:
            logger.error(f"Error registering component {component_id}: {e}")
            return False

    def unregister_realtime_component(self, component_id: str) -> bool:
        """
        Unregister component from real-time updates

        Args:
            component_id: Component to unregister

        Returns:
            True if successful
        """
        try:
            # Stop update thread
            self._stop_component_updates(component_id)

            # Remove from tracking
            if component_id in self._active_components:
                del self._active_components[component_id]

            if component_id in self._update_stats:
                del self._update_stats[component_id]

            logger.info(f"Component {component_id} unregistered from real-time updates")
            return True

        except Exception as e:
            logger.error(f"Error unregistering component {component_id}: {e}")
            return False

    def pause_realtime_updates(self, component_id: Optional[str] = None):
        """
        Pause real-time updates

        Args:
            component_id: Specific component to pause (None for all)
        """
        if component_id:
            self._stop_component_updates(component_id)
            logger.info(f"Paused real-time updates for {component_id}")
        else:
            self.shared_state.set_state('filters.auto_refresh', False, 'time_state_manager')
            for comp_id in list(self._update_threads.keys()):
                self._stop_component_updates(comp_id)
            logger.info("Paused all real-time updates")

    def resume_realtime_updates(self, component_id: Optional[str] = None):
        """
        Resume real-time updates

        Args:
            component_id: Specific component to resume (None for all)
        """
        if component_id:
            if component_id in self._active_components:
                self._start_component_updates(component_id)
                logger.info(f"Resumed real-time updates for {component_id}")
        else:
            self.shared_state.set_state('filters.auto_refresh', True, 'time_state_manager')
            if self.shared_state.get_state('filters.is_realtime'):
                for comp_id in self._active_components:
                    self._start_component_updates(comp_id)
            logger.info("Resumed all real-time updates")

    def get_realtime_stats(self) -> Dict[str, RealTimeUpdate]:
        """Get real-time update statistics"""
        return self._update_stats.copy()

    def set_nasa_mission_time(self, mission: str, component_id: str = "unknown") -> bool:
        """
        Set NASA mission time mode

        Args:
            mission: 'earth', 'mars', or 'mission'
            component_id: Component making the change

        Returns:
            True if successful
        """
        try:
            valid_modes = ['earth', 'mars', 'mission']
            if mission not in valid_modes:
                logger.error(f"Invalid mission time mode: {mission}")
                return False

            success = self.shared_state.set_state('filters.nasa_mission_time', mission, component_id)

            if success:
                # Adjust time calculations for mission time
                self._adjust_for_mission_time(mission)
                logger.info(f"NASA mission time set to {mission} by {component_id}")

            return success

        except Exception as e:
            logger.error(f"Error setting NASA mission time: {e}")
            return False

    def _on_time_range_change(self, key_path: str, old_value: Any, new_value: Any):
        """Handle time range changes from shared state"""
        try:
            if new_value != old_value:
                window = self.time_control.set_time_range(new_value)

                # Update related state
                self.shared_state.update_multiple({
                    'filters.start_time': window.start_time.isoformat(),
                    'filters.end_time': window.end_time.isoformat(),
                    'filters.is_realtime': window.is_realtime,
                    'filters.refresh_interval': window.refresh_interval or 60000
                }, 'time_state_sync')

                self._update_realtime_processing()

        except Exception as e:
            logger.error(f"Error handling time range change: {e}")

    def _on_realtime_toggle(self, key_path: str, old_value: Any, new_value: Any):
        """Handle real-time toggle changes"""
        try:
            if new_value and not old_value:
                # Starting real-time mode
                for comp_id in self._active_components:
                    self._start_component_updates(comp_id)
                logger.info("Real-time mode enabled")

            elif old_value and not new_value:
                # Stopping real-time mode
                for comp_id in list(self._update_threads.keys()):
                    self._stop_component_updates(comp_id)
                logger.info("Real-time mode disabled")

        except Exception as e:
            logger.error(f"Error handling real-time toggle: {e}")

    def _on_refresh_interval_change(self, key_path: str, old_value: Any, new_value: Any):
        """Handle refresh interval changes"""
        try:
            if new_value != old_value:
                # Restart active update threads with new interval
                active_components = list(self._update_threads.keys())

                for comp_id in active_components:
                    self._stop_component_updates(comp_id)
                    self._start_component_updates(comp_id)

                logger.info(f"Refresh interval changed to {new_value}ms")

        except Exception as e:
            logger.error(f"Error handling refresh interval change: {e}")

    def _start_component_updates(self, component_id: str):
        """Start real-time updates for specific component"""
        try:
            if component_id in self._update_threads:
                return  # Already running

            if component_id not in self._active_components:
                return  # Component not registered

            refresh_interval = self.shared_state.get_state('filters.refresh_interval') or 60000
            auto_refresh = self.shared_state.get_state('filters.auto_refresh')

            if not auto_refresh:
                return

            def update_loop():
                while component_id in self._update_threads:
                    try:
                        component_info = self._active_components[component_id]
                        callback = component_info['callback']

                        # Call component update callback
                        callback()

                        # Update statistics
                        component_info['update_count'] += 1
                        self._update_stats[component_id].update_count += 1
                        self._update_stats[component_id].last_update = datetime.now()

                        # Sleep for refresh interval
                        time.sleep(refresh_interval / 1000.0)

                    except Exception as e:
                        logger.error(f"Error in update loop for {component_id}: {e}")
                        component_info['error_count'] += 1
                        self._update_stats[component_id].error_count += 1
                        time.sleep(5)  # Error backoff

            # Start update thread
            thread = threading.Thread(target=update_loop, daemon=True)
            self._update_threads[component_id] = thread
            thread.start()

            logger.debug(f"Started real-time updates for {component_id}")

        except Exception as e:
            logger.error(f"Error starting updates for {component_id}: {e}")

    def _stop_component_updates(self, component_id: str):
        """Stop real-time updates for specific component"""
        try:
            if component_id in self._update_threads:
                # Thread will exit when removed from _update_threads
                del self._update_threads[component_id]
                logger.debug(f"Stopped real-time updates for {component_id}")

        except Exception as e:
            logger.error(f"Error stopping updates for {component_id}: {e}")

    def _update_realtime_processing(self):
        """Update real-time processing based on current state"""
        try:
            is_realtime = self.shared_state.get_state('filters.is_realtime')
            auto_refresh = self.shared_state.get_state('filters.auto_refresh')

            if is_realtime and auto_refresh:
                # Start updates for all registered components
                for comp_id in self._active_components:
                    if comp_id not in self._update_threads:
                        self._start_component_updates(comp_id)
            else:
                # Stop all updates
                for comp_id in list(self._update_threads.keys()):
                    self._stop_component_updates(comp_id)

        except Exception as e:
            logger.error(f"Error updating real-time processing: {e}")

    def _adjust_for_mission_time(self, mission: str):
        """Adjust time calculations for mission-specific time"""
        try:
            if mission == 'mars':
                # Mars sol is ~24h 37m
                # Adjust time ranges proportionally
                pass  # Implement Mars time adjustments if needed
            elif mission == 'mission':
                # Use mission elapsed time
                pass  # Implement mission time adjustments if needed

            # For 'earth' mode, use standard Earth time (default)

        except Exception as e:
            logger.error(f"Error adjusting for mission time: {e}")


# Global instance for use across the dashboard
time_state_manager = TimeStateManager()