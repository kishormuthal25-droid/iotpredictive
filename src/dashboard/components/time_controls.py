"""
Time Control Manager
Unified time range management for synchronized time controls across the dashboard
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class TimeRange(Enum):
    """Predefined time range options"""
    REAL_TIME = "real_time"
    LAST_1M = "1m"
    LAST_5M = "5m"
    LAST_15M = "15m"
    LAST_1H = "1h"
    LAST_6H = "6h"
    LAST_24H = "24h"
    LAST_7D = "7d"
    LAST_30D = "30d"
    CUSTOM = "custom"


@dataclass
class TimeWindow:
    """Time window configuration"""
    start_time: datetime
    end_time: datetime
    range_type: TimeRange
    label: str
    refresh_interval: Optional[int] = None  # Milliseconds
    is_realtime: bool = False


@dataclass
class TimeControlConfig:
    """Configuration for time controls"""
    default_range: TimeRange = TimeRange.LAST_1H
    enable_realtime: bool = True
    enable_custom_range: bool = True
    show_refresh_button: bool = True
    auto_refresh_intervals: Dict[str, int] = None  # Range -> milliseconds

    def __post_init__(self):
        if self.auto_refresh_intervals is None:
            self.auto_refresh_intervals = {
                TimeRange.REAL_TIME.value: 1000,  # 1 second
                TimeRange.LAST_1M.value: 5000,    # 5 seconds
                TimeRange.LAST_5M.value: 10000,   # 10 seconds
                TimeRange.LAST_15M.value: 30000,  # 30 seconds
                TimeRange.LAST_1H.value: 60000,   # 1 minute
                TimeRange.LAST_6H.value: 300000,  # 5 minutes
                TimeRange.LAST_24H.value: 600000, # 10 minutes
                TimeRange.LAST_7D.value: 1800000, # 30 minutes
                TimeRange.LAST_30D.value: 3600000 # 1 hour
            }


class TimeControlManager:
    """
    Unified time control manager for synchronized time range selection
    Handles time windows, real-time updates, and time synchronization across views
    """

    def __init__(self, config: Optional[TimeControlConfig] = None):
        """Initialize time control manager

        Args:
            config: Time control configuration
        """
        self.config = config or TimeControlConfig()
        self.current_window = self._get_default_window()
        self.subscribers = {}  # Component subscriptions for time updates
        self.global_time_state = {
            'current_range': self.config.default_range,
            'start_time': self.current_window.start_time,
            'end_time': self.current_window.end_time,
            'is_realtime': False,
            'last_update': datetime.now()
        }

        # NASA mission-specific time ranges
        self.nasa_time_ranges = {
            'smap_orbit': timedelta(minutes=99),      # SMAP orbital period
            'msl_sol': timedelta(hours=24, minutes=37), # Mars sol duration
            'communication_window': timedelta(minutes=15), # Typical comm window
            'mission_critical': timedelta(hours=1),    # Critical operations window
        }

        logger.info("TimeControlManager initialized with NASA mission-specific ranges")

    def get_time_range_options(self, include_nasa_ranges: bool = True) -> List[Dict[str, Any]]:
        """
        Get available time range options for dropdowns

        Args:
            include_nasa_ranges: Include NASA mission-specific ranges

        Returns:
            List of time range options for dropdown
        """
        options = [
            {"label": "🔴 Real-time", "value": TimeRange.REAL_TIME.value},
            {"label": "📊 Last 1 minute", "value": TimeRange.LAST_1M.value},
            {"label": "📊 Last 5 minutes", "value": TimeRange.LAST_5M.value},
            {"label": "📊 Last 15 minutes", "value": TimeRange.LAST_15M.value},
            {"label": "⏰ Last 1 hour", "value": TimeRange.LAST_1H.value},
            {"label": "⏰ Last 6 hours", "value": TimeRange.LAST_6H.value},
            {"label": "📅 Last 24 hours", "value": TimeRange.LAST_24H.value},
            {"label": "📅 Last 7 days", "value": TimeRange.LAST_7D.value},
            {"label": "📅 Last 30 days", "value": TimeRange.LAST_30D.value},
            {"label": "🔧 Custom Range", "value": TimeRange.CUSTOM.value}
        ]

        if include_nasa_ranges:
            nasa_options = [
                {"label": "🛰️ SMAP Orbit (99min)", "value": "smap_orbit"},
                {"label": "🚗 MSL Sol (24h37m)", "value": "msl_sol"},
                {"label": "📡 Comm Window (15min)", "value": "communication_window"},
                {"label": "🚨 Mission Critical (1h)", "value": "mission_critical"}
            ]
            # Insert NASA options after real-time but before standard ranges
            options = options[:1] + nasa_options + options[1:]

        return options

    def get_refresh_interval_options(self) -> List[Dict[str, Any]]:
        """Get refresh interval options for auto-refresh controls"""
        return [
            {"label": "🔴 Real-time (1s)", "value": 1000},
            {"label": "⚡ Fast (5s)", "value": 5000},
            {"label": "🔄 Normal (10s)", "value": 10000},
            {"label": "⏳ Slow (30s)", "value": 30000},
            {"label": "⏸️ Manual", "value": 0}
        ]

    def set_time_range(self, range_type: str, custom_start: Optional[datetime] = None,
                      custom_end: Optional[datetime] = None) -> TimeWindow:
        """
        Set current time range

        Args:
            range_type: Time range type (enum value or NASA range key)
            custom_start: Custom start time for CUSTOM range
            custom_end: Custom end time for CUSTOM range

        Returns:
            Updated time window
        """
        try:
            if range_type == TimeRange.CUSTOM.value:
                if not custom_start or not custom_end:
                    raise ValueError("Custom range requires start and end times")

                self.current_window = TimeWindow(
                    start_time=custom_start,
                    end_time=custom_end,
                    range_type=TimeRange.CUSTOM,
                    label=f"Custom: {custom_start.strftime('%m/%d %H:%M')} - {custom_end.strftime('%m/%d %H:%M')}",
                    is_realtime=False
                )

            elif range_type in self.nasa_time_ranges:
                # NASA mission-specific range
                end_time = datetime.now()
                start_time = end_time - self.nasa_time_ranges[range_type]

                self.current_window = TimeWindow(
                    start_time=start_time,
                    end_time=end_time,
                    range_type=TimeRange.CUSTOM,  # Treat as custom for processing
                    label=f"NASA {range_type.replace('_', ' ').title()}",
                    refresh_interval=self.config.auto_refresh_intervals.get(TimeRange.LAST_1H.value),
                    is_realtime=False
                )

            else:
                # Standard time range
                time_range = TimeRange(range_type)
                self.current_window = self._calculate_time_window(time_range)

            # Update global state
            self.global_time_state.update({
                'current_range': range_type,
                'start_time': self.current_window.start_time,
                'end_time': self.current_window.end_time,
                'is_realtime': self.current_window.is_realtime,
                'last_update': datetime.now()
            })

            # Notify subscribers
            self._notify_subscribers()

            logger.info(f"Time range set to: {self.current_window.label}")
            return self.current_window

        except Exception as e:
            logger.error(f"Error setting time range: {e}")
            return self.current_window

    def get_current_window(self) -> TimeWindow:
        """Get current time window"""
        # Update real-time windows
        if self.current_window.is_realtime:
            self.current_window = self._calculate_time_window(TimeRange.REAL_TIME)
            self.global_time_state.update({
                'start_time': self.current_window.start_time,
                'end_time': self.current_window.end_time,
                'last_update': datetime.now()
            })

        return self.current_window

    def get_refresh_interval(self, range_type: Optional[str] = None) -> int:
        """
        Get appropriate refresh interval for time range

        Args:
            range_type: Time range type (uses current if None)

        Returns:
            Refresh interval in milliseconds
        """
        if range_type is None:
            range_type = self.global_time_state['current_range']

        return self.config.auto_refresh_intervals.get(
            range_type,
            self.config.auto_refresh_intervals[TimeRange.LAST_1H.value]
        )

    def filter_data_by_time(self, data: pd.DataFrame,
                           timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Filter data by current time window

        Args:
            data: DataFrame to filter
            timestamp_column: Name of timestamp column

        Returns:
            Filtered DataFrame
        """
        try:
            if data.empty or timestamp_column not in data.columns:
                return data

            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
                data[timestamp_column] = pd.to_datetime(data[timestamp_column])

            # Get current window
            window = self.get_current_window()

            # Filter data
            filtered_data = data[
                (data[timestamp_column] >= window.start_time) &
                (data[timestamp_column] <= window.end_time)
            ]

            logger.debug(f"Filtered data: {len(filtered_data)}/{len(data)} records in time window")
            return filtered_data

        except Exception as e:
            logger.error(f"Error filtering data by time: {e}")
            return data

    def subscribe_to_time_updates(self, component_id: str, callback_func):
        """
        Subscribe component to time updates

        Args:
            component_id: Unique component identifier
            callback_func: Function to call on time updates
        """
        self.subscribers[component_id] = callback_func
        logger.info(f"Component {component_id} subscribed to time updates")

    def unsubscribe_from_time_updates(self, component_id: str):
        """
        Unsubscribe component from time updates

        Args:
            component_id: Component identifier to unsubscribe
        """
        if component_id in self.subscribers:
            del self.subscribers[component_id]
            logger.info(f"Component {component_id} unsubscribed from time updates")

    def get_time_state(self) -> Dict[str, Any]:
        """Get current global time state"""
        return self.global_time_state.copy()

    def create_time_display_string(self, include_range: bool = True) -> str:
        """
        Create human-readable time display string

        Args:
            include_range: Include range type in display

        Returns:
            Formatted time display string
        """
        window = self.get_current_window()

        if window.is_realtime:
            return "🔴 Real-time streaming"

        start_str = window.start_time.strftime("%m/%d %H:%M")
        end_str = window.end_time.strftime("%m/%d %H:%M")

        if include_range:
            return f"{window.label}: {start_str} - {end_str}"
        else:
            return f"{start_str} - {end_str}"

    def get_time_controls_layout(self, component_prefix: str = ""):
        """
        Get standard time controls layout components

        Args:
            component_prefix: Prefix for component IDs

        Returns:
            Dash components for time controls
        """
        from dash import html, dcc
        import dash_bootstrap_components as dbc

        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("⏰ Time Range", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id=f"{component_prefix}time-range-select",
                        options=self.get_time_range_options(include_nasa_ranges=True),
                        value=self.config.default_range.value,
                        clearable=False,
                        className="mb-2"
                    )
                ], width=4),

                dbc.Col([
                    html.Label("🔄 Refresh Rate", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id=f"{component_prefix}refresh-interval-select",
                        options=self.get_refresh_interval_options(),
                        value=self.get_refresh_interval(),
                        clearable=False,
                        className="mb-2"
                    )
                ], width=3),

                dbc.Col([
                    html.Label("📅 Custom Range", className="fw-bold mb-2"),
                    dcc.DatePickerRange(
                        id=f"{component_prefix}custom-date-range",
                        start_date=datetime.now().date() - timedelta(days=1),
                        end_date=datetime.now().date(),
                        display_format='MM/DD/YYYY',
                        style={'display': 'none'}  # Hidden by default
                    )
                ], width=3),

                dbc.Col([
                    html.Label("Actions", className="fw-bold mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("🔄", id=f"{component_prefix}refresh-now-btn",
                                 size="sm", color="primary", title="Refresh Now"),
                        dbc.Button("⏸️", id=f"{component_prefix}pause-updates-btn",
                                 size="sm", color="secondary", title="Pause Updates")
                    ], size="sm")
                ], width=2)
            ], className="time-controls-row")
        ], className="time-controls-container")

    def validate_time_window(self, start_time: datetime, end_time: datetime) -> Dict[str, bool]:
        """
        Validate time window parameters

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Validation results
        """
        validation = {
            'valid_range': end_time > start_time,
            'reasonable_duration': (end_time - start_time) <= timedelta(days=365),
            'not_future': end_time <= datetime.now() + timedelta(minutes=5),
            'not_too_old': start_time >= datetime.now() - timedelta(days=365)
        }

        validation['all_valid'] = all(validation.values())
        return validation

    def _get_default_window(self) -> TimeWindow:
        """Get default time window"""
        return self._calculate_time_window(self.config.default_range)

    def _calculate_time_window(self, range_type: TimeRange) -> TimeWindow:
        """Calculate time window for given range type"""
        now = datetime.now()

        if range_type == TimeRange.REAL_TIME:
            return TimeWindow(
                start_time=now - timedelta(minutes=5),  # Last 5 minutes for real-time
                end_time=now,
                range_type=range_type,
                label="Real-time",
                refresh_interval=self.config.auto_refresh_intervals[range_type.value],
                is_realtime=True
            )

        time_deltas = {
            TimeRange.LAST_1M: timedelta(minutes=1),
            TimeRange.LAST_5M: timedelta(minutes=5),
            TimeRange.LAST_15M: timedelta(minutes=15),
            TimeRange.LAST_1H: timedelta(hours=1),
            TimeRange.LAST_6H: timedelta(hours=6),
            TimeRange.LAST_24H: timedelta(hours=24),
            TimeRange.LAST_7D: timedelta(days=7),
            TimeRange.LAST_30D: timedelta(days=30)
        }

        delta = time_deltas.get(range_type, timedelta(hours=1))
        start_time = now - delta

        return TimeWindow(
            start_time=start_time,
            end_time=now,
            range_type=range_type,
            label=range_type.value.replace('_', ' ').title(),
            refresh_interval=self.config.auto_refresh_intervals.get(range_type.value),
            is_realtime=False
        )

    def _notify_subscribers(self):
        """Notify all subscribers of time updates"""
        for component_id, callback_func in self.subscribers.items():
            try:
                callback_func(self.global_time_state)
            except Exception as e:
                logger.error(f"Error notifying subscriber {component_id}: {e}")


# Global instance for use across the dashboard
time_control_manager = TimeControlManager()