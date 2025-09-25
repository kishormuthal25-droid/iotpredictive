"""
Filter State Persistence
Handles filter state persistence across page navigation and browser sessions
"""

import logging
from typing import Dict, Any, Optional, List
import json
import base64
from urllib.parse import urlencode, parse_qs
from datetime import datetime
import uuid

from .shared_state import shared_state_manager
from ..components.filter_manager import filter_manager, FilterCriteria

logger = logging.getLogger(__name__)


class FilterStatePersistence:
    """
    Manages filter state persistence across page navigation and browser sessions
    Handles localStorage, URL parameters, and session restoration
    """

    def __init__(self):
        """Initialize filter state persistence"""
        self.shared_state = shared_state_manager
        self.filter_manager = filter_manager

        # Persistence configuration
        self.storage_key = "iot_dashboard_filters"
        self.url_param_key = "filters"
        self.session_key = "iot_session_id"

        # State tracking
        self.current_session_id = self._generate_session_id()
        self.last_saved_state = {}
        self.persistence_enabled = True

        # Subscribe to state changes for auto-save
        self._setup_auto_save()

        logger.info("FilterStatePersistence initialized")

    def save_filter_state(self, include_url: bool = False) -> Dict[str, Any]:
        """
        Save current filter state to localStorage and optionally URL

        Args:
            include_url: Include state in URL parameters

        Returns:
            Saved state dictionary
        """
        try:
            # Get current filter state
            filter_state = self._get_current_filter_state()

            # Add metadata
            state_with_metadata = {
                'filters': filter_state,
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'session_id': self.current_session_id,
                    'page_path': self.shared_state.get_state('ui.current_page'),
                    'version': '1.0'
                }
            }

            # Save to localStorage (client-side implementation needed)
            self._save_to_storage(state_with_metadata)

            # Save to URL if requested
            if include_url:
                self._save_to_url(filter_state)

            self.last_saved_state = state_with_metadata.copy()

            logger.info("Filter state saved successfully")
            return state_with_metadata

        except Exception as e:
            logger.error(f"Error saving filter state: {e}")
            return {}

    def load_filter_state(self, source: str = "auto") -> bool:
        """
        Load filter state from specified source

        Args:
            source: 'localStorage', 'url', 'auto' (try URL first, then localStorage)

        Returns:
            True if state loaded successfully
        """
        try:
            loaded_state = None

            if source == "url" or source == "auto":
                loaded_state = self._load_from_url()
                if loaded_state:
                    logger.info("Filter state loaded from URL")

            if not loaded_state and (source == "localStorage" or source == "auto"):
                loaded_state = self._load_from_storage()
                if loaded_state:
                    logger.info("Filter state loaded from localStorage")

            if loaded_state:
                return self._apply_loaded_state(loaded_state)
            else:
                logger.info("No saved filter state found")
                return False

        except Exception as e:
            logger.error(f"Error loading filter state: {e}")
            return False

    def clear_persisted_state(self, clear_url: bool = True) -> bool:
        """
        Clear all persisted filter state

        Args:
            clear_url: Also clear URL parameters

        Returns:
            True if successful
        """
        try:
            # Clear localStorage (client-side implementation needed)
            self._clear_storage()

            # Clear URL if requested
            if clear_url:
                self._clear_url()

            # Reset current state
            self.last_saved_state = {}

            logger.info("Persisted filter state cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing persisted state: {e}")
            return False

    def create_shareable_url(self, base_url: str, filters: Optional[FilterCriteria] = None) -> str:
        """
        Create shareable URL with current filter state

        Args:
            base_url: Base URL to append filters to
            filters: Filter criteria (uses current if None)

        Returns:
            URL with encoded filter parameters
        """
        try:
            if filters is None:
                filter_state = self._get_current_filter_state()
            else:
                filter_state = self._filter_criteria_to_dict(filters)

            # Encode filter state
            encoded_filters = self._encode_filter_state(filter_state)

            # Create URL parameters
            params = {self.url_param_key: encoded_filters}

            # Add current page if available
            current_page = self.shared_state.get_state('ui.current_page')
            if current_page and current_page != '/':
                params['page'] = current_page

            # Construct URL
            param_string = urlencode(params)
            separator = '&' if '?' in base_url else '?'

            shareable_url = f"{base_url}{separator}{param_string}"

            logger.info("Shareable URL created")
            return shareable_url

        except Exception as e:
            logger.error(f"Error creating shareable URL: {e}")
            return base_url

    def restore_from_shareable_url(self, url: str) -> bool:
        """
        Restore filter state from shareable URL

        Args:
            url: URL containing encoded filter state

        Returns:
            True if restoration successful
        """
        try:
            # Parse URL parameters
            if '?' not in url:
                return False

            query_part = url.split('?', 1)[1]
            params = parse_qs(query_part)

            # Get encoded filter data
            if self.url_param_key not in params:
                return False

            encoded_filters = params[self.url_param_key][0]
            filter_state = self._decode_filter_state(encoded_filters)

            if filter_state:
                return self._apply_loaded_state({'filters': filter_state})

            return False

        except Exception as e:
            logger.error(f"Error restoring from shareable URL: {e}")
            return False

    def get_filter_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get filter state history from storage

        Args:
            limit: Maximum number of history entries

        Returns:
            List of historical filter states
        """
        try:
            # This would be implemented with actual storage backend
            # For now, return filter manager history
            history = self.filter_manager.filter_history[-limit:]

            formatted_history = []
            for entry in history:
                formatted_history.append({
                    'timestamp': entry['timestamp'].isoformat(),
                    'filters': self._filter_criteria_to_dict(entry['filters']),
                    'summary': entry['result_summary']
                })

            return formatted_history

        except Exception as e:
            logger.error(f"Error getting filter history: {e}")
            return []

    def export_filter_configuration(self) -> str:
        """
        Export current filter configuration as JSON

        Returns:
            JSON string of filter configuration
        """
        try:
            config = {
                'current_filters': self._get_current_filter_state(),
                'presets': self.filter_manager.get_filter_presets(),
                'session_info': {
                    'session_id': self.current_session_id,
                    'exported_at': datetime.now().isoformat(),
                    'current_page': self.shared_state.get_state('ui.current_page')
                }
            }

            return json.dumps(config, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error exporting filter configuration: {e}")
            return "{}"

    def import_filter_configuration(self, config_json: str) -> bool:
        """
        Import filter configuration from JSON

        Args:
            config_json: JSON string containing filter configuration

        Returns:
            True if import successful
        """
        try:
            config = json.loads(config_json)

            if 'current_filters' in config:
                filter_state = config['current_filters']
                return self._apply_loaded_state({'filters': filter_state})

            return False

        except Exception as e:
            logger.error(f"Error importing filter configuration: {e}")
            return False

    def _get_current_filter_state(self) -> Dict[str, Any]:
        """Get current filter state from shared state and filter manager"""
        try:
            # Get from shared state
            shared_filters = {
                'equipment_id': self.shared_state.get_state('selections.equipment_id'),
                'sensor_id': self.shared_state.get_state('selections.sensor_id'),
                'metric_id': self.shared_state.get_state('selections.metric_id'),
                'subsystem': self.shared_state.get_state('selections.subsystem'),
                'time_range': self.shared_state.get_state('filters.time_range'),
                'chart_type': self.shared_state.get_state('filters.chart_type'),
                'show_anomalies': self.shared_state.get_state('filters.show_anomalies'),
                'show_thresholds': self.shared_state.get_state('filters.show_thresholds'),
                'is_realtime': self.shared_state.get_state('filters.is_realtime')
            }

            # Get from filter manager
            active_filters = self.filter_manager.active_filters
            filter_dict = self._filter_criteria_to_dict(active_filters)

            # Merge states
            combined_state = {**shared_filters, **filter_dict}

            # Remove None values
            return {k: v for k, v in combined_state.items() if v is not None}

        except Exception as e:
            logger.error(f"Error getting current filter state: {e}")
            return {}

    def _filter_criteria_to_dict(self, criteria: FilterCriteria) -> Dict[str, Any]:
        """Convert FilterCriteria to dictionary"""
        try:
            return {
                'equipment_ids': criteria.equipment_ids,
                'sensor_ids': criteria.sensor_ids,
                'metric_ids': criteria.metric_ids,
                'subsystems': criteria.subsystems,
                'criticality_levels': criteria.criticality_levels,
                'spacecraft': criteria.spacecraft,
                'min_value': criteria.min_value,
                'max_value': criteria.max_value,
                'anomaly_threshold': criteria.anomaly_threshold,
                'include_anomalies_only': criteria.include_anomalies_only,
                'include_normal_only': criteria.include_normal_only,
                'active_alerts_only': criteria.active_alerts_only
            }
        except Exception as e:
            logger.error(f"Error converting filter criteria: {e}")
            return {}

    def _dict_to_filter_criteria(self, data: Dict[str, Any]) -> FilterCriteria:
        """Convert dictionary to FilterCriteria"""
        try:
            return FilterCriteria(
                equipment_ids=data.get('equipment_ids'),
                sensor_ids=data.get('sensor_ids'),
                metric_ids=data.get('metric_ids'),
                subsystems=data.get('subsystems'),
                criticality_levels=data.get('criticality_levels'),
                spacecraft=data.get('spacecraft'),
                min_value=data.get('min_value'),
                max_value=data.get('max_value'),
                anomaly_threshold=data.get('anomaly_threshold'),
                include_anomalies_only=data.get('include_anomalies_only', False),
                include_normal_only=data.get('include_normal_only', False),
                active_alerts_only=data.get('active_alerts_only', False)
            )
        except Exception as e:
            logger.error(f"Error converting to filter criteria: {e}")
            return FilterCriteria()

    def _apply_loaded_state(self, state_data: Dict[str, Any]) -> bool:
        """Apply loaded state to current session"""
        try:
            if 'filters' not in state_data:
                return False

            filter_state = state_data['filters']

            # Update shared state
            shared_updates = {}
            for key in ['equipment_id', 'sensor_id', 'metric_id', 'subsystem']:
                if key in filter_state:
                    shared_updates[f'selections.{key}'] = filter_state[key]

            for key in ['time_range', 'chart_type', 'show_anomalies', 'show_thresholds', 'is_realtime']:
                if key in filter_state:
                    shared_updates[f'filters.{key}'] = filter_state[key]

            if shared_updates:
                self.shared_state.update_multiple(shared_updates, 'filter_persistence')

            # Update filter manager
            filter_criteria = self._dict_to_filter_criteria(filter_state)
            self.filter_manager.active_filters = filter_criteria

            logger.info("Filter state applied successfully")
            return True

        except Exception as e:
            logger.error(f"Error applying loaded state: {e}")
            return False

    def _encode_filter_state(self, state: Dict[str, Any]) -> str:
        """Encode filter state for URL"""
        try:
            # Convert to JSON and encode
            json_str = json.dumps(state, default=str)
            encoded = base64.urlsafe_b64encode(json_str.encode()).decode()
            return encoded
        except Exception as e:
            logger.error(f"Error encoding filter state: {e}")
            return ""

    def _decode_filter_state(self, encoded: str) -> Dict[str, Any]:
        """Decode filter state from URL"""
        try:
            # Decode and parse JSON
            json_str = base64.urlsafe_b64decode(encoded.encode()).decode()
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error decoding filter state: {e}")
            return {}

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return str(uuid.uuid4())

    def _setup_auto_save(self):
        """Setup automatic saving on state changes"""
        try:
            # Subscribe to relevant state changes
            self.shared_state.subscribe(
                'selections.*',
                self._on_auto_save_trigger,
                'filter_persistence'
            )
            self.shared_state.subscribe(
                'filters.*',
                self._on_auto_save_trigger,
                'filter_persistence'
            )
        except Exception as e:
            logger.error(f"Error setting up auto-save: {e}")

    def _on_auto_save_trigger(self, key_path: str, old_value: Any, new_value: Any):
        """Handle auto-save triggers"""
        try:
            if self.persistence_enabled and new_value != old_value:
                # Auto-save with debouncing (simplified)
                self.save_filter_state(include_url=False)
        except Exception as e:
            logger.error(f"Error in auto-save trigger: {e}")

    # Storage methods (to be implemented with actual client-side storage)

    def _save_to_storage(self, state: Dict[str, Any]):
        """Save state to localStorage (client-side implementation needed)"""
        # This would use Dash clientside callbacks or custom components
        # to interact with browser localStorage
        logger.debug("Saving to storage (client-side implementation needed)")

    def _load_from_storage(self) -> Optional[Dict[str, Any]]:
        """Load state from localStorage (client-side implementation needed)"""
        # This would use Dash clientside callbacks or custom components
        # to read from browser localStorage
        logger.debug("Loading from storage (client-side implementation needed)")
        return None

    def _clear_storage(self):
        """Clear localStorage (client-side implementation needed)"""
        logger.debug("Clearing storage (client-side implementation needed)")

    def _save_to_url(self, state: Dict[str, Any]):
        """Save state to URL parameters (requires browser URL manipulation)"""
        logger.debug("Saving to URL (browser implementation needed)")

    def _load_from_url(self) -> Optional[Dict[str, Any]]:
        """Load state from URL parameters"""
        # This would be implemented with actual URL parsing
        logger.debug("Loading from URL (implementation needed)")
        return None

    def _clear_url(self):
        """Clear URL parameters"""
        logger.debug("Clearing URL (browser implementation needed)")


# Global instance for use across the dashboard
filter_state_persistence = FilterStatePersistence()