"""
Filter Manager
Cross-view filtering system for Equipment→Sensor→Metric with persistence
"""

import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FilterCriteria:
    """Filter criteria for data filtering"""
    equipment_ids: Optional[List[str]] = None
    sensor_ids: Optional[List[str]] = None
    metric_ids: Optional[List[str]] = None
    subsystems: Optional[List[str]] = None
    criticality_levels: Optional[List[str]] = None
    spacecraft: Optional[List[str]] = None  # ['smap', 'msl']

    # Value-based filters
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    anomaly_threshold: Optional[float] = None

    # Time-based filters (handled by TimeStateManager)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Status filters
    include_anomalies_only: bool = False
    include_normal_only: bool = False
    active_alerts_only: bool = False


@dataclass
class FilterResult:
    """Result of applying filters"""
    filtered_data: pd.DataFrame
    original_count: int
    filtered_count: int
    filter_summary: Dict[str, Any]
    equipment_matched: Set[str]
    sensor_matched: Set[str]
    applied_filters: FilterCriteria


class FilterManager:
    """
    Cross-view filtering system for consistent Equipment→Sensor→Metric filtering
    Integrates with DropdownStateManager and SharedStateManager
    """

    def __init__(self):
        """Initialize filter manager"""
        # Import managers (avoid circular imports)
        self.dropdown_manager = None
        self.shared_state = None
        self._initialize_managers()

        # Filter state
        self.active_filters = FilterCriteria()
        self.filter_history = []
        self.max_history = 50

        # NASA equipment categorization
        self.nasa_categories = {
            'critical_systems': {
                'smap': ['POWER', 'ATTITUDE'],
                'msl': ['MOBILITY', 'POWER', 'NAVIGATION']
            },
            'high_priority': {
                'smap': ['COMMUNICATION', 'THERMAL', 'PAYLOAD'],
                'msl': ['COMMUNICATION', 'SCIENTIFIC']
            },
            'medium_priority': {
                'smap': [],
                'msl': ['ENVIRONMENTAL']
            }
        }

        # Common filter presets
        self.filter_presets = {
            'critical_equipment': FilterCriteria(
                criticality_levels=['CRITICAL'],
                equipment_ids=self._get_critical_equipment_ids()
            ),
            'smap_only': FilterCriteria(
                spacecraft=['smap']
            ),
            'msl_only': FilterCriteria(
                spacecraft=['msl']
            ),
            'power_systems': FilterCriteria(
                subsystems=['POWER']
            ),
            'communication_systems': FilterCriteria(
                subsystems=['COMMUNICATION']
            ),
            'anomalies_only': FilterCriteria(
                include_anomalies_only=True
            ),
            'recent_alerts': FilterCriteria(
                active_alerts_only=True
            )
        }

        logger.info("FilterManager initialized with NASA-specific filtering")

    def _initialize_managers(self):
        """Initialize manager dependencies"""
        try:
            from .dropdown_manager import dropdown_state_manager
            from ..state.shared_state import shared_state_manager

            self.dropdown_manager = dropdown_state_manager
            self.shared_state = shared_state_manager

            # Subscribe to filter state changes
            self.shared_state.subscribe(
                'selections.*',
                self._on_selection_change,
                'filter_manager'
            )

        except ImportError as e:
            logger.warning(f"Could not import managers: {e}")

    def apply_filters(self, data: pd.DataFrame,
                     filters: Optional[FilterCriteria] = None,
                     update_state: bool = True) -> FilterResult:
        """
        Apply filters to data

        Args:
            data: DataFrame to filter
            filters: Filter criteria (uses active filters if None)
            update_state: Update shared state with filter results

        Returns:
            FilterResult with filtered data and statistics
        """
        try:
            if filters is None:
                filters = self.active_filters

            original_count = len(data)
            filtered_data = data.copy()
            equipment_matched = set()
            sensor_matched = set()

            # Apply equipment filters
            if filters.equipment_ids:
                if 'equipment_id' in filtered_data.columns:
                    mask = filtered_data['equipment_id'].isin(filters.equipment_ids)
                    filtered_data = filtered_data[mask]
                    equipment_matched.update(filtered_data['equipment_id'].unique())

            # Apply sensor filters
            if filters.sensor_ids:
                if 'sensor_id' in filtered_data.columns:
                    mask = filtered_data['sensor_id'].isin(filters.sensor_ids)
                    filtered_data = filtered_data[mask]
                    sensor_matched.update(filtered_data['sensor_id'].unique())

            # Apply subsystem filters
            if filters.subsystems:
                if 'subsystem' in filtered_data.columns:
                    mask = filtered_data['subsystem'].isin(filters.subsystems)
                    filtered_data = filtered_data[mask]

            # Apply spacecraft filters
            if filters.spacecraft:
                if 'spacecraft' in filtered_data.columns:
                    mask = filtered_data['spacecraft'].isin(filters.spacecraft)
                    filtered_data = filtered_data[mask]
                elif 'equipment_id' in filtered_data.columns:
                    # Infer spacecraft from equipment_id
                    spacecraft_mask = False
                    for spacecraft in filters.spacecraft:
                        if spacecraft.lower() == 'smap':
                            spacecraft_mask |= filtered_data['equipment_id'].str.startswith('SMAP')
                        elif spacecraft.lower() == 'msl':
                            spacecraft_mask |= filtered_data['equipment_id'].str.startswith('MSL')
                    filtered_data = filtered_data[spacecraft_mask]

            # Apply criticality filters
            if filters.criticality_levels:
                if 'criticality' in filtered_data.columns:
                    mask = filtered_data['criticality'].isin(filters.criticality_levels)
                    filtered_data = filtered_data[mask]

            # Apply value-based filters
            if filters.min_value is not None or filters.max_value is not None:
                value_column = self._find_value_column(filtered_data)
                if value_column:
                    if filters.min_value is not None:
                        filtered_data = filtered_data[filtered_data[value_column] >= filters.min_value]
                    if filters.max_value is not None:
                        filtered_data = filtered_data[filtered_data[value_column] <= filters.max_value]

            # Apply anomaly filters
            if filters.include_anomalies_only:
                if 'anomaly_detected' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['anomaly_detected'] == True]
                elif 'anomaly_score' in filtered_data.columns and filters.anomaly_threshold:
                    filtered_data = filtered_data[filtered_data['anomaly_score'] > filters.anomaly_threshold]

            if filters.include_normal_only:
                if 'anomaly_detected' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['anomaly_detected'] == False]

            # Apply alert filters
            if filters.active_alerts_only:
                if 'alert_active' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['alert_active'] == True]

            # Create filter summary
            filter_summary = {
                'equipment_count': len(equipment_matched),
                'sensor_count': len(sensor_matched),
                'reduction_ratio': (original_count - len(filtered_data)) / original_count if original_count > 0 else 0,
                'filters_applied': self._get_applied_filter_summary(filters)
            }

            result = FilterResult(
                filtered_data=filtered_data,
                original_count=original_count,
                filtered_count=len(filtered_data),
                filter_summary=filter_summary,
                equipment_matched=equipment_matched,
                sensor_matched=sensor_matched,
                applied_filters=filters
            )

            # Update shared state if requested
            if update_state and self.shared_state:
                self._update_filter_state(result)

            # Add to history
            self.filter_history.append({
                'timestamp': datetime.now(),
                'filters': filters,
                'result_summary': filter_summary
            })

            # Trim history
            if len(self.filter_history) > self.max_history:
                self.filter_history = self.filter_history[-self.max_history:]

            logger.info(f"Filters applied: {original_count} → {len(filtered_data)} records")
            return result

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return FilterResult(
                filtered_data=data,
                original_count=len(data),
                filtered_count=len(data),
                filter_summary={},
                equipment_matched=set(),
                sensor_matched=set(),
                applied_filters=filters or FilterCriteria()
            )

    def set_equipment_filter(self, equipment_ids: List[str],
                           component_id: str = "unknown") -> bool:
        """
        Set equipment filter and update related dropdowns

        Args:
            equipment_ids: List of equipment IDs to filter by
            component_id: Component making the change

        Returns:
            True if successful
        """
        try:
            self.active_filters.equipment_ids = equipment_ids

            # Update shared state
            if self.shared_state:
                self.shared_state.set_state('selections.equipment_id',
                                          equipment_ids[0] if equipment_ids else None,
                                          component_id)

            # Get available sensors for selected equipment
            available_sensors = self._get_sensors_for_equipment(equipment_ids)

            # Update sensor filter if current selection is not available
            if self.active_filters.sensor_ids:
                valid_sensors = [s for s in self.active_filters.sensor_ids if s in available_sensors]
                if not valid_sensors:
                    self.active_filters.sensor_ids = None

            logger.info(f"Equipment filter set: {equipment_ids} by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error setting equipment filter: {e}")
            return False

    def set_sensor_filter(self, sensor_ids: List[str],
                         component_id: str = "unknown") -> bool:
        """
        Set sensor filter and update related dropdowns

        Args:
            sensor_ids: List of sensor IDs to filter by
            component_id: Component making the change

        Returns:
            True if successful
        """
        try:
            self.active_filters.sensor_ids = sensor_ids

            # Update shared state
            if self.shared_state:
                self.shared_state.set_state('selections.sensor_id',
                                          sensor_ids[0] if sensor_ids else None,
                                          component_id)

            # Get available metrics for selected sensors
            available_metrics = self._get_metrics_for_sensors(sensor_ids)

            # Update metric filter if current selection is not available
            if self.active_filters.metric_ids:
                valid_metrics = [m for m in self.active_filters.metric_ids if m in available_metrics]
                if not valid_metrics:
                    self.active_filters.metric_ids = None

            logger.info(f"Sensor filter set: {sensor_ids} by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error setting sensor filter: {e}")
            return False

    def set_metric_filter(self, metric_ids: List[str],
                         component_id: str = "unknown") -> bool:
        """
        Set metric filter

        Args:
            metric_ids: List of metric IDs to filter by
            component_id: Component making the change

        Returns:
            True if successful
        """
        try:
            self.active_filters.metric_ids = metric_ids

            # Update shared state
            if self.shared_state:
                self.shared_state.set_state('selections.metric_id',
                                          metric_ids[0] if metric_ids else None,
                                          component_id)

            logger.info(f"Metric filter set: {metric_ids} by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error setting metric filter: {e}")
            return False

    def apply_preset_filter(self, preset_name: str,
                           component_id: str = "unknown") -> bool:
        """
        Apply predefined filter preset

        Args:
            preset_name: Name of preset to apply
            component_id: Component applying preset

        Returns:
            True if successful
        """
        try:
            if preset_name not in self.filter_presets:
                logger.error(f"Unknown filter preset: {preset_name}")
                return False

            preset_filters = self.filter_presets[preset_name]
            self.active_filters = preset_filters

            # Update shared state
            if self.shared_state:
                updates = {}
                if preset_filters.equipment_ids:
                    updates['selections.equipment_id'] = preset_filters.equipment_ids[0]
                if preset_filters.sensor_ids:
                    updates['selections.sensor_id'] = preset_filters.sensor_ids[0]
                if preset_filters.metric_ids:
                    updates['selections.metric_id'] = preset_filters.metric_ids[0]

                if updates:
                    self.shared_state.update_multiple(updates, component_id)

            logger.info(f"Applied filter preset '{preset_name}' by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error applying preset filter: {e}")
            return False

    def get_available_options(self, current_filters: Optional[FilterCriteria] = None) -> Dict[str, List]:
        """
        Get available filter options based on current filters

        Args:
            current_filters: Current filter state (uses active if None)

        Returns:
            Dictionary of available options
        """
        try:
            if current_filters is None:
                current_filters = self.active_filters

            options = {
                'equipment': [],
                'sensors': [],
                'metrics': [],
                'subsystems': [],
                'spacecraft': ['smap', 'msl'],
                'criticality': ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            }

            # Get equipment options from dropdown manager
            if self.dropdown_manager:
                equipment_options = self.dropdown_manager.get_equipment_options(include_all=False)
                options['equipment'] = [opt.value for opt in equipment_options if not opt.disabled]

                # Get sensors for current equipment selection
                if current_filters.equipment_ids:
                    for equipment_id in current_filters.equipment_ids:
                        sensor_options = self.dropdown_manager.get_sensor_options_for_equipment(
                            equipment_id, include_all=False
                        )
                        options['sensors'].extend([opt.value for opt in sensor_options if not opt.disabled])

                # Get metrics for current sensor selection
                if current_filters.equipment_ids and current_filters.sensor_ids:
                    for equipment_id in current_filters.equipment_ids:
                        for sensor_id in current_filters.sensor_ids:
                            metric_options = self.dropdown_manager.get_metric_options_for_sensor(
                                equipment_id, sensor_id, include_calculated=True
                            )
                            options['metrics'].extend([opt.value for opt in metric_options if not opt.disabled])

            # Remove duplicates
            for key in options:
                if isinstance(options[key], list):
                    options[key] = list(set(options[key]))

            return options

        except Exception as e:
            logger.error(f"Error getting available options: {e}")
            return {}

    def get_filter_presets(self) -> Dict[str, str]:
        """Get available filter presets with descriptions"""
        return {
            'critical_equipment': '🚨 Critical Equipment Only',
            'smap_only': '🛰️ SMAP Satellite Only',
            'msl_only': '🚗 MSL Mars Rover Only',
            'power_systems': '⚡ Power Systems',
            'communication_systems': '📡 Communication Systems',
            'anomalies_only': '⚠️ Anomalies Only',
            'recent_alerts': '🔔 Recent Alerts Only'
        }

    def clear_filters(self, component_id: str = "unknown") -> bool:
        """
        Clear all active filters

        Args:
            component_id: Component clearing filters

        Returns:
            True if successful
        """
        try:
            self.active_filters = FilterCriteria()

            # Update shared state
            if self.shared_state:
                updates = {
                    'selections.equipment_id': None,
                    'selections.sensor_id': None,
                    'selections.metric_id': None,
                    'selections.subsystem': None
                }
                self.shared_state.update_multiple(updates, component_id)

            logger.info(f"All filters cleared by {component_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing filters: {e}")
            return False

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of active filters"""
        active_count = 0
        summary = {}

        if self.active_filters.equipment_ids:
            active_count += 1
            summary['equipment'] = len(self.active_filters.equipment_ids)

        if self.active_filters.sensor_ids:
            active_count += 1
            summary['sensors'] = len(self.active_filters.sensor_ids)

        if self.active_filters.metric_ids:
            active_count += 1
            summary['metrics'] = len(self.active_filters.metric_ids)

        if self.active_filters.subsystems:
            active_count += 1
            summary['subsystems'] = len(self.active_filters.subsystems)

        if self.active_filters.spacecraft:
            active_count += 1
            summary['spacecraft'] = len(self.active_filters.spacecraft)

        summary['total_active_filters'] = active_count
        summary['has_value_filters'] = (
            self.active_filters.min_value is not None or
            self.active_filters.max_value is not None or
            self.active_filters.anomaly_threshold is not None
        )

        return summary

    def _get_critical_equipment_ids(self) -> List[str]:
        """Get list of critical equipment IDs"""
        critical_ids = []

        # Add SMAP critical systems
        for subsystem in self.nasa_categories['critical_systems']['smap']:
            # Generate equipment IDs for subsystem (simplified)
            critical_ids.append(f"SMAP-{subsystem[:3]}-001")

        # Add MSL critical systems
        for subsystem in self.nasa_categories['critical_systems']['msl']:
            critical_ids.append(f"MSL-{subsystem[:3]}-001")

        return critical_ids

    def _get_sensors_for_equipment(self, equipment_ids: List[str]) -> Set[str]:
        """Get available sensors for equipment list"""
        sensors = set()

        if self.dropdown_manager:
            for equipment_id in equipment_ids:
                sensor_options = self.dropdown_manager.get_sensor_options_for_equipment(
                    equipment_id, include_all=False
                )
                sensors.update([opt.value for opt in sensor_options if not opt.disabled])

        return sensors

    def _get_metrics_for_sensors(self, sensor_ids: List[str]) -> Set[str]:
        """Get available metrics for sensor list"""
        metrics = set()

        if self.dropdown_manager and self.active_filters.equipment_ids:
            for equipment_id in self.active_filters.equipment_ids:
                for sensor_id in sensor_ids:
                    metric_options = self.dropdown_manager.get_metric_options_for_sensor(
                        equipment_id, sensor_id, include_calculated=True
                    )
                    metrics.update([opt.value for opt in metric_options if not opt.disabled])

        return metrics

    def _find_value_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the main value column in DataFrame"""
        value_columns = ['value', 'sensor_value', 'measurement', 'reading']

        for col in value_columns:
            if col in data.columns:
                return col

        # Look for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]

        return None

    def _get_applied_filter_summary(self, filters: FilterCriteria) -> Dict[str, Any]:
        """Get summary of applied filters"""
        summary = {}

        if filters.equipment_ids:
            summary['equipment'] = f"{len(filters.equipment_ids)} equipment"
        if filters.sensor_ids:
            summary['sensors'] = f"{len(filters.sensor_ids)} sensors"
        if filters.metric_ids:
            summary['metrics'] = f"{len(filters.metric_ids)} metrics"
        if filters.subsystems:
            summary['subsystems'] = filters.subsystems
        if filters.spacecraft:
            summary['spacecraft'] = filters.spacecraft
        if filters.criticality_levels:
            summary['criticality'] = filters.criticality_levels

        return summary

    def _update_filter_state(self, result: FilterResult):
        """Update shared state with filter results"""
        try:
            if self.shared_state:
                filter_state = {
                    'filters.active_count': len(result.filter_summary.get('filters_applied', {})),
                    'filters.filtered_count': result.filtered_count,
                    'filters.total_count': result.original_count,
                    'filters.reduction_ratio': result.filter_summary.get('reduction_ratio', 0)
                }

                self.shared_state.update_multiple(filter_state, 'filter_manager')

        except Exception as e:
            logger.error(f"Error updating filter state: {e}")

    def _on_selection_change(self, key_path: str, old_value: Any, new_value: Any):
        """Handle selection changes from shared state"""
        try:
            if 'equipment_id' in key_path and new_value != old_value:
                if new_value:
                    self.set_equipment_filter([new_value], 'shared_state')
                else:
                    self.active_filters.equipment_ids = None

            elif 'sensor_id' in key_path and new_value != old_value:
                if new_value:
                    self.set_sensor_filter([new_value], 'shared_state')
                else:
                    self.active_filters.sensor_ids = None

            elif 'metric_id' in key_path and new_value != old_value:
                if new_value:
                    self.set_metric_filter([new_value], 'shared_state')
                else:
                    self.active_filters.metric_ids = None

        except Exception as e:
            logger.error(f"Error handling selection change: {e}")


# Global instance for use across the dashboard
filter_manager = FilterManager()