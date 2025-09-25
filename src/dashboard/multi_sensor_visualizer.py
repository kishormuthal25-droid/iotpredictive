"""
Multi-Sensor Visualizer for Advanced Real-Time Plotting
Provides sophisticated visualization capabilities for multiple sensors with anomaly highlighting
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from src.dashboard.unified_data_orchestrator import unified_data_orchestrator

logger = logging.getLogger(__name__)


class MultiSensorVisualizer:
    """
    Advanced visualizer for multiple sensor time series with anomaly detection
    Supports various chart types and interactive features
    """

    def __init__(self):
        """Initialize multi-sensor visualizer"""
        self.orchestrator = unified_data_orchestrator

        # Color palettes for different subsystems
        self.subsystem_colors = {
            'POWER': ['#FF6B6B', '#FF8E8E', '#FFB1B1'],
            'COMMUNICATION': ['#4ECDC4', '#6FD5CE', '#90DDD8'],
            'MOBILITY': ['#45B7D1', '#67C4DB', '#89D1E5'],
            'ATTITUDE': ['#96CEB4', '#A8D5C2', '#BADCD0'],
            'THERMAL': ['#FFEAA7', '#FFEF9F', '#FFF4B7'],
            'PAYLOAD': ['#DDA0DD', '#E4B3E4', '#EBC6EB'],
            'ENVIRONMENTAL': ['#98D8C8', '#A8DDD0', '#B8E2D8'],
            'SCIENTIFIC': ['#F7DC6F', '#F9E489', '#FBECA3'],
            'NAVIGATION': ['#BB8FCE', '#C7A2D6', '#D3B5DE']
        }

        # Chart style settings
        self.chart_style = {
            'template': 'plotly_white',
            'height': 600,
            'margin': dict(l=50, r=50, t=60, b=50),
            'font': dict(size=12),
            'showlegend': True,
            'hovermode': 'x unified'
        }

        logger.info("Multi-sensor visualizer initialized")

    def create_multi_sensor_time_series(self, sensor_ids: List[str],
                                       time_window: str = "5min",
                                       chart_type: str = "unified",
                                       show_anomalies: bool = True,
                                       normalize_values: bool = False) -> go.Figure:
        """
        Create multi-sensor time series visualization

        Args:
            sensor_ids: List of sensor IDs to plot
            time_window: Time window for data
            chart_type: Chart type ('unified', 'subplots', 'heatmap')
            show_anomalies: Whether to highlight anomalies
            normalize_values: Whether to normalize sensor values

        Returns:
            Plotly figure with multi-sensor visualization
        """
        try:
            if not sensor_ids:
                return self._create_empty_chart("No sensors selected")

            # Get sensor stream data
            sensor_data = self.orchestrator.get_sensor_stream_data(
                sensor_ids=sensor_ids,
                time_window=time_window
            )

            if not sensor_data:
                return self._create_empty_chart("No sensor data available")

            if chart_type == "unified":
                return self._create_unified_time_series(
                    sensor_data, show_anomalies, normalize_values
                )
            elif chart_type == "subplots":
                return self._create_subplot_time_series(
                    sensor_data, show_anomalies, normalize_values
                )
            elif chart_type == "heatmap":
                return self._create_sensor_heatmap(sensor_data)
            else:
                return self._create_unified_time_series(
                    sensor_data, show_anomalies, normalize_values
                )

        except Exception as e:
            logger.error(f"Error creating multi-sensor visualization: {e}")
            return self._create_error_chart(str(e))

    def _create_unified_time_series(self, sensor_data: Dict[str, Any],
                                   show_anomalies: bool = True,
                                   normalize_values: bool = False) -> go.Figure:
        """Create unified time series chart with all sensors on same plot"""
        fig = go.Figure()

        # Track y-axis assignments
        y_axis_map = {}
        next_y_axis = 1

        # Group sensors by unit for y-axis assignment
        unit_groups = defaultdict(list)
        for sensor_id, data in sensor_data.items():
            unit = data['metadata'].get('unit', 'value')  # Default to 'value' if unit not specified
            unit_groups[unit].append((sensor_id, data))

        # Create traces for each sensor
        color_idx = 0
        total_colors = sum(len(colors) for colors in self.subsystem_colors.values())

        for unit, sensors_with_data in unit_groups.items():
            # Assign y-axis for this unit group
            if len(unit_groups) > 1:
                if unit not in y_axis_map:
                    if next_y_axis == 1:
                        y_axis_map[unit] = 'y'
                    else:
                        y_axis_map[unit] = f'y{next_y_axis}'
                    next_y_axis += 1

            for sensor_id, data in sensors_with_data:
                metadata = data['metadata']
                sensor_name = data['sensor_name']
                subsystem = data['subsystem']
                timestamps = data['data']['timestamps']
                values = data['data']['values']
                anomaly_flags = data['data'].get('anomaly_flags', [])

                if not timestamps or not values:
                    continue

                # Convert timestamps to datetime objects
                timestamps = [pd.to_datetime(ts) for ts in timestamps]

                # Normalize values if requested
                if normalize_values:
                    values = self._normalize_values(values, metadata)
                    unit_suffix = " (normalized)"
                else:
                    unit_suffix = f" ({metadata.get('unit', 'value')})"

                # Get color for this sensor
                subsystem_color_palette = self.subsystem_colors.get(subsystem, ['#888888'])
                color = subsystem_color_palette[color_idx % len(subsystem_color_palette)]
                color_idx += 1

                # Create main trace
                y_axis = y_axis_map.get(metadata.get('unit', 'value'), 'y') if len(unit_groups) > 1 else 'y'

                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=f"{sensor_name}",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    yaxis=y_axis,
                    legendgroup=subsystem,
                    legendgrouptitle_text=subsystem,
                    hovertemplate=f'<b>{sensor_name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  f'Value: %{{y:.3f}} {metadata.get("unit", "value")}<br>' +
                                  f'Equipment: {data["equipment_id"]}<br>' +
                                  '<extra></extra>'
                ))

                # Add anomaly markers if requested
                if show_anomalies and anomaly_flags:
                    anomaly_timestamps = [timestamps[i] for i, flag in enumerate(anomaly_flags) if flag and i < len(timestamps)]
                    anomaly_values = [values[i] for i, flag in enumerate(anomaly_flags) if flag and i < len(values)]

                    if anomaly_timestamps:
                        fig.add_trace(go.Scatter(
                            x=anomaly_timestamps,
                            y=anomaly_values,
                            mode='markers',
                            name=f"{sensor_name} Anomalies",
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='x',
                                line=dict(color='white', width=1)
                            ),
                            yaxis=y_axis,
                            legendgroup=subsystem,
                            showlegend=False,
                            hovertemplate=f'<b>ANOMALY</b><br>' +
                                          f'{sensor_name}<br>' +
                                          'Time: %{x}<br>' +
                                          f'Value: %{{y:.3f}} {metadata.get("unit", "value")}<br>' +
                                          '<extra></extra>'
                        ))

        # Configure layout
        layout_config = self.chart_style.copy()
        layout_config.update({
            'title': f'Multi-Sensor Time Series ({len(sensor_data)} sensors)',
            'xaxis_title': 'Time',
            'yaxis_title': 'Sensor Values',
            'legend': dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                groupclick="toggleitem"
            )
        })

        # Configure multiple y-axes if needed
        if len(unit_groups) > 1:
            layout_config['yaxis_title'] = list(unit_groups.keys())[0]

            # Add secondary y-axes
            for i, unit in enumerate(list(unit_groups.keys())[1:], 2):
                axis_name = f'yaxis{i}'
                layout_config[axis_name] = dict(
                    title=unit,
                    overlaying='y',
                    side='right',
                    position=1 - (i-2) * 0.05  # Offset each additional axis
                )

        fig.update_layout(**layout_config)

        return fig

    def _create_subplot_time_series(self, sensor_data: Dict[str, Any],
                                   show_anomalies: bool = True,
                                   normalize_values: bool = False) -> go.Figure:
        """Create subplot time series with each sensor in its own subplot"""
        num_sensors = len(sensor_data)
        if num_sensors == 0:
            return self._create_empty_chart("No sensor data")

        # Calculate subplot layout
        if num_sensors <= 2:
            rows, cols = num_sensors, 1
        elif num_sensors <= 4:
            rows, cols = 2, 2
        elif num_sensors <= 6:
            rows, cols = 3, 2
        elif num_sensors <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3  # Max 12 sensors

        # Create subplot titles
        subplot_titles = []
        for sensor_id, data in list(sensor_data.items())[:rows*cols]:
            subplot_titles.append(f"{data['sensor_name']} ({data['metadata'].get('unit', 'value')})")

        # Create subplots
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            vertical_spacing=0.08
        )

        # Add traces to subplots
        for i, (sensor_id, data) in enumerate(sensor_data.items()):
            if i >= rows * cols:
                break

            row = (i // cols) + 1
            col = (i % cols) + 1

            metadata = data['metadata']
            sensor_name = data['sensor_name']
            subsystem = data['subsystem']
            timestamps = data['data']['timestamps']
            values = data['data']['values']
            anomaly_flags = data['data'].get('anomaly_flags', [])

            if not timestamps or not values:
                continue

            # Convert timestamps
            timestamps = [pd.to_datetime(ts) for ts in timestamps]

            # Normalize if requested
            if normalize_values:
                values = self._normalize_values(values, metadata)

            # Get color
            subsystem_color_palette = self.subsystem_colors.get(subsystem, ['#888888'])
            color = subsystem_color_palette[0]

            # Main trace
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines+markers',
                    name=sensor_name,
                    line=dict(color=color, width=2),
                    marker=dict(size=3),
                    showlegend=False,
                    hovertemplate=f'<b>{sensor_name}</b><br>' +
                                  'Time: %{x}<br>' +
                                  f'Value: %{{y:.3f}} {metadata.get("unit", "value")}<br>' +
                                  '<extra></extra>'
                ),
                row=row, col=col
            )

            # Anomaly markers
            if show_anomalies and anomaly_flags:
                anomaly_timestamps = [timestamps[i] for i, flag in enumerate(anomaly_flags) if flag and i < len(timestamps)]
                anomaly_values = [values[i] for i, flag in enumerate(anomaly_flags) if flag and i < len(values)]

                if anomaly_timestamps:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_timestamps,
                            y=anomaly_values,
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='x'),
                            showlegend=False,
                            hovertemplate='<b>ANOMALY</b><br>' +
                                          'Time: %{x}<br>' +
                                          f'Value: %{{y:.3f}}<br>' +
                                          '<extra></extra>'
                        ),
                        row=row, col=col
                    )

        # Update layout
        fig.update_layout(
            height=max(400, rows * 200),
            title=f'Multi-Sensor Subplots ({num_sensors} sensors)',
            template=self.chart_style['template'],
            margin=self.chart_style['margin']
        )

        # Update x-axis titles for bottom row
        for col in range(1, cols + 1):
            fig.update_xaxes(title_text="Time", row=rows, col=col)

        return fig

    def _create_sensor_heatmap(self, sensor_data: Dict[str, Any]) -> go.Figure:
        """Create sensor value heatmap"""
        if not sensor_data:
            return self._create_empty_chart("No sensor data for heatmap")

        # Prepare data for heatmap
        sensor_names = []
        timestamps = None
        value_matrix = []

        for sensor_id, data in sensor_data.items():
            sensor_names.append(data['sensor_name'])
            current_timestamps = data['data']['timestamps']
            values = data['data']['values']

            if timestamps is None:
                timestamps = current_timestamps

            # Ensure all sensors have same timestamp length
            if len(current_timestamps) != len(timestamps):
                # Interpolate or pad to match
                if len(values) < len(timestamps):
                    values.extend([values[-1]] * (len(timestamps) - len(values)))
                elif len(values) > len(timestamps):
                    values = values[:len(timestamps)]

            value_matrix.append(values)

        if not value_matrix or not timestamps:
            return self._create_empty_chart("Insufficient data for heatmap")

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=value_matrix,
            x=[pd.to_datetime(ts).strftime('%H:%M:%S') for ts in timestamps],
            y=sensor_names,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                          'Time: %{x}<br>' +
                          'Value: %{z:.3f}<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            title='Sensor Value Heatmap',
            xaxis_title='Time',
            yaxis_title='Sensors',
            **self.chart_style
        )

        return fig

    def _normalize_values(self, values: List[float], metadata: Dict[str, Any]) -> List[float]:
        """Normalize sensor values to 0-1 range"""
        try:
            min_val = metadata.get('min_value', min(values))
            max_val = metadata.get('max_value', max(values))

            if max_val == min_val:
                return [0.5] * len(values)  # All values are the same

            return [(v - min_val) / (max_val - min_val) for v in values]

        except Exception as e:
            logger.error(f"Error normalizing values: {e}")
            return values

    def create_sensor_correlation_matrix(self, sensor_ids: List[str],
                                        time_window: str = "5min") -> go.Figure:
        """Create correlation matrix between sensors"""
        try:
            # Get sensor data
            sensor_data = self.orchestrator.get_sensor_stream_data(
                sensor_ids=sensor_ids,
                time_window=time_window
            )

            if not sensor_data or len(sensor_data) < 2:
                return self._create_empty_chart("Need at least 2 sensors for correlation")

            # Prepare data matrix
            df_data = {}
            for sensor_id, data in sensor_data.items():
                values = data['data']['values']
                sensor_name = data['sensor_name']
                if values:
                    df_data[sensor_name] = values

            if not df_data:
                return self._create_empty_chart("No valid sensor data")

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(df_data)
            correlation_matrix = df.corr()

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>' +
                              'Correlation: %{z:.3f}<br>' +
                              '<extra></extra>'
            ))

            fig.update_layout(
                title='Sensor Correlation Matrix',
                xaxis_title='Sensors',
                yaxis_title='Sensors',
                **self.chart_style
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return self._create_error_chart(str(e))

    def create_anomaly_summary_chart(self, sensor_ids: List[str],
                                    time_window: str = "5min") -> go.Figure:
        """Create anomaly summary visualization"""
        try:
            # Get sensor data
            sensor_data = self.orchestrator.get_sensor_stream_data(
                sensor_ids=sensor_ids,
                time_window=time_window
            )

            if not sensor_data:
                return self._create_empty_chart("No sensor data for anomaly summary")

            # Calculate anomaly statistics
            sensor_names = []
            anomaly_counts = []
            total_points = []
            anomaly_rates = []

            for sensor_id, data in sensor_data.items():
                sensor_names.append(data['sensor_name'])
                anomaly_flags = data['data'].get('anomaly_flags', [])

                anomaly_count = sum(anomaly_flags)
                total_count = len(anomaly_flags)

                anomaly_counts.append(anomaly_count)
                total_points.append(total_count)
                anomaly_rates.append((anomaly_count / total_count * 100) if total_count > 0 else 0)

            # Create bar chart
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=sensor_names,
                y=anomaly_rates,
                text=[f'{count}/{total}' for count, total in zip(anomaly_counts, total_points)],
                textposition='auto',
                marker_color=['red' if rate > 10 else 'orange' if rate > 5 else 'green' for rate in anomaly_rates],
                hovertemplate='<b>%{x}</b><br>' +
                              'Anomaly Rate: %{y:.1f}%<br>' +
                              'Count: %{text}<br>' +
                              '<extra></extra>'
            ))

            fig.update_layout(
                title='Sensor Anomaly Summary',
                xaxis_title='Sensors',
                yaxis_title='Anomaly Rate (%)',
                **self.chart_style
            )

            return fig

        except Exception as e:
            logger.error(f"Error creating anomaly summary: {e}")
            return self._create_error_chart(str(e))

    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title="Multi-Sensor Visualization",
            template=self.chart_style['template'],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig

    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart with error message"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message[:100]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title="Multi-Sensor Visualization - Error",
            template=self.chart_style['template'],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig


# Global instance for dashboard integration
multi_sensor_visualizer = MultiSensorVisualizer()