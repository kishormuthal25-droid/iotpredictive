"""
Chart Manager
Unified chart management for consistent chart type switching across the dashboard
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types"""
    LINE = "line"
    CANDLESTICK = "candlestick"
    AREA = "area"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"


@dataclass
class ChartConfig:
    """Configuration for chart display"""
    chart_type: ChartType
    title: str
    height: int = 400
    show_legend: bool = True
    show_grid: bool = True
    template: str = "plotly_white"

    # Chart-specific options
    show_anomalies: bool = True
    show_confidence_intervals: bool = False
    show_thresholds: bool = True
    normalize_values: bool = False

    # Color scheme
    color_scheme: str = "viridis"
    anomaly_color: str = "red"
    threshold_color: str = "orange"


@dataclass
class ChartData:
    """Standardized data structure for charts"""
    timestamps: List[datetime]
    values: List[float]
    labels: List[str]
    equipment_id: str
    sensor_id: str
    metric_id: str

    # Optional data
    anomaly_scores: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    confidence_lower: Optional[List[float]] = None
    thresholds: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ChartManager:
    """
    Unified chart manager for consistent chart type switching and data visualization
    Handles line, candlestick, area, heatmap charts with NASA IoT data
    """

    def __init__(self):
        """Initialize chart manager"""
        self.supported_types = list(ChartType)
        self.default_config = ChartConfig(
            chart_type=ChartType.LINE,
            title="Sensor Data",
            height=400
        )

        # NASA-specific color schemes
        self.nasa_colors = {
            'smap': '#1f77b4',  # Blue for SMAP
            'msl': '#ff7f0e',   # Orange for MSL
            'critical': '#d62728',  # Red for critical equipment
            'high': '#ff7f0e',      # Orange for high priority
            'medium': '#2ca02c',    # Green for medium priority
            'low': '#17becf'        # Cyan for low priority
        }

        logger.info("ChartManager initialized with support for NASA IoT data visualization")

    def create_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """
        Create chart based on type and configuration

        Args:
            data: Chart data in standardized format
            config: Chart configuration

        Returns:
            Plotly figure object
        """
        try:
            if config.chart_type == ChartType.LINE:
                return self._create_line_chart(data, config)
            elif config.chart_type == ChartType.CANDLESTICK:
                return self._create_candlestick_chart(data, config)
            elif config.chart_type == ChartType.AREA:
                return self._create_area_chart(data, config)
            elif config.chart_type == ChartType.HEATMAP:
                return self._create_heatmap_chart(data, config)
            elif config.chart_type == ChartType.SCATTER:
                return self._create_scatter_chart(data, config)
            elif config.chart_type == ChartType.BAR:
                return self._create_bar_chart(data, config)
            elif config.chart_type == ChartType.HISTOGRAM:
                return self._create_histogram_chart(data, config)
            else:
                logger.warning(f"Unsupported chart type: {config.chart_type}")
                return self._create_line_chart(data, config)  # Fallback to line chart

        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return self._create_error_chart(str(e))

    def convert_chart_type(self, figure: go.Figure, new_type: ChartType,
                          data: ChartData, config: ChartConfig) -> go.Figure:
        """
        Convert existing chart to new type while preserving data

        Args:
            figure: Existing plotly figure
            new_type: Target chart type
            data: Original chart data
            config: Chart configuration

        Returns:
            New figure with converted chart type
        """
        try:
            # Update config with new type
            new_config = config
            new_config.chart_type = new_type

            # Create new chart with updated configuration
            return self.create_chart(data, new_config)

        except Exception as e:
            logger.error(f"Error converting chart type: {e}")
            return figure  # Return original figure on error

    def add_anomaly_overlays(self, figure: go.Figure, data: ChartData,
                           config: ChartConfig) -> go.Figure:
        """
        Add anomaly overlays to existing chart

        Args:
            figure: Plotly figure to modify
            data: Chart data with anomaly information
            config: Chart configuration

        Returns:
            Modified figure with anomaly overlays
        """
        try:
            if not config.show_anomalies or not data.anomaly_scores:
                return figure

            # Find anomaly points (scores above threshold)
            anomaly_threshold = data.thresholds.get('anomaly', 0.7) if data.thresholds else 0.7
            anomaly_indices = [i for i, score in enumerate(data.anomaly_scores)
                             if score > anomaly_threshold]

            if anomaly_indices:
                anomaly_times = [data.timestamps[i] for i in anomaly_indices]
                anomaly_values = [data.values[i] for i in anomaly_indices]

                # Add anomaly scatter trace
                figure.add_trace(go.Scatter(
                    x=anomaly_times,
                    y=anomaly_values,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color=config.anomaly_color,
                        size=10,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='<b>Anomaly Detected</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y}<br>' +
                                '<extra></extra>'
                ))

            return figure

        except Exception as e:
            logger.error(f"Error adding anomaly overlays: {e}")
            return figure

    def add_threshold_lines(self, figure: go.Figure, data: ChartData,
                          config: ChartConfig) -> go.Figure:
        """
        Add threshold lines to chart

        Args:
            figure: Plotly figure to modify
            data: Chart data with threshold information
            config: Chart configuration

        Returns:
            Modified figure with threshold lines
        """
        try:
            if not config.show_thresholds or not data.thresholds:
                return figure

            for threshold_name, threshold_value in data.thresholds.items():
                if threshold_name != 'anomaly':  # Skip anomaly threshold (handled separately)
                    figure.add_hline(
                        y=threshold_value,
                        line_dash="dash",
                        line_color=config.threshold_color,
                        annotation_text=f"{threshold_name.title()} Threshold",
                        annotation_position="bottom right"
                    )

            return figure

        except Exception as e:
            logger.error(f"Error adding threshold lines: {e}")
            return figure

    def _create_line_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create line chart"""
        figure = go.Figure()

        # Main data trace
        figure.add_trace(go.Scatter(
            x=data.timestamps,
            y=data.values,
            mode='lines',
            name=f"{data.sensor_id} - {data.metric_id}",
            line=dict(
                color=self._get_equipment_color(data.equipment_id),
                width=2
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y}<br>' +
                        '<extra></extra>'
        ))

        # Add confidence intervals if available
        if config.show_confidence_intervals and data.confidence_upper and data.confidence_lower:
            figure.add_trace(go.Scatter(
                x=data.timestamps + data.timestamps[::-1],
                y=data.confidence_upper + data.confidence_lower[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                hoverinfo="skip"
            ))

        return self._apply_chart_styling(figure, data, config)

    def _create_candlestick_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create candlestick chart (useful for time-series data with OHLC values)"""
        figure = go.Figure()

        # For single value time series, create synthetic OHLC data
        if len(data.values) > 0:
            # Create moving averages to simulate OHLC
            window = min(5, len(data.values))
            rolling_data = pd.Series(data.values).rolling(window=window, center=True)

            opens = rolling_data.min().fillna(data.values)
            highs = rolling_data.max().fillna(data.values)
            lows = rolling_data.min().fillna(data.values)
            closes = data.values

            figure.add_trace(go.Candlestick(
                x=data.timestamps,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name=f"{data.sensor_id} - {data.metric_id}",
                increasing_line_color=self.nasa_colors['medium'],
                decreasing_line_color=self.nasa_colors['critical']
            ))

        return self._apply_chart_styling(figure, data, config)

    def _create_area_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create area chart"""
        figure = go.Figure()

        figure.add_trace(go.Scatter(
            x=data.timestamps,
            y=data.values,
            mode='lines',
            name=f"{data.sensor_id} - {data.metric_id}",
            fill='tonexty' if len(figure.data) > 0 else 'tozeroy',
            fillcolor=f"rgba{(*self._hex_to_rgb(self._get_equipment_color(data.equipment_id)), 0.3)}",
            line=dict(
                color=self._get_equipment_color(data.equipment_id),
                width=2
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y}<br>' +
                        '<extra></extra>'
        ))

        return self._apply_chart_styling(figure, data, config)

    def _create_heatmap_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create heatmap chart (useful for correlation analysis)"""
        figure = go.Figure()

        # For single sensor data, create time-based heatmap
        if len(data.values) > 0:
            # Reshape data for heatmap (time periods vs values)
            time_buckets = 24  # 24 time buckets
            bucket_size = max(1, len(data.values) // time_buckets)

            heatmap_data = []
            for i in range(0, len(data.values), bucket_size):
                bucket_values = data.values[i:i + bucket_size]
                if bucket_values:
                    heatmap_data.append([np.mean(bucket_values)])

            if heatmap_data:
                figure.add_trace(go.Heatmap(
                    z=heatmap_data,
                    x=[f"{data.sensor_id}"],
                    y=[f"Period {i+1}" for i in range(len(heatmap_data))],
                    colorscale=config.color_scheme,
                    name=f"{data.sensor_id} - {data.metric_id}",
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Period: %{y}<br>' +
                                'Sensor: %{x}<br>' +
                                'Avg Value: %{z}<br>' +
                                '<extra></extra>'
                ))

        return self._apply_chart_styling(figure, data, config)

    def _create_scatter_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create scatter chart"""
        figure = go.Figure()

        # Color points by anomaly score if available
        marker_colors = data.anomaly_scores if data.anomaly_scores else None

        figure.add_trace(go.Scatter(
            x=data.timestamps,
            y=data.values,
            mode='markers',
            name=f"{data.sensor_id} - {data.metric_id}",
            marker=dict(
                color=marker_colors or self._get_equipment_color(data.equipment_id),
                size=8,
                colorscale='Viridis' if marker_colors else None,
                showscale=bool(marker_colors),
                colorbar=dict(title="Anomaly Score") if marker_colors else None
            ),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y}<br>' +
                        '<extra></extra>'
        ))

        return self._apply_chart_styling(figure, data, config)

    def _create_bar_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create bar chart"""
        figure = go.Figure()

        figure.add_trace(go.Bar(
            x=data.timestamps,
            y=data.values,
            name=f"{data.sensor_id} - {data.metric_id}",
            marker_color=self._get_equipment_color(data.equipment_id),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y}<br>' +
                        '<extra></extra>'
        ))

        return self._apply_chart_styling(figure, data, config)

    def _create_histogram_chart(self, data: ChartData, config: ChartConfig) -> go.Figure:
        """Create histogram chart"""
        figure = go.Figure()

        figure.add_trace(go.Histogram(
            x=data.values,
            name=f"{data.sensor_id} - {data.metric_id}",
            marker_color=self._get_equipment_color(data.equipment_id),
            opacity=0.7,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                        'Value Range: %{x}<br>' +
                        'Count: %{y}<br>' +
                        '<extra></extra>'
        ))

        # Update x-axis title for histogram
        figure.update_xaxes(title_text=f"{data.metric_id} Values")
        figure.update_yaxes(title_text="Frequency")

        return self._apply_chart_styling(figure, data, config)

    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when chart creation fails"""
        figure = go.Figure()
        figure.add_annotation(
            text=f"Chart Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="red")
        )
        figure.update_layout(
            title="Chart Error",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return figure

    def _apply_chart_styling(self, figure: go.Figure, data: ChartData,
                           config: ChartConfig) -> go.Figure:
        """Apply consistent styling to charts"""
        try:
            # Add anomaly overlays
            figure = self.add_anomaly_overlays(figure, data, config)

            # Add threshold lines
            figure = self.add_threshold_lines(figure, data, config)

            # Apply layout styling
            figure.update_layout(
                title=config.title,
                height=config.height,
                showlegend=config.show_legend,
                template=config.template,
                hovermode='x unified',
                margin=dict(l=50, r=50, t=50, b=50)
            )

            # Grid configuration
            figure.update_xaxes(
                showgrid=config.show_grid,
                title_text="Time"
            )
            figure.update_yaxes(
                showgrid=config.show_grid,
                title_text=data.metric_id
            )

            return figure

        except Exception as e:
            logger.error(f"Error applying chart styling: {e}")
            return figure

    def _get_equipment_color(self, equipment_id: str) -> str:
        """Get color based on equipment type"""
        if equipment_id.startswith('SMAP'):
            return self.nasa_colors['smap']
        elif equipment_id.startswith('MSL'):
            return self.nasa_colors['msl']
        else:
            return self.nasa_colors['medium']

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_chart_type_options(self) -> List[Dict[str, str]]:
        """Get available chart type options for dropdown"""
        return [
            {"label": "📈 Line Chart", "value": ChartType.LINE.value},
            {"label": "🕯️ Candlestick", "value": ChartType.CANDLESTICK.value},
            {"label": "📊 Area Chart", "value": ChartType.AREA.value},
            {"label": "🔥 Heatmap", "value": ChartType.HEATMAP.value},
            {"label": "⚫ Scatter Plot", "value": ChartType.SCATTER.value},
            {"label": "📊 Bar Chart", "value": ChartType.BAR.value},
            {"label": "📊 Histogram", "value": ChartType.HISTOGRAM.value}
        ]

    def validate_chart_config(self, config: ChartConfig) -> Dict[str, bool]:
        """Validate chart configuration"""
        validation = {
            'chart_type_valid': config.chart_type in self.supported_types,
            'height_valid': 200 <= config.height <= 1000,
            'title_valid': bool(config.title and config.title.strip()),
            'template_valid': config.template in ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white']
        }

        validation['config_valid'] = all(validation.values())
        return validation


# Global instance for use across the dashboard
chart_manager = ChartManager()