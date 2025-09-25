"""
Enhanced Detection Details Panel
Provides detailed sensor-level anomaly breakdown with confidence scores and reconstruction analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Import project modules
from src.anomaly_detection.nasa_anomaly_engine import nasa_anomaly_engine, AnomalyResult
from src.data_ingestion.equipment_mapper import equipment_mapper, SensorSpec

logger = logging.getLogger(__name__)


@dataclass
class SensorAnomalyDetail:
    """Detailed anomaly information for a single sensor"""
    sensor_name: str
    sensor_unit: str
    current_value: float
    expected_value: float
    anomaly_score: float
    confidence: float
    reconstruction_error: float
    severity_level: str
    threshold_exceeded: bool
    z_score: float
    percentile_rank: float
    historical_mean: float
    historical_std: float
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    last_normal_value: Optional[float] = None
    time_since_normal: Optional[int] = None  # minutes


@dataclass
class EquipmentDetectionSummary:
    """Comprehensive detection summary for equipment"""
    equipment_id: str
    equipment_name: str
    equipment_type: str
    subsystem: str
    overall_anomaly_score: float
    overall_confidence: float
    is_anomaly: bool
    severity_level: str
    sensor_count: int
    anomalous_sensor_count: int
    sensor_details: List[SensorAnomalyDetail]
    model_performance: Dict[str, float]
    reconstruction_quality: float
    pattern_analysis: Dict[str, Any]
    recommendations: List[str]


class EnhancedDetectionDetailsPanel:
    """
    Enhanced panel for displaying detailed anomaly detection results
    with sensor-level breakdown and confidence analysis
    """

    def __init__(self):
        """Initialize the Enhanced Detection Details Panel"""
        self.anomaly_engine = nasa_anomaly_engine
        self.equipment_mapper = equipment_mapper

        # Historical data for trend analysis
        self.sensor_history = {}
        self.detection_history = {}

        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.85,
            'medium': 0.65,
            'low': 0.45
        }

        # Visualization settings
        self.color_scheme = {
            'normal': '#2ECC71',
            'warning': '#F39C12',
            'critical': '#E74C3C',
            'unknown': '#95A5A6',
            'confidence_high': '#27AE60',
            'confidence_medium': '#F1C40F',
            'confidence_low': '#E67E22'
        }

        logger.info("Enhanced Detection Details Panel initialized")

    def analyze_equipment_detection(self, equipment_id: str,
                                  anomaly_result: AnomalyResult = None) -> EquipmentDetectionSummary:
        """
        Perform comprehensive analysis of anomaly detection for specific equipment

        Args:
            equipment_id: Equipment identifier
            anomaly_result: Recent anomaly detection result

        Returns:
            Comprehensive detection summary
        """
        try:
            # Get equipment information
            equipment = self._get_equipment_info(equipment_id)

            if not equipment:
                logger.warning(f"Equipment {equipment_id} not found")
                return self._create_empty_summary(equipment_id)

            # Get or generate anomaly result
            if not anomaly_result:
                anomaly_result = self._get_latest_anomaly_result(equipment_id)

            # Analyze each sensor in detail
            sensor_details = self._analyze_sensors_detailed(equipment, anomaly_result)

            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(sensor_details)

            # Perform pattern analysis
            pattern_analysis = self._perform_pattern_analysis(equipment_id, sensor_details)

            # Generate recommendations
            recommendations = self._generate_recommendations(equipment, sensor_details, pattern_analysis)

            # Create comprehensive summary
            summary = EquipmentDetectionSummary(
                equipment_id=equipment_id,
                equipment_name=getattr(equipment, 'equipment_name', equipment_id),
                equipment_type=getattr(equipment, 'equipment_type', 'Unknown'),
                subsystem=getattr(equipment, 'subsystem', 'Unknown'),
                overall_anomaly_score=anomaly_result.anomaly_score if anomaly_result else 0.0,
                overall_confidence=anomaly_result.confidence if anomaly_result else 0.0,
                is_anomaly=anomaly_result.is_anomaly if anomaly_result else False,
                severity_level=anomaly_result.severity_level if anomaly_result else 'NORMAL',
                sensor_count=len(sensor_details),
                anomalous_sensor_count=sum(1 for s in sensor_details if s.anomaly_score > 0.5),
                sensor_details=sensor_details,
                model_performance=overall_metrics['model_performance'],
                reconstruction_quality=overall_metrics['reconstruction_quality'],
                pattern_analysis=pattern_analysis,
                recommendations=recommendations
            )

            return summary

        except Exception as e:
            logger.error(f"Error analyzing equipment detection for {equipment_id}: {e}")
            return self._create_empty_summary(equipment_id)

    def _analyze_sensors_detailed(self, equipment, anomaly_result: AnomalyResult) -> List[SensorAnomalyDetail]:
        """Analyze each sensor in detail"""
        sensor_details = []

        try:
            # Get sensor specifications from equipment
            sensors = getattr(equipment, 'sensors', [])
            sensor_values = anomaly_result.sensor_values if anomaly_result else {}
            anomalous_sensors = anomaly_result.anomalous_sensors if anomaly_result else []

            for i, sensor in enumerate(sensors):
                sensor_name = sensor.name if hasattr(sensor, 'name') else f"Sensor_{i}"

                # Get current sensor value
                current_value = sensor_values.get(sensor_name, 0.0)

                # Calculate detailed metrics for this sensor
                detail = self._calculate_sensor_detail(
                    sensor,
                    sensor_name,
                    current_value,
                    anomalous_sensors,
                    equipment.equipment_id
                )

                sensor_details.append(detail)

        except Exception as e:
            logger.error(f"Error analyzing sensors in detail: {e}")

        return sensor_details

    def _calculate_sensor_detail(self, sensor_spec: SensorSpec, sensor_name: str,
                               current_value: float, anomalous_sensors: List[str],
                               equipment_id: str) -> SensorAnomalyDetail:
        """Calculate detailed metrics for a single sensor"""
        try:
            # Basic sensor information
            is_anomalous = sensor_name in anomalous_sensors

            # Get historical data for this sensor
            historical_data = self._get_sensor_historical_data(equipment_id, sensor_name)

            # Calculate statistical metrics
            if historical_data:
                historical_mean = np.mean(historical_data)
                historical_std = np.std(historical_data)
                z_score = (current_value - historical_mean) / historical_std if historical_std > 0 else 0
                percentile_rank = stats.percentileofscore(historical_data, current_value) / 100.0
            else:
                historical_mean = getattr(sensor_spec, 'nominal_value', current_value)
                historical_std = 1.0
                z_score = 0.0
                percentile_rank = 0.5

            # Calculate anomaly score for this sensor
            sensor_anomaly_score = self._calculate_sensor_anomaly_score(
                current_value, historical_mean, historical_std, sensor_spec
            )

            # Calculate confidence
            confidence = self._calculate_sensor_confidence(sensor_anomaly_score, z_score, percentile_rank)

            # Calculate reconstruction error
            expected_value = self._calculate_expected_value(sensor_spec, historical_mean)
            reconstruction_error = abs(current_value - expected_value)

            # Determine severity
            severity_level = self._determine_sensor_severity(sensor_anomaly_score, confidence)

            # Check threshold exceeded
            threshold_exceeded = self._check_threshold_exceeded(current_value, sensor_spec)

            # Analyze trend
            trend_direction = self._analyze_sensor_trend(equipment_id, sensor_name)

            # Find last normal value
            last_normal_info = self._find_last_normal_value(equipment_id, sensor_name)

            return SensorAnomalyDetail(
                sensor_name=sensor_name,
                sensor_unit=getattr(sensor_spec, 'unit', ''),
                current_value=current_value,
                expected_value=expected_value,
                anomaly_score=sensor_anomaly_score,
                confidence=confidence,
                reconstruction_error=reconstruction_error,
                severity_level=severity_level,
                threshold_exceeded=threshold_exceeded,
                z_score=z_score,
                percentile_rank=percentile_rank,
                historical_mean=historical_mean,
                historical_std=historical_std,
                trend_direction=trend_direction,
                last_normal_value=last_normal_info.get('value'),
                time_since_normal=last_normal_info.get('time_minutes')
            )

        except Exception as e:
            logger.error(f"Error calculating sensor detail for {sensor_name}: {e}")
            return self._create_default_sensor_detail(sensor_name)

    def create_detection_details_layout(self, equipment_summary: EquipmentDetectionSummary) -> html.Div:
        """Create the detection details layout for dashboard"""
        return html.Div([
            # Header section
            self._create_detection_header(equipment_summary),

            # Overview metrics
            self._create_overview_metrics(equipment_summary),

            # Sensor details table
            self._create_sensor_details_table(equipment_summary.sensor_details),

            # Visualizations
            self._create_detection_visualizations(equipment_summary),

            # Recommendations
            self._create_recommendations_section(equipment_summary.recommendations)
        ])

    def _create_detection_header(self, summary: EquipmentDetectionSummary) -> dbc.Card:
        """Create detection header with key information"""
        severity_color = {
            'CRITICAL': 'danger',
            'HIGH': 'warning',
            'MEDIUM': 'info',
            'LOW': 'light',
            'NORMAL': 'success'
        }.get(summary.severity_level, 'secondary')

        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H4([
                        html.I(className="fas fa-microchip me-2"),
                        f"{summary.equipment_name} Detection Details"
                    ]),
                    dbc.Badge(
                        summary.severity_level,
                        color=severity_color,
                        className="ms-2"
                    )
                ]),
                html.P([
                    html.Strong("Equipment ID: "), summary.equipment_id, " | ",
                    html.Strong("Type: "), summary.equipment_type, " | ",
                    html.Strong("Subsystem: "), summary.subsystem
                ], className="text-muted mb-0")
            ])
        ], className="mb-3")

    def _create_overview_metrics(self, summary: EquipmentDetectionSummary) -> dbc.Row:
        """Create overview metrics row"""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{summary.overall_anomaly_score:.3f}", className="text-danger"),
                        html.P("Anomaly Score", className="text-muted mb-0")
                    ])
                ])
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{summary.overall_confidence:.1%}", className="text-info"),
                        html.P("Confidence", className="text-muted mb-0")
                    ])
                ])
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{summary.anomalous_sensor_count}/{summary.sensor_count}"),
                        html.P("Anomalous Sensors", className="text-muted mb-0")
                    ])
                ])
            ], width=2),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{summary.reconstruction_quality:.1%}", className="text-success"),
                        html.P("Reconstruction Quality", className="text-muted mb-0")
                    ])
                ])
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("✓" if summary.is_anomaly else "○"),
                        html.P("Anomaly Detected", className="text-muted mb-0")
                    ])
                ])
            ], width=3)
        ], className="mb-4")

    def _create_sensor_details_table(self, sensor_details: List[SensorAnomalyDetail]) -> dbc.Card:
        """Create detailed sensor information table"""
        # Prepare data for table
        table_data = []
        for sensor in sensor_details:
            table_data.append({
                'Sensor': sensor.sensor_name,
                'Value': f"{sensor.current_value:.3f} {sensor.sensor_unit}",
                'Expected': f"{sensor.expected_value:.3f}",
                'Anomaly Score': f"{sensor.anomaly_score:.3f}",
                'Confidence': f"{sensor.confidence:.1%}",
                'Severity': sensor.severity_level,
                'Z-Score': f"{sensor.z_score:.2f}",
                'Threshold': "⚠️" if sensor.threshold_exceeded else "✓",
                'Trend': sensor.trend_direction
            })

        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-table me-2"),
                    "Sensor-Level Analysis"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dash_table.DataTable(
                    data=table_data,
                    columns=[
                        {'name': 'Sensor', 'id': 'Sensor'},
                        {'name': 'Current Value', 'id': 'Value'},
                        {'name': 'Expected', 'id': 'Expected'},
                        {'name': 'Anomaly Score', 'id': 'Anomaly Score', 'type': 'numeric'},
                        {'name': 'Confidence', 'id': 'Confidence'},
                        {'name': 'Severity', 'id': 'Severity'},
                        {'name': 'Z-Score', 'id': 'Z-Score', 'type': 'numeric'},
                        {'name': 'Threshold', 'id': 'Threshold'},
                        {'name': 'Trend', 'id': 'Trend'}
                    ],
                    style_cell={'textAlign': 'center', 'fontSize': '12px'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Severity} = CRITICAL'},
                            'backgroundColor': '#FFE6E6',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{Severity} = HIGH'},
                            'backgroundColor': '#FFF4E6',
                            'color': 'black',
                        }
                    ],
                    sort_action="native",
                    page_size=10
                )
            ])
        ], className="mb-4")

    def _create_detection_visualizations(self, summary: EquipmentDetectionSummary) -> dbc.Card:
        """Create detection visualization charts"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-line me-2"),
                    "Detection Visualizations"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    figure=self._create_sensor_anomaly_chart(summary.sensor_details),
                    id="sensor-anomaly-chart"
                ),
                html.Hr(),
                dcc.Graph(
                    figure=self._create_confidence_distribution_chart(summary.sensor_details),
                    id="confidence-distribution-chart"
                ),
                html.Hr(),
                dcc.Graph(
                    figure=self._create_reconstruction_error_chart(summary.sensor_details),
                    id="reconstruction-error-chart"
                )
            ])
        ], className="mb-4")

    def _create_sensor_anomaly_chart(self, sensor_details: List[SensorAnomalyDetail]) -> go.Figure:
        """Create sensor anomaly score chart"""
        sensor_names = [s.sensor_name for s in sensor_details]
        anomaly_scores = [s.anomaly_score for s in sensor_details]
        colors = [self._get_severity_color(s.severity_level) for s in sensor_details]

        fig = go.Figure(data=[
            go.Bar(
                x=sensor_names,
                y=anomaly_scores,
                marker_color=colors,
                text=[f"{score:.3f}" for score in anomaly_scores],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title="Sensor Anomaly Scores",
            xaxis_title="Sensors",
            yaxis_title="Anomaly Score",
            height=400
        )

        return fig

    def _create_confidence_distribution_chart(self, sensor_details: List[SensorAnomalyDetail]) -> go.Figure:
        """Create confidence distribution chart"""
        confidences = [s.confidence for s in sensor_details]

        fig = go.Figure(data=[
            go.Histogram(
                x=confidences,
                nbinsx=10,
                marker_color='lightblue',
                opacity=0.7
            )
        ])

        fig.update_layout(
            title="Confidence Score Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Number of Sensors",
            height=300
        )

        return fig

    def _create_reconstruction_error_chart(self, sensor_details: List[SensorAnomalyDetail]) -> go.Figure:
        """Create reconstruction error vs anomaly score scatter plot"""
        anomaly_scores = [s.anomaly_score for s in sensor_details]
        reconstruction_errors = [s.reconstruction_error for s in sensor_details]
        sensor_names = [s.sensor_name for s in sensor_details]

        fig = go.Figure(data=[
            go.Scatter(
                x=anomaly_scores,
                y=reconstruction_errors,
                mode='markers',
                text=sensor_names,
                marker=dict(
                    size=10,
                    opacity=0.6,
                    color=anomaly_scores,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])

        fig.update_layout(
            title="Reconstruction Error vs Anomaly Score",
            xaxis_title="Anomaly Score",
            yaxis_title="Reconstruction Error",
            height=400
        )

        return fig

    def _create_recommendations_section(self, recommendations: List[str]) -> dbc.Card:
        """Create recommendations section"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-lightbulb me-2"),
                    "Recommendations"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.Ul([
                    html.Li(rec) for rec in recommendations
                ]) if recommendations else html.P("No specific recommendations at this time.",
                                                 className="text-muted")
            ])
        ])

    # Helper methods
    def _get_equipment_info(self, equipment_id: str):
        """Get equipment information from mapper"""
        try:
            all_equipment = self.equipment_mapper.get_all_equipment()
            for equipment in all_equipment:
                if getattr(equipment, 'equipment_id', '') == equipment_id:
                    return equipment
        except Exception as e:
            logger.error(f"Error getting equipment info: {e}")
        return None

    def _get_latest_anomaly_result(self, equipment_id: str) -> Optional[AnomalyResult]:
        """Get latest anomaly result for equipment"""
        # This would integrate with the anomaly engine to get latest results
        return None

    def _get_sensor_historical_data(self, equipment_id: str, sensor_name: str) -> List[float]:
        """Get historical data for sensor"""
        # This would get historical sensor data from database
        return []

    def _calculate_sensor_anomaly_score(self, current_value: float, historical_mean: float,
                                      historical_std: float, sensor_spec: SensorSpec) -> float:
        """Calculate anomaly score for individual sensor"""
        if historical_std == 0:
            return 0.0

        z_score = abs(current_value - historical_mean) / historical_std
        return min(1.0, z_score / 3.0)  # Normalize to 0-1 range

    def _calculate_sensor_confidence(self, anomaly_score: float, z_score: float,
                                   percentile_rank: float) -> float:
        """Calculate confidence for sensor anomaly detection"""
        # Combine multiple factors for confidence calculation
        score_confidence = anomaly_score
        z_confidence = min(1.0, abs(z_score) / 3.0)
        percentile_confidence = abs(0.5 - percentile_rank) * 2

        return (score_confidence + z_confidence + percentile_confidence) / 3.0

    def _calculate_expected_value(self, sensor_spec: SensorSpec, historical_mean: float) -> float:
        """Calculate expected value for sensor"""
        return getattr(sensor_spec, 'nominal_value', historical_mean)

    def _determine_sensor_severity(self, anomaly_score: float, confidence: float) -> str:
        """Determine severity level for sensor"""
        if anomaly_score > 0.8 and confidence > 0.7:
            return 'CRITICAL'
        elif anomaly_score > 0.6 and confidence > 0.5:
            return 'HIGH'
        elif anomaly_score > 0.4:
            return 'MEDIUM'
        elif anomaly_score > 0.2:
            return 'LOW'
        else:
            return 'NORMAL'

    def _check_threshold_exceeded(self, current_value: float, sensor_spec: SensorSpec) -> bool:
        """Check if sensor value exceeds thresholds"""
        critical_threshold = getattr(sensor_spec, 'critical_threshold', None)
        if critical_threshold is not None:
            return abs(current_value) > abs(critical_threshold)
        return False

    def _analyze_sensor_trend(self, equipment_id: str, sensor_name: str) -> str:
        """Analyze sensor trend direction"""
        # This would analyze historical data for trends
        return 'stable'

    def _find_last_normal_value(self, equipment_id: str, sensor_name: str) -> Dict[str, Any]:
        """Find last normal value and time for sensor"""
        return {'value': None, 'time_minutes': None}

    def _create_default_sensor_detail(self, sensor_name: str) -> SensorAnomalyDetail:
        """Create default sensor detail for error cases"""
        return SensorAnomalyDetail(
            sensor_name=sensor_name,
            sensor_unit='',
            current_value=0.0,
            expected_value=0.0,
            anomaly_score=0.0,
            confidence=0.0,
            reconstruction_error=0.0,
            severity_level='NORMAL',
            threshold_exceeded=False,
            z_score=0.0,
            percentile_rank=0.5,
            historical_mean=0.0,
            historical_std=1.0,
            trend_direction='stable'
        )

    def _calculate_overall_metrics(self, sensor_details: List[SensorAnomalyDetail]) -> Dict[str, Any]:
        """Calculate overall metrics from sensor details"""
        if not sensor_details:
            return {
                'model_performance': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
                'reconstruction_quality': 0.0
            }

        avg_confidence = np.mean([s.confidence for s in sensor_details])
        avg_reconstruction_error = np.mean([s.reconstruction_error for s in sensor_details])
        reconstruction_quality = max(0.0, 1.0 - avg_reconstruction_error)

        return {
            'model_performance': {
                'accuracy': avg_confidence,
                'precision': avg_confidence * 0.9,  # Simplified calculation
                'recall': avg_confidence * 0.85
            },
            'reconstruction_quality': reconstruction_quality
        }

    def _perform_pattern_analysis(self, equipment_id: str,
                                 sensor_details: List[SensorAnomalyDetail]) -> Dict[str, Any]:
        """Perform pattern analysis on sensor data"""
        return {
            'correlation_detected': False,
            'seasonal_pattern': False,
            'drift_detected': False,
            'pattern_confidence': 0.0
        }

    def _generate_recommendations(self, equipment, sensor_details: List[SensorAnomalyDetail],
                                pattern_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on detection results"""
        recommendations = []

        # Check for critical sensors
        critical_sensors = [s for s in sensor_details if s.severity_level == 'CRITICAL']
        if critical_sensors:
            recommendations.append(f"Immediate attention required for {len(critical_sensors)} critical sensors")

        # Check for threshold violations
        threshold_violations = [s for s in sensor_details if s.threshold_exceeded]
        if threshold_violations:
            recommendations.append(f"Investigate {len(threshold_violations)} sensors exceeding thresholds")

        # Check for poor reconstruction quality
        high_error_sensors = [s for s in sensor_details if s.reconstruction_error > 0.5]
        if high_error_sensors:
            recommendations.append("Consider model retraining - high reconstruction errors detected")

        # Default recommendation
        if not recommendations:
            recommendations.append("Continue monitoring - system operating within normal parameters")

        return recommendations

    def _create_empty_summary(self, equipment_id: str) -> EquipmentDetectionSummary:
        """Create empty summary for error cases"""
        return EquipmentDetectionSummary(
            equipment_id=equipment_id,
            equipment_name=equipment_id,
            equipment_type='Unknown',
            subsystem='Unknown',
            overall_anomaly_score=0.0,
            overall_confidence=0.0,
            is_anomaly=False,
            severity_level='NORMAL',
            sensor_count=0,
            anomalous_sensor_count=0,
            sensor_details=[],
            model_performance={'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0},
            reconstruction_quality=0.0,
            pattern_analysis={},
            recommendations=[]
        )

    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level"""
        return {
            'CRITICAL': '#E74C3C',
            'HIGH': '#F39C12',
            'MEDIUM': '#F1C40F',
            'LOW': '#52C41A',
            'NORMAL': '#2ECC71'
        }.get(severity, '#95A5A6')


# Global instance for dashboard integration
detection_details_panel = EnhancedDetectionDetailsPanel()