"""
Risk Matrix Visualization System
Interactive risk assessment and visualization for IoT equipment portfolio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import json
import colorsys

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.colors import qualitative, sequential
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.forecasting.failure_probability import FailurePrediction, SeverityLevel, FailureMode
from src.forecasting.scenario_analysis import ScenarioResult
from config.settings import settings, get_config

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for the risk matrix"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for a component"""
    equipment_id: str
    component_id: str
    probability_score: float  # 0-1
    impact_score: float  # 0-1
    severity_score: float  # 0-1
    timeline_score: float  # 0-1 (urgency)
    overall_risk_score: float  # 0-1
    risk_level: RiskLevel
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    mitigation_actions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    # Additional risk dimensions
    financial_impact: float = 0.0  # $ cost impact
    operational_impact: float = 0.0  # Operational disruption (0-1)
    safety_impact: float = 0.0  # Safety risk (0-1)
    environmental_impact: float = 0.0  # Environmental risk (0-1)

    # Uncertainty measures
    assessment_confidence: float = 0.8  # Confidence in assessment (0-1)
    data_quality: float = 0.8  # Quality of underlying data (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'equipment_id': self.equipment_id,
            'component_id': self.component_id,
            'probability_score': self.probability_score,
            'impact_score': self.impact_score,
            'severity_score': self.severity_score,
            'timeline_score': self.timeline_score,
            'overall_risk_score': self.overall_risk_score,
            'risk_level': self.risk_level.value,
            'contributing_factors': self.contributing_factors,
            'mitigation_actions': self.mitigation_actions,
            'financial_impact': self.financial_impact,
            'operational_impact': self.operational_impact,
            'safety_impact': self.safety_impact,
            'environmental_impact': self.environmental_impact,
            'assessment_confidence': self.assessment_confidence,
            'data_quality': self.data_quality,
            'last_updated': self.last_updated.isoformat()
        }


class RiskMatrixCalculator:
    """Calculator for risk matrix scores and assessments"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk calculator

        Args:
            config: Risk assessment configuration
        """
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default risk assessment configuration"""
        return {
            'risk_matrix': {
                'probability_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9],
                'impact_thresholds': [0.1, 0.3, 0.5, 0.7, 0.9],
                'risk_levels': {
                    'very_low': {'min': 0.0, 'max': 0.1, 'color': '#2E8B57'},
                    'low': {'min': 0.1, 'max': 0.3, 'color': '#32CD32'},
                    'medium': {'min': 0.3, 'max': 0.5, 'color': '#FFD700'},
                    'high': {'min': 0.5, 'max': 0.7, 'color': '#FF8C00'},
                    'very_high': {'min': 0.7, 'max': 0.9, 'color': '#FF4500'},
                    'critical': {'min': 0.9, 'max': 1.0, 'color': '#DC143C'}
                }
            },
            'weights': {
                'probability': 0.3,
                'impact': 0.25,
                'severity': 0.25,
                'timeline': 0.2
            },
            'impact_factors': {
                'financial_weight': 0.4,
                'operational_weight': 0.3,
                'safety_weight': 0.2,
                'environmental_weight': 0.1
            }
        }

    def calculate_risk_assessment(self,
                                failure_prediction: FailurePrediction,
                                operational_context: Optional[Dict] = None) -> RiskAssessment:
        """
        Calculate comprehensive risk assessment from failure prediction

        Args:
            failure_prediction: Failure probability prediction
            operational_context: Additional operational context

        Returns:
            RiskAssessment object
        """

        # Calculate probability score (already 0-1)
        probability_score = failure_prediction.failure_probability

        # Calculate impact score from multiple factors
        impact_score = self._calculate_impact_score(failure_prediction, operational_context)

        # Calculate severity score
        severity_score = self._calculate_severity_score(failure_prediction)

        # Calculate timeline/urgency score
        timeline_score = self._calculate_timeline_score(failure_prediction)

        # Calculate overall risk score
        weights = self.config['weights']
        overall_risk_score = (
            weights['probability'] * probability_score +
            weights['impact'] * impact_score +
            weights['severity'] * severity_score +
            weights['timeline'] * timeline_score
        )

        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk_score)

        # Calculate detailed impact scores
        financial_impact = self._calculate_financial_impact(failure_prediction, operational_context)
        operational_impact = self._calculate_operational_impact(failure_prediction, operational_context)
        safety_impact = self._calculate_safety_impact(failure_prediction)
        environmental_impact = self._calculate_environmental_impact(failure_prediction)

        # Assessment confidence based on data quality
        confidence = self._calculate_assessment_confidence(failure_prediction)
        data_quality = self._assess_data_quality(failure_prediction)

        return RiskAssessment(
            equipment_id=failure_prediction.equipment_id,
            component_id=failure_prediction.component_id,
            probability_score=probability_score,
            impact_score=impact_score,
            severity_score=severity_score,
            timeline_score=timeline_score,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            contributing_factors=failure_prediction.contributing_factors,
            financial_impact=financial_impact,
            operational_impact=operational_impact,
            safety_impact=safety_impact,
            environmental_impact=environmental_impact,
            assessment_confidence=confidence,
            data_quality=data_quality
        )

    def _calculate_impact_score(self,
                              failure_prediction: FailurePrediction,
                              operational_context: Optional[Dict]) -> float:
        """Calculate composite impact score"""

        # Base impact from severity
        severity_impact = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.LOW: 0.2
        }.get(failure_prediction.severity, 0.5)

        # Equipment criticality impact
        criticality_multiplier = self._get_criticality_multiplier(
            failure_prediction.equipment_id, failure_prediction.component_id
        )

        # Cascade failure impact
        cascade_impact = self._calculate_cascade_impact(failure_prediction)

        # Operational context impact
        context_impact = 0.0
        if operational_context:
            # High load increases impact
            load_factor = operational_context.get('load_factor', 0.5)
            context_impact += load_factor * 0.3

            # Mission criticality
            mission_critical = operational_context.get('mission_critical', False)
            if mission_critical:
                context_impact += 0.4

        # Combine impact factors
        total_impact = min(
            severity_impact * criticality_multiplier + cascade_impact + context_impact,
            1.0
        )

        return total_impact

    def _calculate_severity_score(self, failure_prediction: FailurePrediction) -> float:
        """Calculate severity score"""
        severity_mapping = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.LOW: 0.2
        }
        return severity_mapping.get(failure_prediction.severity, 0.5)

    def _calculate_timeline_score(self, failure_prediction: FailurePrediction) -> float:
        """Calculate urgency score based on time to failure"""
        if failure_prediction.time_to_failure is None:
            return 0.5  # Default medium urgency

        # Convert time to failure to urgency score
        # Shorter time = higher urgency score
        time_hours = failure_prediction.time_to_failure

        if time_hours <= 24:  # Less than 1 day
            return 1.0
        elif time_hours <= 168:  # Less than 1 week
            return 0.8
        elif time_hours <= 720:  # Less than 1 month
            return 0.6
        elif time_hours <= 2160:  # Less than 3 months
            return 0.4
        else:
            return 0.2

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from overall score"""
        risk_levels = self.config['risk_matrix']['risk_levels']

        for level_name, thresholds in risk_levels.items():
            if thresholds['min'] <= risk_score <= thresholds['max']:
                return RiskLevel(level_name)

        return RiskLevel.MEDIUM  # Default

    def _get_criticality_multiplier(self, equipment_id: str, component_id: str) -> float:
        """Get criticality multiplier for equipment/component"""

        # Equipment configuration from CLAUDE.md
        equipment_criticality = {
            'smap': {
                'power_system': 1.0,
                'attitude_control': 1.0,
                'communication': 0.8,
                'thermal_control': 0.8,
                'payload_sensors': 0.8
            },
            'msl': {
                'mobility_front': 1.0,
                'mobility_rear': 1.0,
                'power_system': 1.0,
                'navigation': 1.0,
                'scientific': 0.8,
                'communication': 0.8,
                'environmental': 0.6
            }
        }

        equipment_type = equipment_id.lower()
        if equipment_type in equipment_criticality:
            component_criticality = equipment_criticality[equipment_type]
            return component_criticality.get(component_id, 0.7)

        return 0.7  # Default multiplier

    def _calculate_cascade_impact(self, failure_prediction: FailurePrediction) -> float:
        """Calculate potential cascade failure impact"""

        # Define component dependencies
        cascade_risks = {
            'power_system': 0.5,  # Power failure affects many systems
            'navigation': 0.4,    # Navigation affects mobility
            'communication': 0.3, # Communication affects operations
            'mobility_front': 0.3, # Mobility affects mission capability
            'attitude_control': 0.4  # Attitude control affects mission
        }

        return cascade_risks.get(failure_prediction.component_id, 0.1)

    def _calculate_financial_impact(self,
                                  failure_prediction: FailurePrediction,
                                  operational_context: Optional[Dict]) -> float:
        """Calculate financial impact in dollars"""

        # Base repair costs
        base_costs = {
            SeverityLevel.CRITICAL: 50000,
            SeverityLevel.HIGH: 20000,
            SeverityLevel.MEDIUM: 10000,
            SeverityLevel.LOW: 2000
        }

        repair_cost = base_costs.get(failure_prediction.severity, 10000)

        # Downtime costs
        downtime_hours = {
            SeverityLevel.CRITICAL: 48,
            SeverityLevel.HIGH: 24,
            SeverityLevel.MEDIUM: 8,
            SeverityLevel.LOW: 2
        }.get(failure_prediction.severity, 8)

        downtime_cost_per_hour = 1000  # Default
        if operational_context:
            downtime_cost_per_hour = operational_context.get('downtime_cost_per_hour', 1000)

        downtime_cost = downtime_hours * downtime_cost_per_hour

        return repair_cost + downtime_cost

    def _calculate_operational_impact(self,
                                    failure_prediction: FailurePrediction,
                                    operational_context: Optional[Dict]) -> float:
        """Calculate operational impact score (0-1)"""

        # Base operational impact
        base_impact = {
            SeverityLevel.CRITICAL: 0.9,
            SeverityLevel.HIGH: 0.7,
            SeverityLevel.MEDIUM: 0.4,
            SeverityLevel.LOW: 0.1
        }.get(failure_prediction.severity, 0.4)

        # Adjust for component criticality
        criticality_adjustment = self._get_criticality_multiplier(
            failure_prediction.equipment_id, failure_prediction.component_id
        )

        return min(base_impact * criticality_adjustment, 1.0)

    def _calculate_safety_impact(self, failure_prediction: FailurePrediction) -> float:
        """Calculate safety impact score (0-1)"""

        # Critical components have higher safety impact
        safety_critical_components = [
            'power_system', 'navigation', 'attitude_control', 'mobility_front', 'mobility_rear'
        ]

        if failure_prediction.component_id in safety_critical_components:
            severity_impact = {
                SeverityLevel.CRITICAL: 0.8,
                SeverityLevel.HIGH: 0.6,
                SeverityLevel.MEDIUM: 0.3,
                SeverityLevel.LOW: 0.1
            }.get(failure_prediction.severity, 0.3)
        else:
            severity_impact = {
                SeverityLevel.CRITICAL: 0.4,
                SeverityLevel.HIGH: 0.2,
                SeverityLevel.MEDIUM: 0.1,
                SeverityLevel.LOW: 0.05
            }.get(failure_prediction.severity, 0.1)

        return severity_impact

    def _calculate_environmental_impact(self, failure_prediction: FailurePrediction) -> float:
        """Calculate environmental impact score (0-1)"""

        # Environmental impact varies by component and mission
        environmental_components = ['thermal_control', 'environmental', 'power_system']

        if failure_prediction.component_id in environmental_components:
            return 0.3 if failure_prediction.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH] else 0.1

        return 0.05  # Minimal environmental impact for other components

    def _calculate_assessment_confidence(self, failure_prediction: FailurePrediction) -> float:
        """Calculate confidence in the assessment"""

        confidence_factors = []

        # Data availability
        if hasattr(failure_prediction, 'historical_mtbf') and failure_prediction.historical_mtbf:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)

        # Prediction uncertainty
        if hasattr(failure_prediction, 'uncertainty'):
            uncertainty_impact = 1.0 - failure_prediction.uncertainty
            confidence_factors.append(uncertainty_impact)
        else:
            confidence_factors.append(0.7)

        # Model maturity
        confidence_factors.append(0.8)  # Assume reasonable model maturity

        return np.mean(confidence_factors)

    def _assess_data_quality(self, failure_prediction: FailurePrediction) -> float:
        """Assess quality of underlying data"""

        quality_factors = []

        # Time series completeness
        quality_factors.append(0.8)  # Assume good data completeness

        # Sensor reliability
        quality_factors.append(0.85)  # Assume good sensor reliability

        # Historical data availability
        if hasattr(failure_prediction, 'historical_mtbf') and failure_prediction.historical_mtbf:
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.6)

        return np.mean(quality_factors)


class RiskMatrixVisualizer:
    """Risk matrix visualization engine"""

    def __init__(self, calculator: RiskMatrixCalculator):
        """
        Initialize visualizer

        Args:
            calculator: Risk assessment calculator
        """
        self.calculator = calculator
        self.config = calculator.config

    def create_risk_matrix_plot(self, risk_assessments: List[RiskAssessment]) -> go.Figure:
        """Create classic risk matrix (probability vs impact)"""

        if not risk_assessments:
            return go.Figure()

        # Prepare data
        df = pd.DataFrame([assessment.to_dict() for assessment in risk_assessments])

        # Create scatter plot
        fig = go.Figure()

        # Add risk level background regions
        self._add_risk_level_regions(fig)

        # Color mapping for risk levels
        color_map = {level_name: data['color']
                    for level_name, data in self.config['risk_matrix']['risk_levels'].items()}

        # Add scatter points
        for risk_level in RiskLevel:
            level_data = df[df['risk_level'] == risk_level.value]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data['probability_score'],
                    y=level_data['impact_score'],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color=color_map.get(risk_level.value, '#888888'),
                        line=dict(width=2, color='white')
                    ),
                    text=level_data['component_id'],
                    textposition='top center',
                    name=risk_level.value.replace('_', ' ').title(),
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        'Equipment: %{customdata[0]}<br>' +
                        'Probability: %{x:.3f}<br>' +
                        'Impact: %{y:.3f}<br>' +
                        'Overall Risk: %{customdata[1]:.3f}<br>' +
                        'Financial Impact: $%{customdata[2]:,.0f}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=level_data[['equipment_id', 'overall_risk_score', 'financial_impact']].values
                ))

        # Update layout
        fig.update_layout(
            title='Equipment Risk Matrix',
            xaxis_title='Failure Probability',
            yaxis_title='Impact Score',
            xaxis=dict(range=[0, 1], showgrid=True),
            yaxis=dict(range=[0, 1], showgrid=True),
            height=600,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def _add_risk_level_regions(self, fig: go.Figure):
        """Add colored background regions for risk levels"""

        # Define risk regions (simplified 5x5 grid)
        risk_regions = [
            # Very Low Risk
            {'x': [0, 0.2, 0.2, 0], 'y': [0, 0, 0.2, 0.2], 'color': '#2E8B57', 'opacity': 0.1},
            {'x': [0.2, 0.4, 0.4, 0.2], 'y': [0, 0, 0.2, 0.2], 'color': '#2E8B57', 'opacity': 0.1},
            {'x': [0, 0.2, 0.2, 0], 'y': [0.2, 0.2, 0.4, 0.4], 'color': '#2E8B57', 'opacity': 0.1},

            # Low Risk
            {'x': [0.4, 0.6, 0.6, 0.4], 'y': [0, 0, 0.2, 0.2], 'color': '#32CD32', 'opacity': 0.1},
            {'x': [0.2, 0.4, 0.4, 0.2], 'y': [0.2, 0.2, 0.4, 0.4], 'color': '#32CD32', 'opacity': 0.1},
            {'x': [0, 0.2, 0.2, 0], 'y': [0.4, 0.4, 0.6, 0.6], 'color': '#32CD32', 'opacity': 0.1},

            # Medium Risk
            {'x': [0.6, 0.8, 0.8, 0.6], 'y': [0, 0, 0.2, 0.2], 'color': '#FFD700', 'opacity': 0.1},
            {'x': [0.4, 0.6, 0.6, 0.4], 'y': [0.2, 0.2, 0.4, 0.4], 'color': '#FFD700', 'opacity': 0.1},
            {'x': [0.2, 0.4, 0.4, 0.2], 'y': [0.4, 0.4, 0.6, 0.6], 'color': '#FFD700', 'opacity': 0.1},
            {'x': [0, 0.2, 0.2, 0], 'y': [0.6, 0.6, 0.8, 0.8], 'color': '#FFD700', 'opacity': 0.1},

            # High Risk
            {'x': [0.8, 1.0, 1.0, 0.8], 'y': [0, 0, 0.2, 0.2], 'color': '#FF8C00', 'opacity': 0.1},
            {'x': [0.6, 0.8, 0.8, 0.6], 'y': [0.2, 0.2, 0.4, 0.4], 'color': '#FF8C00', 'opacity': 0.1},
            {'x': [0.4, 0.6, 0.6, 0.4], 'y': [0.4, 0.4, 0.6, 0.6], 'color': '#FF8C00', 'opacity': 0.1},
            {'x': [0.2, 0.4, 0.4, 0.2], 'y': [0.6, 0.6, 0.8, 0.8], 'color': '#FF8C00', 'opacity': 0.1},
            {'x': [0, 0.2, 0.2, 0], 'y': [0.8, 0.8, 1.0, 1.0], 'color': '#FF8C00', 'opacity': 0.1},

            # Very High/Critical Risk
            {'x': [0.8, 1.0, 1.0, 0.8], 'y': [0.2, 0.2, 1.0, 1.0], 'color': '#FF4500', 'opacity': 0.1},
            {'x': [0.6, 0.8, 0.8, 0.6], 'y': [0.4, 0.4, 1.0, 1.0], 'color': '#FF4500', 'opacity': 0.1},
            {'x': [0.4, 0.6, 0.6, 0.4], 'y': [0.6, 0.6, 1.0, 1.0], 'color': '#FF4500', 'opacity': 0.1},
            {'x': [0.2, 1.0, 1.0, 0.2], 'y': [0.8, 0.8, 1.0, 1.0], 'color': '#DC143C', 'opacity': 0.1},
        ]

        for region in risk_regions:
            fig.add_shape(
                type="path",
                path=f"M {region['x'][0]},{region['y'][0]} L {region['x'][1]},{region['y'][1]} L {region['x'][2]},{region['y'][2]} L {region['x'][3]},{region['y'][3]} Z",
                fillcolor=region['color'],
                opacity=region['opacity'],
                line=dict(width=0),
                layer="below"
            )

    def create_risk_heatmap(self, risk_assessments: List[RiskAssessment]) -> go.Figure:
        """Create risk heatmap by equipment and component"""

        if not risk_assessments:
            return go.Figure()

        # Prepare data
        df = pd.DataFrame([assessment.to_dict() for assessment in risk_assessments])

        # Create pivot table
        pivot_data = df.pivot_table(
            values='overall_risk_score',
            index='component_id',
            columns='equipment_id',
            fill_value=0
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn_r',  # Red-Yellow-Green reversed
            colorbar=dict(title="Risk Score"),
            hovetemplate=(
                'Equipment: %{x}<br>' +
                'Component: %{y}<br>' +
                'Risk Score: %{z:.3f}<br>' +
                '<extra></extra>'
            )
        ))

        fig.update_layout(
            title='Risk Heatmap by Equipment and Component',
            xaxis_title='Equipment',
            yaxis_title='Component',
            height=500,
            template='plotly_white'
        )

        return fig

    def create_risk_dashboard(self, risk_assessments: List[RiskAssessment]) -> go.Figure:
        """Create comprehensive risk dashboard with multiple views"""

        if not risk_assessments:
            return go.Figure()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Risk Matrix', 'Risk Distribution', 'Impact Analysis', 'Timeline View'],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        df = pd.DataFrame([assessment.to_dict() for assessment in risk_assessments])

        # 1. Risk Matrix (top-left)
        color_map = {level_name: data['color']
                    for level_name, data in self.config['risk_matrix']['risk_levels'].items()}

        for risk_level in df['risk_level'].unique():
            level_data = df[df['risk_level'] == risk_level]
            fig.add_trace(
                go.Scatter(
                    x=level_data['probability_score'],
                    y=level_data['impact_score'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_map.get(risk_level, '#888888')
                    ),
                    name=risk_level.replace('_', ' ').title(),
                    showlegend=True
                ),
                row=1, col=1
            )

        # 2. Risk Distribution (top-right)
        risk_counts = df['risk_level'].value_counts()
        fig.add_trace(
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker_color=[color_map.get(level, '#888888') for level in risk_counts.index],
                name='Risk Distribution',
                showlegend=False
            ),
            row=1, col=2
        )

        # 3. Impact Analysis (bottom-left)
        avg_impacts = df.groupby('equipment_id')[['financial_impact', 'operational_impact', 'safety_impact']].mean()

        fig.add_trace(
            go.Bar(
                x=avg_impacts.index,
                y=avg_impacts['financial_impact'],
                name='Financial Impact',
                showlegend=False
            ),
            row=2, col=1
        )

        # 4. Timeline View (bottom-right)
        fig.add_trace(
            go.Scatter(
                x=df['timeline_score'],
                y=df['overall_risk_score'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['probability_score'],
                    colorscale='Reds',
                    showscale=False
                ),
                text=df['component_id'],
                name='Timeline vs Risk',
                showlegend=False
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title='Comprehensive Risk Assessment Dashboard',
            height=800,
            template='plotly_white'
        )

        # Update axes
        fig.update_xaxes(title_text="Probability", row=1, col=1)
        fig.update_yaxes(title_text="Impact", row=1, col=1)
        fig.update_xaxes(title_text="Risk Level", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Equipment", row=2, col=1)
        fig.update_yaxes(title_text="Financial Impact ($)", row=2, col=1)
        fig.update_xaxes(title_text="Timeline Score", row=2, col=2)
        fig.update_yaxes(title_text="Overall Risk", row=2, col=2)

        return fig

    def create_risk_trends_plot(self,
                               historical_assessments: Dict[datetime, List[RiskAssessment]]) -> go.Figure:
        """Create risk trends over time"""

        if not historical_assessments:
            return go.Figure()

        fig = go.Figure()

        # Calculate average risk scores over time
        timestamps = sorted(historical_assessments.keys())
        avg_risks = []
        high_risk_counts = []
        critical_risk_counts = []

        for timestamp in timestamps:
            assessments = historical_assessments[timestamp]
            if assessments:
                risks = [a.overall_risk_score for a in assessments]
                avg_risks.append(np.mean(risks))

                high_risk_counts.append(
                    sum(1 for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH])
                )
                critical_risk_counts.append(
                    sum(1 for a in assessments if a.risk_level == RiskLevel.CRITICAL)
                )
            else:
                avg_risks.append(0)
                high_risk_counts.append(0)
                critical_risk_counts.append(0)

        # Add traces
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=avg_risks,
            mode='lines+markers',
            name='Average Risk Score',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=high_risk_counts,
            mode='lines+markers',
            name='High Risk Components',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))

        fig.add_trace(go.Scatter(
            x=timestamps,
            y=critical_risk_counts,
            mode='lines+markers',
            name='Critical Risk Components',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))

        # Update layout with dual y-axis
        fig.update_layout(
            title='Risk Trends Over Time',
            xaxis_title='Time',
            yaxis=dict(title='Average Risk Score', side='left'),
            yaxis2=dict(title='Component Count', side='right', overlaying='y'),
            height=600,
            template='plotly_white'
        )

        return fig


class RiskMatrixSystem:
    """Complete risk matrix system with calculation and visualization"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize risk matrix system

        Args:
            config: System configuration
        """
        self.calculator = RiskMatrixCalculator(config)
        self.visualizer = RiskMatrixVisualizer(self.calculator)
        self.risk_assessments = {}  # Equipment_Component -> RiskAssessment
        self.historical_assessments = {}  # datetime -> List[RiskAssessment]

    def update_risk_assessments(self,
                              failure_predictions: List[FailurePrediction],
                              operational_context: Optional[Dict] = None) -> List[RiskAssessment]:
        """Update risk assessments from failure predictions"""

        assessments = []

        for prediction in failure_predictions:
            assessment = self.calculator.calculate_risk_assessment(
                prediction, operational_context
            )

            key = f"{assessment.equipment_id}_{assessment.component_id}"
            self.risk_assessments[key] = assessment
            assessments.append(assessment)

        # Store historical snapshot
        self.historical_assessments[datetime.now()] = assessments.copy()

        return assessments

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""

        if not self.risk_assessments:
            return {'error': 'No risk assessments available'}

        assessments = list(self.risk_assessments.values())

        # Calculate summary statistics
        risk_levels = [a.risk_level for a in assessments]
        risk_scores = [a.overall_risk_score for a in assessments]

        level_counts = {level: risk_levels.count(level) for level in RiskLevel}

        summary = {
            'total_components': len(assessments),
            'average_risk_score': np.mean(risk_scores),
            'highest_risk_score': np.max(risk_scores),
            'risk_level_distribution': {k.value: v for k, v in level_counts.items()},
            'critical_components': [
                a.component_id for a in assessments
                if a.risk_level == RiskLevel.CRITICAL
            ],
            'high_risk_components': [
                a.component_id for a in assessments
                if a.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
            ],
            'total_financial_impact': sum(a.financial_impact for a in assessments),
            'average_confidence': np.mean([a.assessment_confidence for a in assessments]),
            'last_updated': max(a.last_updated for a in assessments).isoformat()
        }

        return summary

    def get_top_risks(self, n: int = 10, sort_by: str = 'overall_risk_score') -> List[RiskAssessment]:
        """Get top N risk components"""

        assessments = list(self.risk_assessments.values())

        if sort_by == 'overall_risk_score':
            sorted_assessments = sorted(assessments, key=lambda x: x.overall_risk_score, reverse=True)
        elif sort_by == 'financial_impact':
            sorted_assessments = sorted(assessments, key=lambda x: x.financial_impact, reverse=True)
        elif sort_by == 'timeline_score':
            sorted_assessments = sorted(assessments, key=lambda x: x.timeline_score, reverse=True)
        else:
            sorted_assessments = assessments

        return sorted_assessments[:n]

    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""

        summary = self.get_risk_summary()
        top_risks = self.get_top_risks(5)

        # Create visualizations
        all_assessments = list(self.risk_assessments.values())
        risk_matrix_fig = self.visualizer.create_risk_matrix_plot(all_assessments)
        risk_dashboard_fig = self.visualizer.create_risk_dashboard(all_assessments)

        report = {
            'summary': summary,
            'top_risks': [risk.to_dict() for risk in top_risks],
            'recommendations': self._generate_recommendations(all_assessments),
            'visualizations': {
                'risk_matrix': risk_matrix_fig.to_json(),
                'dashboard': risk_dashboard_fig.to_json()
            },
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _generate_recommendations(self, assessments: List[RiskAssessment]) -> List[str]:
        """Generate risk management recommendations"""

        recommendations = []

        # Critical risk recommendations
        critical_risks = [a for a in assessments if a.risk_level == RiskLevel.CRITICAL]
        if critical_risks:
            recommendations.append(
                f"URGENT: {len(critical_risks)} components at critical risk require immediate attention"
            )

        # High financial impact recommendations
        high_financial = [a for a in assessments if a.financial_impact > 50000]
        if high_financial:
            recommendations.append(
                f"Consider priority maintenance for {len(high_financial)} high-cost components"
            )

        # Timeline recommendations
        urgent_timeline = [a for a in assessments if a.timeline_score > 0.8]
        if urgent_timeline:
            recommendations.append(
                f"Schedule immediate inspections for {len(urgent_timeline)} time-critical components"
            )

        # Cascade failure recommendations
        power_risks = [a for a in assessments if a.component_id == 'power_system' and a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if power_risks:
            recommendations.append(
                "Power system issues detected - review backup power and dependent systems"
            )

        # Low confidence recommendations
        low_confidence = [a for a in assessments if a.assessment_confidence < 0.6]
        if len(low_confidence) > len(assessments) * 0.3:
            recommendations.append(
                "Consider additional data collection to improve risk assessment confidence"
            )

        return recommendations

    def save_risk_data(self, filepath: Path):
        """Save risk assessment data"""

        export_data = {
            'current_assessments': {
                key: assessment.to_dict()
                for key, assessment in self.risk_assessments.items()
            },
            'historical_assessments': {
                timestamp.isoformat(): [a.to_dict() for a in assessments]
                for timestamp, assessments in self.historical_assessments.items()
            },
            'config': self.calculator.config,
            'exported_at': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Risk data saved to {filepath}")


if __name__ == "__main__":
    # Demo and testing
    print("\n" + "="*60)
    print("Testing Risk Matrix System")
    print("="*60)

    # Create risk matrix system
    risk_system = RiskMatrixSystem()

    # Create mock failure predictions
    print("\n1. Creating mock failure predictions...")

    mock_predictions = [
        FailurePrediction(
            equipment_id='SMAP',
            component_id='power_system',
            failure_probability=0.7,
            time_to_failure=48,
            severity=SeverityLevel.CRITICAL,
            contributing_factors={'degradation': 0.5, 'anomalies': 0.3}
        ),
        FailurePrediction(
            equipment_id='SMAP',
            component_id='communication',
            failure_probability=0.3,
            time_to_failure=168,
            severity=SeverityLevel.MEDIUM,
            contributing_factors={'forecast_trends': 0.2}
        ),
        FailurePrediction(
            equipment_id='MSL',
            component_id='mobility_front',
            failure_probability=0.9,
            time_to_failure=24,
            severity=SeverityLevel.CRITICAL,
            contributing_factors={'degradation': 0.7, 'operational_stress': 0.4}
        ),
        FailurePrediction(
            equipment_id='MSL',
            component_id='environmental',
            failure_probability=0.1,
            time_to_failure=720,
            severity=SeverityLevel.LOW,
            contributing_factors={'forecast_trends': 0.1}
        )
    ]

    # Update risk assessments
    print("2. Calculating risk assessments...")

    operational_context = {
        'downtime_cost_per_hour': 1000,
        'mission_critical': True,
        'load_factor': 0.8
    }

    assessments = risk_system.update_risk_assessments(mock_predictions, operational_context)

    print(f"Generated {len(assessments)} risk assessments")

    # Display risk summary
    print("\n3. Risk Summary:")
    summary = risk_system.get_risk_summary()

    print(f"  Total Components: {summary['total_components']}")
    print(f"  Average Risk Score: {summary['average_risk_score']:.3f}")
    print(f"  Highest Risk Score: {summary['highest_risk_score']:.3f}")
    print(f"  Critical Components: {summary['critical_components']}")
    print(f"  Total Financial Impact: ${summary['total_financial_impact']:,.0f}")

    # Display top risks
    print("\n4. Top Risk Components:")
    top_risks = risk_system.get_top_risks(3)

    for i, risk in enumerate(top_risks, 1):
        print(f"  {i}. {risk.equipment_id}/{risk.component_id}")
        print(f"     Risk Level: {risk.risk_level.value}")
        print(f"     Risk Score: {risk.overall_risk_score:.3f}")
        print(f"     Financial Impact: ${risk.financial_impact:,.0f}")
        print(f"     Time to Failure: {risk.timeline_score:.3f}")

    # Generate visualizations
    print("\n5. Generating visualizations...")

    risk_matrix_fig = risk_system.visualizer.create_risk_matrix_plot(assessments)
    print(f"  Risk matrix plot created with {len(assessments)} points")

    risk_heatmap_fig = risk_system.visualizer.create_risk_heatmap(assessments)
    print(f"  Risk heatmap created")

    dashboard_fig = risk_system.visualizer.create_risk_dashboard(assessments)
    print(f"  Risk dashboard created")

    # Generate recommendations
    print("\n6. Risk Management Recommendations:")
    report = risk_system.generate_risk_report()

    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")

    print("\n" + "="*60)
    print("Risk matrix system test complete")
    print("="*60)