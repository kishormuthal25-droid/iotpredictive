"""
Equipment-Specific Threshold Management System
Dynamic threshold management for CRITICAL, HIGH, and MEDIUM priority equipment
"""

import numpy as np
import pandas as pd
from dash import html, dcc, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import project modules
from src.anomaly_detection.nasa_anomaly_engine import nasa_anomaly_engine, EquipmentThreshold
from src.data_ingestion.equipment_mapper import equipment_mapper, EquipmentComponent

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfiguration:
    """Configuration for equipment thresholds"""
    equipment_id: str
    equipment_type: str
    subsystem: str
    criticality: str

    # Current thresholds
    critical_threshold: float
    high_threshold: float
    medium_threshold: float
    warning_threshold: float

    # Adaptive parameters
    adaptation_rate: float = 0.1
    sensitivity_multiplier: float = 1.0
    seasonal_adjustment: bool = True
    learning_enabled: bool = True

    # Performance metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    # Historical effectiveness
    effectiveness_history: List[float] = field(default_factory=list)
    threshold_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ThresholdOptimizationResult:
    """Result of threshold optimization"""
    equipment_id: str
    optimization_type: str
    old_thresholds: Dict[str, float]
    new_thresholds: Dict[str, float]
    improvement_score: float
    confidence: float
    justification: str
    applied: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class EquipmentSpecificThresholdManager:
    """
    Manages equipment-specific thresholds with dynamic optimization
    Handles CRITICAL vs HIGH vs MEDIUM equipment priority differences
    """

    def __init__(self):
        """Initialize the Equipment-Specific Threshold Manager"""
        self.anomaly_engine = nasa_anomaly_engine
        self.equipment_mapper = equipment_mapper

        # Threshold configurations for all equipment
        self.threshold_configs = {}

        # Optimization history
        self.optimization_history = {}

        # Performance tracking
        self.performance_metrics = {}

        # Equipment priority configurations
        self.priority_configs = {
            'CRITICAL': {
                'threshold_sensitivity': 1.2,    # More sensitive thresholds
                'adaptation_rate': 0.05,          # Slower adaptation
                'min_threshold_gap': 0.1,         # Minimum gap between thresholds
                'escalation_speed': 'immediate',
                'false_positive_tolerance': 0.05  # Low tolerance
            },
            'HIGH': {
                'threshold_sensitivity': 1.0,
                'adaptation_rate': 0.1,
                'min_threshold_gap': 0.08,
                'escalation_speed': 'fast',
                'false_positive_tolerance': 0.1
            },
            'MEDIUM': {
                'threshold_sensitivity': 0.8,
                'adaptation_rate': 0.15,
                'min_threshold_gap': 0.05,
                'escalation_speed': 'normal',
                'false_positive_tolerance': 0.15
            },
            'LOW': {
                'threshold_sensitivity': 0.6,
                'adaptation_rate': 0.2,
                'min_threshold_gap': 0.03,
                'escalation_speed': 'scheduled',
                'false_positive_tolerance': 0.25
            }
        }

        # Subsystem-specific configurations
        self.subsystem_configs = {
            'POWER': {
                'base_thresholds': {'critical': 0.90, 'high': 0.75, 'medium': 0.60},
                'volatility_factor': 0.8,        # Power systems are more stable
                'seasonal_variance': 0.1
            },
            'MOBILITY': {
                'base_thresholds': {'critical': 0.85, 'high': 0.70, 'medium': 0.55},
                'volatility_factor': 1.2,        # Higher variance due to terrain
                'seasonal_variance': 0.15
            },
            'COMMUNICATION': {
                'base_thresholds': {'critical': 0.80, 'high': 0.65, 'medium': 0.50},
                'volatility_factor': 1.0,
                'seasonal_variance': 0.2          # Distance/orbital variations
            },
            'THERMAL': {
                'base_thresholds': {'critical': 0.85, 'high': 0.70, 'medium': 0.55},
                'volatility_factor': 1.1,
                'seasonal_variance': 0.25         # High seasonal variance
            },
            'ATTITUDE': {
                'base_thresholds': {'critical': 0.88, 'high': 0.73, 'medium': 0.58},
                'volatility_factor': 0.9,
                'seasonal_variance': 0.05
            },
            'PAYLOAD': {
                'base_thresholds': {'critical': 0.82, 'high': 0.67, 'medium': 0.52},
                'volatility_factor': 1.0,
                'seasonal_variance': 0.1
            }
        }

        self._initialize_threshold_configurations()
        logger.info("Equipment-Specific Threshold Manager initialized")

    def _initialize_threshold_configurations(self):
        """Initialize threshold configurations for all equipment"""
        try:
            all_equipment = self.equipment_mapper.get_all_equipment()

            for equipment in all_equipment:
                config = self._create_threshold_configuration(equipment)
                self.threshold_configs[equipment.equipment_id] = config

            logger.info(f"Initialized threshold configurations for {len(self.threshold_configs)} equipment")

        except Exception as e:
            logger.error(f"Error initializing threshold configurations: {e}")

    def _create_threshold_configuration(self, equipment: EquipmentComponent) -> ThresholdConfiguration:
        """Create threshold configuration for specific equipment"""
        equipment_id = equipment.equipment_id
        subsystem = equipment.subsystem
        criticality = equipment.criticality

        # Get base thresholds for subsystem
        subsystem_config = self.subsystem_configs.get(subsystem, self.subsystem_configs['POWER'])
        base_thresholds = subsystem_config['base_thresholds']

        # Get priority adjustments
        priority_config = self.priority_configs.get(criticality, self.priority_configs['MEDIUM'])
        sensitivity = priority_config['threshold_sensitivity']

        # Calculate adjusted thresholds
        critical_threshold = base_thresholds['critical'] * sensitivity
        high_threshold = base_thresholds['high'] * sensitivity
        medium_threshold = base_thresholds['medium'] * sensitivity
        warning_threshold = medium_threshold * 0.8

        # Ensure proper ordering and minimum gaps
        min_gap = priority_config['min_threshold_gap']
        warning_threshold = max(warning_threshold, 0.1)
        medium_threshold = max(medium_threshold, warning_threshold + min_gap)
        high_threshold = max(high_threshold, medium_threshold + min_gap)
        critical_threshold = max(critical_threshold, high_threshold + min_gap)
        critical_threshold = min(critical_threshold, 0.95)  # Cap at 95%

        return ThresholdConfiguration(
            equipment_id=equipment_id,
            equipment_type=equipment.equipment_type,
            subsystem=subsystem,
            criticality=criticality,
            critical_threshold=critical_threshold,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
            warning_threshold=warning_threshold,
            adaptation_rate=priority_config['adaptation_rate'],
            sensitivity_multiplier=sensitivity
        )

    def create_threshold_management_interface(self) -> html.Div:
        """Create threshold management interface"""
        return html.Div([
            # Header
            self._create_threshold_header(),

            # Equipment filter and selection
            self._create_equipment_filter(),

            # Threshold overview dashboard
            self._create_threshold_overview(),

            # Detailed threshold configuration
            self._create_threshold_configuration_panel(),

            # Optimization recommendations
            self._create_optimization_recommendations(),

            # Performance analytics
            self._create_performance_analytics()
        ])

    def _create_threshold_header(self) -> dbc.Card:
        """Create threshold management header"""
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H3([
                        html.I(className="fas fa-sliders-h me-2"),
                        "Equipment-Specific Threshold Management"
                    ]),
                    html.P("Dynamic threshold optimization for CRITICAL, HIGH, and MEDIUM priority equipment",
                           className="text-muted mb-0")
                ])
            ])
        ], className="mb-4")

    def _create_equipment_filter(self) -> dbc.Card:
        """Create equipment filtering interface"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-filter me-2"),
                    "Equipment Selection & Filtering"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Spacecraft:"),
                        dcc.Dropdown(
                            id="threshold-spacecraft-filter",
                            options=[
                                {"label": "All Spacecraft", "value": "all"},
                                {"label": "SMAP Satellite", "value": "SMAP"},
                                {"label": "MSL Mars Rover", "value": "MSL"}
                            ],
                            value="all"
                        )
                    ], width=3),

                    dbc.Col([
                        dbc.Label("Subsystem:"),
                        dcc.Dropdown(
                            id="threshold-subsystem-filter",
                            options=[
                                {"label": "All Subsystems", "value": "all"},
                                {"label": "Power Systems", "value": "POWER"},
                                {"label": "Mobility Systems", "value": "MOBILITY"},
                                {"label": "Communication Systems", "value": "COMMUNICATION"},
                                {"label": "Thermal Systems", "value": "THERMAL"},
                                {"label": "Attitude Control", "value": "ATTITUDE"},
                                {"label": "Payload Systems", "value": "PAYLOAD"}
                            ],
                            value="all"
                        )
                    ], width=3),

                    dbc.Col([
                        dbc.Label("Criticality:"),
                        dcc.Dropdown(
                            id="threshold-criticality-filter",
                            options=[
                                {"label": "All Priorities", "value": "all"},
                                {"label": "CRITICAL", "value": "CRITICAL"},
                                {"label": "HIGH", "value": "HIGH"},
                                {"label": "MEDIUM", "value": "MEDIUM"},
                                {"label": "LOW", "value": "LOW"}
                            ],
                            value="all"
                        )
                    ], width=3),

                    dbc.Col([
                        dbc.Label("Actions:"),
                        dbc.ButtonGroup([
                            dbc.Button("Auto-Optimize", id="auto-optimize-btn", color="primary", size="sm"),
                            dbc.Button("Reset to Defaults", id="reset-thresholds-btn", color="secondary", size="sm"),
                            dbc.Button("Export Config", id="export-thresholds-btn", color="info", size="sm")
                        ])
                    ], width=3)
                ])
            ])
        ], className="mb-4")

    def _create_threshold_overview(self) -> dbc.Card:
        """Create threshold overview dashboard"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-tachometer-alt me-2"),
                    "Threshold Overview Dashboard"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Summary metrics
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("85", className="text-success"),
                                html.P("Equipment Monitored", className="text-muted mb-0")
                            ])
                        ])
                    ], width=2),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("94.5%", className="text-info"),
                                html.P("Avg Accuracy", className="text-muted mb-0")
                            ])
                        ])
                    ], width=2),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("7.2%", className="text-warning"),
                                html.P("False Positive Rate", className="text-muted mb-0")
                            ])
                        ])
                    ], width=2),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("12", className="text-primary"),
                                html.P("Auto-Optimized Today", className="text-muted mb-0")
                            ])
                        ])
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("3", className="text-danger"),
                                html.P("Need Attention", className="text-muted mb-0")
                            ])
                        ])
                    ], width=3)
                ], className="mb-4"),

                # Threshold distribution chart
                dcc.Graph(
                    id="threshold-distribution-chart",
                    figure=self._create_threshold_distribution_chart()
                )
            ])
        ], className="mb-4")

    def _create_threshold_configuration_panel(self) -> dbc.Card:
        """Create detailed threshold configuration panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-cogs me-2"),
                    "Threshold Configuration Details"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                # Equipment selection
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Equipment:"),
                        dcc.Dropdown(
                            id="threshold-equipment-selector",
                            options=self._get_equipment_options(),
                            placeholder="Select equipment to configure..."
                        )
                    ], width=6),

                    dbc.Col([
                        dbc.Label("Configuration Mode:"),
                        dcc.Dropdown(
                            id="threshold-config-mode",
                            options=[
                                {"label": "Manual Configuration", "value": "manual"},
                                {"label": "Adaptive Learning", "value": "adaptive"},
                                {"label": "Performance Optimization", "value": "optimize"}
                            ],
                            value="manual"
                        )
                    ], width=6)
                ], className="mb-4"),

                # Threshold sliders
                html.Div(id="threshold-sliders-container"),

                # Configuration buttons
                dbc.ButtonGroup([
                    dbc.Button("Apply Changes", id="apply-threshold-changes", color="success"),
                    dbc.Button("Preview Impact", id="preview-threshold-impact", color="info"),
                    dbc.Button("Revert Changes", id="revert-threshold-changes", color="warning")
                ], className="mt-3")
            ])
        ], className="mb-4")

    def _create_optimization_recommendations(self) -> dbc.Card:
        """Create optimization recommendations panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-lightbulb me-2"),
                    "Optimization Recommendations"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                html.Div(id="optimization-recommendations-content")
            ])
        ], className="mb-4")

    def _create_performance_analytics(self) -> dbc.Card:
        """Create performance analytics panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H5([
                    html.I(className="fas fa-chart-line me-2"),
                    "Threshold Performance Analytics"
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="threshold-performance-chart",
                    figure=self._create_performance_analytics_chart()
                )
            ])
        ])

    def optimize_thresholds_for_equipment(self, equipment_id: str,
                                        optimization_type: str = "accuracy") -> ThresholdOptimizationResult:
        """
        Optimize thresholds for specific equipment

        Args:
            equipment_id: Equipment identifier
            optimization_type: Type of optimization ('accuracy', 'false_positive', 'sensitivity')

        Returns:
            Optimization result
        """
        try:
            config = self.threshold_configs.get(equipment_id)
            if not config:
                raise ValueError(f"No configuration found for equipment {equipment_id}")

            # Get historical performance data
            performance_data = self._get_equipment_performance_data(equipment_id)

            # Current thresholds
            old_thresholds = {
                'critical': config.critical_threshold,
                'high': config.high_threshold,
                'medium': config.medium_threshold,
                'warning': config.warning_threshold
            }

            # Optimize based on type
            if optimization_type == "accuracy":
                new_thresholds = self._optimize_for_accuracy(config, performance_data)
            elif optimization_type == "false_positive":
                new_thresholds = self._optimize_for_false_positives(config, performance_data)
            elif optimization_type == "sensitivity":
                new_thresholds = self._optimize_for_sensitivity(config, performance_data)
            else:
                new_thresholds = old_thresholds

            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(
                old_thresholds, new_thresholds, performance_data
            )

            # Generate justification
            justification = self._generate_optimization_justification(
                optimization_type, old_thresholds, new_thresholds, improvement_score
            )

            # Calculate confidence
            confidence = self._calculate_optimization_confidence(performance_data, improvement_score)

            result = ThresholdOptimizationResult(
                equipment_id=equipment_id,
                optimization_type=optimization_type,
                old_thresholds=old_thresholds,
                new_thresholds=new_thresholds,
                improvement_score=improvement_score,
                confidence=confidence,
                justification=justification
            )

            # Save to history
            if equipment_id not in self.optimization_history:
                self.optimization_history[equipment_id] = []
            self.optimization_history[equipment_id].append(result)

            logger.info(f"Optimized thresholds for {equipment_id}: {improvement_score:.3f} improvement")

            return result

        except Exception as e:
            logger.error(f"Error optimizing thresholds for {equipment_id}: {e}")
            return ThresholdOptimizationResult(
                equipment_id=equipment_id,
                optimization_type=optimization_type,
                old_thresholds={},
                new_thresholds={},
                improvement_score=0.0,
                confidence=0.0,
                justification=f"Optimization failed: {str(e)}"
            )

    def apply_threshold_optimization(self, equipment_id: str,
                                   optimization_result: ThresholdOptimizationResult) -> bool:
        """Apply threshold optimization result"""
        try:
            config = self.threshold_configs.get(equipment_id)
            if not config:
                return False

            new_thresholds = optimization_result.new_thresholds

            # Update configuration
            config.critical_threshold = new_thresholds['critical']
            config.high_threshold = new_thresholds['high']
            config.medium_threshold = new_thresholds['medium']
            config.warning_threshold = new_thresholds['warning']
            config.last_updated = datetime.now()
            config.update_count += 1

            # Store in history
            config.threshold_history.append({
                'timestamp': datetime.now().isoformat(),
                'thresholds': new_thresholds.copy(),
                'optimization_type': optimization_result.optimization_type,
                'improvement_score': optimization_result.improvement_score
            })

            # Update in anomaly engine
            self._update_anomaly_engine_thresholds(equipment_id, config)

            # Mark as applied
            optimization_result.applied = True

            logger.info(f"Applied optimized thresholds for {equipment_id}")
            return True

        except Exception as e:
            logger.error(f"Error applying threshold optimization for {equipment_id}: {e}")
            return False

    def get_threshold_recommendations(self, equipment_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get threshold optimization recommendations"""
        recommendations = []

        try:
            # Filter equipment based on criteria
            filtered_equipment = self._filter_equipment(equipment_filter or {})

            for equipment_id in filtered_equipment:
                config = self.threshold_configs.get(equipment_id)
                if not config:
                    continue

                # Analyze current performance
                performance = self._analyze_threshold_performance(equipment_id)

                # Generate recommendations based on performance
                if performance['false_positive_rate'] > 0.15:
                    recommendations.append({
                        'equipment_id': equipment_id,
                        'type': 'false_positive_reduction',
                        'priority': 'HIGH' if config.criticality == 'CRITICAL' else 'MEDIUM',
                        'description': f"High false positive rate ({performance['false_positive_rate']:.1%})",
                        'suggested_action': 'Increase threshold sensitivity',
                        'expected_improvement': 0.1
                    })

                if performance['accuracy_score'] < 0.85:
                    recommendations.append({
                        'equipment_id': equipment_id,
                        'type': 'accuracy_improvement',
                        'priority': 'HIGH',
                        'description': f"Low accuracy score ({performance['accuracy_score']:.1%})",
                        'suggested_action': 'Retune threshold levels',
                        'expected_improvement': 0.15
                    })

                if config.update_count == 0 and config.learning_enabled:
                    recommendations.append({
                        'equipment_id': equipment_id,
                        'type': 'initial_optimization',
                        'priority': 'MEDIUM',
                        'description': 'Thresholds never optimized',
                        'suggested_action': 'Run initial optimization',
                        'expected_improvement': 0.08
                    })

        except Exception as e:
            logger.error(f"Error generating threshold recommendations: {e}")

        return recommendations

    # Helper methods and chart creation
    def _create_threshold_distribution_chart(self) -> go.Figure:
        """Create threshold distribution chart"""
        # Prepare data for visualization
        equipment_data = []
        for equipment_id, config in self.threshold_configs.items():
            equipment_data.append({
                'equipment_id': equipment_id,
                'subsystem': config.subsystem,
                'criticality': config.criticality,
                'critical_threshold': config.critical_threshold,
                'high_threshold': config.high_threshold,
                'medium_threshold': config.medium_threshold
            })

        df = pd.DataFrame(equipment_data)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Threshold Distribution by Criticality', 'Threshold Levels by Subsystem'),
            specs=[[{"type": "violin"}, {"type": "box"}]]
        )

        # Violin plot by criticality
        for criticality in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            subset = df[df['criticality'] == criticality]
            if not subset.empty:
                fig.add_trace(
                    go.Violin(
                        y=subset['critical_threshold'],
                        name=criticality,
                        box_visible=True,
                        meanline_visible=True
                    ),
                    row=1, col=1
                )

        # Box plot by subsystem
        for subsystem in df['subsystem'].unique():
            subset = df[df['subsystem'] == subsystem]
            fig.add_trace(
                go.Box(
                    y=subset['critical_threshold'],
                    name=subsystem,
                    boxpoints='all'
                ),
                row=1, col=2
            )

        fig.update_layout(
            height=400,
            title_text="Threshold Distribution Analysis",
            showlegend=False
        )

        return fig

    def _create_performance_analytics_chart(self) -> go.Figure:
        """Create performance analytics chart"""
        # Generate sample performance data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accuracy Trends',
                'False Positive Rates',
                'Optimization Impact',
                'Threshold Stability'
            )
        )

        # Accuracy trends
        accuracy_data = np.random.uniform(0.85, 0.98, len(dates))
        fig.add_trace(
            go.Scatter(x=dates, y=accuracy_data, name='Accuracy', line=dict(color='green')),
            row=1, col=1
        )

        # False positive rates
        fp_data = np.random.uniform(0.02, 0.15, len(dates))
        fig.add_trace(
            go.Scatter(x=dates, y=fp_data, name='False Positive Rate', line=dict(color='red')),
            row=1, col=2
        )

        # Optimization impact
        optimization_impact = np.random.uniform(-0.05, 0.20, 10)
        equipment_names = [f'Eq_{i:02d}' for i in range(10)]
        fig.add_trace(
            go.Bar(x=equipment_names, y=optimization_impact, name='Improvement'),
            row=2, col=1
        )

        # Threshold stability
        stability_data = np.random.uniform(0.8, 1.0, len(dates))
        fig.add_trace(
            go.Scatter(x=dates, y=stability_data, name='Stability Index', line=dict(color='blue')),
            row=2, col=2
        )

        fig.update_layout(height=600, title_text="Threshold Performance Analytics")
        return fig

    def _get_equipment_options(self) -> List[Dict[str, str]]:
        """Get equipment options for dropdown"""
        options = []
        for equipment_id, config in self.threshold_configs.items():
            options.append({
                'label': f"{equipment_id} ({config.subsystem} - {config.criticality})",
                'value': equipment_id
            })
        return sorted(options, key=lambda x: x['label'])

    def _get_equipment_performance_data(self, equipment_id: str) -> Dict[str, Any]:
        """Get historical performance data for equipment"""
        # This would get real performance data from database
        return {
            'false_positive_rate': np.random.uniform(0.05, 0.20),
            'false_negative_rate': np.random.uniform(0.02, 0.10),
            'accuracy_score': np.random.uniform(0.80, 0.95),
            'detection_count': np.random.randint(50, 200),
            'alert_count': np.random.randint(5, 25)
        }

    def _optimize_for_accuracy(self, config: ThresholdConfiguration,
                             performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize thresholds for accuracy"""
        # Simplified optimization logic
        current_accuracy = performance_data['accuracy_score']

        # Adjust thresholds based on current performance
        adjustment = 0.05 if current_accuracy < 0.85 else -0.02

        return {
            'critical': min(0.95, config.critical_threshold + adjustment),
            'high': min(0.90, config.high_threshold + adjustment),
            'medium': min(0.85, config.medium_threshold + adjustment),
            'warning': min(0.80, config.warning_threshold + adjustment)
        }

    def _optimize_for_false_positives(self, config: ThresholdConfiguration,
                                    performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize thresholds to reduce false positives"""
        fp_rate = performance_data['false_positive_rate']

        # Increase thresholds if false positive rate is high
        adjustment = 0.08 if fp_rate > 0.15 else 0.03

        return {
            'critical': min(0.95, config.critical_threshold + adjustment),
            'high': min(0.90, config.high_threshold + adjustment),
            'medium': min(0.85, config.medium_threshold + adjustment),
            'warning': min(0.80, config.warning_threshold + adjustment)
        }

    def _optimize_for_sensitivity(self, config: ThresholdConfiguration,
                                performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Optimize thresholds for sensitivity"""
        fn_rate = performance_data['false_negative_rate']

        # Decrease thresholds if false negative rate is high
        adjustment = -0.05 if fn_rate > 0.08 else -0.02

        return {
            'critical': max(0.50, config.critical_threshold + adjustment),
            'high': max(0.45, config.high_threshold + adjustment),
            'medium': max(0.40, config.medium_threshold + adjustment),
            'warning': max(0.35, config.warning_threshold + adjustment)
        }

    def _calculate_improvement_score(self, old_thresholds: Dict[str, float],
                                   new_thresholds: Dict[str, float],
                                   performance_data: Dict[str, Any]) -> float:
        """Calculate expected improvement score"""
        # Simplified improvement calculation
        threshold_change = sum(abs(new_thresholds[k] - old_thresholds[k]) for k in old_thresholds)
        base_improvement = min(0.20, threshold_change * 0.5)

        # Adjust based on current performance
        if performance_data['accuracy_score'] < 0.85:
            base_improvement *= 1.5
        if performance_data['false_positive_rate'] > 0.15:
            base_improvement *= 1.2

        return base_improvement

    def _generate_optimization_justification(self, optimization_type: str,
                                           old_thresholds: Dict[str, float],
                                           new_thresholds: Dict[str, float],
                                           improvement_score: float) -> str:
        """Generate justification for optimization"""
        justifications = {
            'accuracy': f"Optimized for accuracy improvement. Expected gain: {improvement_score:.1%}",
            'false_positive': f"Reduced false positive alerts. Expected reduction: {improvement_score:.1%}",
            'sensitivity': f"Increased sensitivity for better detection. Expected improvement: {improvement_score:.1%}"
        }
        return justifications.get(optimization_type, "Threshold optimization completed")

    def _calculate_optimization_confidence(self, performance_data: Dict[str, Any],
                                         improvement_score: float) -> float:
        """Calculate confidence in optimization"""
        # Base confidence on data quality and expected improvement
        data_confidence = min(1.0, performance_data['detection_count'] / 100.0)
        improvement_confidence = min(1.0, improvement_score * 5.0)

        return (data_confidence + improvement_confidence) / 2.0

    def _update_anomaly_engine_thresholds(self, equipment_id: str, config: ThresholdConfiguration):
        """Update thresholds in the anomaly engine"""
        # This would update the actual anomaly engine with new thresholds
        pass

    def _filter_equipment(self, filter_criteria: Dict[str, Any]) -> List[str]:
        """Filter equipment based on criteria"""
        filtered = []
        for equipment_id, config in self.threshold_configs.items():
            if self._equipment_matches_filter(config, filter_criteria):
                filtered.append(equipment_id)
        return filtered

    def _equipment_matches_filter(self, config: ThresholdConfiguration,
                                filter_criteria: Dict[str, Any]) -> bool:
        """Check if equipment matches filter criteria"""
        for key, value in filter_criteria.items():
            if value == 'all':
                continue
            if key == 'spacecraft' and value not in config.equipment_id:
                return False
            if key == 'subsystem' and value != config.subsystem:
                return False
            if key == 'criticality' and value != config.criticality:
                return False
        return True

    def _analyze_threshold_performance(self, equipment_id: str) -> Dict[str, float]:
        """Analyze threshold performance for equipment"""
        # This would analyze actual performance data
        return {
            'false_positive_rate': np.random.uniform(0.05, 0.20),
            'false_negative_rate': np.random.uniform(0.02, 0.10),
            'accuracy_score': np.random.uniform(0.80, 0.95)
        }


# Global instance for dashboard integration
threshold_manager = EquipmentSpecificThresholdManager()