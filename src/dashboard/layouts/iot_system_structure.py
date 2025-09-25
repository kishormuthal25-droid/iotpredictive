"""
IoT System Structure Visualization
Interactive dashboard page showing the complete NASA IoT system architecture
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

# from ..unified_data_orchestrator import unified_data_orchestrator  # Not needed for static structure view

logger = logging.getLogger(__name__)


class IoTSystemStructure:
    """Interactive visualization of IoT system structure"""

    def __init__(self):
        self.system_data = self._get_system_structure_data()
        logger.info("Initialized IoT System Structure visualization")

    def _get_system_structure_data(self) -> Dict[str, Any]:
        """Get structured data about the IoT systems"""
        return {
            "smap_system": {
                "name": "🛰️ SMAP Satellite",
                "location": "Earth Orbit",
                "total_sensors": 25,
                "subsystems": {
                    "POWER": {"equipment": 1, "sensors": 6, "criticality": "CRITICAL"},
                    "COMMUNICATION": {"equipment": 1, "sensors": 5, "criticality": "HIGH"},
                    "ATTITUDE": {"equipment": 1, "sensors": 6, "criticality": "CRITICAL"},
                    "THERMAL": {"equipment": 1, "sensors": 4, "criticality": "HIGH"},
                    "PAYLOAD": {"equipment": 1, "sensors": 4, "criticality": "HIGH"}
                }
            },
            "msl_system": {
                "name": "🤖 MSL Mars Rover",
                "location": "Mars Surface",
                "total_sensors": 55,
                "subsystems": {
                    "MOBILITY": {"equipment": 2, "sensors": 18, "criticality": "CRITICAL"},
                    "POWER": {"equipment": 1, "sensors": 8, "criticality": "CRITICAL"},
                    "ENVIRONMENTAL": {"equipment": 1, "sensors": 12, "criticality": "MEDIUM"},
                    "SCIENTIFIC": {"equipment": 1, "sensors": 10, "criticality": "HIGH"},
                    "COMMUNICATION": {"equipment": 1, "sensors": 6, "criticality": "HIGH"},
                    "NAVIGATION": {"equipment": 1, "sensors": 1, "criticality": "CRITICAL"}
                }
            }
        }

    def create_system_overview_chart(self) -> go.Figure:
        """Create a high-level system overview chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['System Distribution', 'Sensor Distribution', 'Criticality Levels', 'Subsystem Coverage'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )

        # System distribution pie chart
        systems = ['SMAP Satellite', 'MSL Mars Rover']
        sensors = [25, 55]
        colors = ['#1f77b4', '#ff7f0e']

        fig.add_trace(
            go.Pie(
                labels=systems,
                values=sensors,
                marker=dict(colors=colors),
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Sensors: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=1
        )

        # Sensor distribution by subsystem
        subsystem_data = []
        for system_name, system_info in self.system_data.items():
            for subsystem, details in system_info["subsystems"].items():
                subsystem_data.append({
                    'Subsystem': subsystem,
                    'Sensors': details['sensors'],
                    'System': system_info['name']
                })

        df_subsystems = pd.DataFrame(subsystem_data)

        for system in df_subsystems['System'].unique():
            system_data = df_subsystems[df_subsystems['System'] == system]
            fig.add_trace(
                go.Bar(
                    x=system_data['Subsystem'],
                    y=system_data['Sensors'],
                    name=system,
                    text=system_data['Sensors'],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>System: ' + system + '<br>Sensors: %{y}<extra></extra>'
                ),
                row=1, col=2
            )

        # Criticality levels pie chart
        criticality_count = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        for system_info in self.system_data.values():
            for subsystem_info in system_info["subsystems"].values():
                criticality_count[subsystem_info["criticality"]] += 1

        criticality_colors = {'CRITICAL': '#d62728', 'HIGH': '#ff7f0e', 'MEDIUM': '#2ca02c'}

        fig.add_trace(
            go.Pie(
                labels=list(criticality_count.keys()),
                values=list(criticality_count.values()),
                marker=dict(colors=[criticality_colors[level] for level in criticality_count.keys()]),
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Subsystems: %{value}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=2, col=1
        )

        # Equipment distribution
        equipment_data = []
        for system_name, system_info in self.system_data.items():
            for subsystem, details in system_info["subsystems"].items():
                equipment_data.append({
                    'Subsystem': subsystem,
                    'Equipment': details['equipment'],
                    'System': system_info['name']
                })

        df_equipment = pd.DataFrame(equipment_data)

        for system in df_equipment['System'].unique():
            system_data = df_equipment[df_equipment['System'] == system]
            fig.add_trace(
                go.Bar(
                    x=system_data['Subsystem'],
                    y=system_data['Equipment'],
                    name=f"{system} Equipment",
                    text=system_data['Equipment'],
                    textposition='auto',
                    showlegend=False,
                    hovertemplate='<b>%{x}</b><br>System: ' + system + '<br>Equipment: %{y}<extra></extra>'
                ),
                row=2, col=2
            )

        fig.update_layout(
            title="NASA IoT Predictive Maintenance System Overview",
            height=800,
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def create_system_hierarchy_chart(self) -> go.Figure:
        """Create an interactive system hierarchy tree"""
        fig = go.Figure()

        # Create hierarchical structure for treemap
        labels = ["NASA IoT Systems"]
        parents = [""]
        values = [80]  # Total sensors
        colors = [0.5]

        # Add main systems
        labels.extend(["🛰️ SMAP Satellite", "🤖 MSL Mars Rover"])
        parents.extend(["NASA IoT Systems", "NASA IoT Systems"])
        values.extend([25, 55])
        colors.extend([0.8, 0.2])

        # Add SMAP subsystems
        for subsystem, details in self.system_data["smap_system"]["subsystems"].items():
            labels.append(f"SMAP-{subsystem}")
            parents.append("🛰️ SMAP Satellite")
            values.append(details["sensors"])
            colors.append(0.9 if details["criticality"] == "CRITICAL" else 0.7 if details["criticality"] == "HIGH" else 0.5)

        # Add MSL subsystems
        for subsystem, details in self.system_data["msl_system"]["subsystems"].items():
            labels.append(f"MSL-{subsystem}")
            parents.append("🤖 MSL Mars Rover")
            values.append(details["sensors"])
            colors.append(0.1 if details["criticality"] == "CRITICAL" else 0.3 if details["criticality"] == "HIGH" else 0.5)

        fig.add_trace(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Sensors: %{value}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=3,
            marker=dict(
                colorscale='RdYlBu_r',
                colorbar=dict(
                    title="Criticality Level",
                    tickvals=[0.1, 0.5, 0.9],
                    ticktext=["Critical", "Medium", "High"]
                ),
                line=dict(width=2)
            )
        ))

        fig.update_layout(
            title="Interactive IoT System Hierarchy - Sensor Distribution",
            height=600,
            template="plotly_white"
        )

        return fig

    def create_network_diagram(self) -> go.Figure:
        """Create a network diagram showing system relationships"""
        fig = go.Figure()

        # Define node positions for a clean layout
        positions = {
            "NASA IoT": (0, 0),
            "SMAP Satellite": (-2, 1),
            "MSL Mars Rover": (2, 1),
            # SMAP subsystems
            "SMAP-POWER": (-3, 2),
            "SMAP-COMMUNICATION": (-2.5, 2.5),
            "SMAP-ATTITUDE": (-2, 2.5),
            "SMAP-THERMAL": (-1.5, 2.5),
            "SMAP-PAYLOAD": (-1, 2),
            # MSL subsystems
            "MSL-MOBILITY": (3, 2),
            "MSL-POWER": (2.5, 2.5),
            "MSL-ENVIRONMENTAL": (2, 2.5),
            "MSL-SCIENTIFIC": (1.5, 2.5),
            "MSL-COMMUNICATION": (1, 2),
            "MSL-NAVIGATION": (2, 3)
        }

        # Add edges (connections)
        edge_x = []
        edge_y = []

        # Connect root to main systems
        for system in ["SMAP Satellite", "MSL Mars Rover"]:
            x0, y0 = positions["NASA IoT"]
            x1, y1 = positions[system]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Connect systems to subsystems
        connections = {
            "SMAP Satellite": ["SMAP-POWER", "SMAP-COMMUNICATION", "SMAP-ATTITUDE", "SMAP-THERMAL", "SMAP-PAYLOAD"],
            "MSL Mars Rover": ["MSL-MOBILITY", "MSL-POWER", "MSL-ENVIRONMENTAL", "MSL-SCIENTIFIC", "MSL-COMMUNICATION", "MSL-NAVIGATION"]
        }

        for parent, children in connections.items():
            for child in children:
                x0, y0 = positions[parent]
                x1, y1 = positions[child]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        # Add edges to plot
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightblue'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))

        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []

        for node, (x, y) in positions.items():
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            if node == "NASA IoT":
                node_colors.append('gold')
                node_sizes.append(40)
            elif "Satellite" in node or "Rover" in node:
                node_colors.append('lightcoral')
                node_sizes.append(30)
            else:
                # Color by criticality
                if "POWER" in node or "ATTITUDE" in node or "MOBILITY" in node or "NAVIGATION" in node:
                    node_colors.append('red')  # Critical
                elif "COMMUNICATION" in node or "SCIENTIFIC" in node or "PAYLOAD" in node or "THERMAL" in node:
                    node_colors.append('orange')  # High
                else:
                    node_colors.append('green')  # Medium
                node_sizes.append(20)

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))

        fig.update_layout(
            title="NASA IoT System Network Architecture",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="🔴 Critical | 🟠 High | 🟢 Medium Priority",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            height=600
        )

        return fig

    def get_layout(self) -> html.Div:
        """Get the complete layout for the IoT system structure page"""
        return html.Div([
            # Header
            html.Div([
                html.H2("🏗️ IoT System Architecture", className="text-primary mb-4"),
                html.P([
                    "Interactive visualization of NASA's predictive maintenance IoT systems. ",
                    "This page shows the complete structure of our SMAP satellite and MSL Mars rover systems ",
                    "with their subsystems, equipment components, and 80 individual sensors."
                ], className="lead mb-4")
            ], className="mb-4"),

            # System Statistics Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H4("🛰️ SMAP Satellite", className="card-title"),
                        html.H2("25", className="text-primary"),
                        html.P("Sensors", className="card-text"),
                        html.Small("5 Subsystems | Earth Orbit", className="text-muted")
                    ], className="card-body text-center")
                ], className="card"),

                html.Div([
                    html.Div([
                        html.H4("🤖 MSL Mars Rover", className="card-title"),
                        html.H2("55", className="text-primary"),
                        html.P("Sensors", className="card-text"),
                        html.Small("6 Subsystems | Mars Surface", className="text-muted")
                    ], className="card-body text-center")
                ], className="card"),

                html.Div([
                    html.Div([
                        html.H4("[SYSTEM] Total System", className="card-title"),
                        html.H2("80", className="text-success"),
                        html.P("Total Sensors", className="card-text"),
                        html.Small("12 Equipment Components", className="text-muted")
                    ], className="card-body text-center")
                ], className="card"),

                html.Div([
                    html.Div([
                        html.H4("🔍 Monitoring", className="card-title"),
                        html.H2("Real-time", className="text-info"),
                        html.P("Anomaly Detection", className="card-text"),
                        html.Small("Pre-trained AI Models", className="text-muted")
                    ], className="card-body text-center")
                ], className="card")
            ], className="row g-3 mb-4", style={"display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(250px, 1fr))"}),

            # Visualization Selection
            html.Div([
                html.H5("📈 Visualization Type:", className="mb-3"),
                dcc.RadioItems(
                    id="viz-type-selector",
                    options=[
                        {"label": " [SYSTEM] System Overview Dashboard", "value": "overview"},
                        {"label": " 🌳 Hierarchical Structure Tree", "value": "hierarchy"},
                        {"label": " 🔗 Network Architecture Diagram", "value": "network"}
                    ],
                    value="overview",
                    className="mb-4",
                    labelStyle={"display": "block", "margin": "10px 0"}
                )
            ], className="mb-4"),

            # Main Visualization
            html.Div([
                dcc.Graph(
                    id="system-structure-chart",
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                    }
                )
            ], className="mb-4"),

            # Detailed System Information
            html.Div([
                html.H5("[DETAIL] Detailed System Breakdown:", className="mb-3"),

                # SMAP System Details
                html.Div([
                    html.H6("🛰️ SMAP Satellite System", className="text-primary mb-3"),
                    html.Div(id="smap-details", className="mb-4")
                ]),

                # MSL System Details
                html.Div([
                    html.H6("🤖 MSL Mars Rover System", className="text-warning mb-3"),
                    html.Div(id="msl-details", className="mb-4")
                ])
            ], className="mb-4"),

            # Footer Information
            html.Div([
                html.Hr(),
                html.P([
                    "💡 ", html.Strong("Note:"), " This IoT system uses real NASA SMAP and MSL telemetry data ",
                    "for predictive maintenance with AI-powered anomaly detection across all 80 sensors. ",
                    "The system provides real-time monitoring and maintenance scheduling for space mission equipment."
                ], className="text-muted text-center")
            ])
        ], className="container-fluid")

    def create_detailed_breakdown(self, system_key: str) -> html.Div:
        """Create detailed breakdown for a specific system"""
        system_info = self.system_data[system_key]

        cards = []
        for subsystem, details in system_info["subsystems"].items():
            criticality_color = {
                "CRITICAL": "danger",
                "HIGH": "warning",
                "MEDIUM": "success"
            }.get(details["criticality"], "secondary")

            cards.append(
                html.Div([
                    html.Div([
                        html.H6(f"{subsystem}", className="card-title"),
                        html.P([
                            html.Strong("Equipment: "), f"{details['equipment']} units", html.Br(),
                            html.Strong("Sensors: "), f"{details['sensors']} sensors", html.Br(),
                            html.Span(f"{details['criticality']}", className=f"badge bg-{criticality_color}")
                        ], className="card-text")
                    ], className="card-body")
                ], className="card h-100")
            )

        return html.Div(cards, className="row g-3", style={"display": "grid", "grid-template-columns": "repeat(auto-fit, minmax(200px, 1fr))"})


# Initialize the layout
iot_structure = IoTSystemStructure()

def create_layout():
    """Create the IoT system structure layout for dashboard integration"""
    return iot_structure.get_layout()


# Callbacks
@callback(
    Output('system-structure-chart', 'figure'),
    Input('viz-type-selector', 'value')
)
def update_visualization(viz_type):
    """Update the main visualization based on selection"""
    try:
        if viz_type == "hierarchy":
            return iot_structure.create_system_hierarchy_chart()
        elif viz_type == "network":
            return iot_structure.create_network_diagram()
        else:  # overview
            return iot_structure.create_system_overview_chart()
    except Exception as e:
        logger.error(f"Error updating visualization: {e}")
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error loading visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


@callback(
    [Output('smap-details', 'children'),
     Output('msl-details', 'children')],
    Input('viz-type-selector', 'value')  # Trigger on any change
)
def update_system_details(_):
    """Update the detailed system breakdown"""
    try:
        smap_details = iot_structure.create_detailed_breakdown("smap_system")
        msl_details = iot_structure.create_detailed_breakdown("msl_system")
        return smap_details, msl_details
    except Exception as e:
        logger.error(f"Error updating system details: {e}")
        error_div = html.Div(f"Error loading details: {str(e)}", className="alert alert-warning")
        return error_div, error_div