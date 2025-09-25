"""
Equipment Anomaly Heatmap Component
Real-time visual heatmap of NASA SMAP/MSL equipment health based on anomaly scores
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math

logger = logging.getLogger(__name__)


@dataclass
class EquipmentHeatmapData:
    """Equipment data for heatmap visualization"""
    equipment_id: str
    equipment_name: str
    equipment_type: str  # SMAP, MSL
    subsystem: str      # Power, Communication, etc.
    location: str       # Satellite, Mars Rover
    criticality: str    # CRITICAL, HIGH, MEDIUM, LOW

    # Current status
    anomaly_score: float
    is_anomaly: bool
    status: str         # NORMAL, WARNING, CRITICAL
    last_update: datetime

    # Grid position for visualization
    grid_row: int
    grid_col: int

    # Additional metrics
    sensor_count: int
    model_accuracy: float
    uptime_percentage: float


class AnomalyHeatmapManager:
    """
    Manages the equipment anomaly heatmap visualization
    Organizes NASA equipment by mission and subsystem for optimal layout
    """

    def __init__(self):
        """Initialize anomaly heatmap manager"""
        # Import managers with error handling
        self.unified_orchestrator = None
        self.equipment_mapper = None
        self.model_manager = None
        self._initialize_services()

        # Equipment organization for heatmap layout
        self.equipment_grid = {}
        self.subsystem_colors = {
            'Power': '#FF6B6B',         # Red-ish
            'Communication': '#4ECDC4', # Teal
            'Attitude': '#45B7D1',      # Blue
            'Thermal': '#96CEB4',       # Green
            'Payload': '#FFEAA7',       # Yellow
            'Mobility': '#DDA0DD',      # Purple
            'Environmental': '#98D8C8', # Mint
            'Science': '#F7DC6F',       # Light Yellow
            'Navigation': '#BB8FCE'     # Light Purple
        }

        # Status color mapping
        self.status_colors = {
            'NORMAL': '#2ECC71',      # Green
            'WARNING': '#F39C12',     # Orange
            'CRITICAL': '#E74C3C',    # Red
            'OFFLINE': '#95A5A6',     # Gray
            'UNKNOWN': '#BDC3C7'      # Light Gray
        }

        # Initialize equipment grid layout
        self._initialize_equipment_grid()

        logger.info("Anomaly Heatmap Manager initialized")

    def _initialize_services(self):
        """Initialize service connections with error handling"""
        try:
            from src.dashboard.unified_data_orchestrator import unified_data_orchestrator
            self.unified_orchestrator = unified_data_orchestrator
            logger.info("Connected to unified data orchestrator")
        except ImportError as e:
            logger.warning(f"Unified data orchestrator not available: {e}")

        try:
            from src.data_ingestion.equipment_mapper import equipment_mapper
            self.equipment_mapper = equipment_mapper
            logger.info("Connected to equipment mapper")
        except ImportError as e:
            logger.warning(f"Equipment mapper not available: {e}")

        try:
            from src.dashboard.model_manager import pretrained_model_manager
            self.model_manager = pretrained_model_manager
            logger.info("Connected to model manager")
        except ImportError as e:
            logger.warning(f"Model manager not available: {e}")

    def _initialize_equipment_grid(self):
        """Initialize equipment grid layout for optimal visualization"""
        try:
            if not self.equipment_mapper:
                # Create mock grid for demonstration
                self._create_mock_equipment_grid()
                return

            # Get all equipment from mapper
            all_equipment = self.equipment_mapper.get_all_equipment()

            # Organize by spacecraft and subsystem
            smap_equipment = []
            msl_equipment = []

            for equipment in all_equipment:
                if hasattr(equipment, 'equipment_type'):
                    equipment_type = equipment.equipment_type
                else:
                    equipment_type = getattr(equipment, 'spacecraft', 'UNKNOWN')

                if 'SMAP' in equipment_type.upper():
                    smap_equipment.append(equipment)
                elif 'MSL' in equipment_type.upper():
                    msl_equipment.append(equipment)

            # Create grid layout
            self._arrange_equipment_grid(smap_equipment, msl_equipment)

        except Exception as e:
            logger.error(f"Error initializing equipment grid: {e}")
            self._create_mock_equipment_grid()

    def _arrange_equipment_grid(self, smap_equipment: List, msl_equipment: List):
        """Arrange equipment in optimal grid layout"""
        grid_data = {}

        # SMAP equipment layout (top half of grid)
        smap_subsystems = self._group_by_subsystem(smap_equipment)
        row_offset = 0

        for subsystem, equipment_list in smap_subsystems.items():
            col = 0
            for equipment in equipment_list[:8]:  # Max 8 per row
                equipment_id = getattr(equipment, 'equipment_id', f'SMAP_{subsystem}_{col}')

                grid_data[equipment_id] = EquipmentHeatmapData(
                    equipment_id=equipment_id,
                    equipment_name=getattr(equipment, 'equipment_name', f'SMAP {subsystem} {col+1}'),
                    equipment_type='SMAP',
                    subsystem=subsystem,
                    location='Satellite',
                    criticality=getattr(equipment, 'criticality', 'MEDIUM'),
                    anomaly_score=0.0,
                    is_anomaly=False,
                    status='NORMAL',
                    last_update=datetime.now(),
                    grid_row=row_offset,
                    grid_col=col,
                    sensor_count=getattr(equipment, 'sensor_count', 5),
                    model_accuracy=0.95,
                    uptime_percentage=99.5
                )
                col += 1

            row_offset += 1

        # MSL equipment layout (bottom half of grid)
        msl_subsystems = self._group_by_subsystem(msl_equipment)
        row_offset += 1  # Gap between SMAP and MSL

        for subsystem, equipment_list in msl_subsystems.items():
            col = 0
            for equipment in equipment_list[:8]:  # Max 8 per row
                equipment_id = getattr(equipment, 'equipment_id', f'MSL_{subsystem}_{col}')

                grid_data[equipment_id] = EquipmentHeatmapData(
                    equipment_id=equipment_id,
                    equipment_name=getattr(equipment, 'equipment_name', f'MSL {subsystem} {col+1}'),
                    equipment_type='MSL',
                    subsystem=subsystem,
                    location='Mars Rover',
                    criticality=getattr(equipment, 'criticality', 'MEDIUM'),
                    anomaly_score=0.0,
                    is_anomaly=False,
                    status='NORMAL',
                    last_update=datetime.now(),
                    grid_row=row_offset,
                    grid_col=col,
                    sensor_count=getattr(equipment, 'sensor_count', 5),
                    model_accuracy=0.95,
                    uptime_percentage=99.5
                )
                col += 1

            row_offset += 1

        self.equipment_grid = grid_data

    def _group_by_subsystem(self, equipment_list: List) -> Dict[str, List]:
        """Group equipment by subsystem"""
        subsystems = {}

        for equipment in equipment_list:
            subsystem = getattr(equipment, 'subsystem', 'Unknown')
            if subsystem not in subsystems:
                subsystems[subsystem] = []
            subsystems[subsystem].append(equipment)

        return subsystems

    def _create_mock_equipment_grid(self):
        """Create mock equipment grid for demonstration"""
        grid_data = {}

        # SMAP equipment (25 total)
        smap_subsystems = ['Power', 'Communication', 'Attitude', 'Thermal', 'Payload']
        smap_row = 0

        for i, subsystem in enumerate(smap_subsystems):
            for j in range(5):  # 5 equipment per subsystem
                equipment_id = f'SMAP_{i*5 + j:02d}'

                grid_data[equipment_id] = EquipmentHeatmapData(
                    equipment_id=equipment_id,
                    equipment_name=f'SMAP {subsystem} {j+1}',
                    equipment_type='SMAP',
                    subsystem=subsystem,
                    location='Satellite',
                    criticality=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'][j % 4],
                    anomaly_score=np.random.uniform(0.0, 0.3),
                    is_anomaly=np.random.choice([True, False], p=[0.1, 0.9]),
                    status=np.random.choice(['NORMAL', 'WARNING', 'CRITICAL'], p=[0.8, 0.15, 0.05]),
                    last_update=datetime.now(),
                    grid_row=smap_row,
                    grid_col=j,
                    sensor_count=np.random.randint(3, 8),
                    model_accuracy=np.random.uniform(0.90, 0.98),
                    uptime_percentage=np.random.uniform(95.0, 99.9)
                )

            smap_row += 1

        # MSL equipment (55 total)
        msl_subsystems = ['Power', 'Mobility', 'Environmental', 'Science', 'Communication', 'Navigation']
        msl_row = smap_row + 1  # Gap between SMAP and MSL

        equipment_count = 0
        for i, subsystem in enumerate(msl_subsystems):
            items_in_subsystem = [8, 18, 12, 10, 6, 1][i]  # Distribution matching NASA MSL

            for j in range(items_in_subsystem):
                if equipment_count >= 55:  # Limit to 55 MSL equipment
                    break

                equipment_id = f'MSL_{equipment_count + 25:02d}'

                grid_data[equipment_id] = EquipmentHeatmapData(
                    equipment_id=equipment_id,
                    equipment_name=f'MSL {subsystem} {j+1}',
                    equipment_type='MSL',
                    subsystem=subsystem,
                    location='Mars Rover',
                    criticality=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'][j % 4],
                    anomaly_score=np.random.uniform(0.0, 0.4),
                    is_anomaly=np.random.choice([True, False], p=[0.12, 0.88]),
                    status=np.random.choice(['NORMAL', 'WARNING', 'CRITICAL'], p=[0.75, 0.2, 0.05]),
                    last_update=datetime.now(),
                    grid_row=msl_row,
                    grid_col=j % 8,  # Max 8 per row
                    sensor_count=np.random.randint(3, 10),
                    model_accuracy=np.random.uniform(0.88, 0.96),
                    uptime_percentage=np.random.uniform(92.0, 99.5)
                )

                equipment_count += 1

                if j > 0 and j % 8 == 7:  # New row every 8 items
                    msl_row += 1

            if equipment_count < 55:
                msl_row += 1

        self.equipment_grid = grid_data
        logger.info(f"Created mock equipment grid with {len(grid_data)} equipment items")

    def update_real_time_data(self):
        """Update equipment data with real-time anomaly scores"""
        try:
            if self.unified_orchestrator:
                # Get real-time data from unified orchestrator
                equipment_status = self.unified_orchestrator.get_all_equipment_status()

                for equipment_id, status_data in equipment_status.items():
                    if equipment_id in self.equipment_grid and status_data:
                        equipment = self.equipment_grid[equipment_id]

                        # Update anomaly scores from real data
                        equipment.anomaly_score = status_data.get('current_anomaly_score', 0.0)
                        equipment.is_anomaly = status_data.get('anomaly_detected', False)
                        equipment.last_update = datetime.now()

                        # Determine status based on anomaly score
                        equipment.status = self._determine_status(equipment.anomaly_score)
            else:
                # Simulate real-time updates
                self._simulate_real_time_updates()

        except Exception as e:
            logger.error(f"Error updating real-time data: {e}")
            self._simulate_real_time_updates()

    def _simulate_real_time_updates(self):
        """Simulate real-time updates for demonstration"""
        for equipment_id, equipment in self.equipment_grid.items():
            # Simulate some variance in anomaly scores
            if np.random.random() < 0.1:  # 10% chance of update
                # Add some realistic patterns
                if equipment.equipment_type == 'SMAP':
                    # SMAP has generally lower anomaly rates
                    equipment.anomaly_score = max(0.0, min(1.0,
                        equipment.anomaly_score + np.random.normal(0, 0.02)))
                else:  # MSL
                    # MSL on Mars has slightly higher anomaly rates
                    equipment.anomaly_score = max(0.0, min(1.0,
                        equipment.anomaly_score + np.random.normal(0, 0.025)))

                equipment.is_anomaly = equipment.anomaly_score > 0.15
                equipment.status = self._determine_status(equipment.anomaly_score)
                equipment.last_update = datetime.now()

    def _determine_status(self, anomaly_score: float) -> str:
        """Determine equipment status based on anomaly score"""
        if anomaly_score >= 0.3:
            return 'CRITICAL'
        elif anomaly_score >= 0.15:
            return 'WARNING'
        else:
            return 'NORMAL'

    def get_heatmap_data(self) -> Dict[str, Any]:
        """Get current heatmap data for visualization

        Returns:
            Heatmap data dictionary
        """
        self.update_real_time_data()

        # Organize data for heatmap
        grid_rows = max([eq.grid_row for eq in self.equipment_grid.values()]) + 1
        grid_cols = max([eq.grid_col for eq in self.equipment_grid.values()]) + 1

        # Create matrices for heatmap
        anomaly_matrix = np.zeros((grid_rows, grid_cols))
        text_matrix = [[''] * grid_cols for _ in range(grid_rows)]
        hover_matrix = [[''] * grid_cols for _ in range(grid_rows)]
        color_matrix = [[''] * grid_cols for _ in range(grid_rows)]

        equipment_info = {}

        for equipment_id, equipment in self.equipment_grid.items():
            row, col = equipment.grid_row, equipment.grid_col

            if row < grid_rows and col < grid_cols:
                anomaly_matrix[row][col] = equipment.anomaly_score
                text_matrix[row][col] = equipment.equipment_id
                color_matrix[row][col] = self.status_colors[equipment.status]

                hover_matrix[row][col] = (
                    f"{equipment.equipment_name}<br>"
                    f"Type: {equipment.equipment_type}<br>"
                    f"Subsystem: {equipment.subsystem}<br>"
                    f"Status: {equipment.status}<br>"
                    f"Anomaly Score: {equipment.anomaly_score:.3f}<br>"
                    f"Sensors: {equipment.sensor_count}<br>"
                    f"Uptime: {equipment.uptime_percentage:.1f}%"
                )

                equipment_info[equipment_id] = equipment

        return {
            'anomaly_matrix': anomaly_matrix,
            'text_matrix': text_matrix,
            'hover_matrix': hover_matrix,
            'color_matrix': color_matrix,
            'equipment_info': equipment_info,
            'grid_dimensions': (grid_rows, grid_cols)
        }

    def create_heatmap_figure(self) -> go.Figure:
        """Create Plotly heatmap figure

        Returns:
            Plotly Figure object
        """
        heatmap_data = self.get_heatmap_data()

        anomaly_matrix = heatmap_data['anomaly_matrix']
        text_matrix = heatmap_data['text_matrix']
        hover_matrix = heatmap_data['hover_matrix']

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=anomaly_matrix,
            text=text_matrix,
            hovertext=hover_matrix,
            hovertemplate='%{hovertext}<extra></extra>',
            texttemplate='%{text}',
            textfont={"size": 8},
            colorscale=[
                [0.0, '#2ECC71'],   # Green (Normal)
                [0.15, '#F39C12'],  # Orange (Warning)
                [0.3, '#E74C3C'],   # Red (Critical)
                [1.0, '#8B0000']    # Dark Red (Severe)
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Anomaly Score",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.2
            )
        ))

        # Add annotations for subsystem labels
        self._add_subsystem_annotations(fig, heatmap_data)

        # Update layout
        fig.update_layout(
            title={
                'text': "🛰️ NASA Equipment Anomaly Heatmap - Real-time Status",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title="Equipment Units →",
            yaxis_title="↑ Subsystems by Mission",
            font=dict(size=10),
            height=600,
            margin=dict(l=100, r=100, t=80, b=60)
        )

        # Reverse y-axis to show SMAP at top
        fig.update_yaxes(autorange="reversed")

        return fig

    def _add_subsystem_annotations(self, fig: go.Figure, heatmap_data: Dict):
        """Add subsystem labels to heatmap"""
        equipment_info = heatmap_data['equipment_info']

        # Group by subsystem for labeling
        subsystem_positions = {}

        for equipment in equipment_info.values():
            subsystem_key = f"{equipment.equipment_type}_{equipment.subsystem}"
            if subsystem_key not in subsystem_positions:
                subsystem_positions[subsystem_key] = {
                    'rows': [],
                    'cols': [],
                    'name': f"{equipment.equipment_type} {equipment.subsystem}",
                    'type': equipment.equipment_type
                }

            subsystem_positions[subsystem_key]['rows'].append(equipment.grid_row)
            subsystem_positions[subsystem_key]['cols'].append(equipment.grid_col)

        # Add annotations for subsystem centers
        for subsystem_data in subsystem_positions.values():
            if subsystem_data['rows']:
                center_row = np.mean(subsystem_data['rows'])
                min_col = min(subsystem_data['cols']) - 0.8

                fig.add_annotation(
                    x=min_col,
                    y=center_row,
                    text=subsystem_data['name'],
                    showarrow=False,
                    font=dict(size=9, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    xanchor='right'
                )

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard display

        Returns:
            Summary statistics
        """
        self.update_real_time_data()

        total_equipment = len(self.equipment_grid)
        smap_count = sum(1 for eq in self.equipment_grid.values() if eq.equipment_type == 'SMAP')
        msl_count = sum(1 for eq in self.equipment_grid.values() if eq.equipment_type == 'MSL')

        # Status counts
        status_counts = {'NORMAL': 0, 'WARNING': 0, 'CRITICAL': 0, 'OFFLINE': 0}
        anomaly_count = 0
        total_sensors = 0
        avg_uptime = 0

        for equipment in self.equipment_grid.values():
            status_counts[equipment.status] = status_counts.get(equipment.status, 0) + 1
            if equipment.is_anomaly:
                anomaly_count += 1
            total_sensors += equipment.sensor_count
            avg_uptime += equipment.uptime_percentage

        avg_uptime = avg_uptime / total_equipment if total_equipment > 0 else 0

        return {
            'total_equipment': total_equipment,
            'smap_equipment': smap_count,
            'msl_equipment': msl_count,
            'normal_count': status_counts['NORMAL'],
            'warning_count': status_counts['WARNING'],
            'critical_count': status_counts['CRITICAL'],
            'offline_count': status_counts.get('OFFLINE', 0),
            'anomaly_count': anomaly_count,
            'anomaly_rate': (anomaly_count / total_equipment * 100) if total_equipment > 0 else 0,
            'total_sensors': total_sensors,
            'average_uptime': avg_uptime,
            'last_update': max([eq.last_update for eq in self.equipment_grid.values()]) if self.equipment_grid else datetime.now()
        }


# Global instance for dashboard integration
anomaly_heatmap_manager = AnomalyHeatmapManager()