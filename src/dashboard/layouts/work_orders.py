"""
Work Orders Management Dashboard Layout
Provides interface for viewing, updating, and tracking maintenance work orders
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple

# Import project modules - with fallback error handling
try:
    from src.utils.logger import get_logger
    from src.maintenance.work_order_manager import WorkOrderManager
    from src.maintenance.priority_calculator import PriorityCalculator
    from src.data_ingestion.database_manager import DatabaseManager
    from config.settings import Settings

    # Initialize components
    logger = get_logger(__name__)
    settings = Settings()
    work_order_manager = WorkOrderManager()
    priority_calculator = PriorityCalculator()
    db_manager = DatabaseManager()

except ImportError as e:
    # Fallback for dashboard-only mode
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Some modules not available in dashboard-only mode: {e}")

    # Create mock objects
    settings = None
    work_order_manager = None
    priority_calculator = None
    db_manager = None

# Define color schemes for priority levels
PRIORITY_COLORS = {
    'CRITICAL': '#dc3545',  # Red
    'HIGH': '#fd7e14',      # Orange
    'MEDIUM': '#ffc107',    # Yellow
    'LOW': '#28a745',       # Green
    'SCHEDULED': '#17a2b8'  # Cyan
}

STATUS_COLORS = {
    'PENDING': '#6c757d',      # Gray
    'ASSIGNED': '#17a2b8',     # Cyan
    'IN_PROGRESS': '#ffc107',  # Yellow
    'ON_HOLD': '#fd7e14',      # Orange
    'COMPLETED': '#28a745',    # Green
    'CANCELLED': '#dc3545'     # Red
}

def create_layout():
    """Create the work orders dashboard layout"""
    return html.Div([
        # Header
        html.Div([
            html.H2('Work Order Management', className='dashboard-header'),
            html.P('Monitor and manage maintenance work orders in real-time',
                   className='dashboard-subtitle')
        ], className='header-section'),
        
        # Top Stats Cards
        html.Div([
            html.Div([
                create_stat_card('total-orders', 'Total Work Orders', '0', 'primary'),
                create_stat_card('pending-orders', 'Pending', '0', 'warning'),
                create_stat_card('in-progress-orders', 'In Progress', '0', 'info'),
                create_stat_card('completed-orders', 'Completed Today', '0', 'success'),
            ], className='stat-cards-row')
        ], className='stats-section'),
        
        # Filters and Controls
        html.Div([
            html.Div([
                # Date Range Filter
                html.Div([
                    html.Label('Date Range:', className='filter-label'),
                    dcc.DatePickerRange(
                        id='work-order-date-range',
                        start_date=(datetime.now() - timedelta(days=7)).date(),
                        end_date=datetime.now().date(),
                        display_format='YYYY-MM-DD',
                        className='date-picker'
                    )
                ], className='filter-group'),
                
                # Status Filter
                html.Div([
                    html.Label('Status:', className='filter-label'),
                    dcc.Dropdown(
                        id='work-order-status-filter',
                        options=[
                            {'label': 'All', 'value': 'ALL'},
                            {'label': 'Pending', 'value': 'PENDING'},
                            {'label': 'Assigned', 'value': 'ASSIGNED'},
                            {'label': 'In Progress', 'value': 'IN_PROGRESS'},
                            {'label': 'On Hold', 'value': 'ON_HOLD'},
                            {'label': 'Completed', 'value': 'COMPLETED'}
                        ],
                        value='ALL',
                        className='filter-dropdown'
                    )
                ], className='filter-group'),
                
                # Priority Filter
                html.Div([
                    html.Label('Priority:', className='filter-label'),
                    dcc.Dropdown(
                        id='work-order-priority-filter',
                        options=[
                            {'label': 'All', 'value': 'ALL'},
                            {'label': 'Critical', 'value': 'CRITICAL'},
                            {'label': 'High', 'value': 'HIGH'},
                            {'label': 'Medium', 'value': 'MEDIUM'},
                            {'label': 'Low', 'value': 'LOW'}
                        ],
                        value='ALL',
                        className='filter-dropdown'
                    )
                ], className='filter-group'),
                
                # Technician Filter
                html.Div([
                    html.Label('Technician:', className='filter-label'),
                    dcc.Dropdown(
                        id='work-order-technician-filter',
                        options=[],  # Will be populated dynamically
                        value='ALL',
                        className='filter-dropdown'
                    )
                ], className='filter-group'),
                
                # Refresh Button
                html.Button('Refresh', id='refresh-work-orders-btn', 
                           className='btn btn-primary')
            ], className='filters-container')
        ], className='filters-section'),
        
        # Main Content Area - Two Column Layout
        html.Div([
            # Left Column - Work Orders Table
            html.Div([
                html.H4('Active Work Orders', className='section-title'),
                html.Div([
                    dash_table.DataTable(
                        id='work-orders-table',
                        columns=[
                            {'name': 'ID', 'id': 'work_order_id'},
                            {'name': 'Equipment', 'id': 'equipment_id'},
                            {'name': 'Priority', 'id': 'priority'},
                            {'name': 'Status', 'id': 'status'},
                            {'name': 'Technician', 'id': 'assigned_technician'},
                            {'name': 'Created', 'id': 'created_at'},
                            {'name': 'Due Date', 'id': 'due_date'},
                            {'name': 'Actions', 'id': 'actions', 'presentation': 'markdown'}
                        ],
                        data=[],
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'priority', 'filter_query': '{priority} = CRITICAL'},
                                'backgroundColor': '#ffebee',
                                'color': '#c62828'
                            },
                            {
                                'if': {'column_id': 'priority', 'filter_query': '{priority} = HIGH'},
                                'backgroundColor': '#fff3e0',
                                'color': '#e65100'
                            },
                            {
                                'if': {'column_id': 'status', 'filter_query': '{status} = IN_PROGRESS'},
                                'backgroundColor': '#e3f2fd',
                                'color': '#1565c0'
                            }
                        ],
                        style_header={
                            'backgroundColor': '#f8f9fa',
                            'fontWeight': 'bold'
                        },
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                        row_selectable='single',
                        selected_rows=[]
                    )
                ], className='table-container')
            ], className='left-column'),
            
            # Right Column - Details and Charts
            html.Div([
                # Work Order Details Panel
                html.Div([
                    html.H4('Work Order Details', className='section-title'),
                    html.Div(id='work-order-details', className='details-panel')
                ], className='details-section'),
                
                # Timeline Chart
                html.Div([
                    html.H4('Work Order Timeline', className='section-title'),
                    dcc.Graph(id='work-order-timeline', className='timeline-chart')
                ], className='timeline-section')
            ], className='right-column')
        ], className='main-content-row'),
        
        # Bottom Section - Analytics
        html.Div([
            # Work Order Distribution Charts
            html.Div([
                html.Div([
                    html.H4('Work Orders by Status', className='section-title'),
                    dcc.Graph(id='work-orders-by-status-chart')
                ], className='chart-container'),
                
                html.Div([
                    html.H4('Work Orders by Priority', className='section-title'),
                    dcc.Graph(id='work-orders-by-priority-chart')
                ], className='chart-container'),
                
                html.Div([
                    html.H4('Technician Workload', className='section-title'),
                    dcc.Graph(id='technician-workload-chart')
                ], className='chart-container')
            ], className='analytics-row')
        ], className='analytics-section'),
        
        # Hidden components for state management
        dcc.Store(id='work-orders-data-store'),
        dcc.Store(id='selected-work-order-store'),
        dcc.Interval(id='work-orders-interval', interval=30000),  # Update every 30 seconds
        
        # Modal for Work Order Updates
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Update Work Order', className='modal-title'),
                    html.Button('×', id='close-modal-btn', className='close-button')
                ], className='modal-header'),
                
                html.Div([
                    html.Div(id='modal-work-order-content')
                ], className='modal-body'),
                
                html.Div([
                    html.Button('Save Changes', id='save-work-order-btn', 
                               className='btn btn-primary'),
                    html.Button('Cancel', id='cancel-work-order-btn', 
                               className='btn btn-secondary')
                ], className='modal-footer')
            ], className='modal-content')
        ], id='work-order-modal', className='modal', style={'display': 'none'})
    ], className='work-orders-dashboard')

def create_stat_card(card_id: str, title: str, value: str, card_type: str) -> html.Div:
    """Create a statistics card component"""
    return html.Div([
        html.Div([
            html.H6(title, className='stat-title'),
            html.H3(value, id=card_id, className='stat-value'),
        ], className=f'stat-card stat-card-{card_type}')
    ], className='stat-card-wrapper')

def fetch_work_orders(start_date: str, end_date: str, status: str, 
                     priority: str, technician: str) -> pd.DataFrame:
    """Fetch work orders from database based on filters"""
    try:
        # Build query based on filters
        query = """
            SELECT 
                wo.work_order_id,
                wo.equipment_id,
                wo.priority,
                wo.status,
                wo.assigned_technician,
                wo.created_at,
                wo.due_date,
                wo.estimated_duration,
                wo.actual_duration,
                wo.description,
                wo.anomaly_id,
                e.equipment_name,
                e.location,
                e.criticality
            FROM work_orders wo
            LEFT JOIN equipment e ON wo.equipment_id = e.equipment_id
            WHERE wo.created_at BETWEEN %s AND %s
        """
        
        params = [start_date, end_date]
        
        if status != 'ALL':
            query += " AND wo.status = %s"
            params.append(status)
            
        if priority != 'ALL':
            query += " AND wo.priority = %s"
            params.append(priority)
            
        if technician != 'ALL':
            query += " AND wo.assigned_technician = %s"
            params.append(technician)
            
        query += " ORDER BY wo.created_at DESC"
        
        # Execute query
        df = db_manager.execute_query(query, params)
        
        # Add action buttons
        df['actions'] = df.apply(
            lambda row: f"[View](#{row['work_order_id']}) | [Update](#{row['work_order_id']})",
            axis=1
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching work orders: {str(e)}")
        return pd.DataFrame()

def generate_timeline_chart(work_orders_df: pd.DataFrame) -> go.Figure:
    """Generate Gantt chart for work order timeline"""
    if work_orders_df.empty:
        return go.Figure()
    
    # Prepare data for Gantt chart
    timeline_data = []
    for _, row in work_orders_df.iterrows():
        if pd.notna(row['due_date']):
            timeline_data.append({
                'Task': f"WO-{row['work_order_id']}",
                'Start': row['created_at'],
                'Finish': row['due_date'],
                'Resource': row['assigned_technician'] or 'Unassigned',
                'Priority': row['priority'],
                'Status': row['status']
            })
    
    if not timeline_data:
        return go.Figure()
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create Gantt chart
    fig = px.timeline(
        df_timeline,
        x_start='Start',
        x_end='Finish',
        y='Task',
        color='Priority',
        color_discrete_map=PRIORITY_COLORS,
        hover_data=['Resource', 'Status'],
        title='Work Order Schedule'
    )
    
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(
        height=400,
        showlegend=True,
        xaxis_title='Timeline',
        yaxis_title='Work Orders',
        hovermode='closest'
    )
    
    return fig

def generate_status_chart(work_orders_df: pd.DataFrame) -> go.Figure:
    """Generate pie chart for work order status distribution"""
    if work_orders_df.empty:
        return go.Figure()
    
    status_counts = work_orders_df['status'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker=dict(colors=[STATUS_COLORS.get(s, '#cccccc') for s in status_counts.index])
        )
    ])
    
    fig.update_layout(
        height=300,
        showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    return fig

def generate_priority_chart(work_orders_df: pd.DataFrame) -> go.Figure:
    """Generate bar chart for work order priority distribution"""
    if work_orders_df.empty:
        return go.Figure()
    
    priority_counts = work_orders_df['priority'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=priority_counts.index,
            y=priority_counts.values,
            marker_color=[PRIORITY_COLORS.get(p, '#cccccc') for p in priority_counts.index],
            text=priority_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        height=300,
        xaxis_title='Priority Level',
        yaxis_title='Number of Work Orders',
        showlegend=False,
        margin=dict(t=20, b=40, l=40, r=20)
    )
    
    return fig

def generate_workload_chart(work_orders_df: pd.DataFrame) -> go.Figure:
    """Generate bar chart for technician workload"""
    if work_orders_df.empty:
        return go.Figure()
    
    # Filter only active work orders
    active_df = work_orders_df[work_orders_df['status'].isin(['ASSIGNED', 'IN_PROGRESS'])]
    
    if active_df.empty:
        return go.Figure()
    
    workload = active_df.groupby('assigned_technician').agg({
        'work_order_id': 'count',
        'estimated_duration': 'sum'
    }).reset_index()
    
    workload.columns = ['Technician', 'Work Orders', 'Total Hours']
    workload['Total Hours'] = workload['Total Hours'].fillna(0) / 60  # Convert to hours
    
    fig = go.Figure()
    
    # Add work order count bars
    fig.add_trace(go.Bar(
        x=workload['Technician'],
        y=workload['Work Orders'],
        name='Work Orders',
        marker_color='#17a2b8',
        yaxis='y',
        offsetgroup=1
    ))
    
    # Add total hours bars
    fig.add_trace(go.Bar(
        x=workload['Technician'],
        y=workload['Total Hours'],
        name='Total Hours',
        marker_color='#28a745',
        yaxis='y2',
        offsetgroup=2
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title='Technician',
        yaxis=dict(title='Work Orders', side='left'),
        yaxis2=dict(title='Total Hours', overlaying='y', side='right'),
        barmode='group',
        showlegend=True,
        margin=dict(t=20, b=40, l=40, r=40)
    )
    
    return fig

def format_work_order_details(work_order: pd.Series) -> html.Div:
    """Format work order details for display"""
    if work_order is None or work_order.empty:
        return html.Div('Select a work order to view details', className='no-selection')
    
    return html.Div([
        html.Div([
            html.H5(f"Work Order #{work_order['work_order_id']}", className='detail-title'),
            html.Span(work_order['status'], 
                     className=f"status-badge status-{work_order['status'].lower()}")
        ], className='detail-header'),
        
        html.Div([
            # Equipment Information
            html.Div([
                html.H6('Equipment Information', className='detail-section-title'),
                html.P([
                    html.Strong('Equipment: '),
                    html.Span(work_order.get('equipment_name', work_order['equipment_id']))
                ]),
                html.P([
                    html.Strong('Location: '),
                    html.Span(work_order.get('location', 'N/A'))
                ]),
                html.P([
                    html.Strong('Criticality: '),
                    html.Span(work_order.get('criticality', 'N/A'))
                ])
            ], className='detail-section'),
            
            # Work Order Details
            html.Div([
                html.H6('Work Order Details', className='detail-section-title'),
                html.P([
                    html.Strong('Priority: '),
                    html.Span(work_order['priority'],
                             className=f"priority-badge priority-{work_order['priority'].lower()}")
                ]),
                html.P([
                    html.Strong('Created: '),
                    html.Span(pd.to_datetime(work_order['created_at']).strftime('%Y-%m-%d %H:%M'))
                ]),
                html.P([
                    html.Strong('Due Date: '),
                    html.Span(pd.to_datetime(work_order['due_date']).strftime('%Y-%m-%d %H:%M')
                             if pd.notna(work_order['due_date']) else 'N/A')
                ]),
                html.P([
                    html.Strong('Estimated Duration: '),
                    html.Span(f"{work_order.get('estimated_duration', 0)} minutes")
                ])
            ], className='detail-section'),
            
            # Assignment Information
            html.Div([
                html.H6('Assignment', className='detail-section-title'),
                html.P([
                    html.Strong('Technician: '),
                    html.Span(work_order.get('assigned_technician', 'Unassigned'))
                ]),
                html.P([
                    html.Strong('Anomaly ID: '),
                    html.Span(work_order.get('anomaly_id', 'N/A'))
                ])
            ], className='detail-section'),
            
            # Description
            html.Div([
                html.H6('Description', className='detail-section-title'),
                html.P(work_order.get('description', 'No description available'))
            ], className='detail-section'),
            
            # Action Buttons
            html.Div([
                html.Button('Update Status', id='update-status-btn', 
                           className='btn btn-primary btn-sm'),
                html.Button('Reassign', id='reassign-btn', 
                           className='btn btn-secondary btn-sm'),
                html.Button('Print', id='print-wo-btn', 
                           className='btn btn-info btn-sm')
            ], className='detail-actions')
        ], className='detail-content')
    ], className='work-order-details')

def calculate_statistics(work_orders_df: pd.DataFrame) -> Dict[str, int]:
    """Calculate work order statistics"""
    if work_orders_df.empty:
        return {
            'total': 0,
            'pending': 0,
            'in_progress': 0,
            'completed_today': 0
        }
    
    today = datetime.now().date()
    
    return {
        'total': len(work_orders_df),
        'pending': len(work_orders_df[work_orders_df['status'] == 'PENDING']),
        'in_progress': len(work_orders_df[work_orders_df['status'] == 'IN_PROGRESS']),
        'completed_today': len(work_orders_df[
            (work_orders_df['status'] == 'COMPLETED') &
            (pd.to_datetime(work_orders_df['created_at']).dt.date == today)
        ])
    }

def get_available_technicians() -> List[Dict[str, str]]:
    """Get list of available technicians"""
    try:
        query = """
            SELECT DISTINCT assigned_technician 
            FROM work_orders 
            WHERE assigned_technician IS NOT NULL
            ORDER BY assigned_technician
        """
        
        result = db_manager.execute_query(query)
        
        technicians = [{'label': 'All', 'value': 'ALL'}]
        if not result.empty:
            technicians.extend([
                {'label': tech, 'value': tech} 
                for tech in result['assigned_technician'].tolist()
            ])
        
        return technicians
        
    except Exception as e:
        logger.error(f"Error fetching technicians: {str(e)}")
        return [{'label': 'All', 'value': 'ALL'}]

def update_work_order_status(work_order_id: str, new_status: str, 
                            notes: Optional[str] = None) -> bool:
    """Update work order status in database"""
    try:
        query = """
            UPDATE work_orders 
            SET status = %s, 
                updated_at = %s,
                notes = COALESCE(%s, notes)
            WHERE work_order_id = %s
        """
        
        params = [new_status, datetime.now(), notes, work_order_id]
        db_manager.execute_update(query, params)
        
        logger.info(f"Updated work order {work_order_id} status to {new_status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating work order status: {str(e)}")
        return False

def reassign_work_order(work_order_id: str, new_technician: str) -> bool:
    """Reassign work order to different technician"""
    try:
        query = """
            UPDATE work_orders 
            SET assigned_technician = %s,
                status = 'ASSIGNED',
                updated_at = %s
            WHERE work_order_id = %s
        """
        
        params = [new_technician, datetime.now(), work_order_id]
        db_manager.execute_update(query, params)
        
        logger.info(f"Reassigned work order {work_order_id} to {new_technician}")
        return True
        
    except Exception as e:
        logger.error(f"Error reassigning work order: {str(e)}")
        return False

def export_work_orders(work_orders_df: pd.DataFrame, format: str = 'csv') -> str:
    """Export work orders to file"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filename = f"work_orders_{timestamp}.csv"
            filepath = f"exports/{filename}"
            work_orders_df.to_csv(filepath, index=False)
        elif format == 'excel':
            filename = f"work_orders_{timestamp}.xlsx"
            filepath = f"exports/{filename}"
            work_orders_df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported work orders to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting work orders: {str(e)}")
        return None

# Register callbacks
def register_callbacks(app, data_service=None):
    """Register all callbacks for work orders dashboard"""
    
    @app.callback(
        [Output('work-orders-data-store', 'data'),
         Output('work-orders-table', 'data'),
         Output('total-orders', 'children'),
         Output('pending-orders', 'children'),
         Output('in-progress-orders', 'children'),
         Output('completed-orders', 'children'),
         Output('work-order-timeline', 'figure'),
         Output('work-orders-by-status-chart', 'figure'),
         Output('work-orders-by-priority-chart', 'figure'),
         Output('technician-workload-chart', 'figure'),
         Output('work-order-technician-filter', 'options')],
        [Input('refresh-work-orders-btn', 'n_clicks'),
         Input('work-orders-interval', 'n_intervals')],
        [State('work-order-date-range', 'start_date'),
         State('work-order-date-range', 'end_date'),
         State('work-order-status-filter', 'value'),
         State('work-order-priority-filter', 'value'),
         State('work-order-technician-filter', 'value')]
    )
    def update_work_orders_dashboard(n_clicks, n_intervals, start_date, end_date, 
                                    status, priority, technician):
        """Update all work orders dashboard components"""
        try:
            # Fetch work orders
            work_orders_df = fetch_work_orders(start_date, end_date, status, 
                                              priority, technician)
            
            # Calculate statistics
            stats = calculate_statistics(work_orders_df)
            
            # Generate charts
            timeline_fig = generate_timeline_chart(work_orders_df)
            status_fig = generate_status_chart(work_orders_df)
            priority_fig = generate_priority_chart(work_orders_df)
            workload_fig = generate_workload_chart(work_orders_df)
            
            # Get technicians
            technicians = get_available_technicians()
            
            # Prepare table data
            table_data = work_orders_df.to_dict('records')
            
            return (
                work_orders_df.to_json(date_format='iso', orient='split'),
                table_data,
                str(stats['total']),
                str(stats['pending']),
                str(stats['in_progress']),
                str(stats['completed_today']),
                timeline_fig,
                status_fig,
                priority_fig,
                workload_fig,
                technicians
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            return (None, [], '0', '0', '0', '0', 
                   go.Figure(), go.Figure(), go.Figure(), go.Figure(), [])
    
    @app.callback(
        Output('work-order-details', 'children'),
        [Input('work-orders-table', 'selected_rows')],
        [State('work-orders-data-store', 'data')]
    )
    def display_work_order_details(selected_rows, data_json):
        """Display details for selected work order"""
        if not selected_rows or not data_json:
            return html.Div('Select a work order to view details', className='no-selection')
        
        try:
            work_orders_df = pd.read_json(data_json, orient='split')
            selected_order = work_orders_df.iloc[selected_rows[0]]
            return format_work_order_details(selected_order)
            
        except Exception as e:
            logger.error(f"Error displaying work order details: {str(e)}")
            return html.Div('Error loading work order details', className='error-message')
    
    @app.callback(
        [Output('work-order-modal', 'style'),
         Output('modal-work-order-content', 'children')],
        [Input('update-status-btn', 'n_clicks'),
         Input('reassign-btn', 'n_clicks'),
         Input('close-modal-btn', 'n_clicks'),
         Input('cancel-work-order-btn', 'n_clicks')],
        [State('work-orders-table', 'selected_rows'),
         State('work-orders-data-store', 'data')]
    )
    def toggle_modal(update_clicks, reassign_clicks, close_clicks, 
                    cancel_clicks, selected_rows, data_json):
        """Toggle work order update modal"""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            return {'display': 'none'}, ''
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id in ['close-modal-btn', 'cancel-work-order-btn']:
            return {'display': 'none'}, ''
        
        if trigger_id in ['update-status-btn', 'reassign-btn'] and selected_rows and data_json:
            try:
                work_orders_df = pd.read_json(data_json, orient='split')
                selected_order = work_orders_df.iloc[selected_rows[0]]
                
                if trigger_id == 'update-status-btn':
                    content = create_status_update_form(selected_order)
                else:
                    content = create_reassignment_form(selected_order)
                
                return {'display': 'block'}, content
                
            except Exception as e:
                logger.error(f"Error opening modal: {str(e)}")
                return {'display': 'none'}, ''
        
        return {'display': 'none'}, ''

def create_status_update_form(work_order: pd.Series) -> html.Div:
    """Create form for updating work order status"""
    return html.Div([
        html.H5(f"Update Status for WO-{work_order['work_order_id']}"),
        html.Div([
            html.Label('New Status:'),
            dcc.Dropdown(
                id='new-status-dropdown',
                options=[
                    {'label': 'Pending', 'value': 'PENDING'},
                    {'label': 'Assigned', 'value': 'ASSIGNED'},
                    {'label': 'In Progress', 'value': 'IN_PROGRESS'},
                    {'label': 'On Hold', 'value': 'ON_HOLD'},
                    {'label': 'Completed', 'value': 'COMPLETED'},
                    {'label': 'Cancelled', 'value': 'CANCELLED'}
                ],
                value=work_order['status']
            )
        ], className='form-group'),
        html.Div([
            html.Label('Notes:'),
            dcc.Textarea(
                id='status-update-notes',
                placeholder='Add notes about this status change...',
                style={'width': '100%', 'height': 100}
            )
        ], className='form-group')
    ])

def create_reassignment_form(work_order: pd.Series) -> html.Div:
    """Create form for reassigning work order"""
    technicians = get_available_technicians()
    
    return html.Div([
        html.H5(f"Reassign WO-{work_order['work_order_id']}"),
        html.Div([
            html.Label('Assign to Technician:'),
            dcc.Dropdown(
                id='new-technician-dropdown',
                options=technicians[1:],  # Exclude 'All' option
                value=work_order.get('assigned_technician', '')
            )
        ], className='form-group'),
        html.Div([
            html.Label('Reason for Reassignment:'),
            dcc.Textarea(
                id='reassignment-reason',
                placeholder='Provide reason for reassignment...',
                style={'width': '100%', 'height': 100}
            )
        ], className='form-group')
    ])