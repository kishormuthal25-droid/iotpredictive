"""
Dashboard Callbacks Module
Integrates all dashboard components and manages application state
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
from collections import deque

# Import project modules - with fallback error handling
try:
    from src.utils.logger import get_logger
    from src.data_ingestion.kafka_consumer import KafkaConsumer
    from src.data_ingestion.database_manager import DatabaseManager
    from src.anomaly_detection.model_evaluator import ModelEvaluator
    from src.forecasting.forecast_evaluator import ForecastEvaluator
    from src.maintenance.scheduler import MaintenanceScheduler
    from src.maintenance.work_order_manager import WorkOrderManager
    from src.alerts.alert_manager import AlertManager
    from src.preprocessing.feature_engineering import FeatureEngineer
    from config.settings import Settings
    logger = get_logger(__name__)
    settings = Settings()

except ImportError as e:
    # Fallback imports for dashboard-only mode
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"Some modules not available in dashboard-only mode: {e}")

    # Create mock classes for dashboard-only operation
    class MockDataManager:
        def execute_query(self, query, params=None):
            return pd.DataFrame()

    class MockSettings:
        def __init__(self):
            self.database = {'host': 'localhost', 'port': 5432, 'name': 'iot_db', 'user': 'user', 'password': 'pass'}
            self.kafka = {'bootstrap_servers': 'localhost:9092', 'topics': {'smap': 'smap_data', 'msl': 'msl_data', 'anomalies': 'anomalies'}}
            self.anomaly = {'threshold_percentile': 95, 'severity_levels': {'high': 0.8, 'medium': 0.6}, 'min_consecutive_anomalies': 3}
            self.alerts = {'cooldown_minutes': 30}
            self.maintenance = {'constraints': {}}
            self.models = {'anomaly_detection': {}}

    # Use mock classes
    settings = MockSettings()
    db_manager = MockDataManager()
    kafka_consumer = None
    model_evaluator = None
    forecast_evaluator = None
    maintenance_scheduler = None
    work_order_manager = None
    alert_manager = None
    feature_engineer = None

# Initialize components (only if not already initialized in try/except block)
if 'db_manager' not in locals():
    try:
        db_manager = DatabaseManager()
        kafka_consumer = KafkaConsumer()
        model_evaluator = ModelEvaluator()
        forecast_evaluator = ForecastEvaluator()
        maintenance_scheduler = MaintenanceScheduler()
        work_order_manager = WorkOrderManager()
        alert_manager = AlertManager()
        feature_engineer = FeatureEngineer()
    except Exception as e:
        logger.warning(f"Could not initialize all components: {e}")
        # Use mock objects as fallback
        db_manager = MockDataManager() if 'MockDataManager' in locals() else None

# Global data stores for real-time updates
TELEMETRY_BUFFER = deque(maxlen=1000)
ANOMALY_BUFFER = deque(maxlen=500)
ALERT_QUEUE = queue.Queue(maxsize=100)
ACTIVE_MODELS = {}
STREAMING_THREADS = {}

class DashboardCallbacks:
    """Main callbacks manager for the dashboard"""
    
    def __init__(self, app):
        self.app = app
        self.register_all_callbacks()
        self.initialize_streaming()
        
    def initialize_streaming(self):
        """Initialize real-time data streaming threads"""
        try:
            # Start Kafka consumer threads
            if 'telemetry' not in STREAMING_THREADS:
                telemetry_thread = threading.Thread(
                    target=self.stream_telemetry_data,
                    daemon=True
                )
                telemetry_thread.start()
                STREAMING_THREADS['telemetry'] = telemetry_thread
                
            if 'anomalies' not in STREAMING_THREADS:
                anomaly_thread = threading.Thread(
                    target=self.stream_anomaly_data,
                    daemon=True
                )
                anomaly_thread.start()
                STREAMING_THREADS['anomalies'] = anomaly_thread
                
            logger.info("Streaming threads initialized")
            
        except Exception as e:
            logger.error(f"Error initializing streaming: {str(e)}")
    
    def stream_telemetry_data(self):
        """Stream telemetry data from Kafka"""
        while True:
            try:
                messages = kafka_consumer.consume_messages(
                    topic=settings.kafka['topics']['smap'],
                    timeout=1.0
                )
                
                for message in messages:
                    if len(TELEMETRY_BUFFER) >= TELEMETRY_BUFFER.maxlen:
                        TELEMETRY_BUFFER.popleft()
                    TELEMETRY_BUFFER.append(message)
                    
            except Exception as e:
                logger.error(f"Error streaming telemetry: {str(e)}")
                
    def stream_anomaly_data(self):
        """Stream anomaly detection results"""
        while True:
            try:
                messages = kafka_consumer.consume_messages(
                    topic=settings.kafka['topics']['anomalies'],
                    timeout=1.0
                )
                
                for message in messages:
                    if len(ANOMALY_BUFFER) >= ANOMALY_BUFFER.maxlen:
                        ANOMALY_BUFFER.popleft()
                    ANOMALY_BUFFER.append(message)
                    
                    # Check if alert needed
                    if message.get('severity', 0) > settings.anomaly['severity_levels']['medium']:
                        if not ALERT_QUEUE.full():
                            ALERT_QUEUE.put(message)
                            
            except Exception as e:
                logger.error(f"Error streaming anomalies: {str(e)}")
    
    def register_all_callbacks(self):
        """Register all dashboard callbacks"""
        self.register_navigation_callbacks()
        self.register_data_update_callbacks()
        self.register_model_callbacks()
        self.register_alert_callbacks()
        self.register_export_callbacks()
        self.register_settings_callbacks()
        
    def register_navigation_callbacks(self):
        """Register navigation and page routing callbacks"""
        
        @self.app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            """Route to appropriate dashboard page"""
            if pathname == '/' or pathname == '/overview':
                from src.dashboard.layouts.overview import create_layout
                return create_layout()
            elif pathname == '/anomalies':
                from src.dashboard.layouts.anomaly_monitor import create_layout
                return create_layout()
            elif pathname == '/forecast':
                from src.dashboard.layouts.forecast_view import create_layout
                return create_layout()
            elif pathname == '/maintenance':
                from src.dashboard.layouts.maintenance_scheduler import create_layout
                return create_layout()
            elif pathname == '/work-orders':
                from src.dashboard.layouts.work_orders import create_layout
                return create_layout()
            else:
                return html.Div([
                    html.H1('404: Page not found'),
                    html.P(f'The pathname {pathname} was not recognized...'),
                    dcc.Link('Go to Overview', href='/')
                ])
        
        @self.app.callback(
            Output('nav-sidebar', 'className'),
            [Input('nav-toggle', 'n_clicks')],
            [State('nav-sidebar', 'className')]
        )
        def toggle_sidebar(n_clicks, current_class):
            """Toggle navigation sidebar"""
            if n_clicks:
                if 'collapsed' in current_class:
                    return current_class.replace('collapsed', '')
                else:
                    return current_class + ' collapsed'
            return current_class
    
    def register_data_update_callbacks(self):
        """Register callbacks for real-time data updates"""
        
        @self.app.callback(
            [Output('live-telemetry-store', 'data'),
             Output('live-anomaly-store', 'data'),
             Output('system-metrics-store', 'data')],
            [Input('main-interval-component', 'n_intervals')]
        )
        def update_live_data(n_intervals):
            """Update live data stores from buffers"""
            try:
                # Get latest telemetry data
                telemetry_data = list(TELEMETRY_BUFFER)[-100:] if TELEMETRY_BUFFER else []
                
                # Get latest anomaly data
                anomaly_data = list(ANOMALY_BUFFER)[-50:] if ANOMALY_BUFFER else []
                
                # Calculate system metrics
                system_metrics = self.calculate_system_metrics()
                
                return telemetry_data, anomaly_data, system_metrics
                
            except Exception as e:
                logger.error(f"Error updating live data: {str(e)}")
                return [], [], {}
        
        @self.app.callback(
            Output('equipment-health-store', 'data'),
            [Input('health-check-interval', 'n_intervals')]
        )
        def update_equipment_health(n_intervals):
            """Update equipment health scores"""
            try:
                query = """
                    SELECT 
                        e.equipment_id,
                        e.equipment_name,
                        e.criticality,
                        COUNT(a.anomaly_id) as anomaly_count,
                        AVG(a.anomaly_score) as avg_anomaly_score,
                        MAX(a.created_at) as last_anomaly,
                        COUNT(wo.work_order_id) as active_work_orders
                    FROM equipment e
                    LEFT JOIN anomalies a ON e.equipment_id = a.equipment_id
                        AND a.created_at > NOW() - INTERVAL '7 days'
                    LEFT JOIN work_orders wo ON e.equipment_id = wo.equipment_id
                        AND wo.status IN ('PENDING', 'IN_PROGRESS')
                    GROUP BY e.equipment_id, e.equipment_name, e.criticality
                """
                
                health_df = db_manager.execute_query(query)
                
                # Calculate health scores
                health_df['health_score'] = health_df.apply(
                    lambda row: self.calculate_health_score(row), axis=1
                )
                
                return health_df.to_dict('records')
                
            except Exception as e:
                logger.error(f"Error updating equipment health: {str(e)}")
                return []
        
        @self.app.callback(
            [Output('telemetry-chart', 'figure'),
             Output('anomaly-timeline', 'figure'),
             Output('health-heatmap', 'figure')],
            [Input('live-telemetry-store', 'data'),
             Input('live-anomaly-store', 'data'),
             Input('equipment-health-store', 'data')],
            [State('selected-equipment', 'value'),
             State('selected-timerange', 'value')]
        )
        def update_main_charts(telemetry_data, anomaly_data, health_data, 
                              selected_equipment, timerange):
            """Update main dashboard charts"""
            try:
                # Create telemetry chart
                telemetry_fig = self.create_telemetry_chart(
                    telemetry_data, selected_equipment, timerange
                )
                
                # Create anomaly timeline
                anomaly_fig = self.create_anomaly_timeline(
                    anomaly_data, selected_equipment, timerange
                )
                
                # Create health heatmap
                health_fig = self.create_health_heatmap(health_data)
                
                return telemetry_fig, anomaly_fig, health_fig
                
            except Exception as e:
                logger.error(f"Error updating charts: {str(e)}")
                return go.Figure(), go.Figure(), go.Figure()
    
    def register_model_callbacks(self):
        """Register callbacks for ML model management"""
        
        @self.app.callback(
            [Output('model-performance-chart', 'figure'),
             Output('model-comparison-table', 'data'),
             Output('active-model-indicator', 'children')],
            [Input('model-selector', 'value'),
             Input('model-refresh-btn', 'n_clicks')]
        )
        def update_model_dashboard(selected_model, n_clicks):
            """Update model performance dashboard"""
            try:
                # Get model performance metrics
                performance_data = model_evaluator.get_model_performance(selected_model)
                
                # Create performance chart
                perf_fig = self.create_model_performance_chart(performance_data)
                
                # Get comparison data
                comparison_data = model_evaluator.compare_models()
                
                # Get active model info
                active_model = ACTIVE_MODELS.get(selected_model, {})
                active_indicator = f"Active: {active_model.get('name', 'None')}"
                
                return perf_fig, comparison_data, active_indicator
                
            except Exception as e:
                logger.error(f"Error updating model dashboard: {str(e)}")
                return go.Figure(), [], "Error loading model"
        
        @self.app.callback(
            Output('retrain-status', 'children'),
            [Input('retrain-model-btn', 'n_clicks')],
            [State('model-selector', 'value'),
             State('retrain-params', 'value')]
        )
        def retrain_model(n_clicks, model_name, params):
            """Trigger model retraining"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                # Parse parameters
                train_params = json.loads(params) if params else {}
                
                # Start retraining in background
                threading.Thread(
                    target=self.retrain_model_background,
                    args=(model_name, train_params),
                    daemon=True
                ).start()
                
                return html.Div([
                    html.Span('Model retraining started...', 
                             className='status-message info')
                ])
                
            except Exception as e:
                logger.error(f"Error starting retraining: {str(e)}")
                return html.Div([
                    html.Span(f'Error: {str(e)}', 
                             className='status-message error')
                ])
    
    def register_alert_callbacks(self):
        """Register callbacks for alert management"""
        
        @self.app.callback(
            [Output('alert-banner', 'children'),
             Output('alert-banner', 'style'),
             Output('alert-count', 'children'),
             Output('alert-list', 'children')],
            [Input('alert-check-interval', 'n_intervals')]
        )
        def update_alerts(n_intervals):
            """Update alert displays"""
            try:
                alerts = []
                while not ALERT_QUEUE.empty() and len(alerts) < 10:
                    alerts.append(ALERT_QUEUE.get_nowait())
                
                if not alerts:
                    return '', {'display': 'none'}, '0', []
                
                # Create alert banner for highest priority
                highest_alert = max(alerts, key=lambda x: x.get('severity', 0))
                banner = self.create_alert_banner(highest_alert)
                
                # Style based on severity
                severity = highest_alert.get('severity', 0)
                if severity > 0.95:
                    style = {'backgroundColor': '#dc3545', 'color': 'white'}
                elif severity > 0.85:
                    style = {'backgroundColor': '#fd7e14', 'color': 'white'}
                else:
                    style = {'backgroundColor': '#ffc107', 'color': 'black'}
                
                # Create alert list
                alert_list = [self.create_alert_item(alert) for alert in alerts]
                
                return banner, style, str(len(alerts)), alert_list
                
            except Exception as e:
                logger.error(f"Error updating alerts: {str(e)}")
                return '', {'display': 'none'}, '0', []
        
        @self.app.callback(
            Output('alert-response-status', 'children'),
            [Input({'type': 'acknowledge-alert', 'index': ALL}, 'n_clicks'),
             Input({'type': 'dismiss-alert', 'index': ALL}, 'n_clicks')],
            [State({'type': 'alert-id', 'index': ALL}, 'children')]
        )
        def handle_alert_response(ack_clicks, dismiss_clicks, alert_ids):
            """Handle alert acknowledgment and dismissal"""
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            try:
                triggered = ctx.triggered[0]
                prop_id = json.loads(triggered['prop_id'].split('.')[0])
                action = prop_id['type'].replace('-alert', '')
                index = prop_id['index']
                alert_id = alert_ids[index] if index < len(alert_ids) else None
                
                if alert_id:
                    if action == 'acknowledge':
                        alert_manager.acknowledge_alert(alert_id)
                        return f"Alert {alert_id} acknowledged"
                    elif action == 'dismiss':
                        alert_manager.dismiss_alert(alert_id)
                        return f"Alert {alert_id} dismissed"
                
                return ""
                
            except Exception as e:
                logger.error(f"Error handling alert response: {str(e)}")
                return f"Error: {str(e)}"
    
    def register_export_callbacks(self):
        """Register callbacks for data export functionality"""
        
        @self.app.callback(
            [Output('export-status', 'children'),
             Output('download-link', 'href')],
            [Input('export-data-btn', 'n_clicks')],
            [State('export-type', 'value'),
             State('export-daterange', 'start_date'),
             State('export-daterange', 'end_date'),
             State('export-format', 'value')]
        )
        def export_data(n_clicks, export_type, start_date, end_date, format):
            """Export data based on selection"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                # Generate export based on type
                if export_type == 'telemetry':
                    filepath = self.export_telemetry_data(
                        start_date, end_date, format
                    )
                elif export_type == 'anomalies':
                    filepath = self.export_anomaly_data(
                        start_date, end_date, format
                    )
                elif export_type == 'work_orders':
                    filepath = self.export_work_orders(
                        start_date, end_date, format
                    )
                elif export_type == 'report':
                    filepath = self.generate_report(
                        start_date, end_date, format
                    )
                else:
                    return "Invalid export type", ""
                
                if filepath:
                    return f"Export completed: {filepath}", f"/download/{filepath}"
                else:
                    return "Export failed", ""
                    
            except Exception as e:
                logger.error(f"Error exporting data: {str(e)}")
                return f"Error: {str(e)}", ""
    
    def register_settings_callbacks(self):
        """Register callbacks for system settings"""
        
        @self.app.callback(
            Output('settings-save-status', 'children'),
            [Input('save-settings-btn', 'n_clicks')],
            [State('anomaly-threshold', 'value'),
             State('alert-cooldown', 'value'),
             State('maintenance-constraints', 'value'),
             State('model-params', 'value')]
        )
        def save_settings(n_clicks, threshold, cooldown, constraints, model_params):
            """Save system settings"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                # Update settings
                settings.anomaly['threshold_percentile'] = threshold
                settings.alerts['cooldown_minutes'] = cooldown
                
                if constraints:
                    constraint_dict = json.loads(constraints)
                    settings.maintenance['constraints'].update(constraint_dict)
                
                if model_params:
                    params_dict = json.loads(model_params)
                    settings.models['anomaly_detection'].update(params_dict)
                
                # Save to file
                settings.save()
                
                return html.Div([
                    html.Span('Settings saved successfully', 
                             className='status-message success')
                ])
                
            except Exception as e:
                logger.error(f"Error saving settings: {str(e)}")
                return html.Div([
                    html.Span(f'Error: {str(e)}', 
                             className='status-message error')
                ])
        
        @self.app.callback(
            [Output('anomaly-threshold', 'value'),
             Output('alert-cooldown', 'value'),
             Output('maintenance-constraints', 'value'),
             Output('model-params', 'value')],
            [Input('load-settings-btn', 'n_clicks')]
        )
        def load_settings(n_clicks):
            """Load current system settings"""
            if not n_clicks:
                raise PreventUpdate
                
            try:
                return (
                    settings.anomaly['threshold_percentile'],
                    settings.alerts['cooldown_minutes'],
                    json.dumps(settings.maintenance['constraints'], indent=2),
                    json.dumps(settings.models['anomaly_detection'], indent=2)
                )
                
            except Exception as e:
                logger.error(f"Error loading settings: {str(e)}")
                return 95, 30, "{}", "{}"
    
    # Helper methods for callbacks
    
    def calculate_system_metrics(self) -> Dict[str, Any]:
        """Calculate overall system metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'active_equipment': 0,
                'total_anomalies_24h': 0,
                'avg_health_score': 0,
                'pending_work_orders': 0,
                'model_accuracy': 0,
                'system_uptime': 0
            }
            
            # Query database for metrics
            query = """
                SELECT 
                    COUNT(DISTINCT e.equipment_id) as active_equipment,
                    COUNT(DISTINCT a.anomaly_id) as total_anomalies,
                    AVG(CASE 
                        WHEN a.anomaly_score IS NOT NULL 
                        THEN (1 - a.anomaly_score) * 100 
                        ELSE 100 
                    END) as avg_health_score,
                    COUNT(DISTINCT wo.work_order_id) as pending_work_orders
                FROM equipment e
                LEFT JOIN anomalies a ON e.equipment_id = a.equipment_id
                    AND a.created_at > NOW() - INTERVAL '24 hours'
                LEFT JOIN work_orders wo ON e.equipment_id = wo.equipment_id
                    AND wo.status IN ('PENDING', 'ASSIGNED')
            """
            
            result = db_manager.execute_query(query)
            
            if not result.empty:
                metrics.update(result.iloc[0].to_dict())
            
            # Add model metrics
            if ACTIVE_MODELS:
                accuracies = [m.get('accuracy', 0) for m in ACTIVE_MODELS.values()]
                metrics['model_accuracy'] = np.mean(accuracies) if accuracies else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {str(e)}")
            return {}
    
    def calculate_health_score(self, equipment_row: pd.Series) -> float:
        """Calculate health score for equipment"""
        try:
            base_score = 100.0
            
            # Deduct for anomalies
            anomaly_penalty = min(equipment_row['anomaly_count'] * 5, 30)
            base_score -= anomaly_penalty
            
            # Deduct for average anomaly score
            if equipment_row['avg_anomaly_score']:
                score_penalty = equipment_row['avg_anomaly_score'] * 20
                base_score -= score_penalty
            
            # Deduct for active work orders
            work_order_penalty = equipment_row['active_work_orders'] * 10
            base_score -= work_order_penalty
            
            # Factor in criticality
            if equipment_row['criticality'] == 'HIGH':
                base_score *= 0.9
            elif equipment_row['criticality'] == 'CRITICAL':
                base_score *= 0.8
            
            return max(0, min(100, base_score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {str(e)}")
            return 50.0
    
    def create_telemetry_chart(self, telemetry_data: List, 
                               equipment: str, timerange: str) -> go.Figure:
        """Create telemetry time series chart"""
        try:
            if not telemetry_data:
                return go.Figure()
            
            df = pd.DataFrame(telemetry_data)
            
            if equipment and equipment != 'ALL':
                df = df[df['equipment_id'] == equipment]
            
            # Filter by timerange
            if timerange:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff = datetime.now() - timedelta(hours=int(timerange))
                df = df[df['timestamp'] > cutoff]
            
            fig = go.Figure()
            
            # Add traces for each sensor
            for column in df.select_dtypes(include=[np.number]).columns:
                if column not in ['equipment_id', 'anomaly_score']:
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[column],
                        mode='lines',
                        name=column,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title='Telemetry Data',
                xaxis_title='Time',
                yaxis_title='Value',
                hovermode='x unified',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating telemetry chart: {str(e)}")
            return go.Figure()
    
    def create_anomaly_timeline(self, anomaly_data: List, 
                               equipment: str, timerange: str) -> go.Figure:
        """Create anomaly timeline chart"""
        try:
            if not anomaly_data:
                return go.Figure()
            
            df = pd.DataFrame(anomaly_data)
            
            if equipment and equipment != 'ALL':
                df = df[df['equipment_id'] == equipment]
            
            # Filter by timerange
            if timerange:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                cutoff = datetime.now() - timedelta(hours=int(timerange))
                df = df[df['timestamp'] > cutoff]
            
            fig = go.Figure()
            
            # Add scatter plot for anomalies
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['anomaly_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['anomaly_score'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title='Anomaly Score')
                ),
                text=df['equipment_id'],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<br>Time: %{x}'
            ))
            
            # Add threshold line
            threshold = settings.anomaly['threshold_percentile'] / 100
            fig.add_hline(
                y=threshold,
                line_dash='dash',
                line_color='red',
                annotation_text='Threshold'
            )
            
            fig.update_layout(
                title='Anomaly Detection Timeline',
                xaxis_title='Time',
                yaxis_title='Anomaly Score',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anomaly timeline: {str(e)}")
            return go.Figure()
    
    def create_health_heatmap(self, health_data: List) -> go.Figure:
        """Create equipment health heatmap"""
        try:
            if not health_data:
                return go.Figure()
            
            df = pd.DataFrame(health_data)
            
            # Pivot data for heatmap
            pivot_df = df.pivot_table(
                values='health_score',
                index='equipment_name',
                aggfunc='mean'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_df.values.reshape(-1, 1),
                y=pivot_df.index,
                colorscale='RdYlGn',
                zmid=50,
                text=pivot_df.values.reshape(-1, 1),
                texttemplate='%{text:.1f}',
                textfont={"size": 12},
                colorbar=dict(title='Health Score')
            ))
            
            fig.update_layout(
                title='Equipment Health Status',
                xaxis_title='',
                yaxis_title='Equipment',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating health heatmap: {str(e)}")
            return go.Figure()
    
    def create_model_performance_chart(self, performance_data: Dict) -> go.Figure:
        """Create model performance comparison chart"""
        try:
            if not performance_data:
                return go.Figure()
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            models = list(performance_data.keys())
            
            fig = go.Figure()
            
            for metric in metrics:
                values = [performance_data[model].get(metric, 0) for model in models]
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=models,
                    y=values,
                    text=[f'{v:.2%}' for v in values],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Model',
                yaxis_title='Score',
                barmode='group',
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model performance chart: {str(e)}")
            return go.Figure()
    
    def create_alert_banner(self, alert: Dict) -> html.Div:
        """Create alert banner component"""
        return html.Div([
            html.Span('[!] ', className='alert-icon'),
            html.Strong(f"ALERT: {alert.get('type', 'Unknown')} "),
            html.Span(f"on {alert.get('equipment_id', 'Unknown Equipment')} - "),
            html.Span(f"Severity: {alert.get('severity', 0):.2%} "),
            html.Button('Acknowledge', 
                       id={'type': 'acknowledge-alert', 'index': 0},
                       className='btn btn-sm btn-light ml-2'),
            html.Button('Dismiss', 
                       id={'type': 'dismiss-alert', 'index': 0},
                       className='btn btn-sm btn-light ml-1')
        ], className='alert-banner-content')
    
    def create_alert_item(self, alert: Dict) -> html.Div:
        """Create alert list item"""
        severity = alert.get('severity', 0)
        if severity > 0.95:
            severity_class = 'critical'
        elif severity > 0.85:
            severity_class = 'high'
        elif severity > 0.7:
            severity_class = 'medium'
        else:
            severity_class = 'low'
        
        return html.Div([
            html.Div([
                html.Span(alert.get('timestamp', ''), className='alert-time'),
                html.Span(alert.get('type', 'Unknown'), 
                         className=f'alert-type severity-{severity_class}')
            ], className='alert-header'),
            html.Div([
                html.P(f"Equipment: {alert.get('equipment_id', 'Unknown')}"),
                html.P(f"Score: {alert.get('anomaly_score', 0):.3f}"),
                html.P(alert.get('description', 'No description'))
            ], className='alert-body'),
            html.Div([
                html.Span(alert.get('alert_id', ''), 
                         id={'type': 'alert-id', 'index': alert.get('index', 0)},
                         style={'display': 'none'}),
                html.Button('Acknowledge', 
                           id={'type': 'acknowledge-alert', 
                               'index': alert.get('index', 0)},
                           className='btn btn-sm btn-primary'),
                html.Button('Create Work Order', 
                           id={'type': 'create-wo-alert', 
                               'index': alert.get('index', 0)},
                           className='btn btn-sm btn-secondary')
            ], className='alert-actions')
        ], className=f'alert-item severity-{severity_class}')
    
    def retrain_model_background(self, model_name: str, params: Dict):
        """Retrain model in background thread"""
        try:
            logger.info(f"Starting retraining for model: {model_name}")
            
            # Load training data
            query = """
                SELECT * FROM telemetry_data 
                WHERE timestamp > NOW() - INTERVAL '30 days'
                ORDER BY timestamp
            """
            training_data = db_manager.execute_query(query)
            
            # Perform feature engineering
            features = feature_engineer.engineer_features(training_data)
            
            # Train model based on type
            if 'lstm' in model_name.lower():
                from src.anomaly_detection.lstm_detector import LSTMDetector
                model = LSTMDetector()
            elif 'autoencoder' in model_name.lower():
                from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
                model = LSTMAutoencoder()
            elif 'vae' in model_name.lower():
                from src.anomaly_detection.lstm_vae import LSTMVAE
                model = LSTMVAE()
            else:
                logger.error(f"Unknown model type: {model_name}")
                return
            
            # Train model
            model.train(features, **params)
            
            # Save model
            model.save(f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Update active models
            ACTIVE_MODELS[model_name] = {
                'name': model_name,
                'model': model,
                'trained_at': datetime.now(),
                'params': params,
                'accuracy': model.evaluate(features)
            }
            
            logger.info(f"Model {model_name} retrained successfully")
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}")
    
    def export_telemetry_data(self, start_date: str, end_date: str, 
                             format: str) -> Optional[str]:
        """Export telemetry data to file"""
        try:
            query = """
                SELECT * FROM telemetry_data
                WHERE timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            
            data = db_manager.execute_query(query, [start_date, end_date])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'csv':
                filepath = f"exports/telemetry_{timestamp}.csv"
                data.to_csv(filepath, index=False)
            elif format == 'parquet':
                filepath = f"exports/telemetry_{timestamp}.parquet"
                data.to_parquet(filepath, index=False)
            else:
                filepath = f"exports/telemetry_{timestamp}.json"
                data.to_json(filepath, orient='records', date_format='iso')
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting telemetry data: {str(e)}")
            return None
    
    def export_anomaly_data(self, start_date: str, end_date: str, 
                           format: str) -> Optional[str]:
        """Export anomaly data to file"""
        try:
            query = """
                SELECT 
                    a.*,
                    e.equipment_name,
                    e.location,
                    e.criticality
                FROM anomalies a
                JOIN equipment e ON a.equipment_id = e.equipment_id
                WHERE a.created_at BETWEEN %s AND %s
                ORDER BY a.created_at
            """
            
            data = db_manager.execute_query(query, [start_date, end_date])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'csv':
                filepath = f"exports/anomalies_{timestamp}.csv"
                data.to_csv(filepath, index=False)
            elif format == 'excel':
                filepath = f"exports/anomalies_{timestamp}.xlsx"
                data.to_excel(filepath, index=False, engine='openpyxl')
            else:
                filepath = f"exports/anomalies_{timestamp}.json"
                data.to_json(filepath, orient='records', date_format='iso')
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting anomaly data: {str(e)}")
            return None
    
    def export_work_orders(self, start_date: str, end_date: str, 
                          format: str) -> Optional[str]:
        """Export work orders to file"""
        try:
            query = """
                SELECT 
                    wo.*,
                    e.equipment_name,
                    e.location,
                    a.anomaly_score
                FROM work_orders wo
                LEFT JOIN equipment e ON wo.equipment_id = e.equipment_id
                LEFT JOIN anomalies a ON wo.anomaly_id = a.anomaly_id
                WHERE wo.created_at BETWEEN %s AND %s
                ORDER BY wo.created_at
            """
            
            data = db_manager.execute_query(query, [start_date, end_date])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'csv':
                filepath = f"exports/work_orders_{timestamp}.csv"
                data.to_csv(filepath, index=False)
            elif format == 'excel':
                filepath = f"exports/work_orders_{timestamp}.xlsx"
                data.to_excel(filepath, index=False, engine='openpyxl')
            else:
                filepath = f"exports/work_orders_{timestamp}.json"
                data.to_json(filepath, orient='records', date_format='iso')
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting work orders: {str(e)}")
            return None
    
    def generate_report(self, start_date: str, end_date: str, 
                       format: str) -> Optional[str]:
        """Generate comprehensive system report"""
        try:
            # Gather all report data
            report_data = {
                'period': {'start': start_date, 'end': end_date},
                'generated_at': datetime.now().isoformat(),
                'system_metrics': self.calculate_system_metrics(),
                'anomaly_summary': self.get_anomaly_summary(start_date, end_date),
                'maintenance_summary': self.get_maintenance_summary(start_date, end_date),
                'model_performance': model_evaluator.get_all_performance(),
                'equipment_health': self.get_equipment_health_summary()
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'json':
                filepath = f"reports/system_report_{timestamp}.json"
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            elif format == 'html':
                filepath = f"reports/system_report_{timestamp}.html"
                html_content = self.generate_html_report(report_data)
                with open(filepath, 'w') as f:
                    f.write(html_content)
            else:
                # Default to PDF using reportlab or similar
                filepath = f"reports/system_report_{timestamp}.pdf"
                self.generate_pdf_report(report_data, filepath)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
    
    def get_anomaly_summary(self, start_date: str, end_date: str) -> Dict:
        """Get anomaly summary for reporting"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_anomalies,
                    AVG(anomaly_score) as avg_score,
                    MAX(anomaly_score) as max_score,
                    COUNT(DISTINCT equipment_id) as affected_equipment,
                    COUNT(CASE WHEN anomaly_score > 0.95 THEN 1 END) as critical_count,
                    COUNT(CASE WHEN anomaly_score > 0.85 THEN 1 END) as high_count
                FROM anomalies
                WHERE created_at BETWEEN %s AND %s
            """
            
            result = db_manager.execute_query(query, [start_date, end_date])
            
            return result.iloc[0].to_dict() if not result.empty else {}
            
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {str(e)}")
            return {}
    
    def get_maintenance_summary(self, start_date: str, end_date: str) -> Dict:
        """Get maintenance summary for reporting"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_work_orders,
                    COUNT(CASE WHEN status = 'COMPLETED' THEN 1 END) as completed,
                    COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending,
                    AVG(actual_duration) as avg_duration,
                    COUNT(DISTINCT assigned_technician) as technicians_involved
                FROM work_orders
                WHERE created_at BETWEEN %s AND %s
            """
            
            result = db_manager.execute_query(query, [start_date, end_date])
            
            return result.iloc[0].to_dict() if not result.empty else {}
            
        except Exception as e:
            logger.error(f"Error getting maintenance summary: {str(e)}")
            return {}
    
    def get_equipment_health_summary(self) -> List[Dict]:
        """Get equipment health summary for reporting"""
        try:
            query = """
                SELECT 
                    e.equipment_id,
                    e.equipment_name,
                    e.criticality,
                    COUNT(a.anomaly_id) as recent_anomalies,
                    AVG(a.anomaly_score) as avg_anomaly_score,
                    COUNT(wo.work_order_id) as recent_work_orders
                FROM equipment e
                LEFT JOIN anomalies a ON e.equipment_id = a.equipment_id
                    AND a.created_at > NOW() - INTERVAL '30 days'
                LEFT JOIN work_orders wo ON e.equipment_id = wo.equipment_id
                    AND wo.created_at > NOW() - INTERVAL '30 days'
                GROUP BY e.equipment_id, e.equipment_name, e.criticality
                ORDER BY e.criticality DESC, recent_anomalies DESC
            """
            
            result = db_manager.execute_query(query)
            
            return result.to_dict('records') if not result.empty else []
            
        except Exception as e:
            logger.error(f"Error getting equipment health summary: {str(e)}")
            return []
    
    def generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML format report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IoT System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>IoT Predictive Maintenance System Report</h1>
            <p>Period: {report_data['period']['start']} to {report_data['period']['end']}</p>
            <p>Generated: {report_data['generated_at']}</p>
            
            <h2>System Metrics</h2>
            <div>
                {self._format_metrics_html(report_data['system_metrics'])}
            </div>
            
            <h2>Anomaly Summary</h2>
            {self._format_summary_html(report_data['anomaly_summary'])}
            
            <h2>Maintenance Summary</h2>
            {self._format_summary_html(report_data['maintenance_summary'])}
            
            <h2>Model Performance</h2>
            {self._format_performance_html(report_data['model_performance'])}
        </body>
        </html>
        """
        return html
    
    def _format_metrics_html(self, metrics: Dict) -> str:
        """Format metrics for HTML report"""
        html = ""
        for key, value in metrics.items():
            html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        return html
    
    def _format_summary_html(self, summary: Dict) -> str:
        """Format summary data for HTML report"""
        html = "<table>"
        for key, value in summary.items():
            html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
        html += "</table>"
        return html
    
    def _format_performance_html(self, performance: Dict) -> str:
        """Format model performance for HTML report"""
        html = "<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th></tr>"
        for model, metrics in performance.items():
            html += f"""<tr>
                <td>{model}</td>
                <td>{metrics.get('accuracy', 0):.2%}</td>
                <td>{metrics.get('precision', 0):.2%}</td>
                <td>{metrics.get('recall', 0):.2%}</td>
            </tr>"""
        html += "</table>"
        return html
    
    def generate_pdf_report(self, report_data: Dict, filepath: str):
        """Generate PDF report (requires reportlab or similar)"""
        # Placeholder for PDF generation
        # In production, use reportlab or weasyprint
        logger.info(f"PDF report generation not implemented. Data saved to {filepath}.json")
        with open(f"{filepath}.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

# Initialize callbacks when module is imported
def init_callbacks(app):
    """Initialize all dashboard callbacks"""
    return DashboardCallbacks(app)