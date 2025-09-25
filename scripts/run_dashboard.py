#!/usr/bin/env python3
"""
Dashboard Launch Script for IoT Anomaly Detection System
Starts the interactive Dash dashboard for monitoring and control
"""

import os
import sys
import argparse
import signal
import threading
import time
import json
import yaml
import queue
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask, send_from_directory
from flask_caching import Cache
import redis

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import dashboard components
from src.dashboard.layouts.overview import create_layout as create_overview_layout
from src.dashboard.layouts.anomaly_monitor import create_layout as create_anomaly_layout
from src.dashboard.layouts.forecast_view import create_layout as create_forecast_layout
from src.dashboard.layouts.maintenance_scheduler import create_layout as create_maintenance_layout
from src.dashboard.layouts.work_orders import create_layout as create_work_orders_layout
from src.dashboard.callbacks.dashboard_callbacks import DashboardCallbacks

# Import with fallback for dependency issues
try:
    from src.data_ingestion.database_manager import DatabaseManager
    database_available = True
except ImportError as e:
    print(f"Database manager not available: {e}")
    database_available = False
    DatabaseManager = None

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

try:
    from config.settings import Settings
except ImportError as e:
    logger.warning(f"Settings not available: {e}")
    class MockSettings:
        def __init__(self, config_path=None):
            self.database = {'host': 'localhost', 'port': 5432, 'name': 'iot_db', 'user': 'user', 'password': 'pass'}
    Settings = MockSettings

# Logger is already initialized above in the try/except block

class IoTDashboard:
    """Main dashboard application for IoT anomaly detection system"""
    
    def __init__(self, config_path: str = None, debug: bool = False, port: int = 8050):
        """Initialize dashboard
        
        Args:
            config_path: Path to configuration file
            debug: Debug mode flag
            port: Port to run dashboard on
        """
        self.settings = Settings()
        # Settings is a singleton, but we can still pass config_path to __init__
        if config_path:
            self.settings.__init__(config_path)
        self.debug = debug
        self.port = port
        
        # Initialize Flask server
        self.server = Flask(__name__)
        self.server.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ],
            suppress_callback_exceptions=True,
            meta_tags=[
                {"name": "viewport", "content": "width=device-width, initial-scale=1"}
            ]
        )
        
        # Set app title
        self.app.title = "IoT Predictive Maintenance Dashboard"
        
        # Initialize cache
        self._setup_cache()
        
        # Initialize database connection
        self._setup_database()
        
        # Setup layout
        self._setup_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        # Setup custom routes
        self._setup_routes()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize real-time data monitoring
        self._init_realtime_monitoring()

        logger.info(f"Dashboard initialized on port {self.port}")

    def _init_realtime_monitoring(self):
        """Initialize real-time data monitoring"""
        self.last_data_timestamps = {
            'telemetry': datetime.now() - timedelta(hours=1),
            'anomalies': datetime.now() - timedelta(hours=1),
            'work_orders': datetime.now() - timedelta(hours=1)
        }

        # Data update queues for real-time streaming
        self.data_update_queue = queue.Queue(maxsize=1000)
        self.websocket_clients = set()

        # Start background data monitoring thread
        if hasattr(self, 'database_connected') and self.database_connected:
            self.monitoring_thread = threading.Thread(
                target=self._monitor_database_changes,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Real-time data monitoring started")

    def _monitor_database_changes(self):
        """Monitor database for new data and stream to dashboard"""
        while True:
            try:
                # Check for new telemetry data
                new_telemetry = self._check_for_new_data('telemetry_data', 'timestamp', 'telemetry')
                if new_telemetry:
                    # Limit data size for performance
                    limited_telemetry = new_telemetry[:20] if len(new_telemetry) > 20 else new_telemetry
                    if not self.data_update_queue.full():
                        self.data_update_queue.put({
                            'type': 'telemetry',
                            'data': limited_telemetry,
                            'count': len(new_telemetry),
                            'timestamp': datetime.now().isoformat()
                        })

                # Check for new anomalies
                new_anomalies = self._check_for_new_data('anomalies', 'detected_at', 'anomalies')
                if new_anomalies:
                    self.data_update_queue.put({
                        'type': 'anomalies',
                        'data': new_anomalies,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Send alert if critical anomalies
                    critical_anomalies = [a for a in new_anomalies
                                        if a.get('severity') in ['CRITICAL', 'HIGH']]
                    if critical_anomalies:
                        self.data_update_queue.put({
                            'type': 'alerts',
                            'data': critical_anomalies,
                            'timestamp': datetime.now().isoformat()
                        })

                # Check for new work orders
                new_work_orders = self._check_for_new_data('work_orders', 'created_at', 'work_orders')
                if new_work_orders:
                    self.data_update_queue.put({
                        'type': 'work_orders',
                        'data': new_work_orders,
                        'timestamp': datetime.now().isoformat()
                    })

                # Dynamic sleep based on data activity
                if new_telemetry or new_anomalies or new_work_orders:
                    # Faster checking when data is actively flowing
                    time.sleep(1)
                else:
                    # Slower checking when no new data
                    time.sleep(5)

            except Exception as e:
                logger.error(f"Error monitoring database changes: {e}")
                time.sleep(5)  # Wait longer on error

    def _check_for_new_data(self, table_name, timestamp_column, data_type):
        """Check for new data in specified table"""
        try:
            last_timestamp = self.last_data_timestamps[data_type]

            if self.database_connected:
                query = f"""
                    SELECT * FROM {table_name}
                    WHERE {timestamp_column} > %s
                    ORDER BY {timestamp_column} DESC
                    LIMIT 50
                """
                result = self.db_manager.execute_query(query, [last_timestamp])
            else:
                # Return empty for demo mode (no new data simulation)
                return []

            if not result.empty:
                # Update last timestamp
                self.last_data_timestamps[data_type] = result[timestamp_column].max()
                return result.to_dict('records')

            return []

        except Exception as e:
            logger.error(f"Error checking for new {data_type}: {e}")
            return []
    
    def _setup_cache(self):
        """Setup caching for performance"""
        # Try Redis cache first, fallback to simple cache
        try:
            cache_config = {
                'CACHE_TYPE': 'redis',
                'CACHE_REDIS_HOST': self.settings.get('redis_host', 'localhost'),
                'CACHE_REDIS_PORT': self.settings.get('redis_port', 6379),
                'CACHE_REDIS_DB': self.settings.get('redis_db', 0),
                'CACHE_DEFAULT_TIMEOUT': 300
            }
            
            # Test Redis connection
            r = redis.Redis(
                host=cache_config['CACHE_REDIS_HOST'],
                port=cache_config['CACHE_REDIS_PORT'],
                db=cache_config['CACHE_REDIS_DB']
            )
            r.ping()
            
            logger.info("Using Redis cache")
            
        except:
            # Fallback to simple cache
            cache_config = {
                'CACHE_TYPE': 'simple',
                'CACHE_DEFAULT_TIMEOUT': 300
            }
            logger.info("Using simple cache (Redis not available)")
        
        self.cache = Cache(self.server, config=cache_config)
    
    def _setup_database(self):
        """Setup database connection with fallback"""
        if not database_available or DatabaseManager is None:
            logger.info("Database manager not available, running in demo mode")
            self.database_connected = False
            self.db_manager = self._create_mock_database()
            return

        try:
            # Try to establish database connection
            # Get database config from settings
            db_config = getattr(self.settings, 'database', None)
            if not db_config:
                # Try to get from _config if available
                db_config = getattr(self.settings, '_config', {}).get('database', {
                    'host': 'localhost', 'port': 5432, 'name': 'iot_db',
                    'user': 'user', 'password': 'pass'
                })

            self.db_manager = DatabaseManager(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('name', 'iot_db'),
                user=db_config.get('user', 'user'),
                password=db_config.get('password', 'pass')
            )

            # Test the connection
            test_query = "SELECT 1 as test"
            result = self.db_manager.execute_query(test_query)
            if result is not None:
                logger.info("Database connection established and tested successfully")
                self.database_connected = True
            else:
                raise Exception("Database connection test failed")

        except Exception as e:
            logger.warning(f"Database connection failed: {str(e)}")
            logger.info("Running dashboard in demo mode with sample data")
            self.database_connected = False
            self.db_manager = self._create_mock_database()

    def _create_mock_database(self):
        """Create mock database manager for demo mode"""
        class MockDatabaseManager:
            def execute_query(self, query, params=None):
                """Return sample data for demo mode"""
                import pandas as pd

                if 'telemetry_data' in query.lower():
                    # Return sample telemetry data
                    data = {
                        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
                        'equipment_id': ['PUMP_001'] * 100,
                        'sensor_data': [{'temp': 25.5, 'pressure': 101.3}] * 100
                    }
                    return pd.DataFrame(data)

                elif 'anomalies' in query.lower():
                    # Return sample anomaly data
                    data = {
                        'anomaly_id': range(1, 11),
                        'equipment_id': ['PUMP_001'] * 10,
                        'anomaly_score': [0.85, 0.92, 0.78, 0.95, 0.81, 0.89, 0.76, 0.93, 0.87, 0.82],
                        'detected_at': pd.date_range(start='2024-01-01', periods=10, freq='H'),
                        'severity': ['HIGH', 'CRITICAL', 'MEDIUM', 'CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'CRITICAL', 'HIGH', 'HIGH']
                    }
                    return pd.DataFrame(data)

                elif 'work_orders' in query.lower():
                    # Return sample work order data
                    data = {
                        'work_order_id': ['WO-001', 'WO-002', 'WO-003'],
                        'equipment_id': ['PUMP_001', 'PUMP_002', 'PUMP_001'],
                        'status': ['PENDING', 'IN_PROGRESS', 'COMPLETED'],
                        'priority': ['HIGH', 'MEDIUM', 'LOW'],
                        'created_at': pd.date_range(start='2024-01-01', periods=3, freq='D')
                    }
                    return pd.DataFrame(data)

                else:
                    # Return empty DataFrame for unknown queries
                    return pd.DataFrame()

            def execute(self, query, params=None):
                """Mock execute method"""
                pass

            def close(self):
                """Mock close method"""
                pass

        return MockDatabaseManager()
    
    def _setup_layout(self):
        """Setup main dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-industry fa-2x text-primary"),
                        html.H1("IoT Predictive Maintenance", 
                               className="d-inline-block ml-3 mb-0"),
                    ], className="d-flex align-items-center"),
                    
                    html.Div([
                        html.Span(id="connection-status", className="badge badge-success mr-3"),
                        html.Span(id="current-time", className="text-muted mr-3"),
                        html.Button(
                            html.I(className="fas fa-cog"),
                            id="settings-btn",
                            className="btn btn-sm btn-outline-secondary"
                        )
                    ], className="d-flex align-items-center")
                ], className="d-flex justify-content-between align-items-center")
            ], className="header-bar bg-white shadow-sm p-3 mb-4"),
            
            # Navigation Tabs
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Overview", href="/", id="overview-link", active=True)),
                dbc.NavItem(dbc.NavLink("Anomaly Monitor", href="/anomalies", id="anomaly-link")),
                dbc.NavItem(dbc.NavLink("Forecasting", href="/forecast", id="forecast-link")),
                dbc.NavItem(dbc.NavLink("Maintenance", href="/maintenance", id="maintenance-link")),
                dbc.NavItem(dbc.NavLink("Work Orders", href="/work-orders", id="work-orders-link")),
            ], pills=True, className="mb-4 justify-content-center"),
            
            # Alert Banner (hidden by default)
            html.Div(id="alert-banner", className="alert-banner", style={'display': 'none'}),
            
            # URL for routing
            dcc.Location(id='url', refresh=False),
            
            # Main content area
            html.Div(id='page-content', className="container-fluid"),
            
            # Intervals for real-time updates
            dcc.Interval(id='main-interval', interval=5000),  # 5 seconds
            dcc.Interval(id='slow-interval', interval=30000),  # 30 seconds
            dcc.Interval(id='clock-interval', interval=1000),  # 1 second
            
            # Hidden stores for shared data
            dcc.Store(id='shared-data-store', storage_type='session'),
            dcc.Store(id='user-settings-store', storage_type='local'),
            dcc.Store(id='cache-store', storage_type='memory'),
            
            # Settings Modal
            dbc.Modal([
                dbc.ModalHeader("Dashboard Settings"),
                dbc.ModalBody([
                    html.H5("Data Refresh"),
                    dbc.Form([
                        html.Div([  # Replace FormGroup with Div
                            dbc.Label("Refresh Interval (seconds)"),
                            dbc.Input(
                                id="refresh-interval-input",
                                type="number",
                                value=5,
                                min=1,
                                max=60
                            )
                        ]),
                        html.Div([  # Replace FormGroup with Div
                            dbc.Label("Theme"),
                            dbc.Select(
                                id="theme-select",
                                options=[
                                    {"label": "Light", "value": "light"},
                                    {"label": "Dark", "value": "dark"}
                                ],
                                value="light"
                            )
                        ]),
                        html.Div([  # Replace FormGroup with Div
                            dbc.Label("Notifications"),
                            dbc.Checklist(
                                id="notification-settings",
                                options=[
                                    {"label": "Enable Sound Alerts", "value": "sound"},
                                    {"label": "Enable Desktop Notifications", "value": "desktop"},
                                    {"label": "Enable Email Alerts", "value": "email"}
                                ],
                                value=["sound"]
                            )
                        ])
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button("Save", id="save-settings-btn", color="primary"),
                    dbc.Button("Cancel", id="cancel-settings-btn", color="secondary")
                ])
            ], id="settings-modal", is_open=False),
            
            # Load custom CSS
            html.Link(
                rel='stylesheet',
                href='/assets/style.css'
            ),

            # Real-time updates JavaScript
            html.Script("""
                // Real-time updates via Server-Sent Events
                let eventSource = null;
                let reconnectInterval = 5000; // 5 seconds

                function initRealTimeUpdates() {
                    try {
                        eventSource = new EventSource('/api/realtime-updates');

                        eventSource.onopen = function(event) {
                            console.log('Real-time connection established');
                            updateConnectionIndicator('connected');
                        };

                        eventSource.onmessage = function(event) {
                            try {
                                const data = JSON.parse(event.data);
                                handleRealTimeUpdate(data);
                            } catch (e) {
                                console.error('Error parsing real-time data:', e);
                            }
                        };

                        eventSource.onerror = function(event) {
                            console.error('Real-time connection error:', event);
                            updateConnectionIndicator('error');

                            // Attempt to reconnect
                            setTimeout(() => {
                                if (eventSource.readyState === EventSource.CLOSED) {
                                    console.log('Attempting to reconnect...');
                                    initRealTimeUpdates();
                                }
                            }, reconnectInterval);
                        };

                    } catch (e) {
                        console.error('Failed to initialize real-time updates:', e);
                        setTimeout(initRealTimeUpdates, reconnectInterval);
                    }
                }

                function handleRealTimeUpdate(data) {
                    // Handle different types of real-time updates
                    switch(data.type) {
                        case 'telemetry':
                            console.log('New telemetry data:', data.data.length, 'records');
                            // Trigger dashboard refresh or update specific components
                            triggerComponentUpdate('telemetry', data.data);
                            break;

                        case 'anomalies':
                            console.log('New anomalies detected:', data.data.length);
                            triggerComponentUpdate('anomalies', data.data);
                            showAnomalyAlert(data.data);
                            break;

                        case 'alerts':
                            console.log('Critical alerts:', data.data.length);
                            showCriticalAlert(data.data);
                            break;

                        case 'work_orders':
                            console.log('New work orders:', data.data.length);
                            triggerComponentUpdate('work_orders', data.data);
                            break;

                        case 'heartbeat':
                            // Update last seen timestamp
                            updateConnectionIndicator('alive');
                            break;

                        default:
                            console.log('Unknown update type:', data.type);
                    }
                }

                function triggerComponentUpdate(component, data) {
                    // Force Dash components to refresh by updating store
                    const store = document.getElementById('shared-data-store');
                    if (store) {
                        const currentData = JSON.parse(store.textContent || '{}');
                        currentData[component + '_updates'] = {
                            data: data,
                            timestamp: new Date().toISOString()
                        };
                        store.textContent = JSON.stringify(currentData);

                        // Trigger a synthetic event to force Dash update
                        const event = new Event('change', { bubbles: true });
                        store.dispatchEvent(event);
                    }
                }

                function showAnomalyAlert(anomalies) {
                    // Show visual alert for new anomalies
                    const alertBanner = document.getElementById('alert-banner');
                    if (alertBanner && anomalies.length > 0) {
                        const highSeverity = anomalies.filter(a => a.severity === 'HIGH' || a.severity === 'CRITICAL');
                        if (highSeverity.length > 0) {
                            alertBanner.innerHTML = `
                                <div class="alert alert-warning alert-dismissible fade show" role="alert">
                                    <strong>New Anomaly Detected!</strong> ${highSeverity.length} high-severity anomalies require attention.
                                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                                </div>
                            `;
                            alertBanner.style.display = 'block';
                        }
                    }
                }

                function showCriticalAlert(alerts) {
                    // Show critical alert banner
                    const alertBanner = document.getElementById('alert-banner');
                    if (alertBanner && alerts.length > 0) {
                        alertBanner.innerHTML = `
                            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                                <strong>CRITICAL ALERT!</strong> ${alerts.length} critical anomalies detected. Immediate action required.
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        `;
                        alertBanner.style.display = 'block';
                    }
                }

                function updateConnectionIndicator(status) {
                    const indicator = document.getElementById('connection-status');
                    if (indicator) {
                        switch(status) {
                            case 'connected':
                                indicator.className = 'badge badge-success';
                                indicator.textContent = 'Live Updates Active';
                                break;
                            case 'error':
                                indicator.className = 'badge badge-warning';
                                indicator.textContent = 'Connection Issues';
                                break;
                            case 'alive':
                                indicator.className = 'badge badge-success';
                                // Keep current text but update timestamp
                                break;
                        }
                    }
                }

                // Initialize when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('Initializing real-time updates...');
                    initRealTimeUpdates();
                });

                // Cleanup on page unload
                window.addEventListener('beforeunload', function() {
                    if (eventSource) {
                        eventSource.close();
                    }
                });
            """, type="text/javascript")
        ])
    
    def _register_callbacks(self):
        """Register all dashboard callbacks"""
        
        # Initialize callbacks manager
        self.callbacks = DashboardCallbacks(self.app)
        
        # Page routing callback
        @self.app.callback(
            [Output('page-content', 'children'),
             Output('overview-link', 'active'),
             Output('anomaly-link', 'active'),
             Output('forecast-link', 'active'),
             Output('maintenance-link', 'active'),
             Output('work-orders-link', 'active')],
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            """Route to appropriate page based on URL"""
            # Reset all active states
            active_states = [False] * 5
            
            if pathname == '/' or pathname == '/overview':
                layout = create_overview_layout()
                active_states[0] = True
            elif pathname == '/anomalies':
                layout = create_anomaly_layout()
                active_states[1] = True
            elif pathname == '/forecast':
                layout = create_forecast_layout()
                active_states[2] = True
            elif pathname == '/maintenance':
                layout = create_maintenance_layout()
                active_states[3] = True
            elif pathname == '/work-orders':
                layout = create_work_orders_layout()
                active_states[4] = True
            else:
                layout = html.Div([
                    html.H1("404: Page not found", className="text-center mt-5"),
                    html.P(f"The pathname {pathname} was not recognized."),
                    html.A("Go to Overview", href="/", className="btn btn-primary")
                ])
                active_states[0] = True
            
            return [layout] + active_states
        
        # Clock update callback
        @self.app.callback(
            Output('current-time', 'children'),
            [Input('clock-interval', 'n_intervals')]
        )
        def update_clock(n):
            """Update current time display"""
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Connection status callback
        @self.app.callback(
            Output('connection-status', 'children'),
            Output('connection-status', 'className'),
            [Input('slow-interval', 'n_intervals')]
        )
        @self.cache.memoize(timeout=30)
        def update_connection_status(n):
            """Check and update connection status"""
            try:
                if hasattr(self, 'database_connected') and self.database_connected:
                    # Test database connection
                    test_result = self.db_manager.execute_query("SELECT 1 as test")
                    if test_result is not None:
                        return "Database Connected", "badge badge-success"
                    else:
                        return "Database Error", "badge badge-warning"
                elif hasattr(self, 'database_connected') and not self.database_connected:
                    return "Demo Mode", "badge badge-info"
                else:
                    return "Initializing", "badge badge-secondary"
            except Exception as e:
                logger.error(f"Connection status check failed: {e}")
                return "Connection Error", "badge badge-warning"
        
        # Settings modal callbacks
        @self.app.callback(
            Output('settings-modal', 'is_open'),
            [Input('settings-btn', 'n_clicks'),
             Input('save-settings-btn', 'n_clicks'),
             Input('cancel-settings-btn', 'n_clicks')],
            [State('settings-modal', 'is_open')]
        )
        def toggle_settings_modal(settings_clicks, save_clicks, cancel_clicks, is_open):
            """Toggle settings modal"""
            ctx = callback_context
            if not ctx.triggered:
                return is_open
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'settings-btn':
                return not is_open
            elif trigger_id in ['save-settings-btn', 'cancel-settings-btn']:
                return False
            
            return is_open
        
        # Save settings callback
        @self.app.callback(
            [Output('main-interval', 'interval'),
             Output('user-settings-store', 'data')],
            [Input('save-settings-btn', 'n_clicks')],
            [State('refresh-interval-input', 'value'),
             State('theme-select', 'value'),
             State('notification-settings', 'value'),
             State('user-settings-store', 'data')]
        )
        def save_settings(n_clicks, refresh_interval, theme, notifications, current_settings):
            """Save user settings"""
            if not n_clicks:
                raise PreventUpdate
            
            # Update settings
            settings = current_settings or {}
            settings.update({
                'refresh_interval': refresh_interval * 1000,  # Convert to milliseconds
                'theme': theme,
                'notifications': notifications
            })
            
            logger.info(f"Settings updated: {settings}")
            
            return settings['refresh_interval'], settings
        
        # Real-time data update callback (cached)
        @self.app.callback(
            Output('cache-store', 'data'),
            [Input('main-interval', 'n_intervals')]
        )
        @self.cache.memoize(timeout=5)
        def update_cache_data(n):
            """Update cached data for all components"""
            try:
                data = {}

                if self.db_manager:
                    if hasattr(self, 'database_connected') and self.database_connected:
                        # Real database queries
                        telemetry_query = """
                            SELECT * FROM telemetry_data
                            WHERE timestamp > NOW() - INTERVAL '1 hour'
                            ORDER BY timestamp DESC
                            LIMIT 1000
                        """
                        anomaly_query = """
                            SELECT * FROM anomalies
                            WHERE detected_at > NOW() - INTERVAL '24 hours'
                            ORDER BY detected_at DESC
                            LIMIT 100
                        """
                        wo_stats_query = """
                            SELECT status, COUNT(*) as count
                            FROM work_orders
                            WHERE created_at > NOW() - INTERVAL '7 days'
                            GROUP BY status
                        """
                    else:
                        # Demo mode queries (mock data will be returned)
                        telemetry_query = "SELECT * FROM telemetry_data LIMIT 100"
                        anomaly_query = "SELECT * FROM anomalies LIMIT 10"
                        wo_stats_query = "SELECT status, COUNT(*) as count FROM work_orders GROUP BY status"

                    # Execute queries (will return real or mock data)
                    telemetry_df = self.db_manager.execute_query(telemetry_query)
                    data['telemetry'] = telemetry_df.to_dict('records') if not telemetry_df.empty else []

                    anomaly_df = self.db_manager.execute_query(anomaly_query)
                    data['anomalies'] = anomaly_df.to_dict('records') if not anomaly_df.empty else []

                    wo_stats_df = self.db_manager.execute_query(wo_stats_query)
                    data['work_order_stats'] = wo_stats_df.to_dict('records') if not wo_stats_df.empty else []

                    # Calculate system metrics
                    total_anomalies = len(data['anomalies'])
                    pending_work_orders = sum(
                        stat['count'] for stat in data['work_order_stats']
                        if stat.get('status') == 'PENDING'
                    )

                    data['metrics'] = {
                        'total_equipment': 25,
                        'active_anomalies': total_anomalies,
                        'pending_work_orders': pending_work_orders,
                        'system_health': max(50, 100 - (total_anomalies * 5) - (pending_work_orders * 2)),
                        'database_status': 'connected' if self.database_connected else 'demo_mode'
                    }

                else:
                    # No database manager available
                    data = {
                        'telemetry': [],
                        'anomalies': [],
                        'work_order_stats': [],
                        'metrics': {
                            'total_equipment': 0,
                            'active_anomalies': 0,
                            'pending_work_orders': 0,
                            'system_health': 0,
                            'database_status': 'disconnected'
                        }
                    }

                data['timestamp'] = datetime.now().isoformat()
                return data

            except Exception as e:
                logger.error(f"Error updating cache: {str(e)}")
                return {
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'total_equipment': 0,
                        'active_anomalies': 0,
                        'pending_work_orders': 0,
                        'system_health': 0,
                        'database_status': 'error'
                    },
                    'telemetry': [],
                    'anomalies': [],
                    'work_order_stats': []
                }
    
    def _setup_routes(self):
        """Setup custom Flask routes"""
        
        @self.server.route('/assets/<path:path>')
        def serve_assets(path):
            """Serve static assets"""
            assets_folder = os.path.join(os.path.dirname(__file__), '..', 'src', 'dashboard', 'assets')
            return send_from_directory(assets_folder, path)
        
        @self.server.route('/api/health')
        def health_check():
            """Health check endpoint"""
            db_status = 'disconnected'
            if hasattr(self, 'database_connected'):
                db_status = 'connected' if self.database_connected else 'demo_mode'
            elif self.db_manager:
                db_status = 'unknown'

            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'database': db_status,
                'mode': 'production' if db_status == 'connected' else 'demo'
            }
        
        @self.server.route('/api/metrics')
        @self.cache.cached(timeout=30)
        def get_metrics():
            """Get current system metrics"""
            try:
                metrics = {}
                
                if self.db_manager:
                    # Get various metrics from database
                    queries = {
                        'total_telemetry': "SELECT COUNT(*) FROM telemetry_data WHERE timestamp > NOW() - INTERVAL '1 hour'",
                        'total_anomalies': "SELECT COUNT(*) FROM anomalies WHERE detected_at > NOW() - INTERVAL '24 hours'",
                        'active_work_orders': "SELECT COUNT(*) FROM work_orders WHERE status IN ('PENDING', 'IN_PROGRESS')",
                        'total_alerts': "SELECT COUNT(*) FROM alerts WHERE sent_at > NOW() - INTERVAL '24 hours'"
                    }
                    
                    for key, query in queries.items():
                        result = self.db_manager.execute_query(query)
                        metrics[key] = result.iloc[0, 0] if not result.empty else 0
                
                return metrics
                
            except Exception as e:
                logger.error(f"Error getting metrics: {str(e)}")
                return {}
        
        @self.server.route('/api/export/<data_type>')
        def export_data(data_type):
            """Export data endpoint"""
            try:
                if data_type == 'anomalies':
                    query = "SELECT * FROM anomalies ORDER BY detected_at DESC LIMIT 10000"
                elif data_type == 'work_orders':
                    query = "SELECT * FROM work_orders ORDER BY created_at DESC LIMIT 10000"
                elif data_type == 'telemetry':
                    query = "SELECT * FROM telemetry_data ORDER BY timestamp DESC LIMIT 50000"
                else:
                    return {'error': 'Invalid data type'}, 400
                
                if self.db_manager:
                    df = self.db_manager.execute_query(query)
                    
                    # Return as CSV
                    response = df.to_csv(index=False)
                    return response, 200, {
                        'Content-Type': 'text/csv',
                        'Content-Disposition': f'attachment; filename={data_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    }
                else:
                    return {'error': 'Database not connected'}, 500
                    
            except Exception as e:
                logger.error(f"Error exporting data: {str(e)}")
                return {'error': str(e)}, 500

        @self.server.route('/api/realtime-updates')
        def realtime_updates():
            """Server-Sent Events endpoint for real-time updates"""
            def event_stream():
                """Generator function for streaming events"""
                heartbeat_counter = 0
                max_heartbeat_interval = 30  # seconds

                while True:
                    try:
                        updates_sent = 0
                        # Process multiple updates if available (batch processing)
                        while not self.data_update_queue.empty() and updates_sent < 10:
                            try:
                                update = self.data_update_queue.get_nowait()
                                yield f"data: {json.dumps(update)}\n\n"
                                updates_sent += 1
                            except queue.Empty:
                                break

                        if updates_sent > 0:
                            heartbeat_counter = 0  # Reset heartbeat counter
                        else:
                            # Send heartbeat periodically when no updates
                            heartbeat_counter += 1
                            if heartbeat_counter >= max_heartbeat_interval:
                                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                                heartbeat_counter = 0

                        # Short sleep to prevent CPU overload
                        time.sleep(1)

                    except GeneratorExit:
                        logger.info("Client disconnected from event stream")
                        break
                    except Exception as e:
                        logger.error(f"Error in event stream: {e}")
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Stream error occurred'})}\n\n"
                        break

            from flask import Response
            return Response(
                event_stream(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )

        @self.server.route('/api/pipeline-status')
        def pipeline_status():
            """Get current pipeline status"""
            try:
                # Check if pipeline is running by looking at recent data
                recent_telemetry = self._check_for_new_data('telemetry_data', 'timestamp', 'telemetry')
                pipeline_active = len(recent_telemetry) > 0

                status = {
                    'pipeline_active': pipeline_active,
                    'database_connected': self.database_connected,
                    'last_data_update': max(self.last_data_timestamps.values()).isoformat() if self.last_data_timestamps else None,
                    'queue_size': self.data_update_queue.qsize(),
                    'timestamp': datetime.now().isoformat()
                }

                return status

            except Exception as e:
                logger.error(f"Error getting pipeline status: {e}")
                return {
                    'pipeline_active': False,
                    'database_connected': self.database_connected if hasattr(self, 'database_connected') else False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }, 500
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down dashboard...")

        # Stop monitoring thread
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            logger.info("Stopping monitoring thread...")
            # The thread will stop when the main process exits since it's a daemon thread

        # Close database connection
        if self.db_manager:
            self.db_manager.close()

        logger.info("Dashboard shutdown complete")
        sys.exit(0)
    
    def run(self):
        """Run the dashboard"""
        logger.info(f"""
        ╔══════════════════════════════════════════════════════╗
        ║     IoT Predictive Maintenance Dashboard            ║
        ║                                                      ║
        ║     Access the dashboard at:                        ║
        ║     http://localhost:{self.port}                         ║
        ║                                                      ║
        ║     Press Ctrl+C to stop                           ║
        ╚══════════════════════════════════════════════════════╝
        """)
        
        # Run the server
        self.app.run(
            debug=self.debug,
            host='0.0.0.0',
            port=self.port,
            dev_tools_hot_reload=self.debug
        )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run IoT Anomaly Detection Dashboard')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to run dashboard on')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo data (no database required)')
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.debug:
        os.environ['FLASK_ENV'] = 'development'
    else:
        os.environ['FLASK_ENV'] = 'production'
    
    if args.no_cache:
        os.environ['CACHE_TYPE'] = 'null'
    
    # Initialize and run dashboard
    try:
        dashboard = IoTDashboard(
            config_path=args.config,
            debug=args.debug,
            port=args.port
        )
        
        if args.demo:
            logger.info("Running in demo mode with sample data")
            # Could initialize with demo data here
        
        dashboard.run()
        
    except KeyboardInterrupt:
        logger.info("\nDashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()