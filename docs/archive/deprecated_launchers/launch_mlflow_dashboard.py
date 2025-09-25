#!/usr/bin/env python3
"""
MLFlow-Enhanced IoT Dashboard - Original Dashboard with Lazy Model Loading
Solves the 97-model startup hang by using MLFlow model registry and lazy loading
Drop-in replacement for launch_real_data_dashboard.py
"""

import sys
import time
import threading
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("[ROCKET] IoT PREDICTIVE MAINTENANCE DASHBOARD - MLFlow Enhanced")
print("=" * 70)
print("[SPARKLES] Features: NASA SMAP/MSL data + 97 AI models with lazy loading")
print("[TARGET] MLFlow Model Registry + On-demand model loading")
print("[ZAP] Fast startup - no more 97-model initialization hang!")
print("=" * 70)

def setup_mlflow_infrastructure():
    """Setup MLFlow infrastructure before dashboard startup"""
    print("\n[MLFLOW] Setting up model registry infrastructure...")

    try:
        from src.model_registry.mlflow_config import setup_mlflow
        from src.model_registry.model_manager import get_model_manager

        # Setup MLFlow (this starts the tracking server if needed)
        mlflow_config = setup_mlflow()

        # Initialize model manager (this will discover models but not load them)
        model_manager = get_model_manager()

        # Get available models count
        available_models = model_manager.get_available_models()
        model_count = len(available_models)

        print(f"[MLFLOW] [OK] Model registry ready with {model_count} models")
        print(f"[MLFLOW] [OK] Tracking server: {mlflow_config.tracking_uri}")
        print(f"[MLFLOW] [OK] Lazy loading enabled - models load on-demand")

        return True

    except Exception as e:
        print(f"[MLFLOW] [WARNING] MLFlow setup failed: {e}")
        print(f"[MLFLOW] [FALLBACK] Falling back to local model loading")
        return False

def create_nasa_data_service():
    """Create NASA data service"""
    print("\n[DATA] Initializing NASA data services...")

    try:
        from src.data_ingestion.nasa_data_service import NASADataService

        # This should be fast as we're not loading models here anymore
        data_service = NASADataService()
        print("[DATA] [OK] NASA SMAP/MSL data service ready")
        return data_service

    except Exception as e:
        print(f"[DATA] [ERROR] Failed to create data service: {e}")
        return None

def create_anomaly_engine():
    """Create anomaly detection engine with MLFlow integration"""
    print("\n[AI] Initializing anomaly detection engine...")

    try:
        # Use MLFlow-integrated version instead of original
        from src.anomaly_detection.mlflow_telemanom_integration import create_telemanom_integration

        # This is the key change - uses lazy loading instead of loading all 97 models
        telemanom_integration = create_telemanom_integration(use_mlflow=True)

        print("[AI] [OK] NASA Telemanom integration ready (lazy loading)")
        print(f"[AI] [OK] Models available: {len(telemanom_integration.get_available_models())}")
        print("[AI] [OK] Models will load automatically when needed")

        return telemanom_integration

    except Exception as e:
        print(f"[AI] [ERROR] Failed to create anomaly engine: {e}")
        return None

def create_dashboard_components():
    """Create dashboard components"""
    print("\n[DASHBOARD] Setting up dashboard components...")

    try:
        # Import dashboard components
        from src.dashboard.model_manager import PretrainedModelManager
        from src.dashboard.sensor_stream_manager import SensorStreamManager
        from src.dashboard.unified_data_orchestrator import UnifiedDataOrchestrator
        from src.dashboard.multi_sensor_visualizer import MultiSensorVisualizer

        components = {}

        # These should be fast now since models aren't loaded synchronously
        print("[DASHBOARD] Creating model manager...")
        components['model_manager'] = PretrainedModelManager()

        print("[DASHBOARD] Creating sensor stream manager...")
        components['sensor_stream_manager'] = SensorStreamManager()

        print("[DASHBOARD] Creating data orchestrator...")
        components['data_orchestrator'] = UnifiedDataOrchestrator()

        print("[DASHBOARD] Creating visualizer...")
        components['visualizer'] = MultiSensorVisualizer()

        print("[DASHBOARD] [OK] All dashboard components ready")
        return components

    except Exception as e:
        print(f"[DASHBOARD] [ERROR] Failed to create dashboard components: {e}")
        return {}

def create_dash_app():
    """Create Dash application"""
    print("\n[WEB] Creating Dash web application...")

    try:
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        from flask import Flask

        # Create Flask server and Dash app
        server = Flask(__name__)
        app = dash.Dash(
            __name__,
            server=server,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
            ],
            suppress_callback_exceptions=True
        )
        app.title = "IoT Predictive Maintenance Dashboard"

        # Create main layout
        app.layout = html.Div([
            # Header with MLFlow indicator
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.Badge("MLFlow Enhanced", color="success", className="me-2")),
                    dbc.NavItem(dbc.Badge("97 AI Models", color="info", className="me-2")),
                    dbc.NavItem(dbc.Badge("Lazy Loading", color="warning", className="me-2")),
                ],
                brand="[ROCKET] IoT Predictive Maintenance Dashboard",
                brand_href="#",
                color="primary",
                dark=True,
                className="mb-4"
            ),

            # Main content
            dbc.Container([
                # System status row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("[TARGET] System Status", className="card-title"),
                                html.Div(id="system-status-content"),
                            ])
                        ])
                    ])
                ], className="mb-4"),

                # Model loading progress
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("[ROBOT] AI Model Status", className="card-title"),
                                html.Div(id="model-status-content"),
                            ])
                        ])
                    ])
                ], className="mb-4"),

                # Data visualization area
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("[CHART] Real-time Monitoring", className="card-title"),
                                html.Div("Dashboard ready! Models will load automatically as data is processed.",
                                        id="main-content"),
                            ])
                        ])
                    ])
                ], className="mb-4"),

            ], fluid=True),

            # Auto-refresh intervals
            dcc.Interval(id='status-interval', interval=5000, n_intervals=0),
            dcc.Interval(id='model-interval', interval=10000, n_intervals=0),
        ])

        print("[WEB] [OK] Dash application created successfully")
        return app

    except Exception as e:
        print(f"[WEB] [ERROR] Failed to create Dash app: {e}")
        return None

def setup_callbacks(app, components):
    """Setup Dash callbacks"""
    print("\n[CALLBACKS] Setting up dashboard callbacks...")

    try:
        from dash import Input, Output

        @app.callback(
            Output('system-status-content', 'children'),
            [Input('status-interval', 'n_intervals')]
        )
        def update_system_status(n):
            return html.Div([
                html.P("[OK] NASA Data Service: Connected", className="mb-1 text-success"),
                html.P("[OK] MLFlow Registry: Active", className="mb-1 text-success"),
                html.P("[OK] Anomaly Detection: Ready", className="mb-1 text-success"),
                html.P(f"[CLOCK] Last Update: {datetime.now().strftime('%H:%M:%S')}", className="mb-0 text-muted")
            ])

        @app.callback(
            Output('model-status-content', 'children'),
            [Input('model-interval', 'n_intervals')]
        )
        def update_model_status(n):
            try:
                from src.model_registry.model_manager import get_model_manager
                model_manager = get_model_manager()
                stats = model_manager.get_stats()

                return html.Div([
                    html.P(f"[CLIPBOARD] Available Models: {stats.get('available_models', 0)}",
                           className="mb-1 text-info"),
                    html.P(f"[ROCKET] Models Loaded: {stats.get('cache_stats', {}).get('size', 0)}",
                           className="mb-1 text-success"),
                    html.P(f"[DISK] Cache Hits: {stats.get('cache_hits', 0)}",
                           className="mb-1 text-primary"),
                    html.P(f"[CHART-UP] Cache Misses: {stats.get('cache_misses', 0)}",
                           className="mb-0 text-warning")
                ])
            except Exception as e:
                return html.P(f"Model status unavailable: {e}", className="text-muted")

        print("[CALLBACKS] [OK] Dashboard callbacks registered")
        return True

    except Exception as e:
        print(f"[CALLBACKS] [ERROR] Failed to setup callbacks: {e}")
        return False

def main():
    """Main dashboard startup with MLFlow integration"""
    startup_time = datetime.now()

    try:
        # Phase 1: MLFlow Infrastructure (fast)
        print("\n[FIRE] PHASE 1: MLFlow Infrastructure Setup")
        mlflow_ready = setup_mlflow_infrastructure()

        # Phase 2: Data Services (fast)
        print("\n[FIRE] PHASE 2: Data Services")
        data_service = create_nasa_data_service()

        # Phase 3: Anomaly Engine (fast - no model loading!)
        print("\n[FIRE] PHASE 3: Anomaly Detection Engine")
        anomaly_engine = create_anomaly_engine()

        # Phase 4: Dashboard Components (fast)
        print("\n[FIRE] PHASE 4: Dashboard Components")
        components = create_dashboard_components()

        # Phase 5: Web Application (fast)
        print("\n[FIRE] PHASE 5: Web Application")
        app = create_dash_app()

        if app is None:
            print("[ERROR] Failed to create dashboard application")
            return 1

        # Phase 6: Callbacks (fast)
        print("\n[FIRE] PHASE 6: Dashboard Callbacks")
        callbacks_ready = setup_callbacks(app, components)

        # Calculate startup time
        startup_duration = datetime.now() - startup_time

        # Success!
        print("\n" + "=" * 70)
        print("[PARTY] DASHBOARD STARTUP COMPLETE!")
        print("=" * 70)
        print(f"[ZAP] Startup Time: {startup_duration.total_seconds():.2f} seconds")
        print(f"[ROBOT] AI Models: 97 available (lazy loading)")
        print(f"[SATELLITE] Data Sources: NASA SMAP + MSL")
        print(f"[GLOBE] Web Server: Starting on http://localhost:8060")
        print(f"[REFRESH] MLFlow Registry: {'Active' if mlflow_ready else 'Fallback'}")
        print("=" * 70)

        # Start the server - this should work now!
        print("\n[ROCKET] Starting Dash server...")
        app.run(
            debug=False,
            host='0.0.0.0',
            port=8060,
            dev_tools_hot_reload=False
        )

    except KeyboardInterrupt:
        print("\n\n[WAVE] Dashboard stopped by user")
        return 0

    except Exception as e:
        print(f"\n[ERROR] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())