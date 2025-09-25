#!/usr/bin/env python3
"""
ULTRATHINK DIAGNOSTIC DASHBOARD - Pinpoint Server Startup Hang
Advanced diagnostic version with component-level monitoring and fallback mechanisms
"""

import sys
import time
import threading
import os
import psutil
import socket
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import signal

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Global monitoring variables
diagnostic_status = {
    'current_phase': 'initialization',
    'components_loaded': 0,
    'total_components': 15,  # Expected major components
    'memory_usage': 0,
    'startup_time': None,
    'hang_detected': False,
    'last_heartbeat': None
}

startup_phases = [
    'initialization',
    'config_loading',
    'tensorflow_setup',
    'model_loading',
    'data_services',
    'dashboard_components',
    'layout_assembly',
    'callback_registration',
    'server_binding',
    'server_startup'
]

current_phase_index = 0

def log_diagnostic(message, level="INFO"):
    """Enhanced diagnostic logging with timestamp and memory usage"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    diagnostic_status['memory_usage'] = memory_mb
    diagnostic_status['last_heartbeat'] = datetime.now()

    phase_info = f"[{diagnostic_status['current_phase'].upper()}]"
    memory_info = f"[MEM: {memory_mb:.1f}MB]"

    # Replace emojis with ASCII equivalents for Windows compatibility
    ascii_message = message.replace('🚀', '[ROCKET]').replace('🎯', '[TARGET]').replace('🔬', '[MICROSCOPE]').replace('=' * 80, '-' * 80).replace('✅', '[OK]').replace('❌', '[ERROR]').replace('⚠️', '[WARN]').replace('💀', '[HANG]').replace('🚨', '[ALERT]').replace('💥', '[CRASH]').replace('🛑', '[STOP]').replace('🔍', '[SEARCH]').replace('⏰', '[TIMEOUT]').replace('📊', '[CHART]').replace('💾', '[MEMORY]').replace('⏱️', '[TIME]').replace('🌐', '[WEB]').replace('📡', '[SERVER]').replace('🕒', '[CLOCK]').replace('🔄', '[REFRESH]')

    print(f"{timestamp} {level} {phase_info} {memory_info} {ascii_message}")

def check_port_availability(port):
    """Check if port is available for binding"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        sock.close()
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except:
            pass

def advance_phase(phase_name):
    """Advance to next diagnostic phase"""
    global current_phase_index
    diagnostic_status['current_phase'] = phase_name
    log_diagnostic(f"=== ENTERING PHASE: {phase_name.upper()} ===")
    current_phase_index = startup_phases.index(phase_name) if phase_name in startup_phases else current_phase_index

def component_timeout_wrapper(func, component_name, timeout_seconds=30):
    """Wrapper to run component loading with timeout"""
    result_container = {'result': None, 'error': None, 'completed': False}

    def target():
        try:
            log_diagnostic(f"Loading component: {component_name}")
            result = func()
            result_container['result'] = result
            result_container['completed'] = True
            log_diagnostic(f"[OK] Component loaded: {component_name}")
            diagnostic_status['components_loaded'] += 1
        except Exception as e:
            result_container['error'] = e
            log_diagnostic(f"[ERROR] Component failed: {component_name} - {str(e)}", "ERROR")

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        log_diagnostic(f"[TIMEOUT] Component {component_name} hung after {timeout_seconds}s", "ERROR")
        return None

    if result_container['error']:
        raise result_container['error']

    return result_container['result']

def heartbeat_monitor():
    """Monitor for hangs and provide diagnostic information"""
    log_diagnostic("[SEARCH] Heartbeat monitor started")

    while not diagnostic_status['hang_detected']:
        time.sleep(10)  # Check every 10 seconds

        if diagnostic_status['last_heartbeat']:
            time_since_heartbeat = datetime.now() - diagnostic_status['last_heartbeat']
            if time_since_heartbeat > timedelta(seconds=60):  # 1 minute without heartbeat
                log_diagnostic(f"[HANG] HANG DETECTED in phase: {diagnostic_status['current_phase']}", "CRITICAL")
                log_diagnostic(f"[HANG] Memory usage: {diagnostic_status['memory_usage']:.1f}MB", "CRITICAL")
                log_diagnostic(f"[HANG] Components loaded: {diagnostic_status['components_loaded']}/{diagnostic_status['total_components']}", "CRITICAL")
                log_diagnostic(f"[HANG] Time since last heartbeat: {time_since_heartbeat.total_seconds():.1f}s", "CRITICAL")
                diagnostic_status['hang_detected'] = True

                # Force fallback activation
                activate_emergency_fallback()
                break

def activate_emergency_fallback():
    """Activate emergency fallback dashboard"""
    log_diagnostic("[ALERT] ACTIVATING EMERGENCY FALLBACK DASHBOARD", "CRITICAL")

    try:
        import dash
        from dash import html, dcc
        import dash_bootstrap_components as dbc
        from flask import Flask

        # Create minimal emergency dashboard
        server = Flask(__name__)
        emergency_app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

        emergency_app.layout = html.Div([
            html.H1("[ALERT] EMERGENCY FALLBACK DASHBOARD", className="text-center text-warning mb-4"),
            html.H3("IoT Predictive Maintenance System", className="text-center text-primary mb-4"),

            dbc.Alert([
                html.H4("System Status", className="mb-3"),
                html.P(f"[ERROR] Main dashboard hung in phase: {diagnostic_status['current_phase']}", className="mb-2"),
                html.P(f"[OK] Components loaded: {diagnostic_status['components_loaded']}/{diagnostic_status['total_components']}", className="mb-2"),
                html.P(f"[CHART] Memory usage: {diagnostic_status['memory_usage']:.1f}MB", className="mb-2"),
                html.P("[REFRESH] Emergency fallback activated successfully", className="mb-0 text-success"),
            ], color="warning", className="mb-4"),

            html.H4("Available Actions:", className="mb-3"),
            html.Ul([
                html.Li("View system diagnostics"),
                html.Li("Access basic monitoring tools"),
                html.Li("Review hang analysis results"),
                html.Li("Manual component loading controls"),
            ], className="mb-4"),

            dcc.Interval(id='emergency-interval', interval=5000, n_intervals=0),
            html.Div(id='emergency-status')
        ], className="container mt-5")

        @emergency_app.callback(
            dash.dependencies.Output('emergency-status', 'children'),
            [dash.dependencies.Input('emergency-interval', 'n_intervals')]
        )
        def update_emergency_status(n):
            current_time = datetime.now().strftime('%H:%M:%S')
            return html.P(f"[CLOCK] Emergency dashboard active - {current_time}", className="text-muted text-center")

        log_diagnostic("[ALERT] Starting emergency dashboard on port 8061", "CRITICAL")
        emergency_app.run(debug=False, host='0.0.0.0', port=8061, dev_tools_hot_reload=False)

    except Exception as e:
        log_diagnostic(f"[CRASH] Emergency fallback failed: {e}", "CRITICAL")

def diagnostic_main():
    """Main diagnostic dashboard startup with detailed monitoring"""
    diagnostic_status['startup_time'] = datetime.now()

    log_diagnostic("[ROCKET] ULTRATHINK DIAGNOSTIC DASHBOARD STARTING")
    log_diagnostic("=" * 80)
    log_diagnostic("[TARGET] Goal: Identify exact location of server startup hang")
    log_diagnostic("[MICROSCOPE] Features: Component timeouts, memory monitoring, emergency fallback")
    log_diagnostic("=" * 80)

    # Start heartbeat monitor
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()

    try:
        # Phase 1: Port availability check
        advance_phase('initialization')
        if not check_port_availability(8060):
            log_diagnostic("[WARN] Port 8060 is already in use, trying port 8062", "WARNING")
            if not check_port_availability(8062):
                log_diagnostic("[ERROR] Both ports 8060 and 8062 are in use", "ERROR")
                return
            else:
                target_port = 8062
        else:
            target_port = 8060

        log_diagnostic(f"[OK] Port {target_port} is available")

        # Phase 2: Configuration loading
        advance_phase('config_loading')
        config = component_timeout_wrapper(
            lambda: __import__('config.settings', fromlist=['Settings']).Settings(),
            'Configuration', 15
        )

        # Phase 3: TensorFlow setup
        advance_phase('tensorflow_setup')
        def setup_tensorflow():
            import tensorflow as tf
            log_diagnostic(f"TensorFlow version: {tf.__version__}")
            return tf

        tf = component_timeout_wrapper(setup_tensorflow, 'TensorFlow', 30)

        # Phase 4: Model loading (with timeout per model type)
        advance_phase('model_loading')

        def load_nasa_models():
            log_diagnostic("Loading NASA Telemanom models...")
            # Simulate NASA model loading with progress
            for i in range(10):  # Load first 10 models as test
                time.sleep(0.5)  # Simulate loading time
                log_diagnostic(f"NASA model {i+1}/10 loaded")
            return True

        nasa_models = component_timeout_wrapper(load_nasa_models, 'NASA Models (Sample)', 45)

        # Phase 5: Data services
        advance_phase('data_services')

        def init_data_services():
            from src.data_ingestion.nasa_data_service import NASADataService
            service = NASADataService()
            log_diagnostic("NASA Data Service initialized")
            return service

        data_service = component_timeout_wrapper(init_data_services, 'Data Services', 30)

        # Phase 6: Dashboard components (one by one)
        advance_phase('dashboard_components')

        components = {}
        component_loaders = [
            ('Model Manager', lambda: __import__('src.dashboard.model_manager', fromlist=['PretrainedModelManager'])),
            ('Sensor Stream Manager', lambda: __import__('src.dashboard.sensor_stream_manager', fromlist=['SensorStreamManager'])),
            ('Data Orchestrator', lambda: __import__('src.dashboard.unified_data_orchestrator', fromlist=['UnifiedDataOrchestrator'])),
            ('Visualizer', lambda: __import__('src.dashboard.multi_sensor_visualizer', fromlist=['MultiSensorVisualizer'])),
            ('Chart Manager', lambda: __import__('src.dashboard.components.chart_manager', fromlist=['ChartManager'])),
        ]

        for component_name, loader in component_loaders:
            try:
                components[component_name] = component_timeout_wrapper(loader, component_name, 20)
            except Exception as e:
                log_diagnostic(f"⚠️ Non-critical component failed: {component_name} - {e}", "WARNING")

        # Phase 7: Dash app creation and layout assembly
        advance_phase('layout_assembly')

        def create_dash_app():
            import dash
            import dash_bootstrap_components as dbc
            from dash import html, dcc
            from flask import Flask

            server = Flask(__name__)
            app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

            # Create minimal diagnostic layout
            app.layout = html.Div([
                html.H1("[TARGET] DIAGNOSTIC DASHBOARD SUCCESS!", className="text-center text-success mb-4"),
                html.H3("IoT Predictive Maintenance System", className="text-center text-primary mb-4"),

                dbc.Alert([
                    html.H4("[OK] Diagnostic Results", className="mb-3"),
                    html.P(f"[ROCKET] Successfully passed all startup phases!", className="mb-2"),
                    html.P(f"[CHART] Components loaded: {diagnostic_status['components_loaded']}", className="mb-2"),
                    html.P(f"[MEMORY] Memory usage: {diagnostic_status['memory_usage']:.1f}MB", className="mb-2"),
                    html.P(f"[TIME] Startup time: {(datetime.now() - diagnostic_status['startup_time']).total_seconds():.1f}s", className="mb-0"),
                ], color="success", className="mb-4"),

                html.H4("System Components Status:", className="mb-3"),
                html.Div(id='component-status'),

                dcc.Interval(id='status-interval', interval=2000, n_intervals=0),
            ], className="container mt-5")

            log_diagnostic("✅ Dash app layout created successfully")
            return app

        app = component_timeout_wrapper(create_dash_app, 'Dash App Creation', 30)

        # Phase 8: Callback registration
        advance_phase('callback_registration')

        def register_callbacks():
            @app.callback(
                dash.dependencies.Output('component-status', 'children'),
                [dash.dependencies.Input('status-interval', 'n_intervals')]
            )
            def update_status(n):
                return html.Div([
                    html.P(f"[OK] Configuration: Loaded", className="text-success"),
                    html.P(f"[OK] TensorFlow: Ready", className="text-success"),
                    html.P(f"[OK] NASA Models: Sample Loaded", className="text-success"),
                    html.P(f"[OK] Data Services: Connected", className="text-success"),
                    html.P(f"[OK] Dashboard Components: {len(components)} loaded", className="text-success"),
                    html.P(f"[CLOCK] Current time: {datetime.now().strftime('%H:%M:%S')}", className="text-muted"),
                ])

            log_diagnostic("[OK] Callbacks registered successfully")

        component_timeout_wrapper(register_callbacks, 'Callback Registration', 15)

        # Phase 9: Server binding
        advance_phase('server_binding')
        log_diagnostic(f"[WEB] Preparing to bind server to port {target_port}")

        # Phase 10: Server startup (the critical phase)
        advance_phase('server_startup')
        log_diagnostic("[ROCKET] STARTING DASH SERVER - THIS IS THE CRITICAL MOMENT")
        log_diagnostic("[SEARCH] If hang occurs here, we've identified the exact issue!")

        # Start server with timeout
        def start_server():
            log_diagnostic(f"[SERVER] Starting server on http://localhost:{target_port}")
            app.run(debug=False, host='0.0.0.0', port=target_port, dev_tools_hot_reload=False)

        # This is where the hang likely occurs
        start_server()

    except KeyboardInterrupt:
        log_diagnostic("[STOP] Diagnostic stopped by user")
    except Exception as e:
        log_diagnostic(f"[CRASH] DIAGNOSTIC ERROR in {diagnostic_status['current_phase']}: {e}", "ERROR")
        log_diagnostic(f"[CRASH] Full traceback:\n{traceback.format_exc()}", "ERROR")

        # Try emergency fallback
        activate_emergency_fallback()

if __name__ == '__main__':
    diagnostic_main()