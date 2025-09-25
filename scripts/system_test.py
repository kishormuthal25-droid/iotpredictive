#!/usr/bin/env python3
"""
System Test Script for IoT Anomaly Detection System
Tests the core functionality without ML models that require TensorFlow
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path

def test_configuration():
    """Test configuration system"""
    print("=== Testing Configuration System ===")
    try:
        # Add project root to path
        sys.path.append(str(Path(__file__).parent.parent))
        from config.settings import settings

        # Test configuration loading
        print(f"[OK] Configuration loaded from: {settings.config_file}")

        # Test configuration access
        debug_mode = settings.debug
        dashboard_port = settings.get('dashboard.server.port')
        data_root = settings.get('paths.data_root')

        print(f"[OK] Debug mode: {debug_mode}")
        print(f"[OK] Dashboard port: {dashboard_port}")
        print(f"[OK] Data root: {data_root}")

        return True
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n=== Testing Data Loading ===")
    try:
        # Test SMAP data
        smap_train = np.load('data/raw/smap/train.npy')
        smap_labels = np.load('data/raw/smap/train_labels.npy')
        smap_test = np.load('data/raw/smap/test.npy')

        print(f"[OK] SMAP train data: {smap_train.shape}")
        print(f"[OK] SMAP train labels: {smap_labels.shape}")
        print(f"[OK] SMAP test data: {smap_test.shape}")

        # Test MSL data
        msl_train = np.load('data/raw/msl/train.npy')
        msl_labels = np.load('data/raw/msl/train_labels.npy')

        print(f"[OK] MSL train data: {msl_train.shape}")
        print(f"[OK] MSL train labels: {msl_labels.shape}")

        # Test labels CSV
        with open('data/raw/labeled_anomalies.csv', 'r') as f:
            lines = f.readlines()
        print(f"[OK] Labels CSV: {len(lines)} lines")

        # Basic data validation
        assert smap_train.shape[1] == 25, "SMAP should have 25 features"
        assert msl_train.shape[1] == 55, "MSL should have 55 features"
        assert len(smap_labels) == len(smap_train), "Labels and data length mismatch"

        print("[OK] Data validation passed")
        return True

    except Exception as e:
        print(f"[ERROR] Data loading test failed: {e}")
        return False

def test_database():
    """Test database functionality"""
    print("\n=== Testing Database ===")
    try:
        # Test database connection
        db_path = 'data/iot_telemetry.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in cursor.fetchall()]

        expected_tables = [
            'telemetry_data', 'anomaly_detections', 'model_metrics',
            'maintenance_schedule', 'work_orders', 'alerts', 'system_config'
        ]

        for table in expected_tables:
            if table in tables:
                print(f"[OK] Table '{table}' exists")
            else:
                print(f"[ERROR] Table '{table}' missing")
                return False

        # Test data insertion
        cursor.execute("""
            INSERT INTO telemetry_data (spacecraft, channel_name, value, anomaly_score)
            VALUES ('SMAP_TEST', 'temperature', 25.5, 0.1)
        """)

        # Test data retrieval
        cursor.execute("SELECT COUNT(*) FROM telemetry_data WHERE spacecraft='SMAP_TEST'")
        count = cursor.fetchone()[0]

        conn.commit()
        conn.close()

        print(f"[OK] Database operations successful, test records: {count}")
        return True

    except Exception as e:
        print(f"[ERROR] Database test failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("\n=== Testing Dashboard Components ===")
    try:
        # Test basic dashboard imports
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
        import plotly.express as px
        import plotly.graph_objects as go

        print(f"[OK] Dash {dash.__version__}")
        print(f"[OK] Dash Bootstrap Components")
        print(f"[OK] Plotly components")

        # Test basic dashboard creation
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        # Create sample data for visualization
        sample_data = np.random.randn(100, 5)

        # Create basic layout
        app.layout = dbc.Container([
            html.H1("IoT Anomaly Detection System", className="text-center mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Status", className="card-title"),
                            html.P("Dashboard components are working!", className="card-text"),
                            dbc.Badge("Online", color="success")
                        ])
                    ])
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Data Summary", className="card-title"),
                            html.P(f"SMAP: 5000 samples, 25 features", className="card-text"),
                            html.P(f"MSL: 5000 samples, 55 features", className="card-text")
                        ])
                    ])
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Database", className="card-title"),
                            html.P(f"Tables: 7", className="card-text"),
                            dbc.Badge("Connected", color="success")
                        ])
                    ])
                ], width=4)
            ]),

            html.Hr(),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=px.line(
                            x=range(len(sample_data)),
                            y=sample_data[:, 0],
                            title="Sample Telemetry Data",
                            labels={"x": "Time", "y": "Value"}
                        )
                    )
                ])
            ])
        ])

        print("[OK] Dashboard layout created successfully")
        return True

    except Exception as e:
        print(f"[ERROR] Dashboard test failed: {e}")
        return False

def test_basic_data_processing():
    """Test basic data processing capabilities"""
    print("\n=== Testing Data Processing ===")
    try:
        # Load sample data
        smap_data = np.load('data/raw/smap/train.npy')

        # Basic preprocessing
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        # Test normalization
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(smap_data)

        print(f"[OK] MinMax normalization: {normalized_data.shape}")
        print(f"    Original range: [{smap_data.min():.3f}, {smap_data.max():.3f}]")
        print(f"    Normalized range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

        # Test standardization
        std_scaler = StandardScaler()
        standardized_data = std_scaler.fit_transform(smap_data)

        print(f"[OK] Standard normalization: {standardized_data.shape}")
        print(f"    Mean: {standardized_data.mean():.3f}, Std: {standardized_data.std():.3f}")

        # Test windowing
        window_size = 50
        stride = 10
        windowed_data = []

        for i in range(0, len(smap_data) - window_size, stride):
            window = smap_data[i:i + window_size]
            windowed_data.append(window)

        windowed_data = np.array(windowed_data)
        print(f"[OK] Windowing: {windowed_data.shape} (window_size={window_size}, stride={stride})")

        return True

    except Exception as e:
        print(f"[ERROR] Data processing test failed: {e}")
        return False

def main():
    """Run all system tests"""
    print("=" * 60)
    print("IoT Anomaly Detection System - Comprehensive Test")
    print("=" * 60)

    tests = [
        ("Configuration", test_configuration),
        ("Data Loading", test_data_loading),
        ("Database", test_database),
        ("Dashboard Components", test_dashboard_components),
        ("Data Processing", test_basic_data_processing)
    ]

    results = {}

    for test_name, test_func in tests:
        results[test_name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(tests)

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All core functionality is working!")
        print("\nNext steps:")
        print("1. Install compatible TensorFlow version for ML models")
        print("2. Run: python scripts/train_models.py (after TensorFlow is fixed)")
        print("3. Run: python scripts/run_dashboard.py")
        print("4. Access dashboard at http://localhost:8050")
    else:
        print(f"\n[WARNING] {total - passed} tests failed. Please fix issues before proceeding.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)