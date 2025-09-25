#!/usr/bin/env python3
"""
Database Setup Script for IoT Anomaly Detection System
Creates necessary database tables and initializes the database schema
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.settings import settings
except ImportError:
    print("Warning: Could not import settings, using default database path")
    settings = None

def create_sqlite_database():
    """Create SQLite database and tables for local development"""

    # Determine database path
    if settings:
        db_config = settings.get_database_config()
        if db_config.type == 'sqlite':
            db_path = db_config.sqlite_path
        else:
            db_path = './data/iot_telemetry.db'
    else:
        db_path = './data/iot_telemetry.db'

    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"Creating SQLite database at: {db_path}")

    # Connect to database (creates file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    create_tables(cursor)

    # Commit changes and close
    conn.commit()
    conn.close()

    print(f"[OK] Database created successfully at: {db_path}")
    return db_path

def create_tables(cursor):
    """Create necessary database tables"""

    # Telemetry data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            spacecraft TEXT NOT NULL,
            channel_name TEXT,
            value REAL NOT NULL,
            anomaly_score REAL DEFAULT 0.0,
            is_anomaly BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Anomaly detection results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            spacecraft TEXT NOT NULL,
            model_name TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            threshold REAL NOT NULL,
            is_anomaly BOOLEAN NOT NULL,
            confidence REAL,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Model performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            dataset TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            training_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Maintenance schedule table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS maintenance_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            equipment_id TEXT NOT NULL,
            scheduled_date DATETIME NOT NULL,
            priority TEXT NOT NULL,
            task_type TEXT NOT NULL,
            estimated_duration INTEGER,
            technician_assigned TEXT,
            status TEXT DEFAULT 'scheduled',
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Work orders table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_number TEXT UNIQUE NOT NULL,
            equipment_id TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT DEFAULT 'open',
            description TEXT,
            created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
            due_date DATETIME,
            completed_date DATETIME,
            assigned_technician TEXT,
            estimated_cost REAL,
            actual_cost REAL
        )
    ''')

    # Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            equipment_id TEXT,
            spacecraft TEXT,
            anomaly_score REAL,
            status TEXT DEFAULT 'active',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at DATETIME,
            resolved_at DATETIME,
            acknowledged_by TEXT,
            resolution_notes TEXT
        )
    ''')

    # System configuration table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT UNIQUE NOT NULL,
            config_value TEXT NOT NULL,
            description TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    print("[OK] Database tables created")

def insert_sample_config(db_path):
    """Insert sample configuration data"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Sample configuration values
    config_data = [
        ('anomaly_threshold', '0.8', 'Default anomaly detection threshold'),
        ('email_alerts_enabled', 'true', 'Enable email notifications'),
        ('maintenance_horizon_days', '7', 'Maintenance planning horizon in days'),
        ('system_status', 'active', 'Current system operational status')
    ]

    for key, value, description in config_data:
        cursor.execute('''
            INSERT OR REPLACE INTO system_config (config_key, config_value, description)
            VALUES (?, ?, ?)
        ''', (key, value, description))

    conn.commit()
    conn.close()
    print("[OK] Sample configuration data inserted")

def verify_database(db_path):
    """Verify database was created correctly"""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    expected_tables = [
        'telemetry_data', 'anomaly_detections', 'model_metrics',
        'maintenance_schedule', 'work_orders', 'alerts', 'system_config'
    ]

    existing_tables = [table[0] for table in tables]

    print(f"Expected tables: {len(expected_tables)}")
    print(f"Created tables: {len(existing_tables)}")

    for table in expected_tables:
        if table in existing_tables:
            print(f"  [OK] {table}")
        else:
            print(f"  [MISSING] {table}")

    conn.close()

    # Check if all expected tables exist
    missing_tables = [table for table in expected_tables if table not in existing_tables]
    if missing_tables:
        print(f"Missing tables: {missing_tables}")
        return False
    return True

def main():
    """Main database setup function"""

    print("=" * 50)
    print("IoT Anomaly Detection System - Database Setup")
    print("=" * 50)

    try:
        # Create SQLite database
        db_path = create_sqlite_database()

        # Insert sample configuration
        insert_sample_config(db_path)

        # Verify database creation
        if verify_database(db_path):
            print("\n[OK] Database setup completed successfully!")
            print(f"Database location: {os.path.abspath(db_path)}")

            # Show database file size
            file_size = os.path.getsize(db_path)
            print(f"Database size: {file_size} bytes")

        else:
            print("\n[ERROR] Database verification failed!")
            return False

    except Exception as e:
        print(f"[ERROR] Database setup failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)