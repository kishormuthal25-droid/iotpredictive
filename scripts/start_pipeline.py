#!/usr/bin/env python3
"""
Real-time Pipeline Orchestration Script for IoT Anomaly Detection System
Manages data streaming, anomaly detection, forecasting, and maintenance scheduling
"""

import os
import sys
import argparse
import signal
import threading
import time
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import tensorflow as tf
from tensorflow import keras
import schedule
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_ingestion.stream_simulator import StreamSimulator
from src.data_ingestion.kafka_producer import KafkaProducer as CustomKafkaProducer
from src.data_ingestion.kafka_consumer import KafkaConsumer as CustomKafkaConsumer
from src.data_ingestion.database_manager import DatabaseManager
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly_detection.lstm_detector import LSTMDetector
from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
from src.anomaly_detection.model_evaluator import ModelEvaluator
from src.forecasting.transformer_forecaster import TransformerForecaster
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.maintenance.scheduler import MaintenanceScheduler
from src.maintenance.work_order_manager import WorkOrderManager
from src.maintenance.priority_calculator import PriorityCalculator
from src.alerts.alert_manager import AlertManager
from src.utils.logger import get_logger
from src.utils.metrics import StreamingMetrics
from config.settings import Settings

# Initialize logger
logger = get_logger(__name__)

class IoTPipeline:
    """Main pipeline orchestrator for IoT anomaly detection system"""
    
    def __init__(self, config_path: str = None, mode: str = 'simulation'):
        """Initialize pipeline
        
        Args:
            config_path: Path to configuration file
            mode: Pipeline mode ('simulation', 'kafka', 'direct')
        """
        self.settings = Settings(config_path)
        self.mode = mode
        self.running = False
        self.threads = []
        
        # Initialize components
        self._initialize_components()
        
        # Load models
        self._load_models()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Metrics tracking
        self.metrics = {
            'processed_count': 0,
            'anomaly_count': 0,
            'alert_count': 0,
            'work_order_count': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        # Data buffers
        self.telemetry_buffer = []
        self.anomaly_buffer = []
        self.forecast_buffer = []
        
        logger.info(f"Pipeline initialized in {mode} mode")
    
    def _initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # Database
        self.db_manager = DatabaseManager(
            host=self.settings.database['host'],
            port=self.settings.database['port'],
            database=self.settings.database['name'],
            user=self.settings.database['user'],
            password=self.settings.database['password']
        )
        
        # Create tables if needed
        self._setup_database()
        
        # Data processing
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        
        # Streaming components
        if self.mode == 'simulation':
            # Load sample data for simulation
            data_path = Path(self.settings.data['smap_path']) / 'train.npy'
            if data_path.exists():
                data = np.load(data_path)
            else:
                # Generate sample data if not available
                data = np.random.randn(1000, 25)
            
            self.stream_simulator = StreamSimulator(
                data=data,
                stream_rate=self.settings.data['streaming']['frequency_seconds'],
                buffer_size=self.settings.data['streaming']['buffer_size']
            )
        
        elif self.mode == 'kafka':
            # Kafka components
            self.kafka_producer = CustomKafkaProducer(
                bootstrap_servers=self.settings.kafka['bootstrap_servers'],
                topic=self.settings.kafka['topics']['smap']
            )
            
            self.kafka_consumer = CustomKafkaConsumer(
                bootstrap_servers=self.settings.kafka['bootstrap_servers'],
                topics=[
                    self.settings.kafka['topics']['smap'],
                    self.settings.kafka['topics']['msl']
                ],
                group_id='iot_pipeline'
            )
        
        # Maintenance components
        self.work_order_manager = WorkOrderManager()
        self.priority_calculator = PriorityCalculator()
        self.maintenance_scheduler = MaintenanceScheduler(
            max_technicians=self.settings.maintenance['constraints']['max_technicians'],
            work_hours=(
                self.settings.maintenance['constraints']['work_hours_start'],
                self.settings.maintenance['constraints']['work_hours_end']
            )
        )
        
        # Alert manager
        self.alert_manager = AlertManager(
            smtp_server=self.settings.alerts['email']['smtp_server'],
            smtp_port=self.settings.alerts['email']['smtp_port'],
            sender_email=self.settings.alerts['email']['sender'],
            password=self.settings.alerts['email']['password'],
            recipients=self.settings.alerts['email']['recipients']
        )
        
        # Streaming metrics
        self.streaming_metrics = StreamingMetrics(window_size=1000)
        
        logger.info("Components initialized successfully")
    
    def _setup_database(self):
        """Setup database tables"""
        try:
            # Create telemetry table
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    equipment_id VARCHAR(50) NOT NULL,
                    sensor_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create anomalies table
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    anomaly_id SERIAL PRIMARY KEY,
                    equipment_id VARCHAR(50) NOT NULL,
                    anomaly_score FLOAT NOT NULL,
                    severity VARCHAR(20),
                    detected_at TIMESTAMPTZ NOT NULL,
                    model_name VARCHAR(50),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create work orders table
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS work_orders (
                    work_order_id VARCHAR(50) PRIMARY KEY,
                    equipment_id VARCHAR(50) NOT NULL,
                    priority VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    description TEXT,
                    anomaly_id INTEGER REFERENCES anomalies(anomaly_id),
                    assigned_technician VARCHAR(50),
                    estimated_duration INTEGER,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create alerts table
            self.db_manager.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id SERIAL PRIMARY KEY,
                    anomaly_id INTEGER REFERENCES anomalies(anomaly_id),
                    alert_type VARCHAR(50),
                    severity VARCHAR(20),
                    message TEXT,
                    sent_at TIMESTAMPTZ DEFAULT NOW(),
                    acknowledged BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create indexes for performance
            self.db_manager.execute("""
                CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp 
                ON telemetry_data(timestamp DESC);
                
                CREATE INDEX IF NOT EXISTS idx_anomalies_equipment 
                ON anomalies(equipment_id, detected_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_work_orders_status 
                ON work_orders(status, created_at DESC);
            """)
            
            # Convert to TimescaleDB hypertables if available
            try:
                self.db_manager.execute("""
                    SELECT create_hypertable('telemetry_data', 'timestamp',
                        if_not_exists => TRUE);
                """)
                logger.info("TimescaleDB hypertable created for telemetry_data")
            except:
                logger.info("Using standard PostgreSQL tables")
            
            logger.info("Database setup completed")
            
        except Exception as e:
            logger.error(f"Database setup failed: {str(e)}")
            raise
    
    def _load_models(self):
        """Load trained models"""
        logger.info("Loading trained models...")
        
        models_path = Path(self.settings.data['models_path'])
        self.models = {}
        
        try:
            # Load anomaly detection models
            lstm_path = models_path / 'lstm_detector_latest'
            if lstm_path.exists():
                self.models['lstm_detector'] = LSTMDetector.load(str(lstm_path))
                logger.info("Loaded LSTM Detector")
            
            ae_path = models_path / 'lstm_autoencoder_latest'
            if ae_path.exists():
                self.models['lstm_autoencoder'] = LSTMAutoencoder.load(str(ae_path))
                logger.info("Loaded LSTM Autoencoder")
            
            # Load forecasting models
            transformer_path = models_path / 'transformer_forecaster_latest'
            if transformer_path.exists():
                self.models['transformer_forecaster'] = TransformerForecaster.load(str(transformer_path))
                logger.info("Loaded Transformer Forecaster")
            
            lstm_fc_path = models_path / 'lstm_forecaster_latest'
            if lstm_fc_path.exists():
                self.models['lstm_forecaster'] = LSTMForecaster.load(str(lstm_fc_path))
                logger.info("Loaded LSTM Forecaster")
            
            if not self.models:
                logger.warning("No models found. Please train models first.")
                # Load or create default models
                self._create_default_models()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default models for testing"""
        logger.info("Creating default models for testing...")
        
        # Create simple models with default parameters
        self.models['lstm_detector'] = LSTMDetector(
            sequence_length=50,
            n_features=25,
            hidden_units=[32, 16]
        )
        
        self.models['lstm_autoencoder'] = LSTMAutoencoder(
            sequence_length=50,
            n_features=25,
            encoding_dim=16,
            latent_dim=8
        )
        
        logger.info("Default models created")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.stop()
    
    def process_telemetry_data(self, data: Dict[str, Any]):
        """Process incoming telemetry data
        
        Args:
            data: Telemetry data dictionary
        """
        try:
            # Extract data
            timestamp = data.get('timestamp', datetime.now())
            equipment_id = data.get('equipment_id', 'UNKNOWN')
            sensor_values = data.get('values', [])
            
            # Store in buffer
            self.telemetry_buffer.append({
                'timestamp': timestamp,
                'equipment_id': equipment_id,
                'values': sensor_values
            })
            
            # Keep buffer size limited
            if len(self.telemetry_buffer) > self.settings.data['streaming']['buffer_size']:
                self.telemetry_buffer.pop(0)
            
            # Check if we have enough data for sequence
            window_size = self.settings.data['streaming']['window_size']
            if len(self.telemetry_buffer) >= window_size:
                # Create sequence for anomaly detection
                sequence = np.array([item['values'] for item in self.telemetry_buffer[-window_size:]])
                
                # Preprocess sequence
                sequence = self.preprocessor.normalize(sequence)
                
                # Run anomaly detection
                anomaly_results = self.detect_anomalies(sequence, equipment_id)
                
                if anomaly_results['is_anomaly']:
                    self.handle_anomaly(anomaly_results, equipment_id)
                
                # Run forecasting
                forecast_results = self.generate_forecast(sequence, equipment_id)
                
                # Store results in database
                self.store_results(data, anomaly_results, forecast_results)
            
            # Update metrics
            self.metrics['processed_count'] += 1
            
            # Update streaming metrics
            if 'anomaly_score' in locals():
                self.streaming_metrics.update(
                    y_true=0,  # Placeholder for real labels
                    y_pred=anomaly_results['anomaly_score']
                )
            
        except Exception as e:
            logger.error(f"Error processing telemetry data: {str(e)}")
            self.metrics['errors'].append({
                'timestamp': datetime.now(),
                'error': str(e)
            })
    
    def detect_anomalies(self, sequence: np.ndarray, equipment_id: str) -> Dict[str, Any]:
        """Run anomaly detection models
        
        Args:
            sequence: Input sequence
            equipment_id: Equipment identifier
        
        Returns:
            Anomaly detection results
        """
        results = {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'model_scores': {},
            'timestamp': datetime.now(),
            'equipment_id': equipment_id
        }
        
        try:
            # Reshape for models
            sequence = sequence.reshape(1, *sequence.shape)
            
            # Run LSTM Detector
            if 'lstm_detector' in self.models:
                lstm_score = self.models['lstm_detector'].predict(sequence)[0][0]
                results['model_scores']['lstm_detector'] = float(lstm_score)
            
            # Run LSTM Autoencoder
            if 'lstm_autoencoder' in self.models:
                reconstruction_error = self.models['lstm_autoencoder'].calculate_reconstruction_error(sequence)[0]
                # Normalize to 0-1 range
                ae_score = min(reconstruction_error / 10.0, 1.0)
                results['model_scores']['lstm_autoencoder'] = float(ae_score)
            
            # Ensemble scoring
            if results['model_scores']:
                results['anomaly_score'] = np.mean(list(results['model_scores'].values()))
                
                # Check against threshold
                threshold = self.settings.anomaly['threshold_percentile'] / 100
                results['is_anomaly'] = results['anomaly_score'] > threshold
                
                # Determine severity
                if results['is_anomaly']:
                    if results['anomaly_score'] > self.settings.anomaly['severity_levels']['high']:
                        results['severity'] = 'HIGH'
                    elif results['anomaly_score'] > self.settings.anomaly['severity_levels']['medium']:
                        results['severity'] = 'MEDIUM'
                    else:
                        results['severity'] = 'LOW'
            
            # Add to anomaly buffer
            if results['is_anomaly']:
                self.anomaly_buffer.append(results)
                self.metrics['anomaly_count'] += 1
                
                # Check for consecutive anomalies
                min_consecutive = self.settings.anomaly['min_consecutive_anomalies']
                recent_anomalies = [a for a in self.anomaly_buffer[-min_consecutive:]
                                  if a['equipment_id'] == equipment_id]
                
                if len(recent_anomalies) >= min_consecutive:
                    results['confirmed_anomaly'] = True
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
        
        return results
    
    def generate_forecast(self, sequence: np.ndarray, equipment_id: str) -> Dict[str, Any]:
        """Generate time series forecast
        
        Args:
            sequence: Input sequence
            equipment_id: Equipment identifier
        
        Returns:
            Forecast results
        """
        results = {
            'forecast': None,
            'confidence_interval': None,
            'horizon': self.settings.models['forecasting']['horizon'],
            'timestamp': datetime.now(),
            'equipment_id': equipment_id
        }
        
        try:
            # Reshape for models
            sequence = sequence.reshape(1, *sequence.shape)
            
            # Run Transformer Forecaster
            if 'transformer_forecaster' in self.models:
                forecast = self.models['transformer_forecaster'].forecast(
                    sequence,
                    steps=results['horizon']
                )
                results['forecast'] = forecast[0].tolist()
                
                # Generate confidence intervals
                lower, upper = self.calculate_confidence_intervals(forecast[0])
                results['confidence_interval'] = {
                    'lower': lower.tolist(),
                    'upper': upper.tolist()
                }
            
            # Alternative: LSTM Forecaster
            elif 'lstm_forecaster' in self.models:
                forecast = self.models['lstm_forecaster'].forecast(
                    sequence,
                    steps=results['horizon']
                )
                results['forecast'] = forecast[0].tolist()
            
            # Add to forecast buffer
            if results['forecast'] is not None:
                self.forecast_buffer.append(results)
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
        
        return results
    
    def calculate_confidence_intervals(self, forecast: np.ndarray, 
                                      confidence: float = 0.95) -> tuple:
        """Calculate confidence intervals for forecast
        
        Args:
            forecast: Forecast values
            confidence: Confidence level
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Simple confidence interval based on historical variance
        std = np.std(forecast) * 0.1  # Placeholder
        z_score = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
        
        lower = forecast - z_score * std
        upper = forecast + z_score * std
        
        return lower, upper
    
    def handle_anomaly(self, anomaly_results: Dict[str, Any], equipment_id: str):
        """Handle detected anomaly
        
        Args:
            anomaly_results: Anomaly detection results
            equipment_id: Equipment identifier
        """
        try:
            logger.warning(f"Anomaly detected on {equipment_id}: {anomaly_results}")
            
            # Store anomaly in database
            anomaly_id = self.store_anomaly(anomaly_results)
            
            # Calculate priority
            equipment_info = self.get_equipment_info(equipment_id)
            priority = self.priority_calculator.calculate_priority(
                anomaly_results,
                equipment_info
            )
            
            # Create work order if high priority
            if priority > 70 or anomaly_results.get('severity') in ['HIGH', 'CRITICAL']:
                work_order_id = self.create_work_order(
                    equipment_id,
                    anomaly_id,
                    priority,
                    anomaly_results
                )
                
                logger.info(f"Work order created: {work_order_id}")
                self.metrics['work_order_count'] += 1
            
            # Send alert if needed
            if anomaly_results.get('confirmed_anomaly', False):
                self.send_alert(equipment_id, anomaly_results, priority)
                self.metrics['alert_count'] += 1
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {str(e)}")
    
    def store_anomaly(self, anomaly_results: Dict[str, Any]) -> int:
        """Store anomaly in database
        
        Args:
            anomaly_results: Anomaly detection results
        
        Returns:
            Anomaly ID
        """
        query = """
            INSERT INTO anomalies (equipment_id, anomaly_score, severity, detected_at, model_name)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING anomaly_id;
        """
        
        params = [
            anomaly_results['equipment_id'],
            anomaly_results['anomaly_score'],
            anomaly_results.get('severity', 'LOW'),
            anomaly_results['timestamp'],
            'ensemble'
        ]
        
        result = self.db_manager.execute_query(query, params)
        return result[0][0] if result else None
    
    def get_equipment_info(self, equipment_id: str) -> Dict[str, Any]:
        """Get equipment information
        
        Args:
            equipment_id: Equipment identifier
        
        Returns:
            Equipment information dictionary
        """
        # In production, this would query the equipment database
        # For now, return mock data
        return {
            'equipment_id': equipment_id,
            'criticality': 'HIGH',
            'location': 'Building A',
            'downtime_cost_per_hour': 1000,
            'last_maintenance': datetime.now() - timedelta(days=30),
            'mtbf': 180  # days
        }
    
    def create_work_order(self, equipment_id: str, anomaly_id: int,
                         priority: float, anomaly_results: Dict[str, Any]) -> str:
        """Create maintenance work order
        
        Args:
            equipment_id: Equipment identifier
            anomaly_id: Anomaly ID
            priority: Priority score
            anomaly_results: Anomaly detection results
        
        Returns:
            Work order ID
        """
        # Determine priority class
        if priority > 85:
            priority_class = 'CRITICAL'
        elif priority > 70:
            priority_class = 'HIGH'
        elif priority > 50:
            priority_class = 'MEDIUM'
        else:
            priority_class = 'LOW'
        
        # Create work order
        work_order = {
            'equipment_id': equipment_id,
            'priority': priority_class,
            'description': f"Anomaly detected with score {anomaly_results['anomaly_score']:.3f}",
            'anomaly_id': anomaly_id,
            'estimated_duration': 120,  # Default 2 hours
            'required_skills': ['mechanical', 'electrical']
        }
        
        work_order_id = self.work_order_manager.create_work_order(**work_order)
        
        # Store in database
        query = """
            INSERT INTO work_orders (work_order_id, equipment_id, priority, status, 
                                    description, anomaly_id, estimated_duration)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        
        params = [
            work_order_id,
            equipment_id,
            priority_class,
            'PENDING',
            work_order['description'],
            anomaly_id,
            work_order['estimated_duration']
        ]
        
        self.db_manager.execute(query, params)
        
        return work_order_id
    
    def send_alert(self, equipment_id: str, anomaly_results: Dict[str, Any], 
                  priority: float):
        """Send alert notification
        
        Args:
            equipment_id: Equipment identifier
            anomaly_results: Anomaly detection results
            priority: Priority score
        """
        try:
            alert_data = {
                'equipment_id': equipment_id,
                'anomaly_score': anomaly_results['anomaly_score'],
                'severity': anomaly_results.get('severity', 'LOW'),
                'priority': priority,
                'timestamp': anomaly_results['timestamp'],
                'message': f"Critical anomaly detected on {equipment_id}"
            }
            
            # Send email alert
            self.alert_manager.send_alert(alert_data)
            
            # Store alert in database
            query = """
                INSERT INTO alerts (anomaly_id, alert_type, severity, message)
                VALUES (%s, %s, %s, %s);
            """
            
            params = [
                anomaly_results.get('anomaly_id'),
                'EMAIL',
                anomaly_results.get('severity', 'LOW'),
                alert_data['message']
            ]
            
            self.db_manager.execute(query, params)
            
            logger.info(f"Alert sent for {equipment_id}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    def store_results(self, telemetry: Dict[str, Any], 
                     anomaly_results: Dict[str, Any],
                     forecast_results: Dict[str, Any]):
        """Store processing results in database
        
        Args:
            telemetry: Original telemetry data
            anomaly_results: Anomaly detection results
            forecast_results: Forecast results
        """
        try:
            # Store telemetry data
            query = """
                INSERT INTO telemetry_data (timestamp, equipment_id, sensor_data)
                VALUES (%s, %s, %s);
            """
            
            params = [
                telemetry.get('timestamp', datetime.now()),
                telemetry.get('equipment_id', 'UNKNOWN'),
                json.dumps({
                    'values': telemetry.get('values', []),
                    'anomaly_score': anomaly_results.get('anomaly_score', 0),
                    'forecast': forecast_results.get('forecast', [])
                })
            ]
            
            self.db_manager.execute(query, params)
            
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
    
    def run_simulation_mode(self):
        """Run pipeline in simulation mode"""
        logger.info("Starting simulation mode...")
        
        while self.running:
            try:
                # Get next batch from simulator
                batch = self.stream_simulator.get_next_batch(batch_size=1)
                
                # Create telemetry data
                telemetry_data = {
                    'timestamp': datetime.now(),
                    'equipment_id': f'PUMP_{np.random.randint(1, 6):03d}',
                    'values': batch[0].tolist()
                }
                
                # Process data
                self.process_telemetry_data(telemetry_data)
                
                # Sleep to simulate real-time streaming
                time.sleep(1.0 / self.settings.data['streaming']['frequency_seconds'])
                
            except Exception as e:
                logger.error(f"Error in simulation mode: {str(e)}")
                time.sleep(1)
    
    def run_kafka_mode(self):
        """Run pipeline in Kafka mode"""
        logger.info("Starting Kafka mode...")
        
        while self.running:
            try:
                # Consume messages from Kafka
                messages = self.kafka_consumer.consume_messages(
                    max_messages=10,
                    timeout=1.0
                )
                
                for message in messages:
                    # Process each message
                    self.process_telemetry_data(message)
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in Kafka mode: {str(e)}")
                time.sleep(1)
    
    def run_scheduler(self):
        """Run maintenance scheduler periodically"""
        logger.info("Starting maintenance scheduler...")
        
        def schedule_maintenance():
            try:
                # Get pending work orders
                pending_orders = self.work_order_manager.get_work_orders_by_status('PENDING')
                
                if pending_orders:
                    # Get available technicians (mock data)
                    technicians = pd.DataFrame([
                        {'technician_id': 'TECH_001', 'skills': ['electrical', 'mechanical'], 'availability': True},
                        {'technician_id': 'TECH_002', 'skills': ['mechanical'], 'availability': True},
                        {'technician_id': 'TECH_003', 'skills': ['electrical'], 'availability': True}
                    ])
                    
                    # Get equipment info (mock data)
                    equipment = pd.DataFrame([
                        {'equipment_id': f'PUMP_{i:03d}', 'criticality': 'HIGH', 'location': f'Building {chr(65+i%3)}'}
                        for i in range(1, 6)
                    ])
                    
                    # Create schedule
                    work_orders_df = pd.DataFrame(pending_orders)
                    schedule = self.maintenance_scheduler.create_schedule(
                        work_orders_df,
                        technicians,
                        equipment
                    )
                    
                    logger.info(f"Created schedule for {len(schedule)} work orders")
                    
                    # Update work order statuses
                    for _, assignment in schedule.iterrows():
                        if pd.notna(assignment['assigned_technician']):
                            self.work_order_manager.assign_work_order(
                                assignment['work_order_id'],
                                assignment['assigned_technician']
                            )
            
            except Exception as e:
                logger.error(f"Error in scheduler: {str(e)}")
        
        # Schedule to run every hour
        schedule.every().hour.do(schedule_maintenance)
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def run_metrics_reporter(self):
        """Run metrics reporting thread"""
        logger.info("Starting metrics reporter...")
        
        while self.running:
            try:
                # Calculate runtime
                runtime = datetime.now() - self.metrics['start_time']
                
                # Get current streaming metrics
                current_metrics = self.streaming_metrics.get_current_metrics('regression')
                
                # Log metrics
                logger.info(f"""
                Pipeline Metrics:
                - Runtime: {runtime}
                - Processed: {self.metrics['processed_count']}
                - Anomalies: {self.metrics['anomaly_count']}
                - Alerts: {self.metrics['alert_count']}
                - Work Orders: {self.metrics['work_order_count']}
                - Current MAE: {current_metrics.get('mae', 0):.4f}
                - Errors: {len(self.metrics['errors'])}
                """)
                
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                logger.info(f"System: CPU {cpu_percent}%, Memory {memory_percent}%")
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in metrics reporter: {str(e)}")
                time.sleep(60)
    
    def start(self):
        """Start the pipeline"""
        logger.info("Starting IoT anomaly detection pipeline...")
        self.running = True
        
        # Start main processing thread based on mode
        if self.mode == 'simulation':
            main_thread = threading.Thread(target=self.run_simulation_mode)
        elif self.mode == 'kafka':
            main_thread = threading.Thread(target=self.run_kafka_mode)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        main_thread.daemon = True
        main_thread.start()
        self.threads.append(main_thread)
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        self.threads.append(scheduler_thread)
        
        # Start metrics reporter thread
        metrics_thread = threading.Thread(target=self.run_metrics_reporter)
        metrics_thread.daemon = True
        metrics_thread.start()
        self.threads.append(metrics_thread)
        
        logger.info(f"Pipeline started with {len(self.threads)} threads")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping pipeline...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        # Close connections
        if hasattr(self, 'db_manager'):
            self.db_manager.close()
        
        if hasattr(self, 'kafka_producer'):
            self.kafka_producer.close()
        
        if hasattr(self, 'kafka_consumer'):
            self.kafka_consumer.close()
        
        # Final metrics report
        logger.info(f"""
        Final Pipeline Metrics:
        - Total Processed: {self.metrics['processed_count']}
        - Total Anomalies: {self.metrics['anomaly_count']}
        - Total Alerts: {self.metrics['alert_count']}
        - Total Work Orders: {self.metrics['work_order_count']}
        - Total Errors: {len(self.metrics['errors'])}
        """)
        
        logger.info("Pipeline stopped successfully")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Start IoT Anomaly Detection Pipeline')
    parser.add_argument('--mode', type=str, default='simulation',
                       choices=['simulation', 'kafka', 'direct'],
                       help='Pipeline mode')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (limited duration)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     IoT Anomaly Detection Pipeline - Starting...    ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Initialize pipeline
    pipeline = IoTPipeline(
        config_path=args.config,
        mode=args.mode
    )
    
    try:
        if args.test:
            # Run in test mode for 1 minute
            logger.info("Running in test mode for 60 seconds...")
            test_thread = threading.Thread(target=pipeline.start)
            test_thread.daemon = True
            test_thread.start()
            
            time.sleep(60)
            pipeline.stop()
        else:
            # Run normally
            pipeline.start()
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()