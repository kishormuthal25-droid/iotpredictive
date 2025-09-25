"""
Real-time Streaming Service
Continuously streams NASA telemetry data to database for real-time dashboard updates
"""

import logging
import numpy as np
import threading
import time
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import json

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.data_ingestion.database_manager import DatabaseManager
from src.data_ingestion.unified_data_access import unified_data_access
from src.data_ingestion.equipment_mapper import equipment_mapper
from src.dashboard.model_manager import pretrained_model_manager

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for streaming service"""
    enabled: bool = True
    refresh_interval: float = 1.0  # seconds
    batch_size: int = 50
    max_queue_size: int = 1000
    anomaly_detection_enabled: bool = True
    simulate_realistic_patterns: bool = True


class RealTimeStreamingService:
    """
    Real-time streaming service for NASA telemetry data
    Simulates live data feed and performs real-time anomaly detection
    """

    def __init__(self, database_manager: Optional[DatabaseManager] = None,
                 config: Optional[StreamConfig] = None):
        """Initialize real-time streaming service"""
        self.db_manager = database_manager or DatabaseManager()
        self.config = config or StreamConfig()

        # Streaming state
        self.is_running = False
        self.is_paused = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Data queues
        self.telemetry_queue = Queue(maxsize=self.config.max_queue_size)
        self.anomaly_queue = Queue(maxsize=self.config.max_queue_size)

        # Threading
        self._threads = []
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

        # Performance tracking
        self.stats = {
            'start_time': None,
            'records_streamed': 0,
            'anomalies_detected': 0,
            'queue_size': 0,
            'processing_rate': 0.0,
            'last_update': None
        }

        # Error tracking
        self.error_count = 0

        # Available equipment for streaming
        self.available_equipment = {}
        self._initialize_equipment()

        # Realistic data patterns
        self.equipment_patterns = {}
        self._initialize_patterns()

        logger.info("Real-time Streaming Service initialized")

    def _initialize_equipment(self):
        """Initialize available equipment from database or defaults"""
        try:
            # Try to get from database first
            available = unified_data_access.get_available_equipment()
            if available:
                self.available_equipment = available
                logger.info(f"Loaded {sum(len(eq_list) for eq_list in available.values())} equipment from database")
                return

            # Fallback to equipment mapper
            self.available_equipment = {
                'SMAP': [],
                'MSL': []
            }

            # Get all equipment from mapper
            for spacecraft in ['SMAP', 'MSL']:
                for i in range(1, 26):  # A-1 to T-X pattern
                    for prefix in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']:
                        equipment_id = f"{spacecraft}_{prefix}-{i}"
                        if equipment_mapper.get_equipment_info(equipment_id):
                            self.available_equipment[spacecraft].append(equipment_id)

            # If still empty, create default set
            if not any(self.available_equipment.values()):
                self.available_equipment = {
                    'SMAP': [f"SMAP_A-{i}" for i in range(1, 26)],
                    'MSL': [f"MSL_B-{i}" for i in range(1, 31)]
                }

            logger.info(f"Initialized with default equipment: SMAP={len(self.available_equipment['SMAP'])}, MSL={len(self.available_equipment['MSL'])}")

        except Exception as e:
            logger.error(f"Failed to initialize equipment: {e}")
            # Minimal fallback
            self.available_equipment = {
                'SMAP': ['SMAP_A-1', 'SMAP_A-2', 'SMAP_A-3'],
                'MSL': ['MSL_B-1', 'MSL_B-2', 'MSL_B-3']
            }

    def _initialize_patterns(self):
        """Initialize realistic data patterns for each equipment"""
        for spacecraft, equipment_list in self.available_equipment.items():
            for equipment_id in equipment_list:
                # Create equipment-specific pattern
                pattern = self._generate_equipment_pattern(equipment_id)
                self.equipment_patterns[equipment_id] = pattern

    def _generate_equipment_pattern(self, equipment_id: str) -> Dict[str, Any]:
        """Generate realistic pattern for specific equipment"""
        # Get equipment info
        equipment_info = equipment_mapper.get_equipment_info(equipment_id)

        if not equipment_info:
            # Default pattern
            return {
                'base_value': 0.5,
                'noise_level': 0.1,
                'trend_factor': 0.0,
                'seasonal_amplitude': 0.2,
                'seasonal_period': 100,
                'anomaly_probability': 0.02
            }

        # Pattern based on equipment type
        subsystem = equipment_info.get('subsystem', 'unknown').lower()
        equipment_type = equipment_info.get('equipment_type', 'unknown').lower()

        if 'power' in subsystem or 'pwr' in equipment_id.lower():
            # Power systems: more stable, periodic patterns
            pattern = {
                'base_value': 0.7,
                'noise_level': 0.05,
                'trend_factor': 0.001,
                'seasonal_amplitude': 0.15,
                'seasonal_period': 120,
                'anomaly_probability': 0.015
            }
        elif 'mobility' in subsystem or 'mob' in equipment_id.lower():
            # Mobility systems: more variable, movement patterns
            pattern = {
                'base_value': 0.4,
                'noise_level': 0.15,
                'trend_factor': -0.002,
                'seasonal_amplitude': 0.3,
                'seasonal_period': 80,
                'anomaly_probability': 0.025
            }
        elif 'communication' in subsystem or 'com' in equipment_id.lower():
            # Communication systems: signal strength patterns
            pattern = {
                'base_value': 0.6,
                'noise_level': 0.08,
                'trend_factor': 0.0005,
                'seasonal_amplitude': 0.25,
                'seasonal_period': 150,
                'anomaly_probability': 0.02
            }
        else:
            # Generic sensor pattern
            pattern = {
                'base_value': 0.5,
                'noise_level': 0.1,
                'trend_factor': 0.0,
                'seasonal_amplitude': 0.2,
                'seasonal_period': 100,
                'anomaly_probability': 0.02
            }

        # Add some randomization
        pattern['base_value'] += random.uniform(-0.1, 0.1)
        pattern['noise_level'] *= random.uniform(0.8, 1.2)
        pattern['seasonal_period'] = int(pattern['seasonal_period'] * random.uniform(0.8, 1.2))

        return pattern

    def _generate_realistic_value(self, equipment_id: str, timestamp: datetime) -> Tuple[float, bool]:
        """Generate realistic sensor value for equipment

        Args:
            equipment_id: Equipment identifier
            timestamp: Current timestamp

        Returns:
            Tuple of (sensor_value, is_anomalous)
        """
        if equipment_id not in self.equipment_patterns:
            # Fallback to simple random
            return random.uniform(0.0, 1.0), False

        pattern = self.equipment_patterns[equipment_id]

        # Time-based components
        time_factor = timestamp.timestamp() % pattern['seasonal_period']
        seasonal = pattern['seasonal_amplitude'] * np.sin(2 * np.pi * time_factor / pattern['seasonal_period'])

        # Trend component
        trend = pattern['trend_factor'] * time_factor

        # Noise component
        noise = random.gauss(0, pattern['noise_level'])

        # Base value
        base_value = pattern['base_value'] + seasonal + trend + noise

        # Check for anomaly
        is_anomaly = random.random() < pattern['anomaly_probability']
        if is_anomaly:
            # Introduce anomalous behavior
            anomaly_type = random.choice(['spike', 'drop', 'drift'])
            if anomaly_type == 'spike':
                base_value += random.uniform(0.3, 0.8)
            elif anomaly_type == 'drop':
                base_value -= random.uniform(0.3, 0.8)
            else:  # drift
                base_value += random.uniform(-0.4, 0.4)

        # Clamp to [0, 1] range
        base_value = max(0.0, min(1.0, base_value))

        return base_value, is_anomaly

    def _generate_telemetry_batch(self) -> List[Dict[str, Any]]:
        """Generate batch of telemetry data"""
        current_time = datetime.now()
        telemetry_batch = []

        # Select random subset of equipment for this batch
        all_equipment = []
        for spacecraft, equipment_list in self.available_equipment.items():
            all_equipment.extend(equipment_list)

        selected_equipment = random.sample(
            all_equipment,
            min(self.config.batch_size, len(all_equipment))
        )

        for equipment_id in selected_equipment:
            # Parse equipment ID
            if '_' in equipment_id:
                spacecraft, channel = equipment_id.split('_', 1)
            else:
                spacecraft, channel = 'UNKNOWN', equipment_id

            # Generate realistic sensor data array
            sequence_length = random.randint(10, 50)
            sensor_values = []
            is_batch_anomalous = False

            for i in range(sequence_length):
                timestamp_offset = current_time + timedelta(seconds=i)
                value, is_anomaly = self._generate_realistic_value(equipment_id, timestamp_offset)
                sensor_values.append(value)
                if is_anomaly:
                    is_batch_anomalous = True

            # Create telemetry record
            telemetry_record = {
                'timestamp': current_time,
                'spacecraft': spacecraft,
                'channel': channel,
                'sequence_id': int(time.time()) % 10000,  # Simple sequence ID
                'data': sensor_values,
                'data_shape': f"({len(sensor_values)},)",
                'mean_value': float(np.mean(sensor_values)),
                'std_value': float(np.std(sensor_values)),
                'min_value': float(np.min(sensor_values)),
                'max_value': float(np.max(sensor_values)),
                'is_anomaly': is_batch_anomalous,
                'anomaly_score': 0.0,  # Will be updated by ML model
                'event_metadata': {
                    'equipment_id': equipment_id,
                    'generated_at': current_time.isoformat(),
                    'sequence_length': sequence_length,
                    'simulation_source': 'realtime_streaming'
                }
            }

            telemetry_batch.append(telemetry_record)

        return telemetry_batch

    def _stream_telemetry_worker(self):
        """Worker thread for streaming telemetry data"""
        logger.info("Telemetry streaming worker started")

        while not self._stop_event.is_set():
            try:
                if self.is_paused:
                    self._pause_event.wait()
                    continue

                # Generate batch of telemetry data
                telemetry_batch = self._generate_telemetry_batch()

                # Add to queue for database insertion
                for record in telemetry_batch:
                    if not self.telemetry_queue.full():
                        self.telemetry_queue.put(record)
                    else:
                        logger.warning("Telemetry queue is full, dropping record")

                # Update statistics
                self.stats['records_streamed'] += len(telemetry_batch)
                self.stats['queue_size'] = self.telemetry_queue.qsize()
                self.stats['last_update'] = datetime.now()

                # Wait before next batch
                time.sleep(self.config.refresh_interval)

            except Exception as e:
                logger.error(f"Error in telemetry streaming worker: {e}")
                self.error_count += 1
                time.sleep(self.config.refresh_interval)

        logger.info("Telemetry streaming worker stopped")

    def _database_writer_worker(self):
        """Worker thread for writing data to database"""
        logger.info("Database writer worker started")

        batch_buffer = []
        last_write_time = time.time()
        write_interval = 5.0  # Write to database every 5 seconds

        while not self._stop_event.is_set():
            try:
                # Collect records from queue
                try:
                    record = self.telemetry_queue.get(timeout=1.0)
                    batch_buffer.append(record)
                except Empty:
                    pass

                # Write to database when buffer is full or enough time has passed
                current_time = time.time()
                should_write = (
                    len(batch_buffer) >= self.config.batch_size or
                    (batch_buffer and current_time - last_write_time >= write_interval)
                )

                if should_write:
                    try:
                        # Insert batch to database
                        self.db_manager.insert_telemetry_batch(batch_buffer)
                        logger.debug(f"Inserted {len(batch_buffer)} records to database")

                        # Clear buffer
                        batch_buffer.clear()
                        last_write_time = current_time

                    except Exception as e:
                        logger.error(f"Failed to insert batch to database: {e}")
                        # Keep records in buffer for retry
                        time.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in database writer worker: {e}")
                self.error_count += 1
                time.sleep(1.0)

        # Write remaining records
        if batch_buffer:
            try:
                self.db_manager.insert_telemetry_batch(batch_buffer)
                logger.info(f"Final insert: {len(batch_buffer)} records")
            except Exception as e:
                logger.error(f"Failed to insert final batch: {e}")

        logger.info("Database writer worker stopped")

    def _anomaly_detection_worker(self):
        """Worker thread for real-time anomaly detection"""
        logger.info("Anomaly detection worker started")

        while not self._stop_event.is_set():
            try:
                if not self.config.anomaly_detection_enabled:
                    time.sleep(5.0)
                    continue

                # Get recent data for anomaly detection
                recent_records = []
                try:
                    for _ in range(min(10, self.telemetry_queue.qsize())):
                        record = self.telemetry_queue.get_nowait()
                        recent_records.append(record)
                        # Put back in queue for database writer
                        if not self.telemetry_queue.full():
                            self.telemetry_queue.put(record)
                except Empty:
                    pass

                # Run anomaly detection on recent records
                for record in recent_records:
                    equipment_id = record.get('event_metadata', {}).get('equipment_id', '')
                    sensor_data = record.get('data', [])

                    if sensor_data and equipment_id:
                        try:
                            # Prepare data for model (reshape for LSTM)
                            sensor_array = np.array(sensor_data).reshape(1, -1, 1)

                            # Use pretrained model for anomaly detection
                            result = pretrained_model_manager.predict_anomaly(
                                equipment_id, sensor_array
                            )

                            if result and not result.get('error'):
                                # Update anomaly score in record
                                record['anomaly_score'] = result.get('anomaly_score', 0.0)
                                record['is_anomaly'] = result.get('is_anomaly', False)

                                # Track anomaly statistics
                                if result.get('is_anomaly'):
                                    self.stats['anomalies_detected'] += 1

                        except Exception as e:
                            logger.debug(f"Anomaly detection failed for {equipment_id}: {e}")

                time.sleep(2.0)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Error in anomaly detection worker: {e}")
                self.error_count += 1
                time.sleep(5.0)

        logger.info("Anomaly detection worker stopped")

    def start(self):
        """Start the real-time streaming service"""
        if self.is_running:
            logger.warning("Streaming service is already running")
            return

        self.is_running = True
        self.is_paused = False
        self._stop_event.clear()
        self._pause_event.set()  # Initially not paused

        self.stats['start_time'] = datetime.now()
        self.stats['records_streamed'] = 0
        self.stats['anomalies_detected'] = 0

        # Start worker threads
        self._threads = [
            threading.Thread(target=self._stream_telemetry_worker, name="TelemetryStreamer"),
            threading.Thread(target=self._database_writer_worker, name="DatabaseWriter"),
            threading.Thread(target=self._anomaly_detection_worker, name="AnomalyDetector")
        ]

        for thread in self._threads:
            thread.start()

        logger.info("Real-time streaming service started")

    def stop(self):
        """Stop the real-time streaming service"""
        if not self.is_running:
            logger.warning("Streaming service is not running")
            return

        logger.info("Stopping real-time streaming service...")

        # Signal threads to stop
        self._stop_event.set()
        self._pause_event.set()  # Unpause if paused

        # Wait for threads to finish
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not stop gracefully")

        self.is_running = False
        self.is_paused = False

        # Calculate final statistics
        if self.stats['start_time']:
            runtime = (datetime.now() - self.stats['start_time']).total_seconds()
            if runtime > 0:
                self.stats['processing_rate'] = self.stats['records_streamed'] / runtime

        logger.info("Real-time streaming service stopped")

    def pause(self):
        """Pause the streaming service"""
        if not self.is_running:
            return

        self.is_paused = True
        self._pause_event.clear()
        logger.info("Real-time streaming service paused")

    def resume(self):
        """Resume the streaming service"""
        if not self.is_running:
            return

        self.is_paused = False
        self._pause_event.set()
        logger.info("Real-time streaming service resumed")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        stats = self.stats.copy()
        stats['is_running'] = self.is_running
        stats['is_paused'] = self.is_paused
        stats['queue_size'] = self.telemetry_queue.qsize()

        # Enhanced statistics for pipeline monitor
        stats['telemetry_queue_size'] = self.telemetry_queue.qsize()
        stats['anomaly_queue_size'] = getattr(self, 'anomaly_queue', queue.Queue()).qsize()
        stats['data_sources'] = {
            'smap_active': True,
            'msl_active': True,
            'models_active': True
        }

        # Calculate processing rate and runtime
        if self.stats['start_time']:
            runtime = (datetime.now() - self.stats['start_time']).total_seconds()
            stats['runtime_seconds'] = runtime
            stats['runtime_hours'] = runtime / 3600
            if runtime > 0:
                stats['processing_rate'] = self.stats['records_streamed'] / runtime
                # Calculate instantaneous rate (last 60 seconds)
                recent_time = 60  # seconds
                if runtime >= recent_time:
                    stats['recent_processing_rate'] = self.stats['records_streamed'] / min(runtime, recent_time)
                else:
                    stats['recent_processing_rate'] = stats['processing_rate']
            else:
                stats['processing_rate'] = 0.0
                stats['recent_processing_rate'] = 0.0
        else:
            stats['runtime_seconds'] = 0
            stats['runtime_hours'] = 0
            stats['processing_rate'] = 0.0
            stats['recent_processing_rate'] = 0.0

        # Calculate anomaly rate
        if self.stats['records_streamed'] > 0:
            stats['anomaly_rate'] = self.stats['anomalies_detected'] / self.stats['records_streamed']
        else:
            stats['anomaly_rate'] = 0.0

        # Add error tracking
        stats['error_count'] = getattr(self, 'error_count', 0)

        return stats

    def configure(self, config: StreamConfig):
        """Update streaming configuration"""
        old_config = self.config
        self.config = config

        logger.info(f"Streaming service configuration updated")
        logger.debug(f"Config changed: {old_config} -> {config}")


# Global instance for dashboard integration
realtime_streaming_service = RealTimeStreamingService()


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Streaming Service")
    parser.add_argument('--duration', type=int, default=60,
                       help='Streaming duration in seconds')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for telemetry generation')
    parser.add_argument('--refresh-interval', type=float, default=1.0,
                       help='Refresh interval in seconds')

    args = parser.parse_args()

    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure service
    config = StreamConfig(
        batch_size=args.batch_size,
        refresh_interval=args.refresh_interval,
        anomaly_detection_enabled=True
    )

    service = RealTimeStreamingService(config=config)

    try:
        print(f"Starting real-time streaming for {args.duration} seconds...")
        service.start()

        # Monitor for specified duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            stats = service.get_statistics()
            print(f"Records streamed: {stats['records_streamed']}, "
                  f"Anomalies: {stats['anomalies_detected']}, "
                  f"Rate: {stats.get('processing_rate', 0):.1f}/sec, "
                  f"Queue: {stats['queue_size']}")
            time.sleep(10)

        print("Stopping streaming service...")
        service.stop()

        # Final statistics
        final_stats = service.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Total records: {final_stats['records_streamed']}")
        print(f"  Total anomalies: {final_stats['anomalies_detected']}")
        print(f"  Processing rate: {final_stats.get('processing_rate', 0):.2f} records/sec")
        print(f"  Runtime: {final_stats.get('runtime_seconds', 0):.1f} seconds")

    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        service.stop()
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        service.stop()
        raise


if __name__ == "__main__":
    main()