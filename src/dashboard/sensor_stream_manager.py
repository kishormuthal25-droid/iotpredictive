"""
Sensor Stream Manager for Real-Time Individual Sensor Data Streaming
Manages individual sensor data streams for all 80 NASA SMAP/MSL sensors with anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
import asyncio
import aiohttp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import optimized memory management
from src.utils.memory_manager import (
    memory_manager,
    add_sensor_data,
    get_sensor_data,
    get_all_sensors_data,
    optimize_for_sensors
)

# Import async processing capabilities
from src.utils.async_processor import (
    async_processor,
    process_sensors_async,
    detect_anomalies_async,
    async_sensor_task
)

from src.data_ingestion.equipment_mapper import equipment_mapper
from src.data_ingestion.nasa_data_service import nasa_data_service
from src.data_ingestion.unified_data_access import (
    unified_data_access, DataQuery, TimeRange, TelemetryRecord
)
from src.dashboard.model_manager import pretrained_model_manager

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging for sensor stream manager


@dataclass
class SensorStream:
    """Individual sensor data stream with optimized memory management"""
    sensor_id: str
    sensor_name: str
    equipment_id: str
    subsystem: str
    unit: str
    min_value: float
    max_value: float
    nominal_value: float
    critical_threshold: float

    # Stream metadata (data stored in memory manager)
    last_update: datetime
    update_frequency: float  # Hz
    buffer_size: int
    is_active: bool

    # Performance metrics
    data_points_added: int = 0
    anomalies_detected: int = 0
    memory_usage_bytes: int = 0

    def add_data_point(self, timestamp: datetime, value: float,
                      anomaly_score: float = 0.0, anomaly_flag: bool = False) -> bool:
        """Add data point using optimized memory manager"""
        success = add_sensor_data(self.sensor_id, timestamp, value, anomaly_score, anomaly_flag)

        if success:
            self.last_update = timestamp
            self.data_points_added += 1
            if anomaly_flag:
                self.anomalies_detected += 1

        return success

    def get_recent_data(self, count: int = None) -> Optional[Dict[str, np.ndarray]]:
        """Get recent data points using memory manager"""
        return get_sensor_data(self.sensor_id, count)

    def get_time_range_data(self, start_time: datetime, end_time: datetime) -> Optional[Dict[str, np.ndarray]]:
        """Get data within time range using memory manager"""
        return memory_manager.get_sensor_time_range(self.sensor_id, start_time, end_time)

    def get_latest_value(self) -> Optional[float]:
        """Get the most recent sensor value"""
        data = self.get_recent_data(1)
        if data and len(data['values']) > 0:
            return float(data['values'][-1])
        return None

    def get_data_count(self) -> int:
        """Get current number of data points"""
        data = self.get_recent_data(1)
        return len(data['values']) if data else 0


@dataclass
class SensorAnomalyResult:
    """Result of sensor-level anomaly detection"""
    sensor_id: str
    timestamp: datetime
    value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    confidence: float
    threshold_exceeded: bool
    model_name: str


class SensorStreamManager:
    """
    Manages individual sensor data streams for real-time visualization
    Provides comprehensive streaming for all 80 NASA sensors
    """

    def __init__(self, buffer_size: int = 1000, update_frequency: float = 1.0):
        """
        Initialize sensor stream manager with optimized memory management

        Args:
            buffer_size: Maximum number of data points per sensor stream
            update_frequency: Update frequency in Hz
        """
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency

        # Core components
        self.equipment_mapper = equipment_mapper
        self.nasa_service = nasa_data_service

        # Sensor streams storage (metadata only, data in memory manager)
        self.sensor_streams: Dict[str, SensorStream] = {}
        self.equipment_sensors: Dict[str, List[str]] = defaultdict(list)
        self.subsystem_sensors: Dict[str, List[str]] = defaultdict(list)

        # Streaming control
        self.is_streaming = False
        self.stream_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Async processing support
        self.async_loop = None
        self.async_tasks = set()
        self.sensor_queues: Dict[str, asyncio.Queue] = {}
        self.async_enabled = True

        # Performance metrics
        self.stream_stats = {
            'total_updates': 0,
            'anomalies_detected': 0,
            'last_update_time': None,
            'update_rate': 0.0,
            'active_sensors': 0,
            'memory_usage_mb': 0.0,
            'avg_window_size': 0.0
        }

        # Initialize sensor streams
        self._initialize_sensor_streams()

        # Optimize memory manager for sensor count
        sensor_count = len(self.sensor_streams)
        optimize_for_sensors(sensor_count)

        logger.info(f"Initialized SensorStreamManager with {sensor_count} sensors")
        logger.info(f"SMAP sensors: {len([s for s in self.sensor_streams if 'SMAP' in s])}")
        logger.info(f"MSL sensors: {len([s for s in self.sensor_streams if 'MSL' in s])}")
        logger.info(f"Memory manager optimized for {sensor_count} sensors")

    def _initialize_sensor_streams(self):
        """Initialize individual sensor streams for all equipment"""
        all_equipment = self.equipment_mapper.get_all_equipment()

        sensor_id_counter = 0

        for equipment in all_equipment:
            equipment_id = equipment.equipment_id

            for sensor_spec in equipment.sensors:
                # Create unique sensor ID
                sensor_id = f"{equipment_id}_{sensor_spec.name.replace(' ', '_').replace('.', '').lower()}"

                # Create sensor stream (data will be stored in memory manager)
                sensor_stream = SensorStream(
                    sensor_id=sensor_id,
                    sensor_name=sensor_spec.name,
                    equipment_id=equipment_id,
                    subsystem=equipment.subsystem,
                    unit=sensor_spec.unit,
                    min_value=sensor_spec.min_value,
                    max_value=sensor_spec.max_value,
                    nominal_value=sensor_spec.nominal_value,
                    critical_threshold=sensor_spec.critical_threshold,
                    last_update=datetime.now(),
                    update_frequency=self.update_frequency,
                    buffer_size=self.buffer_size,
                    is_active=True
                )

                # Create sliding window in memory manager
                memory_manager.create_sensor_window(sensor_id, self.buffer_size)

                self.sensor_streams[sensor_id] = sensor_stream
                self.equipment_sensors[equipment_id].append(sensor_id)
                self.subsystem_sensors[equipment.subsystem].append(sensor_id)

                sensor_id_counter += 1

        self.stream_stats['active_sensors'] = len(self.sensor_streams)
        logger.info(f"Initialized {sensor_id_counter} individual sensor streams")

    def start_streaming(self):
        """Start real-time sensor data streaming with async support"""
        if self.is_streaming:
            logger.warning("Sensor streaming is already active")
            return

        self.is_streaming = True

        # Start async processing if enabled
        if self.async_enabled:
            # Create event loop in separate thread
            self.stream_thread = threading.Thread(target=self._async_stream_wrapper, daemon=True)
        else:
            # Fallback to traditional threading
            self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)

        self.stream_thread.start()

        logger.info(f"Started real-time sensor streaming (async={'enabled' if self.async_enabled else 'disabled'})")

    def _async_stream_wrapper(self):
        """Wrapper to run async streaming in separate thread"""
        try:
            # Create new event loop for this thread
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)

            # Start async processor
            self.async_loop.run_until_complete(async_processor.start())

            # Run main async streaming loop
            self.async_loop.run_until_complete(self._async_stream_loop())

        except Exception as e:
            logger.error(f"Async streaming error: {e}")
        finally:
            if self.async_loop:
                self.async_loop.close()

    async def _async_stream_loop(self):
        """Main async streaming loop for 80-sensor processing"""
        logger.info("Starting async sensor streaming for 80 sensors")

        # Initialize sensor queues
        for sensor_id in self.sensor_streams.keys():
            self.sensor_queues[sensor_id] = asyncio.Queue(maxsize=100)

        while self.is_streaming:
            try:
                # Get latest NASA data
                telemetry_data = self._get_latest_nasa_telemetry()

                if telemetry_data:
                    # Process sensor data in parallel batches
                    await self._async_update_sensor_streams(telemetry_data)

                    # Update performance metrics
                    self.stream_stats['total_updates'] += 1
                    self.stream_stats['last_update_time'] = datetime.now()

                # Sleep to maintain update frequency
                await asyncio.sleep(1.0 / self.update_frequency)

            except Exception as e:
                logger.error(f"Error in async sensor streaming loop: {e}")
                await asyncio.sleep(1.0)  # Error recovery delay

    async def _async_update_sensor_streams(self, telemetry_data: List[Dict[str, Any]]):
        """Update sensor streams using async parallel processing"""
        current_time = datetime.now()

        # Group telemetry data by equipment for efficient processing
        equipment_data = defaultdict(list)
        for record in telemetry_data:
            equipment_id = record.get('equipment_id')
            if equipment_id:
                equipment_data[equipment_id].append(record)

        # Process equipment in parallel batches
        batch_size = 20  # Process 20 equipment units at a time
        equipment_items = list(equipment_data.items())

        for i in range(0, len(equipment_items), batch_size):
            batch = equipment_items[i:i + batch_size]

            # Create async tasks for this batch
            tasks = []
            for equipment_id, records in batch:
                task = asyncio.create_task(
                    self._process_equipment_async(equipment_id, records, current_time)
                )
                tasks.append(task)

            # Execute batch in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    equipment_id = batch[j][0]
                    logger.error(f"Async processing failed for equipment {equipment_id}: {result}")

    async def _process_equipment_async(self, equipment_id: str,
                                     records: List[Dict[str, Any]],
                                     current_time: datetime):
        """Process equipment data asynchronously"""
        if equipment_id not in self.equipment_sensors:
            return

        sensors_updated = 0
        equipment_sensor_ids = self.equipment_sensors[equipment_id]

        # Process all records for this equipment
        for record in records:
            timestamp = record.get('timestamp', current_time)
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # Extract sensor values
            sensor_values = {}
            skip_fields = {
                'timestamp', 'equipment_id', 'equipment_type', 'subsystem',
                'criticality', 'anomaly_score', 'is_anomaly', 'model_name'
            }

            for key, value in record.items():
                if key not in skip_fields and isinstance(value, (int, float)):
                    sensor_values[key] = value

            # Process sensors for this equipment in parallel
            sensor_tasks = []
            sensor_values_list = list(sensor_values.values())

            for i, sensor_id in enumerate(equipment_sensor_ids):
                if sensor_id in self.sensor_streams:
                    # Determine sensor value using strategy from original implementation
                    sensor_value = None
                    sensor_stream = self.sensor_streams[sensor_id]

                    # Strategy 1: Name matching
                    for sensor_name, value in sensor_values.items():
                        if self._match_sensor_name(sensor_stream.sensor_name, sensor_name):
                            sensor_value = float(value)
                            break

                    # Strategy 2: Index-based mapping
                    if sensor_value is None and i < len(sensor_values_list):
                        sensor_value = float(sensor_values_list[i])

                    if sensor_value is not None:
                        # Create async task for sensor processing
                        task = asyncio.create_task(
                            self._process_sensor_data_async(sensor_stream, sensor_value, timestamp)
                        )
                        sensor_tasks.append(task)

            # Execute sensor processing in parallel
            if sensor_tasks:
                await asyncio.gather(*sensor_tasks, return_exceptions=True)
                sensors_updated += len(sensor_tasks)

        # Debug logging
        if equipment_id == "SMAP-PWR-001":
            logger.debug(f"Async updated {sensors_updated}/{len(equipment_sensor_ids)} sensors for {equipment_id}")

    @async_sensor_task
    async def _process_sensor_data_async(self, sensor_stream: SensorStream,
                                       sensor_value: float, timestamp: datetime):
        """Process individual sensor data asynchronously"""
        try:
            # Perform sensor-level anomaly detection
            anomaly_result = await self._detect_sensor_anomaly_async(
                sensor_stream, sensor_value, timestamp
            )

            # Update sensor stream using memory manager
            success = sensor_stream.add_data_point(
                timestamp, sensor_value,
                anomaly_result.anomaly_score,
                anomaly_result.is_anomaly
            )

            if success and anomaly_result.is_anomaly:
                self.stream_stats['anomalies_detected'] += 1

            return success

        except Exception as e:
            logger.error(f"Async sensor processing error for {sensor_stream.sensor_id}: {e}")
            return False

    async def _detect_sensor_anomaly_async(self, sensor_stream: SensorStream,
                                         value: float, timestamp: datetime) -> SensorAnomalyResult:
        """Async version of anomaly detection"""
        # Run the existing anomaly detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._detect_sensor_anomaly,
            sensor_stream,
            value,
            timestamp
        )
        return result

    def stop_streaming(self):
        """Stop real-time sensor data streaming"""
        self.is_streaming = False

        # Clean up async resources
        if self.async_loop and self.async_enabled:
            try:
                # Schedule cleanup in the async loop
                future = asyncio.run_coroutine_threadsafe(
                    self._async_cleanup(), self.async_loop
                )
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Error during async cleanup: {e}")

        if self.stream_thread:
            self.stream_thread.join(timeout=5.0)

        logger.info("Stopped sensor streaming")

    async def _async_cleanup(self):
        """Clean up async resources"""
        try:
            # Stop async processor
            await async_processor.stop()

            # Cancel any remaining tasks
            if self.async_tasks:
                for task in self.async_tasks:
                    if not task.done():
                        task.cancel()

                await asyncio.gather(*self.async_tasks, return_exceptions=True)
                self.async_tasks.clear()

            # Clear sensor queues
            self.sensor_queues.clear()

        except Exception as e:
            logger.error(f"Async cleanup error: {e}")

    def toggle_async_processing(self, enabled: bool = True):
        """Enable or disable async processing"""
        was_streaming = self.is_streaming

        if was_streaming:
            self.stop_streaming()

        self.async_enabled = enabled

        if was_streaming:
            self.start_streaming()

        logger.info(f"Async processing {'enabled' if enabled else 'disabled'}")

    def get_async_performance_stats(self) -> Dict[str, Any]:
        """Get async processor performance statistics"""
        if not self.async_enabled:
            return {'async_enabled': False}

        try:
            return {
                'async_enabled': True,
                'async_processor_stats': async_processor.get_performance_stats(),
                'sensor_queues': len(self.sensor_queues),
                'active_async_tasks': len(self.async_tasks)
            }
        except Exception as e:
            logger.error(f"Error getting async stats: {e}")
            return {'async_enabled': True, 'error': str(e)}

    def _stream_loop(self):
        """Main streaming loop"""
        last_update = time.time()

        while self.is_streaming:
            try:
                current_time = time.time()

                # Get latest NASA data
                telemetry_data = self._get_latest_nasa_telemetry()

                if telemetry_data:
                    # Update sensor streams with new data
                    self._update_sensor_streams(telemetry_data)

                    # Update performance metrics
                    self.stream_stats['total_updates'] += 1
                    self.stream_stats['last_update_time'] = datetime.now()
                    self.stream_stats['update_rate'] = 1.0 / (current_time - last_update) if current_time > last_update else 0

                last_update = current_time

                # Sleep to maintain update frequency
                sleep_time = max(0, (1.0 / self.update_frequency) - (time.time() - current_time))
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in sensor streaming loop: {e}")
                time.sleep(1.0)  # Error recovery delay

    def _get_latest_nasa_telemetry(self) -> List[Dict[str, Any]]:
        """Get latest NASA telemetry data"""
        try:
            # Get real-time data from NASA service
            telemetry_data = self.nasa_service.get_real_time_telemetry(
                time_window="30s",
                max_records=50
            )

            return telemetry_data

        except Exception as e:
            logger.error(f"Error getting NASA telemetry: {e}")
            return []

    def _update_sensor_streams(self, telemetry_data: List[Dict[str, Any]]):
        """Update individual sensor streams with new telemetry data"""
        current_time = datetime.now()

        for record in telemetry_data:
            equipment_id = record.get('equipment_id')
            timestamp = record.get('timestamp', current_time)

            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

            # Extract sensor values from flattened record format
            # The NASA data service flattens sensor_values into the main record
            sensor_values = {}

            # Skip non-sensor fields
            skip_fields = {
                'timestamp', 'equipment_id', 'equipment_type', 'subsystem',
                'criticality', 'anomaly_score', 'is_anomaly', 'model_name'
            }

            for key, value in record.items():
                if key not in skip_fields and isinstance(value, (int, float)):
                    sensor_values[key] = value

            # Debug logging to understand the data structure
            if equipment_id == "SMAP-PWR-001" and sensor_values:  # Log first equipment to avoid spam
                logger.debug(f"Equipment {equipment_id}: Found sensor values: {list(sensor_values.keys())}")

            # Update each sensor stream for this equipment
            if equipment_id in self.equipment_sensors:
                sensors_updated = 0
                equipment_sensor_ids = self.equipment_sensors[equipment_id]
                sensor_values_list = list(sensor_values.values())

                for i, sensor_id in enumerate(equipment_sensor_ids):
                    sensor_stream = self.sensor_streams[sensor_id]

                    # Strategy 1: Try name matching first
                    sensor_value = None
                    matched_key = None
                    for sensor_name, value in sensor_values.items():
                        if self._match_sensor_name(sensor_stream.sensor_name, sensor_name):
                            sensor_value = float(value)
                            matched_key = sensor_name
                            break

                    # Strategy 2: If no name match, use index-based mapping
                    # This ensures all sensors get data even if names don't match
                    if sensor_value is None and i < len(sensor_values_list):
                        sensor_value = float(sensor_values_list[i])
                        matched_key = f"sensor_{i}"

                    if sensor_value is not None:
                        # Perform sensor-level anomaly detection
                        anomaly_result = self._detect_sensor_anomaly(
                            sensor_stream, sensor_value, timestamp
                        )

                        # Update sensor stream using memory manager
                        success = sensor_stream.add_data_point(
                            timestamp, sensor_value,
                            anomaly_result.anomaly_score,
                            anomaly_result.is_anomaly
                        )

                        if success:
                            # Update anomaly count
                            if anomaly_result.is_anomaly:
                                self.stream_stats['anomalies_detected'] += 1
                        else:
                            logger.warning(f"Failed to add data point for sensor {sensor_id}")

                        sensors_updated += 1
                    else:
                        # Log failed matches for debugging
                        if equipment_id == "SMAP-PWR-001":  # Log only for one equipment to avoid spam
                            logger.debug(f"No match for sensor '{sensor_stream.sensor_name}' in available keys: {list(sensor_values.keys())}")

                if equipment_id == "SMAP-PWR-001":
                    logger.debug(f"Updated {sensors_updated}/{len(self.equipment_sensors[equipment_id])} sensors for {equipment_id}")
            else:
                if equipment_id:  # Only log if equipment_id is not None
                    logger.debug(f"Equipment {equipment_id} not found in equipment_sensors")

    def _match_sensor_name(self, stream_sensor_name: str, data_sensor_name: str) -> bool:
        """Match sensor names between stream and telemetry data"""
        # Normalize names for comparison
        stream_name = stream_sensor_name.lower().replace(' ', '_').replace('.', '').replace('-', '')
        data_name = data_sensor_name.lower().replace(' ', '_').replace('.', '').replace('-', '')

        # Try exact match first
        if stream_name == data_name:
            return True

        # Try partial matches (both directions)
        if stream_name in data_name or data_name in stream_name:
            return True

        # Try matching with common sensor prefixes/suffixes
        # For NASA data, sensors might be numbered like sensor_1, sensor_2, etc.
        if 'sensor' in data_name and any(keyword in stream_name for keyword in ['voltage', 'current', 'temperature', 'pressure', 'flow']):
            return True

        return False

    def _detect_sensor_anomaly(self, sensor_stream: SensorStream,
                              value: float, timestamp: datetime) -> SensorAnomalyResult:
        """Detect anomalies for individual sensor using pretrained models"""
        try:
            # Initialize default values
            anomaly_score = 0.0
            is_anomaly = False
            severity = "NORMAL"
            confidence = 0.5
            threshold_exceeded = False

            # 1. Use pretrained model if available for this equipment
            equipment_id = sensor_stream.equipment_id
            if equipment_id in pretrained_model_manager.get_available_models():
                try:
                    # Prepare sensor data for model prediction
                    # Need sequence of values for LSTM model (shape: [1, sequence_length, n_features])
                    recent_data = sensor_stream.get_recent_data(50)  # Get last 50 values
                    if recent_data and len(recent_data['values']) >= 10:  # Need minimum history
                        recent_values = recent_data['values']

                        # Create sensor array for this equipment
                        # Get equipment info to know how many sensors
                        equipment_info = equipment_mapper.get_equipment_info(equipment_id)
                        n_sensors = len(equipment_info.get('sensors', [])) if equipment_info else 5

                        # Create normalized sensor data array
                        sensor_data = np.zeros((1, len(recent_values), n_sensors))

                        # Fill with normalized values (simple min-max normalization)
                        values_array = np.array(recent_values)
                        if sensor_stream.max_value > sensor_stream.min_value:
                            normalized_values = (values_array - sensor_stream.min_value) / (sensor_stream.max_value - sensor_stream.min_value)
                        else:
                            normalized_values = values_array / max(abs(sensor_stream.max_value), 1.0)

                        # Fill first sensor channel with this sensor's data
                        sensor_data[0, :, 0] = np.clip(normalized_values, 0.0, 1.0)

                        # Fill other channels with simulated correlated data
                        for i in range(1, n_sensors):
                            noise = np.random.normal(0, 0.1, len(recent_values))
                            sensor_data[0, :, i] = np.clip(normalized_values + noise, 0.0, 1.0)

                        # Get prediction from pretrained model
                        prediction = pretrained_model_manager.predict_anomaly(equipment_id, sensor_data)

                        if 'error' not in prediction:
                            anomaly_score = prediction.get('anomaly_score', 0.0)
                            is_anomaly = prediction.get('is_anomaly', False)
                            confidence = 0.9  # High confidence if using actual model

                            logger.debug(f"Model prediction for {equipment_id}: score={anomaly_score:.3f}, anomaly={is_anomaly}")
                        else:
                            logger.debug(f"Model prediction error for {equipment_id}: {prediction['error']}")

                except Exception as model_error:
                    logger.warning(f"Error using pretrained model for {equipment_id}: {model_error}")
                    # Fall back to statistical methods

            # 2. Threshold-based detection (always check)
            if value > sensor_stream.critical_threshold:
                threshold_exceeded = True
                anomaly_score = max(anomaly_score, 0.4)

            # 3. Statistical anomaly detection (if we have enough history and no model prediction)
            statistical_data = sensor_stream.get_recent_data(100)  # Get recent data for statistics
            if statistical_data and len(statistical_data['values']) > 10 and anomaly_score == 0.0:
                values_array = statistical_data['values']
                mean_val = np.mean(values_array)
                std_val = np.std(values_array)

                if std_val > 0:
                    z_score = abs((value - mean_val) / std_val)
                    if z_score > 3:  # 3-sigma rule
                        anomaly_score += 0.3
                    elif z_score > 2:
                        anomaly_score += 0.2

            # 3. Range-based detection
            value_range = sensor_stream.max_value - sensor_stream.min_value
            if value_range > 0:
                normalized_deviation = abs(value - sensor_stream.nominal_value) / value_range
                if normalized_deviation > 0.8:
                    anomaly_score += 0.2
                elif normalized_deviation > 0.6:
                    anomaly_score += 0.1

            # 4. Trend-based detection (if we have recent history)
            trend_data = sensor_stream.get_recent_data(5)  # Get last 5 values for trend
            if trend_data and len(trend_data['values']) > 5:
                recent_values = trend_data['values']
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]

                # Detect rapid changes
                if abs(trend) > value_range * 0.1:  # 10% of range per time step
                    anomaly_score += 0.1

            # Determine if this is an anomaly
            is_anomaly = anomaly_score > 0.3
            confidence = min(0.95, max(0.1, anomaly_score))

            # Determine severity
            if anomaly_score > 0.7:
                severity = "CRITICAL"
            elif anomaly_score > 0.5:
                severity = "HIGH"
            elif anomaly_score > 0.3:
                severity = "MEDIUM"
            elif anomaly_score > 0.1:
                severity = "LOW"

            return SensorAnomalyResult(
                sensor_id=sensor_stream.sensor_id,
                timestamp=timestamp,
                value=value,
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                severity=severity,
                confidence=confidence,
                threshold_exceeded=threshold_exceeded,
                model_name="SensorThreshold+Statistical"
            )

        except Exception as e:
            logger.error(f"Error detecting sensor anomaly: {e}")
            return SensorAnomalyResult(
                sensor_id=sensor_stream.sensor_id,
                timestamp=timestamp,
                value=value,
                anomaly_score=0.0,
                is_anomaly=False,
                severity="NORMAL",
                confidence=0.0,
                threshold_exceeded=False,
                model_name="Error"
            )

    def get_sensor_data(self, sensor_ids: Optional[List[str]] = None,
                       time_window: str = "5min",
                       use_database: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get sensor data for specified sensors and time window

        Args:
            sensor_ids: List of sensor IDs to retrieve (None for all)
            time_window: Time window for data ("1min", "5min", "15min", "1hour")
            use_database: Whether to use database via unified data access (preferred)

        Returns:
            Dictionary of sensor data
        """
        if use_database:
            return self.get_sensor_data_from_database(sensor_ids, time_window)

        # Fallback to original in-memory stream approach
        result = {}

        # Parse time window
        time_delta = self._parse_time_window(time_window)
        cutoff_time = datetime.now() - time_delta

        # Get sensors to process
        sensors_to_process = sensor_ids or list(self.sensor_streams.keys())

        for sensor_id in sensors_to_process:
            if sensor_id not in self.sensor_streams:
                continue

            sensor_stream = self.sensor_streams[sensor_id]

            # Get data using memory manager with time range
            filtered_data = sensor_stream.get_time_range_data(cutoff_time, datetime.now())

            if filtered_data:
                # Convert to expected format
                formatted_data = {
                    'timestamps': [ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) for ts in filtered_data['timestamps']],
                    'values': filtered_data['values'].tolist(),
                    'anomaly_scores': filtered_data['anomaly_scores'].tolist(),
                    'anomaly_flags': filtered_data['anomaly_flags'].tolist()
                }
            else:
                formatted_data = {
                    'timestamps': [],
                    'values': [],
                    'anomaly_scores': [],
                    'anomaly_flags': []
                }

            result[sensor_id] = {
                'sensor_name': sensor_stream.sensor_name,
                'equipment_id': sensor_stream.equipment_id,
                'subsystem': sensor_stream.subsystem,
                'unit': sensor_stream.unit,
                'data': formatted_data,
                'metadata': {
                    'min_value': sensor_stream.min_value,
                    'max_value': sensor_stream.max_value,
                    'nominal_value': sensor_stream.nominal_value,
                    'critical_threshold': sensor_stream.critical_threshold,
                    'is_active': sensor_stream.is_active,
                    'last_update': sensor_stream.last_update.isoformat() if sensor_stream.last_update else None,
                    'data_points': sensor_stream.data_points_added,
                    'anomalies_detected': sensor_stream.anomalies_detected
                }
            }

        return result

    def get_sensor_data_from_database(self, sensor_ids: Optional[List[str]] = None,
                                    time_window: str = "5min") -> Dict[str, Dict[str, Any]]:
        """
        Get sensor data from database using unified data access layer

        Args:
            sensor_ids: List of sensor IDs to retrieve (None for all)
            time_window: Time window for data ("1min", "5min", "15min", "1hour")

        Returns:
            Dictionary of sensor data in consistent format
        """
        try:
            # Map time window to TimeRange enum
            time_range_map = {
                "1min": TimeRange.LAST_HOUR,
                "5min": TimeRange.LAST_HOUR,
                "15min": TimeRange.LAST_6_HOURS,
                "1hour": TimeRange.LAST_DAY,
                "1day": TimeRange.LAST_DAY
            }

            time_range = time_range_map.get(time_window, TimeRange.LAST_HOUR)

            # Create query for telemetry data
            query = DataQuery(
                time_range=time_range,
                limit=1000,  # Limit for performance
                order_by="timestamp",
                order_desc=True
            )

            # If specific sensor IDs are requested, convert to equipment IDs
            if sensor_ids:
                equipment_ids = []
                for sensor_id in sensor_ids:
                    # Try to get equipment info
                    equipment_info = equipment_mapper.get_equipment_info(sensor_id)
                    if equipment_info:
                        equipment_ids.append(sensor_id)
                    else:
                        # Try alternative formats
                        if '_' not in sensor_id:
                            for spacecraft in ['SMAP', 'MSL']:
                                alt_id = f"{spacecraft}_{sensor_id}"
                                if equipment_mapper.get_equipment_info(alt_id):
                                    equipment_ids.append(alt_id)
                                    break
                        else:
                            equipment_ids.append(sensor_id)

                query.equipment_ids = equipment_ids if equipment_ids else None

            # Get telemetry data from unified access layer
            telemetry_records = unified_data_access.get_telemetry_data(query)

            # Convert to sensor stream format
            result = {}

            # Group records by equipment/channel
            grouped_data = defaultdict(list)
            for record in telemetry_records:
                equipment_key = f"{record.spacecraft}_{record.channel}"
                grouped_data[equipment_key].append(record)

            # Process each equipment's data
            for equipment_id, records in grouped_data.items():
                if not records:
                    continue

                # Sort by timestamp
                records.sort(key=lambda x: x.timestamp, reverse=True)

                # Get equipment info for metadata
                equipment_info = records[0].equipment_info

                # Extract time series data from records
                timestamps = []
                values = []
                anomaly_scores = []
                anomaly_flags = []

                for record in records:
                    # Convert record values to time series
                    if record.values:
                        # Each record contains an array of values with timestamps
                        base_timestamp = record.timestamp
                        for i, value in enumerate(record.values):
                            timestamps.append((base_timestamp + timedelta(seconds=i)).isoformat())
                            values.append(float(value))
                            anomaly_scores.append(record.anomaly_score)
                            anomaly_flags.append(record.is_anomaly)
                    else:
                        # Single value record
                        timestamps.append(record.timestamp.isoformat())
                        values.append(record.mean_value)
                        anomaly_scores.append(record.anomaly_score)
                        anomaly_flags.append(record.is_anomaly)

                # Filter by time window if more precision needed
                time_delta = self._parse_time_window(time_window)
                cutoff_time = datetime.now() - time_delta

                filtered_timestamps = []
                filtered_values = []
                filtered_anomaly_scores = []
                filtered_anomaly_flags = []

                for i, timestamp_str in enumerate(timestamps):
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp >= cutoff_time:
                        filtered_timestamps.append(timestamp_str)
                        filtered_values.append(values[i])
                        filtered_anomaly_scores.append(anomaly_scores[i])
                        filtered_anomaly_flags.append(anomaly_flags[i])

                # Build result for this equipment
                result[equipment_id] = {
                    'sensor_name': equipment_info.get('name', equipment_id),
                    'equipment_id': equipment_id,
                    'subsystem': equipment_info.get('subsystem', 'Unknown'),
                    'unit': equipment_info.get('unit', 'value'),
                    'data': {
                        'timestamps': filtered_timestamps,
                        'values': filtered_values,
                        'anomaly_scores': filtered_anomaly_scores,
                        'anomaly_flags': filtered_anomaly_flags
                    },
                    'metadata': {
                        'min_value': min(filtered_values) if filtered_values else 0.0,
                        'max_value': max(filtered_values) if filtered_values else 1.0,
                        'nominal_value': equipment_info.get('nominal_value', 0.5),
                        'critical_threshold': equipment_info.get('critical_threshold', 0.8),
                        'is_active': bool(filtered_values),
                        'last_update': filtered_timestamps[0] if filtered_timestamps else None,
                        'record_count': len(filtered_values)
                    }
                }

            logger.info(f"Retrieved sensor data from database: {len(result)} sensors, {sum(len(r['data']['values']) for r in result.values())} data points")
            return result

        except Exception as e:
            logger.error(f"Failed to get sensor data from database: {e}")
            # Fallback to in-memory approach
            return self.get_sensor_data(sensor_ids, time_window, use_database=False)

    def _filter_sensor_data_by_time(self, sensor_stream: SensorStream,
                                   cutoff_time: datetime) -> Dict[str, List]:
        """Filter sensor stream data by time window"""
        filtered_timestamps = []
        filtered_values = []
        filtered_anomaly_scores = []
        filtered_anomaly_flags = []

        for i, timestamp in enumerate(sensor_stream.timestamps):
            if timestamp >= cutoff_time:
                filtered_timestamps.append(timestamp.isoformat())
                if i < len(sensor_stream.values):
                    filtered_values.append(sensor_stream.values[i])
                if i < len(sensor_stream.anomaly_scores):
                    filtered_anomaly_scores.append(sensor_stream.anomaly_scores[i])
                if i < len(sensor_stream.anomaly_flags):
                    filtered_anomaly_flags.append(sensor_stream.anomaly_flags[i])

        return {
            'timestamps': filtered_timestamps,
            'values': filtered_values,
            'anomaly_scores': filtered_anomaly_scores,
            'anomaly_flags': filtered_anomaly_flags
        }

    def _parse_time_window(self, time_window: str) -> timedelta:
        """Parse time window string to timedelta"""
        if time_window == "1min":
            return timedelta(minutes=1)
        elif time_window == "5min":
            return timedelta(minutes=5)
        elif time_window == "15min":
            return timedelta(minutes=15)
        elif time_window == "1hour":
            return timedelta(hours=1)
        else:
            return timedelta(minutes=5)  # Default

    def get_equipment_sensors(self, equipment_id: str) -> List[Dict[str, Any]]:
        """Get all sensors for a specific equipment"""
        sensor_list = []

        if equipment_id in self.equipment_sensors:
            for sensor_id in self.equipment_sensors[equipment_id]:
                sensor_stream = self.sensor_streams[sensor_id]
                sensor_list.append({
                    'sensor_id': sensor_id,
                    'sensor_name': sensor_stream.sensor_name,
                    'unit': sensor_stream.unit,
                    'subsystem': sensor_stream.subsystem,
                    'is_active': sensor_stream.is_active,
                    'last_value': sensor_stream.get_latest_value(),
                    'last_update': sensor_stream.last_update.isoformat() if sensor_stream.last_update else None,
                    'data_count': sensor_stream.get_data_count(),
                    'anomalies_detected': sensor_stream.anomalies_detected
                })

        return sensor_list

    def get_subsystem_sensors(self, subsystem: str) -> List[Dict[str, Any]]:
        """Get all sensors for a specific subsystem"""
        sensor_list = []

        if subsystem in self.subsystem_sensors:
            for sensor_id in self.subsystem_sensors[subsystem]:
                sensor_stream = self.sensor_streams[sensor_id]
                sensor_list.append({
                    'sensor_id': sensor_id,
                    'sensor_name': sensor_stream.sensor_name,
                    'equipment_id': sensor_stream.equipment_id,
                    'unit': sensor_stream.unit,
                    'is_active': sensor_stream.is_active,
                    'last_value': sensor_stream.get_latest_value(),
                    'last_update': sensor_stream.last_update.isoformat() if sensor_stream.last_update else None,
                    'data_count': sensor_stream.get_data_count(),
                    'anomalies_detected': sensor_stream.anomalies_detected
                })

        return sensor_list

    def get_all_sensors_summary(self) -> Dict[str, Any]:
        """Get summary of all sensors"""
        total_sensors = len(self.sensor_streams)
        active_sensors = sum(1 for s in self.sensor_streams.values() if s.is_active)

        # Count by subsystem
        subsystem_counts = defaultdict(int)
        for sensor_stream in self.sensor_streams.values():
            subsystem_counts[sensor_stream.subsystem] += 1

        # Count by equipment
        equipment_counts = defaultdict(int)
        for sensor_stream in self.sensor_streams.values():
            equipment_counts[sensor_stream.equipment_id] += 1

        # Update memory stats
        memory_stats = memory_manager.get_memory_stats()
        self.stream_stats['memory_usage_mb'] = memory_stats.total_memory_mb
        self.stream_stats['avg_window_size'] = memory_stats.avg_window_size

        return {
            'total_sensors': total_sensors,
            'active_sensors': active_sensors,
            'smap_sensors': len([s for s in self.sensor_streams if 'SMAP' in s]),
            'msl_sensors': len([s for s in self.sensor_streams if 'MSL' in s]),
            'subsystem_counts': dict(subsystem_counts),
            'equipment_counts': dict(equipment_counts),
            'stream_stats': self.stream_stats.copy(),
            'memory_stats': {
                'total_memory_mb': memory_stats.total_memory_mb,
                'avg_window_size': memory_stats.avg_window_size,
                'compressed_windows': memory_stats.compressed_windows,
                'last_cleanup': memory_stats.last_cleanup.isoformat(),
                'gc_cycles': memory_stats.gc_cycles
            },
            'is_streaming': self.is_streaming
        }

    def get_sensor_options_for_dropdown(self, equipment_id: Optional[str] = None,
                                       subsystem: Optional[str] = None) -> List[Dict[str, str]]:
        """Get sensor options formatted for dropdown selection"""
        options = []

        # Filter sensors based on criteria
        filtered_sensors = []
        for sensor_id, sensor_stream in self.sensor_streams.items():
            if equipment_id and sensor_stream.equipment_id != equipment_id:
                continue
            if subsystem and sensor_stream.subsystem != subsystem:
                continue
            filtered_sensors.append((sensor_id, sensor_stream))

        # Sort by equipment and sensor name
        filtered_sensors.sort(key=lambda x: (x[1].equipment_id, x[1].sensor_name))

        # Create dropdown options
        for sensor_id, sensor_stream in filtered_sensors:
            label = f"{sensor_stream.sensor_name} ({sensor_stream.equipment_id})"
            options.append({
                'label': label,
                'value': sensor_id
            })

        return options


# Create global sensor stream manager instance
sensor_stream_manager = SensorStreamManager()

# Auto-start streaming when module is imported
if sensor_stream_manager and not sensor_stream_manager.is_streaming:
    try:
        sensor_stream_manager.start_streaming()
        logger.info("Auto-started sensor streaming")
    except Exception as e:
        logger.error(f"Failed to auto-start sensor streaming: {e}")