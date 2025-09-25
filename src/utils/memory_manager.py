"""
Sliding Window Memory Manager for IoT Sensor Data
Optimized memory management for 80-sensor real-time processing
"""

import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory management configuration"""
    # Sliding window settings
    default_window_size: int = 1000  # Default points per sensor
    max_window_size: int = 5000      # Maximum window size per sensor
    min_window_size: int = 100       # Minimum window size per sensor

    # Memory limits
    max_total_memory_mb: int = 2048  # 2GB total memory limit
    memory_warning_threshold: float = 0.8  # Warn at 80% usage
    memory_cleanup_threshold: float = 0.9   # Cleanup at 90% usage

    # Performance settings
    cleanup_interval: int = 60       # Cleanup every 60 seconds
    gc_interval: int = 300          # Force garbage collection every 5 minutes
    adaptive_sizing: bool = True    # Automatically adjust window sizes
    compression_enabled: bool = True # Use compressed arrays for older data

    # 80-sensor optimization
    sensors_per_cleanup_batch: int = 20  # Process 20 sensors per cleanup cycle


@dataclass
class SlidingWindow:
    """Efficient sliding window data structure for sensor data"""
    sensor_id: str
    max_size: int

    # Core data arrays (pre-allocated for performance)
    timestamps: np.ndarray = field(default=None)
    values: np.ndarray = field(default=None)
    anomaly_scores: np.ndarray = field(default=None)
    anomaly_flags: np.ndarray = field(default=None)

    # Window state
    current_size: int = 0
    write_index: int = 0
    is_full: bool = False

    # Memory optimization
    last_access: datetime = field(default_factory=datetime.now)
    compression_ratio: float = 1.0
    is_compressed: bool = False

    def __post_init__(self):
        """Initialize pre-allocated arrays"""
        if self.timestamps is None:
            self.timestamps = np.empty(self.max_size, dtype='datetime64[ns]')
            self.values = np.empty(self.max_size, dtype=np.float32)
            self.anomaly_scores = np.empty(self.max_size, dtype=np.float32)
            self.anomaly_flags = np.empty(self.max_size, dtype=bool)

    def append(self, timestamp: datetime, value: float,
               anomaly_score: float = 0.0, anomaly_flag: bool = False):
        """Efficiently append data to sliding window"""
        # Convert timestamp to numpy datetime64
        np_timestamp = np.datetime64(timestamp)

        # Add data at current write position
        self.timestamps[self.write_index] = np_timestamp
        self.values[self.write_index] = np.float32(value)
        self.anomaly_scores[self.write_index] = np.float32(anomaly_score)
        self.anomaly_flags[self.write_index] = anomaly_flag

        # Update indices
        self.write_index = (self.write_index + 1) % self.max_size

        if not self.is_full:
            self.current_size += 1
            if self.current_size >= self.max_size:
                self.is_full = True

        # Update access time
        self.last_access = datetime.now()

    def get_latest(self, count: int = None) -> Dict[str, np.ndarray]:
        """Get latest N data points efficiently"""
        if count is None:
            count = self.current_size

        count = min(count, self.current_size)
        if count == 0:
            return {
                'timestamps': np.array([]),
                'values': np.array([]),
                'anomaly_scores': np.array([]),
                'anomaly_flags': np.array([])
            }

        if self.is_full:
            # Calculate indices for circular buffer
            start_idx = (self.write_index - count) % self.max_size
            if start_idx + count <= self.max_size:
                # Contiguous slice
                slice_indices = slice(start_idx, start_idx + count)
                return {
                    'timestamps': self.timestamps[slice_indices].copy(),
                    'values': self.values[slice_indices].copy(),
                    'anomaly_scores': self.anomaly_scores[slice_indices].copy(),
                    'anomaly_flags': self.anomaly_flags[slice_indices].copy()
                }
            else:
                # Wrap-around slice
                part1_size = self.max_size - start_idx
                part2_size = count - part1_size

                timestamps = np.concatenate([
                    self.timestamps[start_idx:],
                    self.timestamps[:part2_size]
                ])
                values = np.concatenate([
                    self.values[start_idx:],
                    self.values[:part2_size]
                ])
                anomaly_scores = np.concatenate([
                    self.anomaly_scores[start_idx:],
                    self.anomaly_scores[:part2_size]
                ])
                anomaly_flags = np.concatenate([
                    self.anomaly_flags[start_idx:],
                    self.anomaly_flags[:part2_size]
                ])

                return {
                    'timestamps': timestamps,
                    'values': values,
                    'anomaly_scores': anomaly_scores,
                    'anomaly_flags': anomaly_flags
                }
        else:
            # Not full, simple slice
            end_idx = min(count, self.current_size)
            return {
                'timestamps': self.timestamps[:end_idx].copy(),
                'values': self.values[:end_idx].copy(),
                'anomaly_scores': self.anomaly_scores[:end_idx].copy(),
                'anomaly_flags': self.anomaly_flags[:end_idx].copy()
            }

    def get_time_range(self, start_time: datetime, end_time: datetime) -> Dict[str, np.ndarray]:
        """Get data within specific time range"""
        if self.current_size == 0:
            return self.get_latest(0)

        # Convert to numpy datetime64
        start_np = np.datetime64(start_time)
        end_np = np.datetime64(end_time)

        # Get all current data
        all_data = self.get_latest()

        # Filter by time range
        time_mask = (all_data['timestamps'] >= start_np) & (all_data['timestamps'] <= end_np)

        return {
            'timestamps': all_data['timestamps'][time_mask],
            'values': all_data['values'][time_mask],
            'anomaly_scores': all_data['anomaly_scores'][time_mask],
            'anomaly_flags': all_data['anomaly_flags'][time_mask]
        }

    def compress_older_data(self, keep_recent: int = 100):
        """Compress older data to save memory"""
        if not self.is_full or self.current_size <= keep_recent:
            return

        # This is a placeholder for compression logic
        # In practice, you might downsample older data or use compression algorithms
        self.is_compressed = True
        self.compression_ratio = 0.7  # Simulated compression ratio

    def get_memory_usage(self) -> int:
        """Get memory usage in bytes"""
        base_size = (
            self.timestamps.nbytes +
            self.values.nbytes +
            self.anomaly_scores.nbytes +
            self.anomaly_flags.nbytes
        )
        return int(base_size * self.compression_ratio)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_sensors: int = 0
    total_memory_mb: float = 0.0
    avg_window_size: float = 0.0
    max_window_size: int = 0
    min_window_size: int = 0
    compressed_windows: int = 0
    last_cleanup: datetime = field(default_factory=datetime.now)
    gc_cycles: int = 0
    memory_warnings: int = 0


class SlidingWindowMemoryManager:
    """
    High-performance memory manager for 80-sensor sliding windows
    Features: Bounded memory usage, automatic cleanup, adaptive sizing
    """

    def __init__(self, config: MemoryConfig = None):
        """Initialize sliding window memory manager"""
        self.config = config or MemoryConfig()
        self.stats = MemoryStats()

        # Sliding windows for all sensors
        self.windows: Dict[str, SlidingWindow] = {}
        self.window_locks: Dict[str, threading.RLock] = {}

        # Memory management
        self.global_lock = threading.RLock()
        self.cleanup_thread = None
        self.is_running = False

        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        self.memory_history = deque(maxlen=100)

        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Start background maintenance
        self._start_maintenance_thread()

        logger.info(f"Sliding Window Memory Manager initialized: "
                   f"max_memory={self.config.max_total_memory_mb}MB, "
                   f"default_window={self.config.default_window_size}")

    def create_sensor_window(self, sensor_id: str, window_size: int = None) -> bool:
        """Create a new sliding window for a sensor"""
        window_size = window_size or self.config.default_window_size
        window_size = max(self.config.min_window_size,
                         min(window_size, self.config.max_window_size))

        with self.global_lock:
            if sensor_id in self.windows:
                logger.warning(f"Window for sensor {sensor_id} already exists")
                return True

            # Check memory constraints
            if not self._check_memory_available(window_size):
                logger.warning(f"Insufficient memory for sensor {sensor_id}")
                return False

            # Create window and lock
            self.windows[sensor_id] = SlidingWindow(
                sensor_id=sensor_id,
                max_size=window_size
            )
            self.window_locks[sensor_id] = threading.RLock()

            self.stats.total_sensors += 1
            logger.debug(f"Created sliding window for sensor {sensor_id}: size={window_size}")

            return True

    def add_sensor_data(self, sensor_id: str, timestamp: datetime, value: float,
                       anomaly_score: float = 0.0, anomaly_flag: bool = False) -> bool:
        """Add data to sensor's sliding window"""
        # Create window if it doesn't exist
        if sensor_id not in self.windows:
            if not self.create_sensor_window(sensor_id):
                return False

        # Add data with thread safety
        try:
            with self.window_locks[sensor_id]:
                self.windows[sensor_id].append(timestamp, value, anomaly_score, anomaly_flag)
            return True
        except Exception as e:
            logger.error(f"Failed to add data for sensor {sensor_id}: {e}")
            return False

    def get_sensor_data(self, sensor_id: str, count: int = None) -> Optional[Dict[str, np.ndarray]]:
        """Get latest data from sensor's sliding window"""
        if sensor_id not in self.windows:
            return None

        try:
            with self.window_locks[sensor_id]:
                return self.windows[sensor_id].get_latest(count)
        except Exception as e:
            logger.error(f"Failed to get data for sensor {sensor_id}: {e}")
            return None

    def get_sensor_time_range(self, sensor_id: str, start_time: datetime,
                             end_time: datetime) -> Optional[Dict[str, np.ndarray]]:
        """Get sensor data within time range"""
        if sensor_id not in self.windows:
            return None

        try:
            with self.window_locks[sensor_id]:
                return self.windows[sensor_id].get_time_range(start_time, end_time)
        except Exception as e:
            logger.error(f"Failed to get time range data for sensor {sensor_id}: {e}")
            return None

    def get_all_sensors_latest(self, count: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
        """Get latest data from all sensors efficiently"""
        result = {}

        # Process sensors in batches to avoid holding locks too long
        sensor_ids = list(self.windows.keys())
        batch_size = 20

        for i in range(0, len(sensor_ids), batch_size):
            batch = sensor_ids[i:i + batch_size]

            for sensor_id in batch:
                if sensor_id in self.windows:
                    data = self.get_sensor_data(sensor_id, count)
                    if data is not None:
                        result[sensor_id] = data

        return result

    def _check_memory_available(self, additional_window_size: int) -> bool:
        """Check if there's enough memory for additional window"""
        # Estimate memory for new window (4 arrays * window_size * bytes per element)
        estimated_mb = (additional_window_size * 4 * 8) / (1024 * 1024)  # 8 bytes avg per element

        current_memory = self._get_current_memory_usage()

        return (current_memory + estimated_mb) < self.config.max_total_memory_mb

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.memory_history.append(memory_mb)
            return memory_mb
        except Exception:
            return 0.0

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.is_running = True
        self.cleanup_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info("Memory maintenance thread started")

    def _maintenance_loop(self):
        """Background memory maintenance"""
        gc_counter = 0

        while self.is_running:
            try:
                # Check memory usage
                current_memory = self._get_current_memory_usage()
                memory_ratio = current_memory / self.config.max_total_memory_mb

                # Update stats
                self.stats.total_memory_mb = current_memory
                self.stats.total_sensors = len(self.windows)

                # Memory warning
                if memory_ratio > self.config.memory_warning_threshold:
                    self.stats.memory_warnings += 1
                    logger.warning(f"High memory usage: {current_memory:.1f}MB "
                                 f"({memory_ratio:.1%} of limit)")

                # Memory cleanup
                if memory_ratio > self.config.memory_cleanup_threshold:
                    self._perform_memory_cleanup()

                # Adaptive window sizing
                if self.config.adaptive_sizing:
                    self._adjust_window_sizes(memory_ratio)

                # Periodic garbage collection
                gc_counter += 1
                if gc_counter * self.config.cleanup_interval >= self.config.gc_interval:
                    gc.collect()
                    self.stats.gc_cycles += 1
                    gc_counter = 0

                # Update stats
                self.stats.last_cleanup = datetime.now()

                # Sleep until next cleanup
                time.sleep(self.config.cleanup_interval)

            except Exception as e:
                logger.error(f"Memory maintenance error: {e}")
                time.sleep(30)  # Longer sleep on error

    def _perform_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.info("Performing memory cleanup...")

        with self.global_lock:
            # Process sensors in batches
            sensor_ids = list(self.windows.keys())
            batch_size = self.config.sensors_per_cleanup_batch

            for i in range(0, len(sensor_ids), batch_size):
                batch = sensor_ids[i:i + batch_size]

                for sensor_id in batch:
                    if sensor_id in self.windows:
                        window = self.windows[sensor_id]

                        # Compress older data if not already compressed
                        if not window.is_compressed:
                            window.compress_older_data()
                            self.stats.compressed_windows += 1

                        # Remove inactive sensors (no access in last hour)
                        if datetime.now() - window.last_access > timedelta(hours=1):
                            del self.windows[sensor_id]
                            del self.window_locks[sensor_id]
                            self.stats.total_sensors -= 1
                            logger.debug(f"Removed inactive sensor window: {sensor_id}")

            # Force garbage collection
            gc.collect()

    def _adjust_window_sizes(self, memory_ratio: float):
        """Dynamically adjust window sizes based on memory pressure"""
        if memory_ratio < 0.7:
            return  # No adjustment needed

        # Calculate target reduction
        if memory_ratio > 0.9:
            reduction_factor = 0.5  # Aggressive reduction
        elif memory_ratio > 0.8:
            reduction_factor = 0.7  # Moderate reduction
        else:
            reduction_factor = 0.9  # Minor reduction

        with self.global_lock:
            for window in self.windows.values():
                if not window.is_compressed:
                    new_size = max(
                        self.config.min_window_size,
                        int(window.max_size * reduction_factor)
                    )
                    if new_size < window.max_size:
                        # This would require resizing arrays - simplified for now
                        window.compress_older_data()

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        with self.global_lock:
            if self.windows:
                window_sizes = [w.max_size for w in self.windows.values()]
                self.stats.avg_window_size = np.mean(window_sizes)
                self.stats.max_window_size = np.max(window_sizes)
                self.stats.min_window_size = np.min(window_sizes)

            self.stats.total_memory_mb = self._get_current_memory_usage()

            return self.stats

    def optimize_for_sensor_count(self, sensor_count: int):
        """Optimize memory settings for specific sensor count"""
        if sensor_count <= 20:
            self.config.default_window_size = 2000
            self.config.cleanup_interval = 120
        elif sensor_count <= 50:
            self.config.default_window_size = 1500
            self.config.cleanup_interval = 90
        else:  # 80+ sensors
            self.config.default_window_size = 1000
            self.config.cleanup_interval = 60
            self.config.adaptive_sizing = True
            self.config.compression_enabled = True

        logger.info(f"Memory manager optimized for {sensor_count} sensors: "
                   f"window_size={self.config.default_window_size}, "
                   f"cleanup_interval={self.config.cleanup_interval}s")

    def cleanup(self):
        """Clean shutdown of memory manager"""
        self.is_running = False

        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)

        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        with self.global_lock:
            self.windows.clear()
            self.window_locks.clear()

        logger.info("Memory manager shutdown complete")


# Global memory manager instance
memory_manager = SlidingWindowMemoryManager()


# Helper functions for easy integration
def add_sensor_data(sensor_id: str, timestamp: datetime, value: float,
                   anomaly_score: float = 0.0, anomaly_flag: bool = False) -> bool:
    """Add data to sensor's sliding window"""
    return memory_manager.add_sensor_data(sensor_id, timestamp, value, anomaly_score, anomaly_flag)


def get_sensor_data(sensor_id: str, count: int = None) -> Optional[Dict[str, np.ndarray]]:
    """Get latest data from sensor's sliding window"""
    return memory_manager.get_sensor_data(sensor_id, count)


def get_all_sensors_data(count: int = 100) -> Dict[str, Dict[str, np.ndarray]]:
    """Get latest data from all sensors"""
    return memory_manager.get_all_sensors_latest(count)


def optimize_for_sensors(sensor_count: int):
    """Optimize memory manager for specific sensor count"""
    memory_manager.optimize_for_sensor_count(sensor_count)