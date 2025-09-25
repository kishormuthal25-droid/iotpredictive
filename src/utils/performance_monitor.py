"""
Enhanced Performance Monitoring System
Provides comprehensive performance monitoring for the IoT Predictive Maintenance System
"""

import time
import psutil
import threading
import tracemalloc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import json
from pathlib import Path
import weakref
import logging
import os

from .logger import get_logger, MetricsLogger

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

@dataclass
class SystemSnapshot:
    """System resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    process_count: int = 0
    thread_count: int = 0

class PerformanceTracker:
    """Track performance of specific operations"""

    def __init__(self, name: str):
        """Initialize performance tracker

        Args:
            name: Name of the operation being tracked
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.logger = get_logger(f"performance.{name}")

    def start(self):
        """Start performance tracking"""
        self.start_time = time.perf_counter()
        self.logger.debug(f"Started tracking: {self.name}")

        # Start memory tracking if available
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        return self

    def stop(self) -> float:
        """Stop performance tracking and return elapsed time

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("Performance tracking not started")

        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time

        # Get memory usage if tracking
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            self.metrics['memory_current_mb'] = current / 1024 / 1024
            self.metrics['memory_peak_mb'] = peak / 1024 / 1024

        self.metrics['elapsed_seconds'] = elapsed
        self.logger.info(f"Completed tracking: {self.name} - {elapsed:.3f}s")

        return elapsed

    def add_metric(self, name: str, value: float, unit: str = ""):
        """Add custom metric

        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit
        """
        self.metrics[name] = value
        self.logger.debug(f"Added metric {name}: {value} {unit}")

    def get_metrics(self) -> Dict[str, float]:
        """Get all collected metrics

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

class SystemMonitor:
    """Monitor system-wide performance metrics"""

    def __init__(self, sample_interval: float = 60.0, history_size: int = 1000):
        """Initialize system monitor

        Args:
            sample_interval: Sampling interval in seconds
            history_size: Maximum number of samples to keep
        """
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.snapshots = deque(maxlen=history_size)
        self.running = False
        self.thread = None
        self.logger = get_logger('performance.system')

        try:
            self.metrics_logger = MetricsLogger('system_metrics')
        except:
            # Fallback if MetricsLogger not available
            self.metrics_logger = None

        # Initialize network counters
        try:
            self._last_network_stats = psutil.net_io_counters()
        except:
            self._last_network_stats = None

    def start(self):
        """Start system monitoring"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        self.logger.info("System monitoring started")

    def stop(self):
        """Stop system monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self._log_metrics(snapshot)

            except Exception as e:
                self.logger.error(f"Error taking system snapshot: {e}")

            time.sleep(self.sample_interval)

    def _take_snapshot(self) -> SystemSnapshot:
        """Take a system resource snapshot

        Returns:
            SystemSnapshot object
        """
        # Get basic system stats
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get network stats with delta calculation
        net_sent_delta = 0
        net_recv_delta = 0

        if self._last_network_stats:
            try:
                current_network = psutil.net_io_counters()
                net_sent_delta = current_network.bytes_sent - self._last_network_stats.bytes_sent
                net_recv_delta = current_network.bytes_recv - self._last_network_stats.bytes_recv
                self._last_network_stats = current_network
            except:
                pass

        # Create snapshot
        snapshot = SystemSnapshot(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=disk.percent,
            network_bytes_sent=net_sent_delta,
            network_bytes_recv=net_recv_delta,
            process_count=len(psutil.pids()) if hasattr(psutil, 'pids') else 0,
            thread_count=threading.active_count()
        )

        return snapshot

    def _log_metrics(self, snapshot: SystemSnapshot):
        """Log metrics from snapshot

        Args:
            snapshot: System snapshot to log
        """
        if not self.metrics_logger:
            return

        try:
            self.metrics_logger.log_gauge('cpu_percent', snapshot.cpu_percent)
            self.metrics_logger.log_gauge('memory_percent', snapshot.memory_percent)
            self.metrics_logger.log_gauge('memory_used_mb', snapshot.memory_used_mb)
            self.metrics_logger.log_gauge('disk_usage_percent', snapshot.disk_usage_percent)
            self.metrics_logger.log_counter('network_bytes_sent', snapshot.network_bytes_sent)
            self.metrics_logger.log_counter('network_bytes_recv', snapshot.network_bytes_recv)
            self.metrics_logger.log_gauge('process_count', snapshot.process_count)
            self.metrics_logger.log_gauge('thread_count', snapshot.thread_count)
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")

    def get_current_stats(self) -> Optional[SystemSnapshot]:
        """Get the most recent system snapshot

        Returns:
            Most recent SystemSnapshot or None
        """
        return self.snapshots[-1] if self.snapshots else None

    def get_average_stats(self, minutes: int = 5) -> Optional[Dict[str, float]]:
        """Get average statistics over specified time period

        Args:
            minutes: Number of minutes to average over

        Returns:
            Dictionary of averaged statistics
        """
        if not self.snapshots:
            return None

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return None

        return {
            'cpu_percent': statistics.mean(s.cpu_percent for s in recent_snapshots),
            'memory_percent': statistics.mean(s.memory_percent for s in recent_snapshots),
            'memory_used_mb': statistics.mean(s.memory_used_mb for s in recent_snapshots),
            'disk_usage_percent': statistics.mean(s.disk_usage_percent for s in recent_snapshots),
            'process_count': statistics.mean(s.process_count for s in recent_snapshots),
            'thread_count': statistics.mean(s.thread_count for s in recent_snapshots)
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status

        Returns:
            Dictionary with system health information
        """
        current_stats = self.get_current_stats()
        avg_stats = self.get_average_stats(minutes=5)

        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy'
        }

        if current_stats:
            health['current'] = {
                'cpu_percent': current_stats.cpu_percent,
                'memory_percent': current_stats.memory_percent,
                'memory_used_mb': current_stats.memory_used_mb,
                'disk_usage_percent': current_stats.disk_usage_percent,
                'thread_count': current_stats.thread_count
            }

            # Determine health status
            if (current_stats.cpu_percent > 90 or
                current_stats.memory_percent > 90 or
                current_stats.disk_usage_percent > 95):
                health['status'] = 'critical'
            elif (current_stats.cpu_percent > 70 or
                  current_stats.memory_percent > 80 or
                  current_stats.disk_usage_percent > 85):
                health['status'] = 'warning'

        if avg_stats:
            health['averages_5min'] = avg_stats

        return health

# Global system monitor instance
_system_monitor = SystemMonitor()

# Convenience functions
def start_monitoring():
    """Start system monitoring"""
    _system_monitor.start()

def stop_monitoring():
    """Stop system monitoring"""
    _system_monitor.stop()

def get_system_health() -> Dict[str, Any]:
    """Get system health information

    Returns:
        Dictionary with system health data
    """
    return _system_monitor.get_system_health()

def create_tracker(name: str) -> PerformanceTracker:
    """Create a performance tracker

    Args:
        name: Name of the operation to track

    Returns:
        PerformanceTracker instance
    """
    return PerformanceTracker(name)

def track_operation(name: str):
    """Decorator for tracking operation performance

    Args:
        name: Name of the operation

    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with create_tracker(f"{name}.{func.__name__}") as tracker:
                try:
                    result = func(*args, **kwargs)
                    tracker.add_metric('success', 1)
                    return result
                except Exception as e:
                    tracker.add_metric('error', 1)
                    raise
            return result
        return wrapper
    return decorator

# Auto-start monitoring on import
_system_monitor.start()
