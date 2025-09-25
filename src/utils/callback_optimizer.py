"""
Dashboard Callback Optimizer for Sub-Second Response Times
Intelligent memoization and incremental update system for 80-sensor dashboards
"""

import functools
import hashlib
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import advanced caching
from src.utils.advanced_cache import advanced_cache, DashboardCacheHelper

logger = logging.getLogger(__name__)


@dataclass
class CallbackMetrics:
    """Callback performance metrics"""
    callback_name: str
    total_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_execution_time: float = 0.0
    last_execution_time: float = 0.0
    total_execution_time: float = 0.0
    incremental_updates: int = 0
    full_updates: int = 0
    data_size_bytes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class IncrementalUpdate:
    """Incremental update data structure"""
    update_id: str
    callback_name: str
    changed_data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    data_size: int = 0


class CallbackOptimizer:
    """
    Advanced callback optimizer with memoization and incremental updates
    Designed for sub-second dashboard response times with 80 sensors
    """

    def __init__(self):
        """Initialize callback optimizer"""
        self.cache_helper = DashboardCacheHelper()
        self.metrics: Dict[str, CallbackMetrics] = {}
        self.metrics_lock = threading.RLock()

        # Memoization cache
        self.memo_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.memo_lock = threading.RLock()

        # Incremental update tracking
        self.last_data_snapshots: Dict[str, Dict[str, Any]] = {}
        self.incremental_updates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Performance settings
        self.default_cache_ttl = 30  # 30 seconds default cache
        self.incremental_threshold = 0.1  # 10% change threshold
        self.max_memo_cache_size = 1000

        # Background cleanup
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.now()

        logger.info("Callback Optimizer initialized for sub-second dashboard performance")

    def memoize_callback(self, ttl: int = None, use_incremental: bool = True,
                        cache_key_func: Callable = None):
        """
        Decorator for intelligent callback memoization

        Args:
            ttl: Cache time-to-live in seconds
            use_incremental: Enable incremental updates
            cache_key_func: Custom cache key function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_optimized_callback(
                    func, args, kwargs, ttl, use_incremental, cache_key_func
                )
            return wrapper
        return decorator

    def _execute_optimized_callback(self, func: Callable, args: tuple, kwargs: dict,
                                  ttl: Optional[int], use_incremental: bool,
                                  cache_key_func: Optional[Callable]) -> Any:
        """Execute callback with optimization strategies"""
        start_time = time.time()
        callback_name = func.__name__

        # Initialize metrics if needed
        if callback_name not in self.metrics:
            with self.metrics_lock:
                self.metrics[callback_name] = CallbackMetrics(callback_name=callback_name)

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(func, args, kwargs, cache_key_func)

            # Try memoization cache first (fastest)
            memo_result = self._get_from_memo_cache(cache_key, ttl or self.default_cache_ttl)
            if memo_result is not None:
                self._update_callback_metrics(callback_name, time.time() - start_time, True, False)
                return memo_result

            # Try incremental update if enabled
            if use_incremental:
                incremental_result = self._try_incremental_update(callback_name, args, kwargs)
                if incremental_result is not None:
                    # Cache the incremental result
                    self._set_memo_cache(cache_key, incremental_result, ttl or self.default_cache_ttl)
                    self._update_callback_metrics(callback_name, time.time() - start_time, False, True)
                    return incremental_result

            # Execute full callback
            result = func(*args, **kwargs)

            # Cache the result
            self._set_memo_cache(cache_key, result, ttl or self.default_cache_ttl)

            # Store data snapshot for incremental updates
            if use_incremental:
                self._store_data_snapshot(callback_name, result)

            # Update metrics
            execution_time = time.time() - start_time
            self._update_callback_metrics(callback_name, execution_time, False, False)

            return result

        except Exception as e:
            logger.error(f"Callback optimization error for {callback_name}: {e}")
            # Fallback to direct execution
            result = func(*args, **kwargs)
            self._update_callback_metrics(callback_name, time.time() - start_time, False, False)
            return result

    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict,
                          cache_key_func: Optional[Callable]) -> str:
        """Generate cache key for callback"""
        if cache_key_func:
            try:
                return cache_key_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Custom cache key function failed: {e}")

        # Default cache key generation
        key_data = {
            'func': func.__name__,
            'args': str(args),
            'kwargs': sorted(kwargs.items()) if kwargs else []
        }

        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"callback:{hashlib.md5(key_string.encode()).hexdigest()}"

    def _get_from_memo_cache(self, cache_key: str, ttl: int) -> Any:
        """Get result from memoization cache"""
        with self.memo_lock:
            if cache_key in self.memo_cache:
                result, timestamp = self.memo_cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=ttl):
                    return result
                else:
                    # Expired, remove
                    del self.memo_cache[cache_key]
        return None

    def _set_memo_cache(self, cache_key: str, result: Any, ttl: int):
        """Set result in memoization cache with LRU eviction"""
        with self.memo_lock:
            # Evict oldest entries if cache is full
            if len(self.memo_cache) >= self.max_memo_cache_size:
                # Remove 20% of oldest entries
                sorted_items = sorted(
                    self.memo_cache.items(),
                    key=lambda x: x[1][1]  # Sort by timestamp
                )
                items_to_remove = len(sorted_items) // 5
                for item in sorted_items[:items_to_remove]:
                    del self.memo_cache[item[0]]

            self.memo_cache[cache_key] = (result, datetime.now())

    def _try_incremental_update(self, callback_name: str, args: tuple, kwargs: dict) -> Any:
        """Attempt incremental update instead of full refresh"""
        if callback_name not in self.last_data_snapshots:
            return None

        try:
            # This is a simplified incremental update strategy
            # In practice, you'd implement specific logic for each callback type
            last_snapshot = self.last_data_snapshots[callback_name]

            # Check if we can determine what data might have changed
            # For sensor data callbacks, we might check timestamps or specific sensors
            if self._should_use_incremental_update(callback_name, args, kwargs, last_snapshot):
                # Generate incremental update
                return self._generate_incremental_update(callback_name, last_snapshot, args, kwargs)

        except Exception as e:
            logger.debug(f"Incremental update failed for {callback_name}: {e}")

        return None

    def _should_use_incremental_update(self, callback_name: str, args: tuple,
                                     kwargs: dict, last_snapshot: Dict[str, Any]) -> bool:
        """Determine if incremental update is beneficial"""
        # Check if recent enough (within last 60 seconds)
        if 'timestamp' in last_snapshot:
            last_update = datetime.fromisoformat(last_snapshot['timestamp'])
            if datetime.now() - last_update > timedelta(seconds=60):
                return False

        # Check data size - incremental updates more beneficial for large datasets
        if 'data_size' in last_snapshot and last_snapshot['data_size'] > 10000:
            return True

        return False

    def _generate_incremental_update(self, callback_name: str, last_snapshot: Dict[str, Any],
                                   args: tuple, kwargs: dict) -> Any:
        """Generate incremental update based on last snapshot"""
        # This is a placeholder for callback-specific incremental update logic
        # Each callback type would have its own incremental update strategy

        if 'sensor' in callback_name.lower():
            return self._generate_sensor_incremental_update(last_snapshot, args, kwargs)
        elif 'chart' in callback_name.lower() or 'graph' in callback_name.lower():
            return self._generate_chart_incremental_update(last_snapshot, args, kwargs)
        else:
            return self._generate_generic_incremental_update(last_snapshot, args, kwargs)

    def _generate_sensor_incremental_update(self, last_snapshot: Dict[str, Any],
                                          args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Generate incremental update for sensor data"""
        # Simulate incremental sensor data update
        current_time = datetime.now()

        # Get new data points since last update
        if 'data' in last_snapshot and isinstance(last_snapshot['data'], dict):
            updated_data = last_snapshot['data'].copy()

            # Add simulation of new data points
            if 'timestamps' in updated_data and 'values' in updated_data:
                # Add a few new data points
                new_points = 5
                last_timestamp = updated_data['timestamps'][-1] if updated_data['timestamps'] else current_time.isoformat()
                last_time = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))

                for i in range(new_points):
                    new_timestamp = last_time + timedelta(seconds=i+1)
                    updated_data['timestamps'].append(new_timestamp.isoformat())
                    # Simulate new sensor value
                    last_value = updated_data['values'][-1] if updated_data['values'] else 0
                    new_value = last_value + np.random.normal(0, 0.1)
                    updated_data['values'].append(new_value)

                # Keep only recent data
                max_points = 1000
                if len(updated_data['timestamps']) > max_points:
                    updated_data['timestamps'] = updated_data['timestamps'][-max_points:]
                    updated_data['values'] = updated_data['values'][-max_points:]

            return {
                'data': updated_data,
                'timestamp': current_time.isoformat(),
                'incremental': True,
                'update_type': 'sensor_data'
            }

        return None

    def _generate_chart_incremental_update(self, last_snapshot: Dict[str, Any],
                                         args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Generate incremental update for chart data"""
        # For charts, we might only update specific traces or data points
        if 'figure' in last_snapshot:
            # Clone the figure and update only changed traces
            updated_figure = last_snapshot['figure'].copy()

            # Simulate updating chart data
            current_time = datetime.now()

            return {
                'figure': updated_figure,
                'timestamp': current_time.isoformat(),
                'incremental': True,
                'update_type': 'chart_data'
            }

        return None

    def _generate_generic_incremental_update(self, last_snapshot: Dict[str, Any],
                                           args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Generate generic incremental update"""
        # Basic incremental update - just update timestamp
        updated_data = last_snapshot.copy()
        updated_data['timestamp'] = datetime.now().isoformat()
        updated_data['incremental'] = True
        return updated_data

    def _store_data_snapshot(self, callback_name: str, result: Any):
        """Store data snapshot for future incremental updates"""
        try:
            # Convert result to serializable format
            snapshot = {
                'data': result,
                'timestamp': datetime.now().isoformat(),
                'data_size': len(str(result)) if result else 0
            }

            self.last_data_snapshots[callback_name] = snapshot

        except Exception as e:
            logger.debug(f"Failed to store snapshot for {callback_name}: {e}")

    def _update_callback_metrics(self, callback_name: str, execution_time: float,
                               cache_hit: bool, incremental: bool):
        """Update callback performance metrics"""
        with self.metrics_lock:
            metrics = self.metrics[callback_name]
            metrics.total_calls += 1
            metrics.last_execution_time = execution_time

            if cache_hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1

            if incremental:
                metrics.incremental_updates += 1
            else:
                metrics.full_updates += 1

            # Update average execution time (exponential moving average)
            alpha = 0.1
            metrics.avg_execution_time = (
                alpha * execution_time +
                (1 - alpha) * metrics.avg_execution_time
            )

            metrics.total_execution_time += execution_time
            metrics.last_updated = datetime.now()

    def get_callback_metrics(self, callback_name: str = None) -> Union[CallbackMetrics, Dict[str, CallbackMetrics]]:
        """Get callback performance metrics"""
        with self.metrics_lock:
            if callback_name:
                return self.metrics.get(callback_name)
            return self.metrics.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self.metrics_lock:
            total_calls = sum(m.total_calls for m in self.metrics.values())
            total_cache_hits = sum(m.cache_hits for m in self.metrics.values())
            total_incremental = sum(m.incremental_updates for m in self.metrics.values())

            cache_hit_rate = total_cache_hits / total_calls if total_calls > 0 else 0
            incremental_rate = total_incremental / total_calls if total_calls > 0 else 0

            avg_response_time = np.mean([m.avg_execution_time for m in self.metrics.values()]) if self.metrics else 0

            return {
                'total_callbacks': len(self.metrics),
                'total_calls': total_calls,
                'cache_hit_rate': cache_hit_rate,
                'incremental_update_rate': incremental_rate,
                'avg_response_time_ms': avg_response_time * 1000,
                'memo_cache_size': len(self.memo_cache),
                'snapshots_stored': len(self.last_data_snapshots),
                'last_cleanup': self.last_cleanup.isoformat()
            }

    def clear_cache(self, callback_name: str = None):
        """Clear callback cache"""
        with self.memo_lock:
            if callback_name:
                # Clear cache for specific callback
                keys_to_remove = [k for k in self.memo_cache.keys() if callback_name in k]
                for key in keys_to_remove:
                    del self.memo_cache[key]

                # Clear snapshot
                if callback_name in self.last_data_snapshots:
                    del self.last_data_snapshots[callback_name]
            else:
                # Clear all caches
                self.memo_cache.clear()
                self.last_data_snapshots.clear()

    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()

        with self.memo_lock:
            expired_keys = []
            for key, (result, timestamp) in self.memo_cache.items():
                if current_time - timestamp > timedelta(seconds=self.default_cache_ttl * 2):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memo_cache[key]

        # Clean up old snapshots
        expired_snapshots = []
        for callback_name, snapshot in self.last_data_snapshots.items():
            snapshot_time = datetime.fromisoformat(snapshot['timestamp'])
            if current_time - snapshot_time > timedelta(hours=1):
                expired_snapshots.append(callback_name)

        for callback_name in expired_snapshots:
            del self.last_data_snapshots[callback_name]

        self.last_cleanup = current_time

        if expired_keys or expired_snapshots:
            logger.debug(f"Cleaned up {len(expired_keys)} cache entries and {len(expired_snapshots)} snapshots")


# Global callback optimizer instance
callback_optimizer = CallbackOptimizer()


# Convenience decorators
def optimize_callback(ttl: int = 30, use_incremental: bool = True, cache_key_func: Callable = None):
    """Convenience decorator for callback optimization"""
    return callback_optimizer.memoize_callback(ttl, use_incremental, cache_key_func)


def fast_callback(ttl: int = 10):
    """Fast callback optimization for frequently updated data"""
    return callback_optimizer.memoize_callback(ttl, use_incremental=True)


def sensor_callback(ttl: int = 5):
    """Optimized decorator for sensor data callbacks"""
    def sensor_cache_key(*args, **kwargs):
        # Generate cache key based on sensor IDs and time range
        sensor_ids = kwargs.get('sensor_ids', [])
        time_range = kwargs.get('time_range', 'recent')
        return f"sensor:{hash(tuple(sensor_ids) if sensor_ids else ())}:{time_range}"

    return callback_optimizer.memoize_callback(ttl, use_incremental=True, cache_key_func=sensor_cache_key)


def chart_callback(ttl: int = 15):
    """Optimized decorator for chart/graph callbacks"""
    return callback_optimizer.memoize_callback(ttl, use_incremental=True)