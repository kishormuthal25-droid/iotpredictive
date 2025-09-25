"""
Advanced Multi-Level Caching System for IoT Performance Optimization
Implements Redis-based distributed caching with intelligent cache strategies
"""

import redis
import json
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
import threading
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import asyncio
import os
import gzip
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_retrieval_time: float = 0.0
    memory_usage: int = 0
    redis_usage: int = 0


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # L1 Cache (In-Memory)
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl: int = 300  # 5 minutes

    # L2 Cache (Redis)
    l2_enabled: bool = True
    l2_ttl: int = 1800  # 30 minutes

    # Performance settings
    compression_enabled: bool = True
    async_operations: bool = True
    batch_operations: bool = True

    # Sensor-specific settings
    sensor_data_ttl: int = 60  # 1 minute for real-time sensor data
    aggregated_data_ttl: int = 600  # 10 minutes for aggregated data
    dashboard_component_ttl: int = 120  # 2 minutes for dashboard components


class AdvancedCacheManager:
    """
    Multi-level caching system with Redis backend and intelligent strategies
    Optimized for 80-sensor real-time processing and sub-second dashboard response
    """

    def __init__(self, config: CacheConfig = None):
        """Initialize advanced cache manager"""
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()

        # L1 Cache (In-Memory) - Fastest access
        self.l1_cache: Dict[str, tuple] = {}  # (value, timestamp, ttl)
        self.l1_access_times: Dict[str, datetime] = {}  # LRU tracking

        # L2 Cache options: Redis (preferred) or File (fallback)
        self.redis_client = None
        self.file_cache_dir = None
        self.cache_fallback_chain = []

        # Initialize cache hierarchy
        self._initialize_redis()
        self._setup_fallback_chain()

        # Cache key patterns for efficient invalidation
        self.key_patterns = {
            'sensor_data': 'sensor:{sensor_id}:{time_window}',
            'equipment_data': 'equipment:{equipment_id}:{time_range}',
            'aggregated_metrics': 'metrics:{subsystem}:{period}',
            'dashboard_component': 'dashboard:{component}:{params_hash}',
            'anomaly_results': 'anomaly:{equipment_id}:{time_range}',
            'forecast_data': 'forecast:{model}:{equipment_id}:{horizon}'
        }

        # Background maintenance
        self._start_maintenance_thread()

        logger.info("Advanced Cache Manager initialized successfully")

    def _initialize_redis(self):
        """Initialize Redis connection with error handling"""
        if not self.config.redis_enabled:
            logger.info("Redis caching disabled in configuration")
            return

        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,  # Handle binary data
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis connection established: {self.config.redis_host}:{self.config.redis_port}")

        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to file cache + in-memory cache.")
            self.redis_client = None
            self.config.l2_enabled = False
            # Initialize file-based fallback cache
            self._initialize_file_cache()

    def _initialize_file_cache(self):
        """Initialize file-based cache as Redis fallback"""
        try:
            # Create cache directory
            self.file_cache_dir = Path("./cache/file_cache")
            self.file_cache_dir.mkdir(parents=True, exist_ok=True)

            # Test write permission
            test_file = self.file_cache_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()

            logger.info(f"File cache initialized at: {self.file_cache_dir}")

        except Exception as e:
            logger.warning(f"File cache initialization failed: {e}. Using memory-only cache.")
            self.file_cache_dir = None

    def _setup_fallback_chain(self):
        """Setup cache fallback chain based on available options"""
        self.cache_fallback_chain = ["memory"]  # Always available

        if self.redis_client is not None:
            self.cache_fallback_chain.insert(0, "redis")
        elif self.file_cache_dir is not None:
            self.cache_fallback_chain.insert(0, "file")

        logger.info(f"Cache fallback chain: {' -> '.join(self.cache_fallback_chain)}")

    def _get_file_cache_path(self, key: str) -> Path:
        """Generate file path for cache key"""
        if not self.file_cache_dir:
            return None

        # Create safe filename from cache key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.file_cache_dir / f"{safe_key}.cache"

    def _write_file_cache(self, key: str, value: Any, ttl: int) -> bool:
        """Write to file cache"""
        try:
            if not self.file_cache_dir:
                return False

            cache_path = self._get_file_cache_path(key)
            cache_data = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl,
                'key': key  # For debugging
            }

            # Use gzip compression for larger objects
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            return True

        except Exception as e:
            logger.error(f"Error writing file cache for key {key}: {e}")
            return False

    def _read_file_cache(self, key: str) -> Optional[Any]:
        """Read from file cache"""
        try:
            if not self.file_cache_dir:
                return None

            cache_path = self._get_file_cache_path(key)
            if not cache_path.exists():
                return None

            with gzip.open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Check if expired
            current_time = time.time()
            if current_time - cache_data['timestamp'] > cache_data['ttl']:
                # Clean up expired file
                cache_path.unlink(missing_ok=True)
                return None

            return cache_data['value']

        except Exception as e:
            logger.error(f"Error reading file cache for key {key}: {e}")
            return None

    def _generate_cache_key(self, pattern: str, **kwargs) -> str:
        """Generate standardized cache key from pattern"""
        try:
            key = pattern.format(**kwargs)
            return f"iot_cache:{key}"
        except KeyError as e:
            logger.error(f"Missing parameter for cache key pattern: {e}")
            return f"iot_cache:generic:{hash(str(kwargs))}"

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage with compression"""
        try:
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value, default=str).encode('utf-8')
            elif isinstance(value, (np.ndarray, pd.DataFrame)):
                serialized = pickle.dumps(value)
            else:
                serialized = pickle.dumps(value)

            # Apply compression if enabled and beneficial
            if self.config.compression_enabled and len(serialized) > 1024:
                import gzip
                compressed = gzip.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # Only if 20%+ compression
                    return b'compressed:' + compressed

            return serialized

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return pickle.dumps(str(value))

    def _deserialize_value(self, serialized: bytes) -> Any:
        """Deserialize value with decompression support"""
        try:
            if serialized.startswith(b'compressed:'):
                import gzip
                serialized = gzip.decompress(serialized[11:])

            # Try JSON first for simple types
            try:
                return json.loads(serialized.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return pickle.loads(serialized)

        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from multi-level cache with performance tracking"""
        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # L1 Cache check (fastest)
            if self.config.l1_enabled:
                l1_value = self._get_from_l1(key)
                if l1_value is not None:
                    self.metrics.hits += 1
                    self._update_metrics(start_time)
                    return l1_value

            # L2 Cache check (Redis)
            if self.config.l2_enabled and self.redis_client:
                l2_value = self._get_from_l2(key)
                if l2_value is not None:
                    # Store in L1 for faster future access
                    if self.config.l1_enabled:
                        self._set_to_l1(key, l2_value, self.config.l1_ttl)

                    self.metrics.hits += 1
                    self._update_metrics(start_time)
                    return l2_value

            # Cache miss
            self.metrics.misses += 1
            self._update_metrics(start_time)
            return default

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.metrics.misses += 1
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multi-level cache"""
        ttl = ttl or self.config.l1_ttl
        success = True

        try:
            # Set in L1 Cache
            if self.config.l1_enabled:
                success &= self._set_to_l1(key, value, ttl)

            # Set in L2 Cache (Redis)
            if self.config.l2_enabled and self.redis_client:
                success &= self._set_to_l2(key, value, ttl)

            return success

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    def _get_from_l1(self, key: str) -> Any:
        """Get from L1 in-memory cache"""
        with self.lock:
            if key in self.l1_cache:
                value, timestamp, ttl = self.l1_cache[key]

                # Check expiration
                if datetime.now() - timestamp < timedelta(seconds=ttl):
                    # Update access time for LRU
                    self.l1_access_times[key] = datetime.now()
                    return value
                else:
                    # Expired, remove
                    del self.l1_cache[key]
                    if key in self.l1_access_times:
                        del self.l1_access_times[key]

            return None

    def _set_to_l1(self, key: str, value: Any, ttl: int) -> bool:
        """Set in L1 in-memory cache with LRU eviction"""
        with self.lock:
            # Check if cache is full and evict LRU items
            if len(self.l1_cache) >= self.config.l1_max_size:
                self._evict_lru_items()

            self.l1_cache[key] = (value, datetime.now(), ttl)
            self.l1_access_times[key] = datetime.now()
            return True

    def _get_from_l2(self, key: str) -> Any:
        """Get from L2 Redis cache"""
        try:
            if not self.redis_client:
                return None

            serialized = self.redis_client.get(key)
            if serialized:
                return self._deserialize_value(serialized)
            return None

        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def _set_to_l2(self, key: str, value: Any, ttl: int) -> bool:
        """Set in L2 Redis cache"""
        try:
            if not self.redis_client:
                return False

            serialized = self._serialize_value(value)
            return self.redis_client.setex(key, ttl, serialized)

        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def _evict_lru_items(self):
        """Evict least recently used items from L1 cache"""
        if not self.l1_access_times:
            return

        # Sort by access time and remove oldest 20%
        sorted_items = sorted(self.l1_access_times.items(), key=lambda x: x[1])
        items_to_remove = max(1, len(sorted_items) // 5)

        for key, _ in sorted_items[:items_to_remove]:
            if key in self.l1_cache:
                del self.l1_cache[key]
            if key in self.l1_access_times:
                del self.l1_access_times[key]

    def _update_metrics(self, start_time: float):
        """Update cache performance metrics"""
        retrieval_time = time.time() - start_time

        # Update hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = self.metrics.hits / self.metrics.total_requests

        # Update average retrieval time (exponential moving average)
        alpha = 0.1
        self.metrics.avg_retrieval_time = (
            alpha * retrieval_time +
            (1 - alpha) * self.metrics.avg_retrieval_time
        )

    def invalidate_pattern(self, pattern: str, **kwargs) -> int:
        """Invalidate cache entries matching pattern"""
        key_prefix = self._generate_cache_key(pattern, **kwargs)
        count = 0

        # Invalidate L1 cache
        with self.lock:
            keys_to_remove = [k for k in self.l1_cache.keys() if k.startswith(key_prefix.split('*')[0])]
            for key in keys_to_remove:
                del self.l1_cache[key]
                if key in self.l1_access_times:
                    del self.l1_access_times[key]
                count += 1

        # Invalidate L2 cache (Redis)
        if self.redis_client:
            try:
                redis_keys = self.redis_client.keys(key_prefix)
                if redis_keys:
                    count += self.redis_client.delete(*redis_keys)
            except Exception as e:
                logger.error(f"Redis pattern invalidation error: {e}")

        logger.debug(f"Invalidated {count} cache entries for pattern: {pattern}")
        return count

    def clear_all(self):
        """Clear all cache levels"""
        # Clear L1
        with self.lock:
            self.l1_cache.clear()
            self.l1_access_times.clear()

        # Clear L2 (Redis)
        if self.redis_client:
            try:
                keys = self.redis_client.keys("iot_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Redis clear error: {e}")

        logger.info("All cache levels cleared")

    def get_metrics(self) -> CacheMetrics:
        """Get current cache performance metrics"""
        # Update memory usage
        with self.lock:
            self.metrics.memory_usage = len(self.l1_cache)

        if self.redis_client:
            try:
                info = self.redis_client.info('memory')
                self.metrics.redis_usage = info.get('used_memory', 0)
            except:
                self.metrics.redis_usage = 0

        return self.metrics

    def _start_maintenance_thread(self):
        """Start background maintenance thread"""
        def maintenance_loop():
            while True:
                try:
                    # Clean expired L1 entries
                    with self.lock:
                        expired_keys = []
                        now = datetime.now()
                        for key, (value, timestamp, ttl) in self.l1_cache.items():
                            if now - timestamp >= timedelta(seconds=ttl):
                                expired_keys.append(key)

                        for key in expired_keys:
                            del self.l1_cache[key]
                            if key in self.l1_access_times:
                                del self.l1_access_times[key]

                    # Sleep for 60 seconds before next maintenance
                    time.sleep(60)

                except Exception as e:
                    logger.error(f"Cache maintenance error: {e}")
                    time.sleep(60)

        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()


def cache_result(ttl: int = 300, key_pattern: str = None):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_pattern:
                cache_key = key_pattern.format(**kwargs)
            else:
                # Generate key from function name and arguments
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"func:{func.__name__}:{hashlib.md5(args_str.encode()).hexdigest()}"

            # Try to get from cache
            cached_result = advanced_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            advanced_cache.set(cache_key, result, ttl)
            return result

        return wrapper
    return decorator


# Global cache manager instance
advanced_cache = AdvancedCacheManager()


# Sensor-specific cache helpers
class SensorCacheHelper:
    """Helper class for sensor-specific caching operations"""

    @staticmethod
    def cache_sensor_data(sensor_id: str, time_window: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache sensor data with standardized key"""
        ttl = ttl or advanced_cache.config.sensor_data_ttl
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['sensor_data'],
            sensor_id=sensor_id,
            time_window=time_window
        )
        return advanced_cache.set(key, data, ttl)

    @staticmethod
    def get_sensor_data(sensor_id: str, time_window: str) -> Optional[Dict[str, Any]]:
        """Get cached sensor data"""
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['sensor_data'],
            sensor_id=sensor_id,
            time_window=time_window
        )
        return advanced_cache.get(key)

    @staticmethod
    def invalidate_sensor_data(sensor_id: str = None):
        """Invalidate sensor data cache"""
        if sensor_id:
            pattern = advanced_cache.key_patterns['sensor_data'].replace('{time_window}', '*')
            advanced_cache.invalidate_pattern(pattern, sensor_id=sensor_id)
        else:
            # Invalidate all sensor data
            advanced_cache.invalidate_pattern("sensor:*")


# Equipment-specific cache helpers
class EquipmentCacheHelper:
    """Helper class for equipment-specific caching operations"""

    @staticmethod
    def cache_equipment_data(equipment_id: str, time_range: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Cache equipment data with standardized key"""
        ttl = ttl or advanced_cache.config.aggregated_data_ttl
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['equipment_data'],
            equipment_id=equipment_id,
            time_range=time_range
        )
        return advanced_cache.set(key, data, ttl)

    @staticmethod
    def get_equipment_data(equipment_id: str, time_range: str) -> Optional[Dict[str, Any]]:
        """Get cached equipment data"""
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['equipment_data'],
            equipment_id=equipment_id,
            time_range=time_range
        )
        return advanced_cache.get(key)


# Dashboard-specific cache helpers
class DashboardCacheHelper:
    """Helper class for dashboard component caching"""

    @staticmethod
    def cache_component(component: str, data: Any, params: Dict = None, ttl: int = None) -> bool:
        """Cache dashboard component with parameters hash"""
        ttl = ttl or advanced_cache.config.dashboard_component_ttl
        params_hash = hashlib.md5(str(sorted((params or {}).items())).encode()).hexdigest()[:8]
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['dashboard_component'],
            component=component,
            params_hash=params_hash
        )
        return advanced_cache.set(key, data, ttl)

    @staticmethod
    def get_component(component: str, params: Dict = None) -> Any:
        """Get cached dashboard component"""
        params_hash = hashlib.md5(str(sorted((params or {}).items())).encode()).hexdigest()[:8]
        key = advanced_cache._generate_cache_key(
            advanced_cache.key_patterns['dashboard_component'],
            component=component,
            params_hash=params_hash
        )
        return advanced_cache.get(key)