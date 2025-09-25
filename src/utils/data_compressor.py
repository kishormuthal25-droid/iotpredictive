"""
Intelligent Data Compression and Streaming for Dashboard Updates
Optimized for 80-sensor real-time data with minimal network overhead
"""

import gzip
import json
import pickle
import zlib
import lz4.frame
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time
from collections import deque
import base64
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Compression performance metrics"""
    total_operations: int = 0
    total_original_size: int = 0
    total_compressed_size: int = 0
    avg_compression_ratio: float = 0.0
    avg_compression_time: float = 0.0
    compression_method_stats: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class StreamingConfig:
    """Configuration for streaming optimizations"""
    # Compression settings
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress data larger than 1KB
    preferred_method: str = "lz4"      # lz4, gzip, zlib, pickle

    # Streaming settings
    chunk_size: int = 8192            # 8KB chunks for streaming
    max_payload_size: int = 102400    # 100KB max payload
    enable_delta_compression: bool = True

    # Performance settings
    compression_level: int = 1        # Fast compression
    enable_parallel: bool = True      # Parallel compression for large data


class IntelligentDataCompressor:
    """
    Intelligent data compression system for dashboard updates
    Optimized for 80-sensor real-time streaming with minimal latency
    """

    def __init__(self, config: StreamingConfig = None):
        """Initialize data compressor"""
        self.config = config or StreamingConfig()
        self.metrics = CompressionMetrics()
        self.metrics_lock = threading.RLock()

        # Compression methods registry
        self.compression_methods = {
            'gzip': self._compress_gzip,
            'zlib': self._compress_zlib,
            'lz4': self._compress_lz4,
            'pickle': self._compress_pickle,
            'json': self._compress_json
        }

        self.decompression_methods = {
            'gzip': self._decompress_gzip,
            'zlib': self._decompress_zlib,
            'lz4': self._decompress_lz4,
            'pickle': self._decompress_pickle,
            'json': self._decompress_json
        }

        # Delta compression cache
        self.delta_cache: Dict[str, Dict[str, Any]] = {}
        self.delta_lock = threading.RLock()

        logger.info(f"Data Compressor initialized: method={self.config.preferred_method}, "
                   f"threshold={self.config.compression_threshold}B")

    def compress_dashboard_data(self, data: Any, data_key: str = None,
                              force_method: str = None) -> Dict[str, Any]:
        """
        Compress dashboard data with intelligent method selection

        Args:
            data: Data to compress
            data_key: Unique key for delta compression
            force_method: Force specific compression method

        Returns:
            Compressed data package with metadata
        """
        start_time = time.time()

        try:
            # Serialize data first
            serialized_data = self._serialize_data(data)
            original_size = len(serialized_data)

            # Check if compression is needed
            if original_size < self.config.compression_threshold:
                return {
                    'data': serialized_data,
                    'compressed': False,
                    'method': 'none',
                    'original_size': original_size,
                    'compressed_size': original_size,
                    'compression_ratio': 1.0,
                    'compression_time': time.time() - start_time
                }

            # Try delta compression first if enabled
            if self.config.enable_delta_compression and data_key:
                delta_result = self._try_delta_compression(serialized_data, data_key)
                if delta_result:
                    self._update_metrics(original_size, len(delta_result), 'delta', time.time() - start_time)
                    return {
                        'data': delta_result,
                        'compressed': True,
                        'method': 'delta',
                        'original_size': original_size,
                        'compressed_size': len(delta_result),
                        'compression_ratio': len(delta_result) / original_size,
                        'compression_time': time.time() - start_time,
                        'delta_key': data_key
                    }

            # Select compression method
            method = force_method or self._select_compression_method(data, original_size)

            # Compress data
            compressed_data = self.compression_methods[method](serialized_data)
            compressed_size = len(compressed_data)

            # Update metrics
            self._update_metrics(original_size, compressed_size, method, time.time() - start_time)

            # Store for delta compression if enabled
            if self.config.enable_delta_compression and data_key:
                with self.delta_lock:
                    self.delta_cache[data_key] = {
                        'data': serialized_data,
                        'timestamp': datetime.now(),
                        'size': original_size
                    }

            return {
                'data': base64.b64encode(compressed_data).decode('utf-8'),
                'compressed': True,
                'method': method,
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compressed_size / original_size,
                'compression_time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Fallback to uncompressed
            serialized_data = self._serialize_data(data)
            return {
                'data': serialized_data,
                'compressed': False,
                'method': 'fallback',
                'original_size': len(serialized_data),
                'compressed_size': len(serialized_data),
                'compression_ratio': 1.0,
                'compression_time': time.time() - start_time,
                'error': str(e)
            }

    def decompress_dashboard_data(self, compressed_package: Dict[str, Any]) -> Any:
        """
        Decompress dashboard data package

        Args:
            compressed_package: Compressed data package

        Returns:
            Original data
        """
        try:
            if not compressed_package.get('compressed', False):
                return self._deserialize_data(compressed_package['data'])

            method = compressed_package['method']
            compressed_data = compressed_package['data']

            if method == 'delta':
                return self._decompress_delta(compressed_data, compressed_package.get('delta_key'))

            # Decode base64 and decompress
            compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
            decompressed_data = self.decompression_methods[method](compressed_bytes)

            return self._deserialize_data(decompressed_data)

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

    def compress_sensor_stream(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress sensor stream data with sensor-specific optimizations

        Args:
            sensor_data: List of sensor data points

        Returns:
            Compressed sensor stream package
        """
        if not sensor_data:
            return {'data': [], 'compressed': False, 'method': 'none'}

        try:
            # Optimize sensor data structure for compression
            optimized_data = self._optimize_sensor_data(sensor_data)

            # Use LZ4 for fast compression of streaming data
            return self.compress_dashboard_data(optimized_data, force_method='lz4')

        except Exception as e:
            logger.error(f"Sensor stream compression failed: {e}")
            return {'data': sensor_data, 'compressed': False, 'method': 'fallback', 'error': str(e)}

    def compress_chart_data(self, chart_figure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress chart/graph data with chart-specific optimizations

        Args:
            chart_figure: Plotly figure dictionary

        Returns:
            Compressed chart package
        """
        try:
            # Optimize chart data for compression
            optimized_figure = self._optimize_chart_data(chart_figure)

            # Use gzip for charts (good compression ratio)
            return self.compress_dashboard_data(optimized_figure, force_method='gzip')

        except Exception as e:
            logger.error(f"Chart compression failed: {e}")
            return {'data': chart_figure, 'compressed': False, 'method': 'fallback', 'error': str(e)}

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str, separators=(',', ':')).encode('utf-8')
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, bytes):
            return data
        else:
            return pickle.dumps(data)

    def _deserialize_data(self, data: Union[str, bytes]) -> Any:
        """Deserialize data from bytes or string"""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        elif isinstance(data, bytes):
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)
        return data

    def _select_compression_method(self, data: Any, size: int) -> str:
        """Select optimal compression method based on data characteristics"""
        # For real-time streaming, prioritize speed
        if size < 10000:  # < 10KB
            return 'lz4'  # Fastest
        elif isinstance(data, (dict, list)) and size < 100000:  # < 100KB
            return 'gzip'  # Good balance
        else:
            return self.config.preferred_method

    def _optimize_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize sensor data structure for better compression"""
        if not sensor_data:
            return {}

        # Convert to columnar format for better compression
        optimized = {
            'timestamps': [],
            'sensor_ids': [],
            'values': [],
            'anomaly_scores': [],
            'anomaly_flags': []
        }

        for record in sensor_data:
            optimized['timestamps'].append(record.get('timestamp'))
            optimized['sensor_ids'].append(record.get('sensor_id'))
            optimized['values'].append(record.get('value'))
            optimized['anomaly_scores'].append(record.get('anomaly_score', 0.0))
            optimized['anomaly_flags'].append(record.get('anomaly_flag', False))

        return optimized

    def _optimize_chart_data(self, chart_figure: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize chart data for compression"""
        if not isinstance(chart_figure, dict):
            return chart_figure

        optimized = chart_figure.copy()

        # Remove or compress heavy data in chart
        if 'data' in optimized:
            for trace in optimized['data']:
                # Round numeric values to reduce precision and improve compression
                if 'x' in trace and isinstance(trace['x'], list):
                    trace['x'] = [round(x, 3) if isinstance(x, (int, float)) else x for x in trace['x']]
                if 'y' in trace and isinstance(trace['y'], list):
                    trace['y'] = [round(y, 3) if isinstance(y, (int, float)) else y for y in trace['y']]

        return optimized

    def _try_delta_compression(self, current_data: bytes, data_key: str) -> Optional[bytes]:
        """Try delta compression against cached data"""
        with self.delta_lock:
            if data_key not in self.delta_cache:
                return None

            cached_entry = self.delta_cache[data_key]
            cached_data = cached_entry['data']

            # Check if cache is recent enough
            if datetime.now() - cached_entry['timestamp'] > timedelta(minutes=5):
                return None

            # Simple delta - just check if data is similar enough
            # In practice, you'd implement sophisticated delta algorithms
            if len(current_data) > len(cached_data) * 1.5:
                return None  # Too different

            # For now, return LZ4 compressed delta marker
            delta_info = {
                'type': 'delta',
                'base_key': data_key,
                'data': current_data.decode('utf-8', errors='ignore')
            }

            return self._compress_lz4(json.dumps(delta_info).encode('utf-8'))

    def _decompress_delta(self, delta_data: bytes, data_key: str) -> bytes:
        """Decompress delta compressed data"""
        # Placeholder for delta decompression
        decompressed = self._decompress_lz4(delta_data)
        delta_info = json.loads(decompressed.decode('utf-8'))
        return delta_info['data'].encode('utf-8')

    def _compress_gzip(self, data: bytes) -> bytes:
        """Compress using gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)

    def _decompress_gzip(self, data: bytes) -> bytes:
        """Decompress using gzip"""
        return gzip.decompress(data)

    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress using zlib"""
        return zlib.compress(data, level=self.config.compression_level)

    def _decompress_zlib(self, data: bytes) -> bytes:
        """Decompress using zlib"""
        return zlib.decompress(data)

    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress using LZ4"""
        return lz4.frame.compress(data, compression_level=self.config.compression_level)

    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress using LZ4"""
        return lz4.frame.decompress(data)

    def _compress_pickle(self, data: bytes) -> bytes:
        """Compress using pickle + gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)

    def _decompress_pickle(self, data: bytes) -> bytes:
        """Decompress using pickle + gzip"""
        return gzip.decompress(data)

    def _compress_json(self, data: bytes) -> bytes:
        """Compress JSON with gzip"""
        return gzip.compress(data, compresslevel=self.config.compression_level)

    def _decompress_json(self, data: bytes) -> bytes:
        """Decompress JSON with gzip"""
        return gzip.decompress(data)

    def _update_metrics(self, original_size: int, compressed_size: int,
                       method: str, compression_time: float):
        """Update compression metrics"""
        with self.metrics_lock:
            self.metrics.total_operations += 1
            self.metrics.total_original_size += original_size
            self.metrics.total_compressed_size += compressed_size

            # Update compression ratio (exponential moving average)
            current_ratio = compressed_size / original_size
            alpha = 0.1
            self.metrics.avg_compression_ratio = (
                alpha * current_ratio +
                (1 - alpha) * self.metrics.avg_compression_ratio
            )

            # Update compression time
            self.metrics.avg_compression_time = (
                alpha * compression_time +
                (1 - alpha) * self.metrics.avg_compression_time
            )

            # Update method stats
            if method not in self.metrics.compression_method_stats:
                self.metrics.compression_method_stats[method] = 0
            self.metrics.compression_method_stats[method] += 1

            self.metrics.last_updated = datetime.now()

    def get_compression_metrics(self) -> CompressionMetrics:
        """Get current compression metrics"""
        with self.metrics_lock:
            return CompressionMetrics(
                total_operations=self.metrics.total_operations,
                total_original_size=self.metrics.total_original_size,
                total_compressed_size=self.metrics.total_compressed_size,
                avg_compression_ratio=self.metrics.avg_compression_ratio,
                avg_compression_time=self.metrics.avg_compression_time,
                compression_method_stats=self.metrics.compression_method_stats.copy(),
                last_updated=self.metrics.last_updated
            )

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get detailed compression statistics"""
        metrics = self.get_compression_metrics()

        total_savings = metrics.total_original_size - metrics.total_compressed_size
        savings_percentage = (total_savings / metrics.total_original_size * 100) if metrics.total_original_size > 0 else 0

        return {
            'total_operations': metrics.total_operations,
            'total_savings_bytes': total_savings,
            'savings_percentage': savings_percentage,
            'avg_compression_ratio': metrics.avg_compression_ratio,
            'avg_compression_time_ms': metrics.avg_compression_time * 1000,
            'method_usage': metrics.compression_method_stats,
            'cache_entries': len(self.delta_cache)
        }

    def clear_delta_cache(self):
        """Clear delta compression cache"""
        with self.delta_lock:
            self.delta_cache.clear()


# Global data compressor instance
data_compressor = IntelligentDataCompressor()


# Decorator for automatic data compression
def compress_response(method: str = None, cache_key: str = None):
    """Decorator to automatically compress function responses"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # Generate cache key if not provided
            if not cache_key:
                key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            else:
                key = cache_key

            # Compress the result
            compressed = data_compressor.compress_dashboard_data(
                result, data_key=key, force_method=method
            )

            return compressed

        return wrapper
    return decorator


# Convenience functions
def compress_sensor_data(sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compress sensor data for transmission"""
    return data_compressor.compress_sensor_stream(sensor_data)


def compress_chart_figure(figure: Dict[str, Any]) -> Dict[str, Any]:
    """Compress chart figure for transmission"""
    return data_compressor.compress_chart_data(figure)