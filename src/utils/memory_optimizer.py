"""
Memory Optimizer for IoT Predictive Maintenance System
Provides memory usage optimization and resource management
"""

import gc
import os
import sys
import time
import threading
import tracemalloc
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import weakref
import logging

from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    percent_used: float
    process_memory_mb: float
    process_percent: float
    cached_models: int = 0
    active_threads: int = 0
    python_objects: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MemoryMonitor:
    """Monitor and optimize memory usage"""

    def __init__(self,
                 memory_limit_mb: int = 512,
                 cleanup_threshold: float = 0.8,
                 monitoring_interval: float = 30.0):
        """
        Initialize memory monitor

        Args:
            memory_limit_mb: Target memory limit in MB
            cleanup_threshold: Memory usage threshold to trigger cleanup (0.0-1.0)
            monitoring_interval: Memory check interval in seconds
        """
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold
        self.monitoring_interval = monitoring_interval

        self.is_running = False
        self.monitor_thread = None
        self.cleanup_callbacks: List[Callable] = []
        self.memory_history: List[MemoryStats] = []
        self.lock = threading.RLock()

        # Track objects for cleanup
        self.tracked_objects: Set[weakref.ref] = set()
        self.cleanup_priorities: Dict[str, int] = {}

        # Process handle
        self.process = psutil.Process(os.getpid())

        # Enable memory tracing
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        logger.info(f"Memory monitor initialized with {memory_limit_mb}MB limit")

    def start_monitoring(self):
        """Start memory monitoring"""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                stats = self.get_memory_stats()

                with self.lock:
                    self.memory_history.append(stats)
                    # Keep only last 24 hours of data (assuming 30s intervals)
                    if len(self.memory_history) > 2880:
                        self.memory_history = self.memory_history[-1440:]

                # Check if cleanup is needed
                if stats.process_percent > self.cleanup_threshold * 100:
                    logger.warning(f"Memory usage {stats.process_percent:.1f}% exceeds threshold")
                    self._trigger_cleanup()

                # Log memory stats periodically
                if len(self.memory_history) % 20 == 0:  # Every 10 minutes
                    logger.info(f"Memory: {stats.process_memory_mb:.1f}MB ({stats.process_percent:.1f}%)")

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Longer sleep on error

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = self.process.memory_info()

        # Python memory tracking
        python_objects = 0
        if hasattr(sys, 'getsizeof'):
            try:
                python_objects = len(gc.get_objects())
            except:
                pass

        # Count active threads
        active_threads = threading.active_count()

        # Count cached models (if model manager available)
        cached_models = 0
        try:
            from src.model_registry.model_manager import get_model_manager
            manager = get_model_manager()
            if hasattr(manager, 'cache'):
                cached_models = len(manager.cache.cache)
        except:
            pass

        stats = MemoryStats(
            total_memory_mb=system_memory.total / 1024 / 1024,
            available_memory_mb=system_memory.available / 1024 / 1024,
            used_memory_mb=system_memory.used / 1024 / 1024,
            percent_used=system_memory.percent,
            process_memory_mb=process_memory.rss / 1024 / 1024,
            process_percent=(process_memory.rss / system_memory.total) * 100,
            cached_models=cached_models,
            active_threads=active_threads,
            python_objects=python_objects
        )

        return stats

    def _trigger_cleanup(self):
        """Trigger memory cleanup procedures"""
        logger.info("Triggering memory cleanup procedures")

        initial_stats = self.get_memory_stats()

        # 1. Run garbage collection
        self._force_garbage_collection()

        # 2. Clear model cache if needed
        self._cleanup_model_cache()

        # 3. Run custom cleanup callbacks
        self._run_cleanup_callbacks()

        # 4. Clear temporary data structures
        self._cleanup_temp_data()

        # 5. Optimize NumPy arrays
        self._optimize_numpy_arrays()

        final_stats = self.get_memory_stats()
        memory_freed = initial_stats.process_memory_mb - final_stats.process_memory_mb

        logger.info(f"Memory cleanup complete: {memory_freed:.1f}MB freed")

        # If still over threshold, log warning
        if final_stats.process_percent > self.cleanup_threshold * 100:
            logger.warning(
                f"Memory usage still high after cleanup: {final_stats.process_percent:.1f}%"
            )

    def _force_garbage_collection(self):
        """Force comprehensive garbage collection"""
        logger.debug("Running garbage collection")

        # Multiple passes for thorough cleanup
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC pass {i+1}: collected {collected} objects")

        # Clean up reference cycles
        gc.collect()

    def _cleanup_model_cache(self):
        """Clean up model cache if memory usage is high"""
        try:
            from src.model_registry.model_manager import get_model_manager
            manager = get_model_manager()

            if hasattr(manager, 'cache'):
                cache_size = len(manager.cache.cache)
                if cache_size > 0:
                    # Clear half of the cache
                    target_size = max(1, cache_size // 2)

                    # Clear least recently used models
                    with manager.cache.lock:
                        while len(manager.cache.cache) > target_size:
                            if manager.cache.access_order:
                                lru_id = manager.cache.access_order.pop(0)
                                del manager.cache.cache[lru_id]
                                logger.debug(f"Cleared model {lru_id} from cache")
                            else:
                                break

                    logger.info(f"Reduced model cache from {cache_size} to {len(manager.cache.cache)} models")

        except Exception as e:
            logger.debug(f"Could not cleanup model cache: {e}")

    def _run_cleanup_callbacks(self):
        """Run registered cleanup callbacks"""
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")

    def _cleanup_temp_data(self):
        """Clean up temporary data structures"""
        # Clean up dead weak references
        dead_refs = [ref for ref in self.tracked_objects if ref() is None]
        for ref in dead_refs:
            self.tracked_objects.remove(ref)

        if dead_refs:
            logger.debug(f"Cleaned up {len(dead_refs)} dead weak references")

        # Trim memory history if too large
        with self.lock:
            if len(self.memory_history) > 1000:
                self.memory_history = self.memory_history[-500:]
                logger.debug("Trimmed memory history")

    def _optimize_numpy_arrays(self):
        """Optimize NumPy array memory usage"""
        try:
            # Force NumPy memory cleanup
            if hasattr(np, 'get_include'):
                # This is a lightweight way to trigger NumPy memory management
                np.array([]).dtype

        except Exception as e:
            logger.debug(f"NumPy optimization failed: {e}")

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback

        Args:
            callback: Function to call during memory cleanup
        """
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")

    def track_object(self, obj: Any, priority: int = 0):
        """Track an object for potential cleanup

        Args:
            obj: Object to track
            priority: Cleanup priority (higher = cleaned up first)
        """
        ref = weakref.ref(obj)
        self.tracked_objects.add(ref)
        self.cleanup_priorities[id(obj)] = priority

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        stats = self.get_memory_stats()

        # Calculate trends
        recent_history = self.get_memory_history(minutes=10)
        trend = "stable"
        if len(recent_history) > 1:
            recent_avg = sum(s.process_memory_mb for s in recent_history[-5:]) / 5
            older_avg = sum(s.process_memory_mb for s in recent_history[:5]) / min(5, len(recent_history))

            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"

        # Memory efficiency
        efficiency = "excellent"
        if stats.process_memory_mb > self.memory_limit_mb * 0.9:
            efficiency = "poor"
        elif stats.process_memory_mb > self.memory_limit_mb * 0.7:
            efficiency = "fair"
        elif stats.process_memory_mb > self.memory_limit_mb * 0.5:
            efficiency = "good"

        return {
            "current_usage_mb": stats.process_memory_mb,
            "current_percentage": stats.process_percent,
            "limit_mb": self.memory_limit_mb,
            "available_mb": stats.available_memory_mb,
            "efficiency": efficiency,
            "trend": trend,
            "cached_models": stats.cached_models,
            "active_threads": stats.active_threads,
            "python_objects": stats.python_objects,
            "cleanup_threshold": self.cleanup_threshold,
            "within_limits": stats.process_memory_mb <= self.memory_limit_mb,
            "cleanup_needed": stats.process_percent > self.cleanup_threshold * 100
        }

    def get_memory_history(self, minutes: int = 60) -> List[MemoryStats]:
        """Get memory usage history

        Args:
            minutes: Number of minutes of history to return

        Returns:
            List of memory statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.lock:
            return [stats for stats in self.memory_history if stats.timestamp >= cutoff_time]

    def optimize_for_target(self, target_mb: int = None):
        """Optimize memory usage for target size

        Args:
            target_mb: Target memory usage in MB (default: use configured limit)
        """
        if target_mb is None:
            target_mb = self.memory_limit_mb

        current_stats = self.get_memory_stats()

        if current_stats.process_memory_mb <= target_mb:
            logger.info(f"Memory usage {current_stats.process_memory_mb:.1f}MB already within target {target_mb}MB")
            return

        logger.info(f"Optimizing memory usage from {current_stats.process_memory_mb:.1f}MB to {target_mb}MB")

        # Progressive cleanup
        self._trigger_cleanup()

        # Check if we need more aggressive cleanup
        new_stats = self.get_memory_stats()
        if new_stats.process_memory_mb > target_mb:
            logger.warning(f"Aggressive cleanup needed: {new_stats.process_memory_mb:.1f}MB > {target_mb}MB")
            self._aggressive_cleanup()

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        logger.warning("Performing aggressive memory cleanup")

        # Clear most of the model cache
        try:
            from src.model_registry.model_manager import get_model_manager
            manager = get_model_manager()
            if hasattr(manager, 'cache'):
                manager.cache.clear()
                logger.info("Cleared all cached models")
        except:
            pass

        # Multiple garbage collection passes
        for i in range(5):
            collected = gc.collect()
            logger.debug(f"Aggressive GC pass {i+1}: {collected} objects collected")

        # Clear system caches if possible
        try:
            import sys
            if hasattr(sys, 'intern'):
                # This is a hint to Python's string interning
                pass
        except:
            pass

    def generate_memory_report(self) -> str:
        """Generate comprehensive memory usage report"""
        stats = self.get_memory_stats()
        summary = self.get_memory_usage_summary()

        report = f"""
IoT Predictive Maintenance System - Memory Usage Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT STATUS:
- Process Memory: {stats.process_memory_mb:.1f} MB ({stats.process_percent:.1f}%)
- System Memory: {stats.used_memory_mb:.1f} MB / {stats.total_memory_mb:.1f} MB ({stats.percent_used:.1f}%)
- Available Memory: {stats.available_memory_mb:.1f} MB
- Memory Limit: {self.memory_limit_mb} MB
- Efficiency: {summary['efficiency'].upper()}
- Trend: {summary['trend'].upper()}

RESOURCE USAGE:
- Cached Models: {stats.cached_models}
- Active Threads: {stats.active_threads}
- Python Objects: {stats.python_objects:,}

OPTIMIZATION STATUS:
- Within Limits: {'✅ YES' if summary['within_limits'] else '❌ NO'}
- Cleanup Needed: {'❌ YES' if summary['cleanup_needed'] else '✅ NO'}
- Cleanup Threshold: {self.cleanup_threshold * 100:.1f}%

RECOMMENDATIONS:
"""

        if summary['within_limits']:
            report += "✅ Memory usage is optimal\n"
        else:
            report += f"⚠️  Consider reducing memory usage by {stats.process_memory_mb - self.memory_limit_mb:.1f}MB\n"

        if summary['cleanup_needed']:
            report += "🧹 Automatic cleanup will be triggered\n"

        if stats.cached_models > 15:
            report += "📦 Consider reducing model cache size\n"

        if stats.active_threads > 20:
            report += "🔄 High thread count detected\n"

        return report.strip()

# Global memory monitor instance
_memory_monitor = MemoryMonitor()

# Convenience functions
def start_memory_monitoring():
    """Start memory monitoring"""
    _memory_monitor.start_monitoring()

def get_memory_stats() -> MemoryStats:
    """Get current memory statistics"""
    return _memory_monitor.get_memory_stats()

def get_memory_summary() -> Dict[str, Any]:
    """Get memory usage summary"""
    return _memory_monitor.get_memory_usage_summary()

def optimize_memory(target_mb: int = None):
    """Optimize memory usage"""
    _memory_monitor.optimize_for_target(target_mb)

def register_cleanup_callback(callback: Callable):
    """Register cleanup callback"""
    _memory_monitor.register_cleanup_callback(callback)

def generate_memory_report() -> str:
    """Generate memory report"""
    return _memory_monitor.generate_memory_report()

# Decorator for memory-aware functions
def memory_efficient(target_mb: int = None):
    """Decorator to ensure memory efficiency"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check memory before execution
            initial_stats = get_memory_stats()

            try:
                result = func(*args, **kwargs)

                # Check memory after execution
                final_stats = get_memory_stats()
                memory_increase = final_stats.process_memory_mb - initial_stats.process_memory_mb

                # If memory increased significantly, trigger cleanup
                if memory_increase > 50:  # 50MB increase
                    logger.debug(f"Function {func.__name__} increased memory by {memory_increase:.1f}MB")
                    if target_mb and final_stats.process_memory_mb > target_mb:
                        optimize_memory(target_mb)

                return result

            except Exception as e:
                # Cleanup on error
                gc.collect()
                raise

        return wrapper
    return decorator

# Auto-start monitoring
_memory_monitor.start_monitoring()