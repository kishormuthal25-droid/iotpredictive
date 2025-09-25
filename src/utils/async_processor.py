"""
Async Parallel Processor for IoT Sensor Stream Management
High-performance async processing for 80-sensor concurrent operations
"""

import asyncio
import aiohttp
import aiodns
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
from functools import wraps
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class AsyncProcessorConfig:
    """Configuration for async processor"""
    # Async settings
    max_concurrent_tasks: int = 80  # Support 80 sensors concurrently
    max_workers: int = 16          # Thread pool size
    timeout: int = 30              # Task timeout in seconds

    # Batch processing
    batch_size: int = 20           # Process sensors in batches
    batch_timeout: float = 0.1     # Max time to wait for batch

    # Performance settings
    enable_process_pool: bool = False  # Use process pool for CPU-intensive tasks
    process_pool_workers: int = 4      # Process pool size

    # Memory optimization
    task_result_cache_size: int = 1000
    cleanup_interval: int = 60         # Cleanup every 60 seconds


@dataclass
class AsyncTaskResult:
    """Result of an async task"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProcessorMetrics:
    """Async processor performance metrics"""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_execution_time: float = 0.0
    concurrent_tasks: int = 0
    batch_operations: int = 0
    total_processing_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AsyncSensorProcessor:
    """
    High-performance async processor for sensor data processing
    Optimized for 80-sensor concurrent operations with sub-second response
    """

    def __init__(self, config: AsyncProcessorConfig = None):
        """Initialize async sensor processor"""
        self.config = config or AsyncProcessorConfig()
        self.metrics = ProcessorMetrics()

        # Event loop and semaphore for concurrency control
        self.loop = None
        self.semaphore = None
        self.is_running = False

        # Task management
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: deque = deque(maxlen=self.config.task_result_cache_size)
        self.task_queue = asyncio.Queue(maxsize=self.config.max_concurrent_tasks * 2)

        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = None
        if self.config.enable_process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.process_pool_workers)

        # Batch processing
        self.batch_queue = asyncio.Queue()
        self.batch_tasks = {}

        # Performance tracking
        self.execution_times = deque(maxlen=1000)
        self.metrics_lock = threading.Lock()

        # Cleanup management
        self.cleanup_task = None

        logger.info(f"Async Sensor Processor initialized: "
                   f"max_concurrent={self.config.max_concurrent_tasks}, "
                   f"workers={self.config.max_workers}")

    async def start(self):
        """Start the async processor"""
        if self.is_running:
            logger.warning("Async processor is already running")
            return

        self.loop = asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        self.is_running = True

        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Async sensor processor started")

    async def stop(self):
        """Stop the async processor"""
        self.is_running = False

        # Cancel active tasks
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
            self.active_tasks.clear()

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

        logger.info("Async sensor processor stopped")

    async def process_sensor_batch(self, sensor_data_batch: List[Dict[str, Any]],
                                 processor_func: Callable) -> List[AsyncTaskResult]:
        """
        Process a batch of sensor data asynchronously
        Optimized for 80-sensor concurrent processing
        """
        if not self.is_running:
            await self.start()

        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Create tasks for each sensor in the batch
            tasks = []
            for i, sensor_data in enumerate(sensor_data_batch):
                task_id = f"{batch_id}_sensor_{i}"
                task = asyncio.create_task(
                    self._process_single_sensor(task_id, sensor_data, processor_func)
                )
                tasks.append(task)
                self.active_tasks[task_id] = task

            # Execute batch with timeout
            batch_timeout = self.config.timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=batch_timeout
            )

            # Process results
            processed_results = []
            for i, result in enumerate(results):
                task_id = f"{batch_id}_sensor_{i}"

                if isinstance(result, Exception):
                    processed_results.append(AsyncTaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(result),
                        execution_time=time.time() - start_time
                    ))
                else:
                    processed_results.append(result)

                # Clean up task reference
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]

            # Update metrics
            self._update_batch_metrics(len(sensor_data_batch), time.time() - start_time)

            return processed_results

        except asyncio.TimeoutError:
            logger.error(f"Batch {batch_id} timed out after {batch_timeout}s")
            # Cancel remaining tasks
            for task_id in [f"{batch_id}_sensor_{i}" for i in range(len(sensor_data_batch))]:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].cancel()
                    del self.active_tasks[task_id]

            return [AsyncTaskResult(
                task_id=f"{batch_id}_sensor_{i}",
                success=False,
                error="Timeout",
                execution_time=time.time() - start_time
            ) for i in range(len(sensor_data_batch))]

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return [AsyncTaskResult(
                task_id=f"{batch_id}_error",
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )]

    async def _process_single_sensor(self, task_id: str, sensor_data: Dict[str, Any],
                                   processor_func: Callable) -> AsyncTaskResult:
        """Process a single sensor with concurrency control"""
        async with self.semaphore:  # Limit concurrent operations
            start_time = time.time()

            try:
                # Execute processor function in thread pool for CPU-intensive tasks
                if asyncio.iscoroutinefunction(processor_func):
                    result = await processor_func(sensor_data)
                else:
                    # Run sync function in thread pool
                    result = await self.loop.run_in_executor(
                        self.thread_pool, processor_func, sensor_data
                    )

                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)

                task_result = AsyncTaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time
                )

                self.task_results.append(task_result)
                self._update_task_metrics(True, execution_time)

                return task_result

            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)

                logger.error(f"Sensor processing error for {task_id}: {error_msg}")

                task_result = AsyncTaskResult(
                    task_id=task_id,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time
                )

                self.task_results.append(task_result)
                self._update_task_metrics(False, execution_time)

                return task_result

    async def process_sensor_stream(self, sensor_id: str, data_stream: asyncio.Queue,
                                  processor_func: Callable) -> None:
        """
        Process continuous sensor data stream
        Optimized for real-time 80-sensor processing
        """
        logger.info(f"Starting stream processing for sensor {sensor_id}")

        while self.is_running:
            try:
                # Wait for data with timeout
                sensor_data = await asyncio.wait_for(
                    data_stream.get(),
                    timeout=1.0
                )

                # Process data asynchronously
                task_id = f"stream_{sensor_id}_{int(time.time() * 1000)}"

                # Don't wait for completion to maintain stream flow
                task = asyncio.create_task(
                    self._process_single_sensor(task_id, sensor_data, processor_func)
                )

                # Store task reference but don't block
                self.active_tasks[task_id] = task

                # Clean up completed tasks periodically
                if len(self.active_tasks) > self.config.max_concurrent_tasks:
                    await self._cleanup_completed_tasks()

            except asyncio.TimeoutError:
                # No data available, continue
                continue
            except Exception as e:
                logger.error(f"Stream processing error for {sensor_id}: {e}")
                await asyncio.sleep(1)  # Brief pause on error

    async def parallel_anomaly_detection(self, sensor_data_list: List[Dict[str, Any]],
                                       detection_func: Callable) -> List[Dict[str, Any]]:
        """
        Parallel anomaly detection for multiple sensors
        Optimized for 80-sensor concurrent anomaly detection
        """
        # Split into optimal batch sizes
        batch_size = min(self.config.batch_size, len(sensor_data_list))
        batches = [
            sensor_data_list[i:i + batch_size]
            for i in range(0, len(sensor_data_list), batch_size)
        ]

        all_results = []

        # Process batches in parallel
        batch_tasks = []
        for batch in batches:
            task = asyncio.create_task(
                self.process_sensor_batch(batch, detection_func)
            )
            batch_tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch anomaly detection failed: {batch_result}")
                continue

            for task_result in batch_result:
                if task_result.success:
                    all_results.append(task_result.result)
                else:
                    logger.warning(f"Anomaly detection failed for task {task_result.task_id}: {task_result.error}")

        return all_results

    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks to prevent memory buildup"""
        completed_tasks = []

        for task_id, task in list(self.active_tasks.items()):
            if task.done():
                completed_tasks.append(task_id)

        for task_id in completed_tasks:
            del self.active_tasks[task_id]

        if completed_tasks:
            logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.is_running:
            try:
                await self._cleanup_completed_tasks()

                # Update concurrent task count
                with self.metrics_lock:
                    self.metrics.concurrent_tasks = len(self.active_tasks)

                await asyncio.sleep(self.config.cleanup_interval)

            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(10)

    def _update_task_metrics(self, success: bool, execution_time: float):
        """Update task performance metrics"""
        with self.metrics_lock:
            if success:
                self.metrics.tasks_completed += 1
            else:
                self.metrics.tasks_failed += 1

            self.metrics.tasks_submitted += 1
            self.metrics.total_processing_time += execution_time

            # Update average execution time (exponential moving average)
            alpha = 0.1
            self.metrics.avg_execution_time = (
                alpha * execution_time +
                (1 - alpha) * self.metrics.avg_execution_time
            )

            self.metrics.last_updated = datetime.now()

    def _update_batch_metrics(self, batch_size: int, execution_time: float):
        """Update batch processing metrics"""
        with self.metrics_lock:
            self.metrics.batch_operations += 1

    def get_metrics(self) -> ProcessorMetrics:
        """Get current processor metrics"""
        with self.metrics_lock:
            # Update concurrent tasks count
            self.metrics.concurrent_tasks = len(self.active_tasks)
            return ProcessorMetrics(
                tasks_submitted=self.metrics.tasks_submitted,
                tasks_completed=self.metrics.tasks_completed,
                tasks_failed=self.metrics.tasks_failed,
                avg_execution_time=self.metrics.avg_execution_time,
                concurrent_tasks=self.metrics.concurrent_tasks,
                batch_operations=self.metrics.batch_operations,
                total_processing_time=self.metrics.total_processing_time,
                last_updated=self.metrics.last_updated
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        metrics = self.get_metrics()

        success_rate = 0.0
        if metrics.tasks_submitted > 0:
            success_rate = metrics.tasks_completed / metrics.tasks_submitted

        return {
            'success_rate': success_rate,
            'tasks_per_second': metrics.tasks_completed / max(1, metrics.total_processing_time),
            'avg_execution_time': metrics.avg_execution_time,
            'concurrent_tasks': metrics.concurrent_tasks,
            'batch_operations': metrics.batch_operations,
            'memory_usage': len(self.task_results),
            'active_tasks': len(self.active_tasks),
            'recent_execution_times': list(self.execution_times)[-10:] if self.execution_times else []
        }


# Global async processor instance
async_processor = AsyncSensorProcessor()


# Convenience functions for async operations
async def process_sensors_async(sensor_data_list: List[Dict[str, Any]],
                               processor_func: Callable) -> List[AsyncTaskResult]:
    """Process multiple sensors asynchronously"""
    if not async_processor.is_running:
        await async_processor.start()

    return await async_processor.process_sensor_batch(sensor_data_list, processor_func)


async def detect_anomalies_async(sensor_data_list: List[Dict[str, Any]],
                                detection_func: Callable) -> List[Dict[str, Any]]:
    """Detect anomalies across multiple sensors asynchronously"""
    if not async_processor.is_running:
        await async_processor.start()

    return await async_processor.parallel_anomaly_detection(sensor_data_list, detection_func)


def async_sensor_task(func: Callable) -> Callable:
    """Decorator to make sensor processing functions async-compatible"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await loop.run_in_executor(None, func, *args, **kwargs)

    return async_wrapper