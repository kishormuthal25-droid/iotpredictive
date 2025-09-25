"""
Test Performance Monitor

Real-time performance monitoring and metrics collection during testing:
- System resource monitoring (CPU, Memory, Disk, Network)
- Application performance metrics
- Test execution metrics
- Performance regression detection
- Real-time alerting and reporting
"""

import os
import time
import threading
import json
import psutil
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import logging
import queue
import asyncio
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_io_read_mb_s: float
    disk_io_write_mb_s: float
    network_bytes_sent_s: float
    network_bytes_recv_s: float
    process_count: int
    thread_count: int
    open_files: int
    test_specific_metrics: Dict[str, Any]


@dataclass
class TestExecutionMetrics:
    """Container for test execution metrics"""
    test_name: str
    test_suite: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    status: str  # running, passed, failed, error
    assertions_count: int
    data_processed: int
    operations_performed: int
    memory_peak_mb: float
    cpu_peak_percent: float
    error_details: Optional[str]


@dataclass
class PerformanceAlert:
    """Container for performance alerts"""
    alert_id: str
    timestamp: datetime
    severity: str  # info, warning, critical
    metric_name: str
    current_value: float
    threshold_value: float
    description: str
    test_context: Optional[str]


class TestPerformanceMonitor:
    """Real-time performance monitoring for testing"""

    def __init__(self, monitoring_interval: float = 1.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()

        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.test_execution_history = []
        self.performance_alerts = []

        # Current test context
        self.current_test = None
        self.test_start_metrics = None

        # Performance thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_io_warning': 100.0,  # MB/s
            'disk_io_critical': 500.0,  # MB/s
            'test_duration_warning': 300.0,  # seconds
            'test_duration_critical': 600.0,  # seconds
            'memory_leak_threshold': 50.0  # MB increase per test
        }

        # Baseline metrics for comparison
        self.baseline_metrics = None

        # Performance callbacks
        self.alert_callbacks = []
        self.metrics_callbacks = []

        # Initialize process reference
        self.process = psutil.Process()

        logger.info("TestPerformanceMonitor initialized")

    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Performance monitoring stopped")

    def start_test_monitoring(self, test_name: str, test_suite: str = "unknown"):
        """Start monitoring specific test execution"""
        current_metrics = self._collect_current_metrics()

        self.current_test = TestExecutionMetrics(
            test_name=test_name,
            test_suite=test_suite,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=0.0,
            status="running",
            assertions_count=0,
            data_processed=0,
            operations_performed=0,
            memory_peak_mb=current_metrics.memory_used_mb,
            cpu_peak_percent=current_metrics.cpu_percent,
            error_details=None
        )

        self.test_start_metrics = current_metrics
        logger.info(f"Started monitoring test: {test_name}")

    def end_test_monitoring(self, status: str = "passed", error_details: str = None):
        """End monitoring specific test execution"""
        if not self.current_test:
            logger.warning("No active test monitoring to end")
            return

        end_time = datetime.now()
        self.current_test.end_time = end_time
        self.current_test.duration_seconds = (end_time - self.current_test.start_time).total_seconds()
        self.current_test.status = status
        self.current_test.error_details = error_details

        # Check for performance issues during test
        self._analyze_test_performance()

        # Store test execution record
        self.test_execution_history.append(self.current_test)

        logger.info(f"Ended monitoring test: {self.current_test.test_name} ({status})")
        self.current_test = None
        self.test_start_metrics = None

    def update_test_metrics(self, assertions_count: int = None, data_processed: int = None,
                          operations_performed: int = None):
        """Update test-specific metrics"""
        if not self.current_test:
            return

        if assertions_count is not None:
            self.current_test.assertions_count = assertions_count
        if data_processed is not None:
            self.current_test.data_processed = data_processed
        if operations_performed is not None:
            self.current_test.operations_performed = operations_performed

    def set_performance_thresholds(self, **thresholds):
        """Set custom performance thresholds"""
        self.thresholds.update(thresholds)
        logger.info(f"Updated performance thresholds: {thresholds}")

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def add_metrics_callback(self, callback: Callable[[PerformanceMetrics], None]):
        """Add callback for metrics updates"""
        self.metrics_callbacks.append(callback)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_metrics_history(self, duration_minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def get_test_execution_summary(self) -> Dict[str, Any]:
        """Get summary of test execution performance"""
        if not self.test_execution_history:
            return {}

        completed_tests = [t for t in self.test_execution_history if t.end_time is not None]

        if not completed_tests:
            return {}

        return {
            'total_tests': len(completed_tests),
            'passed_tests': len([t for t in completed_tests if t.status == "passed"]),
            'failed_tests': len([t for t in completed_tests if t.status == "failed"]),
            'error_tests': len([t for t in completed_tests if t.status == "error"]),
            'avg_duration': np.mean([t.duration_seconds for t in completed_tests]),
            'max_duration': max([t.duration_seconds for t in completed_tests]),
            'min_duration': min([t.duration_seconds for t in completed_tests]),
            'avg_memory_peak': np.mean([t.memory_peak_mb for t in completed_tests]),
            'max_memory_peak': max([t.memory_peak_mb for t in completed_tests]),
            'avg_cpu_peak': np.mean([t.cpu_peak_percent for t in completed_tests]),
            'max_cpu_peak': max([t.cpu_peak_percent for t in completed_tests]),
            'total_data_processed': sum([t.data_processed for t in completed_tests]),
            'total_operations': sum([t.operations_performed for t in completed_tests])
        }

    def get_performance_alerts(self, severity: str = None, hours: int = 1) -> List[PerformanceAlert]:
        """Get performance alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.performance_alerts if a.timestamp >= cutoff_time]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def capture_baseline(self, duration_seconds: int = 60):
        """Capture baseline performance metrics"""
        logger.info(f"Capturing baseline metrics for {duration_seconds} seconds...")

        baseline_metrics = []
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            metrics = self._collect_current_metrics()
            baseline_metrics.append(metrics)
            time.sleep(self.monitoring_interval)

        # Calculate baseline averages
        self.baseline_metrics = {
            'cpu_avg': np.mean([m.cpu_percent for m in baseline_metrics]),
            'memory_avg': np.mean([m.memory_percent for m in baseline_metrics]),
            'disk_io_read_avg': np.mean([m.disk_io_read_mb_s for m in baseline_metrics]),
            'disk_io_write_avg': np.mean([m.disk_io_write_mb_s for m in baseline_metrics]),
            'network_sent_avg': np.mean([m.network_bytes_sent_s for m in baseline_metrics]),
            'network_recv_avg': np.mean([m.network_bytes_recv_s for m in baseline_metrics]),
            'capture_time': datetime.now(),
            'sample_count': len(baseline_metrics)
        }

        logger.info(f"Baseline captured: CPU={self.baseline_metrics['cpu_avg']:.1f}%, "
                   f"Memory={self.baseline_metrics['memory_avg']:.1f}%")

    def detect_performance_regression(self, test_name: str, tolerance_percent: float = 20.0) -> Dict[str, Any]:
        """Detect performance regression for a specific test"""
        # Find historical executions of this test
        historical_executions = [
            t for t in self.test_execution_history
            if t.test_name == test_name and t.status == "passed" and t.end_time is not None
        ]

        if len(historical_executions) < 2:
            return {'regression_detected': False, 'reason': 'Insufficient historical data'}

        # Calculate historical averages (excluding most recent)
        historical_data = historical_executions[:-1]
        recent_execution = historical_executions[-1]

        historical_avg_duration = np.mean([t.duration_seconds for t in historical_data])
        historical_avg_memory = np.mean([t.memory_peak_mb for t in historical_data])
        historical_avg_cpu = np.mean([t.cpu_peak_percent for t in historical_data])

        # Check for regressions
        duration_increase = ((recent_execution.duration_seconds - historical_avg_duration) / historical_avg_duration) * 100
        memory_increase = ((recent_execution.memory_peak_mb - historical_avg_memory) / historical_avg_memory) * 100
        cpu_increase = ((recent_execution.cpu_peak_percent - historical_avg_cpu) / historical_avg_cpu) * 100

        regression_detected = (
            duration_increase > tolerance_percent or
            memory_increase > tolerance_percent or
            cpu_increase > tolerance_percent
        )

        return {
            'regression_detected': regression_detected,
            'test_name': test_name,
            'recent_execution': asdict(recent_execution),
            'historical_averages': {
                'duration': historical_avg_duration,
                'memory_peak': historical_avg_memory,
                'cpu_peak': historical_avg_cpu
            },
            'performance_changes': {
                'duration_change_percent': duration_increase,
                'memory_change_percent': memory_increase,
                'cpu_change_percent': cpu_increase
            },
            'tolerance_percent': tolerance_percent,
            'sample_size': len(historical_data)
        }

    def generate_performance_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'report_generation_time': datetime.now().isoformat(),
            'monitoring_summary': {
                'total_monitoring_duration': len(self.metrics_history) * self.monitoring_interval,
                'metrics_collected': len(self.metrics_history),
                'tests_monitored': len(self.test_execution_history),
                'alerts_generated': len(self.performance_alerts)
            },
            'system_performance_summary': self._generate_system_performance_summary(),
            'test_execution_summary': self.get_test_execution_summary(),
            'performance_alerts_summary': self._generate_alerts_summary(),
            'baseline_comparison': self._generate_baseline_comparison(),
            'recommendations': self._generate_performance_recommendations()
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to {output_path}")

        return report

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Performance monitoring loop started")

        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)

                # Update current test metrics if running
                if self.current_test:
                    self._update_current_test_metrics(metrics)

                # Check thresholds and generate alerts
                self._check_performance_thresholds(metrics)

                # Call metrics callbacks
                for callback in self.metrics_callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Metrics callback error: {e}")

                # Add metrics to queue for external consumption
                try:
                    self.metrics_queue.put_nowait(metrics)
                except queue.Full:
                    pass  # Queue full, skip this update

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

        logger.info("Performance monitoring loop ended")

    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                time_delta = time.time() - self._last_disk_time
                read_delta = disk_io.read_bytes - self._last_disk_io.read_bytes
                write_delta = disk_io.write_bytes - self._last_disk_io.write_bytes
                disk_io_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                disk_io_write_mb_s = (write_delta / time_delta) / (1024 * 1024)
            else:
                disk_io_read_mb_s = 0.0
                disk_io_write_mb_s = 0.0

            self._last_disk_io = disk_io
            self._last_disk_time = time.time()

            # Network metrics
            network_io = psutil.net_io_counters()
            if hasattr(self, '_last_network_io'):
                time_delta = time.time() - self._last_network_time
                sent_delta = network_io.bytes_sent - self._last_network_io.bytes_sent
                recv_delta = network_io.bytes_recv - self._last_network_io.bytes_recv
                network_bytes_sent_s = sent_delta / time_delta
                network_bytes_recv_s = recv_delta / time_delta
            else:
                network_bytes_sent_s = 0.0
                network_bytes_recv_s = 0.0

            self._last_network_io = network_io
            self._last_network_time = time.time()

            # Process metrics
            process_count = len(psutil.pids())
            thread_count = self.process.num_threads()
            open_files = len(self.process.open_files())

            # Test-specific metrics
            test_specific_metrics = {}
            if self.current_test:
                test_specific_metrics['current_test'] = self.current_test.test_name
                test_specific_metrics['test_duration'] = (datetime.now() - self.current_test.start_time).total_seconds()

            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_io_read_mb_s=disk_io_read_mb_s,
                disk_io_write_mb_s=disk_io_write_mb_s,
                network_bytes_sent_s=network_bytes_sent_s,
                network_bytes_recv_s=network_bytes_recv_s,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                test_specific_metrics=test_specific_metrics
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return default metrics on error
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_io_read_mb_s=0.0,
                disk_io_write_mb_s=0.0,
                network_bytes_sent_s=0.0,
                network_bytes_recv_s=0.0,
                process_count=0,
                thread_count=0,
                open_files=0,
                test_specific_metrics={}
            )

    def _update_current_test_metrics(self, metrics: PerformanceMetrics):
        """Update current test with peak metrics"""
        if not self.current_test:
            return

        # Update peak values
        self.current_test.memory_peak_mb = max(
            self.current_test.memory_peak_mb, metrics.memory_used_mb
        )
        self.current_test.cpu_peak_percent = max(
            self.current_test.cpu_peak_percent, metrics.cpu_percent
        )

    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and generate alerts"""
        alerts_to_generate = []

        # CPU thresholds
        if metrics.cpu_percent >= self.thresholds['cpu_critical']:
            alerts_to_generate.append(('cpu_critical', metrics.cpu_percent, self.thresholds['cpu_critical']))
        elif metrics.cpu_percent >= self.thresholds['cpu_warning']:
            alerts_to_generate.append(('cpu_warning', metrics.cpu_percent, self.thresholds['cpu_warning']))

        # Memory thresholds
        if metrics.memory_percent >= self.thresholds['memory_critical']:
            alerts_to_generate.append(('memory_critical', metrics.memory_percent, self.thresholds['memory_critical']))
        elif metrics.memory_percent >= self.thresholds['memory_warning']:
            alerts_to_generate.append(('memory_warning', metrics.memory_percent, self.thresholds['memory_warning']))

        # Disk I/O thresholds
        total_disk_io = metrics.disk_io_read_mb_s + metrics.disk_io_write_mb_s
        if total_disk_io >= self.thresholds['disk_io_critical']:
            alerts_to_generate.append(('disk_io_critical', total_disk_io, self.thresholds['disk_io_critical']))
        elif total_disk_io >= self.thresholds['disk_io_warning']:
            alerts_to_generate.append(('disk_io_warning', total_disk_io, self.thresholds['disk_io_warning']))

        # Test duration thresholds (if test is running)
        if self.current_test:
            test_duration = (datetime.now() - self.current_test.start_time).total_seconds()
            if test_duration >= self.thresholds['test_duration_critical']:
                alerts_to_generate.append(('test_duration_critical', test_duration, self.thresholds['test_duration_critical']))
            elif test_duration >= self.thresholds['test_duration_warning']:
                alerts_to_generate.append(('test_duration_warning', test_duration, self.thresholds['test_duration_warning']))

        # Generate alerts
        for alert_type, current_value, threshold_value in alerts_to_generate:
            alert = self._create_alert(alert_type, current_value, threshold_value, metrics.timestamp)
            if alert:
                self.performance_alerts.append(alert)
                # Call alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.warning(f"Alert callback error: {e}")

    def _create_alert(self, alert_type: str, current_value: float, threshold_value: float,
                     timestamp: datetime) -> Optional[PerformanceAlert]:
        """Create performance alert"""
        # Avoid duplicate alerts (check if similar alert was generated recently)
        recent_cutoff = timestamp - timedelta(minutes=1)
        recent_alerts = [a for a in self.performance_alerts if a.timestamp >= recent_cutoff and alert_type in a.alert_id]

        if recent_alerts:
            return None  # Skip duplicate alert

        alert_config = {
            'cpu_warning': ('warning', 'CPU usage high'),
            'cpu_critical': ('critical', 'CPU usage critical'),
            'memory_warning': ('warning', 'Memory usage high'),
            'memory_critical': ('critical', 'Memory usage critical'),
            'disk_io_warning': ('warning', 'Disk I/O high'),
            'disk_io_critical': ('critical', 'Disk I/O critical'),
            'test_duration_warning': ('warning', 'Test duration excessive'),
            'test_duration_critical': ('critical', 'Test duration critical')
        }

        if alert_type not in alert_config:
            return None

        severity, description = alert_config[alert_type]

        return PerformanceAlert(
            alert_id=f"{alert_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            severity=severity,
            metric_name=alert_type.split('_')[0],
            current_value=current_value,
            threshold_value=threshold_value,
            description=f"{description}: {current_value:.1f} >= {threshold_value:.1f}",
            test_context=self.current_test.test_name if self.current_test else None
        )

    def _analyze_test_performance(self):
        """Analyze performance during test execution"""
        if not self.current_test or not self.test_start_metrics:
            return

        # Check for memory leaks
        memory_increase = self.current_test.memory_peak_mb - self.test_start_metrics.memory_used_mb
        if memory_increase > self.thresholds['memory_leak_threshold']:
            alert = PerformanceAlert(
                alert_id=f"memory_leak_{self.current_test.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                severity='warning',
                metric_name='memory_increase',
                current_value=memory_increase,
                threshold_value=self.thresholds['memory_leak_threshold'],
                description=f"Potential memory leak detected: {memory_increase:.1f}MB increase",
                test_context=self.current_test.test_name
            )
            self.performance_alerts.append(alert)

    def _generate_system_performance_summary(self) -> Dict[str, Any]:
        """Generate system performance summary"""
        if not self.metrics_history:
            return {}

        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        disk_read_values = [m.disk_io_read_mb_s for m in self.metrics_history]
        disk_write_values = [m.disk_io_write_mb_s for m in self.metrics_history]

        return {
            'cpu_statistics': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values),
                'std': np.std(cpu_values)
            },
            'memory_statistics': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values),
                'std': np.std(memory_values)
            },
            'disk_io_statistics': {
                'read_avg_mb_s': np.mean(disk_read_values),
                'write_avg_mb_s': np.mean(disk_write_values),
                'read_max_mb_s': np.max(disk_read_values),
                'write_max_mb_s': np.max(disk_write_values)
            },
            'monitoring_period': {
                'start_time': self.metrics_history[0].timestamp.isoformat(),
                'end_time': self.metrics_history[-1].timestamp.isoformat(),
                'duration_minutes': len(self.metrics_history) * self.monitoring_interval / 60
            }
        }

    def _generate_alerts_summary(self) -> Dict[str, Any]:
        """Generate alerts summary"""
        if not self.performance_alerts:
            return {}

        severity_counts = {}
        metric_counts = {}

        for alert in self.performance_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            metric_counts[alert.metric_name] = metric_counts.get(alert.metric_name, 0) + 1

        return {
            'total_alerts': len(self.performance_alerts),
            'alerts_by_severity': severity_counts,
            'alerts_by_metric': metric_counts,
            'most_recent_alert': self.performance_alerts[-1].description if self.performance_alerts else None
        }

    def _generate_baseline_comparison(self) -> Dict[str, Any]:
        """Generate baseline comparison"""
        if not self.baseline_metrics or not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-min(60, len(self.metrics_history)):]  # Last 60 samples

        current_averages = {
            'cpu_avg': np.mean([m.cpu_percent for m in recent_metrics]),
            'memory_avg': np.mean([m.memory_percent for m in recent_metrics]),
            'disk_io_read_avg': np.mean([m.disk_io_read_mb_s for m in recent_metrics]),
            'disk_io_write_avg': np.mean([m.disk_io_write_mb_s for m in recent_metrics])
        }

        return {
            'baseline_metrics': self.baseline_metrics,
            'current_averages': current_averages,
            'performance_delta': {
                'cpu_change_percent': ((current_averages['cpu_avg'] - self.baseline_metrics['cpu_avg']) / max(self.baseline_metrics['cpu_avg'], 1)) * 100,
                'memory_change_percent': ((current_averages['memory_avg'] - self.baseline_metrics['memory_avg']) / max(self.baseline_metrics['memory_avg'], 1)) * 100,
                'disk_read_change_percent': ((current_averages['disk_io_read_avg'] - self.baseline_metrics['disk_io_read_avg']) / max(self.baseline_metrics['disk_io_read_avg'], 0.1)) * 100,
                'disk_write_change_percent': ((current_averages['disk_io_write_avg'] - self.baseline_metrics['disk_io_write_avg']) / max(self.baseline_metrics['disk_io_write_avg'], 0.1)) * 100
            }
        }

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Analyze system performance
        if self.metrics_history:
            avg_cpu = np.mean([m.cpu_percent for m in self.metrics_history])
            avg_memory = np.mean([m.memory_percent for m in self.metrics_history])

            if avg_cpu > 70:
                recommendations.append("High CPU usage detected. Consider optimizing CPU-intensive operations or increasing available CPU resources.")

            if avg_memory > 70:
                recommendations.append("High memory usage detected. Review memory usage patterns and consider increasing available RAM.")

        # Analyze test execution patterns
        test_summary = self.get_test_execution_summary()
        if test_summary:
            if test_summary.get('avg_duration', 0) > 60:
                recommendations.append("Long average test execution time. Consider optimizing test logic or using parallel execution.")

            failure_rate = test_summary.get('failed_tests', 0) / max(test_summary.get('total_tests', 1), 1)
            if failure_rate > 0.1:
                recommendations.append(f"High test failure rate ({failure_rate:.1%}). Review test stability and system reliability.")

        # Analyze alerts
        critical_alerts = len([a for a in self.performance_alerts if a.severity == 'critical'])
        if critical_alerts > 0:
            recommendations.append(f"Critical performance alerts detected ({critical_alerts}). Immediate attention required.")

        if not recommendations:
            recommendations.append("Performance metrics are within acceptable ranges. Continue monitoring.")

        return recommendations


# Utility functions for easy integration
class PerformanceContext:
    """Context manager for performance monitoring during tests"""

    def __init__(self, monitor: TestPerformanceMonitor, test_name: str, test_suite: str = "unknown"):
        self.monitor = monitor
        self.test_name = test_name
        self.test_suite = test_suite

    def __enter__(self):
        self.monitor.start_test_monitoring(self.test_name, self.test_suite)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.monitor.end_test_monitoring("error", str(exc_val))
        else:
            self.monitor.end_test_monitoring("passed")

    def update_metrics(self, **kwargs):
        """Update test metrics"""
        self.monitor.update_test_metrics(**kwargs)


def create_performance_monitor(monitoring_interval: float = 1.0) -> TestPerformanceMonitor:
    """Factory function to create performance monitor"""
    return TestPerformanceMonitor(monitoring_interval=monitoring_interval)


# Example usage and testing
def main():
    """Example usage of TestPerformanceMonitor"""

    # Create monitor
    monitor = TestPerformanceMonitor(monitoring_interval=0.5)

    # Add alert callback
    def alert_handler(alert: PerformanceAlert):
        print(f"ALERT: {alert.severity.upper()} - {alert.description}")

    monitor.add_alert_callback(alert_handler)

    # Start monitoring
    monitor.start_monitoring()

    try:
        # Capture baseline
        print("Capturing baseline...")
        monitor.capture_baseline(duration_seconds=5)

        # Simulate test execution
        with PerformanceContext(monitor, "test_example_performance", "performance_suite") as ctx:
            print("Running test simulation...")

            # Simulate some work
            for i in range(10):
                time.sleep(0.5)
                ctx.update_metrics(assertions_count=i+1, data_processed=(i+1)*100)

                # Simulate CPU load
                x = sum(j*j for j in range(10000))

        # Generate report
        print("Generating performance report...")
        report = monitor.generate_performance_report("performance_report.json")

        print(f"Report generated with {report['monitoring_summary']['metrics_collected']} metrics collected")

        # Show summary
        summary = monitor.get_test_execution_summary()
        print(f"Test execution summary: {summary}")

    finally:
        monitor.stop_monitoring()

    print("Performance monitoring example completed")


if __name__ == "__main__":
    main()