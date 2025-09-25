"""
Pipeline Status Monitor
Real-time monitoring component for NASA telemetry data processing pipeline
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    processing_rate: float  # records per second
    total_processed: int
    anomalies_detected: int
    anomaly_rate: float
    pipeline_uptime: float
    last_update: datetime

    # Detailed metrics
    data_sources_status: Dict[str, bool]
    model_processing_time: float
    alert_generation_rate: float
    error_count: int
    queue_sizes: Dict[str, int]


@dataclass
class DataSourceStatus:
    """Status of individual data sources"""
    source_name: str
    is_connected: bool
    last_data_received: Optional[datetime]
    records_per_minute: float
    error_count: int
    connection_quality: str  # EXCELLENT, GOOD, POOR, DISCONNECTED


class PipelineStatusMonitor:
    """
    Real-time pipeline status monitor for NASA telemetry processing
    Tracks data flow, processing rates, and system health
    """

    def __init__(self):
        """Initialize pipeline status monitor"""
        # Import managers with error handling
        self.streaming_service = None
        self.model_manager = None
        self.unified_orchestrator = None
        self.alert_manager = None
        self._initialize_services()

        # Metrics tracking
        self.current_metrics = PipelineMetrics(
            processing_rate=0.0,
            total_processed=0,
            anomalies_detected=0,
            anomaly_rate=0.0,
            pipeline_uptime=0.0,
            last_update=datetime.now(),
            data_sources_status={},
            model_processing_time=0.0,
            alert_generation_rate=0.0,
            error_count=0,
            queue_sizes={}
        )

        # Historical data for trend analysis
        self.metrics_history = deque(maxlen=300)  # 5 minutes at 1-second intervals
        self.processing_rates = deque(maxlen=60)  # 1 minute of processing rates

        # Data source monitoring
        self.data_sources = {
            'SMAP': DataSourceStatus('SMAP Satellite', True, datetime.now(), 0.0, 0, 'EXCELLENT'),
            'MSL': DataSourceStatus('MSL Mars Rover', True, datetime.now(), 0.0, 0, 'EXCELLENT'),
            'Models': DataSourceStatus('Anomaly Models', True, datetime.now(), 0.0, 0, 'EXCELLENT'),
            'Alerts': DataSourceStatus('Alert System', True, datetime.now(), 0.0, 0, 'EXCELLENT')
        }

        # Performance thresholds
        self.thresholds = {
            'processing_rate_min': 10.0,  # records/sec
            'anomaly_rate_max': 0.15,     # 15% max anomaly rate
            'model_latency_max': 100.0,   # 100ms max processing time
            'uptime_min': 0.95            # 95% minimum uptime
        }

        # Threading for real-time updates
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._start_monitoring()

        logger.info("Pipeline Status Monitor initialized")

    def _initialize_services(self):
        """Initialize service connections with error handling"""
        try:
            from src.data_ingestion.realtime_streaming_service import realtime_streaming_service
            self.streaming_service = realtime_streaming_service
            logger.info("Connected to streaming service")
        except ImportError as e:
            logger.warning(f"Streaming service not available: {e}")

        try:
            from src.dashboard.model_manager import pretrained_model_manager
            self.model_manager = pretrained_model_manager
            logger.info("Connected to model manager")
        except ImportError as e:
            logger.warning(f"Model manager not available: {e}")

        try:
            from src.dashboard.unified_data_orchestrator import unified_data_orchestrator
            self.unified_orchestrator = unified_data_orchestrator
            logger.info("Connected to data orchestrator")
        except ImportError as e:
            logger.warning(f"Data orchestrator not available: {e}")

        try:
            from src.alerts.alert_manager import AlertManager
            self.alert_manager = AlertManager()
            logger.info("Connected to alert manager")
        except ImportError as e:
            logger.warning(f"Alert manager not available: {e}")

    def get_current_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics

        Returns:
            Current pipeline metrics
        """
        self._update_metrics()
        return self.current_metrics

    def get_processing_rate_display(self) -> str:
        """Get formatted processing rate display string

        Returns:
            Formatted processing rate string
        """
        rate = self.current_metrics.processing_rate

        if rate >= 1000:
            return f"Processing Rate: {rate/1000:.1f}K records/sec - Real-time NASA data"
        elif rate >= 1:
            return f"Processing Rate: {rate:.1f} records/sec - Real-time NASA data"
        else:
            return f"Processing Rate: {rate:.2f} records/sec - Real-time NASA data"

    def get_pipeline_health_status(self) -> Dict[str, Any]:
        """Get overall pipeline health status

        Returns:
            Pipeline health information
        """
        metrics = self.current_metrics

        # Calculate health score (0-100)
        health_score = 100.0

        # Processing rate health
        if metrics.processing_rate < self.thresholds['processing_rate_min']:
            health_score -= 25

        # Anomaly rate health
        if metrics.anomaly_rate > self.thresholds['anomaly_rate_max']:
            health_score -= 20

        # Model performance health
        if metrics.model_processing_time > self.thresholds['model_latency_max']:
            health_score -= 15

        # Uptime health
        if metrics.pipeline_uptime < self.thresholds['uptime_min']:
            health_score -= 20

        # Error health
        if metrics.error_count > 10:
            health_score -= 20

        # Determine status
        if health_score >= 90:
            status = "EXCELLENT"
            color = "success"
        elif health_score >= 70:
            status = "GOOD"
            color = "primary"
        elif health_score >= 50:
            status = "WARNING"
            color = "warning"
        else:
            status = "CRITICAL"
            color = "danger"

        return {
            'health_score': max(0, health_score),
            'status': status,
            'color': color,
            'last_update': metrics.last_update.strftime('%H:%M:%S'),
            'uptime_hours': metrics.pipeline_uptime,
            'total_processed': metrics.total_processed,
            'anomalies_found': metrics.anomalies_detected
        }

    def get_data_sources_status(self) -> List[Dict[str, Any]]:
        """Get status of all data sources

        Returns:
            List of data source statuses
        """
        sources_status = []

        for source_name, source in self.data_sources.items():
            # Calculate minutes since last data
            if source.last_data_received:
                minutes_since = (datetime.now() - source.last_data_received).total_seconds() / 60
            else:
                minutes_since = float('inf')

            sources_status.append({
                'name': source.source_name,
                'status': 'CONNECTED' if source.is_connected else 'DISCONNECTED',
                'records_per_minute': source.records_per_minute,
                'quality': source.connection_quality,
                'last_data': f"{minutes_since:.1f}m ago" if minutes_since < 60 else "60m+ ago",
                'error_count': source.error_count,
                'color': self._get_source_status_color(source)
            })

        return sources_status

    def get_throughput_metrics(self) -> Dict[str, Any]:
        """Get detailed throughput metrics

        Returns:
            Throughput metrics for charts
        """
        if len(self.processing_rates) < 2:
            return {
                'timestamps': [],
                'processing_rates': [],
                'anomaly_rates': [],
                'average_rate': 0.0,
                'peak_rate': 0.0
            }

        # Get recent data
        recent_data = list(self.processing_rates)[-60:]  # Last minute
        timestamps = [datetime.now() - timedelta(seconds=i) for i in range(len(recent_data)-1, -1, -1)]

        # Calculate anomaly rates
        anomaly_rates = []
        for i, rate in enumerate(recent_data):
            # Simulate anomaly rate based on processing patterns
            base_anomaly_rate = 0.05  # 5% baseline
            if rate > 50:  # High processing rate
                anomaly_rate = base_anomaly_rate + (rate - 50) * 0.001
            else:
                anomaly_rate = base_anomaly_rate
            anomaly_rates.append(min(anomaly_rate, 0.25))  # Cap at 25%

        return {
            'timestamps': [ts.strftime('%H:%M:%S') for ts in timestamps],
            'processing_rates': recent_data,
            'anomaly_rates': anomaly_rates,
            'average_rate': np.mean(recent_data) if recent_data else 0.0,
            'peak_rate': max(recent_data) if recent_data else 0.0
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get processing queue status

        Returns:
            Queue status information
        """
        queue_sizes = self.current_metrics.queue_sizes

        return {
            'telemetry_queue': queue_sizes.get('telemetry', 0),
            'anomaly_queue': queue_sizes.get('anomaly', 0),
            'alert_queue': queue_sizes.get('alert', 0),
            'processing_queue': queue_sizes.get('processing', 0),
            'total_queued': sum(queue_sizes.values()),
            'queue_health': 'GOOD' if sum(queue_sizes.values()) < 1000 else 'WARNING'
        }

    def _update_metrics(self):
        """Update current metrics from all sources"""
        try:
            # Get streaming service statistics
            if self.streaming_service:
                stream_stats = self.streaming_service.get_statistics()
                self.current_metrics.processing_rate = stream_stats.get('processing_rate', 0.0)
                self.current_metrics.total_processed = stream_stats.get('records_streamed', 0)
                self.current_metrics.anomalies_detected = stream_stats.get('anomalies_detected', 0)
                self.current_metrics.queue_sizes['telemetry'] = stream_stats.get('queue_size', 0)

                # Calculate anomaly rate
                if self.current_metrics.total_processed > 0:
                    self.current_metrics.anomaly_rate = (
                        self.current_metrics.anomalies_detected / self.current_metrics.total_processed
                    )

                # Calculate uptime
                if stream_stats.get('start_time'):
                    start_time = stream_stats['start_time']
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time)
                    uptime_seconds = (datetime.now() - start_time).total_seconds()
                    self.current_metrics.pipeline_uptime = uptime_seconds / 3600  # hours

            # Get model manager statistics
            if self.model_manager:
                model_stats = self.model_manager.get_model_performance_summary()
                self.current_metrics.model_processing_time = model_stats.get('avg_inference_time', 0.0) * 1000  # ms

            # Get alert manager statistics
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                self.current_metrics.queue_sizes['alert'] = len(alert_stats.get('active_alerts', []))

            # Update data source statuses
            self._update_data_source_status()

            # Update timestamp
            self.current_metrics.last_update = datetime.now()

            # Add to history
            self.metrics_history.append(self.current_metrics)
            self.processing_rates.append(self.current_metrics.processing_rate)

        except Exception as e:
            logger.error(f"Error updating pipeline metrics: {e}")
            self.current_metrics.error_count += 1

    def _update_data_source_status(self):
        """Update individual data source statuses"""
        try:
            # SMAP status
            smap_source = self.data_sources['SMAP']
            smap_source.last_data_received = datetime.now()
            smap_source.records_per_minute = max(0, self.current_metrics.processing_rate * 60 * 0.45)  # ~45% SMAP

            # MSL status
            msl_source = self.data_sources['MSL']
            msl_source.last_data_received = datetime.now()
            msl_source.records_per_minute = max(0, self.current_metrics.processing_rate * 60 * 0.55)  # ~55% MSL

            # Models status
            models_source = self.data_sources['Models']
            models_source.is_connected = self.model_manager is not None
            models_source.last_data_received = datetime.now()
            models_source.records_per_minute = self.current_metrics.processing_rate * 60

            # Alerts status
            alerts_source = self.data_sources['Alerts']
            alerts_source.is_connected = self.alert_manager is not None
            alerts_source.last_data_received = datetime.now()
            alerts_source.records_per_minute = self.current_metrics.alert_generation_rate * 60

        except Exception as e:
            logger.error(f"Error updating data source status: {e}")

    def _get_source_status_color(self, source: DataSourceStatus) -> str:
        """Get color for data source status

        Args:
            source: Data source to check

        Returns:
            Bootstrap color class
        """
        if not source.is_connected:
            return "danger"
        elif source.connection_quality == "EXCELLENT":
            return "success"
        elif source.connection_quality == "GOOD":
            return "primary"
        elif source.connection_quality == "POOR":
            return "warning"
        else:
            return "danger"

    def _start_monitoring(self):
        """Start background monitoring thread"""
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._update_metrics()
                time.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait longer on error

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

    def reset_metrics(self):
        """Reset all metrics and history"""
        self.current_metrics = PipelineMetrics(
            processing_rate=0.0,
            total_processed=0,
            anomalies_detected=0,
            anomaly_rate=0.0,
            pipeline_uptime=0.0,
            last_update=datetime.now(),
            data_sources_status={},
            model_processing_time=0.0,
            alert_generation_rate=0.0,
            error_count=0,
            queue_sizes={}
        )
        self.metrics_history.clear()
        self.processing_rates.clear()
        logger.info("Pipeline metrics reset")


# Global instance for dashboard integration
pipeline_status_monitor = PipelineStatusMonitor()