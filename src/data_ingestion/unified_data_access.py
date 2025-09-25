"""
Unified Data Access Layer
Provides standardized data access interface for all dashboard components
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import contextmanager
import json
from enum import Enum
import time
from functools import lru_cache
import threading

# Database imports
from sqlalchemy import func, and_, or_, desc, asc, text
from sqlalchemy.orm import Session

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.data_ingestion.database_manager import (
    DatabaseManager, TelemetryData, AnomalyDetection,
    ForecastResult, MaintenanceSchedule
)
from src.data_ingestion.equipment_mapper import equipment_mapper

# Setup logging
logger = logging.getLogger(__name__)

class TimeRange(Enum):
    """Predefined time ranges for data queries"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_DAY = "24h"
    LAST_WEEK = "7d"
    LAST_MONTH = "30d"
    LAST_QUARTER = "90d"
    CUSTOM = "custom"

@dataclass
class DataQuery:
    """Standardized data query structure"""
    time_range: TimeRange = TimeRange.LAST_DAY
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    spacecraft: Optional[List[str]] = None  # ['SMAP', 'MSL']
    channels: Optional[List[str]] = None    # Specific channels
    equipment_ids: Optional[List[str]] = None
    include_anomalies: bool = True
    anomaly_only: bool = False
    limit: Optional[int] = None
    offset: int = 0
    order_by: str = "timestamp"
    order_desc: bool = True

@dataclass
class TelemetryRecord:
    """Standardized telemetry record format for dashboard use"""
    timestamp: datetime
    spacecraft: str
    channel: str
    equipment_id: str
    values: List[float]
    mean_value: float
    std_value: float
    min_value: float
    max_value: float
    is_anomaly: bool
    anomaly_score: float
    equipment_info: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class AnomalyRecord:
    """Standardized anomaly record format"""
    timestamp: datetime
    spacecraft: str
    channel: str
    equipment_id: str
    detector_type: str
    anomaly_score: float
    threshold: float
    severity: str
    description: str
    is_confirmed: bool

@dataclass
class DataSummary:
    """Summary statistics for dashboard display"""
    total_records: int
    date_range: Tuple[datetime, datetime]
    spacecraft_counts: Dict[str, int]
    channel_counts: Dict[str, int]
    anomaly_count: int
    anomaly_percentage: float
    equipment_health: Dict[str, str]  # equipment_id -> status


class UnifiedDataAccess:
    """
    Unified data access layer for all dashboard components
    Provides consistent, cached, and optimized data access
    """

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """Initialize unified data access layer"""
        self.db_manager = database_manager or DatabaseManager()

        # Cache for frequently accessed data
        self._cache = {}
        self._cache_ttl = {}
        self._cache_lock = threading.Lock()
        self._default_cache_duration = 60  # seconds

        logger.info("Unified Data Access Layer initialized")

    def _get_time_range_bounds(self, query: DataQuery) -> Tuple[datetime, datetime]:
        """Get start and end datetime for time range"""
        now = datetime.now()

        if query.time_range == TimeRange.CUSTOM:
            if not query.start_time or not query.end_time:
                raise ValueError("Custom time range requires start_time and end_time")
            return query.start_time, query.end_time

        time_deltas = {
            TimeRange.LAST_HOUR: timedelta(hours=1),
            TimeRange.LAST_6_HOURS: timedelta(hours=6),
            TimeRange.LAST_DAY: timedelta(days=1),
            TimeRange.LAST_WEEK: timedelta(days=7),
            TimeRange.LAST_MONTH: timedelta(days=30),
            TimeRange.LAST_QUARTER: timedelta(days=90)
        }

        delta = time_deltas.get(query.time_range, timedelta(days=1))
        start_time = now - delta

        return start_time, now

    @contextmanager
    def get_session(self):
        """Get database session with proper cleanup"""
        with self.db_manager.get_session() as session:
            yield session

    def _cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        params = sorted(kwargs.items())
        return f"{prefix}:{hash(str(params))}"

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        with self._cache_lock:
            if key in self._cache:
                if time.time() < self._cache_ttl.get(key, 0):
                    return self._cache[key]
                else:
                    # Remove expired entry
                    del self._cache[key]
                    if key in self._cache_ttl:
                        del self._cache_ttl[key]
        return None

    def _set_cache(self, key: str, value: Any, duration: int = None) -> None:
        """Set cached value with TTL"""
        duration = duration or self._default_cache_duration
        with self._cache_lock:
            self._cache[key] = value
            self._cache_ttl[key] = time.time() + duration

    def get_telemetry_data(self, query: DataQuery) -> List[TelemetryRecord]:
        """Get telemetry data based on query parameters

        Args:
            query: DataQuery object with filter parameters

        Returns:
            List of standardized TelemetryRecord objects
        """
        cache_key = self._cache_key("telemetry",
                                  time_range=query.time_range.value,
                                  spacecraft=query.spacecraft,
                                  channels=query.channels,
                                  limit=query.limit,
                                  anomaly_only=query.anomaly_only)

        # Check cache first
        cached_result = self._get_cached(cache_key)
        if cached_result:
            return cached_result

        try:
            with self.get_session() as session:
                # Base query
                db_query = session.query(TelemetryData)

                # Apply time range filter
                start_time, end_time = self._get_time_range_bounds(query)
                db_query = db_query.filter(
                    TelemetryData.timestamp.between(start_time, end_time)
                )

                # Apply spacecraft filter
                if query.spacecraft:
                    db_query = db_query.filter(TelemetryData.spacecraft.in_(query.spacecraft))

                # Apply channel filter
                if query.channels:
                    db_query = db_query.filter(TelemetryData.channel.in_(query.channels))

                # Apply anomaly filter
                if query.anomaly_only:
                    db_query = db_query.filter(TelemetryData.is_anomaly == True)

                # Apply ordering
                if query.order_desc:
                    db_query = db_query.order_by(desc(getattr(TelemetryData, query.order_by)))
                else:
                    db_query = db_query.order_by(asc(getattr(TelemetryData, query.order_by)))

                # Apply pagination
                if query.offset:
                    db_query = db_query.offset(query.offset)
                if query.limit:
                    db_query = db_query.limit(query.limit)

                # Execute query
                db_results = db_query.all()

                # Convert to standardized format
                telemetry_records = []
                for db_record in db_results:
                    equipment_id = f"{db_record.spacecraft}_{db_record.channel}"
                    equipment_info = equipment_mapper.get_equipment_info(equipment_id) or {}

                    record = TelemetryRecord(
                        timestamp=db_record.timestamp,
                        spacecraft=db_record.spacecraft,
                        channel=db_record.channel,
                        equipment_id=equipment_id,
                        values=db_record.data if isinstance(db_record.data, list) else [],
                        mean_value=db_record.mean_value or 0.0,
                        std_value=db_record.std_value or 0.0,
                        min_value=db_record.min_value or 0.0,
                        max_value=db_record.max_value or 0.0,
                        is_anomaly=db_record.is_anomaly or False,
                        anomaly_score=db_record.anomaly_score or 0.0,
                        equipment_info=equipment_info,
                        metadata=db_record.event_metadata or {}
                    )
                    telemetry_records.append(record)

                # Cache results
                self._set_cache(cache_key, telemetry_records)

                logger.info(f"Retrieved {len(telemetry_records)} telemetry records")
                return telemetry_records

        except Exception as e:
            logger.error(f"Failed to get telemetry data: {e}")
            return []

    def get_real_time_data(self, equipment_ids: Optional[List[str]] = None,
                          max_records: int = 1000) -> List[TelemetryRecord]:
        """Get most recent telemetry data for real-time display

        Args:
            equipment_ids: Optional list of equipment IDs to filter
            max_records: Maximum number of records to return

        Returns:
            List of most recent TelemetryRecord objects
        """
        query = DataQuery(
            time_range=TimeRange.LAST_HOUR,
            equipment_ids=equipment_ids,
            limit=max_records,
            order_by="timestamp",
            order_desc=True
        )

        return self.get_telemetry_data(query)

    def get_anomaly_data(self, query: DataQuery) -> List[AnomalyRecord]:
        """Get anomaly detection results

        Args:
            query: DataQuery object with filter parameters

        Returns:
            List of AnomalyRecord objects
        """
        cache_key = self._cache_key("anomalies",
                                  time_range=query.time_range.value,
                                  spacecraft=query.spacecraft,
                                  channels=query.channels,
                                  limit=query.limit)

        # Check cache first
        cached_result = self._get_cached(cache_key)
        if cached_result:
            return cached_result

        try:
            with self.get_session() as session:
                # Base query
                db_query = session.query(AnomalyDetection)

                # Apply time range filter
                start_time, end_time = self._get_time_range_bounds(query)
                db_query = db_query.filter(
                    AnomalyDetection.timestamp.between(start_time, end_time)
                )

                # Apply spacecraft filter
                if query.spacecraft:
                    db_query = db_query.filter(AnomalyDetection.spacecraft.in_(query.spacecraft))

                # Apply channel filter
                if query.channels:
                    db_query = db_query.filter(AnomalyDetection.channel.in_(query.channels))

                # Apply ordering
                if query.order_desc:
                    db_query = db_query.order_by(desc(AnomalyDetection.timestamp))
                else:
                    db_query = db_query.order_by(asc(AnomalyDetection.timestamp))

                # Apply pagination
                if query.offset:
                    db_query = db_query.offset(query.offset)
                if query.limit:
                    db_query = db_query.limit(query.limit)

                # Execute query
                db_results = db_query.all()

                # Convert to standardized format
                anomaly_records = []
                for db_record in db_results:
                    equipment_id = f"{db_record.spacecraft}_{db_record.channel}"

                    record = AnomalyRecord(
                        timestamp=db_record.timestamp,
                        spacecraft=db_record.spacecraft,
                        channel=db_record.channel,
                        equipment_id=equipment_id,
                        detector_type=db_record.detector_type or "unknown",
                        anomaly_score=db_record.anomaly_score,
                        threshold=db_record.threshold or 0.0,
                        severity=db_record.severity or "unknown",
                        description=db_record.description or "",
                        is_confirmed=db_record.is_confirmed or False
                    )
                    anomaly_records.append(record)

                # Cache results
                self._set_cache(cache_key, anomaly_records)

                logger.info(f"Retrieved {len(anomaly_records)} anomaly records")
                return anomaly_records

        except Exception as e:
            logger.error(f"Failed to get anomaly data: {e}")
            return []

    def get_data_summary(self, query: DataQuery) -> DataSummary:
        """Get summary statistics for dashboard overview

        Args:
            query: DataQuery object with filter parameters

        Returns:
            DataSummary object with statistics
        """
        cache_key = self._cache_key("summary",
                                  time_range=query.time_range.value,
                                  spacecraft=query.spacecraft,
                                  channels=query.channels)

        # Check cache first
        cached_result = self._get_cached(cache_key)
        if cached_result:
            return cached_result

        try:
            with self.get_session() as session:
                start_time, end_time = self._get_time_range_bounds(query)

                # Base query with time filter
                base_query = session.query(TelemetryData).filter(
                    TelemetryData.timestamp.between(start_time, end_time)
                )

                # Apply spacecraft filter
                if query.spacecraft:
                    base_query = base_query.filter(TelemetryData.spacecraft.in_(query.spacecraft))

                # Apply channel filter
                if query.channels:
                    base_query = base_query.filter(TelemetryData.channel.in_(query.channels))

                # Get total count
                total_records = base_query.count()

                # Get date range
                date_stats = base_query.with_entities(
                    func.min(TelemetryData.timestamp),
                    func.max(TelemetryData.timestamp)
                ).first()

                # Get spacecraft counts
                spacecraft_counts = dict(
                    base_query.with_entities(
                        TelemetryData.spacecraft,
                        func.count(TelemetryData.spacecraft)
                    ).group_by(TelemetryData.spacecraft).all()
                )

                # Get channel counts
                channel_counts = dict(
                    base_query.with_entities(
                        TelemetryData.channel,
                        func.count(TelemetryData.channel)
                    ).group_by(TelemetryData.channel).limit(20).all()  # Limit to top 20
                )

                # Get anomaly count
                anomaly_count = base_query.filter(TelemetryData.is_anomaly == True).count()
                anomaly_percentage = (anomaly_count / total_records * 100) if total_records > 0 else 0

                # Get equipment health (simplified)
                equipment_health = {}
                for spacecraft in spacecraft_counts.keys():
                    channels = session.query(TelemetryData.channel).filter(
                        and_(
                            TelemetryData.spacecraft == spacecraft,
                            TelemetryData.timestamp.between(start_time, end_time)
                        )
                    ).distinct().all()

                    for channel_tuple in channels:
                        channel = channel_tuple[0]
                        equipment_id = f"{spacecraft}_{channel}"

                        # Check recent anomalies
                        recent_anomalies = session.query(TelemetryData).filter(
                            and_(
                                TelemetryData.spacecraft == spacecraft,
                                TelemetryData.channel == channel,
                                TelemetryData.is_anomaly == True,
                                TelemetryData.timestamp >= (end_time - timedelta(hours=1))
                            )
                        ).count()

                        if recent_anomalies > 0:
                            equipment_health[equipment_id] = "anomaly"
                        else:
                            equipment_health[equipment_id] = "normal"

                summary = DataSummary(
                    total_records=total_records,
                    date_range=(date_stats[0] or start_time, date_stats[1] or end_time),
                    spacecraft_counts=spacecraft_counts,
                    channel_counts=channel_counts,
                    anomaly_count=anomaly_count,
                    anomaly_percentage=anomaly_percentage,
                    equipment_health=equipment_health
                )

                # Cache results with shorter TTL for summary
                self._set_cache(cache_key, summary, duration=30)

                logger.info(f"Generated data summary: {total_records} records, {anomaly_count} anomalies")
                return summary

        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return DataSummary(
                total_records=0,
                date_range=(datetime.now(), datetime.now()),
                spacecraft_counts={},
                channel_counts={},
                anomaly_count=0,
                anomaly_percentage=0.0,
                equipment_health={}
            )

    def get_time_series_data(self, equipment_id: str, query: DataQuery) -> pd.DataFrame:
        """Get time series data for plotting

        Args:
            equipment_id: Equipment identifier (e.g., 'SMAP_A-1')
            query: DataQuery object with filter parameters

        Returns:
            Pandas DataFrame with time series data
        """
        try:
            # Parse equipment ID
            if '_' in equipment_id:
                spacecraft, channel = equipment_id.split('_', 1)
            else:
                # Fallback - try to determine from equipment mapper
                equipment_info = equipment_mapper.get_equipment_info(equipment_id)
                if equipment_info:
                    spacecraft = equipment_info.get('spacecraft', 'UNKNOWN')
                    channel = equipment_info.get('channel', equipment_id)
                else:
                    spacecraft, channel = 'UNKNOWN', equipment_id

            with self.get_session() as session:
                start_time, end_time = self._get_time_range_bounds(query)

                # Query telemetry data
                db_query = session.query(TelemetryData).filter(
                    and_(
                        TelemetryData.spacecraft == spacecraft,
                        TelemetryData.channel == channel,
                        TelemetryData.timestamp.between(start_time, end_time)
                    )
                ).order_by(TelemetryData.timestamp)

                # Apply limit if specified
                if query.limit:
                    db_query = db_query.limit(query.limit)

                results = db_query.all()

                # Convert to DataFrame
                data_rows = []
                for record in results:
                    # Expand data array into individual points
                    values = record.data if isinstance(record.data, list) else []

                    if values:
                        # Create timestamps for each value in the array
                        base_timestamp = record.timestamp
                        for i, value in enumerate(values):
                            data_rows.append({
                                'timestamp': base_timestamp + timedelta(seconds=i),
                                'value': value,
                                'mean_value': record.mean_value,
                                'std_value': record.std_value,
                                'min_value': record.min_value,
                                'max_value': record.max_value,
                                'is_anomaly': record.is_anomaly,
                                'anomaly_score': record.anomaly_score,
                                'sequence_id': record.sequence_id
                            })
                    else:
                        # Fallback to aggregated values
                        data_rows.append({
                            'timestamp': record.timestamp,
                            'value': record.mean_value,
                            'mean_value': record.mean_value,
                            'std_value': record.std_value,
                            'min_value': record.min_value,
                            'max_value': record.max_value,
                            'is_anomaly': record.is_anomaly,
                            'anomaly_score': record.anomaly_score,
                            'sequence_id': record.sequence_id
                        })

                df = pd.DataFrame(data_rows)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')

                logger.info(f"Retrieved time series data for {equipment_id}: {len(df)} points")
                return df

        except Exception as e:
            logger.error(f"Failed to get time series data for {equipment_id}: {e}")
            return pd.DataFrame()

    def get_available_equipment(self) -> Dict[str, List[str]]:
        """Get list of available equipment IDs by spacecraft

        Returns:
            Dictionary with spacecraft as keys and list of equipment IDs as values
        """
        cache_key = "available_equipment"

        # Check cache first
        cached_result = self._get_cached(cache_key)
        if cached_result:
            return cached_result

        try:
            with self.get_session() as session:
                # Get unique spacecraft-channel combinations
                results = session.query(
                    TelemetryData.spacecraft,
                    TelemetryData.channel
                ).distinct().all()

                equipment_by_spacecraft = {}
                for spacecraft, channel in results:
                    if spacecraft not in equipment_by_spacecraft:
                        equipment_by_spacecraft[spacecraft] = []

                    equipment_id = f"{spacecraft}_{channel}"
                    equipment_by_spacecraft[spacecraft].append(equipment_id)

                # Sort lists for consistent ordering
                for spacecraft in equipment_by_spacecraft:
                    equipment_by_spacecraft[spacecraft].sort()

                # Cache with longer TTL since this changes infrequently
                self._set_cache(cache_key, equipment_by_spacecraft, duration=300)

                return equipment_by_spacecraft

        except Exception as e:
            logger.error(f"Failed to get available equipment: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear all cached data"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_ttl.clear()
        logger.info("Data access cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                'cache_size': len(self._cache),
                'cache_keys': list(self._cache.keys()),
                'cache_memory_usage': sum(len(str(v)) for v in self._cache.values())
            }


# Global instance for dashboard integration
unified_data_access = UnifiedDataAccess()