"""
Database Manager Module for IoT Telemetry Data
Handles PostgreSQL/TimescaleDB and SQLite for time-series data storage
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager
from collections import defaultdict
import threading
import time

# Database imports
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Boolean, JSON, Text, Index, UniqueConstraint, ForeignKey,
    and_, or_, func, text, MetaData, Table, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship, scoped_session
from sqlalchemy.pool import QueuePool, NullPool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.dialects.postgresql import insert as pg_insert
import asyncio
import asyncpg
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, DatabaseConfig
from src.data_ingestion.stream_simulator import StreamEvent

# Setup logging
logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


# Database Models
class TelemetryData(Base):
    """Main telemetry data table"""
    __tablename__ = 'telemetry_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    spacecraft = Column(String(50), nullable=False, index=True)
    channel = Column(String(100), nullable=False, index=True)
    sequence_id = Column(Integer)
    data = Column(JSON)  # Store array as JSON
    data_shape = Column(String(50))
    mean_value = Column(Float)
    std_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    is_anomaly = Column(Boolean, default=False, index=True)
    anomaly_score = Column(Float)
    event_metadata = Column('metadata', JSON)  # Use column name mapping
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_spacecraft_channel_time', 'spacecraft', 'channel', 'timestamp'),
        Index('idx_anomaly_time', 'is_anomaly', 'timestamp'),
        UniqueConstraint('spacecraft', 'channel', 'timestamp', 'sequence_id', 
                        name='unique_telemetry_record'),
    )


class AnomalyDetection(Base):
    """Anomaly detection results table"""
    __tablename__ = 'anomaly_detections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    telemetry_id = Column(Integer, ForeignKey('telemetry_data.id'))
    timestamp = Column(DateTime, nullable=False, index=True)
    spacecraft = Column(String(50), nullable=False)
    channel = Column(String(100), nullable=False)
    detector_type = Column(String(50))  # LSTM, Autoencoder, VAE
    anomaly_score = Column(Float, nullable=False)
    threshold = Column(Float)
    is_confirmed = Column(Boolean, default=False)
    severity = Column(String(20))  # low, medium, high, critical
    description = Column(Text)
    raw_scores = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    telemetry = relationship("TelemetryData", backref="anomalies")
    
    __table_args__ = (
        Index('idx_anomaly_spacecraft_time', 'spacecraft', 'timestamp'),
        Index('idx_anomaly_severity', 'severity', 'timestamp'),
    )


class ForecastResult(Base):
    """Forecasting results table"""
    __tablename__ = 'forecast_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    spacecraft = Column(String(50), nullable=False)
    channel = Column(String(100), nullable=False)
    model_type = Column(String(50))  # Transformer, LSTM
    forecast_horizon = Column(Integer)
    predicted_values = Column(JSON)
    confidence_lower = Column(JSON)
    confidence_upper = Column(JSON)
    actual_values = Column(JSON)
    mae = Column(Float)
    rmse = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_forecast_spacecraft_channel', 'spacecraft', 'channel', 'timestamp'),
    )


class MaintenanceSchedule(Base):
    """Maintenance schedule table"""
    __tablename__ = 'maintenance_schedules'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    anomaly_id = Column(Integer, ForeignKey('anomaly_detections.id'))
    spacecraft = Column(String(50), nullable=False)
    channel = Column(String(100), nullable=False)
    scheduled_time = Column(DateTime, nullable=False, index=True)
    duration_hours = Column(Float)
    priority = Column(Integer)  # 1-1000, higher is more urgent
    status = Column(String(20), default='scheduled')  # scheduled, in_progress, completed, cancelled
    technician_id = Column(String(50))
    cost_estimate = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    anomaly = relationship("AnomalyDetection", backref="maintenance_tasks")
    
    __table_args__ = (
        Index('idx_maintenance_status_time', 'status', 'scheduled_time'),
    )


@dataclass
class BulkInsertConfig:
    """Configuration for bulk insert operations"""
    batch_size: int = 1000
    max_workers: int = 16
    timeout: int = 30
    retry_attempts: int = 3
    use_prepared_statements: bool = True


@dataclass
class PerformanceMetrics:
    """Database performance metrics"""
    queries_executed: int = 0
    avg_query_time: float = 0.0
    connection_pool_size: int = 0
    active_connections: int = 0
    bulk_operations: int = 0
    cache_hits: int = 0
    total_sensors_processed: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class OptimizedDatabaseManager:
    """
    High-performance database manager optimized for 80-sensor concurrent operations
    Features: Connection pooling, bulk operations, prepared statements, async support
    """

    def __init__(self, config: DatabaseConfig = None):
        """Initialize optimized database manager"""
        self.config = config or DatabaseConfig()
        self.bulk_config = BulkInsertConfig()
        self.performance_metrics = PerformanceMetrics()

        # Connection pools
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None

        # Thread pool for bulk operations
        self.thread_pool = ThreadPoolExecutor(max_workers=self.bulk_config.max_workers)

        # Prepared statements cache
        self.prepared_statements = {}
        self.statement_lock = threading.Lock()

        # Performance tracking
        self.query_times = deque(maxlen=1000)
        self.metrics_lock = threading.Lock()

        # Initialize database connections
        self._initialize_connections()

        logger.info("Optimized Database Manager initialized with enhanced connection pooling")

    def _initialize_connections(self):
        """Initialize optimized database connections"""
        try:
            # Enhanced connection pool configuration for 80 sensors
            pool_config = {
                'poolclass': QueuePool,
                'pool_size': 20,  # Base connections
                'max_overflow': 40,  # Additional connections under load
                'pool_pre_ping': True,  # Validate connections
                'pool_recycle': 3600,  # Recycle connections every hour
                'connect_args': {}
            }

            if self.config.type == "postgresql":
                # PostgreSQL optimizations
                pool_config['connect_args'].update({
                    'application_name': 'IoT_Sensor_System',
                    'connect_timeout': 10,
                    'server_side_cursors': True
                })

                # Create optimized engine for PostgreSQL
                self.engine = create_engine(
                    self.config.connection_string,
                    **pool_config,
                    echo=False
                )

            elif self.config.type == "sqlite":
                # SQLite optimizations for concurrent access
                pool_config.update({
                    'poolclass': StaticPool,
                    'connect_args': {
                        'check_same_thread': False,
                        'timeout': 20,
                        'isolation_level': None  # autocommit mode
                    }
                })

                self.engine = create_engine(
                    self.config.connection_string,
                    **pool_config,
                    echo=False
                )

            # Create session factory with optimizations
            self.session_factory = scoped_session(
                sessionmaker(
                    bind=self.engine,
                    autoflush=False,  # Manual flush for better control
                    autocommit=False,
                    expire_on_commit=False  # Keep objects accessible after commit
                )
            )

            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)

            # Pre-warm the connection pool
            self._warm_connection_pool()

            logger.info(f"Database connections initialized: {self.config.type}")

        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise

    def _warm_connection_pool(self):
        """Pre-warm connection pool for faster initial queries"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Connection pool warmed successfully")
        except Exception as e:
            logger.warning(f"Connection pool warm-up failed: {e}")

    def bulk_insert_telemetry(self, telemetry_records: List[Dict[str, Any]]) -> bool:
        """
        High-performance bulk insert for telemetry data
        Optimized for 80-sensor concurrent operations
        """
        if not telemetry_records:
            return True

        start_time = time.time()

        try:
            # Split into batches for optimal performance
            batches = [
                telemetry_records[i:i + self.bulk_config.batch_size]
                for i in range(0, len(telemetry_records), self.bulk_config.batch_size)
            ]

            successful_batches = 0

            # Process batches in parallel
            if len(batches) > 1:
                successful_batches = self._parallel_bulk_insert(batches)
            else:
                successful_batches = 1 if self._insert_single_batch(batches[0]) else 0

            # Update performance metrics
            self._update_performance_metrics(
                len(telemetry_records),
                time.time() - start_time,
                len(batches)
            )

            success_rate = successful_batches / len(batches) if batches else 1
            logger.info(f"Bulk insert completed: {len(telemetry_records)} records, "
                       f"{successful_batches}/{len(batches)} batches successful")

            return success_rate > 0.8  # Consider successful if 80%+ batches succeeded

        except Exception as e:
            logger.error(f"Bulk telemetry insert failed: {e}")
            return False

    def _parallel_bulk_insert(self, batches: List[List[Dict]]) -> int:
        """Execute bulk inserts in parallel"""
        successful_batches = 0

        # Submit all batches to thread pool
        future_to_batch = {
            self.thread_pool.submit(self._insert_single_batch, batch): i
            for i, batch in enumerate(batches)
        }

        # Collect results
        for future in as_completed(future_to_batch, timeout=self.bulk_config.timeout):
            try:
                if future.result():
                    successful_batches += 1
            except Exception as e:
                batch_idx = future_to_batch[future]
                logger.error(f"Batch {batch_idx} insert failed: {e}")

        return successful_batches

    def _insert_single_batch(self, batch: List[Dict[str, Any]]) -> bool:
        """Insert a single batch of telemetry records"""
        session = None
        try:
            session = self.session_factory()

            if self.config.type == "postgresql":
                # Use PostgreSQL UPSERT for better performance
                stmt = pg_insert(TelemetryData).values(batch)
                stmt = stmt.on_conflict_do_update(
                    constraint='unique_telemetry_record',
                    set_=dict(
                        anomaly_score=stmt.excluded.anomaly_score,
                        is_anomaly=stmt.excluded.is_anomaly
                    )
                )
                session.execute(stmt)
            else:
                # Standard bulk insert for SQLite
                session.bulk_insert_mappings(TelemetryData, batch)

            session.commit()
            return True

        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Single batch insert failed: {e}")
            return False
        finally:
            if session:
                session.close()

    def bulk_query_sensor_data(self, sensor_ids: List[str],
                              time_range: Tuple[datetime, datetime],
                              limit_per_sensor: int = 1000) -> Dict[str, List[Dict]]:
        """
        Optimized bulk query for multiple sensors
        Returns data for all sensors in a single database round-trip
        """
        start_time = time.time()

        try:
            session = self.session_factory()

            # Single query for all sensors with time range
            query = session.query(TelemetryData).filter(
                and_(
                    TelemetryData.channel.in_(sensor_ids),
                    TelemetryData.timestamp.between(*time_range)
                )
            ).order_by(
                TelemetryData.channel,
                TelemetryData.timestamp.desc()
            )

            # Execute query
            results = query.all()

            # Group results by sensor
            grouped_results = defaultdict(list)
            sensor_counts = defaultdict(int)

            for record in results:
                sensor_id = record.channel
                if sensor_counts[sensor_id] < limit_per_sensor:
                    grouped_results[sensor_id].append({
                        'timestamp': record.timestamp,
                        'data': record.data,
                        'mean_value': record.mean_value,
                        'anomaly_score': record.anomaly_score,
                        'is_anomaly': record.is_anomaly
                    })
                    sensor_counts[sensor_id] += 1

            # Update metrics
            query_time = time.time() - start_time
            self._track_query_performance(query_time, len(sensor_ids))

            logger.debug(f"Bulk sensor query completed: {len(sensor_ids)} sensors, "
                        f"{sum(len(data) for data in grouped_results.values())} records, "
                        f"{query_time:.3f}s")

            return dict(grouped_results)

        except Exception as e:
            logger.error(f"Bulk sensor query failed: {e}")
            return {}
        finally:
            session.close()

    def get_sensor_aggregates(self, sensor_ids: List[str],
                             time_range: Tuple[datetime, datetime],
                             aggregation_interval: str = "1 hour") -> Dict[str, Dict]:
        """
        Get aggregated sensor data for dashboard performance
        Reduces data transfer for time-series charts
        """
        try:
            session = self.session_factory()

            # Determine aggregation function based on database type
            if self.config.type == "postgresql":
                # PostgreSQL with time bucketing
                time_bucket = func.date_trunc('hour', TelemetryData.timestamp)
            else:
                # SQLite approximation
                time_bucket = func.strftime('%Y-%m-%d %H:00:00', TelemetryData.timestamp)

            # Aggregation query
            query = session.query(
                TelemetryData.channel,
                time_bucket.label('time_bucket'),
                func.avg(TelemetryData.mean_value).label('avg_value'),
                func.max(TelemetryData.mean_value).label('max_value'),
                func.min(TelemetryData.mean_value).label('min_value'),
                func.count().label('data_points'),
                func.avg(TelemetryData.anomaly_score).label('avg_anomaly_score'),
                func.sum(func.cast(TelemetryData.is_anomaly, Integer)).label('anomaly_count')
            ).filter(
                and_(
                    TelemetryData.channel.in_(sensor_ids),
                    TelemetryData.timestamp.between(*time_range)
                )
            ).group_by(
                TelemetryData.channel,
                time_bucket
            ).order_by(
                TelemetryData.channel,
                time_bucket
            )

            results = query.all()

            # Group by sensor
            aggregated_data = defaultdict(list)
            for row in results:
                aggregated_data[row.channel].append({
                    'timestamp': row.time_bucket,
                    'avg_value': float(row.avg_value or 0),
                    'max_value': float(row.max_value or 0),
                    'min_value': float(row.min_value or 0),
                    'data_points': row.data_points,
                    'avg_anomaly_score': float(row.avg_anomaly_score or 0),
                    'anomaly_count': row.anomaly_count or 0
                })

            return dict(aggregated_data)

        except Exception as e:
            logger.error(f"Sensor aggregation query failed: {e}")
            return {}
        finally:
            session.close()

    def _update_performance_metrics(self, records_count: int, execution_time: float, batch_count: int):
        """Update database performance metrics"""
        with self.metrics_lock:
            self.performance_metrics.bulk_operations += 1
            self.performance_metrics.total_sensors_processed += records_count

            # Update average query time (exponential moving average)
            alpha = 0.1
            self.performance_metrics.avg_query_time = (
                alpha * execution_time +
                (1 - alpha) * self.performance_metrics.avg_query_time
            )

            self.performance_metrics.last_updated = datetime.now()

    def _track_query_performance(self, query_time: float, sensor_count: int):
        """Track query performance for optimization"""
        self.query_times.append(query_time)

        with self.metrics_lock:
            self.performance_metrics.queries_executed += 1
            self.performance_metrics.total_sensors_processed += sensor_count

    def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status"""
        try:
            pool = self.engine.pool
            return {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
        except Exception as e:
            logger.error(f"Failed to get connection pool status: {e}")
            return {}

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        # Update connection pool info
        pool_status = self.get_connection_pool_status()
        self.performance_metrics.connection_pool_size = pool_status.get('pool_size', 0)
        self.performance_metrics.active_connections = pool_status.get('checked_out', 0)

        return self.performance_metrics

    def optimize_for_sensor_count(self, sensor_count: int):
        """Dynamically optimize database settings based on sensor count"""
        if sensor_count <= 20:
            self.bulk_config.batch_size = 500
            self.bulk_config.max_workers = 4
        elif sensor_count <= 50:
            self.bulk_config.batch_size = 750
            self.bulk_config.max_workers = 8
        else:  # 80+ sensors
            self.bulk_config.batch_size = 1000
            self.bulk_config.max_workers = 16

        logger.info(f"Database optimized for {sensor_count} sensors: "
                   f"batch_size={self.bulk_config.batch_size}, "
                   f"workers={self.bulk_config.max_workers}")

    def close(self):
        """Close database connections and thread pool"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            if self.session_factory:
                self.session_factory.remove()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connections closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


class SystemMetrics(Base):
    """System performance metrics table"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    metric_type = Column(String(50))  # throughput, latency, error_rate
    metric_value = Column(Float)
    component = Column(String(50))  # ingestion, detection, forecasting
    event_metadata = Column('metadata', JSON)  # Use column name mapping
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class DatabaseStats:
    """Database statistics container"""
    total_records: int = 0
    records_by_spacecraft: Dict[str, int] = field(default_factory=dict)
    records_by_channel: Dict[str, int] = field(default_factory=dict)
    anomaly_count: int = 0
    forecast_count: int = 0
    maintenance_count: int = 0
    database_size_mb: float = 0.0
    oldest_record: Optional[datetime] = None
    newest_record: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_records': self.total_records,
            'records_by_spacecraft': self.records_by_spacecraft,
            'records_by_channel': dict(list(self.records_by_channel.items())[:10]),  # Top 10
            'anomaly_count': self.anomaly_count,
            'forecast_count': self.forecast_count,
            'maintenance_count': self.maintenance_count,
            'database_size_mb': self.database_size_mb,
            'oldest_record': self.oldest_record.isoformat() if self.oldest_record else None,
            'newest_record': self.newest_record.isoformat() if self.newest_record else None,
            'data_range_days': (self.newest_record - self.oldest_record).days if self.oldest_record and self.newest_record else 0
        }


class DatabaseManager:
    """
    Main database manager for IoT telemetry system
    Supports PostgreSQL/TimescaleDB and SQLite
    """
    
    def __init__(self, 
                 config: Optional[DatabaseConfig] = None,
                 echo: bool = False,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: float = 30.0,
                 create_tables: bool = True):
        """
        Initialize database manager
        
        Args:
            config: Database configuration object
            echo: Whether to echo SQL statements
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            create_tables: Whether to create tables on init
        """
        # Load configuration
        self.config = config or settings.get_database_config()
        
        # Connection parameters
        self.echo = echo
        self.pool_size = pool_size if self.config.type != 'sqlite' else 1
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        
        # Create engine
        self.engine = self._create_engine()
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.SessionFactory)
        
        # Create tables if needed
        if create_tables:
            self.create_tables()
        
        # Setup TimescaleDB if enabled
        if self.config.type == 'postgresql' and self.config.timescale_enabled:
            self._setup_timescaledb()
        
        # Statistics cache
        self._stats_cache = None
        self._stats_cache_time = None
        self._stats_cache_ttl = 60  # seconds
        
        logger.info(f"DatabaseManager initialized with {self.config.type} backend")
    
    def _create_engine(self):
        """Create SQLAlchemy engine"""
        connection_string = self.config.connection_string
        
        if self.config.type == 'postgresql':
            # PostgreSQL specific settings
            engine = create_engine(
                connection_string,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_pre_ping=True,  # Verify connections
                connect_args={
                    'connect_timeout': 10,
                    'options': '-c timezone=utc'
                }
            )
        elif self.config.type == 'sqlite':
            # SQLite specific settings
            engine = create_engine(
                connection_string,
                echo=self.echo,
                poolclass=NullPool,  # No pooling for SQLite
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30
                }
            )
        else:
            raise ValueError(f"Unsupported database type: {self.config.type}")
        
        return engine
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise
    
    def _setup_timescaledb(self):
        """Setup TimescaleDB specific features"""
        try:
            with self.engine.connect() as conn:
                # Create TimescaleDB extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                
                # Convert telemetry_data to hypertable
                conn.execute(text(
                    f"SELECT create_hypertable('telemetry_data', 'timestamp', "
                    f"chunk_time_interval => INTERVAL '{self.config.timescale_chunk_interval}', "
                    f"if_not_exists => TRUE);"
                ))
                
                # Add compression policy
                conn.execute(text(
                    f"ALTER TABLE telemetry_data SET ("
                    f"timescaledb.compress, "
                    f"timescaledb.compress_segmentby = 'spacecraft,channel'"
                    f");"
                ))
                
                # Create continuous aggregates for common queries
                conn.execute(text("""
                    CREATE MATERIALIZED VIEW IF NOT EXISTS telemetry_hourly
                    WITH (timescaledb.continuous) AS
                    SELECT 
                        time_bucket('1 hour', timestamp) AS bucket,
                        spacecraft,
                        channel,
                        avg(mean_value) as avg_value,
                        max(max_value) as max_value,
                        min(min_value) as min_value,
                        count(*) as sample_count,
                        sum(CASE WHEN is_anomaly THEN 1 ELSE 0 END) as anomaly_count
                    FROM telemetry_data
                    GROUP BY bucket, spacecraft, channel;
                """))
                
                logger.info("TimescaleDB setup completed")
                
        except Exception as e:
            logger.warning(f"TimescaleDB setup failed: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Get database session with automatic cleanup
        
        Yields:
            Database session
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def insert_telemetry_data(self, 
                             event: StreamEvent,
                             commit: bool = True) -> Optional[int]:
        """
        Insert telemetry data from stream event
        
        Args:
            event: Stream event to insert
            commit: Whether to commit immediately
            
        Returns:
            Record ID if successful
        """
        try:
            with self.get_session() as session:
                # Calculate statistics for the data
                data_array = np.array(event.data)
                
                telemetry = TelemetryData(
                    timestamp=event.timestamp,
                    spacecraft=event.spacecraft,
                    channel=event.channel,
                    sequence_id=event.sequence_id,
                    data=event.data.tolist() if isinstance(event.data, np.ndarray) else event.data,
                    data_shape=str(data_array.shape),
                    mean_value=float(np.mean(data_array)),
                    std_value=float(np.std(data_array)),
                    min_value=float(np.min(data_array)),
                    max_value=float(np.max(data_array)),
                    is_anomaly=event.is_anomaly,
                    anomaly_score=event.metadata.get('anomaly_score'),
                    event_metadata=event.metadata
                )
                
                session.add(telemetry)
                
                if commit:
                    session.commit()
                    
                return telemetry.id
                
        except IntegrityError:
            logger.debug("Duplicate record, skipping")
            return None
        except Exception as e:
            logger.error(f"Error inserting telemetry data: {e}")
            return None
    
    def insert_telemetry_batch(self, 
                              events: List[StreamEvent],
                              batch_size: int = 1000) -> int:
        """
        Insert batch of telemetry data
        
        Args:
            events: List of stream events
            batch_size: Batch size for insertion
            
        Returns:
            Number of records inserted
        """
        inserted_count = 0
        
        try:
            with self.get_session() as session:
                batch = []
                
                for event in events:
                    data_array = np.array(event.data)
                    
                    telemetry = TelemetryData(
                        timestamp=event.timestamp,
                        spacecraft=event.spacecraft,
                        channel=event.channel,
                        sequence_id=event.sequence_id,
                        data=event.data.tolist() if isinstance(event.data, np.ndarray) else event.data,
                        data_shape=str(data_array.shape),
                        mean_value=float(np.mean(data_array)),
                        std_value=float(np.std(data_array)),
                        min_value=float(np.min(data_array)),
                        max_value=float(np.max(data_array)),
                        is_anomaly=event.is_anomaly,
                        anomaly_score=event.metadata.get('anomaly_score'),
                        event_metadata=event.metadata
                    )
                    
                    batch.append(telemetry)
                    
                    if len(batch) >= batch_size:
                        session.bulk_save_objects(batch)
                        session.commit()
                        inserted_count += len(batch)
                        batch = []
                
                # Insert remaining
                if batch:
                    session.bulk_save_objects(batch)
                    session.commit()
                    inserted_count += len(batch)
                
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
        
        return inserted_count
    
    def insert_anomaly(self,
                      telemetry_id: Optional[int],
                      timestamp: datetime,
                      spacecraft: str,
                      channel: str,
                      detector_type: str,
                      anomaly_score: float,
                      threshold: float = None,
                      severity: str = 'medium',
                      description: str = None,
                      raw_scores: Dict = None) -> Optional[int]:
        """
        Insert anomaly detection result
        
        Returns:
            Anomaly record ID if successful
        """
        try:
            with self.get_session() as session:
                anomaly = AnomalyDetection(
                    telemetry_id=telemetry_id,
                    timestamp=timestamp,
                    spacecraft=spacecraft,
                    channel=channel,
                    detector_type=detector_type,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    severity=severity,
                    description=description,
                    raw_scores=raw_scores
                )
                
                session.add(anomaly)
                session.commit()
                
                return anomaly.id
                
        except Exception as e:
            logger.error(f"Error inserting anomaly: {e}")
            return None
    
    def insert_forecast(self,
                       timestamp: datetime,
                       spacecraft: str,
                       channel: str,
                       model_type: str,
                       forecast_horizon: int,
                       predicted_values: List[float],
                       confidence_lower: List[float] = None,
                       confidence_upper: List[float] = None,
                       actual_values: List[float] = None) -> Optional[int]:
        """
        Insert forecast result
        
        Returns:
            Forecast record ID if successful
        """
        try:
            with self.get_session() as session:
                # Calculate error metrics if actual values available
                mae = None
                rmse = None
                if actual_values and predicted_values:
                    errors = np.array(actual_values) - np.array(predicted_values)
                    mae = float(np.mean(np.abs(errors)))
                    rmse = float(np.sqrt(np.mean(errors ** 2)))
                
                forecast = ForecastResult(
                    timestamp=timestamp,
                    spacecraft=spacecraft,
                    channel=channel,
                    model_type=model_type,
                    forecast_horizon=forecast_horizon,
                    predicted_values=predicted_values,
                    confidence_lower=confidence_lower,
                    confidence_upper=confidence_upper,
                    actual_values=actual_values,
                    mae=mae,
                    rmse=rmse
                )
                
                session.add(forecast)
                session.commit()
                
                return forecast.id
                
        except Exception as e:
            logger.error(f"Error inserting forecast: {e}")
            return None
    
    def insert_maintenance_schedule(self,
                                  anomaly_id: Optional[int],
                                  spacecraft: str,
                                  channel: str,
                                  scheduled_time: datetime,
                                  duration_hours: float,
                                  priority: int,
                                  technician_id: str = None,
                                  cost_estimate: float = None,
                                  notes: str = None) -> Optional[int]:
        """
        Insert maintenance schedule entry
        
        Returns:
            Schedule record ID if successful
        """
        try:
            with self.get_session() as session:
                schedule = MaintenanceSchedule(
                    anomaly_id=anomaly_id,
                    spacecraft=spacecraft,
                    channel=channel,
                    scheduled_time=scheduled_time,
                    duration_hours=duration_hours,
                    priority=priority,
                    technician_id=technician_id,
                    cost_estimate=cost_estimate,
                    notes=notes
                )
                
                session.add(schedule)
                session.commit()
                
                return schedule.id
                
        except Exception as e:
            logger.error(f"Error inserting maintenance schedule: {e}")
            return None
    
    def get_telemetry_data(self,
                          spacecraft: Optional[str] = None,
                          channel: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          anomalies_only: bool = False,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Query telemetry data
        
        Args:
            spacecraft: Filter by spacecraft
            channel: Filter by channel
            start_time: Start time filter
            end_time: End time filter
            anomalies_only: Only return anomalies
            limit: Maximum records to return
            
        Returns:
            DataFrame with telemetry data
        """
        try:
            with self.get_session() as session:
                query = session.query(TelemetryData)
                
                if spacecraft:
                    query = query.filter(TelemetryData.spacecraft == spacecraft)
                if channel:
                    query = query.filter(TelemetryData.channel == channel)
                if start_time:
                    query = query.filter(TelemetryData.timestamp >= start_time)
                if end_time:
                    query = query.filter(TelemetryData.timestamp <= end_time)
                if anomalies_only:
                    query = query.filter(TelemetryData.is_anomaly == True)
                
                query = query.order_by(TelemetryData.timestamp.desc()).limit(limit)
                
                # Convert to DataFrame
                records = []
                for row in query:
                    record = {
                        'id': row.id,
                        'timestamp': row.timestamp,
                        'spacecraft': row.spacecraft,
                        'channel': row.channel,
                        'mean_value': row.mean_value,
                        'std_value': row.std_value,
                        'min_value': row.min_value,
                        'max_value': row.max_value,
                        'is_anomaly': row.is_anomaly,
                        'anomaly_score': row.anomaly_score
                    }
                    records.append(record)
                
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.error(f"Error querying telemetry data: {e}")
            return pd.DataFrame()
    
    def get_recent_anomalies(self, 
                           hours: int = 24,
                           min_severity: str = 'low') -> pd.DataFrame:
        """
        Get recent anomalies
        
        Args:
            hours: Number of hours to look back
            min_severity: Minimum severity level
            
        Returns:
            DataFrame with anomaly data
        """
        severity_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        min_severity_value = severity_order.get(min_severity, 1)
        
        try:
            with self.get_session() as session:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                query = session.query(AnomalyDetection).filter(
                    AnomalyDetection.timestamp >= cutoff_time
                )
                
                # Filter by severity
                if min_severity != 'low':
                    severity_filters = [
                        AnomalyDetection.severity == sev
                        for sev, val in severity_order.items()
                        if val >= min_severity_value
                    ]
                    query = query.filter(or_(*severity_filters))
                
                query = query.order_by(AnomalyDetection.timestamp.desc())
                
                # Convert to DataFrame
                records = []
                for row in query:
                    record = {
                        'id': row.id,
                        'timestamp': row.timestamp,
                        'spacecraft': row.spacecraft,
                        'channel': row.channel,
                        'detector_type': row.detector_type,
                        'anomaly_score': row.anomaly_score,
                        'threshold': row.threshold,
                        'severity': row.severity,
                        'description': row.description
                    }
                    records.append(record)
                
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.error(f"Error querying anomalies: {e}")
            return pd.DataFrame()
    
    def get_pending_maintenance(self) -> pd.DataFrame:
        """
        Get pending maintenance tasks
        
        Returns:
            DataFrame with maintenance schedule
        """
        try:
            with self.get_session() as session:
                query = session.query(MaintenanceSchedule).filter(
                    MaintenanceSchedule.status.in_(['scheduled', 'in_progress'])
                ).order_by(MaintenanceSchedule.priority.desc())
                
                # Convert to DataFrame
                records = []
                for row in query:
                    record = {
                        'id': row.id,
                        'spacecraft': row.spacecraft,
                        'channel': row.channel,
                        'scheduled_time': row.scheduled_time,
                        'duration_hours': row.duration_hours,
                        'priority': row.priority,
                        'status': row.status,
                        'technician_id': row.technician_id,
                        'cost_estimate': row.cost_estimate
                    }
                    records.append(record)
                
                return pd.DataFrame(records)
                
        except Exception as e:
            logger.error(f"Error querying maintenance schedule: {e}")
            return pd.DataFrame()
    
    def update_maintenance_status(self, 
                                 schedule_id: int,
                                 status: str,
                                 notes: str = None) -> bool:
        """
        Update maintenance task status
        
        Args:
            schedule_id: Schedule record ID
            status: New status
            notes: Additional notes
            
        Returns:
            True if successful
        """
        try:
            with self.get_session() as session:
                schedule = session.query(MaintenanceSchedule).get(schedule_id)
                if schedule:
                    schedule.status = status
                    if notes:
                        schedule.notes = (schedule.notes or '') + f'\n{datetime.utcnow()}: {notes}'
                    schedule.updated_at = datetime.utcnow()
                    session.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error updating maintenance status: {e}")
            return False
    
    def get_statistics(self, force_refresh: bool = False) -> DatabaseStats:
        """
        Get database statistics
        
        Args:
            force_refresh: Force refresh of cached stats
            
        Returns:
            Database statistics
        """
        # Check cache
        if not force_refresh and self._stats_cache and self._stats_cache_time:
            if (datetime.now() - self._stats_cache_time).seconds < self._stats_cache_ttl:
                return self._stats_cache
        
        stats = DatabaseStats()
        
        try:
            with self.get_session() as session:
                # Total records
                stats.total_records = session.query(TelemetryData).count()
                
                # Records by spacecraft
                spacecraft_counts = session.query(
                    TelemetryData.spacecraft,
                    func.count(TelemetryData.id)
                ).group_by(TelemetryData.spacecraft).all()
                
                stats.records_by_spacecraft = dict(spacecraft_counts)
                
                # Records by channel (top 20)
                channel_counts = session.query(
                    TelemetryData.channel,
                    func.count(TelemetryData.id)
                ).group_by(TelemetryData.channel).order_by(
                    func.count(TelemetryData.id).desc()
                ).limit(20).all()
                
                stats.records_by_channel = dict(channel_counts)
                
                # Anomaly count
                stats.anomaly_count = session.query(AnomalyDetection).count()
                
                # Forecast count
                stats.forecast_count = session.query(ForecastResult).count()
                
                # Maintenance count
                stats.maintenance_count = session.query(MaintenanceSchedule).count()
                
                # Date range
                date_range = session.query(
                    func.min(TelemetryData.timestamp),
                    func.max(TelemetryData.timestamp)
                ).first()
                
                if date_range:
                    stats.oldest_record = date_range[0]
                    stats.newest_record = date_range[1]
                
                # Database size (PostgreSQL specific)
                if self.config.type == 'postgresql':
                    size_query = text(
                        "SELECT pg_database_size(current_database()) / 1024.0 / 1024.0 as size_mb"
                    )
                    result = session.execute(size_query).first()
                    if result:
                        stats.database_size_mb = float(result[0])
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
        
        # Update cache
        self._stats_cache = stats
        self._stats_cache_time = datetime.now()
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old data
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of records deleted
        """
        deleted_count = 0
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        try:
            with self.get_session() as session:
                # Delete old telemetry data
                deleted = session.query(TelemetryData).filter(
                    TelemetryData.timestamp < cutoff_date
                ).delete()
                
                deleted_count += deleted
                
                # Delete old anomalies
                deleted = session.query(AnomalyDetection).filter(
                    AnomalyDetection.timestamp < cutoff_date
                ).delete()
                
                deleted_count += deleted
                
                # Delete old forecasts
                deleted = session.query(ForecastResult).filter(
                    ForecastResult.timestamp < cutoff_date
                ).delete()
                
                deleted_count += deleted
                
                session.commit()
                
                logger.info(f"Cleaned up {deleted_count} old records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
        
        return deleted_count
    
    def export_data(self, 
                   output_file: Path,
                   table: str = 'telemetry_data',
                   format: str = 'csv') -> bool:
        """
        Export data to file
        
        Args:
            output_file: Output file path
            table: Table to export
            format: Export format (csv, json)
            
        Returns:
            True if successful
        """
        try:
            with self.get_session() as session:
                if table == 'telemetry_data':
                    df = self.get_telemetry_data(limit=None)
                elif table == 'anomalies':
                    df = self.get_recent_anomalies(hours=24*365)
                elif table == 'maintenance':
                    df = self.get_pending_maintenance()
                else:
                    logger.error(f"Unknown table: {table}")
                    return False
                
                if format == 'csv':
                    df.to_csv(output_file, index=False)
                elif format == 'json':
                    df.to_json(output_file, orient='records', date_format='iso')
                else:
                    logger.error(f"Unknown format: {format}")
                    return False
                
                logger.info(f"Exported {len(df)} records to {output_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        self.Session.remove()
        self.engine.dispose()
        logger.info("Database connections closed")


# Singleton instance
_db_manager_instance: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Get singleton database manager instance
    
    Returns:
        Database manager instance
    """
    global _db_manager_instance
    
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    
    return _db_manager_instance


if __name__ == "__main__":
    # Test database manager
    print("\n" + "="*60)
    print("Testing Database Manager")
    print("="*60)
    
    # Create database manager
    print("\nCreating database manager...")
    db = DatabaseManager(echo=False)
    
    # Create test event
    test_event = StreamEvent(
        timestamp=datetime.now(),
        spacecraft="smap",
        channel="test_channel",
        data=np.random.randn(100),
        sequence_id=1,
        is_anomaly=False,
        metadata={'test': True}
    )
    
    # Test insertion
    print("\nInserting test telemetry data...")
    record_id = db.insert_telemetry_data(test_event)
    print(f"Inserted record ID: {record_id}")
    
    # Test anomaly insertion
    print("\nInserting test anomaly...")
    anomaly_id = db.insert_anomaly(
        telemetry_id=record_id,
        timestamp=datetime.now(),
        spacecraft="smap",
        channel="test_channel",
        detector_type="LSTM",
        anomaly_score=0.85,
        threshold=0.7,
        severity="high",
        description="Test anomaly"
    )
    print(f"Inserted anomaly ID: {anomaly_id}")
    
    # Test maintenance scheduling
    print("\nInserting maintenance schedule...")
    schedule_id = db.insert_maintenance_schedule(
        anomaly_id=anomaly_id,
        spacecraft="smap",
        channel="test_channel",
        scheduled_time=datetime.now() + timedelta(days=1),
        duration_hours=2.0,
        priority=100,
        notes="Test maintenance task"
    )
    print(f"Inserted schedule ID: {schedule_id}")
    
    # Query data
    print("\nQuerying recent telemetry data...")
    df = db.get_telemetry_data(spacecraft="smap", limit=10)
    print(f"Retrieved {len(df)} records")
    if not df.empty:
        print(df.head())
    
    # Get statistics
    print("\nDatabase Statistics:")
    stats = db.get_statistics()
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")
    
    # Close connections
    db.close()
    
    print("\n" + "="*60)
    print("Database manager test complete")
    print("="*60)
