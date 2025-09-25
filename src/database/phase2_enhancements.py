"""
Phase 2 Database Enhancements
Database schema and models for advanced analytics and alert management
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class AlertActionRecord:
    """Database record for alert actions"""
    action_id: str
    alert_id: str
    action_type: str  # 'acknowledge', 'dismiss', 'escalate', 'create_work_order'
    user_id: str
    timestamp: datetime
    reason: Optional[str] = None
    notes: Optional[str] = None
    work_order_id: Optional[str] = None
    escalation_level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ThresholdHistoryRecord:
    """Database record for threshold changes"""
    change_id: str
    equipment_id: str
    equipment_type: str
    subsystem: str
    criticality: str
    old_thresholds: Dict[str, float]
    new_thresholds: Dict[str, float]
    optimization_type: str
    improvement_score: float
    confidence: float
    justification: str
    user_id: str
    timestamp: datetime
    applied: bool


@dataclass
class SubsystemAnalyticsRecord:
    """Database record for subsystem analytics"""
    analysis_id: str
    subsystem: str
    spacecraft: str  # SMAP or MSL
    analysis_type: str  # 'failure_pattern', 'health_trend', 'correlation'
    analysis_date: datetime
    pattern_data: Dict[str, Any]
    health_score: float
    trend_direction: str
    risk_factors: List[str]
    recommendations: List[str]
    confidence_score: float


class Phase2DatabaseManager:
    """
    Manages Phase 2 database enhancements including alert actions,
    threshold history, and subsystem analytics
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize Phase 2 Database Manager"""
        if db_path is None:
            # Default to project data directory
            from config.settings import get_data_path
            db_path = get_data_path('database') / 'phase2_enhancements.db'

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()
        logger.info(f"Phase 2 Database Manager initialized with DB: {self.db_path}")

    def _initialize_database(self):
        """Initialize database tables for Phase 2 enhancements"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Alert Actions Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alert_actions (
                        action_id TEXT PRIMARY KEY,
                        alert_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        reason TEXT,
                        notes TEXT,
                        work_order_id TEXT,
                        escalation_level INTEGER,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Threshold History Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS threshold_history (
                        change_id TEXT PRIMARY KEY,
                        equipment_id TEXT NOT NULL,
                        equipment_type TEXT NOT NULL,
                        subsystem TEXT NOT NULL,
                        criticality TEXT NOT NULL,
                        old_thresholds TEXT NOT NULL,
                        new_thresholds TEXT NOT NULL,
                        optimization_type TEXT NOT NULL,
                        improvement_score REAL NOT NULL,
                        confidence REAL NOT NULL,
                        justification TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        applied BOOLEAN NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Subsystem Analytics Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS subsystem_analytics (
                        analysis_id TEXT PRIMARY KEY,
                        subsystem TEXT NOT NULL,
                        spacecraft TEXT NOT NULL,
                        analysis_type TEXT NOT NULL,
                        analysis_date TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        health_score REAL NOT NULL,
                        trend_direction TEXT NOT NULL,
                        risk_factors TEXT NOT NULL,
                        recommendations TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Equipment Health Snapshots Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS equipment_health_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        equipment_id TEXT NOT NULL,
                        equipment_type TEXT NOT NULL,
                        subsystem TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        health_score REAL NOT NULL,
                        anomaly_count INTEGER NOT NULL,
                        sensor_count INTEGER NOT NULL,
                        critical_alerts INTEGER NOT NULL,
                        performance_metrics TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Detection Details History Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_details_history (
                        detection_id TEXT PRIMARY KEY,
                        equipment_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        overall_anomaly_score REAL NOT NULL,
                        overall_confidence REAL NOT NULL,
                        is_anomaly BOOLEAN NOT NULL,
                        severity_level TEXT NOT NULL,
                        sensor_details TEXT NOT NULL,
                        model_performance TEXT NOT NULL,
                        reconstruction_quality REAL NOT NULL,
                        pattern_analysis TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Create indexes for performance
                self._create_indexes(cursor)

                conn.commit()
                logger.info("Phase 2 database tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Phase 2 database: {e}")
            raise

    def _create_indexes(self, cursor):
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_alert_actions_alert_id ON alert_actions(alert_id)",
            "CREATE INDEX IF NOT EXISTS idx_alert_actions_timestamp ON alert_actions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_alert_actions_user_id ON alert_actions(user_id)",

            "CREATE INDEX IF NOT EXISTS idx_threshold_history_equipment_id ON threshold_history(equipment_id)",
            "CREATE INDEX IF NOT EXISTS idx_threshold_history_timestamp ON threshold_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_threshold_history_subsystem ON threshold_history(subsystem)",

            "CREATE INDEX IF NOT EXISTS idx_subsystem_analytics_subsystem ON subsystem_analytics(subsystem)",
            "CREATE INDEX IF NOT EXISTS idx_subsystem_analytics_spacecraft ON subsystem_analytics(spacecraft)",
            "CREATE INDEX IF NOT EXISTS idx_subsystem_analytics_date ON subsystem_analytics(analysis_date)",

            "CREATE INDEX IF NOT EXISTS idx_equipment_health_equipment_id ON equipment_health_snapshots(equipment_id)",
            "CREATE INDEX IF NOT EXISTS idx_equipment_health_timestamp ON equipment_health_snapshots(timestamp)",

            "CREATE INDEX IF NOT EXISTS idx_detection_details_equipment_id ON detection_details_history(equipment_id)",
            "CREATE INDEX IF NOT EXISTS idx_detection_details_timestamp ON detection_details_history(timestamp)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Error creating index: {e}")

    # Alert Actions Management
    def record_alert_action(self, action: AlertActionRecord) -> bool:
        """Record an alert action in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO alert_actions (
                        action_id, alert_id, action_type, user_id, timestamp,
                        reason, notes, work_order_id, escalation_level, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    action.action_id,
                    action.alert_id,
                    action.action_type,
                    action.user_id,
                    action.timestamp.isoformat(),
                    action.reason,
                    action.notes,
                    action.work_order_id,
                    action.escalation_level,
                    json.dumps(action.metadata) if action.metadata else None
                ))

                conn.commit()
                logger.info(f"Recorded alert action {action.action_id} for alert {action.alert_id}")
                return True

        except Exception as e:
            logger.error(f"Error recording alert action: {e}")
            return False

    def get_alert_actions(self, alert_id: str) -> List[AlertActionRecord]:
        """Get all actions for a specific alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT * FROM alert_actions
                    WHERE alert_id = ?
                    ORDER BY timestamp DESC
                ''', (alert_id,))

                rows = cursor.fetchall()
                actions = []

                for row in rows:
                    metadata = json.loads(row['metadata']) if row['metadata'] else None

                    action = AlertActionRecord(
                        action_id=row['action_id'],
                        alert_id=row['alert_id'],
                        action_type=row['action_type'],
                        user_id=row['user_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        reason=row['reason'],
                        notes=row['notes'],
                        work_order_id=row['work_order_id'],
                        escalation_level=row['escalation_level'],
                        metadata=metadata
                    )
                    actions.append(action)

                return actions

        except Exception as e:
            logger.error(f"Error getting alert actions for {alert_id}: {e}")
            return []

    def get_alert_action_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get alert action statistics for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get action counts by type
                cursor.execute('''
                    SELECT action_type, COUNT(*) as count
                    FROM alert_actions
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY action_type
                '''.format(days))

                action_counts = dict(cursor.fetchall())

                # Get actions by user
                cursor.execute('''
                    SELECT user_id, COUNT(*) as count
                    FROM alert_actions
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY user_id
                    ORDER BY count DESC
                    LIMIT 10
                '''.format(days))

                user_actions = dict(cursor.fetchall())

                # Get recent trends
                cursor.execute('''
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM alert_actions
                    WHERE timestamp >= datetime('now', '-{} days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                '''.format(days))

                daily_trends = dict(cursor.fetchall())

                return {
                    'action_counts': action_counts,
                    'user_actions': user_actions,
                    'daily_trends': daily_trends,
                    'total_actions': sum(action_counts.values())
                }

        except Exception as e:
            logger.error(f"Error getting alert action statistics: {e}")
            return {}

    # Threshold History Management
    def record_threshold_change(self, record: ThresholdHistoryRecord) -> bool:
        """Record a threshold change in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO threshold_history (
                        change_id, equipment_id, equipment_type, subsystem, criticality,
                        old_thresholds, new_thresholds, optimization_type, improvement_score,
                        confidence, justification, user_id, timestamp, applied
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.change_id,
                    record.equipment_id,
                    record.equipment_type,
                    record.subsystem,
                    record.criticality,
                    json.dumps(record.old_thresholds),
                    json.dumps(record.new_thresholds),
                    record.optimization_type,
                    record.improvement_score,
                    record.confidence,
                    record.justification,
                    record.user_id,
                    record.timestamp.isoformat(),
                    record.applied
                ))

                conn.commit()
                logger.info(f"Recorded threshold change {record.change_id} for equipment {record.equipment_id}")
                return True

        except Exception as e:
            logger.error(f"Error recording threshold change: {e}")
            return False

    def get_threshold_history(self, equipment_id: str, limit: int = 50) -> List[ThresholdHistoryRecord]:
        """Get threshold change history for equipment"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT * FROM threshold_history
                    WHERE equipment_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (equipment_id, limit))

                rows = cursor.fetchall()
                records = []

                for row in rows:
                    record = ThresholdHistoryRecord(
                        change_id=row['change_id'],
                        equipment_id=row['equipment_id'],
                        equipment_type=row['equipment_type'],
                        subsystem=row['subsystem'],
                        criticality=row['criticality'],
                        old_thresholds=json.loads(row['old_thresholds']),
                        new_thresholds=json.loads(row['new_thresholds']),
                        optimization_type=row['optimization_type'],
                        improvement_score=row['improvement_score'],
                        confidence=row['confidence'],
                        justification=row['justification'],
                        user_id=row['user_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        applied=bool(row['applied'])
                    )
                    records.append(record)

                return records

        except Exception as e:
            logger.error(f"Error getting threshold history for {equipment_id}: {e}")
            return []

    # Subsystem Analytics Management
    def record_subsystem_analysis(self, record: SubsystemAnalyticsRecord) -> bool:
        """Record subsystem analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO subsystem_analytics (
                        analysis_id, subsystem, spacecraft, analysis_type, analysis_date,
                        pattern_data, health_score, trend_direction, risk_factors,
                        recommendations, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.analysis_id,
                    record.subsystem,
                    record.spacecraft,
                    record.analysis_type,
                    record.analysis_date.isoformat(),
                    json.dumps(record.pattern_data),
                    record.health_score,
                    record.trend_direction,
                    json.dumps(record.risk_factors),
                    json.dumps(record.recommendations),
                    record.confidence_score
                ))

                conn.commit()
                logger.info(f"Recorded subsystem analysis {record.analysis_id}")
                return True

        except Exception as e:
            logger.error(f"Error recording subsystem analysis: {e}")
            return False

    def get_subsystem_health_trends(self, subsystem: str, spacecraft: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get health trends for a subsystem"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT * FROM subsystem_analytics
                    WHERE subsystem = ? AND spacecraft = ?
                    AND analysis_date >= datetime('now', '-{} days')
                    ORDER BY analysis_date DESC
                '''.format(days), (subsystem, spacecraft))

                rows = cursor.fetchall()
                trends = []

                for row in rows:
                    trend = {
                        'analysis_id': row['analysis_id'],
                        'analysis_date': row['analysis_date'],
                        'health_score': row['health_score'],
                        'trend_direction': row['trend_direction'],
                        'confidence_score': row['confidence_score'],
                        'risk_factors': json.loads(row['risk_factors']),
                        'recommendations': json.loads(row['recommendations'])
                    }
                    trends.append(trend)

                return trends

        except Exception as e:
            logger.error(f"Error getting subsystem health trends: {e}")
            return []

    # Equipment Health Snapshots
    def record_equipment_health_snapshot(self, equipment_id: str, equipment_type: str,
                                       subsystem: str, health_data: Dict[str, Any]) -> bool:
        """Record equipment health snapshot"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                snapshot_id = f"{equipment_id}_{datetime.now().isoformat()}"

                cursor.execute('''
                    INSERT INTO equipment_health_snapshots (
                        snapshot_id, equipment_id, equipment_type, subsystem, timestamp,
                        health_score, anomaly_count, sensor_count, critical_alerts,
                        performance_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot_id,
                    equipment_id,
                    equipment_type,
                    subsystem,
                    datetime.now().isoformat(),
                    health_data.get('health_score', 0.0),
                    health_data.get('anomaly_count', 0),
                    health_data.get('sensor_count', 0),
                    health_data.get('critical_alerts', 0),
                    json.dumps(health_data.get('performance_metrics', {}))
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error recording equipment health snapshot: {e}")
            return False

    # Utility Methods
    def cleanup_old_data(self, days: int = 365):
        """Cleanup old data beyond specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                tables_to_cleanup = [
                    'alert_actions',
                    'equipment_health_snapshots',
                    'detection_details_history'
                ]

                for table in tables_to_cleanup:
                    cursor.execute(f'''
                        DELETE FROM {table}
                        WHERE timestamp < datetime('now', '-{days} days')
                    ''')

                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                tables = [
                    'alert_actions',
                    'threshold_history',
                    'subsystem_analytics',
                    'equipment_health_snapshots',
                    'detection_details_history'
                ]

                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count

                # Database file size
                stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

                return stats

        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}


# Global instance for application use
phase2_db_manager = Phase2DatabaseManager()