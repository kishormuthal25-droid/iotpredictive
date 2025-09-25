"""
NASA Alerts Pipeline Component
Real-time alert generation, display, and management for NASA SMAP/MSL telemetry
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from collections import deque
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels for NASA missions"""
    CRITICAL = "CRITICAL"    # Mission threatening
    HIGH = "HIGH"           # Equipment failure risk
    MEDIUM = "MEDIUM"       # Performance degradation
    LOW = "LOW"            # Minor anomaly
    INFO = "INFO"          # Informational


class AlertStatus(Enum):
    """Alert lifecycle status"""
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class MissionType(Enum):
    """NASA mission types"""
    SMAP = "SMAP"
    MSL = "MSL"
    UNKNOWN = "UNKNOWN"


@dataclass
class NASAAlert:
    """NASA-specific alert structure"""
    alert_id: str
    timestamp: datetime
    mission: MissionType
    equipment_id: str
    subsystem: str
    sensor_id: str

    # Alert details
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus

    # Technical details
    anomaly_score: float
    threshold_exceeded: float
    model_confidence: float

    # NASA mission context
    spacecraft_status: str
    orbit_phase: Optional[str] = None  # For SMAP
    sol_day: Optional[int] = None      # For MSL
    mission_phase: str = "Operations"

    # Alert management
    created_by: str = "Anomaly Detection System"
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # Actions and escalation
    recommended_actions: List[str] = field(default_factory=list)
    escalation_level: int = 0
    work_order_created: bool = False

    # Metadata
    raw_telemetry_data: Dict[str, Any] = field(default_factory=dict)
    related_equipment: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'mission': self.mission.value,
            'equipment_id': self.equipment_id,
            'subsystem': self.subsystem,
            'sensor_id': self.sensor_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'status': self.status.value,
            'anomaly_score': self.anomaly_score,
            'threshold_exceeded': self.threshold_exceeded,
            'model_confidence': self.model_confidence,
            'spacecraft_status': self.spacecraft_status,
            'orbit_phase': self.orbit_phase,
            'sol_day': self.sol_day,
            'mission_phase': self.mission_phase,
            'recommended_actions': self.recommended_actions,
            'escalation_level': self.escalation_level,
            'work_order_created': self.work_order_created,
            'tags': self.tags
        }


class NASAAlertsManager:
    """
    Manages real-time NASA alerts pipeline
    Generates, tracks, and manages alerts from SMAP/MSL anomaly detection
    """

    def __init__(self):
        """Initialize NASA alerts manager"""
        # Import managers with error handling
        self.alert_manager = None
        self.nasa_alert_integration = None
        self.model_manager = None
        self.unified_orchestrator = None
        self._initialize_services()

        # Alert storage and tracking
        self.active_alerts: Dict[str, NASAAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_stream: deque = deque(maxlen=100)  # Recent alerts for live feed

        # Alert generation patterns for NASA missions
        self.alert_generators = {
            'SMAP': self._generate_smap_alerts,
            'MSL': self._generate_msl_alerts
        }

        # Mission-specific alert thresholds
        self.alert_thresholds = {
            'SMAP': {
                'CRITICAL': 0.8,
                'HIGH': 0.6,
                'MEDIUM': 0.4,
                'LOW': 0.2
            },
            'MSL': {
                'CRITICAL': 0.75,  # Lower threshold due to Mars environment
                'HIGH': 0.55,
                'MEDIUM': 0.35,
                'LOW': 0.15
            }
        }

        # Alert statistics
        self.alert_stats = {
            'total_generated': 0,
            'total_acknowledged': 0,
            'total_resolved': 0,
            'by_severity': {s.value: 0 for s in AlertSeverity},
            'by_mission': {m.value: 0 for m in MissionType if m != MissionType.UNKNOWN},
            'avg_resolution_time': 0.0,
            'active_count': 0
        }

        logger.info("NASA Alerts Manager initialized")

    def _initialize_services(self):
        """Initialize service connections with error handling"""
        try:
            from src.alerts.alert_manager import AlertManager
            self.alert_manager = AlertManager()
            logger.info("Connected to alert manager")
        except ImportError as e:
            logger.warning(f"Alert manager not available: {e}")

        try:
            from src.alerts.nasa_alert_integration import NASAAlertIntegration
            from src.data_ingestion.nasa_data_service import nasa_data_service
            from src.data_ingestion.equipment_mapper import equipment_mapper

            self.nasa_alert_integration = NASAAlertIntegration(
                nasa_service=nasa_data_service,
                equipment_mapper=equipment_mapper,
                alert_manager=self.alert_manager
            )
            logger.info("Connected to NASA alert integration")
        except ImportError as e:
            logger.warning(f"NASA alert integration not available: {e}")

        try:
            from src.dashboard.model_manager import pretrained_model_manager
            self.model_manager = pretrained_model_manager
            logger.info("Connected to model manager")
        except ImportError as e:
            logger.warning(f"Model manager not available: {e}")

        try:
            from src.dashboard.unified_data_orchestrator import unified_data_orchestrator
            self.unified_orchestrator = unified_data_orchestrator
            logger.info("Connected to unified data orchestrator")
        except ImportError as e:
            logger.warning(f"Unified data orchestrator not available: {e}")

    def generate_real_time_alerts(self) -> List[NASAAlert]:
        """Generate real-time alerts from current telemetry data

        Returns:
            List of newly generated alerts
        """
        new_alerts = []

        try:
            if self.unified_orchestrator:
                # Get real-time anomaly data
                equipment_status = self.unified_orchestrator.get_all_equipment_status()
                self._process_equipment_anomalies(equipment_status, new_alerts)

            if self.model_manager:
                # Get model predictions for alert generation
                self._process_model_predictions(new_alerts)

            # Generate simulated alerts for demonstration
            if not new_alerts or np.random.random() < 0.1:  # 10% chance of simulated alert
                simulated_alert = self._generate_simulated_alert()
                if simulated_alert:
                    new_alerts.append(simulated_alert)

            # Process new alerts
            for alert in new_alerts:
                self._process_new_alert(alert)

        except Exception as e:
            logger.error(f"Error generating real-time alerts: {e}")

        return new_alerts

    def _process_equipment_anomalies(self, equipment_status: Dict, new_alerts: List[NASAAlert]):
        """Process equipment anomalies for alert generation"""
        for equipment_id, status_data in equipment_status.items():
            if not status_data:
                continue

            anomaly_score = status_data.get('current_anomaly_score', 0.0)
            if anomaly_score > 0.15:  # Threshold for alert generation
                mission = self._determine_mission(equipment_id)
                severity = self._calculate_severity(anomaly_score, mission)

                # Check if we already have an active alert for this equipment
                existing_alert = self._find_active_alert(equipment_id)
                if existing_alert and existing_alert.severity == severity:
                    continue  # Don't duplicate similar alerts

                alert = self._create_equipment_alert(equipment_id, status_data, severity, mission)
                new_alerts.append(alert)

    def _process_model_predictions(self, new_alerts: List[NASAAlert]):
        """Process model predictions for predictive alerts"""
        try:
            if not self.model_manager:
                return

            # Get recent predictions from models
            available_models = self.model_manager.get_available_models()

            for model_id in available_models[:10]:  # Limit to 10 models for performance
                if np.random.random() < 0.05:  # 5% chance per model
                    # Simulate getting prediction data
                    sensor_data = self.model_manager.simulate_real_time_data(model_id)
                    prediction = self.model_manager.predict_anomaly(model_id, sensor_data)

                    if prediction.get('is_anomaly', False):
                        mission = self._determine_mission(model_id)
                        severity = self._calculate_severity(prediction['anomaly_score'], mission)

                        alert = self._create_prediction_alert(model_id, prediction, severity, mission)
                        new_alerts.append(alert)

        except Exception as e:
            logger.error(f"Error processing model predictions: {e}")

    def _generate_simulated_alert(self) -> Optional[NASAAlert]:
        """Generate simulated alert for demonstration"""
        try:
            missions = [MissionType.SMAP, MissionType.MSL]
            mission = np.random.choice(missions)

            if mission == MissionType.SMAP:
                equipment_id = f"SMAP_{np.random.randint(0, 25):02d}"
                subsystems = ["Power", "Communication", "Attitude", "Thermal", "Payload"]
                orbit_phase = np.random.choice(["Ascending", "Descending", "Polar"])
                spacecraft_status = "Nominal"
            else:  # MSL
                equipment_id = f"MSL_{np.random.randint(25, 80):02d}"
                subsystems = ["Power", "Mobility", "Environmental", "Science", "Communication", "Navigation"]
                orbit_phase = None
                spacecraft_status = np.random.choice(["Driving", "Parked", "Science Ops", "Maintenance"])

            subsystem = np.random.choice(subsystems)
            anomaly_score = np.random.uniform(0.2, 0.9)
            severity = self._calculate_severity(anomaly_score, mission)

            # Generate mission-specific alert content
            if mission == MissionType.SMAP:
                alert_content = self._generate_smap_alert_content(subsystem, anomaly_score)
            else:
                alert_content = self._generate_msl_alert_content(subsystem, anomaly_score)

            alert = NASAAlert(
                alert_id=f"NASA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(),
                mission=mission,
                equipment_id=equipment_id,
                subsystem=subsystem,
                sensor_id=equipment_id,
                title=alert_content['title'],
                description=alert_content['description'],
                severity=severity,
                status=AlertStatus.NEW,
                anomaly_score=anomaly_score,
                threshold_exceeded=anomaly_score,
                model_confidence=np.random.uniform(0.8, 0.98),
                spacecraft_status=spacecraft_status,
                orbit_phase=orbit_phase,
                sol_day=np.random.randint(3500, 4000) if mission == MissionType.MSL else None,
                recommended_actions=alert_content['actions'],
                tags=[mission.value, subsystem, severity.value]
            )

            return alert

        except Exception as e:
            logger.error(f"Error generating simulated alert: {e}")
            return None

    def _generate_smap_alert_content(self, subsystem: str, anomaly_score: float) -> Dict[str, Any]:
        """Generate SMAP-specific alert content"""
        content_templates = {
            "Power": {
                "title": "SMAP Power System Anomaly",
                "description": f"Anomaly detected in SMAP power subsystem (score: {anomaly_score:.3f}). Solar array performance may be degraded.",
                "actions": ["Check solar array alignment", "Verify battery charge levels", "Review power consumption patterns"]
            },
            "Communication": {
                "title": "SMAP Communication System Alert",
                "description": f"Communication anomaly detected (score: {anomaly_score:.3f}). Signal strength or data transmission issues possible.",
                "actions": ["Verify antenna pointing", "Check downlink schedule", "Review signal quality metrics"]
            },
            "Attitude": {
                "title": "SMAP Attitude Control Anomaly",
                "description": f"Attitude control system showing anomalous behavior (score: {anomaly_score:.3f}). Spacecraft orientation may be affected.",
                "actions": ["Check reaction wheels", "Verify gyroscope readings", "Review star tracker data"]
            },
            "Thermal": {
                "title": "SMAP Thermal Management Alert",
                "description": f"Thermal system anomaly detected (score: {anomaly_score:.3f}). Temperature regulation may be compromised.",
                "actions": ["Monitor component temperatures", "Check thermal radiator performance", "Verify heater operations"]
            },
            "Payload": {
                "title": "SMAP Science Payload Anomaly",
                "description": f"Science payload showing anomalous readings (score: {anomaly_score:.3f}). L-band radiometer performance may be affected.",
                "actions": ["Calibrate L-band radiometer", "Verify antenna deployment", "Check science data quality"]
            }
        }

        return content_templates.get(subsystem, {
            "title": f"SMAP {subsystem} System Alert",
            "description": f"Anomaly detected in SMAP {subsystem} (score: {anomaly_score:.3f})",
            "actions": ["Investigate anomaly", "Monitor system performance"]
        })

    def _generate_msl_alert_content(self, subsystem: str, anomaly_score: float) -> Dict[str, Any]:
        """Generate MSL-specific alert content"""
        content_templates = {
            "Power": {
                "title": "MSL Power System Anomaly",
                "description": f"Curiosity rover power system anomaly (score: {anomaly_score:.3f}). RTG or battery performance degraded.",
                "actions": ["Check RTG output", "Verify battery health", "Review power distribution"]
            },
            "Mobility": {
                "title": "MSL Mobility System Alert",
                "description": f"Rover mobility anomaly detected (score: {anomaly_score:.3f}). Wheel or suspension issues possible.",
                "actions": ["Inspect wheel condition", "Check suspension systems", "Verify motor performance"]
            },
            "Environmental": {
                "title": "MSL Environmental Monitoring Alert",
                "description": f"Environmental sensor anomaly (score: {anomaly_score:.3f}). Weather station or atmospheric sensors affected.",
                "actions": ["Calibrate weather sensors", "Check dust accumulation", "Verify sensor exposure"]
            },
            "Science": {
                "title": "MSL Science Instrument Anomaly",
                "description": f"Science instrument anomaly detected (score: {anomaly_score:.3f}). ChemCam, MAHLI, or other instruments affected.",
                "actions": ["Calibrate instruments", "Check laser performance", "Verify sample handling"]
            },
            "Communication": {
                "title": "MSL Communication System Alert",
                "description": f"Communication anomaly detected (score: {anomaly_score:.3f}). Earth relay or orbiter communication issues.",
                "actions": ["Check antenna alignment", "Verify UHF relay", "Review data transmission rates"]
            },
            "Navigation": {
                "title": "MSL Navigation System Anomaly",
                "description": f"Navigation system anomaly (score: {anomaly_score:.3f}). Hazard avoidance or positioning affected.",
                "actions": ["Verify IMU readings", "Check visual odometry", "Review terrain assessment"]
            }
        }

        return content_templates.get(subsystem, {
            "title": f"MSL {subsystem} System Alert",
            "description": f"Curiosity rover {subsystem} anomaly (score: {anomaly_score:.3f})",
            "actions": ["Investigate anomaly", "Monitor rover systems"]
        })

    def _determine_mission(self, equipment_id: str) -> MissionType:
        """Determine mission type from equipment ID"""
        if "SMAP" in equipment_id.upper():
            return MissionType.SMAP
        elif "MSL" in equipment_id.upper():
            return MissionType.MSL
        else:
            return MissionType.UNKNOWN

    def _calculate_severity(self, anomaly_score: float, mission: MissionType) -> AlertSeverity:
        """Calculate alert severity based on anomaly score and mission"""
        thresholds = self.alert_thresholds.get(mission.value, self.alert_thresholds['SMAP'])

        if anomaly_score >= thresholds['CRITICAL']:
            return AlertSeverity.CRITICAL
        elif anomaly_score >= thresholds['HIGH']:
            return AlertSeverity.HIGH
        elif anomaly_score >= thresholds['MEDIUM']:
            return AlertSeverity.MEDIUM
        elif anomaly_score >= thresholds['LOW']:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO

    def _find_active_alert(self, equipment_id: str) -> Optional[NASAAlert]:
        """Find active alert for equipment"""
        for alert in self.active_alerts.values():
            if alert.equipment_id == equipment_id and alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]:
                return alert
        return None

    def _create_equipment_alert(self, equipment_id: str, status_data: Dict, severity: AlertSeverity, mission: MissionType) -> NASAAlert:
        """Create alert from equipment status data"""
        subsystem = status_data.get('subsystem', 'Unknown')
        anomaly_score = status_data.get('current_anomaly_score', 0.0)

        if mission == MissionType.SMAP:
            content = self._generate_smap_alert_content(subsystem, anomaly_score)
        else:
            content = self._generate_msl_alert_content(subsystem, anomaly_score)

        return NASAAlert(
            alert_id=f"EQ-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            mission=mission,
            equipment_id=equipment_id,
            subsystem=subsystem,
            sensor_id=equipment_id,
            title=content['title'],
            description=content['description'],
            severity=severity,
            status=AlertStatus.NEW,
            anomaly_score=anomaly_score,
            threshold_exceeded=anomaly_score,
            model_confidence=status_data.get('model_confidence', 0.85),
            spacecraft_status="Operational",
            recommended_actions=content['actions'],
            raw_telemetry_data=status_data,
            tags=[mission.value, subsystem, severity.value, "Equipment"]
        )

    def _create_prediction_alert(self, model_id: str, prediction: Dict, severity: AlertSeverity, mission: MissionType) -> NASAAlert:
        """Create alert from model prediction"""
        return NASAAlert(
            alert_id=f"PRED-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            mission=mission,
            equipment_id=model_id,
            subsystem="Prediction",
            sensor_id=model_id,
            title=f"{mission.value} Predictive Anomaly Alert",
            description=f"Model {model_id} predicted anomaly (score: {prediction['anomaly_score']:.3f})",
            severity=severity,
            status=AlertStatus.NEW,
            anomaly_score=prediction['anomaly_score'],
            threshold_exceeded=prediction.get('threshold', 0.15),
            model_confidence=prediction.get('model_confidence', 0.90),
            spacecraft_status="Predictive",
            recommended_actions=["Investigate predicted anomaly", "Monitor system closely"],
            tags=[mission.value, "Predictive", severity.value, "Model"]
        )

    def _process_new_alert(self, alert: NASAAlert):
        """Process and store new alert"""
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.alert_stream.append(alert)

        # Update statistics
        self.alert_stats['total_generated'] += 1
        self.alert_stats['by_severity'][alert.severity.value] += 1
        self.alert_stats['by_mission'][alert.mission.value] += 1
        self.alert_stats['active_count'] = len([a for a in self.active_alerts.values()
                                              if a.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]])

        # Integration with existing alert manager
        if self.alert_manager:
            try:
                self.alert_manager.create_alert(
                    alert_type=getattr(self.alert_manager.AlertType, 'ANOMALY', 'ANOMALY'),
                    severity=getattr(self.alert_manager.AlertSeverity, alert.severity.value, 'MEDIUM'),
                    source=f"{alert.mission.value}_{alert.equipment_id}",
                    title=alert.title,
                    description=alert.description,
                    details=alert.to_dict(),
                    affected_equipment=[alert.equipment_id]
                )
            except Exception as e:
                logger.warning(f"Could not integrate with alert manager: {e}")

        logger.info(f"New {alert.severity.value} alert generated for {alert.mission.value} {alert.equipment_id}")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "Dashboard User") -> bool:
        """Acknowledge an alert

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User acknowledging the alert

        Returns:
            Success flag
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        self.alert_stats['total_acknowledged'] += 1

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = "Dashboard User", resolution_notes: str = "") -> bool:
        """Resolve an alert

        Args:
            alert_id: Alert ID to resolve
            resolved_by: User resolving the alert
            resolution_notes: Resolution notes

        Returns:
            Success flag
        """
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.now()

        if resolution_notes:
            alert.description += f" [Resolved: {resolution_notes}]"

        # Calculate resolution time
        if alert.acknowledged_at:
            resolution_time = (datetime.now() - alert.acknowledged_at).total_seconds() / 60
        else:
            resolution_time = (datetime.now() - alert.timestamp).total_seconds() / 60

        # Update statistics
        self.alert_stats['total_resolved'] += 1
        current_avg = self.alert_stats['avg_resolution_time']
        total_resolved = self.alert_stats['total_resolved']
        self.alert_stats['avg_resolution_time'] = (current_avg * (total_resolved - 1) + resolution_time) / total_resolved
        self.alert_stats['active_count'] = len([a for a in self.active_alerts.values()
                                              if a.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]])

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    def get_live_alerts_stream(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get live alerts stream for dashboard display

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        recent_alerts = list(self.alert_stream)[-limit:]
        return [alert.to_dict() for alert in reversed(recent_alerts)]

    def get_active_alerts(self, mission: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts, optionally filtered by mission

        Args:
            mission: Optional mission filter ('SMAP', 'MSL', or None for all)

        Returns:
            List of active alerts
        """
        active = [alert for alert in self.active_alerts.values()
                 if alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]]

        if mission:
            active = [alert for alert in active if alert.mission.value == mission]

        return [alert.to_dict() for alert in sorted(active, key=lambda a: a.timestamp, reverse=True)]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics

        Returns:
            Alert statistics dictionary
        """
        # Update active count
        self.alert_stats['active_count'] = len([a for a in self.active_alerts.values()
                                              if a.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]])

        return {
            **self.alert_stats,
            'last_alert': max([a.timestamp for a in self.active_alerts.values()]) if self.active_alerts else None,
            'resolution_rate': (self.alert_stats['total_resolved'] / max(self.alert_stats['total_generated'], 1)) * 100,
            'acknowledgment_rate': (self.alert_stats['total_acknowledged'] / max(self.alert_stats['total_generated'], 1)) * 100
        }


# Global instance for dashboard integration
nasa_alerts_manager = NASAAlertsManager()