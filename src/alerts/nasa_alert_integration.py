"""
NASA Alert Integration Module
Bridges NASA anomaly detection with the alert management system
Provides equipment-specific alert routing and criticality-based notifications
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import time

# Import project modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.alerts.alert_manager import (
    AlertManager, Alert, AlertSeverity, AlertStatus, AlertType,
    NotificationChannel, AlertRule
)
from src.data_ingestion.nasa_data_service import NASADataService
from src.data_ingestion.equipment_mapper import IoTEquipmentMapper
from src.maintenance.work_order_manager import WorkOrderManager, WorkOrderPriority
from config.settings import settings

logger = logging.getLogger(__name__)


class NASAEquipmentCriticality(Enum):
    """NASA Equipment criticality levels for mission operations"""
    MISSION_CRITICAL = 1    # Power, Mobility - Mission failure if down
    OPERATIONS_CRITICAL = 2 # Communication, Navigation - Severely impacts ops
    SCIENCE_CRITICAL = 3    # Scientific instruments - Impacts mission goals
    SUPPORT_CRITICAL = 4    # Environmental, Thermal - Impacts performance


@dataclass
class NASAAlertRule:
    """NASA-specific alert rule configuration"""
    equipment_type: str
    subsystem: str
    criticality: NASAEquipmentCriticality
    anomaly_thresholds: Dict[str, float]  # severity -> threshold
    alert_delays: Dict[str, int]  # severity -> seconds before alert
    notification_channels: Dict[str, List[NotificationChannel]]
    escalation_contacts: Dict[str, List[str]]
    failure_keywords: List[str]  # Keywords that indicate specific failures


class NASAAlertIntegration:
    """Main NASA Alert Integration system"""

    def __init__(self,
                 nasa_service: NASADataService,
                 equipment_mapper: IoTEquipmentMapper,
                 alert_manager: AlertManager,
                 work_order_manager: Optional[WorkOrderManager] = None):
        """Initialize NASA Alert Integration

        Args:
            nasa_service: NASA data service instance
            equipment_mapper: Equipment mapping service
            alert_manager: Alert management system
            work_order_manager: Work order management system (optional)
        """
        self.nasa_service = nasa_service
        self.equipment_mapper = equipment_mapper
        self.alert_manager = alert_manager
        self.work_order_manager = work_order_manager or WorkOrderManager()

        # NASA-specific configurations
        self.nasa_alert_rules = {}
        self.equipment_criticality = {}
        self.alert_history = {}
        self.pending_alerts = {}

        # Alert processing state
        self.monitoring_active = True
        self.last_check_time = datetime.now()

        # Initialize NASA-specific alert rules
        self._initialize_nasa_alert_rules()
        self._setup_equipment_criticality()
        self._register_alert_rules()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_nasa_anomalies)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Initialized NASA Alert Integration")

    def _initialize_nasa_alert_rules(self):
        """Initialize NASA equipment-specific alert rules"""

        # MISSION CRITICAL - Power Systems
        self.nasa_alert_rules['POWER'] = NASAAlertRule(
            equipment_type='POWER',
            subsystem='Power Management',
            criticality=NASAEquipmentCriticality.MISSION_CRITICAL,
            anomaly_thresholds={
                'CRITICAL': 0.90,   # 90% anomaly score triggers critical
                'HIGH': 0.75,       # 75% triggers high
                'MEDIUM': 0.60,     # 60% triggers medium
                'LOW': 0.45         # 45% triggers low
            },
            alert_delays={
                'CRITICAL': 0,      # Immediate for critical power issues
                'HIGH': 30,         # 30 seconds for high
                'MEDIUM': 120,      # 2 minutes for medium
                'LOW': 300          # 5 minutes for low
            },
            notification_channels={
                'CRITICAL': [NotificationChannel.EMAIL, NotificationChannel.SMS,
                           NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                'HIGH': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'MEDIUM': [NotificationChannel.EMAIL],
                'LOW': [NotificationChannel.DASHBOARD]
            },
            escalation_contacts={
                'CRITICAL': ['mission_controller', 'power_engineer', 'system_admin'],
                'HIGH': ['power_engineer', 'system_admin'],
                'MEDIUM': ['system_admin'],
                'LOW': []
            },
            failure_keywords=['voltage_drop', 'battery_failure', 'power_loss', 'solar_panel_fault']
        )

        # MISSION CRITICAL - Mobility Systems (Mars Rover)
        self.nasa_alert_rules['MOBILITY'] = NASAAlertRule(
            equipment_type='MOBILITY',
            subsystem='Mobility Control',
            criticality=NASAEquipmentCriticality.MISSION_CRITICAL,
            anomaly_thresholds={
                'CRITICAL': 0.90,
                'HIGH': 0.75,
                'MEDIUM': 0.60,
                'LOW': 0.45
            },
            alert_delays={
                'CRITICAL': 0,      # Immediate for mobility issues
                'HIGH': 30,
                'MEDIUM': 120,
                'LOW': 300
            },
            notification_channels={
                'CRITICAL': [NotificationChannel.EMAIL, NotificationChannel.SMS,
                           NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                'HIGH': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'MEDIUM': [NotificationChannel.EMAIL],
                'LOW': [NotificationChannel.DASHBOARD]
            },
            escalation_contacts={
                'CRITICAL': ['mission_controller', 'mobility_engineer', 'system_admin'],
                'HIGH': ['mobility_engineer', 'system_admin'],
                'MEDIUM': ['system_admin'],
                'LOW': []
            },
            failure_keywords=['wheel_motor_fault', 'suspension_failure', 'drive_system_error', 'stuck_rover']
        )

        # OPERATIONS CRITICAL - Communication Systems
        self.nasa_alert_rules['COMMUNICATION'] = NASAAlertRule(
            equipment_type='COMMUNICATION',
            subsystem='Communication System',
            criticality=NASAEquipmentCriticality.OPERATIONS_CRITICAL,
            anomaly_thresholds={
                'CRITICAL': 0.85,   # Slightly lower thresholds
                'HIGH': 0.70,
                'MEDIUM': 0.55,
                'LOW': 0.40
            },
            alert_delays={
                'CRITICAL': 60,     # 1 minute delay for comm issues
                'HIGH': 120,
                'MEDIUM': 300,
                'LOW': 600
            },
            notification_channels={
                'CRITICAL': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'HIGH': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'MEDIUM': [NotificationChannel.EMAIL],
                'LOW': [NotificationChannel.DASHBOARD]
            },
            escalation_contacts={
                'CRITICAL': ['comm_engineer', 'system_admin'],
                'HIGH': ['comm_engineer'],
                'MEDIUM': ['system_admin'],
                'LOW': []
            },
            failure_keywords=['signal_loss', 'antenna_fault', 'data_corruption', 'transmission_error']
        )

        # SCIENCE CRITICAL - Scientific Instruments
        self.nasa_alert_rules['SCIENCE'] = NASAAlertRule(
            equipment_type='SCIENCE',
            subsystem='Scientific Instruments',
            criticality=NASAEquipmentCriticality.SCIENCE_CRITICAL,
            anomaly_thresholds={
                'CRITICAL': 0.80,   # Lower thresholds for science equipment
                'HIGH': 0.65,
                'MEDIUM': 0.50,
                'LOW': 0.35
            },
            alert_delays={
                'CRITICAL': 300,    # 5 minutes for science equipment
                'HIGH': 600,
                'MEDIUM': 1800,
                'LOW': 3600
            },
            notification_channels={
                'CRITICAL': [NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'HIGH': [NotificationChannel.EMAIL],
                'MEDIUM': [NotificationChannel.EMAIL],
                'LOW': [NotificationChannel.DASHBOARD]
            },
            escalation_contacts={
                'CRITICAL': ['science_team', 'instrument_engineer'],
                'HIGH': ['instrument_engineer'],
                'MEDIUM': [],
                'LOW': []
            },
            failure_keywords=['sensor_drift', 'calibration_error', 'instrument_failure', 'data_quality']
        )

        # Default rule for other equipment types
        self.nasa_alert_rules['DEFAULT'] = NASAAlertRule(
            equipment_type='DEFAULT',
            subsystem='General Equipment',
            criticality=NASAEquipmentCriticality.SUPPORT_CRITICAL,
            anomaly_thresholds={
                'CRITICAL': 0.80,
                'HIGH': 0.65,
                'MEDIUM': 0.50,
                'LOW': 0.35
            },
            alert_delays={
                'CRITICAL': 600,    # 10 minutes for general equipment
                'HIGH': 1800,
                'MEDIUM': 3600,
                'LOW': 7200
            },
            notification_channels={
                'CRITICAL': [NotificationChannel.EMAIL],
                'HIGH': [NotificationChannel.EMAIL],
                'MEDIUM': [NotificationChannel.DASHBOARD],
                'LOW': [NotificationChannel.DASHBOARD]
            },
            escalation_contacts={
                'CRITICAL': ['system_admin'],
                'HIGH': [],
                'MEDIUM': [],
                'LOW': []
            },
            failure_keywords=['general_fault', 'sensor_error', 'system_warning']
        )

    def _setup_equipment_criticality(self):
        """Setup equipment criticality mapping"""
        all_equipment = self.equipment_mapper.get_all_equipment()

        for equipment in all_equipment:
            equipment_type = equipment.equipment_type.upper()

            # Map NASA equipment to criticality levels
            if equipment_type in ['POWER']:
                self.equipment_criticality[equipment.equipment_id] = NASAEquipmentCriticality.MISSION_CRITICAL
            elif equipment_type in ['MOBILITY']:
                self.equipment_criticality[equipment.equipment_id] = NASAEquipmentCriticality.MISSION_CRITICAL
            elif equipment_type in ['COMMUNICATION', 'NAVIGATION']:
                self.equipment_criticality[equipment.equipment_id] = NASAEquipmentCriticality.OPERATIONS_CRITICAL
            elif equipment_type in ['SCIENCE', 'PAYLOAD']:
                self.equipment_criticality[equipment.equipment_id] = NASAEquipmentCriticality.SCIENCE_CRITICAL
            else:
                self.equipment_criticality[equipment.equipment_id] = NASAEquipmentCriticality.SUPPORT_CRITICAL

    def _register_alert_rules(self):
        """Register NASA alert rules with the alert manager"""
        for rule_name, nasa_rule in self.nasa_alert_rules.items():

            # Create alert rule for each severity level
            for severity_name, threshold in nasa_rule.anomaly_thresholds.items():
                rule_id = f"nasa_{rule_name}_{severity_name.lower()}"

                # Convert severity name to AlertSeverity enum
                alert_severity = getattr(AlertSeverity, severity_name, AlertSeverity.MEDIUM)

                alert_rule = AlertRule(
                    rule_id=rule_id,
                    name=f"NASA {rule_name} {severity_name} Anomaly",
                    description=f"NASA {rule_name} equipment anomaly detection at {severity_name} level",
                    condition=f"anomaly_score >= {threshold} and equipment_type == '{rule_name}'",
                    severity=alert_severity,
                    alert_type=AlertType.ANOMALY,
                    channels=nasa_rule.notification_channels.get(severity_name, [NotificationChannel.DASHBOARD]),
                    recipients=nasa_rule.escalation_contacts.get(severity_name, []),
                    cooldown_minutes=nasa_rule.alert_delays.get(severity_name, 300) // 60,
                    metadata={
                        'nasa_rule': True,
                        'equipment_type': rule_name,
                        'criticality': nasa_rule.criticality.name,
                        'failure_keywords': nasa_rule.failure_keywords
                    }
                )

                self.alert_manager.add_alert_rule(alert_rule)

        logger.info(f"Registered {len(self.nasa_alert_rules) * 4} NASA alert rules")

    def _monitor_nasa_anomalies(self):
        """Background monitoring of NASA anomalies"""
        logger.info("Started NASA anomaly monitoring thread")

        while self.monitoring_active:
            try:
                # Get recent anomalies from NASA service
                recent_anomalies = self.nasa_service.get_anomaly_data(time_window="5min")

                # Process each anomaly for alert generation
                for anomaly_data in recent_anomalies:
                    self._process_nasa_anomaly(anomaly_data)

                # Check for pending alerts that should be sent
                self._process_pending_alerts()

                # Update last check time
                self.last_check_time = datetime.now()

            except Exception as e:
                logger.error(f"Error in NASA anomaly monitoring: {e}")

            # Wait before next check
            time.sleep(30)  # Check every 30 seconds

    def _process_nasa_anomaly(self, anomaly_data: Dict[str, Any]):
        """Process a NASA anomaly for alert generation

        Args:
            anomaly_data: Anomaly data from NASA service
        """
        try:
            equipment_id = anomaly_data.get('equipment', 'UNKNOWN')
            anomaly_score = float(anomaly_data.get('score', 0.0))
            severity = anomaly_data.get('severity', 'LOW')
            timestamp_str = anomaly_data.get('timestamp', '')

            # Parse timestamp
            anomaly_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            # Skip if anomaly is too old (already processed)
            if anomaly_timestamp < self.last_check_time:
                return

            # Get equipment information
            equipment = self.equipment_mapper.get_equipment_by_id(equipment_id)
            if not equipment:
                logger.warning(f"Unknown equipment ID in anomaly: {equipment_id}")
                return

            equipment_type = equipment.equipment_type.upper()

            # Get appropriate NASA alert rule
            nasa_rule = self.nasa_alert_rules.get(equipment_type, self.nasa_alert_rules['DEFAULT'])

            # Check if this anomaly severity meets the alert threshold
            threshold = nasa_rule.anomaly_thresholds.get(severity, 1.0)
            if anomaly_score < threshold:
                return  # Below threshold, no alert needed

            # Check for duplicate recent alerts
            alert_key = f"{equipment_id}_{severity}_{anomaly_timestamp.strftime('%Y%m%d_%H%M')}"
            if alert_key in self.alert_history:
                return  # Already processed this anomaly

            # Determine alert delay
            alert_delay = nasa_rule.alert_delays.get(severity, 300)

            if alert_delay == 0:
                # Send immediately
                self._create_nasa_alert(anomaly_data, equipment, nasa_rule)
            else:
                # Schedule for later
                alert_time = anomaly_timestamp + timedelta(seconds=alert_delay)
                self.pending_alerts[alert_key] = {
                    'alert_time': alert_time,
                    'anomaly_data': anomaly_data,
                    'equipment': equipment,
                    'nasa_rule': nasa_rule
                }

            # Record in history
            self.alert_history[alert_key] = anomaly_timestamp

        except Exception as e:
            logger.error(f"Error processing NASA anomaly: {e}")

    def _process_pending_alerts(self):
        """Process pending alerts that are ready to be sent"""
        current_time = datetime.now()
        ready_alerts = []

        # Find alerts that are ready
        for alert_key, alert_info in self.pending_alerts.items():
            if current_time >= alert_info['alert_time']:
                ready_alerts.append((alert_key, alert_info))

        # Process ready alerts
        for alert_key, alert_info in ready_alerts:
            try:
                self._create_nasa_alert(
                    alert_info['anomaly_data'],
                    alert_info['equipment'],
                    alert_info['nasa_rule']
                )

                # Remove from pending
                del self.pending_alerts[alert_key]

            except Exception as e:
                logger.error(f"Error creating pending alert {alert_key}: {e}")

    def _create_nasa_alert(self,
                          anomaly_data: Dict[str, Any],
                          equipment,
                          nasa_rule: NASAAlertRule):
        """Create NASA anomaly alert

        Args:
            anomaly_data: Anomaly data from NASA service
            equipment: Equipment object
            nasa_rule: NASA alert rule
        """
        try:
            equipment_id = anomaly_data.get('equipment', 'UNKNOWN')
            severity = anomaly_data.get('severity', 'LOW')
            anomaly_score = float(anomaly_data.get('score', 0.0))
            model_name = anomaly_data.get('model', 'Unknown')
            equipment_type = anomaly_data.get('type', 'Unknown')

            # Convert severity to AlertSeverity
            alert_severity = getattr(AlertSeverity, severity, AlertSeverity.MEDIUM)

            # Create alert title
            alert_title = f"NASA {equipment_type} Anomaly - {equipment_id}"

            # Create detailed description
            description = self._generate_alert_description(
                anomaly_data, equipment, nasa_rule
            )

            # Prepare alert details
            alert_details = {
                'nasa_equipment_id': equipment_id,
                'nasa_equipment_type': equipment_type,
                'anomaly_score': anomaly_score,
                'detection_model': model_name,
                'criticality_level': nasa_rule.criticality.name,
                'sensor_count': len(equipment.sensors),
                'subsystem': nasa_rule.subsystem,
                'failure_risk': self._assess_failure_risk(anomaly_score, nasa_rule),
                'recommended_actions': self._get_recommended_actions(severity, equipment_type),
                'raw_anomaly_data': anomaly_data
            }

            # Prepare metrics
            alert_metrics = {
                'anomaly_score': anomaly_score,
                'threshold_exceeded': anomaly_score - nasa_rule.anomaly_thresholds[severity],
                'confidence_level': self._calculate_confidence(anomaly_score),
                'equipment_sensor_count': len(equipment.sensors)
            }

            # Create the alert
            alert = self.alert_manager.create_alert(
                alert_type=AlertType.ANOMALY,
                severity=alert_severity,
                source=f"NASA_{equipment_type}",
                title=alert_title,
                description=description,
                details=alert_details,
                affected_equipment=[equipment_id],
                metrics=alert_metrics,
                rule_id=f"nasa_{equipment_type}_{severity.lower()}"
            )

            logger.info(f"Created NASA alert {alert.alert_id} for {equipment_id} ({severity})")

            # Auto-create work order for critical alerts
            if alert_severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                work_order = self._create_work_order_from_alert(alert, equipment)
                if work_order:
                    alert.work_order_id = work_order.work_order_id
                    alert.details['work_order_created'] = True
                    alert.details['work_order_id'] = work_order.work_order_id

            return alert

        except Exception as e:
            logger.error(f"Error creating NASA alert: {e}")
            return None

    def _generate_alert_description(self,
                                  anomaly_data: Dict[str, Any],
                                  equipment,
                                  nasa_rule: NASAAlertRule) -> str:
        """Generate detailed alert description

        Args:
            anomaly_data: Anomaly data
            equipment: Equipment object
            nasa_rule: NASA alert rule

        Returns:
            Alert description
        """
        equipment_id = anomaly_data.get('equipment', 'UNKNOWN')
        severity = anomaly_data.get('severity', 'LOW')
        anomaly_score = float(anomaly_data.get('score', 0.0))
        equipment_type = anomaly_data.get('type', 'Unknown')

        # Get equipment mission context
        mission_context = self._get_mission_context(equipment_id)

        # Generate equipment-failure specific message
        failure_message = self._generate_failure_specific_message(
            equipment_id, equipment_type, severity, anomaly_score, nasa_rule
        )

        description = f"""
NASA {equipment_type} {failure_message}

Equipment: {equipment_id} ({equipment.equipment_name if equipment else 'Unknown'})
Mission: {mission_context}
Criticality: {nasa_rule.criticality.name}

Anomaly Details:
- Severity: {severity}
- Anomaly Score: {anomaly_score:.3f}
- Detection Model: {anomaly_data.get('model', 'Unknown')}
- Sensors Affected: {len(equipment.sensors) if equipment else 'Unknown'}

Risk Assessment:
{self._assess_failure_risk(anomaly_score, nasa_rule)}

Immediate Impact:
{self._assess_immediate_impact(severity, equipment_type)}

Recommended Actions:
{chr(10).join(f"- {action}" for action in self._get_recommended_actions(severity, equipment_type))}

Emergency Contacts:
{chr(10).join(f"- {contact}" for contact in nasa_rule.escalation_contacts.get(severity, []))}
        """.strip()

        return description

    def _get_mission_context(self, equipment_id: str) -> str:
        """Get mission context for equipment

        Args:
            equipment_id: Equipment ID

        Returns:
            Mission context description
        """
        if equipment_id.startswith('SMAP'):
            return 'SMAP Satellite - Soil Moisture Monitoring Mission'
        elif equipment_id.startswith('MSL'):
            return 'Mars Science Laboratory (Curiosity Rover)'
        else:
            return 'NASA Mission Equipment'

    def _assess_failure_risk(self, anomaly_score: float, nasa_rule: NASAAlertRule) -> str:
        """Assess failure risk based on anomaly score and equipment criticality

        Args:
            anomaly_score: Anomaly detection score
            nasa_rule: NASA alert rule

        Returns:
            Risk assessment description
        """
        if nasa_rule.criticality == NASAEquipmentCriticality.MISSION_CRITICAL:
            if anomaly_score >= 0.90:
                return "EXTREME RISK: Mission-critical equipment showing severe anomalies. Immediate intervention required."
            elif anomaly_score >= 0.75:
                return "HIGH RISK: Mission-critical equipment degradation detected. Priority attention needed."
            else:
                return "MODERATE RISK: Mission-critical equipment showing anomalous behavior. Monitor closely."
        elif nasa_rule.criticality == NASAEquipmentCriticality.OPERATIONS_CRITICAL:
            if anomaly_score >= 0.85:
                return "HIGH RISK: Operations-critical equipment failure imminent. Plan contingency operations."
            else:
                return "MEDIUM RISK: Operations-critical equipment anomaly detected. Assess impact on mission timeline."
        else:
            return "LOW-MEDIUM RISK: Equipment anomaly detected. Schedule diagnostic and maintenance."

    def _assess_immediate_impact(self, severity: str, equipment_type: str) -> str:
        """Assess immediate impact of the anomaly

        Args:
            severity: Anomaly severity
            equipment_type: Type of equipment

        Returns:
            Impact assessment
        """
        impact_matrix = {
            'POWER': {
                'CRITICAL': 'Mission abort risk - power system failure could end mission',
                'HIGH': 'Reduced operational capability - backup power systems may engage',
                'MEDIUM': 'Power efficiency reduced - monitor battery and solar panel performance',
                'LOW': 'Minor power fluctuations - no immediate operational impact'
            },
            'MOBILITY': {
                'CRITICAL': 'Rover immobilization risk - mission objectives severely compromised',
                'HIGH': 'Reduced mobility - limit movement to essential operations only',
                'MEDIUM': 'Mobility system degradation - plan alternative routes and procedures',
                'LOW': 'Minor mobility anomalies - continue operations with increased monitoring'
            },
            'COMMUNICATION': {
                'CRITICAL': 'Communication blackout risk - mission control contact may be lost',
                'HIGH': 'Reduced communication capability - prioritize critical data transmission',
                'MEDIUM': 'Communication degradation - expect intermittent connectivity issues',
                'LOW': 'Minor communication anomalies - no immediate impact on operations'
            },
            'DEFAULT': {
                'CRITICAL': 'Equipment failure imminent - immediate diagnostic required',
                'HIGH': 'Equipment degradation - reduced operational capability',
                'MEDIUM': 'Equipment anomaly - monitor and assess trend',
                'LOW': 'Minor equipment anomaly - schedule routine inspection'
            }
        }

        equipment_impacts = impact_matrix.get(equipment_type, impact_matrix['DEFAULT'])
        return equipment_impacts.get(severity, 'Unknown impact level')

    def _get_recommended_actions(self, severity: str, equipment_type: str) -> List[str]:
        """Get recommended actions for the anomaly

        Args:
            severity: Anomaly severity
            equipment_type: Type of equipment

        Returns:
            List of recommended actions
        """
        action_matrix = {
            'POWER': {
                'CRITICAL': [
                    'Immediate assessment by power systems engineer',
                    'Activate backup power protocols',
                    'Prepare for emergency power conservation mode',
                    'Contact mission control for guidance'
                ],
                'HIGH': [
                    'Review power consumption patterns',
                    'Check solar panel orientation and efficiency',
                    'Monitor battery charge cycles',
                    'Schedule power system diagnostic'
                ],
                'MEDIUM': [
                    'Monitor power trends over next 24 hours',
                    'Review recent operational changes',
                    'Schedule routine power system check'
                ],
                'LOW': [
                    'Continue normal operations',
                    'Log anomaly for trend analysis'
                ]
            },
            'MOBILITY': {
                'CRITICAL': [
                    'Halt all mobility operations immediately',
                    'Run comprehensive mobility diagnostic',
                    'Assess terrain and positioning risks',
                    'Plan emergency communication protocols'
                ],
                'HIGH': [
                    'Reduce mobility operations to essential only',
                    'Check wheel motor temperatures and currents',
                    'Review recent terrain and navigation data',
                    'Prepare alternative operational procedures'
                ],
                'MEDIUM': [
                    'Monitor mobility system performance',
                    'Limit high-stress maneuvers',
                    'Schedule mobility system inspection'
                ],
                'LOW': [
                    'Continue operations with monitoring',
                    'Log anomaly for trend analysis'
                ]
            },
            'COMMUNICATION': {
                'CRITICAL': [
                    'Establish communication with mission control urgently',
                    'Switch to backup communication systems',
                    'Prioritize critical data transmission',
                    'Prepare for autonomous operations if needed'
                ],
                'HIGH': [
                    'Check antenna pointing and signal strength',
                    'Review communication schedules',
                    'Test backup communication channels',
                    'Monitor data transmission quality'
                ],
                'MEDIUM': [
                    'Monitor communication performance',
                    'Schedule communication system diagnostic',
                    'Review recent configuration changes'
                ],
                'LOW': [
                    'Continue normal operations',
                    'Log communication anomaly for analysis'
                ]
            }
        }

        equipment_actions = action_matrix.get(equipment_type, {})
        default_actions = [
            'Review equipment status and recent operational data',
            'Schedule diagnostic assessment',
            'Monitor trends over next operational period',
            'Document anomaly for engineering analysis'
        ]

        return equipment_actions.get(severity, default_actions)

    def _calculate_confidence(self, anomaly_score: float) -> float:
        """Calculate confidence level for anomaly detection

        Args:
            anomaly_score: Anomaly detection score

        Returns:
            Confidence level (0-1)
        """
        # Higher anomaly scores have higher confidence
        # Confidence = min(anomaly_score * 1.1, 1.0)
        return min(anomaly_score * 1.1, 1.0)

    def get_nasa_alert_statistics(self) -> Dict[str, Any]:
        """Get NASA-specific alert statistics

        Returns:
            NASA alert statistics
        """
        # Get all NASA-related alerts
        nasa_alerts = [
            alert for alert in self.alert_manager.alerts.values()
            if alert.source.startswith('NASA_')
        ]

        # Statistics by equipment type
        stats_by_equipment = {}
        for alert in nasa_alerts:
            equipment_type = alert.source.replace('NASA_', '')
            if equipment_type not in stats_by_equipment:
                stats_by_equipment[equipment_type] = {
                    'total': 0,
                    'by_severity': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                    'active': 0,
                    'resolved': 0
                }

            stats_by_equipment[equipment_type]['total'] += 1
            stats_by_equipment[equipment_type]['by_severity'][alert.severity.name] += 1

            if alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED, AlertStatus.IN_PROGRESS]:
                stats_by_equipment[equipment_type]['active'] += 1
            elif alert.status == AlertStatus.RESOLVED:
                stats_by_equipment[equipment_type]['resolved'] += 1

        return {
            'total_nasa_alerts': len(nasa_alerts),
            'active_nasa_alerts': len([a for a in nasa_alerts if a.status not in [AlertStatus.RESOLVED, AlertStatus.CLOSED]]),
            'alerts_by_equipment': stats_by_equipment,
            'pending_alerts': len(self.pending_alerts),
            'alert_rules_registered': len(self.nasa_alert_rules) * 4,
            'equipment_criticality_mapped': len(self.equipment_criticality),
            'monitoring_active': self.monitoring_active,
            'last_check_time': self.last_check_time.isoformat()
        }

    def stop_monitoring(self):
        """Stop the NASA anomaly monitoring"""
        self.monitoring_active = False
        logger.info("Stopped NASA anomaly monitoring")

    def create_manual_nasa_alert(self,
                                equipment_id: str,
                                severity: str,
                                description: str,
                                additional_details: Dict[str, Any] = None) -> Optional[Alert]:
        """Create a manual NASA alert

        Args:
            equipment_id: Equipment ID
            severity: Alert severity (CRITICAL, HIGH, MEDIUM, LOW)
            description: Alert description
            additional_details: Additional alert details

        Returns:
            Created alert or None if failed
        """
        try:
            # Get equipment information
            equipment = self.equipment_mapper.get_equipment_by_id(equipment_id)
            if not equipment:
                logger.error(f"Equipment {equipment_id} not found")
                return None

            equipment_type = equipment.equipment_type.upper()
            nasa_rule = self.nasa_alert_rules.get(equipment_type, self.nasa_alert_rules['DEFAULT'])

            # Convert severity to AlertSeverity
            alert_severity = getattr(AlertSeverity, severity, AlertSeverity.MEDIUM)

            # Create alert details
            alert_details = {
                'nasa_equipment_id': equipment_id,
                'nasa_equipment_type': equipment_type,
                'manual_alert': True,
                'criticality_level': nasa_rule.criticality.name,
                'subsystem': nasa_rule.subsystem,
                'created_by': 'manual_intervention'
            }

            if additional_details:
                alert_details.update(additional_details)

            # Create the alert
            alert = self.alert_manager.create_alert(
                alert_type=AlertType.MAINTENANCE,
                severity=alert_severity,
                source=f"NASA_{equipment_type}",
                title=f"Manual NASA Alert - {equipment_id}",
                description=description,
                details=alert_details,
                affected_equipment=[equipment_id],
                metrics={'manual_alert': 1.0}
            )

            logger.info(f"Created manual NASA alert {alert.alert_id} for {equipment_id}")
            return alert

        except Exception as e:
            logger.error(f"Error creating manual NASA alert: {e}")
            return None

    def _generate_failure_specific_message(self,
                                         equipment_id: str,
                                         equipment_type: str,
                                         severity: str,
                                         anomaly_score: float,
                                         nasa_rule: NASAAlertRule) -> str:
        """Generate equipment-failure specific alert message

        Args:
            equipment_id: Equipment ID
            equipment_type: Equipment type
            severity: Alert severity
            anomaly_score: Anomaly score
            nasa_rule: NASA alert rule

        Returns:
            Failure-specific message
        """
        # Equipment-specific failure messages based on type and severity
        failure_messages = {
            'POWER': {
                'CRITICAL': 'POWER SYSTEM FAILURE IMMINENT - Mission-critical power anomaly detected',
                'HIGH': 'Power System Degradation - Significant power anomaly detected',
                'MEDIUM': 'Power Performance Warning - Power system anomaly detected',
                'LOW': 'Power System Monitoring Alert - Minor power anomaly detected'
            },
            'MOBILITY': {
                'CRITICAL': 'MOBILITY SYSTEM FAILURE - Rover immobilization risk detected',
                'HIGH': 'Mobility System Warning - Significant mobility anomaly detected',
                'MEDIUM': 'Mobility Performance Alert - Mobility system anomaly detected',
                'LOW': 'Mobility System Monitoring - Minor mobility anomaly detected'
            },
            'COMMUNICATION': {
                'CRITICAL': 'COMMUNICATION BLACKOUT RISK - Critical communication anomaly',
                'HIGH': 'Communication System Warning - Significant communication anomaly',
                'MEDIUM': 'Communication Performance Alert - Communication anomaly detected',
                'LOW': 'Communication System Monitoring - Minor communication anomaly'
            },
            'SCIENCE': {
                'CRITICAL': 'SCIENCE INSTRUMENT FAILURE - Critical science system anomaly',
                'HIGH': 'Science System Warning - Significant science instrument anomaly',
                'MEDIUM': 'Science Performance Alert - Science instrument anomaly detected',
                'LOW': 'Science System Monitoring - Minor science instrument anomaly'
            },
            'NAVIGATION': {
                'CRITICAL': 'NAVIGATION SYSTEM FAILURE - Critical navigation anomaly detected',
                'HIGH': 'Navigation System Warning - Significant navigation anomaly',
                'MEDIUM': 'Navigation Performance Alert - Navigation system anomaly',
                'LOW': 'Navigation System Monitoring - Minor navigation anomaly'
            },
            'THERMAL': {
                'CRITICAL': 'THERMAL CONTROL FAILURE - Critical thermal management anomaly',
                'HIGH': 'Thermal System Warning - Significant thermal anomaly detected',
                'MEDIUM': 'Thermal Performance Alert - Thermal system anomaly',
                'LOW': 'Thermal System Monitoring - Minor thermal anomaly'
            }
        }

        # Get failure message for equipment type and severity
        equipment_messages = failure_messages.get(equipment_type, failure_messages.get('SCIENCE', {}))
        base_message = equipment_messages.get(severity, f'{equipment_type} Anomaly Detected')

        # Add score-based intensity
        if anomaly_score >= 0.95:
            intensity = " - EXTREME ANOMALY"
        elif anomaly_score >= 0.90:
            intensity = " - SEVERE ANOMALY"
        elif anomaly_score >= 0.75:
            intensity = " - SIGNIFICANT ANOMALY"
        elif anomaly_score >= 0.60:
            intensity = " - MODERATE ANOMALY"
        else:
            intensity = " - MINOR ANOMALY"

        return base_message + intensity

    def _create_work_order_from_alert(self, alert: Alert, equipment) -> Optional[Any]:
        """Create work order from NASA alert

        Args:
            alert: NASA alert
            equipment: Equipment object

        Returns:
            Created work order or None
        """
        try:
            equipment_id = alert.details.get('nasa_equipment_id', 'Unknown')
            equipment_type = alert.details.get('nasa_equipment_type', 'Unknown')
            severity = alert.severity.name
            criticality = alert.details.get('criticality_level', 'Unknown')

            # Map alert severity to work order priority
            priority_map = {
                'CRITICAL': WorkOrderPriority.CRITICAL,
                'HIGH': WorkOrderPriority.HIGH,
                'MEDIUM': WorkOrderPriority.MEDIUM,
                'LOW': WorkOrderPriority.LOW
            }

            work_order_priority = priority_map.get(severity, WorkOrderPriority.MEDIUM)

            # Generate work order title
            title = f"NASA {equipment_type} Maintenance - {equipment_id}"

            # Generate work order description
            description = f"""
AUTOMATED WORK ORDER: NASA Alert {alert.alert_id}

Equipment: {equipment_id}
Mission: {self._get_mission_context(equipment_id)}
Criticality: {criticality}
Alert Severity: {severity}

ISSUE DESCRIPTION:
{alert.description}

REQUIRED ACTIONS:
{chr(10).join(f"- {action}" for action in self._get_maintenance_actions(equipment_type, severity))}

SAFETY CONSIDERATIONS:
{self._get_safety_considerations(equipment_type, severity)}

PARTS/TOOLS REQUIRED:
{chr(10).join(f"- {item}" for item in self._get_required_parts(equipment_type))}

ESTIMATED DURATION: {self._estimate_repair_duration(equipment_type, severity)}
SKILL LEVEL REQUIRED: {self._get_required_skill_level(equipment_type, severity)}
            """.strip()

            # Create work order using work order manager
            work_order = self.work_order_manager.create_work_order(
                title=title,
                description=description,
                priority=work_order_priority,
                equipment_id=equipment_id,
                alert_id=alert.alert_id,
                estimated_duration=self._estimate_repair_duration_hours(equipment_type, severity),
                required_skills=[self._get_required_skill_level(equipment_type, severity)],
                parts_required=self._get_required_parts(equipment_type)
            )

            logger.info(f"Created work order {work_order.work_order_id} for NASA alert {alert.alert_id}")
            return work_order

        except Exception as e:
            logger.error(f"Error creating work order from NASA alert: {e}")
            return None

    def _get_maintenance_actions(self, equipment_type: str, severity: str) -> List[str]:
        """Get maintenance actions for equipment type and severity"""
        actions = {
            'POWER': {
                'CRITICAL': [
                    'Immediately isolate power system to prevent cascading failures',
                    'Run comprehensive power system diagnostic',
                    'Check all power connections and voltages',
                    'Test battery health and charging systems',
                    'Verify solar panel functionality and orientation',
                    'Replace any failed power components'
                ],
                'HIGH': [
                    'Run power system diagnostic test',
                    'Check battery charge levels and health',
                    'Verify solar panel performance',
                    'Inspect power distribution connections',
                    'Monitor power consumption patterns'
                ],
                'MEDIUM': [
                    'Schedule power system health check',
                    'Monitor power trends over next 48 hours',
                    'Check recent power consumption changes'
                ]
            },
            'MOBILITY': {
                'CRITICAL': [
                    'Immediately halt all rover movement operations',
                    'Run full mobility system diagnostic',
                    'Check wheel motor temperatures and currents',
                    'Inspect suspension and steering mechanisms',
                    'Test rocker-bogie suspension system',
                    'Verify navigation and positioning systems'
                ],
                'HIGH': [
                    'Limit rover movement to essential operations only',
                    'Run mobility system diagnostic',
                    'Check wheel motor performance',
                    'Inspect drive system components'
                ]
            },
            'COMMUNICATION': {
                'CRITICAL': [
                    'Switch to backup communication systems immediately',
                    'Test all communication channels',
                    'Check antenna pointing and orientation',
                    'Verify radio frequency systems',
                    'Test data transmission capabilities'
                ]
            }
        }

        equipment_actions = actions.get(equipment_type, {})
        return equipment_actions.get(severity, [
            f'Perform {equipment_type.lower()} system diagnostic',
            f'Check {equipment_type.lower()} system components',
            f'Monitor {equipment_type.lower()} performance trends'
        ])

    def _get_safety_considerations(self, equipment_type: str, severity: str) -> str:
        """Get safety considerations for maintenance"""
        if equipment_type == 'POWER' and severity == 'CRITICAL':
            return "EXTREME CAUTION: High voltage systems. Risk of mission termination if power lost."
        elif equipment_type == 'MOBILITY' and severity == 'CRITICAL':
            return "Mission safety risk: Rover may become immobilized. Ensure terrain safety before operations."
        elif severity == 'CRITICAL':
            return f"Mission-critical {equipment_type.lower()} system. Follow all safety protocols."
        else:
            return f"Standard safety protocols for {equipment_type.lower()} system maintenance."

    def _get_required_parts(self, equipment_type: str) -> List[str]:
        """Get required parts for equipment maintenance"""
        parts = {
            'POWER': ['Power connectors', 'Voltage regulators', 'Battery modules', 'Solar panel components'],
            'MOBILITY': ['Wheel motors', 'Suspension components', 'Drive gears', 'Position sensors'],
            'COMMUNICATION': ['Antenna components', 'Radio modules', 'Signal amplifiers', 'Data cables'],
            'SCIENCE': ['Sensor modules', 'Calibration standards', 'Instrument covers', 'Data storage'],
            'NAVIGATION': ['IMU components', 'GPS modules', 'Compass sensors', 'Navigation software'],
            'THERMAL': ['Thermal sensors', 'Heater elements', 'Insulation materials', 'Temperature controllers']
        }
        return parts.get(equipment_type, ['Standard replacement parts', 'Diagnostic tools'])

    def _estimate_repair_duration(self, equipment_type: str, severity: str) -> str:
        """Estimate repair duration string"""
        if severity == 'CRITICAL':
            return "4-8 hours (emergency repair)"
        elif severity == 'HIGH':
            return "2-4 hours (priority repair)"
        elif severity == 'MEDIUM':
            return "1-2 hours (scheduled maintenance)"
        else:
            return "30-60 minutes (routine check)"

    def _estimate_repair_duration_hours(self, equipment_type: str, severity: str) -> float:
        """Estimate repair duration in hours"""
        duration_map = {
            'CRITICAL': 6.0,
            'HIGH': 3.0,
            'MEDIUM': 1.5,
            'LOW': 0.5
        }
        return duration_map.get(severity, 1.0)

    def _get_required_skill_level(self, equipment_type: str, severity: str) -> str:
        """Get required skill level for maintenance"""
        if severity == 'CRITICAL':
            return f"Expert {equipment_type.lower()} technician"
        elif severity == 'HIGH':
            return f"Senior {equipment_type.lower()} technician"
        elif severity == 'MEDIUM':
            return f"Qualified {equipment_type.lower()} technician"
        else:
            return "General maintenance technician"