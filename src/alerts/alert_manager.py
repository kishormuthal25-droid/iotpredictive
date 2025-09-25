"""
Alert Manager Module for IoT Anomaly Detection System
Comprehensive alert generation, routing, and management system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import queue
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
from jinja2 import Template
import redis
import hashlib
import time

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = 1   # System failure, immediate action
    HIGH = 2       # Major issue, urgent attention
    MEDIUM = 3     # Significant issue, prompt action
    LOW = 4        # Minor issue, can be scheduled
    INFO = 5       # Informational only


class AlertStatus(Enum):
    """Alert status states"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class AlertType(Enum):
    """Types of alerts"""
    ANOMALY = "anomaly"           # Anomaly detected
    FAILURE = "failure"           # Equipment failure
    THRESHOLD = "threshold"       # Threshold exceeded
    PREDICTION = "prediction"     # Predictive alert
    MAINTENANCE = "maintenance"   # Maintenance required
    SYSTEM = "system"             # System alert
    PERFORMANCE = "performance"   # Performance degradation


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    API = "api"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    alert_type: AlertType
    channels: List[NotificationChannel]
    recipients: List[str]
    cooldown_minutes: int = 15  # Prevent alert flooding
    escalation_rules: Optional[Dict] = None
    suppression_rules: Optional[Dict] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met
        
        Args:
            data: Data to evaluate against condition
            
        Returns:
            True if condition is met
        """
        try:
            # Safe evaluation of condition
            # In production, use a proper expression evaluator
            return eval(self.condition, {"__builtins__": {}}, data)
        except Exception as e:
            logger.error(f"Error evaluating rule {self.rule_id}: {str(e)}")
            return False


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: Optional[str]
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    source: str  # Source system/equipment
    title: str
    description: str
    details: Dict[str, Any]
    affected_equipment: List[str]
    metrics: Dict[str, float]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    work_order_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    child_alert_ids: List[str] = field(default_factory=list)
    notifications_sent: List[Dict] = field(default_factory=list)
    escalation_level: int = 0
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'source': self.source,
            'title': self.title,
            'description': self.description,
            'details': self.details,
            'affected_equipment': self.affected_equipment,
            'metrics': self.metrics,
            'work_order_id': self.work_order_id,
            'escalation_level': self.escalation_level,
            'suppressed': self.suppressed,
            'tags': self.tags
        }


class AlertManager:
    """Main alert management system"""
    
    def __init__(self,
                 redis_config: Optional[Dict] = None,
                 notification_config: Optional[Dict] = None):
        """Initialize Alert Manager
        
        Args:
            redis_config: Redis configuration for distributed alerts
            notification_config: Notification channel configurations
        """
        self.alerts = {}
        self.alert_rules = {}
        self.alert_queue = queue.PriorityQueue()
        self.notification_manager = NotificationManager(notification_config)
        self.deduplicator = AlertDeduplicator()
        self.escalation_manager = EscalationManager()
        self.suppression_manager = SuppressionManager()
        
        # Alert history and metrics
        self.alert_history = deque(maxlen=10000)
        self.alert_metrics = defaultdict(int)
        
        # Redis for distributed alerting
        if redis_config:
            self.redis_client = redis.Redis(**redis_config)
        else:
            self.redis_client = None
            
        # Background executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        self._monitoring_thread = None
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("Initialized Alert Manager")
        
    def create_alert(self,
                    alert_type: AlertType,
                    severity: AlertSeverity,
                    source: str,
                    title: str,
                    description: str,
                    details: Dict[str, Any],
                    affected_equipment: List[str] = None,
                    metrics: Dict[str, float] = None,
                    rule_id: Optional[str] = None) -> Alert:
        """Create a new alert
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            source: Source of alert
            title: Alert title
            description: Alert description
            details: Additional details
            affected_equipment: Affected equipment IDs
            metrics: Associated metrics
            rule_id: Rule that triggered alert
            
        Returns:
            Created alert
        """
        # Check for duplicate alerts
        if self.deduplicator.is_duplicate(source, title, details):
            existing_alert = self.deduplicator.get_existing_alert(source, title)
            if existing_alert:
                logger.info(f"Duplicate alert detected, updating existing: {existing_alert.alert_id}")
                return self._update_existing_alert(existing_alert, details, metrics)
                
        # Create new alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            rule_id=rule_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.NEW,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source=source,
            title=title,
            description=description,
            details=details or {},
            affected_equipment=affected_equipment or [],
            metrics=metrics or {}
        )
        
        # Check suppression rules
        if self.suppression_manager.should_suppress(alert):
            alert.suppressed = True
            alert.suppression_reason = self.suppression_manager.get_suppression_reason(alert)
            alert.status = AlertStatus.SUPPRESSED
            logger.info(f"Alert {alert.alert_id} suppressed: {alert.suppression_reason}")
        else:
            # Add to queue for processing
            priority = severity.value
            self.alert_queue.put((priority, alert.alert_id))
            
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Update metrics
        self.alert_metrics['total_created'] += 1
        self.alert_metrics[f'severity_{severity.name}'] += 1
        
        # Store in Redis if available
        if self.redis_client:
            self._store_alert_redis(alert)
            
        # Process alert asynchronously
        if not alert.suppressed:
            self.executor.submit(self._process_alert, alert)
            
        logger.info(f"Created alert {alert.alert_id}: {title}")
        
        return alert
        
    def _process_alert(self, alert: Alert):
        """Process alert (send notifications, create work orders, etc.)
        
        Args:
            alert: Alert to process
        """
        try:
            # Send notifications
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                # Immediate notification for high priority
                self.notification_manager.send_immediate(alert)
            else:
                # Queue for batch notification
                self.notification_manager.queue_notification(alert)
                
            # Check if work order should be created
            if alert.severity == AlertSeverity.CRITICAL:
                self._create_work_order(alert)
                
            # Start escalation timer if needed
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                self.escalation_manager.start_escalation_timer(alert)
                
            # Correlate with other alerts
            self._correlate_alerts(alert)
            
            # Update alert status
            alert.updated_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {str(e)}")
            
    def acknowledge_alert(self,
                         alert_id: str,
                         acknowledged_by: str,
                         notes: str = "") -> bool:
        """Acknowledge an alert
        
        Args:
            alert_id: Alert ID
            acknowledged_by: User acknowledging
            notes: Additional notes
            
        Returns:
            Success flag
        """
        if alert_id not in self.alerts:
            logger.error(f"Alert {alert_id} not found")
            return False
            
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        alert.updated_at = datetime.now()
        
        if notes:
            alert.details['acknowledgment_notes'] = notes
            
        # Stop escalation
        self.escalation_manager.stop_escalation(alert_id)
        
        # Send acknowledgment notification
        self.notification_manager.send_acknowledgment(alert, acknowledged_by)
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return True
        
    def resolve_alert(self,
                     alert_id: str,
                     resolved_by: str,
                     resolution: str) -> bool:
        """Resolve an alert
        
        Args:
            alert_id: Alert ID
            resolved_by: User resolving
            resolution: Resolution description
            
        Returns:
            Success flag
        """
        if alert_id not in self.alerts:
            logger.error(f"Alert {alert_id} not found")
            return False
            
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()
        alert.details['resolution'] = resolution
        
        # Calculate resolution time
        resolution_time = (alert.resolved_at - alert.created_at).total_seconds() / 60
        alert.metrics['resolution_time_minutes'] = resolution_time
        
        # Update metrics
        self.alert_metrics['total_resolved'] += 1
        self.alert_metrics['avg_resolution_time'] = (
            (self.alert_metrics.get('avg_resolution_time', 0) * 
             (self.alert_metrics['total_resolved'] - 1) + resolution_time) /
            self.alert_metrics['total_resolved']
        )
        
        # Send resolution notification
        self.notification_manager.send_resolution(alert, resolved_by, resolution)
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        
        return True
        
    def escalate_alert(self, alert_id: str) -> bool:
        """Escalate an alert to next level
        
        Args:
            alert_id: Alert ID to escalate
            
        Returns:
            Success flag
        """
        if alert_id not in self.alerts:
            return False
            
        alert = self.alerts[alert_id]
        alert.escalation_level += 1
        alert.status = AlertStatus.ESCALATED
        alert.updated_at = datetime.now()
        
        # Get escalation recipients
        recipients = self.escalation_manager.get_escalation_recipients(
            alert.escalation_level,
            alert.severity
        )
        
        # Send escalation notifications
        self.notification_manager.send_escalation(alert, recipients)
        
        logger.info(f"Alert {alert_id} escalated to level {alert.escalation_level}")
        
        return True
        
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule
        
        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against data
        
        Args:
            data: Data to evaluate
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
                
            # Check cooldown
            if self._is_in_cooldown(rule.rule_id):
                continue
                
            # Evaluate rule
            if rule.evaluate(data):
                # Create alert from rule
                alert = self.create_alert(
                    alert_type=rule.alert_type,
                    severity=rule.severity,
                    source=data.get('source', 'unknown'),
                    title=rule.name,
                    description=rule.description,
                    details=data,
                    affected_equipment=data.get('equipment', []),
                    metrics=data.get('metrics', {}),
                    rule_id=rule.rule_id
                )
                
                triggered_alerts.append(alert)
                
                # Set cooldown
                self._set_cooldown(rule.rule_id, rule.cooldown_minutes)
                
        return triggered_alerts
        
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts
        
        Returns:
            List of active alerts
        """
        return [
            alert for alert in self.alerts.values()
            if alert.status not in [AlertStatus.RESOLVED, AlertStatus.CLOSED, AlertStatus.SUPPRESSED]
        ]
        
    def get_alerts_by_equipment(self, equipment_id: str) -> List[Alert]:
        """Get alerts for specific equipment
        
        Args:
            equipment_id: Equipment ID
            
        Returns:
            List of alerts
        """
        return [
            alert for alert in self.alerts.values()
            if equipment_id in alert.affected_equipment
        ]
        
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity
        
        Args:
            severity: Alert severity
            
        Returns:
            List of alerts
        """
        return [
            alert for alert in self.alerts.values()
            if alert.severity == severity
        ]
        
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics
        
        Returns:
            Alert statistics
        """
        active_alerts = self.get_active_alerts()
        
        stats = {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'by_severity': defaultdict(int),
            'by_status': defaultdict(int),
            'by_type': defaultdict(int),
            'metrics': dict(self.alert_metrics)
        }
        
        for alert in self.alerts.values():
            stats['by_severity'][alert.severity.name] += 1
            stats['by_status'][alert.status.value] += 1
            stats['by_type'][alert.alert_type.value] += 1
            
        return stats
        
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"ALT-{timestamp}-{unique_id}"
        
    def _update_existing_alert(self,
                              alert: Alert,
                              new_details: Dict[str, Any],
                              new_metrics: Dict[str, float]) -> Alert:
        """Update existing alert with new information
        
        Args:
            alert: Existing alert
            new_details: New details to add
            new_metrics: New metrics to add
            
        Returns:
            Updated alert
        """
        alert.updated_at = datetime.now()
        alert.details.update(new_details)
        alert.metrics.update(new_metrics)
        
        # Increment occurrence count
        alert.details['occurrence_count'] = alert.details.get('occurrence_count', 1) + 1
        
        return alert
        
    def _correlate_alerts(self, alert: Alert):
        """Correlate alert with existing alerts
        
        Args:
            alert: Alert to correlate
        """
        # Find related alerts
        related_alerts = []
        
        for existing_alert in self.get_active_alerts():
            if existing_alert.alert_id == alert.alert_id:
                continue
                
            # Check if same equipment
            if set(alert.affected_equipment) & set(existing_alert.affected_equipment):
                related_alerts.append(existing_alert)
                continue
                
            # Check if same source
            if alert.source == existing_alert.source:
                related_alerts.append(existing_alert)
                
        # Link related alerts
        for related in related_alerts:
            if alert.alert_id not in related.child_alert_ids:
                related.child_alert_ids.append(alert.alert_id)
            alert.parent_alert_id = related.alert_id
            
    def _create_work_order(self, alert: Alert):
        """Create work order from alert
        
        Args:
            alert: Alert requiring work order
        """
        # This would integrate with work_order_manager
        logger.info(f"Creating work order for alert {alert.alert_id}")
        # Implementation would call work_order_manager.create_work_order_from_alert()
        
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period
        
        Args:
            rule_id: Rule ID to check
            
        Returns:
            True if in cooldown
        """
        # Check Redis or local cache for cooldown
        if self.redis_client:
            return self.redis_client.exists(f"cooldown:{rule_id}")
        return False
        
    def _set_cooldown(self, rule_id: str, minutes: int):
        """Set cooldown for rule
        
        Args:
            rule_id: Rule ID
            minutes: Cooldown duration
        """
        if self.redis_client:
            self.redis_client.setex(f"cooldown:{rule_id}", minutes * 60, "1")
            
    def _store_alert_redis(self, alert: Alert):
        """Store alert in Redis
        
        Args:
            alert: Alert to store
        """
        try:
            alert_data = json.dumps(alert.to_dict())
            self.redis_client.hset("alerts", alert.alert_id, alert_data)
            
            # Set expiry for resolved alerts
            if alert.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
                self.redis_client.expire(f"alert:{alert.alert_id}", 86400)  # 24 hours
        except Exception as e:
            logger.error(f"Error storing alert in Redis: {str(e)}")
            
    def _start_monitoring(self):
        """Start background monitoring thread"""
        self._monitoring_thread = threading.Thread(target=self._monitor_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Process alert queue
                if not self.alert_queue.empty():
                    priority, alert_id = self.alert_queue.get(timeout=1)
                    if alert_id in self.alerts:
                        alert = self.alerts[alert_id]
                        self._process_alert(alert)
                        
                # Check for escalations
                self.escalation_manager.check_escalations(self.alerts)
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                
            time.sleep(5)
            
    def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for alert_id in list(self.alerts.keys()):
            alert = self.alerts[alert_id]
            if (alert.status == AlertStatus.CLOSED and 
                alert.updated_at < cutoff_date):
                del self.alerts[alert_id]
                logger.debug(f"Cleaned up old alert: {alert_id}")


class NotificationManager:
    """Manage alert notifications across channels"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Notification Manager
        
        Args:
            config: Channel configurations
        """
        self.config = config or {}
        self.channels = {}
        self.notification_queue = queue.Queue()
        self.notification_history = deque(maxlen=1000)
        
        # Initialize channels
        self._initialize_channels()
        
        # Start notification worker
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.executor.submit(self._notification_worker)
        
    def _initialize_channels(self):
        """Initialize notification channels"""
        if 'email' in self.config:
            self.channels[NotificationChannel.EMAIL] = EmailChannel(self.config['email'])
        if 'slack' in self.config:
            self.channels[NotificationChannel.SLACK] = SlackChannel(self.config['slack'])
        if 'webhook' in self.config:
            self.channels[NotificationChannel.WEBHOOK] = WebhookChannel(self.config['webhook'])
            
    def send_immediate(self, alert: Alert):
        """Send immediate notification for critical alerts
        
        Args:
            alert: Alert to notify
        """
        # Determine channels based on severity
        channels = self._get_channels_for_severity(alert.severity)
        
        for channel in channels:
            if channel in self.channels:
                try:
                    self.channels[channel].send(alert)
                    self._record_notification(alert, channel, 'success')
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {str(e)}")
                    self._record_notification(alert, channel, 'failed', str(e))
                    
    def queue_notification(self, alert: Alert):
        """Queue notification for batch sending
        
        Args:
            alert: Alert to queue
        """
        self.notification_queue.put(alert)
        
    def send_acknowledgment(self, alert: Alert, acknowledged_by: str):
        """Send acknowledgment notification
        
        Args:
            alert: Acknowledged alert
            acknowledged_by: User who acknowledged
        """
        message = f"Alert {alert.alert_id} acknowledged by {acknowledged_by}"
        self._send_update(alert, message, [NotificationChannel.EMAIL])
        
    def send_resolution(self, alert: Alert, resolved_by: str, resolution: str):
        """Send resolution notification
        
        Args:
            alert: Resolved alert
            resolved_by: User who resolved
            resolution: Resolution description
        """
        message = f"Alert {alert.alert_id} resolved by {resolved_by}: {resolution}"
        self._send_update(alert, message, [NotificationChannel.EMAIL])
        
    def send_escalation(self, alert: Alert, recipients: List[str]):
        """Send escalation notification
        
        Args:
            alert: Escalated alert
            recipients: Escalation recipients
        """
        message = f"ESCALATION: Alert {alert.alert_id} escalated to level {alert.escalation_level}"
        
        # Send to all critical channels
        for channel in [NotificationChannel.EMAIL, NotificationChannel.SMS, NotificationChannel.SLACK]:
            if channel in self.channels:
                self.channels[channel].send_escalation(alert, recipients)
                
    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[NotificationChannel]:
        """Get notification channels based on severity
        
        Args:
            severity: Alert severity
            
        Returns:
            List of channels
        """
        if severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.SMS, 
                   NotificationChannel.SLACK, NotificationChannel.PAGERDUTY]
        elif severity == AlertSeverity.HIGH:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK]
        elif severity == AlertSeverity.MEDIUM:
            return [NotificationChannel.EMAIL]
        else:
            return [NotificationChannel.DASHBOARD]
            
    def _send_update(self, alert: Alert, message: str, channels: List[NotificationChannel]):
        """Send update notification
        
        Args:
            alert: Alert
            message: Update message
            channels: Channels to notify
        """
        for channel in channels:
            if channel in self.channels:
                self.channels[channel].send_update(alert, message)
                
    def _notification_worker(self):
        """Background worker for processing notification queue"""
        while True:
            try:
                alert = self.notification_queue.get(timeout=30)
                self.send_immediate(alert)
            except queue.Empty:
                # Send batch notifications if any
                pass
            except Exception as e:
                logger.error(f"Notification worker error: {str(e)}")
                
    def _record_notification(self, alert: Alert, channel: NotificationChannel, 
                           status: str, error: str = None):
        """Record notification in history
        
        Args:
            alert: Alert
            channel: Notification channel
            status: Success or failed
            error: Error message if failed
        """
        record = {
            'alert_id': alert.alert_id,
            'channel': channel.value,
            'timestamp': datetime.now(),
            'status': status,
            'error': error
        }
        
        self.notification_history.append(record)
        alert.notifications_sent.append(record)


class EmailChannel:
    """Email notification channel"""
    
    def __init__(self, config: Dict):
        """Initialize Email Channel
        
        Args:
            config: Email configuration
        """
        self.config = config
        self.template = Template("""
        Alert Notification
        ==================
        
        Alert ID: {{ alert.alert_id }}
        Severity: {{ alert.severity.name }}
        Type: {{ alert.alert_type.value }}
        
        Title: {{ alert.title }}
        Description: {{ alert.description }}
        
        Source: {{ alert.source }}
        Affected Equipment: {{ alert.affected_equipment | join(', ') }}
        
        Created: {{ alert.created_at }}
        Status: {{ alert.status.value }}
        
        {% if alert.metrics %}
        Metrics:
        {% for key, value in alert.metrics.items() %}
        - {{ key }}: {{ value }}
        {% endfor %}
        {% endif %}
        
        Please log in to the system for more details and to take action.
        """)
        
    def send(self, alert: Alert):
        """Send email notification
        
        Args:
            alert: Alert to send
        """
        msg = MIMEMultipart()
        msg['From'] = self.config.get('from_email')
        msg['To'] = ', '.join(self.config.get('to_emails', []))
        msg['Subject'] = f"[{alert.severity.name}] Alert: {alert.title}"
        
        body = self.template.render(alert=alert)
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(self.config.get('smtp_host'), self.config.get('smtp_port', 587)) as server:
            if self.config.get('use_tls', True):
                server.starttls()
            if self.config.get('username'):
                server.login(self.config.get('username'), self.config.get('password'))
            server.send_message(msg)
            
    def send_update(self, alert: Alert, message: str):
        """Send update email"""
        # Similar implementation with update template
        pass
        
    def send_escalation(self, alert: Alert, recipients: List[str]):
        """Send escalation email"""
        # Similar implementation with escalation template
        pass


class SlackChannel:
    """Slack notification channel"""
    
    def __init__(self, config: Dict):
        """Initialize Slack Channel
        
        Args:
            config: Slack configuration
        """
        self.config = config
        self.webhook_url = config.get('webhook_url')
        
    def send(self, alert: Alert):
        """Send Slack notification
        
        Args:
            alert: Alert to send
        """
        payload = {
            'text': f"*{alert.severity.name} Alert*: {alert.title}",
            'attachments': [{
                'color': self._get_color(alert.severity),
                'fields': [
                    {'title': 'Alert ID', 'value': alert.alert_id, 'short': True},
                    {'title': 'Type', 'value': alert.alert_type.value, 'short': True},
                    {'title': 'Source', 'value': alert.source, 'short': True},
                    {'title': 'Status', 'value': alert.status.value, 'short': True},
                    {'title': 'Description', 'value': alert.description, 'short': False}
                ],
                'timestamp': int(alert.created_at.timestamp())
            }]
        }
        
        response = requests.post(self.webhook_url, json=payload)
        response.raise_for_status()
        
    def _get_color(self, severity: AlertSeverity) -> str:
        """Get color for severity"""
        colors = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: '#FFA500',
            AlertSeverity.LOW: 'good',
            AlertSeverity.INFO: '#808080'
        }
        return colors.get(severity, '#808080')


class WebhookChannel:
    """Generic webhook notification channel"""
    
    def __init__(self, config: Dict):
        """Initialize Webhook Channel
        
        Args:
            config: Webhook configuration
        """
        self.config = config
        self.url = config.get('url')
        self.headers = config.get('headers', {})
        
    async def send_async(self, alert: Alert):
        """Send webhook notification asynchronously
        
        Args:
            alert: Alert to send
        """
        async with aiohttp.ClientSession() as session:
            payload = alert.to_dict()
            async with session.post(self.url, json=payload, headers=self.headers) as response:
                response.raise_for_status()
                
    def send(self, alert: Alert):
        """Send webhook notification
        
        Args:
            alert: Alert to send
        """
        payload = alert.to_dict()
        response = requests.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()


class AlertDeduplicator:
    """Deduplicate similar alerts"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize Alert Deduplicator
        
        Args:
            similarity_threshold: Threshold for considering alerts duplicate
        """
        self.similarity_threshold = similarity_threshold
        self.alert_cache = {}
        self.hash_cache = {}
        
    def is_duplicate(self, source: str, title: str, details: Dict) -> bool:
        """Check if alert is duplicate
        
        Args:
            source: Alert source
            title: Alert title
            details: Alert details
            
        Returns:
            True if duplicate
        """
        alert_hash = self._compute_hash(source, title, details)
        
        # Check exact match
        if alert_hash in self.hash_cache:
            last_seen = self.hash_cache[alert_hash]
            if (datetime.now() - last_seen).seconds < 300:  # 5 minutes
                return True
                
        # Check similarity
        for cached_hash, cached_alert in self.alert_cache.items():
            similarity = self._compute_similarity(
                (source, title, details),
                cached_alert
            )
            if similarity > self.similarity_threshold:
                return True
                
        # Update cache
        self.hash_cache[alert_hash] = datetime.now()
        self.alert_cache[alert_hash] = (source, title, details)
        
        # Clean old entries
        self._cleanup_cache()
        
        return False
        
    def get_existing_alert(self, source: str, title: str) -> Optional[Alert]:
        """Get existing similar alert
        
        Args:
            source: Alert source
            title: Alert title
            
        Returns:
            Existing alert if found
        """
        # Implementation to find and return existing alert
        return None
        
    def _compute_hash(self, source: str, title: str, details: Dict) -> str:
        """Compute hash for alert
        
        Args:
            source: Alert source
            title: Alert title
            details: Alert details
            
        Returns:
            Hash string
        """
        content = f"{source}:{title}:{json.dumps(details, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _compute_similarity(self, alert1: Tuple, alert2: Tuple) -> float:
        """Compute similarity between alerts
        
        Args:
            alert1: First alert tuple
            alert2: Second alert tuple
            
        Returns:
            Similarity score (0-1)
        """
        # Simple similarity based on source and title
        source_match = 1.0 if alert1[0] == alert2[0] else 0.0
        
        # Title similarity (simple word matching)
        words1 = set(alert1[1].lower().split())
        words2 = set(alert2[1].lower().split())
        title_similarity = len(words1 & words2) / max(len(words1), len(words2)) if words1 or words2 else 0
        
        return (source_match * 0.5 + title_similarity * 0.5)
        
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        cutoff_time = datetime.now() - timedelta(minutes=30)
        
        # Clean hash cache
        self.hash_cache = {
            h: t for h, t in self.hash_cache.items()
            if t > cutoff_time
        }


class EscalationManager:
    """Manage alert escalations"""
    
    def __init__(self):
        """Initialize Escalation Manager"""
        self.escalation_timers = {}
        self.escalation_rules = self._default_escalation_rules()
        
    def start_escalation_timer(self, alert: Alert):
        """Start escalation timer for alert
        
        Args:
            alert: Alert to escalate
        """
        escalation_time = self._get_escalation_time(alert.severity, alert.escalation_level)
        
        timer = threading.Timer(
            escalation_time,
            self._escalate_callback,
            args=[alert.alert_id]
        )
        timer.start()
        
        self.escalation_timers[alert.alert_id] = timer
        
    def stop_escalation(self, alert_id: str):
        """Stop escalation timer
        
        Args:
            alert_id: Alert ID
        """
        if alert_id in self.escalation_timers:
            self.escalation_timers[alert_id].cancel()
            del self.escalation_timers[alert_id]
            
    def check_escalations(self, alerts: Dict[str, Alert]):
        """Check for alerts requiring escalation
        
        Args:
            alerts: Current alerts
        """
        for alert_id, alert in alerts.items():
            if alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED]:
                # Check if escalation needed
                time_since_creation = (datetime.now() - alert.created_at).seconds / 60
                
                if self._should_escalate(alert.severity, alert.escalation_level, time_since_creation):
                    # Trigger escalation
                    self._escalate_callback(alert_id)
                    
    def get_escalation_recipients(self, level: int, severity: AlertSeverity) -> List[str]:
        """Get recipients for escalation level
        
        Args:
            level: Escalation level
            severity: Alert severity
            
        Returns:
            List of recipients
        """
        rules = self.escalation_rules.get(severity.name, {})
        return rules.get(f'level_{level}', [])
        
    def _get_escalation_time(self, severity: AlertSeverity, level: int) -> int:
        """Get escalation time in seconds
        
        Args:
            severity: Alert severity
            level: Current escalation level
            
        Returns:
            Time in seconds
        """
        times = {
            AlertSeverity.CRITICAL: [300, 600, 900],     # 5, 10, 15 minutes
            AlertSeverity.HIGH: [900, 1800, 3600],       # 15, 30, 60 minutes
            AlertSeverity.MEDIUM: [3600, 7200, 14400],   # 1, 2, 4 hours
            AlertSeverity.LOW: [14400, 28800, 86400]     # 4, 8, 24 hours
        }
        
        severity_times = times.get(severity, [3600])
        return severity_times[min(level, len(severity_times) - 1)]
        
    def _should_escalate(self, severity: AlertSeverity, level: int, minutes_elapsed: float) -> bool:
        """Check if escalation is needed
        
        Args:
            severity: Alert severity
            level: Current level
            minutes_elapsed: Minutes since creation
            
        Returns:
            True if should escalate
        """
        thresholds = {
            AlertSeverity.CRITICAL: [5, 10, 15],
            AlertSeverity.HIGH: [15, 30, 60],
            AlertSeverity.MEDIUM: [60, 120, 240],
            AlertSeverity.LOW: [240, 480, 1440]
        }
        
        severity_thresholds = thresholds.get(severity, [60])
        threshold = severity_thresholds[min(level, len(severity_thresholds) - 1)]
        
        return minutes_elapsed >= threshold
        
    def _escalate_callback(self, alert_id: str):
        """Callback for escalation timer
        
        Args:
            alert_id: Alert to escalate
        """
        # This would call back to AlertManager.escalate_alert()
        logger.info(f"Escalation triggered for alert {alert_id}")
        
    def _default_escalation_rules(self) -> Dict:
        """Get default escalation rules
        
        Returns:
            Escalation rules
        """
        return {
            'CRITICAL': {
                'level_1': ['supervisor', 'manager'],
                'level_2': ['director', 'vp'],
                'level_3': ['cto', 'ceo']
            },
            'HIGH': {
                'level_1': ['supervisor'],
                'level_2': ['manager'],
                'level_3': ['director']
            }
        }


class SuppressionManager:
    """Manage alert suppression rules"""
    
    def __init__(self):
        """Initialize Suppression Manager"""
        self.suppression_rules = []
        self.maintenance_windows = []
        
    def should_suppress(self, alert: Alert) -> bool:
        """Check if alert should be suppressed
        
        Args:
            alert: Alert to check
            
        Returns:
            True if should suppress
        """
        # Check maintenance windows
        if self._in_maintenance_window(alert):
            return True
            
        # Check suppression rules
        for rule in self.suppression_rules:
            if self._matches_rule(alert, rule):
                return True
                
        return False
        
    def get_suppression_reason(self, alert: Alert) -> str:
        """Get reason for suppression
        
        Args:
            alert: Suppressed alert
            
        Returns:
            Suppression reason
        """
        if self._in_maintenance_window(alert):
            return "In maintenance window"
            
        for rule in self.suppression_rules:
            if self._matches_rule(alert, rule):
                return rule.get('reason', 'Matched suppression rule')
                
        return "Unknown"
        
    def add_maintenance_window(self, 
                              equipment: List[str],
                              start_time: datetime,
                              end_time: datetime):
        """Add maintenance window
        
        Args:
            equipment: Equipment in maintenance
            start_time: Window start
            end_time: Window end
        """
        self.maintenance_windows.append({
            'equipment': equipment,
            'start': start_time,
            'end': end_time
        })
        
    def _in_maintenance_window(self, alert: Alert) -> bool:
        """Check if alert is in maintenance window
        
        Args:
            alert: Alert to check
            
        Returns:
            True if in maintenance window
        """
        current_time = datetime.now()
        
        for window in self.maintenance_windows:
            if (window['start'] <= current_time <= window['end'] and
                any(eq in window['equipment'] for eq in alert.affected_equipment)):
                return True
                
        return False
        
    def _matches_rule(self, alert: Alert, rule: Dict) -> bool:
        """Check if alert matches suppression rule
        
        Args:
            alert: Alert to check
            rule: Suppression rule
            
        Returns:
            True if matches
        """
        # Check various rule conditions
        if 'source' in rule and alert.source != rule['source']:
            return False
        if 'type' in rule and alert.alert_type.value != rule['type']:
            return False
        if 'severity_below' in rule and alert.severity.value < rule['severity_below']:
            return False
            
        return True