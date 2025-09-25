"""
Email Sender Module for IoT Anomaly Detection System
Advanced email communication with templates, attachments, and scheduling
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import base64
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiosmtplib
from jinja2 import Template, Environment, FileSystemLoader
import premailer
import markdown
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
import os
from enum import Enum
import time
import hashlib
import re

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class EmailPriority(Enum):
    """Email priority levels"""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BATCH = 5


class EmailType(Enum):
    """Types of emails"""
    ALERT = "alert"
    REPORT = "report"
    WORK_ORDER = "work_order"
    NOTIFICATION = "notification"
    SUMMARY = "summary"
    ESCALATION = "escalation"
    CONFIRMATION = "confirmation"


@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_host: str
    smtp_port: int = 587
    use_tls: bool = True
    use_ssl: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    from_email: str = "iot-system@company.com"
    from_name: str = "IoT Anomaly Detection System"
    reply_to: Optional[str] = None
    max_retries: int = 3
    retry_delay: int = 60
    batch_size: int = 50
    rate_limit: int = 10  # emails per second
    timeout: int = 30
    
    # Template settings
    template_dir: str = "templates/email"
    use_html: bool = True
    inline_css: bool = True
    
    # Features
    track_opens: bool = False
    track_clicks: bool = False
    unsubscribe_link: bool = True


@dataclass
class EmailMessage:
    """Email message structure"""
    to_emails: List[str]
    subject: str
    body_text: str
    body_html: Optional[str] = None
    cc_emails: List[str] = field(default_factory=list)
    bcc_emails: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None
    priority: EmailPriority = EmailPriority.NORMAL
    email_type: EmailType = EmailType.NOTIFICATION
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    inline_images: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduled_time: Optional[datetime] = None
    retry_count: int = 0
    
    def get_recipients_count(self) -> int:
        """Get total number of recipients"""
        return len(self.to_emails) + len(self.cc_emails) + len(self.bcc_emails)


class EmailSender:
    """Main email sending service"""
    
    def __init__(self, config: Optional[EmailConfig] = None):
        """Initialize Email Sender
        
        Args:
            config: Email configuration
        """
        self.config = config or self._get_default_config()
        self.template_engine = EmailTemplateEngine(self.config.template_dir)
        self.queue_manager = EmailQueueManager()
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        
        # Email statistics
        self.stats = {
            'sent': 0,
            'failed': 0,
            'queued': 0,
            'retried': 0
        }
        
        # History tracking
        self.send_history = deque(maxlen=1000)
        
        # SMTP connection pool
        self.smtp_pool = SMTPConnectionPool(self.config)
        
        # Background executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event()
        
        # Start background workers
        self._start_workers()
        
        logger.info("Initialized Email Sender")
        
    def send_alert_email(self,
                        alert_data: Dict[str, Any],
                        recipients: List[str],
                        priority: EmailPriority = EmailPriority.HIGH) -> bool:
        """Send alert notification email
        
        Args:
            alert_data: Alert information
            recipients: Email recipients
            priority: Email priority
            
        Returns:
            Success flag
        """
        # Generate email content
        subject = self._generate_alert_subject(alert_data)
        body_html = self.template_engine.render_alert_template(alert_data)
        body_text = self._html_to_text(body_html)
        
        # Create charts if metrics available
        attachments = []
        if 'metrics' in alert_data:
            chart_image = self._create_alert_chart(alert_data['metrics'])
            if chart_image:
                attachments.append({
                    'filename': 'alert_metrics.png',
                    'content': chart_image,
                    'content_type': 'image/png'
                })
                
        # Create email message
        message = EmailMessage(
            to_emails=recipients,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority=priority,
            email_type=EmailType.ALERT,
            attachments=attachments,
            metadata={'alert_id': alert_data.get('alert_id')}
        )
        
        # Send based on priority
        if priority == EmailPriority.URGENT:
            return self.send_immediate(message)
        else:
            return self.queue_email(message)
            
    def send_work_order_email(self,
                             work_order: Dict[str, Any],
                             recipients: List[str]) -> bool:
        """Send work order notification email
        
        Args:
            work_order: Work order details
            recipients: Email recipients
            
        Returns:
            Success flag
        """
        # Generate email content
        subject = f"Work Order {work_order.get('order_id')}: {work_order.get('title', 'Maintenance Required')}"
        body_html = self.template_engine.render_work_order_template(work_order)
        body_text = self._html_to_text(body_html)
        
        # Add calendar attachment for scheduled work
        attachments = []
        if work_order.get('scheduled_start'):
            ics_content = self._create_calendar_event(work_order)
            attachments.append({
                'filename': 'work_order.ics',
                'content': ics_content.encode(),
                'content_type': 'text/calendar'
            })
            
        # Create email message
        message = EmailMessage(
            to_emails=recipients,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority=EmailPriority.HIGH,
            email_type=EmailType.WORK_ORDER,
            attachments=attachments,
            metadata={'work_order_id': work_order.get('order_id')}
        )
        
        return self.send_immediate(message)
        
    def send_report_email(self,
                         report_data: Dict[str, Any],
                         recipients: List[str],
                         attachments: List[Dict] = None) -> bool:
        """Send report email with attachments
        
        Args:
            report_data: Report information
            recipients: Email recipients
            attachments: File attachments
            
        Returns:
            Success flag
        """
        # Generate email content
        subject = report_data.get('title', 'IoT System Report')
        body_html = self.template_engine.render_report_template(report_data)
        body_text = self._html_to_text(body_html)
        
        # Process attachments
        email_attachments = attachments or []
        
        # Add report charts
        if 'charts' in report_data:
            for i, chart in enumerate(report_data['charts']):
                chart_image = self._create_report_chart(chart)
                email_attachments.append({
                    'filename': f'chart_{i+1}.png',
                    'content': chart_image,
                    'content_type': 'image/png',
                    'cid': f'chart_{i+1}'  # For inline display
                })
                
        # Create email message
        message = EmailMessage(
            to_emails=recipients,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority=EmailPriority.NORMAL,
            email_type=EmailType.REPORT,
            attachments=email_attachments,
            metadata={'report_type': report_data.get('type')}
        )
        
        return self.queue_email(message)
        
    def send_summary_email(self,
                          summary_data: Dict[str, Any],
                          recipients: List[str],
                          period: str = "daily") -> bool:
        """Send periodic summary email
        
        Args:
            summary_data: Summary information
            recipients: Email recipients
            period: Summary period (daily, weekly, monthly)
            
        Returns:
            Success flag
        """
        # Generate email content
        subject = f"{period.capitalize()} Summary - {datetime.now().strftime('%Y-%m-%d')}"
        body_html = self.template_engine.render_summary_template(summary_data, period)
        body_text = self._html_to_text(body_html)
        
        # Create summary dashboard image
        dashboard_image = self._create_summary_dashboard(summary_data)
        
        # Create email message
        message = EmailMessage(
            to_emails=recipients,
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            priority=EmailPriority.LOW,
            email_type=EmailType.SUMMARY,
            inline_images=[{
                'filename': 'dashboard.png',
                'content': dashboard_image,
                'content_type': 'image/png',
                'cid': 'dashboard'
            }] if dashboard_image else [],
            metadata={'period': period}
        )
        
        return self.queue_email(message)
        
    def send_immediate(self, message: EmailMessage) -> bool:
        """Send email immediately
        
        Args:
            message: Email message
            
        Returns:
            Success flag
        """
        try:
            # Apply rate limiting
            self.rate_limiter.acquire()
            
            # Get SMTP connection
            smtp = self.smtp_pool.get_connection()
            
            # Build MIME message
            mime_message = self._build_mime_message(message)
            
            # Send email
            smtp.send_message(mime_message)
            
            # Update statistics
            self.stats['sent'] += message.get_recipients_count()
            
            # Record in history
            self._record_send(message, 'success')
            
            logger.info(f"Email sent successfully to {len(message.to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            self.stats['failed'] += 1
            
            # Retry if configured
            if message.retry_count < self.config.max_retries:
                message.retry_count += 1
                self.queue_email(message)
                self.stats['retried'] += 1
                
            self._record_send(message, 'failed', str(e))
            return False
            
    def queue_email(self, message: EmailMessage) -> bool:
        """Queue email for batch sending
        
        Args:
            message: Email message
            
        Returns:
            Success flag
        """
        try:
            self.queue_manager.add_to_queue(message)
            self.stats['queued'] += 1
            logger.info(f"Email queued for {len(message.to_emails)} recipients")
            return True
        except Exception as e:
            logger.error(f"Failed to queue email: {str(e)}")
            return False
            
    def send_batch(self, messages: List[EmailMessage]) -> Dict[str, int]:
        """Send batch of emails
        
        Args:
            messages: List of email messages
            
        Returns:
            Results dictionary
        """
        results = {'sent': 0, 'failed': 0}
        
        # Group by priority
        priority_groups = defaultdict(list)
        for msg in messages:
            priority_groups[msg.priority].append(msg)
            
        # Send in priority order
        for priority in sorted(priority_groups.keys()):
            for message in priority_groups[priority]:
                if self.send_immediate(message):
                    results['sent'] += 1
                else:
                    results['failed'] += 1
                    
                # Rate limiting between emails
                time.sleep(1 / self.config.rate_limit)
                
        return results
        
    def schedule_email(self,
                      message: EmailMessage,
                      send_time: datetime) -> bool:
        """Schedule email for future sending
        
        Args:
            message: Email message
            send_time: When to send
            
        Returns:
            Success flag
        """
        message.scheduled_time = send_time
        return self.queue_manager.schedule_email(message)
        
    def _build_mime_message(self, message: EmailMessage) -> MIMEMultipart:
        """Build MIME message from EmailMessage
        
        Args:
            message: Email message
            
        Returns:
            MIME message
        """
        # Create multipart message
        mime_msg = MIMEMultipart('mixed')
        
        # Set headers
        mime_msg['From'] = f"{self.config.from_name} <{self.config.from_email}>"
        mime_msg['To'] = ', '.join(message.to_emails)
        mime_msg['Subject'] = message.subject
        
        if message.cc_emails:
            mime_msg['Cc'] = ', '.join(message.cc_emails)
        if message.bcc_emails:
            mime_msg['Bcc'] = ', '.join(message.bcc_emails)
        if message.reply_to:
            mime_msg['Reply-To'] = message.reply_to
            
        # Add custom headers
        for key, value in message.headers.items():
            mime_msg[key] = value
            
        # Add priority header
        if message.priority == EmailPriority.URGENT:
            mime_msg['X-Priority'] = '1'
            mime_msg['Importance'] = 'high'
            
        # Create body container
        body_container = MIMEMultipart('alternative')
        
        # Add text part
        text_part = MIMEText(message.body_text, 'plain', 'utf-8')
        body_container.attach(text_part)
        
        # Add HTML part if available
        if message.body_html:
            # Process HTML (inline CSS, etc.)
            processed_html = self._process_html(message.body_html)
            
            # Handle inline images
            if message.inline_images:
                html_container = MIMEMultipart('related')
                html_part = MIMEText(processed_html, 'html', 'utf-8')
                html_container.attach(html_part)
                
                # Add inline images
                for img in message.inline_images:
                    img_part = MIMEImage(img['content'])
                    img_part.add_header('Content-ID', f"<{img['cid']}>")
                    img_part.add_header('Content-Disposition', 'inline')
                    html_container.attach(img_part)
                    
                body_container.attach(html_container)
            else:
                html_part = MIMEText(processed_html, 'html', 'utf-8')
                body_container.attach(html_part)
                
        mime_msg.attach(body_container)
        
        # Add attachments
        for attachment in message.attachments:
            self._add_attachment(mime_msg, attachment)
            
        # Add tracking pixel if configured
        if self.config.track_opens:
            tracking_pixel = self._create_tracking_pixel(message)
            if tracking_pixel:
                mime_msg.attach(tracking_pixel)
                
        return mime_msg
        
    def _add_attachment(self, mime_msg: MIMEMultipart, attachment: Dict):
        """Add attachment to MIME message
        
        Args:
            mime_msg: MIME message
            attachment: Attachment data
        """
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment['content'])
        encoders.encode_base64(part)
        
        part.add_header(
            'Content-Disposition',
            f"attachment; filename= {attachment['filename']}"
        )
        
        if 'content_type' in attachment:
            part.set_type(attachment['content_type'])
            
        mime_msg.attach(part)
        
    def _process_html(self, html: str) -> str:
        """Process HTML content
        
        Args:
            html: HTML content
            
        Returns:
            Processed HTML
        """
        # Inline CSS if configured
        if self.config.inline_css:
            html = premailer.transform(html)
            
        # Add unsubscribe link if configured
        if self.config.unsubscribe_link:
            html = self._add_unsubscribe_link(html)
            
        return html
        
    def _generate_alert_subject(self, alert_data: Dict) -> str:
        """Generate alert email subject
        
        Args:
            alert_data: Alert information
            
        Returns:
            Email subject
        """
        severity = alert_data.get('severity', 'INFO')
        source = alert_data.get('source', 'Unknown')
        title = alert_data.get('title', 'Alert')
        
        return f"[{severity}] Alert from {source}: {title}"
        
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text
        
        Args:
            html: HTML content
            
        Returns:
            Plain text
        """
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', html)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
        
    def _create_alert_chart(self, metrics: Dict) -> bytes:
        """Create chart for alert metrics
        
        Args:
            metrics: Metrics data
            
        Returns:
            Chart image bytes
        """
        try:
            fig = go.Figure()
            
            # Add metric traces
            for name, values in metrics.items():
                if isinstance(values, list):
                    fig.add_trace(go.Scatter(
                        y=values,
                        mode='lines',
                        name=name
                    ))
                    
            fig.update_layout(
                title="Alert Metrics",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white",
                height=400,
                width=600
            )
            
            # Convert to image bytes
            return pio.to_image(fig, format='png')
            
        except Exception as e:
            logger.error(f"Failed to create alert chart: {str(e)}")
            return None
            
    def _create_report_chart(self, chart_data: Dict) -> bytes:
        """Create chart for report
        
        Args:
            chart_data: Chart configuration
            
        Returns:
            Chart image bytes
        """
        # Implementation would create various chart types
        # based on chart_data configuration
        return self._create_alert_chart(chart_data.get('metrics', {}))
        
    def _create_summary_dashboard(self, summary_data: Dict) -> bytes:
        """Create summary dashboard image
        
        Args:
            summary_data: Summary data
            
        Returns:
            Dashboard image bytes
        """
        try:
            from plotly.subplots import make_subplots
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Alerts', 'Work Orders', 'Equipment Health', 'Costs']
            )
            
            # Add various metrics
            # This is simplified - actual implementation would be more detailed
            
            fig.update_layout(
                title="System Summary Dashboard",
                showlegend=False,
                height=600,
                width=800
            )
            
            return pio.to_image(fig, format='png')
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {str(e)}")
            return None
            
    def _create_calendar_event(self, work_order: Dict) -> str:
        """Create iCalendar event for work order
        
        Args:
            work_order: Work order details
            
        Returns:
            iCalendar content
        """
        start_time = work_order.get('scheduled_start', datetime.now())
        end_time = work_order.get('scheduled_end', start_time + timedelta(hours=2))
        
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//IoT System//Work Order//EN
BEGIN:VEVENT
UID:{work_order.get('order_id')}@iot-system
DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{start_time.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{end_time.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:Work Order: {work_order.get('title', 'Maintenance')}
DESCRIPTION:{work_order.get('description', '')}
LOCATION:{work_order.get('location', '')}
STATUS:CONFIRMED
END:VEVENT
END:VCALENDAR"""
        
        return ics_content
        
    def _create_tracking_pixel(self, message: EmailMessage) -> Optional[MIMEImage]:
        """Create tracking pixel for open tracking
        
        Args:
            message: Email message
            
        Returns:
            Tracking pixel image
        """
        # Create unique tracking ID
        tracking_id = hashlib.md5(
            f"{message.to_emails[0]}:{message.subject}:{time.time()}".encode()
        ).hexdigest()
        
        # Create 1x1 transparent pixel
        pixel_data = base64.b64decode(
            'R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
        )
        
        pixel = MIMEImage(pixel_data)
        pixel.add_header('Content-ID', f"<tracking_{tracking_id}>")
        
        return pixel
        
    def _add_unsubscribe_link(self, html: str) -> str:
        """Add unsubscribe link to HTML
        
        Args:
            html: HTML content
            
        Returns:
            HTML with unsubscribe link
        """
        unsubscribe_html = """
        <div style="text-align: center; margin-top: 20px; padding: 10px; 
                    border-top: 1px solid #ccc; font-size: 12px; color: #666;">
            <a href="#" style="color: #666;">Unsubscribe</a> | 
            <a href="#" style="color: #666;">Manage Preferences</a>
        </div>
        """
        
        # Insert before closing body tag
        if '</body>' in html:
            html = html.replace('</body>', f"{unsubscribe_html}</body>")
        else:
            html += unsubscribe_html
            
        return html
        
    def _record_send(self, message: EmailMessage, status: str, error: str = None):
        """Record email send attempt
        
        Args:
            message: Email message
            status: Send status
            error: Error message if failed
        """
        record = {
            'timestamp': datetime.now(),
            'recipients': message.to_emails,
            'subject': message.subject,
            'type': message.email_type.value,
            'priority': message.priority.value,
            'status': status,
            'error': error
        }
        
        self.send_history.append(record)
        
    def _start_workers(self):
        """Start background worker threads"""
        # Queue processor
        self.executor.submit(self._queue_processor)
        
        # Scheduled email processor
        self.executor.submit(self._schedule_processor)
        
    def _queue_processor(self):
        """Process email queue"""
        while not self._stop_event.is_set():
            try:
                # Get batch from queue
                batch = self.queue_manager.get_batch(self.config.batch_size)
                
                if batch:
                    self.send_batch(batch)
                else:
                    time.sleep(10)  # Wait if queue is empty
                    
            except Exception as e:
                logger.error(f"Queue processor error: {str(e)}")
                
    def _schedule_processor(self):
        """Process scheduled emails"""
        while not self._stop_event.is_set():
            try:
                # Get due scheduled emails
                due_emails = self.queue_manager.get_due_scheduled()
                
                for message in due_emails:
                    self.send_immediate(message)
                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Schedule processor error: {str(e)}")
                
    def _get_default_config(self) -> EmailConfig:
        """Get default email configuration
        
        Returns:
            Default configuration
        """
        return EmailConfig(
            smtp_host='localhost',
            smtp_port=587,
            from_email='noreply@iot-system.com'
        )
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get email statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'sent': self.stats['sent'],
            'failed': self.stats['failed'],
            'queued': self.stats['queued'],
            'retried': self.stats['retried'],
            'queue_size': self.queue_manager.get_queue_size(),
            'success_rate': (
                self.stats['sent'] / (self.stats['sent'] + self.stats['failed']) * 100
                if (self.stats['sent'] + self.stats['failed']) > 0 else 0
            )
        }
        
    def stop(self):
        """Stop email sender"""
        self._stop_event.set()
        self.executor.shutdown(wait=True)
        self.smtp_pool.close_all()


class EmailTemplateEngine:
    """Email template rendering engine"""
    
    def __init__(self, template_dir: str):
        """Initialize Template Engine
        
        Args:
            template_dir: Template directory path
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )
        
        # Load default templates
        self._create_default_templates()
        
    def render_alert_template(self, alert_data: Dict) -> str:
        """Render alert email template
        
        Args:
            alert_data: Alert information
            
        Returns:
            Rendered HTML
        """
        template = self.env.get_template('alert.html')
        return template.render(alert=alert_data, timestamp=datetime.now())
        
    def render_work_order_template(self, work_order: Dict) -> str:
        """Render work order email template
        
        Args:
            work_order: Work order information
            
        Returns:
            Rendered HTML
        """
        template = self.env.get_template('work_order.html')
        return template.render(work_order=work_order, timestamp=datetime.now())
        
    def render_report_template(self, report_data: Dict) -> str:
        """Render report email template
        
        Args:
            report_data: Report information
            
        Returns:
            Rendered HTML
        """
        template = self.env.get_template('report.html')
        return template.render(report=report_data, timestamp=datetime.now())
        
    def render_summary_template(self, summary_data: Dict, period: str) -> str:
        """Render summary email template
        
        Args:
            summary_data: Summary information
            period: Summary period
            
        Returns:
            Rendered HTML
        """
        template = self.env.get_template('summary.html')
        return template.render(
            summary=summary_data,
            period=period,
            timestamp=datetime.now()
        )
        
    def _create_default_templates(self):
        """Create default email templates"""
        # Alert template
        alert_template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .alert-header { 
            background-color: #ff4444; 
            color: white; 
            padding: 20px;
            border-radius: 5px 5px 0 0;
        }
        .alert-body { 
            padding: 20px; 
            background-color: #f9f9f9;
        }
        .metric { 
            display: inline-block; 
            margin: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="alert-header">
        <h2>{{ alert.severity }} Alert</h2>
        <p>{{ alert.title }}</p>
    </div>
    <div class="alert-body">
        <p><strong>Source:</strong> {{ alert.source }}</p>
        <p><strong>Description:</strong> {{ alert.description }}</p>
        <p><strong>Time:</strong> {{ alert.created_at }}</p>
        
        {% if alert.metrics %}
        <h3>Metrics:</h3>
        <div>
            {% for key, value in alert.metrics.items() %}
            <span class="metric">
                <strong>{{ key }}:</strong> {{ value }}
            </span>
            {% endfor %}
        </div>
        {% endif %}
        
        <p style="margin-top: 20px;">
            <a href="#" style="background: #4CAF50; color: white; padding: 10px 20px; 
                             text-decoration: none; border-radius: 5px;">
                View in Dashboard
            </a>
        </p>
    </div>
</body>
</html>
        """
        
        # Save template
        template_path = self.template_dir / 'alert.html'
        if not template_path.exists():
            template_path.write_text(alert_template)
            
        # Create other default templates similarly
        self._create_work_order_template()
        self._create_report_template()
        self._create_summary_template()
        
    def _create_work_order_template(self):
        """Create default work order template"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .wo-header { background: #2196F3; color: white; padding: 20px; }
        .wo-details { padding: 20px; background: #f5f5f5; }
        .wo-info { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="wo-header">
        <h2>Work Order: {{ work_order.order_id }}</h2>
    </div>
    <div class="wo-details">
        <div class="wo-info"><strong>Priority:</strong> {{ work_order.priority }}</div>
        <div class="wo-info"><strong>Equipment:</strong> {{ work_order.equipment_id }}</div>
        <div class="wo-info"><strong>Description:</strong> {{ work_order.description }}</div>
        <div class="wo-info"><strong>Scheduled:</strong> {{ work_order.scheduled_start }}</div>
    </div>
</body>
</html>
        """
        
        template_path = self.template_dir / 'work_order.html'
        if not template_path.exists():
            template_path.write_text(template)
            
    def _create_report_template(self):
        """Create default report template"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .report-header { background: #4CAF50; color: white; padding: 20px; }
        .report-content { padding: 20px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <div class="report-header">
        <h2>{{ report.title }}</h2>
    </div>
    <div class="report-content">
        {{ report.content | safe }}
    </div>
</body>
</html>
        """
        
        template_path = self.template_dir / 'report.html'
        if not template_path.exists():
            template_path.write_text(template)
            
    def _create_summary_template(self):
        """Create default summary template"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        .summary-header { background: #9C27B0; color: white; padding: 20px; }
        .summary-stats { display: flex; justify-content: space-around; padding: 20px; }
        .stat-box { text-align: center; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .stat-label { color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="summary-header">
        <h2>{{ period | capitalize }} Summary</h2>
        <p>{{ timestamp.strftime('%Y-%m-%d') }}</p>
    </div>
    <div class="summary-stats">
        {% for key, value in summary.items() %}
        <div class="stat-box">
            <div class="stat-value">{{ value }}</div>
            <div class="stat-label">{{ key | replace('_', ' ') | title }}</div>
        </div>
        {% endfor %}
    </div>
    <div style="text-align: center; padding: 20px;">
        <img src="cid:dashboard" alt="Dashboard" style="max-width: 100%;">
    </div>
</body>
</html>
        """
        
        template_path = self.template_dir / 'summary.html'
        if not template_path.exists():
            template_path.write_text(template)


class EmailQueueManager:
    """Manage email queue and scheduling"""
    
    def __init__(self):
        """Initialize Queue Manager"""
        self.queue = queue.PriorityQueue()
        self.scheduled_emails = []
        self.queue_lock = threading.Lock()
        
    def add_to_queue(self, message: EmailMessage):
        """Add email to queue
        
        Args:
            message: Email message
        """
        priority = message.priority.value
        self.queue.put((priority, time.time(), message))
        
    def get_batch(self, batch_size: int) -> List[EmailMessage]:
        """Get batch of emails from queue
        
        Args:
            batch_size: Number of emails to get
            
        Returns:
            List of email messages
        """
        batch = []
        
        with self.queue_lock:
            while len(batch) < batch_size and not self.queue.empty():
                try:
                    _, _, message = self.queue.get_nowait()
                    batch.append(message)
                except queue.Empty:
                    break
                    
        return batch
        
    def schedule_email(self, message: EmailMessage) -> bool:
        """Schedule email for future sending
        
        Args:
            message: Email message with scheduled_time
            
        Returns:
            Success flag
        """
        if not message.scheduled_time:
            return False
            
        with self.queue_lock:
            self.scheduled_emails.append(message)
            # Sort by scheduled time
            self.scheduled_emails.sort(key=lambda x: x.scheduled_time)
            
        return True
        
    def get_due_scheduled(self) -> List[EmailMessage]:
        """Get scheduled emails that are due
        
        Returns:
            List of due email messages
        """
        due_emails = []
        current_time = datetime.now()
        
        with self.queue_lock:
            while self.scheduled_emails and self.scheduled_emails[0].scheduled_time <= current_time:
                due_emails.append(self.scheduled_emails.pop(0))
                
        return due_emails
        
    def get_queue_size(self) -> int:
        """Get current queue size
        
        Returns:
            Queue size
        """
        return self.queue.qsize()


class SMTPConnectionPool:
    """SMTP connection pool for efficiency"""
    
    def __init__(self, config: EmailConfig, max_connections: int = 5):
        """Initialize Connection Pool
        
        Args:
            config: Email configuration
            max_connections: Maximum connections
        """
        self.config = config
        self.max_connections = max_connections
        self.connections = []
        self.available = queue.Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        
        # Create initial connections
        for _ in range(min(2, max_connections)):
            conn = self._create_connection()
            if conn:
                self.available.put(conn)
                self.connections.append(conn)
                
    def get_connection(self) -> smtplib.SMTP:
        """Get SMTP connection from pool
        
        Returns:
            SMTP connection
        """
        try:
            # Try to get existing connection
            conn = self.available.get_nowait()
            
            # Test if connection is alive
            try:
                conn.noop()
                return conn
            except:
                # Connection is dead, create new one
                conn = self._create_connection()
                if conn:
                    return conn
                    
        except queue.Empty:
            # No available connections, create new one
            with self.lock:
                if len(self.connections) < self.max_connections:
                    conn = self._create_connection()
                    if conn:
                        self.connections.append(conn)
                        return conn
                        
        # Wait for available connection
        return self.available.get(timeout=30)
        
    def return_connection(self, conn: smtplib.SMTP):
        """Return connection to pool
        
        Args:
            conn: SMTP connection
        """
        try:
            self.available.put_nowait(conn)
        except queue.Full:
            # Pool is full, close connection
            try:
                conn.quit()
            except:
                pass
                
    def _create_connection(self) -> Optional[smtplib.SMTP]:
        """Create new SMTP connection
        
        Returns:
            SMTP connection or None
        """
        try:
            if self.config.use_ssl:
                conn = smtplib.SMTP_SSL(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout
                )
            else:
                conn = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port,
                    timeout=self.config.timeout
                )
                
                if self.config.use_tls:
                    conn.starttls()
                    
            # Authenticate if credentials provided
            if self.config.username and self.config.password:
                conn.login(self.config.username, self.config.password)
                
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create SMTP connection: {str(e)}")
            return None
            
    def close_all(self):
        """Close all connections"""
        for conn in self.connections:
            try:
                conn.quit()
            except:
                pass
        self.connections.clear()


class RateLimiter:
    """Rate limiting for email sending"""
    
    def __init__(self, rate: int):
        """Initialize Rate Limiter
        
        Args:
            rate: Emails per second
        """
        self.rate = rate
        self.interval = 1.0 / rate
        self.last_send = 0
        self.lock = threading.Lock()
        
    def acquire(self):
        """Acquire permission to send"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_send
            
            if time_since_last < self.interval:
                sleep_time = self.interval - time_since_last
                time.sleep(sleep_time)
                
            self.last_send = time.time()