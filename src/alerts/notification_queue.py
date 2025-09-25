"""
Notification Queue Module for IoT Anomaly Detection System
Advanced queue management for multi-channel notifications with prioritization and reliability
"""

import asyncio
import aioredis
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
from collections import defaultdict, deque, Counter
from enum import Enum
import threading
import queue
import heapq
from concurrent.futures import ThreadPoolExecutor, Future
import time
import hashlib
import redis
from abc import ABC, abstractmethod
import requests
import aiohttp
from circuitbreaker import circuit
import backoff

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels"""
    CRITICAL = 0    # Immediate delivery required
    HIGH = 1        # Urgent delivery
    NORMAL = 2      # Standard delivery
    LOW = 3         # Can be delayed
    BATCH = 4       # Batch processing


class NotificationChannel(Enum):
    """Notification delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PUSH = "push"
    DASHBOARD = "dashboard"
    VOICE = "voice"
    PAGERDUTY = "pagerduty"


class NotificationStatus(Enum):
    """Notification processing status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DeliveryStrategy(Enum):
    """Notification delivery strategies"""
    IMMEDIATE = "immediate"      # Send immediately
    BATCH = "batch"              # Batch with similar notifications
    SCHEDULED = "scheduled"       # Send at scheduled time
    THROTTLED = "throttled"      # Apply rate limiting
    AGGREGATED = "aggregated"    # Aggregate similar notifications


@dataclass
class NotificationConfig:
    """Notification queue configuration"""
    # Queue settings
    max_queue_size: int = 10000
    batch_size: int = 50
    batch_interval: int = 60  # seconds
    
    # Retry settings
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    retry_backoff: float = 2.0
    
    # Timeout settings
    notification_ttl: int = 3600  # seconds
    processing_timeout: int = 30  # seconds
    
    # Rate limiting
    global_rate_limit: int = 100  # per second
    channel_rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # Aggregation settings
    aggregation_window: int = 300  # seconds
    aggregation_threshold: int = 5  # min notifications to aggregate
    
    # Persistence
    use_redis: bool = False
    redis_config: Dict[str, Any] = field(default_factory=dict)
    persist_history: bool = True
    history_retention_days: int = 30
    
    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: int = 60
    circuit_expected_exception: type = Exception
    
    # Worker settings
    num_workers: int = 4
    worker_poll_interval: int = 1  # seconds


@dataclass
class Notification:
    """Notification message"""
    notification_id: str
    channel: NotificationChannel
    priority: NotificationPriority
    recipients: List[str]
    subject: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery settings
    delivery_strategy: DeliveryStrategy = DeliveryStrategy.IMMEDIATE
    scheduled_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Processing info
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    
    # Tracking
    attempts: int = 0
    last_error: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    
    # Content variants
    html_content: Optional[str] = None
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    template_id: Optional[str] = None
    template_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'notification_id': self.notification_id,
            'channel': self.channel.value,
            'priority': self.priority.value,
            'recipients': self.recipients,
            'subject': self.subject,
            'content': self.content,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
        
    def get_hash(self) -> str:
        """Get notification hash for deduplication"""
        content = f"{self.channel.value}:{':'.join(sorted(self.recipients))}:{self.subject}"
        return hashlib.md5(content.encode()).hexdigest()


class NotificationQueue:
    """Main notification queue manager"""
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """Initialize Notification Queue
        
        Args:
            config: Queue configuration
        """
        self.config = config or NotificationConfig()
        
        # Priority queues for each channel
        self.channel_queues = defaultdict(lambda: [])  # Min heap per channel
        self.global_queue = []  # Global priority queue
        
        # Processing components
        self.aggregator = NotificationAggregator(self.config)
        self.deduplicator = NotificationDeduplicator()
        self.rate_limiter = MultiChannelRateLimiter(self.config)
        self.circuit_breakers = {}
        self.channel_handlers = {}
        
        # Persistent storage
        if self.config.use_redis:
            self.redis_client = redis.Redis(**self.config.redis_config)
            self.persistent_queue = PersistentQueue(self.redis_client)
        else:
            self.redis_client = None
            self.persistent_queue = None
            
        # Dead letter queue
        self.dead_letter_queue = deque(maxlen=1000)
        
        # Statistics
        self.stats = NotificationStatistics()
        
        # Worker management
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.workers = []
        self._stop_event = threading.Event()
        self._queue_lock = threading.Lock()
        
        # History tracking
        self.notification_history = deque(maxlen=10000)
        
        # Initialize channel handlers
        self._initialize_handlers()
        
        # Start workers
        self._start_workers()
        
        logger.info("Initialized Notification Queue")
        
    def enqueue(self, notification: Notification) -> bool:
        """Enqueue notification for processing
        
        Args:
            notification: Notification to enqueue
            
        Returns:
            Success flag
        """
        try:
            # Validate notification
            if not self._validate_notification(notification):
                logger.error(f"Invalid notification: {notification.notification_id}")
                return False
                
            # Check for duplicates
            if self.deduplicator.is_duplicate(notification):
                logger.info(f"Duplicate notification filtered: {notification.notification_id}")
                self.stats.increment('duplicates_filtered')
                return True
                
            # Set expiry if not set
            if not notification.expires_at:
                notification.expires_at = datetime.now() + timedelta(seconds=self.config.notification_ttl)
                
            # Update status
            notification.status = NotificationStatus.QUEUED
            notification.queued_at = datetime.now()
            
            # Apply delivery strategy
            if notification.delivery_strategy == DeliveryStrategy.IMMEDIATE:
                return self._enqueue_immediate(notification)
            elif notification.delivery_strategy == DeliveryStrategy.BATCH:
                return self._enqueue_batch(notification)
            elif notification.delivery_strategy == DeliveryStrategy.SCHEDULED:
                return self._enqueue_scheduled(notification)
            elif notification.delivery_strategy == DeliveryStrategy.AGGREGATED:
                return self._enqueue_aggregated(notification)
            else:
                return self._enqueue_immediate(notification)
                
        except Exception as e:
            logger.error(f"Failed to enqueue notification: {str(e)}")
            self.stats.increment('enqueue_errors')
            return False
            
    def _enqueue_immediate(self, notification: Notification) -> bool:
        """Enqueue for immediate processing
        
        Args:
            notification: Notification to enqueue
            
        Returns:
            Success flag
        """
        with self._queue_lock:
            # Add to channel queue
            heapq.heappush(
                self.channel_queues[notification.channel],
                (notification.priority.value, time.time(), notification)
            )
            
            # Add to global queue
            heapq.heappush(
                self.global_queue,
                (notification.priority.value, time.time(), notification)
            )
            
            # Persist if configured
            if self.persistent_queue:
                self.persistent_queue.push(notification)
                
        self.stats.increment('notifications_queued')
        self.stats.increment(f'queued_{notification.channel.value}')
        
        logger.debug(f"Notification {notification.notification_id} queued for immediate delivery")
        
        return True
        
    def _enqueue_batch(self, notification: Notification) -> bool:
        """Enqueue for batch processing
        
        Args:
            notification: Notification to enqueue
            
        Returns:
            Success flag
        """
        # Add to aggregator for batching
        self.aggregator.add_for_batching(notification)
        self.stats.increment('notifications_batched')
        
        return True
        
    def _enqueue_scheduled(self, notification: Notification) -> bool:
        """Enqueue for scheduled delivery
        
        Args:
            notification: Notification to enqueue
            
        Returns:
            Success flag
        """
        if not notification.scheduled_time:
            notification.scheduled_time = datetime.now()
            
        # Add to scheduled queue
        self.aggregator.add_scheduled(notification)
        self.stats.increment('notifications_scheduled')
        
        return True
        
    def _enqueue_aggregated(self, notification: Notification) -> bool:
        """Enqueue for aggregation
        
        Args:
            notification: Notification to enqueue
            
        Returns:
            Success flag
        """
        # Add to aggregator
        if self.aggregator.should_aggregate(notification):
            self.aggregator.add_for_aggregation(notification)
            self.stats.increment('notifications_aggregated')
            return True
        else:
            # Send immediately if aggregation not needed
            return self._enqueue_immediate(notification)
            
    def process_next(self) -> Optional[Notification]:
        """Process next notification from queue
        
        Returns:
            Processed notification or None
        """
        notification = None
        
        with self._queue_lock:
            # Check global queue first
            if self.global_queue:
                priority, timestamp, notification = heapq.heappop(self.global_queue)
                
                # Remove from channel queue as well
                channel_queue = self.channel_queues[notification.channel]
                if channel_queue:
                    # Find and remove from channel queue
                    for i, (p, t, n) in enumerate(channel_queue):
                        if n.notification_id == notification.notification_id:
                            channel_queue.pop(i)
                            heapq.heapify(channel_queue)
                            break
                            
        if notification:
            # Check if expired
            if notification.expires_at and datetime.now() > notification.expires_at:
                notification.status = NotificationStatus.EXPIRED
                self.stats.increment('notifications_expired')
                self._record_notification(notification)
                return None
                
            # Process notification
            success = self._process_notification(notification)
            
            if not success and notification.attempts < self.config.max_retries:
                # Retry
                self._retry_notification(notification)
            elif not success:
                # Move to dead letter queue
                self._move_to_dead_letter(notification)
                
            return notification
            
        return None
        
    def _process_notification(self, notification: Notification) -> bool:
        """Process single notification
        
        Args:
            notification: Notification to process
            
        Returns:
            Success flag
        """
        try:
            # Update status
            notification.status = NotificationStatus.PROCESSING
            notification.attempts += 1
            
            # Check rate limit
            if not self.rate_limiter.acquire(notification.channel):
                logger.debug(f"Rate limited: {notification.notification_id}")
                # Re-queue
                self._enqueue_immediate(notification)
                return False
                
            # Get handler for channel
            handler = self.channel_handlers.get(notification.channel)
            if not handler:
                logger.error(f"No handler for channel: {notification.channel}")
                notification.last_error = "No handler available"
                return False
                
            # Check circuit breaker
            breaker = self.circuit_breakers.get(notification.channel)
            if breaker and breaker.current_state == 'open':
                logger.warning(f"Circuit breaker open for {notification.channel}")
                notification.last_error = "Circuit breaker open"
                return False
                
            # Send notification
            result = handler.send(notification)
            
            if result:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.now()
                self.stats.increment('notifications_sent')
                self.stats.increment(f'sent_{notification.channel.value}')
                
                # Record success
                self._record_notification(notification)
                
                logger.info(f"Notification {notification.notification_id} sent successfully")
                return True
            else:
                notification.status = NotificationStatus.FAILED
                notification.last_error = "Send failed"
                self.stats.increment('notifications_failed')
                
                # Trip circuit breaker if needed
                if breaker:
                    breaker.record_failure()
                    
                return False
                
        except Exception as e:
            logger.error(f"Error processing notification {notification.notification_id}: {str(e)}")
            notification.status = NotificationStatus.FAILED
            notification.last_error = str(e)
            self.stats.increment('processing_errors')
            return False
            
    def _retry_notification(self, notification: Notification):
        """Retry failed notification
        
        Args:
            notification: Notification to retry
        """
        notification.status = NotificationStatus.RETRYING
        
        # Calculate retry delay with exponential backoff
        delay = self.config.retry_delay * (self.config.retry_backoff ** (notification.attempts - 1))
        
        # Schedule retry
        retry_time = datetime.now() + timedelta(seconds=delay)
        notification.scheduled_time = retry_time
        notification.delivery_strategy = DeliveryStrategy.SCHEDULED
        
        self._enqueue_scheduled(notification)
        
        self.stats.increment('notifications_retried')
        logger.info(f"Notification {notification.notification_id} scheduled for retry at {retry_time}")
        
    def _move_to_dead_letter(self, notification: Notification):
        """Move failed notification to dead letter queue
        
        Args:
            notification: Failed notification
        """
        notification.status = NotificationStatus.FAILED
        self.dead_letter_queue.append(notification)
        
        # Persist if configured
        if self.persistent_queue:
            self.persistent_queue.push_dead_letter(notification)
            
        self.stats.increment('dead_letter_notifications')
        
        logger.warning(f"Notification {notification.notification_id} moved to dead letter queue")
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status
        
        Returns:
            Queue status information
        """
        with self._queue_lock:
            status = {
                'global_queue_size': len(self.global_queue),
                'channel_queues': {
                    channel.value: len(queue)
                    for channel, queue in self.channel_queues.items()
                },
                'dead_letter_size': len(self.dead_letter_queue),
                'statistics': self.stats.get_stats(),
                'rate_limits': self.rate_limiter.get_status(),
                'circuit_breakers': {
                    channel.value: breaker.current_state
                    for channel, breaker in self.circuit_breakers.items()
                }
            }
            
        return status
        
    def process_batch(self) -> int:
        """Process batch of notifications
        
        Returns:
            Number of notifications processed
        """
        processed = 0
        batch_size = min(self.config.batch_size, len(self.global_queue))
        
        for _ in range(batch_size):
            notification = self.process_next()
            if notification:
                processed += 1
                
        # Process aggregated notifications
        aggregated = self.aggregator.get_ready_aggregated()
        for agg_notification in aggregated:
            if self._process_notification(agg_notification):
                processed += 1
                
        # Process scheduled notifications
        scheduled = self.aggregator.get_due_scheduled()
        for sched_notification in scheduled:
            if self._process_notification(sched_notification):
                processed += 1
                
        return processed
        
    def cancel_notification(self, notification_id: str) -> bool:
        """Cancel pending notification
        
        Args:
            notification_id: Notification ID to cancel
            
        Returns:
            Success flag
        """
        with self._queue_lock:
            # Search in global queue
            for i, (priority, timestamp, notification) in enumerate(self.global_queue):
                if notification.notification_id == notification_id:
                    notification.status = NotificationStatus.CANCELLED
                    self.global_queue.pop(i)
                    heapq.heapify(self.global_queue)
                    
                    # Remove from channel queue
                    channel_queue = self.channel_queues[notification.channel]
                    for j, (p, t, n) in enumerate(channel_queue):
                        if n.notification_id == notification_id:
                            channel_queue.pop(j)
                            heapq.heapify(channel_queue)
                            break
                            
                    self.stats.increment('notifications_cancelled')
                    logger.info(f"Notification {notification_id} cancelled")
                    return True
                    
        return False
        
    def reprocess_dead_letter(self, limit: int = 10) -> int:
        """Reprocess notifications from dead letter queue
        
        Args:
            limit: Maximum number to reprocess
            
        Returns:
            Number reprocessed
        """
        reprocessed = 0
        
        for _ in range(min(limit, len(self.dead_letter_queue))):
            notification = self.dead_letter_queue.popleft()
            
            # Reset attempts
            notification.attempts = 0
            notification.status = NotificationStatus.PENDING
            notification.last_error = None
            
            # Re-enqueue
            if self.enqueue(notification):
                reprocessed += 1
                
        logger.info(f"Reprocessed {reprocessed} notifications from dead letter queue")
        return reprocessed
        
    def _validate_notification(self, notification: Notification) -> bool:
        """Validate notification
        
        Args:
            notification: Notification to validate
            
        Returns:
            True if valid
        """
        # Check required fields
        if not notification.recipients:
            logger.error("No recipients specified")
            return False
            
        if not notification.subject and not notification.content:
            logger.error("No content specified")
            return False
            
        # Check channel handler exists
        if notification.channel not in self.channel_handlers:
            logger.error(f"No handler for channel: {notification.channel}")
            return False
            
        return True
        
    def _record_notification(self, notification: Notification):
        """Record notification in history
        
        Args:
            notification: Notification to record
        """
        self.notification_history.append({
            'notification_id': notification.notification_id,
            'channel': notification.channel.value,
            'status': notification.status.value,
            'recipients': notification.recipients,
            'timestamp': datetime.now(),
            'attempts': notification.attempts
        })
        
        # Persist if configured
        if self.config.persist_history and self.redis_client:
            try:
                self.redis_client.hset(
                    'notification_history',
                    notification.notification_id,
                    json.dumps(notification.to_dict())
                )
            except Exception as e:
                logger.error(f"Failed to persist notification history: {str(e)}")
                
    def _initialize_handlers(self):
        """Initialize channel handlers"""
        # These would be actual implementations
        self.channel_handlers[NotificationChannel.EMAIL] = EmailHandler()
        self.channel_handlers[NotificationChannel.SMS] = SMSHandler()
        self.channel_handlers[NotificationChannel.SLACK] = SlackHandler()
        self.channel_handlers[NotificationChannel.WEBHOOK] = WebhookHandler()
        
        # Initialize circuit breakers
        for channel in NotificationChannel:
            self.circuit_breakers[channel] = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout
            )
            
    def _start_workers(self):
        """Start background worker threads"""
        for i in range(self.config.num_workers):
            worker = self.executor.submit(self._worker_loop, i)
            self.workers.append(worker)
            
        # Start aggregator worker
        self.executor.submit(self._aggregator_worker)
        
        logger.info(f"Started {self.config.num_workers} notification workers")
        
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing notifications
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        while not self._stop_event.is_set():
            try:
                # Process next notification
                notification = self.process_next()
                
                if not notification:
                    # No notifications, wait
                    time.sleep(self.config.worker_poll_interval)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                time.sleep(1)
                
        logger.info(f"Worker {worker_id} stopped")
        
    def _aggregator_worker(self):
        """Worker for processing aggregated notifications"""
        while not self._stop_event.is_set():
            try:
                # Process aggregated notifications
                self.aggregator.process_aggregated()
                
                # Process scheduled notifications
                self.aggregator.process_scheduled()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Aggregator worker error: {str(e)}")
                time.sleep(1)
                
    def stop(self):
        """Stop notification queue"""
        logger.info("Stopping notification queue...")
        
        self._stop_event.set()
        
        # Wait for workers
        for worker in self.workers:
            worker.result(timeout=5)
            
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save state if persistent
        if self.persistent_queue:
            self.persistent_queue.save_state(self.global_queue)
            
        logger.info("Notification queue stopped")


class NotificationAggregator:
    """Aggregate similar notifications"""
    
    def __init__(self, config: NotificationConfig):
        """Initialize Aggregator
        
        Args:
            config: Queue configuration
        """
        self.config = config
        self.aggregation_buckets = defaultdict(list)
        self.scheduled_notifications = []
        self.batch_queue = []
        self.lock = threading.Lock()
        
    def add_for_aggregation(self, notification: Notification):
        """Add notification for aggregation
        
        Args:
            notification: Notification to aggregate
        """
        with self.lock:
            # Create aggregation key
            key = self._get_aggregation_key(notification)
            
            # Add to bucket
            bucket = self.aggregation_buckets[key]
            bucket.append(notification)
            
            # Check if ready to send
            if len(bucket) >= self.config.aggregation_threshold:
                self._create_aggregated_notification(key, bucket)
                self.aggregation_buckets[key] = []
                
    def add_for_batching(self, notification: Notification):
        """Add notification for batch processing
        
        Args:
            notification: Notification to batch
        """
        with self.lock:
            self.batch_queue.append(notification)
            
    def add_scheduled(self, notification: Notification):
        """Add scheduled notification
        
        Args:
            notification: Scheduled notification
        """
        with self.lock:
            self.scheduled_notifications.append(notification)
            # Sort by scheduled time
            self.scheduled_notifications.sort(key=lambda x: x.scheduled_time)
            
    def should_aggregate(self, notification: Notification) -> bool:
        """Check if notification should be aggregated
        
        Args:
            notification: Notification to check
            
        Returns:
            True if should aggregate
        """
        # Aggregate if low priority and similar notifications exist
        if notification.priority in [NotificationPriority.LOW, NotificationPriority.BATCH]:
            key = self._get_aggregation_key(notification)
            return len(self.aggregation_buckets.get(key, [])) > 0
            
        return False
        
    def get_ready_aggregated(self) -> List[Notification]:
        """Get aggregated notifications ready to send
        
        Returns:
            List of aggregated notifications
        """
        ready = []
        current_time = datetime.now()
        
        with self.lock:
            for key, bucket in list(self.aggregation_buckets.items()):
                if not bucket:
                    continue
                    
                # Check age of oldest notification
                oldest = min(bucket, key=lambda x: x.created_at)
                age = (current_time - oldest.created_at).total_seconds()
                
                if age >= self.config.aggregation_window:
                    # Create aggregated notification
                    aggregated = self._create_aggregated_notification(key, bucket)
                    if aggregated:
                        ready.append(aggregated)
                    self.aggregation_buckets[key] = []
                    
        return ready
        
    def get_due_scheduled(self) -> List[Notification]:
        """Get scheduled notifications that are due
        
        Returns:
            List of due notifications
        """
        due = []
        current_time = datetime.now()
        
        with self.lock:
            while (self.scheduled_notifications and 
                   self.scheduled_notifications[0].scheduled_time <= current_time):
                due.append(self.scheduled_notifications.pop(0))
                
        return due
        
    def process_aggregated(self):
        """Process aggregated notifications"""
        aggregated = self.get_ready_aggregated()
        # These would be added back to main queue
        
    def process_scheduled(self):
        """Process scheduled notifications"""
        scheduled = self.get_due_scheduled()
        # These would be added back to main queue
        
    def _get_aggregation_key(self, notification: Notification) -> str:
        """Get aggregation key for notification
        
        Args:
            notification: Notification
            
        Returns:
            Aggregation key
        """
        # Aggregate by channel, recipients, and type
        recipients_key = ':'.join(sorted(notification.recipients))
        type_key = notification.metadata.get('type', 'default')
        
        return f"{notification.channel.value}:{recipients_key}:{type_key}"
        
    def _create_aggregated_notification(self, 
                                       key: str,
                                       notifications: List[Notification]) -> Optional[Notification]:
        """Create aggregated notification from bucket
        
        Args:
            key: Aggregation key
            notifications: Notifications to aggregate
            
        Returns:
            Aggregated notification or None
        """
        if not notifications:
            return None
            
        # Use first notification as template
        template = notifications[0]
        
        # Aggregate content
        subjects = [n.subject for n in notifications]
        contents = [n.content for n in notifications]
        
        aggregated = Notification(
            notification_id=str(uuid.uuid4()),
            channel=template.channel,
            priority=template.priority,
            recipients=template.recipients,
            subject=f"[{len(notifications)} Notifications] {template.subject}",
            content=self._format_aggregated_content(subjects, contents),
            metadata={
                'aggregated': True,
                'count': len(notifications),
                'original_ids': [n.notification_id for n in notifications]
            }
        )
        
        return aggregated
        
    def _format_aggregated_content(self, subjects: List[str], contents: List[str]) -> str:
        """Format aggregated content
        
        Args:
            subjects: List of subjects
            contents: List of contents
            
        Returns:
            Formatted content
        """
        formatted = f"You have {len(subjects)} notifications:\n\n"
        
        for i, (subject, content) in enumerate(zip(subjects, contents), 1):
            formatted += f"{i}. {subject}\n{content}\n\n"
            
        return formatted


class NotificationDeduplicator:
    """Deduplicate notifications"""
    
    def __init__(self, window_seconds: int = 300):
        """Initialize Deduplicator
        
        Args:
            window_seconds: Deduplication window
        """
        self.window_seconds = window_seconds
        self.seen_hashes = {}
        self.lock = threading.Lock()
        
    def is_duplicate(self, notification: Notification) -> bool:
        """Check if notification is duplicate
        
        Args:
            notification: Notification to check
            
        Returns:
            True if duplicate
        """
        notification_hash = notification.get_hash()
        current_time = time.time()
        
        with self.lock:
            # Clean old entries
            self._cleanup_old_entries(current_time)
            
            # Check if seen
            if notification_hash in self.seen_hashes:
                last_seen = self.seen_hashes[notification_hash]
                if current_time - last_seen < self.window_seconds:
                    return True
                    
            # Record
            self.seen_hashes[notification_hash] = current_time
            
        return False
        
    def _cleanup_old_entries(self, current_time: float):
        """Clean old hash entries
        
        Args:
            current_time: Current timestamp
        """
        cutoff = current_time - self.window_seconds
        self.seen_hashes = {
            h: t for h, t in self.seen_hashes.items()
            if t > cutoff
        }


class MultiChannelRateLimiter:
    """Rate limiter for multiple channels"""
    
    def __init__(self, config: NotificationConfig):
        """Initialize Rate Limiter
        
        Args:
            config: Queue configuration
        """
        self.config = config
        self.channel_limiters = {}
        self.global_limiter = TokenBucket(config.global_rate_limit)
        
        # Initialize channel limiters
        for channel in NotificationChannel:
            rate = config.channel_rate_limits.get(channel.value, 10)
            self.channel_limiters[channel] = TokenBucket(rate)
            
    def acquire(self, channel: NotificationChannel) -> bool:
        """Acquire permission to send
        
        Args:
            channel: Notification channel
            
        Returns:
            True if permitted
        """
        # Check global limit
        if not self.global_limiter.consume():
            return False
            
        # Check channel limit
        channel_limiter = self.channel_limiters.get(channel)
        if channel_limiter and not channel_limiter.consume():
            # Return global token
            self.global_limiter.tokens = min(
                self.global_limiter.capacity,
                self.global_limiter.tokens + 1
            )
            return False
            
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status
        
        Returns:
            Status information
        """
        return {
            'global': {
                'tokens': self.global_limiter.tokens,
                'capacity': self.global_limiter.capacity
            },
            'channels': {
                channel.value: {
                    'tokens': limiter.tokens,
                    'capacity': limiter.capacity
                }
                for channel, limiter in self.channel_limiters.items()
            }
        }


class TokenBucket:
    """Token bucket for rate limiting"""
    
    def __init__(self, rate: int):
        """Initialize Token Bucket
        
        Args:
            rate: Tokens per second
        """
        self.capacity = rate
        self.tokens = rate
        self.rate = rate
        self.last_update = time.time()
        self.lock = threading.Lock()
        
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens available
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
                
            return False
            
    def _refill(self):
        """Refill tokens based on elapsed time"""
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        # Add tokens based on rate
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = current_time


class CircuitBreaker:
    """Circuit breaker for channel failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize Circuit Breaker
        
        Args:
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.current_state = 'closed'  # closed, open, half_open
        self.lock = threading.Lock()
        
    def record_success(self):
        """Record successful operation"""
        with self.lock:
            if self.current_state == 'half_open':
                self.current_state = 'closed'
                self.failure_count = 0
                
    def record_failure(self):
        """Record failed operation"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.current_state = 'open'
                
    def can_proceed(self) -> bool:
        """Check if operation can proceed
        
        Returns:
            True if can proceed
        """
        with self.lock:
            if self.current_state == 'closed':
                return True
                
            if self.current_state == 'open':
                # Check if recovery timeout passed
                if (self.last_failure_time and 
                    time.time() - self.last_failure_time > self.recovery_timeout):
                    self.current_state = 'half_open'
                    return True
                return False
                
            # Half open - allow one attempt
            return True


class PersistentQueue:
    """Persistent queue using Redis"""
    
    def __init__(self, redis_client: redis.Redis):
        """Initialize Persistent Queue
        
        Args:
            redis_client: Redis client
        """
        self.redis = redis_client
        self.queue_key = 'notification_queue'
        self.dead_letter_key = 'notification_dead_letter'
        
    def push(self, notification: Notification):
        """Push notification to persistent queue
        
        Args:
            notification: Notification to persist
        """
        try:
            self.redis.rpush(
                self.queue_key,
                json.dumps(notification.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to persist notification: {str(e)}")
            
    def pop(self) -> Optional[Notification]:
        """Pop notification from persistent queue
        
        Returns:
            Notification or None
        """
        try:
            data = self.redis.lpop(self.queue_key)
            if data:
                # Reconstruct notification
                return self._deserialize_notification(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to pop from persistent queue: {str(e)}")
            
        return None
        
    def push_dead_letter(self, notification: Notification):
        """Push to dead letter queue
        
        Args:
            notification: Failed notification
        """
        try:
            self.redis.rpush(
                self.dead_letter_key,
                json.dumps(notification.to_dict())
            )
        except Exception as e:
            logger.error(f"Failed to persist to dead letter: {str(e)}")
            
    def save_state(self, queue: List):
        """Save queue state
        
        Args:
            queue: Current queue
        """
        # Implementation to save full queue state
        pass
        
    def _deserialize_notification(self, data: Dict) -> Notification:
        """Deserialize notification from dict
        
        Args:
            data: Notification data
            
        Returns:
            Notification object
        """
        # Implementation to reconstruct Notification
        return Notification(
            notification_id=data['notification_id'],
            channel=NotificationChannel(data['channel']),
            priority=NotificationPriority(data['priority']),
            recipients=data['recipients'],
            subject=data['subject'],
            content=data['content'],
            metadata=data.get('metadata', {})
        )


class NotificationStatistics:
    """Track notification statistics"""
    
    def __init__(self):
        """Initialize Statistics"""
        self.counters = Counter()
        self.timings = defaultdict(list)
        self.lock = threading.Lock()
        
    def increment(self, metric: str, value: int = 1):
        """Increment counter
        
        Args:
            metric: Metric name
            value: Increment value
        """
        with self.lock:
            self.counters[metric] += value
            
    def record_timing(self, metric: str, duration: float):
        """Record timing
        
        Args:
            metric: Metric name
            duration: Duration in seconds
        """
        with self.lock:
            self.timings[metric].append(duration)
            # Keep only last 1000 timings
            if len(self.timings[metric]) > 1000:
                self.timings[metric] = self.timings[metric][-1000:]
                
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            stats = dict(self.counters)
            
            # Add timing statistics
            for metric, values in self.timings.items():
                if values:
                    stats[f'{metric}_avg'] = np.mean(values)
                    stats[f'{metric}_p50'] = np.percentile(values, 50)
                    stats[f'{metric}_p95'] = np.percentile(values, 95)
                    
        return stats


# Channel Handler Base Classes

class NotificationHandler(ABC):
    """Base notification handler"""
    
    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """Send notification
        
        Args:
            notification: Notification to send
            
        Returns:
            Success flag
        """
        pass


class EmailHandler(NotificationHandler):
    """Email notification handler"""
    
    def send(self, notification: Notification) -> bool:
        """Send email notification"""
        # Implementation would use email_sender module
        logger.debug(f"Sending email to {notification.recipients}")
        return True


class SMSHandler(NotificationHandler):
    """SMS notification handler"""
    
    def send(self, notification: Notification) -> bool:
        """Send SMS notification"""
        # Implementation would use SMS service (Twilio, etc.)
        logger.debug(f"Sending SMS to {notification.recipients}")
        return True


class SlackHandler(NotificationHandler):
    """Slack notification handler"""
    
    def send(self, notification: Notification) -> bool:
        """Send Slack notification"""
        # Implementation would use Slack API
        logger.debug(f"Sending Slack notification")
        return True


class WebhookHandler(NotificationHandler):
    """Webhook notification handler"""
    
    def send(self, notification: Notification) -> bool:
        """Send webhook notification"""
        # Implementation would make HTTP request
        logger.debug(f"Sending webhook notification")
        return True