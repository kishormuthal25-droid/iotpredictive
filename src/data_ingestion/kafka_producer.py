"""
Kafka Producer Module for IoT Telemetry Data
Publishes telemetry streams to Kafka topics for distributed processing
"""

import json
import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict
import pickle
import zlib
import base64

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaAdminClient
    from kafka.admin import NewTopic
    from kafka.errors import KafkaError, NoBrokersAvailable
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logging.warning("kafka-python not installed. Kafka functionality will be limited.")

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config
from src.data_ingestion.stream_simulator import StreamSimulator, StreamEvent

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ProducerMetrics:
    """Metrics for monitoring producer performance"""
    messages_sent: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    topics_used: Dict[str, int] = None
    last_send_time: Optional[datetime] = None
    start_time: datetime = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.topics_used is None:
            self.topics_used = defaultdict(int)
        if self.errors is None:
            self.errors = []
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def get_throughput(self) -> float:
        """Calculate messages per second"""
        if not self.last_send_time or not self.start_time:
            return 0.0
        elapsed = (self.last_send_time - self.start_time).total_seconds()
        return self.messages_sent / elapsed if elapsed > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'bytes_sent': self.bytes_sent,
            'topics_used': dict(self.topics_used),
            'throughput': self.get_throughput(),
            'error_rate': self.messages_failed / max(1, self.messages_sent),
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class KafkaStreamProducer:
    """
    Kafka producer for streaming telemetry data
    Handles serialization, partitioning, and error recovery
    """
    
    def __init__(self,
                 bootstrap_servers: Optional[str] = None,
                 topics: Optional[Dict[str, str]] = None,
                 compression_type: str = 'gzip',
                 batch_size: int = 16384,
                 linger_ms: int = 10,
                 max_retries: int = 3,
                 enable_idempotence: bool = True,
                 serialization_format: str = 'json',
                 use_schema_registry: bool = False):
        """
        Initialize Kafka producer
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topics: Topic mapping for different data types
            compression_type: Compression algorithm ('gzip', 'snappy', 'lz4', 'zstd')
            batch_size: Batch size for producer
            linger_ms: Time to wait for batching
            max_retries: Maximum retry attempts
            enable_idempotence: Enable idempotent producer
            serialization_format: Format for serialization ('json', 'avro', 'pickle')
            use_schema_registry: Whether to use schema registry (for Avro)
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed. Install with: pip install kafka-python")
        
        # Load configuration
        kafka_config = get_config('data_ingestion.kafka', {})
        
        self.bootstrap_servers = bootstrap_servers or kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.topics = topics or kafka_config.get('topics', {
            'smap': 'iot-smap-telemetry',
            'msl': 'iot-msl-telemetry',
            'anomalies': 'iot-anomalies',
            'alerts': 'iot-alerts'
        })
        
        # Producer configuration
        self.compression_type = compression_type
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.max_retries = max_retries
        self.enable_idempotence = enable_idempotence
        self.serialization_format = serialization_format
        self.use_schema_registry = use_schema_registry
        
        # Initialize producer
        self.producer: Optional[KafkaProducer] = None
        self.admin_client: Optional[KafkaAdminClient] = None
        
        # Metrics
        self.metrics = ProducerMetrics()
        
        # Callbacks
        self.success_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Connection status
        self.connected = False
        
        logger.info(f"KafkaStreamProducer initialized with servers: {self.bootstrap_servers}")
    
    def connect(self) -> bool:
        """
        Connect to Kafka cluster
        
        Returns:
            True if connection successful
        """
        try:
            # Configure producer
            producer_config = {
                'bootstrap_servers': self.bootstrap_servers.split(','),
                'compression_type': self.compression_type,
                'batch_size': self.batch_size,
                'linger_ms': self.linger_ms,
                'retries': self.max_retries,
                'enable_idempotence': self.enable_idempotence,
                'acks': 'all' if self.enable_idempotence else 1,
                'max_in_flight_requests_per_connection': 5,
                'value_serializer': self._get_serializer(),
                'key_serializer': lambda k: k.encode('utf-8') if k else None
            }
            
            # Create producer
            self.producer = KafkaProducer(**producer_config)
            
            # Create admin client for topic management
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers.split(','),
                client_id='iot-admin'
            )
            
            # Ensure topics exist
            self._ensure_topics()
            
            self.connected = True
            logger.info("Connected to Kafka cluster successfully")
            return True
            
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers available at {self.bootstrap_servers}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.connected = False
            return False
    
    def _get_serializer(self) -> Callable:
        """Get serializer based on configuration"""
        if self.serialization_format == 'json':
            return lambda v: json.dumps(v, default=self._json_encoder).encode('utf-8')
        elif self.serialization_format == 'pickle':
            return lambda v: pickle.dumps(v)
        elif self.serialization_format == 'avro':
            # Simplified Avro serialization (would need schema registry in production)
            return lambda v: json.dumps(v, default=self._json_encoder).encode('utf-8')
        else:
            return lambda v: str(v).encode('utf-8')
    
    def _json_encoder(self, obj):
        """Custom JSON encoder for complex types"""
        if isinstance(obj, np.ndarray):
            # Compress large arrays
            if obj.size > 100:
                # Compress with zlib and encode as base64
                compressed = zlib.compress(obj.tobytes())
                return {
                    '_type': 'ndarray_compressed',
                    'dtype': str(obj.dtype),
                    'shape': obj.shape,
                    'data': base64.b64encode(compressed).decode('utf-8')
                }
            else:
                return {
                    '_type': 'ndarray',
                    'dtype': str(obj.dtype),
                    'shape': obj.shape,
                    'data': obj.tolist()
                }
        elif isinstance(obj, datetime):
            return {'_type': 'datetime', 'value': obj.isoformat()}
        elif isinstance(obj, StreamEvent):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _ensure_topics(self):
        """Ensure all required topics exist"""
        try:
            # Get existing topics
            existing_topics = self.admin_client.list_topics()
            
            # Create missing topics
            new_topics = []
            for topic_name in self.topics.values():
                if topic_name not in existing_topics:
                    new_topic = NewTopic(
                        name=topic_name,
                        num_partitions=3,  # Adjust based on scale needs
                        replication_factor=1  # Set to 3 for production
                    )
                    new_topics.append(new_topic)
            
            if new_topics:
                self.admin_client.create_topics(new_topics, validate_only=False)
                logger.info(f"Created {len(new_topics)} new topics")
                
        except Exception as e:
            logger.warning(f"Error ensuring topics: {e}")
    
    def publish_event(self, 
                      event: StreamEvent,
                      topic_override: Optional[str] = None,
                      partition_key: Optional[str] = None) -> Future:
        """
        Publish single event to Kafka
        
        Args:
            event: Stream event to publish
            topic_override: Override default topic selection
            partition_key: Key for partitioning
            
        Returns:
            Future for tracking send result
        """
        if not self.connected:
            logger.error("Not connected to Kafka")
            self.metrics.messages_failed += 1
            return None
        
        try:
            # Determine topic
            if topic_override:
                topic = topic_override
            elif event.is_anomaly:
                topic = self.topics.get('anomalies')
            else:
                topic = self.topics.get(event.spacecraft, self.topics.get('smap'))
            
            # Determine partition key
            if not partition_key:
                partition_key = f"{event.spacecraft}_{event.channel}"
            
            # Serialize event
            value = event.to_dict()
            
            # Send to Kafka
            future = self.producer.send(
                topic=topic,
                key=partition_key,
                value=value,
                timestamp_ms=int(event.timestamp.timestamp() * 1000)
            )
            
            # Add callbacks
            future.add_callback(lambda metadata: self._on_send_success(metadata, event))
            future.add_errback(lambda e: self._on_send_error(e, event))
            
            return future
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            self.metrics.messages_failed += 1
            self.metrics.errors.append(str(e))
            return None
    
    def publish_batch(self,
                     events: List[StreamEvent],
                     async_send: bool = True) -> List[Future]:
        """
        Publish batch of events
        
        Args:
            events: List of events to publish
            async_send: Whether to send asynchronously
            
        Returns:
            List of futures for tracking
        """
        futures = []
        
        if async_send:
            # Use thread pool for parallel sending
            for event in events:
                future = self.executor.submit(self.publish_event, event)
                futures.append(future)
        else:
            # Sequential sending
            for event in events:
                future = self.publish_event(event)
                if future:
                    futures.append(future)
        
        return futures
    
    def _on_send_success(self, metadata, event: StreamEvent):
        """Handle successful send"""
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += metadata.serialized_value_size or 0
        self.metrics.topics_used[metadata.topic] += 1
        self.metrics.last_send_time = datetime.now()
        
        # Call success callbacks
        for callback in self.success_callbacks:
            try:
                callback(metadata, event)
            except Exception as e:
                logger.error(f"Error in success callback: {e}")
        
        logger.debug(f"Sent event to {metadata.topic}:{metadata.partition}@{metadata.offset}")
    
    def _on_send_error(self, error, event: StreamEvent):
        """Handle send error"""
        self.metrics.messages_failed += 1
        self.metrics.errors.append(str(error))
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error, event)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        logger.error(f"Failed to send event: {error}")
    
    def add_success_callback(self, callback: Callable):
        """Add callback for successful sends"""
        self.success_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for send errors"""
        self.error_callbacks.append(callback)
    
    def flush(self, timeout: float = 10):
        """
        Flush pending messages
        
        Args:
            timeout: Timeout in seconds
        """
        if self.producer:
            self.producer.flush(timeout=timeout)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics"""
        return self.metrics.to_dict()
    
    def close(self):
        """Close producer and cleanup resources"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
        
        self.executor.shutdown(wait=True)
        self.connected = False
        logger.info("Kafka producer closed")


class StreamToKafkaBridge:
    """
    Bridge between StreamSimulator and Kafka
    Automatically publishes stream events to Kafka topics
    """
    
    def __init__(self,
                 simulator: StreamSimulator,
                 producer: Optional[KafkaStreamProducer] = None,
                 batch_publish: bool = True,
                 batch_size: int = 100,
                 batch_timeout: float = 1.0):
        """
        Initialize bridge
        
        Args:
            simulator: Stream simulator instance
            producer: Kafka producer instance (creates new if None)
            batch_publish: Whether to batch events before publishing
            batch_size: Size of batches
            batch_timeout: Timeout for batch collection
        """
        self.simulator = simulator
        self.producer = producer or KafkaStreamProducer()
        self.batch_publish = batch_publish
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch buffer
        self.batch_buffer: List[StreamEvent] = []
        self.batch_lock = threading.Lock()
        self.batch_timer: Optional[threading.Timer] = None
        
        # Statistics
        self.events_bridged = 0
        self.batches_sent = 0
        
        # Running state
        self.is_running = False
        self.bridge_thread: Optional[threading.Thread] = None
        
        logger.info("StreamToKafkaBridge initialized")
    
    def start(self) -> bool:
        """
        Start bridging events from simulator to Kafka
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Bridge already running")
            return False
        
        # Connect producer if not connected
        if not self.producer.connected:
            if not self.producer.connect():
                logger.error("Failed to connect Kafka producer")
                return False
        
        # Subscribe to simulator events
        self.simulator.subscribe(self._handle_event)
        
        self.is_running = True
        
        # Start batch processing thread if batching enabled
        if self.batch_publish:
            self.bridge_thread = threading.Thread(
                target=self._batch_processor,
                daemon=True
            )
            self.bridge_thread.start()
        
        logger.info("StreamToKafkaBridge started")
        return True
    
    def stop(self):
        """Stop bridging"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Unsubscribe from simulator
        self.simulator.unsubscribe(self._handle_event)
        
        # Flush any remaining batched events
        if self.batch_publish:
            self._flush_batch()
        
        # Wait for thread to finish
        if self.bridge_thread:
            self.bridge_thread.join(timeout=5)
        
        # Flush Kafka producer
        self.producer.flush()
        
        logger.info(f"StreamToKafkaBridge stopped. Bridged {self.events_bridged} events")
    
    def _handle_event(self, event: StreamEvent):
        """Handle event from simulator"""
        if self.batch_publish:
            # Add to batch
            with self.batch_lock:
                self.batch_buffer.append(event)
                
                # Check if batch is full
                if len(self.batch_buffer) >= self.batch_size:
                    self._flush_batch()
                else:
                    # Reset timer
                    self._reset_batch_timer()
        else:
            # Publish immediately
            self.producer.publish_event(event)
            self.events_bridged += 1
    
    def _batch_processor(self):
        """Background thread for batch processing"""
        while self.is_running:
            time.sleep(self.batch_timeout)
            
            with self.batch_lock:
                if self.batch_buffer:
                    self._flush_batch()
    
    def _flush_batch(self):
        """Flush current batch to Kafka"""
        if not self.batch_buffer:
            return
        
        # Copy and clear buffer
        events_to_send = self.batch_buffer.copy()
        self.batch_buffer.clear()
        
        # Cancel timer
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Publish batch
        futures = self.producer.publish_batch(events_to_send)
        
        self.events_bridged += len(events_to_send)
        self.batches_sent += 1
        
        logger.debug(f"Flushed batch of {len(events_to_send)} events")
    
    def _reset_batch_timer(self):
        """Reset batch timeout timer"""
        if self.batch_timer:
            self.batch_timer.cancel()
        
        self.batch_timer = threading.Timer(
            self.batch_timeout,
            self._flush_batch_with_lock
        )
        self.batch_timer.daemon = True
        self.batch_timer.start()
    
    def _flush_batch_with_lock(self):
        """Flush batch with lock (for timer callback)"""
        with self.batch_lock:
            self._flush_batch()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        producer_metrics = self.producer.get_metrics()
        
        return {
            'events_bridged': self.events_bridged,
            'batches_sent': self.batches_sent,
            'average_batch_size': self.events_bridged / max(1, self.batches_sent),
            'producer_metrics': producer_metrics,
            'is_running': self.is_running
        }


class MockKafkaProducer:
    """Mock Kafka producer for testing without Kafka cluster"""
    
    def __init__(self):
        """Initialize mock producer"""
        self.messages = []
        self.connected = True
        self.metrics = ProducerMetrics()
        logger.info("Using MockKafkaProducer (no actual Kafka connection)")
    
    def connect(self) -> bool:
        """Mock connect"""
        return True
    
    def publish_event(self, event: StreamEvent, **kwargs):
        """Mock publish"""
        self.messages.append(event)
        self.metrics.messages_sent += 1
        logger.debug(f"Mock published: {event.spacecraft}/{event.channel}")
        return None
    
    def publish_batch(self, events: List[StreamEvent], **kwargs):
        """Mock batch publish"""
        self.messages.extend(events)
        self.metrics.messages_sent += len(events)
        return []
    
    def flush(self, timeout: float = 10):
        """Mock flush"""
        pass
    
    def close(self):
        """Mock close"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mock metrics"""
        return self.metrics.to_dict()


def create_kafka_producer(use_mock: bool = False) -> Union[KafkaStreamProducer, MockKafkaProducer]:
    """
    Factory function to create appropriate producer
    
    Args:
        use_mock: Use mock producer if True or if Kafka not available
        
    Returns:
        Producer instance
    """
    if use_mock or not KAFKA_AVAILABLE:
        return MockKafkaProducer()
    else:
        return KafkaStreamProducer()


if __name__ == "__main__":
    # Test Kafka producer
    print("\n" + "="*60)
    print("Testing Kafka Producer")
    print("="*60)
    
    # Check if Kafka is available
    if not KAFKA_AVAILABLE:
        print("Kafka not installed. Using mock producer for testing.")
        use_mock = True
    else:
        # Check if Kafka is running
        try:
            producer = KafkaStreamProducer()
            if not producer.connect():
                print("Kafka not running. Using mock producer for testing.")
                use_mock = True
            else:
                producer.close()
                use_mock = False
        except:
            use_mock = True
    
    # Create simulator
    print("\nCreating stream simulator...")
    simulator = StreamSimulator(
        spacecraft=["smap"],
        speed_multiplier=10.0,
        window_size=100
    )
    
    # Create producer
    print("Creating Kafka producer...")
    producer = create_kafka_producer(use_mock=use_mock)
    
    if not use_mock:
        producer.connect()
    
    # Create bridge
    print("Creating stream-to-Kafka bridge...")
    bridge = StreamToKafkaBridge(
        simulator=simulator,
        producer=producer,
        batch_publish=True,
        batch_size=10
    )
    
    # Start streaming and bridging
    print("\nStarting stream and bridge...")
    simulator.start_streaming()
    bridge.start()
    
    try:
        # Run for 10 seconds
        time.sleep(10)
        
        # Get statistics
        print("\nStatistics:")
        bridge_stats = bridge.get_statistics()
        print(f"  Events bridged: {bridge_stats['events_bridged']}")
        print(f"  Batches sent: {bridge_stats['batches_sent']}")
        print(f"  Producer throughput: {bridge_stats['producer_metrics']['throughput']:.2f} msg/sec")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        bridge.stop()
        simulator.stop_streaming()
        producer.close()
    
    print("\n" + "="*60)
    print("Kafka producer test complete")
    print("="*60)
