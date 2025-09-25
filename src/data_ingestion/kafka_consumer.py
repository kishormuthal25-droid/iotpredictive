"""
Kafka Consumer Module for IoT Telemetry Data
Consumes telemetry streams from Kafka topics for processing and analysis
"""

import json
import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Union, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import pickle
import zlib
import base64
from enum import Enum
import signal

# Kafka imports
try:
    from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
    from kafka.errors import KafkaError, CommitFailedError, NoBrokersAvailable
    from kafka.consumer.fetcher import ConsumerRecord
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    ConsumerRecord = None
    logging.warning("kafka-python not installed. Kafka functionality will be limited.")

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config
from src.data_ingestion.stream_simulator import StreamEvent

# Setup logging
logger = logging.getLogger(__name__)


class ConsumeMode(Enum):
    """Consumer processing modes"""
    BATCH = "batch"
    STREAM = "stream"
    WINDOW = "window"


@dataclass
class ConsumerMetrics:
    """Metrics for monitoring consumer performance"""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_consumed: int = 0
    topics_consumed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    partitions_assigned: List[TopicPartition] = field(default_factory=list)
    last_consume_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    lag_per_partition: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def get_throughput(self) -> float:
        """Calculate messages per second"""
        if not self.last_consume_time or not self.start_time:
            return 0.0
        elapsed = (self.last_consume_time - self.start_time).total_seconds()
        return self.messages_consumed / elapsed if elapsed > 0 else 0.0
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time in milliseconds"""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'messages_consumed': self.messages_consumed,
            'messages_processed': self.messages_processed,
            'messages_failed': self.messages_failed,
            'bytes_consumed': self.bytes_consumed,
            'topics_consumed': dict(self.topics_consumed),
            'partitions_count': len(self.partitions_assigned),
            'throughput': self.get_throughput(),
            'avg_processing_time_ms': self.get_avg_processing_time(),
            'error_rate': self.messages_failed / max(1, self.messages_consumed),
            'total_lag': sum(self.lag_per_partition.values()),
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds()
        }


class KafkaStreamConsumer:
    """
    Kafka consumer for processing telemetry streams
    Handles deserialization, offset management, and error recovery
    """
    
    def __init__(self,
                 bootstrap_servers: Optional[str] = None,
                 topics: Optional[List[str]] = None,
                 group_id: Optional[str] = None,
                 auto_offset_reset: str = 'latest',
                 enable_auto_commit: bool = False,
                 max_poll_records: int = 500,
                 session_timeout_ms: int = 30000,
                 heartbeat_interval_ms: int = 3000,
                 consume_mode: ConsumeMode = ConsumeMode.BATCH,
                 batch_size: int = 100,
                 batch_timeout: float = 1.0,
                 deserialization_format: str = 'json'):
        """
        Initialize Kafka consumer
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topics: List of topics to subscribe to
            group_id: Consumer group ID
            auto_offset_reset: Where to start reading ('earliest', 'latest')
            enable_auto_commit: Whether to auto-commit offsets
            max_poll_records: Maximum records per poll
            session_timeout_ms: Session timeout
            heartbeat_interval_ms: Heartbeat interval
            consume_mode: Processing mode (batch/stream/window)
            batch_size: Size of batches for batch mode
            batch_timeout: Timeout for batch collection
            deserialization_format: Format for deserialization
        """
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python is not installed. Install with: pip install kafka-python")
        
        # Load configuration
        kafka_config = get_config('data_ingestion.kafka', {})
        consumer_config = kafka_config.get('consumer', {})
        
        self.bootstrap_servers = bootstrap_servers or kafka_config.get('bootstrap_servers', 'localhost:9092')
        
        # Default topics if not specified
        if topics is None:
            topic_mapping = kafka_config.get('topics', {})
            self.topics = list(topic_mapping.values())
        else:
            self.topics = topics
        
        # Consumer configuration
        self.group_id = group_id or consumer_config.get('group_id', 'iot-anomaly-detector')
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.max_poll_records = max_poll_records
        self.session_timeout_ms = session_timeout_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.consume_mode = consume_mode
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.deserialization_format = deserialization_format
        
        # Initialize consumer
        self.consumer: Optional[KafkaConsumer] = None
        
        # Processing handlers
        self.message_handlers: List[Callable] = []
        self.batch_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []
        
        # Metrics
        self.metrics = ConsumerMetrics()
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Control flags
        self.is_consuming = False
        self.consumer_thread: Optional[threading.Thread] = None
        self.shutdown_requested = False
        
        # Window buffer for window mode
        self.window_buffer: deque = deque()
        self.window_size = 100
        self.window_slide = 10
        
        logger.info(f"KafkaStreamConsumer initialized for topics: {self.topics}")
    
    def connect(self) -> bool:
        """
        Connect to Kafka cluster and subscribe to topics
        
        Returns:
            True if connection successful
        """
        try:
            # Configure consumer
            consumer_config = {
                'bootstrap_servers': self.bootstrap_servers.split(','),
                'group_id': self.group_id,
                'auto_offset_reset': self.auto_offset_reset,
                'enable_auto_commit': self.enable_auto_commit,
                'max_poll_records': self.max_poll_records,
                'session_timeout_ms': self.session_timeout_ms,
                'heartbeat_interval_ms': self.heartbeat_interval_ms,
                'value_deserializer': self._get_deserializer(),
                'key_deserializer': lambda k: k.decode('utf-8') if k else None,
                'consumer_timeout_ms': 1000  # Timeout for poll
            }
            
            # Create consumer
            self.consumer = KafkaConsumer(**consumer_config)
            
            # Subscribe to topics
            self.consumer.subscribe(self.topics)
            
            logger.info(f"Connected to Kafka and subscribed to topics: {self.topics}")
            return True
            
        except NoBrokersAvailable:
            logger.error(f"No Kafka brokers available at {self.bootstrap_servers}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False
    
    def _get_deserializer(self) -> Callable:
        """Get deserializer based on configuration"""
        if self.deserialization_format == 'json':
            return lambda v: self._json_decoder(v)
        elif self.deserialization_format == 'pickle':
            return lambda v: pickle.loads(v)
        elif self.deserialization_format == 'avro':
            # Simplified Avro deserialization
            return lambda v: json.loads(v.decode('utf-8'))
        else:
            return lambda v: v.decode('utf-8')
    
    def _json_decoder(self, data: bytes) -> Any:
        """Custom JSON decoder for complex types"""
        try:
            obj = json.loads(data.decode('utf-8'))
            return self._decode_custom_types(obj)
        except Exception as e:
            logger.error(f"Error decoding JSON: {e}")
            return None
    
    def _decode_custom_types(self, obj: Any) -> Any:
        """Recursively decode custom types"""
        if isinstance(obj, dict):
            # Check for custom type markers
            if '_type' in obj:
                if obj['_type'] == 'ndarray':
                    # Reconstruct numpy array
                    return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
                elif obj['_type'] == 'ndarray_compressed':
                    # Decompress numpy array
                    compressed_data = base64.b64decode(obj['data'])
                    decompressed = zlib.decompress(compressed_data)
                    dtype = np.dtype(obj['dtype'])
                    array = np.frombuffer(decompressed, dtype=dtype)
                    return array.reshape(obj['shape'])
                elif obj['_type'] == 'datetime':
                    return datetime.fromisoformat(obj['value'])
            
            # Recursively process dictionary
            return {k: self._decode_custom_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list
            return [self._decode_custom_types(item) for item in obj]
        else:
            return obj
    
    def start_consuming(self) -> bool:
        """
        Start consuming messages
        
        Returns:
            True if started successfully
        """
        if self.is_consuming:
            logger.warning("Consumer already running")
            return False
        
        if not self.consumer:
            if not self.connect():
                logger.error("Failed to connect consumer")
                return False
        
        self.is_consuming = True
        self.shutdown_requested = False
        
        # Start consumer thread
        self.consumer_thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="kafka-consumer"
        )
        self.consumer_thread.start()
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Started consuming in {self.consume_mode.value} mode")
        return True
    
    def stop_consuming(self):
        """Stop consuming messages"""
        if not self.is_consuming:
            return
        
        logger.info("Stopping consumer...")
        self.shutdown_requested = True
        self.is_consuming = False
        
        # Wait for consumer thread to finish
        if self.consumer_thread:
            self.consumer_thread.join(timeout=10)
        
        # Close consumer
        if self.consumer:
            try:
                self.consumer.close()
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=5)
        
        logger.info(f"Consumer stopped. Processed {self.metrics.messages_processed} messages")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_requested = True
    
    def _consume_loop(self):
        """Main consumption loop"""
        logger.info("Consumer loop started")
        
        if self.consume_mode == ConsumeMode.BATCH:
            self._consume_batch_mode()
        elif self.consume_mode == ConsumeMode.STREAM:
            self._consume_stream_mode()
        elif self.consume_mode == ConsumeMode.WINDOW:
            self._consume_window_mode()
    
    def _consume_batch_mode(self):
        """Consume messages in batches"""
        batch = []
        last_batch_time = time.time()
        
        while self.is_consuming and not self.shutdown_requested:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=100, max_records=self.batch_size)
                
                if messages:
                    # Process messages from all partitions
                    for topic_partition, records in messages.items():
                        for record in records:
                            self._update_metrics(record)
                            batch.append(record)
                
                # Check if batch is ready
                current_time = time.time()
                batch_ready = (
                    len(batch) >= self.batch_size or
                    (current_time - last_batch_time) >= self.batch_timeout
                )
                
                if batch_ready and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
            except Exception as e:
                logger.error(f"Error in batch consume loop: {e}")
                self.metrics.errors.append(str(e))
                time.sleep(1)
        
        # Process remaining batch
        if batch:
            self._process_batch(batch)
    
    def _consume_stream_mode(self):
        """Consume messages in streaming mode"""
        while self.is_consuming and not self.shutdown_requested:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=100, max_records=1)
                
                if messages:
                    for topic_partition, records in messages.items():
                        for record in records:
                            self._update_metrics(record)
                            self._process_message(record)
                
            except Exception as e:
                logger.error(f"Error in stream consume loop: {e}")
                self.metrics.errors.append(str(e))
                time.sleep(1)
    
    def _consume_window_mode(self):
        """Consume messages in windowed mode"""
        while self.is_consuming and not self.shutdown_requested:
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=100)
                
                if messages:
                    for topic_partition, records in messages.items():
                        for record in records:
                            self._update_metrics(record)
                            self.window_buffer.append(record)
                            
                            # Check if window is ready
                            if len(self.window_buffer) >= self.window_size:
                                window = list(self.window_buffer)[:self.window_size]
                                self._process_window(window)
                                
                                # Slide window
                                for _ in range(self.window_slide):
                                    if self.window_buffer:
                                        self.window_buffer.popleft()
                
            except Exception as e:
                logger.error(f"Error in window consume loop: {e}")
                self.metrics.errors.append(str(e))
                time.sleep(1)
    
    def _process_message(self, record: ConsumerRecord):
        """Process single message"""
        start_time = time.time()
        
        try:
            # Convert to StreamEvent if possible
            event = self._record_to_event(record)
            
            # Call message handlers
            for handler in self.message_handlers:
                try:
                    handler(event if event else record)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    self.metrics.messages_failed += 1
            
            self.metrics.messages_processed += 1
            
            # Commit offset if not auto-committing
            if not self.enable_auto_commit:
                self._commit_offset(record)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.metrics.messages_failed += 1
            
            # Call error handlers
            for handler in self.error_handlers:
                try:
                    handler(e, record)
                except:
                    pass
        
        finally:
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self.metrics.processing_times.append(processing_time)
    
    def _process_batch(self, batch: List[ConsumerRecord]):
        """Process batch of messages"""
        start_time = time.time()
        
        try:
            # Convert to events
            events = []
            for record in batch:
                event = self._record_to_event(record)
                if event:
                    events.append(event)
            
            # Call batch handlers
            for handler in self.batch_handlers:
                try:
                    handler(events if events else batch)
                except Exception as e:
                    logger.error(f"Error in batch handler: {e}")
                    self.metrics.messages_failed += len(batch)
            
            self.metrics.messages_processed += len(batch)
            
            # Commit offsets if not auto-committing
            if not self.enable_auto_commit:
                self._commit_batch_offsets(batch)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.metrics.messages_failed += len(batch)
            
            # Call error handlers
            for handler in self.error_handlers:
                try:
                    handler(e, batch)
                except:
                    pass
        
        finally:
            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            self.metrics.processing_times.append(processing_time)
    
    def _process_window(self, window: List[ConsumerRecord]):
        """Process window of messages"""
        # Similar to batch processing but for windowed data
        self._process_batch(window)
    
    def _record_to_event(self, record: ConsumerRecord) -> Optional[StreamEvent]:
        """Convert Kafka record to StreamEvent"""
        try:
            if isinstance(record.value, dict):
                # Reconstruct StreamEvent
                return StreamEvent(
                    timestamp=datetime.fromisoformat(record.value.get('timestamp', datetime.now().isoformat())),
                    channel=record.value.get('channel', 'unknown'),
                    spacecraft=record.value.get('spacecraft', 'unknown'),
                    data=np.array(record.value.get('data', [])),
                    sequence_id=record.value.get('sequence_id', 0),
                    is_anomaly=record.value.get('is_anomaly', False),
                    metadata=record.value.get('metadata', {})
                )
            return None
        except Exception as e:
            logger.debug(f"Could not convert record to StreamEvent: {e}")
            return None
    
    def _update_metrics(self, record: ConsumerRecord):
        """Update consumer metrics"""
        self.metrics.messages_consumed += 1
        self.metrics.bytes_consumed += len(record.value) if isinstance(record.value, (bytes, str)) else 0
        self.metrics.topics_consumed[record.topic] += 1
        self.metrics.last_consume_time = datetime.now()
        
        # Update lag if available
        partition_key = f"{record.topic}-{record.partition}"
        if hasattr(self.consumer, 'end_offsets'):
            try:
                end_offset = self.consumer.end_offsets([TopicPartition(record.topic, record.partition)])[
                    TopicPartition(record.topic, record.partition)
                ]
                lag = end_offset - record.offset
                self.metrics.lag_per_partition[partition_key] = lag
            except:
                pass
    
    def _commit_offset(self, record: ConsumerRecord):
        """Commit offset for single record"""
        try:
            tp = TopicPartition(record.topic, record.partition)
            offset = OffsetAndMetadata(record.offset + 1, None)
            self.consumer.commit({tp: offset})
        except CommitFailedError as e:
            logger.error(f"Failed to commit offset: {e}")
    
    def _commit_batch_offsets(self, batch: List[ConsumerRecord]):
        """Commit offsets for batch"""
        try:
            # Group by topic-partition
            offsets = {}
            for record in batch:
                tp = TopicPartition(record.topic, record.partition)
                if tp not in offsets or record.offset > offsets[tp].offset:
                    offsets[tp] = OffsetAndMetadata(record.offset + 1, None)
            
            if offsets:
                self.consumer.commit(offsets)
        except CommitFailedError as e:
            logger.error(f"Failed to commit batch offsets: {e}")
    
    def add_message_handler(self, handler: Callable):
        """Add handler for individual messages"""
        self.message_handlers.append(handler)
        logger.info(f"Added message handler: {handler.__name__}")
    
    def add_batch_handler(self, handler: Callable):
        """Add handler for batches"""
        self.batch_handlers.append(handler)
        logger.info(f"Added batch handler: {handler.__name__}")
    
    def add_error_handler(self, handler: Callable):
        """Add error handler"""
        self.error_handlers.append(handler)
        logger.info(f"Added error handler: {handler.__name__}")
    
    def seek_to_beginning(self):
        """Seek all partitions to beginning"""
        if self.consumer:
            partitions = self.consumer.assignment()
            self.consumer.seek_to_beginning(*partitions)
            logger.info("Seeked to beginning of all partitions")
    
    def seek_to_end(self):
        """Seek all partitions to end"""
        if self.consumer:
            partitions = self.consumer.assignment()
            self.consumer.seek_to_end(*partitions)
            logger.info("Seeked to end of all partitions")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics"""
        return self.metrics.to_dict()
    
    def pause(self):
        """Pause consumption"""
        if self.consumer:
            self.consumer.pause(self.consumer.assignment())
            logger.info("Consumer paused")
    
    def resume(self):
        """Resume consumption"""
        if self.consumer:
            self.consumer.resume(self.consumer.assignment())
            logger.info("Consumer resumed")


class MockKafkaConsumer:
    """Mock Kafka consumer for testing without Kafka cluster"""
    
    def __init__(self, test_data: Optional[List[Dict]] = None):
        """Initialize mock consumer"""
        self.test_data = test_data or []
        self.position = 0
        self.is_consuming = False
        self.message_handlers = []
        self.batch_handlers = []
        self.metrics = ConsumerMetrics()
        logger.info("Using MockKafkaConsumer (no actual Kafka connection)")
    
    def connect(self) -> bool:
        """Mock connect"""
        return True
    
    def start_consuming(self) -> bool:
        """Mock start consuming"""
        self.is_consuming = True
        
        # Simulate consumption in a thread
        thread = threading.Thread(target=self._mock_consume, daemon=True)
        thread.start()
        return True
    
    def _mock_consume(self):
        """Simulate message consumption"""
        while self.is_consuming and self.position < len(self.test_data):
            # Get next message
            message = self.test_data[self.position]
            self.position += 1
            
            # Create mock StreamEvent
            event = StreamEvent(
                timestamp=datetime.now(),
                channel=message.get('channel', 'test_channel'),
                spacecraft=message.get('spacecraft', 'test'),
                data=np.array(message.get('data', [1, 2, 3])),
                sequence_id=self.position,
                is_anomaly=message.get('is_anomaly', False),
                metadata=message.get('metadata', {})
            )
            
            # Call handlers
            for handler in self.message_handlers:
                handler(event)
            
            self.metrics.messages_consumed += 1
            self.metrics.messages_processed += 1
            
            time.sleep(0.1)  # Simulate processing time
    
    def stop_consuming(self):
        """Mock stop consuming"""
        self.is_consuming = False
    
    def add_message_handler(self, handler: Callable):
        """Add message handler"""
        self.message_handlers.append(handler)
    
    def add_batch_handler(self, handler: Callable):
        """Add batch handler"""
        self.batch_handlers.append(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get mock metrics"""
        return self.metrics.to_dict()


def create_kafka_consumer(
    topics: Optional[List[str]] = None,
    use_mock: bool = False,
    test_data: Optional[List[Dict]] = None
) -> Union[KafkaStreamConsumer, MockKafkaConsumer]:
    """
    Factory function to create appropriate consumer
    
    Args:
        topics: Topics to subscribe to
        use_mock: Use mock consumer if True or if Kafka not available
        test_data: Test data for mock consumer
        
    Returns:
        Consumer instance
    """
    if use_mock or not KAFKA_AVAILABLE:
        return MockKafkaConsumer(test_data)
    else:
        return KafkaStreamConsumer(topics=topics)


if __name__ == "__main__":
    # Test Kafka consumer
    print("\n" + "="*60)
    print("Testing Kafka Consumer")
    print("="*60)
    
    # Check if Kafka is available
    if not KAFKA_AVAILABLE:
        print("Kafka not installed. Using mock consumer for testing.")
        use_mock = True
    else:
        # Check if Kafka is running
        try:
            consumer = KafkaStreamConsumer()
            if not consumer.connect():
                print("Kafka not running. Using mock consumer for testing.")
                use_mock = True
            else:
                consumer.stop_consuming()
                use_mock = False
        except:
            use_mock = True
    
    # Create test data for mock consumer
    test_data = [
        {
            'channel': 'channel_1',
            'spacecraft': 'smap',
            'data': np.random.randn(100).tolist(),
            'is_anomaly': False
        },
        {
            'channel': 'channel_2',
            'spacecraft': 'msl',
            'data': np.random.randn(100).tolist(),
            'is_anomaly': True
        }
    ] * 5
    
    # Create consumer
    print("Creating Kafka consumer...")
    consumer = create_kafka_consumer(
        topics=['iot-smap-telemetry', 'iot-msl-telemetry'],
        use_mock=use_mock,
        test_data=test_data
    )
    
    # Define message handler
    def handle_message(message):
        if isinstance(message, StreamEvent):
            print(f"Received: {message.spacecraft}/{message.channel} "
                  f"[{'ANOMALY' if message.is_anomaly else 'NORMAL'}]")
        else:
            print(f"Received message: {message}")
    
    # Define batch handler
    def handle_batch(batch):
        print(f"Processing batch of {len(batch)} messages")
    
    # Add handlers
    consumer.add_message_handler(handle_message)
    
    if not use_mock:
        consumer.add_batch_handler(handle_batch)
    
    # Start consuming
    print("\nStarting consumer...")
    if consumer.connect() if not use_mock else True:
        consumer.start_consuming()
        
        try:
            # Run for 10 seconds
            time.sleep(10)
            
            # Get metrics
            metrics = consumer.get_metrics()
            print("\nConsumer Metrics:")
            print(f"  Messages consumed: {metrics['messages_consumed']}")
            print(f"  Messages processed: {metrics['messages_processed']}")
            print(f"  Throughput: {metrics['throughput']:.2f} msg/sec")
            print(f"  Avg processing time: {metrics['avg_processing_time_ms']:.2f} ms")
            
        except KeyboardInterrupt:
            print("\nStopping consumer...")
        finally:
            consumer.stop_consuming()
    
    print("\n" + "="*60)
    print("Kafka consumer test complete")
    print("="*60)
