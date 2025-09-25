"""
Stream Simulator Module for Real-time Telemetry Data Streaming
Simulates real-time streaming of NASA SMAP/MSL data from static files
"""

import asyncio
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Generator
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import random

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config
from src.data_ingestion.data_loader import DataLoader, TelemetryData

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    """Represents a single streaming event"""
    timestamp: datetime
    channel: str
    spacecraft: str
    data: np.ndarray
    sequence_id: int
    is_anomaly: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'channel': self.channel,
            'spacecraft': self.spacecraft,
            'data': self.data.tolist() if isinstance(self.data, np.ndarray) else self.data,
            'sequence_id': self.sequence_id,
            'is_anomaly': self.is_anomaly,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class StreamStatistics:
    """Statistics for stream monitoring"""
    total_events: int = 0
    events_per_channel: Dict[str, int] = field(default_factory=dict)
    anomalies_detected: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_event_time: datetime = field(default_factory=datetime.now)
    throughput: float = 0.0
    buffer_size: int = 0
    dropped_events: int = 0
    
    def update_throughput(self):
        """Calculate current throughput"""
        elapsed = (self.last_event_time - self.start_time).total_seconds()
        if elapsed > 0:
            self.throughput = self.total_events / elapsed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_events': self.total_events,
            'events_per_channel': self.events_per_channel,
            'anomalies_detected': self.anomalies_detected,
            'start_time': self.start_time.isoformat(),
            'last_event_time': self.last_event_time.isoformat(),
            'throughput': self.throughput,
            'buffer_size': self.buffer_size,
            'dropped_events': self.dropped_events,
            'runtime_seconds': (self.last_event_time - self.start_time).total_seconds()
        }


class StreamSimulator:
    """
    Simulates real-time streaming of telemetry data
    Supports multiple channels, configurable speed, and various output modes
    """
    
    def __init__(self,
                 spacecraft: Union[str, List[str]] = "smap",
                 channels: Optional[List[str]] = None,
                 speed_multiplier: float = 1.0,
                 batch_size: int = 100,
                 buffer_size: int = 1000,
                 window_size: int = 100,
                 stride: int = 10,
                 loop_data: bool = True,
                 add_noise: bool = False,
                 noise_level: float = 0.01):
        """
        Initialize stream simulator
        
        Args:
            spacecraft: Spacecraft name(s) to simulate
            channels: Specific channels to stream (None = all)
            speed_multiplier: Speed of simulation (1.0 = real-time, 2.0 = 2x speed)
            batch_size: Number of samples to process in batch
            buffer_size: Size of internal buffer
            window_size: Size of sliding window for streaming
            stride: Step size for sliding window
            loop_data: Whether to loop when reaching end of data
            add_noise: Whether to add random noise to simulate real conditions
            noise_level: Standard deviation of noise to add
        """
        # Handle spacecraft input
        if isinstance(spacecraft, str):
            self.spacecraft_list = [spacecraft.lower()]
        else:
            self.spacecraft_list = [s.lower() for s in spacecraft]
        
        self.channels = channels
        self.speed_multiplier = max(0.1, speed_multiplier)  # Minimum 0.1x speed
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.stride = stride
        self.loop_data = loop_data
        self.add_noise = add_noise
        self.noise_level = noise_level
        
        # Load configuration
        self.config = {
            'enabled': get_config('data_ingestion.simulation.enabled', True),
            'speed': get_config('data_ingestion.simulation.speed_multiplier', 1.0),
            'batch': get_config('data_ingestion.simulation.batch_size', 100),
            'buffer': get_config('data_ingestion.simulation.buffer_size', 1000)
        }
        
        # Override with config if not explicitly set
        if speed_multiplier == 1.0:
            self.speed_multiplier = self.config['speed']
        
        # Data loaders for each spacecraft
        self.data_loaders: Dict[str, DataLoader] = {}
        self.telemetry_data: Dict[str, Dict[str, TelemetryData]] = {}
        
        # Streaming state
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.event_buffer = queue.Queue(maxsize=self.buffer_size)
        self.subscribers: List[Callable] = []
        
        # Statistics
        self.stats = StreamStatistics()
        
        # Position tracking for each channel
        self.channel_positions: Dict[str, int] = {}
        self.sequence_counter = 0
        
        # Async event loop for async subscribers
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"StreamSimulator initialized for {self.spacecraft_list}")
        logger.info(f"Speed multiplier: {self.speed_multiplier}x")
    
    def load_data(self) -> bool:
        """
        Load telemetry data for all configured spacecraft
        
        Returns:
            True if data loaded successfully
        """
        success = True
        
        for spacecraft in self.spacecraft_list:
            try:
                logger.info(f"Loading data for {spacecraft.upper()}...")
                
                # Create data loader
                loader = DataLoader(
                    spacecraft=spacecraft,
                    normalize=True,
                    cache_processed=True,
                    verbose=False
                )
                
                # Load all data
                data = loader.load_all_data()
                
                if not data:
                    logger.warning(f"No data found for {spacecraft}")
                    success = False
                    continue
                
                self.data_loaders[spacecraft] = loader
                self.telemetry_data[spacecraft] = data
                
                # Filter channels if specified
                if self.channels:
                    filtered_data = {
                        ch: data[ch] for ch in self.channels 
                        if ch in data
                    }
                    self.telemetry_data[spacecraft] = filtered_data
                
                # Initialize positions
                for channel in self.telemetry_data[spacecraft]:
                    self.channel_positions[f"{spacecraft}_{channel}"] = 0
                
                logger.info(f"Loaded {len(self.telemetry_data[spacecraft])} channels for {spacecraft}")
                
            except Exception as e:
                logger.error(f"Error loading data for {spacecraft}: {e}")
                success = False
        
        return success
    
    def start_streaming(self, async_mode: bool = False) -> bool:
        """
        Start the streaming simulation
        
        Args:
            async_mode: Whether to run in async mode
            
        Returns:
            True if streaming started successfully
        """
        if self.is_streaming:
            logger.warning("Streaming already in progress")
            return False
        
        # Load data if not already loaded
        if not self.telemetry_data:
            if not self.load_data():
                logger.error("Failed to load data")
                return False
        
        self.is_streaming = True
        self.stats = StreamStatistics()  # Reset statistics
        
        if async_mode:
            # Start async streaming
            self.loop = asyncio.new_event_loop()
            self.stream_thread = threading.Thread(
                target=self._run_async_stream,
                daemon=True
            )
        else:
            # Start sync streaming
            self.stream_thread = threading.Thread(
                target=self._run_stream,
                daemon=True
            )
        
        self.stream_thread.start()
        logger.info("Streaming started")
        return True
    
    def stop_streaming(self):
        """Stop the streaming simulation"""
        if not self.is_streaming:
            logger.warning("Streaming not in progress")
            return
        
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
            
        if self.loop:
            self.loop.stop()
            
        logger.info("Streaming stopped")
        logger.info(f"Final statistics: {self.stats.to_dict()}")
    
    def _run_stream(self):
        """Main streaming loop (synchronous)"""
        logger.info("Starting synchronous streaming loop")
        
        # Calculate delay between samples based on speed multiplier
        # Assume 1 sample per second in real-time
        sample_delay = 1.0 / self.speed_multiplier
        
        while self.is_streaming:
            try:
                # Generate batch of events
                events = self._generate_batch()
                
                if not events:
                    if self.loop_data:
                        self._reset_positions()
                        continue
                    else:
                        logger.info("End of data reached")
                        break
                
                # Process each event
                for event in events:
                    # Add to buffer
                    try:
                        self.event_buffer.put_nowait(event)
                    except queue.Full:
                        self.stats.dropped_events += 1
                        # Remove oldest event and try again
                        try:
                            self.event_buffer.get_nowait()
                            self.event_buffer.put_nowait(event)
                        except:
                            pass
                    
                    # Notify subscribers
                    self._notify_subscribers(event)
                    
                    # Update statistics
                    self._update_statistics(event)
                    
                    # Sleep to simulate real-time
                    time.sleep(sample_delay)
                    
                    if not self.is_streaming:
                        break
                        
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(1)
    
    def _run_async_stream(self):
        """Main streaming loop (asynchronous)"""
        logger.info("Starting asynchronous streaming loop")
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_stream())
    
    async def _async_stream(self):
        """Async streaming implementation"""
        sample_delay = 1.0 / self.speed_multiplier
        
        while self.is_streaming:
            try:
                events = self._generate_batch()
                
                if not events:
                    if self.loop_data:
                        self._reset_positions()
                        continue
                    else:
                        logger.info("End of data reached")
                        break
                
                # Process events asynchronously
                tasks = []
                for event in events:
                    tasks.append(self._process_event_async(event))
                
                await asyncio.gather(*tasks)
                await asyncio.sleep(sample_delay * len(events))
                
            except Exception as e:
                logger.error(f"Error in async streaming loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_async(self, event: StreamEvent):
        """Process a single event asynchronously"""
        try:
            self.event_buffer.put_nowait(event)
        except queue.Full:
            self.stats.dropped_events += 1
        
        self._notify_subscribers(event)
        self._update_statistics(event)
    
    def _generate_batch(self) -> List[StreamEvent]:
        """
        Generate a batch of streaming events
        
        Returns:
            List of stream events
        """
        events = []
        
        for spacecraft, channels_data in self.telemetry_data.items():
            for channel_name, telemetry in channels_data.items():
                # Get current position
                pos_key = f"{spacecraft}_{channel_name}"
                current_pos = self.channel_positions.get(pos_key, 0)
                
                # Check if we have enough data
                if current_pos + self.window_size > len(telemetry.data):
                    continue
                
                # Extract window of data
                data_window = telemetry.data[current_pos:current_pos + self.window_size]
                
                # Add noise if configured
                if self.add_noise:
                    noise = np.random.normal(0, self.noise_level, data_window.shape)
                    data_window = data_window + noise
                
                # Check if this window contains anomalies
                is_anomaly = self._check_anomaly(
                    telemetry,
                    current_pos,
                    current_pos + self.window_size
                )
                
                # Create stream event
                event = StreamEvent(
                    timestamp=datetime.now(),
                    channel=channel_name,
                    spacecraft=spacecraft,
                    data=data_window,
                    sequence_id=self.sequence_counter,
                    is_anomaly=is_anomaly,
                    metadata={
                        'window_start': current_pos,
                        'window_end': current_pos + self.window_size,
                        'window_size': self.window_size
                    }
                )
                
                events.append(event)
                
                # Update position
                self.channel_positions[pos_key] = current_pos + self.stride
                self.sequence_counter += 1
                
                # Limit batch size
                if len(events) >= self.batch_size:
                    break
            
            if len(events) >= self.batch_size:
                break
        
        return events
    
    def _check_anomaly(self, 
                      telemetry: TelemetryData,
                      start_idx: int,
                      end_idx: int) -> bool:
        """
        Check if the current window contains an anomaly
        
        Args:
            telemetry: Telemetry data object
            start_idx: Start index of window
            end_idx: End index of window
            
        Returns:
            True if window contains anomaly
        """
        if not telemetry.anomaly_sequences:
            return False
        
        for anomaly_start, anomaly_end in telemetry.anomaly_sequences:
            # Check if anomaly overlaps with current window
            if (anomaly_start <= start_idx <= anomaly_end) or \
               (anomaly_start <= end_idx <= anomaly_end) or \
               (start_idx <= anomaly_start <= end_idx):
                return True
        
        return False
    
    def _reset_positions(self):
        """Reset all channel positions to beginning"""
        for key in self.channel_positions:
            self.channel_positions[key] = 0
        logger.info("Data loop - resetting to beginning")
    
    def _notify_subscribers(self, event: StreamEvent):
        """Notify all subscribers of new event"""
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def _update_statistics(self, event: StreamEvent):
        """Update streaming statistics"""
        self.stats.total_events += 1
        self.stats.events_per_channel[event.channel] = \
            self.stats.events_per_channel.get(event.channel, 0) + 1
        
        if event.is_anomaly:
            self.stats.anomalies_detected += 1
        
        self.stats.last_event_time = datetime.now()
        self.stats.buffer_size = self.event_buffer.qsize()
        self.stats.update_throughput()
    
    def subscribe(self, callback: Callable[[StreamEvent], None]):
        """
        Subscribe to stream events
        
        Args:
            callback: Function to call for each event
        """
        self.subscribers.append(callback)
        logger.info(f"Added subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable[[StreamEvent], None]):
        """
        Unsubscribe from stream events
        
        Args:
            callback: Function to remove
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Removed subscriber: {callback.__name__}")
    
    def get_next_event(self, timeout: float = 1.0) -> Optional[StreamEvent]:
        """
        Get next event from buffer
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Next event or None if timeout
        """
        try:
            return self.event_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_batch(self, 
                  batch_size: Optional[int] = None,
                  timeout: float = 1.0) -> List[StreamEvent]:
        """
        Get batch of events from buffer
        
        Args:
            batch_size: Number of events to get
            timeout: Timeout in seconds
            
        Returns:
            List of events
        """
        batch_size = batch_size or self.batch_size
        events = []
        
        end_time = time.time() + timeout
        
        while len(events) < batch_size and time.time() < end_time:
            try:
                event = self.event_buffer.get(timeout=0.1)
                events.append(event)
            except queue.Empty:
                if not self.is_streaming:
                    break
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current streaming statistics"""
        return self.stats.to_dict()
    
    def inject_anomaly(self, 
                      channel: str,
                      spacecraft: str = None,
                      duration: int = 10,
                      severity: float = 2.0):
        """
        Inject synthetic anomaly into stream
        
        Args:
            channel: Channel to inject anomaly
            spacecraft: Spacecraft (if None, use first available)
            duration: Duration of anomaly in samples
            severity: Severity multiplier for anomaly
        """
        if spacecraft is None:
            spacecraft = self.spacecraft_list[0]
        
        if spacecraft not in self.telemetry_data:
            logger.warning(f"Spacecraft {spacecraft} not found")
            return
        
        if channel not in self.telemetry_data[spacecraft]:
            logger.warning(f"Channel {channel} not found")
            return
        
        # Create anomaly event
        telemetry = self.telemetry_data[spacecraft][channel]
        pos_key = f"{spacecraft}_{channel}"
        current_pos = self.channel_positions.get(pos_key, 0)
        
        # Generate anomalous data
        normal_data = telemetry.data[current_pos:current_pos + duration]
        anomaly_data = normal_data * severity + np.random.normal(0, 0.1, normal_data.shape)
        
        # Create anomaly event
        event = StreamEvent(
            timestamp=datetime.now(),
            channel=channel,
            spacecraft=spacecraft,
            data=anomaly_data,
            sequence_id=self.sequence_counter,
            is_anomaly=True,
            metadata={
                'injected': True,
                'severity': severity,
                'duration': duration
            }
        )
        
        # Add to buffer with priority
        self.event_buffer.put(event)
        self.sequence_counter += 1
        
        logger.info(f"Injected anomaly into {spacecraft}/{channel}")
    
    def simulate_failure(self, 
                        channel: str,
                        failure_type: str = "spike",
                        duration: int = 100):
        """
        Simulate specific failure patterns
        
        Args:
            channel: Channel to simulate failure
            failure_type: Type of failure ('spike', 'drift', 'noise', 'flatline')
            duration: Duration of failure
        """
        # Implementation for different failure types
        if failure_type == "spike":
            self.inject_anomaly(channel, duration=1, severity=10.0)
        elif failure_type == "drift":
            for i in range(duration):
                severity = 1.0 + (i / duration) * 2.0
                self.inject_anomaly(channel, duration=1, severity=severity)
                time.sleep(0.1)
        elif failure_type == "noise":
            self.inject_anomaly(channel, duration=duration, severity=0.5)
        elif failure_type == "flatline":
            self.inject_anomaly(channel, duration=duration, severity=0.0)
        else:
            logger.warning(f"Unknown failure type: {failure_type}")


class StreamRecorder:
    """Records streaming data to file for replay"""
    
    def __init__(self, output_file: Path):
        """
        Initialize stream recorder
        
        Args:
            output_file: Path to output file
        """
        self.output_file = Path(output_file)
        self.events: List[StreamEvent] = []
        self.is_recording = False
    
    def start_recording(self):
        """Start recording events"""
        self.is_recording = True
        self.events = []
        logger.info(f"Started recording to {self.output_file}")
    
    def stop_recording(self):
        """Stop recording and save to file"""
        self.is_recording = False
        self._save_events()
        logger.info(f"Saved {len(self.events)} events to {self.output_file}")
    
    def record_event(self, event: StreamEvent):
        """Record a single event"""
        if self.is_recording:
            self.events.append(event)
    
    def _save_events(self):
        """Save recorded events to file"""
        data = [event.to_dict() for event in self.events]
        
        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)


class StreamReplayer:
    """Replays recorded streaming data"""
    
    def __init__(self, input_file: Path):
        """
        Initialize stream replayer
        
        Args:
            input_file: Path to recorded events file
        """
        self.input_file = Path(input_file)
        self.events: List[StreamEvent] = []
        self._load_events()
    
    def _load_events(self):
        """Load events from file"""
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        
        self.events = []
        for item in data:
            event = StreamEvent(
                timestamp=datetime.fromisoformat(item['timestamp']),
                channel=item['channel'],
                spacecraft=item['spacecraft'],
                data=np.array(item['data']),
                sequence_id=item['sequence_id'],
                is_anomaly=item['is_anomaly'],
                metadata=item['metadata']
            )
            self.events.append(event)
        
        logger.info(f"Loaded {len(self.events)} events from {self.input_file}")
    
    def replay(self, speed_multiplier: float = 1.0) -> Generator[StreamEvent, None, None]:
        """
        Replay recorded events
        
        Args:
            speed_multiplier: Speed of replay
            
        Yields:
            Stream events
        """
        if not self.events:
            logger.warning("No events to replay")
            return
        
        start_time = self.events[0].timestamp
        replay_start = datetime.now()
        
        for event in self.events:
            # Calculate when this event should be replayed
            event_offset = (event.timestamp - start_time).total_seconds()
            replay_offset = event_offset / speed_multiplier
            
            # Wait until the right time
            current_offset = (datetime.now() - replay_start).total_seconds()
            if current_offset < replay_offset:
                time.sleep(replay_offset - current_offset)
            
            yield event


if __name__ == "__main__":
    # Test streaming simulator
    print("\n" + "="*60)
    print("Testing Stream Simulator")
    print("="*60)
    
    # Create simulator for both spacecraft
    simulator = StreamSimulator(
        spacecraft=["smap", "msl"],
        speed_multiplier=10.0,  # 10x speed for testing
        window_size=100,
        stride=50,
        add_noise=True
    )
    
    # Define a simple subscriber
    def print_event(event: StreamEvent):
        print(f"[{event.timestamp.strftime('%H:%M:%S')}] "
              f"{event.spacecraft}/{event.channel}: "
              f"shape={event.data.shape}, "
              f"anomaly={event.is_anomaly}")
    
    # Subscribe to events
    simulator.subscribe(print_event)
    
    # Start streaming
    print("\nStarting stream simulation...")
    if simulator.start_streaming():
        try:
            # Run for 10 seconds
            time.sleep(10)
            
            # Get statistics
            stats = simulator.get_statistics()
            print(f"\nStreaming Statistics:")
            print(f"  Total events: {stats['total_events']}")
            print(f"  Throughput: {stats['throughput']:.2f} events/sec")
            print(f"  Anomalies: {stats['anomalies_detected']}")
            
            # Test anomaly injection
            print("\nInjecting anomaly...")
            channels = list(simulator.telemetry_data.get('smap', {}).keys())
            if channels:
                simulator.inject_anomaly(channels[0], severity=5.0)
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nStopping stream...")
        finally:
            simulator.stop_streaming()
    
    print("\n" + "="*60)
    print("Stream simulation test complete")
    print("="*60)
