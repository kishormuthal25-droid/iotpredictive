"""
Scheduler Module for IoT Anomaly Detection System
Automated task scheduling, model training, and system monitoring
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
import asyncio
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from collections import defaultdict, deque
import schedule
import time
import psutil
import traceback
from enum import Enum
from queue import Queue, PriorityQueue
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    IDLE = 5


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class TaskConfig:
    """Configuration for scheduled tasks"""
    name: str
    function: Callable
    schedule_type: str  # 'cron', 'interval', 'once', 'event'
    schedule_params: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 3600  # seconds
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    notification_on_failure: bool = True
    notification_on_success: bool = False
    enabled: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.schedule_type not in ['cron', 'interval', 'once', 'event']:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")


@dataclass
class TaskResult:
    """Result of task execution"""
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    result: Any
    error: Optional[str]
    retries: int
    resource_usage: Dict[str, Any]


class TaskScheduler:
    """Main scheduler for automated tasks"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_redis: bool = False,
                 redis_config: Optional[Dict] = None):
        """Initialize Task Scheduler
        
        Args:
            max_workers: Maximum concurrent workers
            use_redis: Whether to use Redis for distributed scheduling
            redis_config: Redis configuration
        """
        self.max_workers = max_workers
        self.use_redis = use_redis
        self.tasks = {}
        self.task_queue = PriorityQueue()
        self.results_history = deque(maxlen=1000)
        self.running_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max_workers // 2)
        
        # Initialize Redis if configured
        if use_redis and redis_config:
            self.redis_client = redis.Redis(**redis_config)
        else:
            self.redis_client = None
            
        # Task dependencies graph
        self.dependency_graph = defaultdict(list)
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Notification system
        self.notifier = NotificationSystem()
        
        # Schedule monitoring
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        logger.info(f"Initialized Task Scheduler with {max_workers} workers")
        
    def register_task(self, task_config: TaskConfig):
        """Register a new task for scheduling
        
        Args:
            task_config: Task configuration
        """
        if task_config.name in self.tasks:
            logger.warning(f"Task {task_config.name} already registered, updating...")
            
        self.tasks[task_config.name] = task_config
        
        # Set up dependencies
        for dep in task_config.dependencies:
            self.dependency_graph[dep].append(task_config.name)
            
        # Schedule based on type
        if task_config.schedule_type == 'interval':
            self._schedule_interval_task(task_config)
        elif task_config.schedule_type == 'cron':
            self._schedule_cron_task(task_config)
        elif task_config.schedule_type == 'once':
            self._schedule_once_task(task_config)
            
        logger.info(f"Registered task: {task_config.name}")
        
    def _schedule_interval_task(self, task_config: TaskConfig):
        """Schedule task to run at intervals"""
        interval = task_config.schedule_params.get('interval', 3600)
        
        def job():
            self.execute_task(task_config.name)
            
        schedule.every(interval).seconds.do(job)
        
    def _schedule_cron_task(self, task_config: TaskConfig):
        """Schedule task using cron-like syntax"""
        cron_params = task_config.schedule_params
        
        def job():
            self.execute_task(task_config.name)
            
        # Parse cron parameters
        if 'daily_at' in cron_params:
            schedule.every().day.at(cron_params['daily_at']).do(job)
        elif 'hourly_at' in cron_params:
            schedule.every().hour.at(cron_params['hourly_at']).do(job)
        elif 'weekly_on' in cron_params:
            day = cron_params['weekly_on']
            time_at = cron_params.get('at', '00:00')
            getattr(schedule.every(), day.lower()).at(time_at).do(job)
            
    def _schedule_once_task(self, task_config: TaskConfig):
        """Schedule task to run once at specific time"""
        run_at = task_config.schedule_params.get('run_at')
        if isinstance(run_at, str):
            run_at = datetime.fromisoformat(run_at)
            
        delay = (run_at - datetime.now()).total_seconds()
        if delay > 0:
            threading.Timer(delay, lambda: self.execute_task(task_config.name)).start()
            
    def execute_task(self, task_name: str) -> TaskResult:
        """Execute a registered task
        
        Args:
            task_name: Name of the task to execute
            
        Returns:
            TaskResult object
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not registered")
            
        task_config = self.tasks[task_name]
        
        if not task_config.enabled:
            logger.info(f"Task {task_name} is disabled, skipping...")
            return None
            
        # Check resource availability
        if not self._check_resources(task_config):
            logger.warning(f"Insufficient resources for task {task_name}, queuing...")
            self.task_queue.put((task_config.priority.value, task_name))
            return None
            
        # Check dependencies
        if not self._check_dependencies(task_config):
            logger.info(f"Dependencies not met for task {task_name}, deferring...")
            return None
            
        # Execute task
        result = TaskResult(
            task_name=task_name,
            status=TaskStatus.RUNNING,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            result=None,
            error=None,
            retries=0,
            resource_usage={}
        )
        
        self.running_tasks[task_name] = result
        
        # Submit to executor
        if task_config.resource_requirements.get('use_process', False):
            future = self.process_executor.submit(
                self._execute_task_wrapper, task_config, result
            )
        else:
            future = self.executor.submit(
                self._execute_task_wrapper, task_config, result
            )
            
        return result
        
    def _execute_task_wrapper(self, task_config: TaskConfig, result: TaskResult) -> TaskResult:
        """Wrapper for task execution with error handling"""
        max_retries = task_config.max_retries
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Monitor resources during execution
                initial_resources = self.resource_monitor.get_current_usage()
                
                # Execute the task function
                task_output = task_config.function()
                
                # Update result
                result.status = TaskStatus.COMPLETED
                result.end_time = datetime.now()
                result.duration = (result.end_time - result.start_time).total_seconds()
                result.result = task_output
                result.resource_usage = self.resource_monitor.get_resource_delta(initial_resources)
                
                # Send success notification if configured
                if task_config.notification_on_success:
                    self.notifier.send_notification(
                        f"Task {task_config.name} completed successfully",
                        f"Duration: {result.duration:.2f}s"
                    )
                    
                break
                
            except Exception as e:
                retry_count += 1
                result.retries = retry_count
                
                if retry_count <= max_retries:
                    result.status = TaskStatus.RETRYING
                    logger.warning(f"Task {task_config.name} failed (attempt {retry_count}/{max_retries}): {str(e)}")
                    time.sleep(task_config.retry_delay)
                else:
                    result.status = TaskStatus.FAILED
                    result.error = str(e)
                    result.end_time = datetime.now()
                    result.duration = (result.end_time - result.start_time).total_seconds()
                    
                    # Send failure notification
                    if task_config.notification_on_failure:
                        self.notifier.send_notification(
                            f"Task {task_config.name} failed",
                            f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
                        )
                        
                    logger.error(f"Task {task_config.name} failed after {max_retries} retries: {str(e)}")
                    
        # Clean up and store result
        if task_config.name in self.running_tasks:
            del self.running_tasks[task_config.name]
        self.results_history.append(result)
        
        # Trigger dependent tasks if successful
        if result.status == TaskStatus.COMPLETED:
            self._trigger_dependent_tasks(task_config.name)
            
        return result
        
    def _check_resources(self, task_config: TaskConfig) -> bool:
        """Check if required resources are available"""
        requirements = task_config.resource_requirements
        
        if not requirements:
            return True
            
        current_usage = self.resource_monitor.get_current_usage()
        
        # Check CPU
        if 'min_cpu_percent' in requirements:
            if current_usage['cpu_percent'] > (100 - requirements['min_cpu_percent']):
                return False
                
        # Check memory
        if 'min_memory_mb' in requirements:
            if current_usage['memory_available_mb'] < requirements['min_memory_mb']:
                return False
                
        # Check disk
        if 'min_disk_gb' in requirements:
            if current_usage['disk_available_gb'] < requirements['min_disk_gb']:
                return False
                
        return True
        
    def _check_dependencies(self, task_config: TaskConfig) -> bool:
        """Check if task dependencies are satisfied"""
        for dep in task_config.dependencies:
            # Check if dependency has completed successfully
            dep_completed = False
            for result in reversed(self.results_history):
                if result.task_name == dep and result.status == TaskStatus.COMPLETED:
                    # Check if completed recently (within last hour)
                    if (datetime.now() - result.end_time).total_seconds() < 3600:
                        dep_completed = True
                        break
                        
            if not dep_completed:
                return False
                
        return True
        
    def _trigger_dependent_tasks(self, completed_task: str):
        """Trigger tasks that depend on the completed task"""
        dependent_tasks = self.dependency_graph.get(completed_task, [])
        
        for task_name in dependent_tasks:
            if task_name in self.tasks:
                logger.info(f"Triggering dependent task: {task_name}")
                self.execute_task(task_name)
                
    def start(self):
        """Start the scheduler"""
        logger.info("Starting Task Scheduler...")
        
        # Start monitoring thread
        self._monitoring_thread = threading.Thread(target=self._monitor_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        # Start schedule loop
        self._schedule_thread = threading.Thread(target=self._schedule_loop)
        self._schedule_thread.daemon = True
        self._schedule_thread.start()
        
        logger.info("Task Scheduler started")
        
    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping Task Scheduler...")
        
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        if self._schedule_thread:
            self._schedule_thread.join(timeout=5)
            
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Task Scheduler stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Process queued tasks
                if not self.task_queue.empty():
                    priority, task_name = self.task_queue.get(timeout=1)
                    if self._check_resources(self.tasks[task_name]):
                        self.execute_task(task_name)
                    else:
                        # Re-queue if resources still not available
                        self.task_queue.put((priority, task_name))
                        
                # Monitor running tasks
                for task_name, result in list(self.running_tasks.items()):
                    if result.start_time:
                        duration = (datetime.now() - result.start_time).total_seconds()
                        task_config = self.tasks.get(task_name)
                        if task_config and duration > task_config.timeout:
                            logger.warning(f"Task {task_name} exceeded timeout, marking as failed")
                            result.status = TaskStatus.FAILED
                            result.error = "Task timeout exceeded"
                            
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                
            time.sleep(5)  # Check every 5 seconds
            
    def _schedule_loop(self):
        """Main schedule loop"""
        while not self._stop_event.is_set():
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"Error in schedule loop: {str(e)}")
            time.sleep(1)
            
    def get_task_status(self, task_name: str) -> Optional[TaskStatus]:
        """Get current status of a task"""
        if task_name in self.running_tasks:
            return self.running_tasks[task_name].status
            
        # Check recent history
        for result in reversed(self.results_history):
            if result.task_name == task_name:
                return result.status
                
        return None
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = {
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'queued_tasks': self.task_queue.qsize(),
            'completed_tasks': sum(1 for r in self.results_history if r.status == TaskStatus.COMPLETED),
            'failed_tasks': sum(1 for r in self.results_history if r.status == TaskStatus.FAILED),
            'task_details': {}
        }
        
        # Add per-task statistics
        for task_name in self.tasks:
            task_results = [r for r in self.results_history if r.task_name == task_name]
            if task_results:
                stats['task_details'][task_name] = {
                    'executions': len(task_results),
                    'successes': sum(1 for r in task_results if r.status == TaskStatus.COMPLETED),
                    'failures': sum(1 for r in task_results if r.status == TaskStatus.FAILED),
                    'avg_duration': np.mean([r.duration for r in task_results if r.duration])
                }
                
        return stats


class ResourceMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        """Initialize Resource Monitor"""
        self.history = deque(maxlen=100)
        self._monitoring = False
        
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        usage = {
            'timestamp': datetime.now(),
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'memory_percent': memory.percent,
            'memory_used_mb': memory.used / (1024 * 1024),
            'memory_available_mb': memory.available / (1024 * 1024),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024 * 1024 * 1024),
            'disk_available_gb': disk.free / (1024 * 1024 * 1024)
        }
        
        self.history.append(usage)
        return usage
        
    def get_resource_delta(self, initial_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource usage delta"""
        current = self.get_current_usage()
        
        delta = {
            'cpu_percent_delta': current['cpu_percent'] - initial_usage['cpu_percent'],
            'memory_mb_delta': (current['memory_used_mb'] - initial_usage['memory_used_mb']),
            'disk_gb_delta': (current['disk_used_gb'] - initial_usage['disk_used_gb'])
        }
        
        return delta
        
    def is_resource_available(self, requirements: Dict[str, Any]) -> bool:
        """Check if resources meet requirements"""
        current = self.get_current_usage()
        
        if 'max_cpu_percent' in requirements:
            if current['cpu_percent'] > requirements['max_cpu_percent']:
                return False
                
        if 'min_memory_mb' in requirements:
            if current['memory_available_mb'] < requirements['min_memory_mb']:
                return False
                
        if 'min_disk_gb' in requirements:
            if current['disk_available_gb'] < requirements['min_disk_gb']:
                return False
                
        return True


class NotificationSystem:
    """Handle notifications for task events"""
    
    def __init__(self, smtp_config: Optional[Dict] = None):
        """Initialize Notification System
        
        Args:
            smtp_config: SMTP configuration for email notifications
        """
        self.smtp_config = smtp_config or {}
        self.notification_history = deque(maxlen=100)
        
    def send_notification(self, subject: str, message: str, 
                         notification_type: str = 'info'):
        """Send notification
        
        Args:
            subject: Notification subject
            message: Notification message
            notification_type: Type of notification (info, warning, error)
        """
        notification = {
            'timestamp': datetime.now(),
            'subject': subject,
            'message': message,
            'type': notification_type
        }
        
        self.notification_history.append(notification)
        
        # Log the notification
        if notification_type == 'error':
            logger.error(f"{subject}: {message}")
        elif notification_type == 'warning':
            logger.warning(f"{subject}: {message}")
        else:
            logger.info(f"{subject}: {message}")
            
        # Send email if configured
        if self.smtp_config:
            self._send_email(subject, message)
            
    def _send_email(self, subject: str, message: str):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config.get('from_email')
            msg['To'] = self.smtp_config.get('to_email')
            msg['Subject'] = f"[IoT Anomaly Detection] {subject}"
            
            body = f"""
            Timestamp: {datetime.now()}
            
            {message}
            
            ---
            Automated notification from IoT Anomaly Detection System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_config.get('smtp_host'), 
                             self.smtp_config.get('smtp_port', 587)) as server:
                if self.smtp_config.get('use_tls', True):
                    server.starttls()
                if self.smtp_config.get('username'):
                    server.login(
                        self.smtp_config.get('username'),
                        self.smtp_config.get('password')
                    )
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email notification: {str(e)}")


class ModelTrainingScheduler:
    """Specialized scheduler for model training tasks"""
    
    def __init__(self, scheduler: TaskScheduler):
        """Initialize Model Training Scheduler
        
        Args:
            scheduler: Main task scheduler instance
        """
        self.scheduler = scheduler
        self.training_configs = {}
        
    def schedule_model_training(self,
                               model_name: str,
                               train_function: Callable,
                               data_loader: Callable,
                               evaluation_function: Callable,
                               schedule_params: Dict[str, Any],
                               retrain_threshold: float = 0.1):
        """Schedule automated model training
        
        Args:
            model_name: Name of the model
            train_function: Function to train the model
            data_loader: Function to load training data
            evaluation_function: Function to evaluate model
            schedule_params: Scheduling parameters
            retrain_threshold: Performance drop threshold for retraining
        """
        
        def training_task():
            """Complete training task"""
            logger.info(f"Starting training for model: {model_name}")
            
            # Load data
            data = data_loader()
            
            # Train model
            model = train_function(data)
            
            # Evaluate model
            metrics = evaluation_function(model, data)
            
            # Check if retraining is needed
            if self._should_retrain(model_name, metrics, retrain_threshold):
                logger.info(f"Retraining {model_name} due to performance degradation")
                model = train_function(data)
                metrics = evaluation_function(model, data)
                
            # Save model and metrics
            self._save_training_results(model_name, model, metrics)
            
            return {'model': model_name, 'metrics': metrics}
            
        # Create task configuration
        task_config = TaskConfig(
            name=f"train_{model_name}",
            function=training_task,
            schedule_type=schedule_params.get('type', 'interval'),
            schedule_params=schedule_params,
            priority=TaskPriority.HIGH,
            max_retries=2,
            timeout=7200,  # 2 hours
            resource_requirements={'min_cpu_percent': 20, 'min_memory_mb': 1024},
            notification_on_failure=True,
            notification_on_success=True
        )
        
        # Register with main scheduler
        self.scheduler.register_task(task_config)
        
        # Store configuration
        self.training_configs[model_name] = {
            'train_function': train_function,
            'data_loader': data_loader,
            'evaluation_function': evaluation_function,
            'schedule_params': schedule_params,
            'retrain_threshold': retrain_threshold
        }
        
    def _should_retrain(self, model_name: str, 
                       current_metrics: Dict[str, float],
                       threshold: float) -> bool:
        """Check if model should be retrained based on performance"""
        # Load previous metrics
        try:
            with open(f"models/{model_name}_metrics.json", 'r') as f:
                previous_metrics = json.load(f)
                
            # Compare key metrics
            for metric in ['accuracy', 'f1_score', 'rmse']:
                if metric in current_metrics and metric in previous_metrics:
                    if metric in ['accuracy', 'f1_score']:
                        # Higher is better
                        if current_metrics[metric] < previous_metrics[metric] * (1 - threshold):
                            return True
                    else:
                        # Lower is better
                        if current_metrics[metric] > previous_metrics[metric] * (1 + threshold):
                            return True
                            
        except FileNotFoundError:
            # No previous metrics, don't retrain
            pass
            
        return False
        
    def _save_training_results(self, model_name: str, 
                              model: Any, 
                              metrics: Dict[str, float]):
        """Save training results"""
        # Save model
        model_path = f"models/{model_name}_latest.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metrics
        metrics_path = f"models/{model_name}_metrics.json"
        metrics['timestamp'] = datetime.now().isoformat()
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Saved training results for {model_name}")


class DataProcessingScheduler:
    """Specialized scheduler for data processing pipelines"""
    
    def __init__(self, scheduler: TaskScheduler):
        """Initialize Data Processing Scheduler
        
        Args:
            scheduler: Main task scheduler instance
        """
        self.scheduler = scheduler
        self.pipeline_configs = {}
        
    def schedule_data_pipeline(self,
                              pipeline_name: str,
                              pipeline_stages: List[Callable],
                              input_source: str,
                              output_destination: str,
                              schedule_params: Dict[str, Any],
                              batch_size: int = 1000):
        """Schedule data processing pipeline
        
        Args:
            pipeline_name: Name of the pipeline
            pipeline_stages: List of processing functions
            input_source: Data input source
            output_destination: Data output destination
            schedule_params: Scheduling parameters
            batch_size: Batch size for processing
        """
        
        def pipeline_task():
            """Execute data pipeline"""
            logger.info(f"Starting data pipeline: {pipeline_name}")
            
            # Load data in batches
            processed_count = 0
            failed_count = 0
            
            try:
                # Initialize input source
                data_iterator = self._get_data_iterator(input_source, batch_size)
                
                for batch in data_iterator:
                    try:
                        # Process through pipeline stages
                        result = batch
                        for stage in pipeline_stages:
                            result = stage(result)
                            
                        # Save results
                        self._save_results(result, output_destination)
                        processed_count += len(batch)
                        
                    except Exception as e:
                        logger.error(f"Failed to process batch in {pipeline_name}: {str(e)}")
                        failed_count += len(batch)
                        
                logger.info(f"Pipeline {pipeline_name} completed: {processed_count} processed, {failed_count} failed")
                
                return {
                    'pipeline': pipeline_name,
                    'processed': processed_count,
                    'failed': failed_count
                }
                
            except Exception as e:
                logger.error(f"Pipeline {pipeline_name} failed: {str(e)}")
                raise
                
        # Create task configuration
        task_config = TaskConfig(
            name=f"pipeline_{pipeline_name}",
            function=pipeline_task,
            schedule_type=schedule_params.get('type', 'interval'),
            schedule_params=schedule_params,
            priority=TaskPriority.NORMAL,
            max_retries=3,
            timeout=3600,
            resource_requirements={'min_memory_mb': 512},
            notification_on_failure=True
        )
        
        # Register with main scheduler
        self.scheduler.register_task(task_config)
        
        # Store configuration
        self.pipeline_configs[pipeline_name] = {
            'stages': pipeline_stages,
            'input_source': input_source,
            'output_destination': output_destination,
            'batch_size': batch_size
        }
        
    def _get_data_iterator(self, source: str, batch_size: int):
        """Get data iterator based on source type"""
        if source.endswith('.csv'):
            return pd.read_csv(source, chunksize=batch_size)
        elif source.endswith('.parquet'):
            df = pd.read_parquet(source)
            for i in range(0, len(df), batch_size):
                yield df.iloc[i:i+batch_size]
        else:
            # Assume it's a directory
            for file in Path(source).glob('*.csv'):
                yield pd.read_csv(file)
                
    def _save_results(self, data: pd.DataFrame, destination: str):
        """Save processed results"""
        if destination.endswith('.csv'):
            data.to_csv(destination, mode='a', header=False, index=False)
        elif destination.endswith('.parquet'):
            data.to_parquet(destination, engine='pyarrow')
        else:
            # Save to timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{destination}/processed_{timestamp}.parquet"
            data.to_parquet(output_file)