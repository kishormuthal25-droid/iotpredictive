"""
Logger Utility Module
Provides centralized logging configuration for the IoT Anomaly Detection System
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union
from functools import wraps
import threading
import queue
import colorlog
import yaml

# Default configuration
DEFAULT_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/iot_system.log',
    'max_bytes': 10485760,  # 10MB
    'backup_count': 10,
    'enable_console': True,
    'enable_file': True,
    'enable_color': True,
    'enable_json': False,
    'enable_syslog': False,
    'syslog_host': 'localhost',
    'syslog_port': 514
}

# Thread-local storage for context
context = threading.local()

class ContextFilter(logging.Filter):
    """Add contextual information to log records"""
    
    def filter(self, record):
        """Add context data to log record"""
        # Add request ID if available
        record.request_id = getattr(context, 'request_id', 'N/A')
        
        # Add user ID if available
        record.user_id = getattr(context, 'user_id', 'N/A')
        
        # Add equipment ID if available
        record.equipment_id = getattr(context, 'equipment_id', 'N/A')
        
        # Add additional context
        if hasattr(context, 'extra'):
            for key, value in context.extra.items():
                setattr(record, key, value)
        
        return True

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'request_id': getattr(record, 'request_id', 'N/A'),
            'user_id': getattr(record, 'user_id', 'N/A'),
            'equipment_id': getattr(record, 'equipment_id', 'N/A')
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            log_data['traceback'] = traceback.format_exc()
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'message', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread', 'threadName',
                          'exc_info', 'exc_text', 'stack_info', 'request_id',
                          'user_id', 'equipment_id']:
                log_data[key] = value
        
        return json.dumps(log_data)

class AsyncHandler(logging.Handler):
    """Asynchronous logging handler for better performance"""
    
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.queue = queue.Queue(-1)
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()
    
    def _worker(self):
        """Worker thread for processing log records"""
        while True:
            try:
                record = self.queue.get()
                if record is None:
                    break
                self.handler.emit(record)
            except Exception:
                pass
    
    def emit(self, record):
        """Add record to queue"""
        self.queue.put(record)
    
    def close(self):
        """Close the handler"""
        self.queue.put(None)
        self.thread.join()
        self.handler.close()
        super().close()

class LoggerManager:
    """Centralized logger management"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger manager"""
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.loggers = {}
            self.config = self._load_config()
            self.setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from file"""
        config = DEFAULT_CONFIG.copy()
        
        try:
            config_file = Path('config/config.yaml')
            if config_file.exists():
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    if 'logging' in yaml_config:
                        logging_config = yaml_config['logging']
                        # Handle nested file config
                        if 'file' in logging_config and isinstance(logging_config['file'], dict):
                            if 'path' in logging_config['file']:
                                config['file'] = logging_config['file']['path']
                            # Update other file settings
                            config['max_bytes'] = logging_config['file'].get('max_bytes', config['max_bytes'])
                            config['backup_count'] = logging_config['file'].get('backup_count', config['backup_count'])

                        # Update other logging settings
                        for key in ['level', 'enable_console', 'enable_file']:
                            if key in logging_config:
                                config[key] = logging_config[key]
        except Exception as e:
            print(f"Warning: Could not load logging config: {e}")

        return config
    
    def setup_logging(self):
        """Setup root logger configuration"""
        # Create logs directory if it doesn't exist
        log_file = Path(self.config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level']))
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Add context filter
        context_filter = ContextFilter()
        
        # Console handler
        if self.config['enable_console']:
            console_handler = self._create_console_handler()
            console_handler.addFilter(context_filter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.config['enable_file']:
            file_handler = self._create_file_handler()
            file_handler.addFilter(context_filter)
            root_logger.addHandler(file_handler)
        
        # JSON file handler
        if self.config['enable_json']:
            json_handler = self._create_json_handler()
            json_handler.addFilter(context_filter)
            root_logger.addHandler(json_handler)
        
        # Syslog handler
        if self.config['enable_syslog']:
            syslog_handler = self._create_syslog_handler()
            syslog_handler.addFilter(context_filter)
            root_logger.addHandler(syslog_handler)
    
    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with optional color support"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config['enable_color'] and colorlog:
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            formatter = logging.Formatter(
                self.config['format'],
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, self.config['level']))
        
        return console_handler
    
    def _create_file_handler(self) -> logging.Handler:
        """Create rotating file handler"""
        file_handler = logging.handlers.RotatingFileHandler(
            self.config['file'],
            maxBytes=self.config['max_bytes'],
            backupCount=self.config['backup_count']
        )
        
        formatter = logging.Formatter(
            self.config['format'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, self.config['level']))
        
        # Wrap in async handler for better performance
        return AsyncHandler(file_handler)
    
    def _create_json_handler(self) -> logging.Handler:
        """Create JSON file handler for structured logging"""
        json_file = self.config['file'].replace('.log', '.json')
        
        json_handler = logging.handlers.RotatingFileHandler(
            json_file,
            maxBytes=self.config['max_bytes'],
            backupCount=self.config['backup_count']
        )
        
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(getattr(logging, self.config['level']))
        
        # Wrap in async handler for better performance
        return AsyncHandler(json_handler)
    
    def _create_syslog_handler(self) -> logging.Handler:
        """Create syslog handler for centralized logging"""
        syslog_handler = logging.handlers.SysLogHandler(
            address=(self.config['syslog_host'], self.config['syslog_port'])
        )
        
        formatter = logging.Formatter(
            'iot_system: ' + self.config['format'],
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        syslog_handler.setFormatter(formatter)
        syslog_handler.setLevel(getattr(logging, self.config['level']))
        
        return syslog_handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger instance"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def set_level(self, level: str, logger_name: Optional[str] = None):
        """Set logging level for specific logger or root"""
        level_obj = getattr(logging, level.upper())
        
        if logger_name:
            logger = logging.getLogger(logger_name)
            logger.setLevel(level_obj)
        else:
            logging.getLogger().setLevel(level_obj)
    
    def add_context(self, **kwargs):
        """Add context data to current thread"""
        if not hasattr(context, 'extra'):
            context.extra = {}
        context.extra.update(kwargs)
    
    def clear_context(self):
        """Clear context data for current thread"""
        if hasattr(context, 'extra'):
            context.extra.clear()
    
    def set_request_id(self, request_id: str):
        """Set request ID for current thread"""
        context.request_id = request_id
    
    def set_user_id(self, user_id: str):
        """Set user ID for current thread"""
        context.user_id = user_id
    
    def set_equipment_id(self, equipment_id: str):
        """Set equipment ID for current thread"""
        context.equipment_id = equipment_id

# Singleton instance
_logger_manager = LoggerManager()

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    if name is None:
        name = 'iot_system'
    
    return _logger_manager.get_logger(name)

def set_level(level: str, logger_name: Optional[str] = None):
    """Set logging level
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Optional specific logger name
    """
    _logger_manager.set_level(level, logger_name)

def add_context(**kwargs):
    """Add contextual information to logs
    
    Args:
        **kwargs: Key-value pairs to add to log context
    """
    _logger_manager.add_context(**kwargs)

def clear_context():
    """Clear contextual information"""
    _logger_manager.clear_context()

def set_request_id(request_id: str):
    """Set request ID for log correlation
    
    Args:
        request_id: Unique request identifier
    """
    _logger_manager.set_request_id(request_id)

def set_user_id(user_id: str):
    """Set user ID for log tracking
    
    Args:
        user_id: User identifier
    """
    _logger_manager.set_user_id(user_id)

def set_equipment_id(equipment_id: str):
    """Set equipment ID for log tracking
    
    Args:
        equipment_id: Equipment identifier
    """
    _logger_manager.set_equipment_id(equipment_id)

def log_execution_time(func):
    """Decorator to log function execution time
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = datetime.now()
        
        try:
            logger.debug(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {execution_time:.3f} seconds")
            return result
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}")
            raise
    
    return wrapper

def log_exceptions(func):
    """Decorator to log exceptions with full traceback
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return wrapper

def log_api_call(service: str):
    """Decorator to log external API calls
    
    Args:
        service: Name of the external service
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = datetime.now()
            
            logger.info(f"Calling {service} API: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"{service} API call successful: {execution_time:.3f}s")
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"{service} API call failed: {execution_time:.3f}s - {str(e)}")
                raise
        
        return wrapper
    return decorator

def log_database_operation(operation: str):
    """Decorator to log database operations
    
    Args:
        operation: Type of database operation (SELECT, INSERT, UPDATE, DELETE)
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = datetime.now()
            
            logger.debug(f"Database {operation}: {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Log result count for SELECT operations
                if operation == 'SELECT' and hasattr(result, '__len__'):
                    logger.info(f"Database {operation} completed: {len(result)} rows in {execution_time:.3f}s")
                else:
                    logger.info(f"Database {operation} completed in {execution_time:.3f}s")
                
                return result
            
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Database {operation} failed: {execution_time:.3f}s - {str(e)}")
                raise
        
        return wrapper
    return decorator

class LogContext:
    """Context manager for temporary log context"""
    
    def __init__(self, **kwargs):
        """Initialize context manager
        
        Args:
            **kwargs: Context data to add
        """
        self.context_data = kwargs
        self.previous_context = None
    
    def __enter__(self):
        """Enter context"""
        self.previous_context = getattr(context, 'extra', {}).copy() if hasattr(context, 'extra') else {}
        add_context(**self.context_data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        if hasattr(context, 'extra'):
            context.extra = self.previous_context
        return False

class MetricsLogger:
    """Logger for metrics and performance data"""
    
    def __init__(self, name: str = 'metrics'):
        """Initialize metrics logger
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
        self.metrics = {}
    
    def log_metric(self, name: str, value: Union[int, float], 
                   unit: str = None, tags: Dict[str, str] = None):
        """Log a metric value
        
        Args:
            name: Metric name
            value: Metric value
            unit: Optional unit of measurement
            tags: Optional tags for the metric
        """
        metric_data = {
            'metric': name,
            'value': value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if unit:
            metric_data['unit'] = unit
        
        if tags:
            metric_data['tags'] = tags
        
        self.logger.info(f"METRIC: {json.dumps(metric_data)}")
        
        # Store for aggregation
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def log_counter(self, name: str, increment: int = 1, 
                    tags: Dict[str, str] = None):
        """Log a counter increment
        
        Args:
            name: Counter name
            increment: Increment value
            tags: Optional tags
        """
        self.log_metric(f"counter.{name}", increment, tags=tags)
    
    def log_gauge(self, name: str, value: Union[int, float], 
                  tags: Dict[str, str] = None):
        """Log a gauge value
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
        """
        self.log_metric(f"gauge.{name}", value, tags=tags)
    
    def log_histogram(self, name: str, value: Union[int, float], 
                     tags: Dict[str, str] = None):
        """Log a histogram value
        
        Args:
            name: Histogram name
            value: Value to add to histogram
            tags: Optional tags
        """
        self.log_metric(f"histogram.{name}", value, tags=tags)
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric
        
        Args:
            name: Metric name
        
        Returns:
            Dictionary with summary statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'sum': sum(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values)
        }

class AuditLogger:
    """Logger for audit trails"""
    
    def __init__(self, name: str = 'audit'):
        """Initialize audit logger
        
        Args:
            name: Logger name
        """
        self.logger = get_logger(name)
    
    def log_action(self, action: str, entity: str, entity_id: str,
                   user: str = None, details: Dict[str, Any] = None,
                   result: str = 'SUCCESS'):
        """Log an audit action
        
        Args:
            action: Action performed (CREATE, UPDATE, DELETE, etc.)
            entity: Entity type
            entity_id: Entity identifier
            user: User who performed the action
            details: Additional details
            result: Action result (SUCCESS, FAILURE)
        """
        audit_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'entity': entity,
            'entity_id': entity_id,
            'user': user or 'SYSTEM',
            'result': result
        }
        
        if details:
            audit_data['details'] = details
        
        self.logger.info(f"AUDIT: {json.dumps(audit_data)}")
    
    def log_access(self, resource: str, user: str, 
                   permission: str, granted: bool):
        """Log access control decisions
        
        Args:
            resource: Resource accessed
            user: User requesting access
            permission: Permission required
            granted: Whether access was granted
        """
        self.log_action(
            action='ACCESS',
            entity='RESOURCE',
            entity_id=resource,
            user=user,
            details={
                'permission': permission,
                'granted': granted
            },
            result='GRANTED' if granted else 'DENIED'
        )
    
    def log_data_change(self, entity: str, entity_id: str,
                        field: str, old_value: Any, new_value: Any,
                        user: str = None):
        """Log data changes for audit trail
        
        Args:
            entity: Entity type
            entity_id: Entity identifier
            field: Field that changed
            old_value: Previous value
            new_value: New value
            user: User who made the change
        """
        self.log_action(
            action='UPDATE',
            entity=entity,
            entity_id=entity_id,
            user=user,
            details={
                'field': field,
                'old_value': str(old_value),
                'new_value': str(new_value)
            }
        )

# Create singleton instances
metrics_logger = MetricsLogger()
audit_logger = AuditLogger()

# Export convenience functions
log_metric = metrics_logger.log_metric
log_counter = metrics_logger.log_counter
log_gauge = metrics_logger.log_gauge
log_histogram = metrics_logger.log_histogram
log_audit = audit_logger.log_action
log_access = audit_logger.log_access
log_data_change = audit_logger.log_data_change

def setup_test_logging():
    """Setup simplified logging for tests"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def cleanup_old_logs(days: int = 30):
    """Clean up old log files
    
    Args:
        days: Number of days to keep logs
    """
    logger = get_logger('maintenance')
    log_dir = Path('logs')
    
    if not log_dir.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for log_file in log_dir.glob('*.log*'):
        try:
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                logger.info(f"Deleted old log file: {log_file}")
        except Exception as e:
            logger.error(f"Error deleting log file {log_file}: {str(e)}")

# Initialize logging on module import
if __name__ != '__main__':
    _logger_manager.setup_logging()