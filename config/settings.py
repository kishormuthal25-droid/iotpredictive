"""
Settings Manager for IoT Anomaly Detection System
Handles configuration loading, validation, and environment variable management
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get the root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config.yaml"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    enabled: bool = True
    type: str = "sqlite"  # Default to SQLite for local development
    
    # PostgreSQL settings
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "iot_telemetry"
    pg_username: str = "iot_user"
    pg_password: str = "iot_password"
    pg_pool_size: int = 10
    pg_max_overflow: int = 20
    
    # SQLite settings
    sqlite_path: str = "./data/iot_telemetry.db"
    
    # TimescaleDB settings
    timescale_enabled: bool = False
    timescale_chunk_interval: str = "1 day"
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string"""
        if self.type == "postgresql":
            password = os.getenv("DB_PASSWORD", self.pg_password)
            return f"postgresql://{self.pg_username}:{password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        elif self.type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class KafkaConfig:
    """Kafka configuration settings"""
    enabled: bool = False
    bootstrap_servers: str = "localhost:9092"
    topics: Dict[str, str] = field(default_factory=dict)
    producer_config: Dict[str, Any] = field(default_factory=dict)
    consumer_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmailConfig:
    """Email alert configuration"""
    enabled: bool = True
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    sender_email: str = ""
    sender_password: str = ""
    sender_name: str = "IoT Alert System"
    recipients: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Override with environment variables if available"""
        self.sender_email = os.getenv("EMAIL_SENDER", self.sender_email)
        self.sender_password = os.getenv("EMAIL_PASSWORD", self.sender_password)
        self.smtp_server = os.getenv("SMTP_SERVER", self.smtp_server)


@dataclass
class ModelConfig:
    """Model configuration base class"""
    enabled: bool = True
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience
        }


class Settings:
    """
    Central configuration management class
    Singleton pattern to ensure single instance across application
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize settings from configuration file"""
        if not self._initialized:
            self.config_file = config_file or DEFAULT_CONFIG_FILE
            self._config = {}
            self._load_config()
            self._override_with_env()
            self._validate_config()
            self._setup_logging()
            self._initialized = True
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                self._config = yaml.safe_load(f)
            print(f"[OK] Configuration loaded from {self.config_file}")
        except FileNotFoundError:
            print(f"[WARNING] Configuration file not found: {self.config_file}")
            print("  Using default configuration...")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file not found"""
        return {
            'environment': 'development',
            'system': {
                'project_name': 'IoT Anomaly Detection System',
                'version': '1.0.0',
                'debug': True,
                'log_level': 'INFO',
                'timezone': 'UTC',
                'random_seed': 42
            },
            'paths': {
                'data_root': './data',
                'raw_data': './data/raw',
                'processed_data': './data/processed',
                'models': './data/models',
                'logs': './logs',
                'cache': './cache',
                'smap_data': './data/raw/smap',
                'msl_data': './data/raw/msl'
            },
            'data_ingestion': {
                'simulation': {
                    'enabled': True,
                    'speed_multiplier': 1.0,
                    'batch_size': 100
                }
            },
            'dashboard': {
                'server': {
                    'host': '127.0.0.1',
                    'port': 8050,
                    'debug': True
                }
            }
        }
    
    def _override_with_env(self):
        """Override configuration with environment variables"""
        # Environment
        self._config['environment'] = os.getenv('ENVIRONMENT', self._config.get('environment', 'development'))
        
        # Debug mode
        if 'DEBUG' in os.environ:
            self._config['system']['debug'] = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Database
        if 'DATABASE_URL' in os.environ:
            self._config['data_ingestion']['database']['postgresql']['host'] = os.getenv('DB_HOST', 'localhost')
            self._config['data_ingestion']['database']['postgresql']['port'] = int(os.getenv('DB_PORT', 5432))
            self._config['data_ingestion']['database']['postgresql']['database'] = os.getenv('DB_NAME', 'iot_telemetry')
            self._config['data_ingestion']['database']['postgresql']['username'] = os.getenv('DB_USER', 'iot_user')
            self._config['data_ingestion']['database']['postgresql']['password'] = os.getenv('DB_PASSWORD', 'iot_password')
        
        # Kafka
        if 'KAFKA_BOOTSTRAP_SERVERS' in os.environ:
            self._config['data_ingestion']['kafka']['bootstrap_servers'] = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
            self._config['data_ingestion']['kafka']['enabled'] = True
        
        # Dashboard
        if 'DASHBOARD_PORT' in os.environ:
            self._config['dashboard']['server']['port'] = int(os.getenv('DASHBOARD_PORT'))
    
    def _validate_config(self):
        """Validate configuration values"""
        # Check required paths exist or create them
        required_dirs = [
            self.get('paths.data_root'),
            self.get('paths.raw_data'),
            self.get('paths.processed_data'),
            self.get('paths.models'),
            self.get('paths.logs'),
            self.get('paths.smap_data'),
            self.get('paths.msl_data')
        ]
        
        for dir_path in required_dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Validate numeric ranges
        if self.get('data_ingestion.simulation.speed_multiplier', 1.0) <= 0:
            raise ValueError("Speed multiplier must be positive")
        
        if self.get('dashboard.server.port', 8050) < 1 or self.get('dashboard.server.port', 8050) > 65535:
            raise ValueError("Dashboard port must be between 1 and 65535")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.get('system.log_level', 'INFO')
        log_format = self.get('logging.file.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create logs directory
        log_dir = Path(self.get('paths.logs', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_dir / 'iot_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: settings.get('database.postgresql.host')
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation
        Example: settings.set('database.postgresql.port', 5433)
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration object"""
        db_config = self.get('data_ingestion.database', {})
        
        if db_config.get('type') == 'postgresql':
            pg_config = db_config.get('postgresql', {})
            return DatabaseConfig(
                enabled=db_config.get('enabled', True),
                type='postgresql',
                pg_host=pg_config.get('host', 'localhost'),
                pg_port=pg_config.get('port', 5432),
                pg_database=pg_config.get('database', 'iot_telemetry'),
                pg_username=pg_config.get('username', 'iot_user'),
                pg_password=pg_config.get('password', 'iot_password'),
                pg_pool_size=pg_config.get('pool_size', 10),
                pg_max_overflow=pg_config.get('max_overflow', 20),
                timescale_enabled=db_config.get('timescale', {}).get('enabled', False)
            )
        else:
            sqlite_config = db_config.get('sqlite', {})
            return DatabaseConfig(
                enabled=db_config.get('enabled', True),
                type='sqlite',
                sqlite_path=sqlite_config.get('path', './data/iot_telemetry.db')
            )
    
    def get_kafka_config(self) -> KafkaConfig:
        """Get Kafka configuration object"""
        kafka_config = self.get('data_ingestion.kafka', {})
        return KafkaConfig(
            enabled=kafka_config.get('enabled', False),
            bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
            topics=kafka_config.get('topics', {}),
            producer_config=kafka_config.get('producer', {}),
            consumer_config=kafka_config.get('consumer', {})
        )
    
    def get_email_config(self) -> EmailConfig:
        """Get email configuration object"""
        email_config = self.get('alerts.email', {})
        return EmailConfig(
            enabled=email_config.get('enabled', True),
            smtp_server=email_config.get('smtp_server', 'smtp.gmail.com'),
            smtp_port=email_config.get('smtp_port', 587),
            use_tls=email_config.get('use_tls', True),
            sender_email=email_config.get('sender_email', ''),
            sender_password=email_config.get('sender_password', ''),
            sender_name=email_config.get('sender_name', 'IoT Alert System'),
            recipients=email_config.get('recipients', {})
        )
    
    def get_model_config(self, model_type: str) -> ModelConfig:
        """Get model-specific configuration"""
        if model_type == 'lstm_predictor':
            config = self.get('anomaly_detection.lstm_predictor.training', {})
        elif model_type == 'lstm_autoencoder':
            config = self.get('anomaly_detection.lstm_autoencoder.training', {})
        elif model_type == 'lstm_vae':
            config = self.get('anomaly_detection.lstm_vae.training', {})
        else:
            config = {}
        
        return ModelConfig(
            enabled=True,
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            learning_rate=config.get('learning_rate', 0.001),
            validation_split=config.get('validation_split', 0.2),
            early_stopping_patience=config.get('early_stopping_patience', 10)
        )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self._config.get('environment', 'development') == 'development'
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self._config.get('environment', 'development') == 'production'
    
    @property
    def debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.get('system.debug', False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._config.copy()
    
    def save(self, file_path: Optional[Path] = None):
        """Save current configuration to file"""
        save_path = file_path or self.config_file
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        print(f"[OK] Configuration saved to {save_path}")

    def reload(self):
        """Reload configuration from file"""
        self._initialized = False
        self.__init__(self.config_file)
        print("[OK] Configuration reloaded")


# Global settings instance
settings = Settings()


# Convenience functions for quick access
def get_config(key: str, default: Any = None) -> Any:
    """Quick access to configuration values"""
    return settings.get(key, default)


def get_data_path(path_key: str) -> Path:
    """Get path from configuration as Path object"""
    path_str = settings.get(f'paths.{path_key}', '.')
    return Path(path_str).resolve()


def get_model_path(model_name: str) -> Path:
    """Get path for saving/loading models"""
    models_dir = get_data_path('models')
    return models_dir / f"{model_name}.h5"


def is_kafka_enabled() -> bool:
    """Check if Kafka is enabled"""
    return settings.get('data_ingestion.kafka.enabled', False)


def is_redis_enabled() -> bool:
    """Check if Redis is enabled"""
    return settings.get('data_ingestion.redis.enabled', False)


def get_window_config() -> Dict[str, int]:
    """Get time series window configuration"""
    return {
        'size': settings.get('preprocessing.window.size', 100),
        'stride': settings.get('preprocessing.window.stride', 10)
    }


def get_dashboard_url() -> str:
    """Get dashboard URL"""
    host = settings.get('dashboard.server.host', '127.0.0.1')
    port = settings.get('dashboard.server.port', 8050)
    return f"http://{host}:{port}"


# Initialize settings on module import
if __name__ == "__main__":
    # Test configuration loading
    print("\n" + "="*50)
    print("IoT Anomaly Detection System - Settings Test")
    print("="*50 + "\n")
    
    print(f"Environment: {settings.get('environment')}")
    print(f"Debug Mode: {settings.debug}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config File: {settings.config_file}")
    
    print("\nDatabase Configuration:")
    db_config = settings.get_database_config()
    print(f"  Type: {db_config.type}")
    print(f"  Connection: {db_config.connection_string}")
    
    print("\nKafka Configuration:")
    kafka_config = settings.get_kafka_config()
    print(f"  Enabled: {kafka_config.enabled}")
    print(f"  Bootstrap Servers: {kafka_config.bootstrap_servers}")
    
    print("\nEmail Configuration:")
    email_config = settings.get_email_config()
    print(f"  Enabled: {email_config.enabled}")
    print(f"  SMTP Server: {email_config.smtp_server}")
    
    print("\nDashboard Configuration:")
    print(f"  URL: {get_dashboard_url()}")
    
    print("\n[OK] Settings module initialized successfully")
