"""
NASA Anomaly Detection Engine
Enhanced anomaly detection specifically designed for NASA SMAP/MSL data
with equipment-specific models and thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import json
from pathlib import Path
import threading
import time

# Import project modules
from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderConfig
from src.data_ingestion.equipment_mapper import EquipmentComponent, SensorSpec
from config.settings import get_data_path, get_config

logger = logging.getLogger(__name__)


@dataclass
class EquipmentThreshold:
    """Equipment-specific anomaly thresholds"""
    equipment_id: str
    equipment_type: str
    subsystem: str
    criticality: str

    # Anomaly score thresholds
    critical_threshold: float = 0.95    # Critical alerts
    high_threshold: float = 0.85        # High priority alerts
    medium_threshold: float = 0.70      # Medium priority alerts
    warning_threshold: float = 0.50     # Warning level

    # Equipment-specific multipliers
    criticality_multiplier: float = 1.0  # Based on equipment criticality
    sensor_count_weight: float = 1.0     # Based on number of sensors
    historical_factor: float = 1.0       # Based on historical failure rate


@dataclass
class AnomalyResult:
    """Result from anomaly detection"""
    timestamp: datetime
    equipment_id: str
    equipment_type: str
    subsystem: str

    # Anomaly metrics
    anomaly_score: float
    is_anomaly: bool
    severity_level: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    confidence: float

    # Model information
    model_name: str
    reconstruction_error: float

    # Sensor details
    sensor_values: Dict[str, float]
    anomalous_sensors: List[str]

    # Alert information
    requires_alert: bool = False
    alert_message: str = ""


class NASAAnomalyEngine:
    """
    Advanced anomaly detection engine for NASA SMAP/MSL data
    Features equipment-specific models, thresholds, and severity classification
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize NASA Anomaly Detection Engine

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)

        # Equipment-specific models and thresholds
        self.equipment_models: Dict[str, LSTMAutoencoder] = {}
        self.equipment_thresholds: Dict[str, EquipmentThreshold] = {}

        # Training data storage
        self.training_data: Dict[str, List[np.ndarray]] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}

        # Real-time processing
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'models_trained': 0,
            'last_training': None
        }

        # Model paths
        self.models_dir = get_data_path('models') / 'nasa_equipment_models'
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("NASA Anomaly Detection Engine initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load engine configuration"""
        default_config = {
            'training': {
                'min_samples_for_training': 500,
                'training_batch_size': 32,
                'training_epochs': 50,
                'validation_split': 0.2,
                'sequence_length': 50
            },
            'detection': {
                'contamination_rate': 0.05,
                'ensemble_voting': True,
                'adaptive_thresholds': True
            },
            'thresholds': {
                'power_systems': {
                    'critical': 0.90,
                    'high': 0.75,
                    'medium': 0.60
                },
                'mobility_systems': {
                    'critical': 0.90,
                    'high': 0.75,
                    'medium': 0.60
                },
                'communication_systems': {
                    'critical': 0.85,
                    'high': 0.70,
                    'medium': 0.55
                },
                'default': {
                    'critical': 0.80,
                    'high': 0.65,
                    'medium': 0.50
                }
            }
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def initialize_equipment_models(self, equipment_list: List[EquipmentComponent]):
        """Initialize LSTM models for each equipment component

        Args:
            equipment_list: List of equipment components to monitor
        """
        logger.info(f"Initializing models for {len(equipment_list)} equipment components")

        for equipment in equipment_list:
            # Create equipment-specific model configuration
            model_config = self._create_equipment_model_config(equipment)

            # Initialize LSTM Autoencoder
            model = LSTMAutoencoder(
                name=f"{equipment.equipment_id}_anomaly_detector",
                config=model_config,
                contamination=self.config['detection']['contamination_rate']
            )

            self.equipment_models[equipment.equipment_id] = model

            # Initialize training data storage
            self.training_data[equipment.equipment_id] = []

            # Create equipment-specific thresholds
            threshold = self._create_equipment_threshold(equipment)
            self.equipment_thresholds[equipment.equipment_id] = threshold

            logger.debug(f"Initialized model for {equipment.equipment_id}")

        logger.info(f"Initialized {len(self.equipment_models)} equipment-specific models")

    def _create_equipment_model_config(self, equipment: EquipmentComponent) -> LSTMAutoencoderConfig:
        """Create model configuration specific to equipment type

        Args:
            equipment: Equipment component

        Returns:
            Model configuration
        """
        # Safe access to equipment properties with defaults
        sensors = getattr(equipment, 'sensors', None) or []
        criticality = getattr(equipment, 'criticality', None) or 'MEDIUM'

        n_sensors = len(sensors)

        # Ensure minimum sensor count for model stability
        if n_sensors == 0:
            logger.warning(f"Equipment {equipment.equipment_id} has no sensors, using default config")
            n_sensors = 5  # Default sensor count

        # Adjust model complexity based on equipment type and sensor count
        if criticality == 'CRITICAL':
            # More complex models for critical equipment
            encoder_units = [min(64, n_sensors * 4), min(32, n_sensors * 2)]
            latent_dim = min(16, n_sensors)
            epochs = 100
        elif criticality == 'HIGH':
            encoder_units = [min(32, n_sensors * 3), min(16, n_sensors * 2)]
            latent_dim = min(12, n_sensors)
            epochs = 75
        else:
            encoder_units = [min(16, n_sensors * 2), min(8, n_sensors)]
            latent_dim = min(8, n_sensors)
            epochs = 50

        return LSTMAutoencoderConfig(
            encoder_units=encoder_units,
            decoder_units=encoder_units[::-1],  # Mirror encoder
            latent_dim=latent_dim,
            sequence_length=self.config['training']['sequence_length'],
            epochs=epochs,
            batch_size=self.config['training']['training_batch_size'],
            validation_split=self.config['training']['validation_split'],

            # Equipment-specific settings
            use_bidirectional=equipment.criticality == 'CRITICAL',
            use_batch_norm=True,
            dropout_rate=0.2,
            l2_reg=0.01,

            # Adaptive settings based on sensor count
            learning_rate=0.001 if n_sensors > 5 else 0.0005,
            early_stopping_patience=15 if criticality == 'CRITICAL' else 10
        )

    def _create_equipment_threshold(self, equipment: EquipmentComponent) -> EquipmentThreshold:
        """Create equipment-specific anomaly thresholds

        Args:
            equipment: Equipment component

        Returns:
            Equipment threshold configuration
        """
        # Safe access to equipment properties with defaults
        subsystem = getattr(equipment, 'subsystem', None) or 'default'
        criticality = getattr(equipment, 'criticality', None) or 'MEDIUM'
        sensors = getattr(equipment, 'sensors', None) or []
        equipment_type = getattr(equipment, 'equipment_type', None) or 'Unknown'

        # Get subsystem-specific thresholds from config
        subsystem_key = f"{subsystem.lower()}_systems"
        subsystem_thresholds = self.config['thresholds'].get(
            subsystem_key,
            self.config['thresholds']['default']
        )

        # Calculate criticality multiplier
        criticality_multipliers = {
            'CRITICAL': 1.2,
            'HIGH': 1.0,
            'MEDIUM': 0.8,
            'LOW': 0.6
        }
        multiplier = criticality_multipliers.get(criticality, 1.0)

        # Sensor count weight (more sensors = more robust detection)
        sensor_count = len(sensors)
        sensor_count_weight = 1.0 + (sensor_count - 5) * 0.02
        sensor_count_weight = max(0.8, min(1.3, sensor_count_weight))

        return EquipmentThreshold(
            equipment_id=getattr(equipment, 'equipment_id', 'unknown'),
            equipment_type=equipment_type,
            subsystem=subsystem,
            criticality=criticality,

            critical_threshold=subsystem_thresholds['critical'] * multiplier,
            high_threshold=subsystem_thresholds['high'] * multiplier,
            medium_threshold=subsystem_thresholds['medium'] * multiplier,
            warning_threshold=subsystem_thresholds['medium'] * 0.8 * multiplier,

            criticality_multiplier=multiplier,
            sensor_count_weight=sensor_count_weight
        )

    def train_equipment_model(self, equipment_id: str, training_data: np.ndarray) -> bool:
        """Train LSTM model for specific equipment

        Args:
            equipment_id: Equipment identifier
            training_data: Training data array

        Returns:
            Success status
        """
        if equipment_id not in self.equipment_models:
            logger.error(f"No model found for equipment {equipment_id}")
            return False

        try:
            model = self.equipment_models[equipment_id]

            # Validate training data
            if len(training_data) < self.config['training']['min_samples_for_training']:
                logger.warning(f"Insufficient training data for {equipment_id}: {len(training_data)} samples")
                return False

            logger.info(f"Training model for {equipment_id} with {len(training_data)} samples")

            # Train the model (only on normal data)
            model.fit(training_data)

            # Evaluate model performance
            performance = self._evaluate_model_performance(model, training_data)
            self.model_performance[equipment_id] = performance

            # Save trained model
            model_path = self.models_dir / f"{equipment_id}_model.pkl"
            self._save_model(model, model_path)

            # Update statistics
            self.processing_stats['models_trained'] += 1
            self.processing_stats['last_training'] = datetime.now()

            logger.info(f"Successfully trained model for {equipment_id}")
            logger.info(f"Model performance: {performance}")

            return True

        except Exception as e:
            logger.error(f"Error training model for {equipment_id}: {e}")
            return False

    def _evaluate_model_performance(self, model: LSTMAutoencoder, test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data

        Args:
            model: Trained model
            test_data: Test data

        Returns:
            Performance metrics
        """
        try:
            # Get reconstruction errors
            scores = model.score_samples(test_data)
            predictions = model.predict(test_data)

            # Calculate basic statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            anomaly_rate = np.mean(predictions)

            # Calculate reconstruction accuracy (inverse of mean reconstruction error)
            reconstructions = model.reconstruct(test_data)
            reconstruction_error = np.mean(np.square(test_data - reconstructions))
            reconstruction_accuracy = 1.0 / (1.0 + reconstruction_error)

            return {
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'anomaly_rate': float(anomaly_rate),
                'reconstruction_accuracy': float(reconstruction_accuracy),
                'reconstruction_error': float(reconstruction_error)
            }

        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {
                'mean_score': 0.0,
                'std_score': 1.0,
                'anomaly_rate': 0.0,
                'reconstruction_accuracy': 0.0,
                'reconstruction_error': 1.0
            }

    def detect_anomalies(self, equipment_id: str, sensor_data: Dict[str, float],
                        timestamp: datetime) -> AnomalyResult:
        """Detect anomalies for specific equipment

        Args:
            equipment_id: Equipment identifier
            sensor_data: Current sensor readings
            timestamp: Data timestamp

        Returns:
            Anomaly detection result
        """
        try:
            if equipment_id not in self.equipment_models:
                return self._create_default_result(equipment_id, sensor_data, timestamp, "No model available")

            model = self.equipment_models[equipment_id]
            threshold = self.equipment_thresholds.get(equipment_id)

            if not model.is_fitted:
                return self._create_default_result(equipment_id, sensor_data, timestamp, "Model not trained")

            # Prepare sensor data for model
            sensor_values = np.array(list(sensor_data.values())).reshape(1, -1)

            # Get anomaly score from model
            anomaly_scores = model.score_samples(sensor_values)
            anomaly_score = float(anomaly_scores[0])

            # Get prediction
            predictions = model.predict(sensor_values)
            is_anomaly = bool(predictions[0])

            # Calculate reconstruction error
            reconstructions = model.reconstruct(sensor_values)
            reconstruction_error = float(np.mean(np.square(sensor_values - reconstructions)))

            # Determine severity and confidence
            severity, confidence = self._calculate_severity_and_confidence(
                anomaly_score, threshold, reconstruction_error
            )

            # Identify anomalous sensors
            anomalous_sensors = self._identify_anomalous_sensors(
                sensor_data, reconstructions[0], threshold
            )

            # Determine if alert is required
            requires_alert = self._requires_alert(severity, threshold)
            alert_message = self._generate_alert_message(
                equipment_id, severity, anomalous_sensors, anomaly_score
            ) if requires_alert else ""

            # Update processing statistics
            self.processing_stats['total_processed'] += 1
            if is_anomaly:
                self.processing_stats['anomalies_detected'] += 1

            return AnomalyResult(
                timestamp=timestamp,
                equipment_id=equipment_id,
                equipment_type=threshold.equipment_type if threshold else "Unknown",
                subsystem=threshold.subsystem if threshold else "Unknown",

                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                severity_level=severity,
                confidence=confidence,

                model_name=model.name,
                reconstruction_error=reconstruction_error,

                sensor_values=sensor_data,
                anomalous_sensors=anomalous_sensors,

                requires_alert=requires_alert,
                alert_message=alert_message
            )

        except Exception as e:
            logger.error(f"Error detecting anomalies for {equipment_id}: {e}")
            return self._create_default_result(equipment_id, sensor_data, timestamp, f"Detection error: {e}")

    def _calculate_severity_and_confidence(self, anomaly_score: float,
                                         threshold: Optional[EquipmentThreshold],
                                         reconstruction_error: float) -> Tuple[str, float]:
        """Calculate anomaly severity and confidence

        Args:
            anomaly_score: Model anomaly score
            threshold: Equipment threshold configuration
            reconstruction_error: Reconstruction error

        Returns:
            Tuple of (severity_level, confidence)
        """
        if not threshold:
            # Default thresholds
            if anomaly_score > 0.8:
                return "HIGH", 0.8
            elif anomaly_score > 0.6:
                return "MEDIUM", 0.6
            else:
                return "LOW", 0.4

        # Equipment-specific severity calculation
        if anomaly_score >= threshold.critical_threshold:
            severity = "CRITICAL"
            confidence = min(0.95, 0.7 + (anomaly_score - threshold.critical_threshold) * 2)
        elif anomaly_score >= threshold.high_threshold:
            severity = "HIGH"
            confidence = min(0.85, 0.6 + (anomaly_score - threshold.high_threshold) * 2)
        elif anomaly_score >= threshold.medium_threshold:
            severity = "MEDIUM"
            confidence = min(0.75, 0.5 + (anomaly_score - threshold.medium_threshold) * 2)
        elif anomaly_score >= threshold.warning_threshold:
            severity = "LOW"
            confidence = min(0.65, 0.4 + (anomaly_score - threshold.warning_threshold) * 2)
        else:
            severity = "NORMAL"
            confidence = max(0.3, 0.5 - anomaly_score)

        # Adjust confidence based on reconstruction error
        if reconstruction_error > 1.0:
            confidence = min(confidence * 1.2, 0.95)
        elif reconstruction_error < 0.1:
            confidence = max(confidence * 0.8, 0.3)

        return severity, confidence

    def _identify_anomalous_sensors(self, sensor_data: Dict[str, float],
                                   reconstruction: np.ndarray,
                                   threshold: Optional[EquipmentThreshold]) -> List[str]:
        """Identify which specific sensors are anomalous

        Args:
            sensor_data: Original sensor data
            reconstruction: Model reconstruction
            threshold: Equipment threshold

        Returns:
            List of anomalous sensor names
        """
        anomalous_sensors = []

        if len(reconstruction) != len(sensor_data):
            return anomalous_sensors

        sensor_names = list(sensor_data.keys())
        sensor_values = list(sensor_data.values())

        for i, (name, original_value) in enumerate(zip(sensor_names, sensor_values)):
            reconstructed_value = reconstruction[i]
            error = abs(original_value - reconstructed_value)

            # Adaptive threshold based on sensor value magnitude
            adaptive_threshold = max(0.1, abs(original_value) * 0.15)

            if error > adaptive_threshold:
                anomalous_sensors.append(name)

        return anomalous_sensors

    def _requires_alert(self, severity: str, threshold: Optional[EquipmentThreshold]) -> bool:
        """Determine if an alert should be triggered

        Args:
            severity: Anomaly severity level
            threshold: Equipment threshold configuration

        Returns:
            Whether alert is required
        """
        alert_severities = ["CRITICAL", "HIGH"]

        # Critical equipment always alerts on medium and above
        if threshold and threshold.criticality == "CRITICAL":
            alert_severities.append("MEDIUM")

        return severity in alert_severities

    def _generate_alert_message(self, equipment_id: str, severity: str,
                               anomalous_sensors: List[str], anomaly_score: float) -> str:
        """Generate alert message

        Args:
            equipment_id: Equipment identifier
            severity: Severity level
            anomalous_sensors: List of anomalous sensors
            anomaly_score: Anomaly score

        Returns:
            Alert message
        """
        if not anomalous_sensors:
            return f"{severity} anomaly detected on {equipment_id} (Score: {anomaly_score:.3f})"

        sensor_list = ", ".join(anomalous_sensors[:3])  # Show up to 3 sensors
        if len(anomalous_sensors) > 3:
            sensor_list += f" (+{len(anomalous_sensors) - 3} more)"

        return f"{severity} anomaly detected on {equipment_id}: {sensor_list} (Score: {anomaly_score:.3f})"

    def _create_default_result(self, equipment_id: str, sensor_data: Dict[str, float],
                              timestamp: datetime, reason: str) -> AnomalyResult:
        """Create default anomaly result for error cases

        Args:
            equipment_id: Equipment identifier
            sensor_data: Sensor data
            timestamp: Timestamp
            reason: Error reason

        Returns:
            Default anomaly result
        """
        return AnomalyResult(
            timestamp=timestamp,
            equipment_id=equipment_id,
            equipment_type="Unknown",
            subsystem="Unknown",

            anomaly_score=0.0,
            is_anomaly=False,
            severity_level="NORMAL",
            confidence=0.0,

            model_name=reason,
            reconstruction_error=0.0,

            sensor_values=sensor_data,
            anomalous_sensors=[],

            requires_alert=False,
            alert_message=""
        )

    def _save_model(self, model: LSTMAutoencoder, model_path: Path):
        """Save trained model to disk

        Args:
            model: Trained model
            model_path: Path to save model
        """
        try:
            # Save model state
            model_state = {
                'model_config': model.config,
                'model_weights': model.autoencoder.get_weights() if model.autoencoder else None,
                'scaler': model.scaler,
                'reconstruction_stats': model.reconstruction_stats,
                'n_features': getattr(model, 'n_features', None),
                'training_time': datetime.now().isoformat()
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_state, f)

            logger.debug(f"Saved model to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model to {model_path}: {e}")

    def load_model(self, equipment_id: str) -> bool:
        """Load trained model from disk

        Args:
            equipment_id: Equipment identifier

        Returns:
            Success status
        """
        try:
            model_path = self.models_dir / f"{equipment_id}_model.pkl"

            if not model_path.exists():
                logger.warning(f"No saved model found for {equipment_id}")
                return False

            with open(model_path, 'rb') as f:
                model_state = pickle.load(f)

            if equipment_id in self.equipment_models:
                model = self.equipment_models[equipment_id]

                # Restore model state
                model.config = model_state['model_config']
                model.scaler = model_state['scaler']
                model.reconstruction_stats = model_state['reconstruction_stats']

                # Restore n_features from model state (required for _build_model)
                if 'n_features' in model_state:
                    model.n_features = model_state['n_features']
                else:
                    # Fallback: infer from equipment sensors
                    equipment_info = self.equipment_thresholds.get(equipment_id)
                    if equipment_info:
                        sensor_count = getattr(equipment_info, 'sensor_count_weight', 5)
                        model.n_features = max(1, int(sensor_count))
                    else:
                        model.n_features = 5  # Default fallback

                # Rebuild and restore weights
                if model_state.get('model_weights'):
                    try:
                        model._build_model()
                        model.autoencoder.set_weights(model_state['model_weights'])
                        model._fitted = True
                        logger.info(f"Loaded model for {equipment_id}")
                        return True
                    except Exception as build_error:
                        logger.warning(f"Failed to rebuild model for {equipment_id}: {build_error}")
                        return False

                logger.info(f"Loaded model metadata for {equipment_id} (no weights)")
                return True

        except Exception as e:
            logger.error(f"Error loading model for {equipment_id}: {e}")

        return False

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics

        Returns:
            Processing statistics
        """
        return {
            'total_processed': self.processing_stats['total_processed'],
            'anomalies_detected': self.processing_stats['anomalies_detected'],
            'models_trained': self.processing_stats['models_trained'],
            'total_models': len(self.equipment_models),
            'trained_models': len([m for m in self.equipment_models.values() if m.is_fitted]),
            'last_training': self.processing_stats['last_training'].isoformat() if self.processing_stats['last_training'] else None,
            'model_performance': self.model_performance
        }

    def get_equipment_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get equipment threshold configurations

        Returns:
            Equipment thresholds
        """
        return {
            equipment_id: {
                'equipment_type': threshold.equipment_type,
                'subsystem': threshold.subsystem,
                'criticality': threshold.criticality,
                'critical_threshold': threshold.critical_threshold,
                'high_threshold': threshold.high_threshold,
                'medium_threshold': threshold.medium_threshold,
                'warning_threshold': threshold.warning_threshold
            }
            for equipment_id, threshold in self.equipment_thresholds.items()
        }


# Global instance for integration
nasa_anomaly_engine = NASAAnomalyEngine()