"""
Test Data Generator

Generates realistic test data for comprehensive testing across all phases:
- NASA SMAP/MSL sensor data simulation
- Anomaly injection and pattern generation
- Business scenario data creation
- Performance testing datasets
- Failure simulation data
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import random
import hashlib
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestDataConfig:
    """Configuration for test data generation"""
    smap_sensors: int = 25
    msl_sensors: int = 55
    total_sensors: int = 80
    data_points_per_sensor: int = 1000
    anomaly_injection_rate: float = 0.05
    seasonal_patterns: bool = True
    trend_patterns: bool = True
    noise_level: float = 0.05
    output_format: str = 'json'  # json, csv, pickle, numpy
    data_quality_level: str = 'high'  # high, medium, low


@dataclass
class SensorMetadata:
    """Sensor metadata structure"""
    sensor_id: str
    sensor_type: str
    equipment_id: str
    location: str
    criticality: str
    sampling_rate: str
    unit: str
    normal_range: Tuple[float, float]
    alert_threshold: float
    failure_modes: List[str]


@dataclass
class AnomalyPattern:
    """Anomaly pattern definition"""
    anomaly_type: str
    severity: str
    duration: int
    frequency: float
    pattern_signature: Dict[str, Any]


class TestDataGenerator:
    """Comprehensive test data generator for IoT Predictive Maintenance System"""

    def __init__(self, config: TestDataConfig, output_dir: str = "generated_test_data"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize random seeds for reproducibility
        np.random.seed(42)
        random.seed(42)

        # Define sensor types and characteristics
        self.sensor_types = {
            'temperature': {
                'unit': 'celsius',
                'normal_range': (20.0, 80.0),
                'alert_threshold': 85.0,
                'failure_modes': ['overheating', 'sensor_drift', 'calibration_error']
            },
            'pressure': {
                'unit': 'psi',
                'normal_range': (10.0, 100.0),
                'alert_threshold': 110.0,
                'failure_modes': ['pressure_spike', 'sensor_clog', 'membrane_failure']
            },
            'vibration': {
                'unit': 'g',
                'normal_range': (0.1, 2.0),
                'alert_threshold': 3.0,
                'failure_modes': ['bearing_wear', 'imbalance', 'resonance']
            },
            'flow': {
                'unit': 'gpm',
                'normal_range': (5.0, 50.0),
                'alert_threshold': 55.0,
                'failure_modes': ['blockage', 'pump_failure', 'leak']
            },
            'electrical': {
                'unit': 'volts',
                'normal_range': (110.0, 130.0),
                'alert_threshold': 140.0,
                'failure_modes': ['voltage_spike', 'ground_fault', 'insulation_breakdown']
            }
        }

        # Define anomaly patterns
        self.anomaly_patterns = {
            'spike': {
                'description': 'Sudden spike in sensor values',
                'duration_range': (1, 10),
                'amplitude_multiplier': (2.0, 5.0)
            },
            'drift': {
                'description': 'Gradual drift from normal values',
                'duration_range': (50, 200),
                'amplitude_multiplier': (1.2, 2.0)
            },
            'oscillation': {
                'description': 'Abnormal oscillatory behavior',
                'duration_range': (20, 100),
                'frequency_range': (0.1, 2.0)
            },
            'flatline': {
                'description': 'Sensor becomes unresponsive',
                'duration_range': (10, 50),
                'value_type': 'constant'
            },
            'noise_burst': {
                'description': 'High-frequency noise burst',
                'duration_range': (5, 30),
                'noise_multiplier': (5.0, 20.0)
            }
        }

        logger.info(f"TestDataGenerator initialized with config: {asdict(config)}")

    def generate_all_test_data(self) -> Dict[str, str]:
        """Generate all types of test data"""
        logger.info("Starting comprehensive test data generation...")

        generated_files = {}

        # Generate sensor metadata
        metadata_file = self.generate_sensor_metadata()
        generated_files['sensor_metadata'] = metadata_file

        # Generate sensor time series data
        sensor_data_file = self.generate_sensor_time_series()
        generated_files['sensor_data'] = sensor_data_file

        # Generate anomaly data
        anomaly_data_file = self.generate_anomaly_scenarios()
        generated_files['anomaly_data'] = anomaly_data_file

        # Generate business scenario data
        business_data_file = self.generate_business_scenarios()
        generated_files['business_data'] = business_data_file

        # Generate performance testing data
        performance_data_file = self.generate_performance_test_data()
        generated_files['performance_data'] = performance_data_file

        # Generate failure simulation data
        failure_data_file = self.generate_failure_scenarios()
        generated_files['failure_data'] = failure_data_file

        # Generate synthetic NASA-like datasets
        nasa_data_file = self.generate_nasa_like_datasets()
        generated_files['nasa_data'] = nasa_data_file

        # Generate data quality test sets
        quality_data_file = self.generate_data_quality_test_sets()
        generated_files['quality_data'] = quality_data_file

        # Create data manifest
        manifest_file = self.create_data_manifest(generated_files)
        generated_files['manifest'] = manifest_file

        logger.info(f"Test data generation completed. Generated {len(generated_files)} datasets.")
        return generated_files

    def generate_sensor_metadata(self) -> str:
        """Generate comprehensive sensor metadata"""
        logger.info("Generating sensor metadata...")

        metadata = []
        sensor_types = list(self.sensor_types.keys())

        # Generate SMAP sensors
        for i in range(self.config.smap_sensors):
            sensor_type = sensor_types[i % len(sensor_types)]
            type_config = self.sensor_types[sensor_type]

            sensor_metadata = SensorMetadata(
                sensor_id=f"SMAP_{i+1:03d}",
                sensor_type=sensor_type,
                equipment_id=f"SMAP_EQUIPMENT_{(i // 5) + 1:02d}",
                location=f"SATELLITE_ZONE_{(i // 10) + 1}",
                criticality=random.choice(['HIGH', 'MEDIUM', 'LOW']),
                sampling_rate="1_minute",
                unit=type_config['unit'],
                normal_range=type_config['normal_range'],
                alert_threshold=type_config['alert_threshold'],
                failure_modes=type_config['failure_modes']
            )
            metadata.append(asdict(sensor_metadata))

        # Generate MSL sensors
        for i in range(self.config.msl_sensors):
            sensor_type = sensor_types[i % len(sensor_types)]
            type_config = self.sensor_types[sensor_type]

            sensor_metadata = SensorMetadata(
                sensor_id=f"MSL_{i+1:03d}",
                sensor_type=sensor_type,
                equipment_id=f"MSL_EQUIPMENT_{(i // 5) + 1:02d}",
                location=f"ROVER_ZONE_{(i // 10) + 1}",
                criticality=random.choice(['HIGH', 'MEDIUM', 'LOW']),
                sampling_rate="1_minute",
                unit=type_config['unit'],
                normal_range=type_config['normal_range'],
                alert_threshold=type_config['alert_threshold'],
                failure_modes=type_config['failure_modes']
            )
            metadata.append(asdict(sensor_metadata))

        # Save metadata
        metadata_file = self.output_dir / "sensor_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Generated metadata for {len(metadata)} sensors -> {metadata_file}")
        return str(metadata_file)

    def generate_sensor_time_series(self) -> str:
        """Generate realistic sensor time series data"""
        logger.info("Generating sensor time series data...")

        # Load sensor metadata
        metadata_file = self.output_dir / "sensor_metadata.json"
        with open(metadata_file, 'r') as f:
            sensor_metadata = json.load(f)

        sensor_data = {}

        for sensor_meta in sensor_metadata:
            sensor_id = sensor_meta['sensor_id']
            sensor_type = sensor_meta['sensor_type']
            normal_range = sensor_meta['normal_range']

            logger.info(f"Generating time series for {sensor_id}...")

            # Generate time series
            time_series = self._generate_realistic_time_series(
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                normal_range=normal_range,
                data_points=self.config.data_points_per_sensor
            )

            sensor_data[sensor_id] = time_series

        # Save sensor data
        sensor_data_file = self.output_dir / "sensor_time_series.json"
        with open(sensor_data_file, 'w') as f:
            json.dump(sensor_data, f, indent=2, default=str)

        logger.info(f"Generated time series for {len(sensor_data)} sensors -> {sensor_data_file}")
        return str(sensor_data_file)

    def generate_anomaly_scenarios(self) -> str:
        """Generate various anomaly scenarios for testing"""
        logger.info("Generating anomaly scenarios...")

        anomaly_scenarios = []

        # Load sensor metadata
        metadata_file = self.output_dir / "sensor_metadata.json"
        with open(metadata_file, 'r') as f:
            sensor_metadata = json.load(f)

        # Generate anomalies for each sensor
        for sensor_meta in sensor_metadata:
            sensor_id = sensor_meta['sensor_id']
            failure_modes = sensor_meta['failure_modes']

            # Generate multiple anomaly scenarios per sensor
            for anomaly_type in self.anomaly_patterns.keys():
                for failure_mode in failure_modes[:2]:  # Limit to 2 failure modes per sensor
                    scenario = self._generate_anomaly_scenario(
                        sensor_id=sensor_id,
                        anomaly_type=anomaly_type,
                        failure_mode=failure_mode,
                        sensor_meta=sensor_meta
                    )
                    anomaly_scenarios.append(scenario)

        # Save anomaly scenarios
        anomaly_file = self.output_dir / "anomaly_scenarios.json"
        with open(anomaly_file, 'w') as f:
            json.dump(anomaly_scenarios, f, indent=2, default=str)

        logger.info(f"Generated {len(anomaly_scenarios)} anomaly scenarios -> {anomaly_file}")
        return str(anomaly_file)

    def generate_business_scenarios(self) -> str:
        """Generate business logic test scenarios"""
        logger.info("Generating business scenarios...")

        business_scenarios = []

        # Define business scenario templates
        scenario_templates = [
            {
                'scenario_type': 'preventive_maintenance',
                'description': 'Scheduled preventive maintenance scenario',
                'trigger_conditions': ['sensor_degradation', 'time_based'],
                'expected_actions': ['schedule_maintenance', 'notify_technician'],
                'priority': 'MEDIUM',
                'estimated_cost': (500, 2000)
            },
            {
                'scenario_type': 'emergency_shutdown',
                'description': 'Emergency equipment shutdown scenario',
                'trigger_conditions': ['critical_anomaly', 'safety_threshold_exceeded'],
                'expected_actions': ['immediate_shutdown', 'emergency_alert'],
                'priority': 'CRITICAL',
                'estimated_cost': (2000, 10000)
            },
            {
                'scenario_type': 'predictive_replacement',
                'description': 'Predictive component replacement scenario',
                'trigger_conditions': ['end_of_life_prediction', 'performance_degradation'],
                'expected_actions': ['order_parts', 'schedule_replacement'],
                'priority': 'HIGH',
                'estimated_cost': (1000, 5000)
            },
            {
                'scenario_type': 'routine_inspection',
                'description': 'Routine equipment inspection scenario',
                'trigger_conditions': ['inspection_schedule', 'compliance_requirement'],
                'expected_actions': ['schedule_inspection', 'prepare_checklist'],
                'priority': 'LOW',
                'estimated_cost': (100, 500)
            }
        ]

        # Generate scenarios for different equipment combinations
        equipment_ids = set()
        metadata_file = self.output_dir / "sensor_metadata.json"
        with open(metadata_file, 'r') as f:
            sensor_metadata = json.load(f)
            equipment_ids.update(meta['equipment_id'] for meta in sensor_metadata)

        for equipment_id in equipment_ids:
            for template in scenario_templates:
                for scenario_variant in range(3):  # 3 variants per template per equipment
                    scenario = self._generate_business_scenario(equipment_id, template, scenario_variant)
                    business_scenarios.append(scenario)

        # Save business scenarios
        business_file = self.output_dir / "business_scenarios.json"
        with open(business_file, 'w') as f:
            json.dump(business_scenarios, f, indent=2, default=str)

        logger.info(f"Generated {len(business_scenarios)} business scenarios -> {business_file}")
        return str(business_file)

    def generate_performance_test_data(self) -> str:
        """Generate data specifically for performance testing"""
        logger.info("Generating performance test data...")

        performance_datasets = {}

        # Generate datasets of varying sizes for scalability testing
        dataset_sizes = [
            {'name': 'small', 'sensors': 10, 'points': 100},
            {'name': 'medium', 'sensors': 50, 'points': 500},
            {'name': 'large', 'sensors': 100, 'points': 1000},
            {'name': 'xlarge', 'sensors': 200, 'points': 2000},
            {'name': 'stress', 'sensors': 500, 'points': 5000}
        ]

        for dataset_config in dataset_sizes:
            dataset_name = dataset_config['name']
            logger.info(f"Generating {dataset_name} performance dataset...")

            dataset = self._generate_performance_dataset(
                num_sensors=dataset_config['sensors'],
                data_points=dataset_config['points']
            )

            performance_datasets[dataset_name] = dataset

        # Generate concurrent access test data
        concurrent_data = self._generate_concurrent_access_data()
        performance_datasets['concurrent_access'] = concurrent_data

        # Generate memory pressure test data
        memory_pressure_data = self._generate_memory_pressure_data()
        performance_datasets['memory_pressure'] = memory_pressure_data

        # Save performance test data
        performance_file = self.output_dir / "performance_test_data.json"
        with open(performance_file, 'w') as f:
            json.dump(performance_datasets, f, indent=2, default=str)

        logger.info(f"Generated performance test datasets -> {performance_file}")
        return str(performance_file)

    def generate_failure_scenarios(self) -> str:
        """Generate failure simulation scenarios"""
        logger.info("Generating failure scenarios...")

        failure_scenarios = []

        # Define failure types
        failure_types = [
            {
                'failure_type': 'sensor_failure',
                'description': 'Individual sensor failure',
                'impact_scope': 'sensor',
                'recovery_time': (30, 300),  # seconds
                'frequency': 'rare'
            },
            {
                'failure_type': 'equipment_failure',
                'description': 'Equipment-level failure affecting multiple sensors',
                'impact_scope': 'equipment',
                'recovery_time': (300, 1800),  # seconds
                'frequency': 'uncommon'
            },
            {
                'failure_type': 'network_failure',
                'description': 'Network connectivity failure',
                'impact_scope': 'zone',
                'recovery_time': (60, 600),  # seconds
                'frequency': 'occasional'
            },
            {
                'failure_type': 'power_failure',
                'description': 'Power supply failure',
                'impact_scope': 'zone',
                'recovery_time': (600, 3600),  # seconds
                'frequency': 'rare'
            },
            {
                'failure_type': 'software_failure',
                'description': 'Software component failure',
                'impact_scope': 'system',
                'recovery_time': (120, 900),  # seconds
                'frequency': 'uncommon'
            }
        ]

        # Generate failure scenarios
        for failure_type in failure_types:
            for scenario_id in range(10):  # 10 scenarios per failure type
                scenario = self._generate_failure_scenario(failure_type, scenario_id)
                failure_scenarios.append(scenario)

        # Save failure scenarios
        failure_file = self.output_dir / "failure_scenarios.json"
        with open(failure_file, 'w') as f:
            json.dump(failure_scenarios, f, indent=2, default=str)

        logger.info(f"Generated {len(failure_scenarios)} failure scenarios -> {failure_file}")
        return str(failure_file)

    def generate_nasa_like_datasets(self) -> str:
        """Generate synthetic datasets mimicking NASA SMAP/MSL characteristics"""
        logger.info("Generating NASA-like datasets...")

        nasa_datasets = {}

        # Generate SMAP-like satellite data
        smap_data = self._generate_smap_like_data()
        nasa_datasets['smap_synthetic'] = smap_data

        # Generate MSL-like rover data
        msl_data = self._generate_msl_like_data()
        nasa_datasets['msl_synthetic'] = msl_data

        # Generate combined mission data
        combined_data = self._generate_combined_mission_data()
        nasa_datasets['combined_mission'] = combined_data

        # Save NASA-like datasets
        nasa_file = self.output_dir / "nasa_like_datasets.json"
        with open(nasa_file, 'w') as f:
            json.dump(nasa_datasets, f, indent=2, default=str)

        logger.info(f"Generated NASA-like datasets -> {nasa_file}")
        return str(nasa_file)

    def generate_data_quality_test_sets(self) -> str:
        """Generate test sets with various data quality issues"""
        logger.info("Generating data quality test sets...")

        quality_test_sets = {}

        # Generate datasets with different quality levels
        quality_levels = ['high', 'medium', 'low', 'corrupted']

        for quality_level in quality_levels:
            logger.info(f"Generating {quality_level} quality dataset...")

            quality_dataset = self._generate_quality_dataset(quality_level)
            quality_test_sets[quality_level] = quality_dataset

        # Generate specific data quality issue scenarios
        issue_scenarios = [
            'missing_values',
            'outliers',
            'inconsistent_timestamps',
            'duplicate_records',
            'schema_violations',
            'encoding_errors'
        ]

        for issue_type in issue_scenarios:
            issue_dataset = self._generate_quality_issue_dataset(issue_type)
            quality_test_sets[f"issue_{issue_type}"] = issue_dataset

        # Save quality test sets
        quality_file = self.output_dir / "data_quality_test_sets.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_test_sets, f, indent=2, default=str)

        logger.info(f"Generated data quality test sets -> {quality_file}")
        return str(quality_file)

    def create_data_manifest(self, generated_files: Dict[str, str]) -> str:
        """Create a manifest of all generated test data"""
        logger.info("Creating data manifest...")

        manifest = {
            'generation_timestamp': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'config': asdict(self.config),
            'generated_files': generated_files,
            'data_statistics': {},
            'usage_instructions': {
                'sensor_metadata': 'Contains metadata for all sensors including types, ranges, and failure modes',
                'sensor_data': 'Time series data for all sensors with realistic patterns and injected anomalies',
                'anomaly_data': 'Specific anomaly scenarios for testing detection algorithms',
                'business_data': 'Business logic scenarios for testing maintenance decision making',
                'performance_data': 'Datasets of various sizes for performance and scalability testing',
                'failure_data': 'Failure scenarios for testing system resilience and recovery',
                'nasa_data': 'Synthetic datasets mimicking NASA SMAP/MSL mission data',
                'quality_data': 'Test sets with various data quality issues for robustness testing'
            }
        }

        # Calculate data statistics
        for data_type, file_path in generated_files.items():
            if data_type != 'manifest':
                try:
                    file_size = os.path.getsize(file_path)
                    manifest['data_statistics'][data_type] = {
                        'file_size_bytes': file_size,
                        'file_size_mb': round(file_size / (1024 * 1024), 2)
                    }
                except Exception as e:
                    logger.warning(f"Could not get statistics for {data_type}: {e}")

        # Save manifest
        manifest_file = self.output_dir / "test_data_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created data manifest -> {manifest_file}")
        return str(manifest_file)

    def _generate_realistic_time_series(self, sensor_id: str, sensor_type: str,
                                      normal_range: Tuple[float, float], data_points: int) -> Dict[str, Any]:
        """Generate realistic time series with patterns and anomalies"""

        # Base parameters
        min_val, max_val = normal_range
        mid_val = (min_val + max_val) / 2
        amplitude = (max_val - min_val) / 4

        # Generate timestamps
        start_time = datetime.now() - timedelta(minutes=data_points)
        timestamps = [start_time + timedelta(minutes=i) for i in range(data_points)]

        # Generate base signal
        values = np.full(data_points, mid_val)

        # Add trend (if enabled)
        if self.config.trend_patterns:
            trend_strength = random.uniform(-0.1, 0.1)
            trend = np.linspace(0, trend_strength * amplitude, data_points)
            values += trend

        # Add seasonal patterns (if enabled)
        if self.config.seasonal_patterns:
            # Daily pattern
            daily_freq = 2 * np.pi / (24 * 60)  # 24 hour cycle
            daily_pattern = amplitude * 0.3 * np.sin(daily_freq * np.arange(data_points))
            values += daily_pattern

            # Weekly pattern
            weekly_freq = 2 * np.pi / (7 * 24 * 60)  # 7 day cycle
            weekly_pattern = amplitude * 0.1 * np.sin(weekly_freq * np.arange(data_points))
            values += weekly_pattern

        # Add noise
        noise = np.random.normal(0, self.config.noise_level * amplitude, data_points)
        values += noise

        # Inject anomalies
        anomaly_indices = []
        if self.config.anomaly_injection_rate > 0:
            num_anomalies = int(data_points * self.config.anomaly_injection_rate)
            anomaly_positions = np.random.choice(data_points, size=num_anomalies, replace=False)

            for pos in anomaly_positions:
                anomaly_type = random.choice(list(self.anomaly_patterns.keys()))
                anomaly_data = self._inject_anomaly(values, pos, anomaly_type, amplitude)
                values = anomaly_data['values']
                anomaly_indices.extend(anomaly_data['indices'])

        # Ensure values stay within reasonable bounds
        values = np.clip(values, min_val * 0.5, max_val * 1.5)

        return {
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'values': values.tolist(),
            'normal_range': normal_range,
            'anomaly_indices': sorted(set(anomaly_indices)),
            'data_quality_score': self._calculate_quality_score(),
            'generation_metadata': {
                'trend_enabled': self.config.trend_patterns,
                'seasonal_enabled': self.config.seasonal_patterns,
                'noise_level': self.config.noise_level,
                'anomaly_rate': self.config.anomaly_injection_rate
            }
        }

    def _inject_anomaly(self, values: np.ndarray, position: int, anomaly_type: str, amplitude: float) -> Dict[str, Any]:
        """Inject specific anomaly pattern into time series"""

        pattern = self.anomaly_patterns[anomaly_type]
        duration_min, duration_max = pattern['duration_range']
        duration = random.randint(duration_min, min(duration_max, len(values) - position))

        affected_indices = list(range(position, min(position + duration, len(values))))

        if anomaly_type == 'spike':
            multiplier = random.uniform(*pattern['amplitude_multiplier'])
            for idx in affected_indices:
                values[idx] *= multiplier

        elif anomaly_type == 'drift':
            multiplier = random.uniform(*pattern['amplitude_multiplier'])
            for i, idx in enumerate(affected_indices):
                drift_factor = 1 + (multiplier - 1) * (i / len(affected_indices))
                values[idx] *= drift_factor

        elif anomaly_type == 'oscillation':
            frequency = random.uniform(*pattern['frequency_range'])
            for i, idx in enumerate(affected_indices):
                oscillation = amplitude * 0.5 * np.sin(2 * np.pi * frequency * i)
                values[idx] += oscillation

        elif anomaly_type == 'flatline':
            flatline_value = values[position]
            for idx in affected_indices:
                values[idx] = flatline_value

        elif anomaly_type == 'noise_burst':
            noise_multiplier = random.uniform(*pattern['noise_multiplier'])
            for idx in affected_indices:
                noise = np.random.normal(0, amplitude * noise_multiplier)
                values[idx] += noise

        return {
            'values': values,
            'indices': affected_indices,
            'anomaly_type': anomaly_type,
            'duration': duration
        }

    def _generate_anomaly_scenario(self, sensor_id: str, anomaly_type: str,
                                 failure_mode: str, sensor_meta: Dict) -> Dict[str, Any]:
        """Generate a specific anomaly scenario"""

        return {
            'scenario_id': f"{sensor_id}_{anomaly_type}_{failure_mode}",
            'sensor_id': sensor_id,
            'anomaly_type': anomaly_type,
            'failure_mode': failure_mode,
            'severity': random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
            'expected_detection_time': random.uniform(1, 30),  # minutes
            'expected_classification_accuracy': random.uniform(0.8, 0.98),
            'impact_assessment': {
                'safety_risk': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'operational_impact': random.choice(['MINIMAL', 'MODERATE', 'SEVERE']),
                'cost_impact': random.uniform(100, 10000)
            },
            'test_conditions': {
                'ambient_temperature': random.uniform(15, 35),
                'system_load': random.uniform(0.3, 0.9),
                'concurrent_operations': random.randint(1, 10)
            },
            'expected_response': {
                'alert_triggered': True,
                'maintenance_scheduled': random.choice([True, False]),
                'emergency_action': anomaly_type in ['spike', 'critical_drift']
            }
        }

    def _generate_business_scenario(self, equipment_id: str, template: Dict, variant: int) -> Dict[str, Any]:
        """Generate a business logic scenario"""

        scenario_id = f"{equipment_id}_{template['scenario_type']}_v{variant}"
        cost_range = template['estimated_cost']

        return {
            'scenario_id': scenario_id,
            'equipment_id': equipment_id,
            'scenario_type': template['scenario_type'],
            'description': template['description'],
            'trigger_conditions': template['trigger_conditions'],
            'expected_actions': template['expected_actions'],
            'priority': template['priority'],
            'estimated_cost': random.uniform(*cost_range),
            'estimated_duration_hours': random.uniform(1, 48),
            'required_resources': {
                'technicians': random.randint(1, 4),
                'specialist_required': random.choice([True, False]),
                'parts_needed': random.choice([True, False]),
                'external_contractor': random.choice([True, False])
            },
            'business_rules': {
                'cost_approval_required': random.uniform(*cost_range) > 2000,
                'safety_lockout_required': template['priority'] == 'CRITICAL',
                'documentation_required': True,
                'compliance_check_needed': random.choice([True, False])
            },
            'kpis': {
                'equipment_availability_impact': random.uniform(0.0, 0.3),
                'production_loss_estimate': random.uniform(0, 50000),
                'maintenance_efficiency_score': random.uniform(0.7, 0.95)
            }
        }

    def _generate_performance_dataset(self, num_sensors: int, data_points: int) -> Dict[str, Any]:
        """Generate a dataset for performance testing"""

        dataset = {
            'dataset_info': {
                'num_sensors': num_sensors,
                'data_points_per_sensor': data_points,
                'total_data_points': num_sensors * data_points,
                'estimated_size_mb': (num_sensors * data_points * 8) / (1024 * 1024)  # Rough estimate
            },
            'sensors': {}
        }

        for i in range(num_sensors):
            sensor_id = f"PERF_SENSOR_{i+1:04d}"
            sensor_type = random.choice(list(self.sensor_types.keys()))
            normal_range = self.sensor_types[sensor_type]['normal_range']

            # Generate simplified time series for performance testing
            values = np.random.normal(
                (normal_range[0] + normal_range[1]) / 2,
                (normal_range[1] - normal_range[0]) / 6,
                data_points
            )
            values = np.clip(values, *normal_range)

            dataset['sensors'][sensor_id] = {
                'sensor_type': sensor_type,
                'values': values.tolist(),
                'checksum': hashlib.md5(values.tobytes()).hexdigest()
            }

        return dataset

    def _generate_concurrent_access_data(self) -> Dict[str, Any]:
        """Generate data for concurrent access testing"""

        return {
            'user_sessions': [
                {
                    'user_id': f"user_{i+1:03d}",
                    'session_duration_minutes': random.uniform(5, 60),
                    'operations_per_minute': random.uniform(1, 20),
                    'operation_types': random.sample([
                        'sensor_data_query', 'anomaly_detection', 'report_generation',
                        'dashboard_view', 'maintenance_scheduling', 'alert_management'
                    ], random.randint(2, 4))
                }
                for i in range(50)  # 50 concurrent users
            ],
            'data_access_patterns': {
                'hot_sensors': [f"SENSOR_{i:03d}" for i in range(1, 21)],  # Frequently accessed
                'warm_sensors': [f"SENSOR_{i:03d}" for i in range(21, 61)],  # Moderately accessed
                'cold_sensors': [f"SENSOR_{i:03d}" for i in range(61, 81)]   # Rarely accessed
            }
        }

    def _generate_memory_pressure_data(self) -> Dict[str, Any]:
        """Generate data for memory pressure testing"""

        return {
            'large_datasets': [
                {
                    'dataset_id': f"large_dataset_{i+1}",
                    'size_mb': size_mb,
                    'data_points': int(size_mb * 1024 * 1024 / 8),  # Assuming 8 bytes per data point
                    'complexity_level': random.choice(['LOW', 'MEDIUM', 'HIGH'])
                }
                for i, size_mb in enumerate([10, 50, 100, 250, 500, 1000])
            ],
            'memory_allocation_patterns': [
                'sequential_allocation',
                'random_allocation',
                'fragmented_allocation',
                'burst_allocation'
            ]
        }

    def _generate_failure_scenario(self, failure_type: Dict, scenario_id: int) -> Dict[str, Any]:
        """Generate a specific failure scenario"""

        recovery_time_range = failure_type['recovery_time']

        return {
            'scenario_id': f"{failure_type['failure_type']}_{scenario_id:03d}",
            'failure_type': failure_type['failure_type'],
            'description': failure_type['description'],
            'impact_scope': failure_type['impact_scope'],
            'frequency': failure_type['frequency'],
            'trigger_conditions': self._generate_failure_triggers(failure_type),
            'expected_impact': {
                'affected_sensors': random.randint(1, 20) if failure_type['impact_scope'] != 'system' else 80,
                'data_loss_percentage': random.uniform(0, 10),
                'service_degradation': random.uniform(20, 80),
                'recovery_time_seconds': random.uniform(*recovery_time_range)
            },
            'recovery_procedures': self._generate_recovery_procedures(failure_type),
            'test_validation': {
                'detection_required': True,
                'isolation_required': failure_type['impact_scope'] in ['equipment', 'zone'],
                'automatic_recovery': random.choice([True, False]),
                'manual_intervention_required': failure_type['frequency'] == 'rare'
            }
        }

    def _generate_failure_triggers(self, failure_type: Dict) -> List[str]:
        """Generate failure trigger conditions"""

        trigger_map = {
            'sensor_failure': ['calibration_drift', 'physical_damage', 'electrical_fault'],
            'equipment_failure': ['component_wear', 'power_surge', 'environmental_stress'],
            'network_failure': ['connection_timeout', 'packet_loss', 'routing_error'],
            'power_failure': ['grid_outage', 'ups_failure', 'circuit_breaker_trip'],
            'software_failure': ['memory_leak', 'deadlock', 'configuration_error']
        }

        base_triggers = trigger_map.get(failure_type['failure_type'], ['unknown_trigger'])
        return random.sample(base_triggers, random.randint(1, len(base_triggers)))

    def _generate_recovery_procedures(self, failure_type: Dict) -> List[str]:
        """Generate recovery procedures for failure type"""

        procedure_map = {
            'sensor_failure': ['sensor_recalibration', 'sensor_replacement', 'backup_sensor_activation'],
            'equipment_failure': ['equipment_restart', 'component_replacement', 'failover_activation'],
            'network_failure': ['connection_retry', 'route_reconfiguration', 'backup_network_switch'],
            'power_failure': ['power_restoration', 'ups_activation', 'generator_start'],
            'software_failure': ['service_restart', 'configuration_reset', 'rollback_deployment']
        }

        base_procedures = procedure_map.get(failure_type['failure_type'], ['manual_intervention'])
        return random.sample(base_procedures, random.randint(1, len(base_procedures)))

    def _generate_smap_like_data(self) -> Dict[str, Any]:
        """Generate synthetic SMAP-like satellite data"""

        return {
            'mission_info': {
                'mission_name': 'SMAP_SYNTHETIC',
                'mission_type': 'satellite',
                'orbit_parameters': {
                    'altitude_km': 685,
                    'inclination_degrees': 98.1,
                    'period_minutes': 98.5
                }
            },
            'data_characteristics': {
                'sampling_frequency': '1_minute',
                'data_types': ['soil_moisture', 'temperature', 'radar_backscatter'],
                'spatial_resolution': '36_km',
                'temporal_coverage': '2019-2024'
            },
            'synthetic_sensors': [
                {
                    'sensor_id': f"SMAP_SYNTH_{i+1:03d}",
                    'instrument': random.choice(['radiometer', 'radar']),
                    'measurement_type': random.choice(['soil_moisture', 'temperature', 'backscatter']),
                    'frequency_ghz': random.choice([1.4, 1.26]),
                    'polarization': random.choice(['H', 'V', 'HV'])
                }
                for i in range(25)
            ]
        }

    def _generate_msl_like_data(self) -> Dict[str, Any]:
        """Generate synthetic MSL-like rover data"""

        return {
            'mission_info': {
                'mission_name': 'MSL_SYNTHETIC',
                'mission_type': 'rover',
                'landing_site': 'SYNTHETIC_CRATER',
                'mission_duration_sols': 3000
            },
            'rover_systems': {
                'propulsion': ['wheel_motors', 'suspension', 'steering'],
                'power': ['radioisotope_generator', 'batteries'],
                'thermal': ['heating_elements', 'cooling_systems'],
                'communication': ['uhf_antenna', 'x_band_antenna']
            },
            'synthetic_sensors': [
                {
                    'sensor_id': f"MSL_SYNTH_{i+1:03d}",
                    'subsystem': random.choice(['propulsion', 'power', 'thermal', 'communication']),
                    'measurement_type': random.choice(['temperature', 'voltage', 'current', 'pressure']),
                    'criticality': random.choice(['MISSION_CRITICAL', 'IMPORTANT', 'MONITORING']),
                    'sol_range': (1, 3000)
                }
                for i in range(55)
            ]
        }

    def _generate_combined_mission_data(self) -> Dict[str, Any]:
        """Generate combined mission scenario data"""

        return {
            'scenario_name': 'MULTI_MISSION_SYNTHETIC',
            'missions': ['SMAP_SYNTHETIC', 'MSL_SYNTHETIC'],
            'data_fusion_points': [
                'temporal_correlation',
                'environmental_conditions',
                'mission_status_correlation'
            ],
            'cross_mission_analytics': {
                'comparative_analysis': True,
                'failure_pattern_correlation': True,
                'resource_optimization': True
            }
        }

    def _generate_quality_dataset(self, quality_level: str) -> Dict[str, Any]:
        """Generate dataset with specific quality level"""

        quality_config = {
            'high': {
                'missing_rate': 0.01,
                'outlier_rate': 0.02,
                'noise_level': 0.01,
                'timestamp_consistency': 0.99
            },
            'medium': {
                'missing_rate': 0.05,
                'outlier_rate': 0.08,
                'noise_level': 0.05,
                'timestamp_consistency': 0.95
            },
            'low': {
                'missing_rate': 0.15,
                'outlier_rate': 0.20,
                'noise_level': 0.15,
                'timestamp_consistency': 0.85
            },
            'corrupted': {
                'missing_rate': 0.30,
                'outlier_rate': 0.40,
                'noise_level': 0.30,
                'timestamp_consistency': 0.60
            }
        }

        config = quality_config[quality_level]

        return {
            'quality_level': quality_level,
            'quality_metrics': config,
            'data_issues_injected': [
                f"missing_values_{config['missing_rate']*100:.0f}%",
                f"outliers_{config['outlier_rate']*100:.0f}%",
                f"noise_level_{config['noise_level']*100:.0f}%",
                f"timestamp_consistency_{config['timestamp_consistency']*100:.0f}%"
            ],
            'expected_processing_impact': {
                'preprocessing_time_increase': config['missing_rate'] * 2,
                'detection_accuracy_decrease': config['noise_level'] * 0.5,
                'system_reliability_impact': 1 - config['timestamp_consistency']
            }
        }

    def _generate_quality_issue_dataset(self, issue_type: str) -> Dict[str, Any]:
        """Generate dataset with specific quality issues"""

        issue_scenarios = {
            'missing_values': {
                'description': 'Dataset with various missing value patterns',
                'patterns': ['random_missing', 'sequential_missing', 'sensor_specific_missing'],
                'severity_levels': ['5%', '15%', '30%']
            },
            'outliers': {
                'description': 'Dataset with different types of outliers',
                'patterns': ['statistical_outliers', 'contextual_outliers', 'collective_outliers'],
                'severity_levels': ['rare', 'moderate', 'frequent']
            },
            'inconsistent_timestamps': {
                'description': 'Dataset with timestamp inconsistencies',
                'patterns': ['gaps', 'duplicates', 'out_of_order', 'timezone_issues'],
                'severity_levels': ['minor', 'moderate', 'severe']
            },
            'duplicate_records': {
                'description': 'Dataset with duplicate record patterns',
                'patterns': ['exact_duplicates', 'near_duplicates', 'temporal_duplicates'],
                'severity_levels': ['1%', '5%', '15%']
            },
            'schema_violations': {
                'description': 'Dataset with schema violation issues',
                'patterns': ['type_mismatches', 'range_violations', 'format_errors'],
                'severity_levels': ['isolated', 'systematic', 'widespread']
            },
            'encoding_errors': {
                'description': 'Dataset with encoding and format errors',
                'patterns': ['character_encoding', 'decimal_precision', 'unit_inconsistencies'],
                'severity_levels': ['minor', 'moderate', 'critical']
            }
        }

        scenario = issue_scenarios.get(issue_type, {})

        return {
            'issue_type': issue_type,
            'description': scenario.get('description', f'Dataset with {issue_type}'),
            'patterns': scenario.get('patterns', []),
            'severity_levels': scenario.get('severity_levels', []),
            'test_objectives': [
                'data_validation_effectiveness',
                'error_detection_accuracy',
                'system_robustness',
                'recovery_mechanisms'
            ]
        }

    def _calculate_quality_score(self) -> float:
        """Calculate data quality score based on configuration"""

        quality_factors = {
            'high': 0.95,
            'medium': 0.80,
            'low': 0.60
        }

        base_score = quality_factors.get(self.config.data_quality_level, 0.80)

        # Adjust based on noise and anomaly injection
        noise_penalty = self.config.noise_level * 0.1
        anomaly_penalty = self.config.anomaly_injection_rate * 0.1

        final_score = base_score - noise_penalty - anomaly_penalty
        return max(0.0, min(1.0, final_score))


def main():
    """Main function to generate test data"""

    # Configure test data generation
    config = TestDataConfig(
        smap_sensors=25,
        msl_sensors=55,
        total_sensors=80,
        data_points_per_sensor=1000,
        anomaly_injection_rate=0.05,
        seasonal_patterns=True,
        trend_patterns=True,
        noise_level=0.05,
        data_quality_level='high'
    )

    # Create generator and generate all test data
    generator = TestDataGenerator(config)
    generated_files = generator.generate_all_test_data()

    # Print summary
    print("\n" + "="*60)
    print("TEST DATA GENERATION COMPLETED")
    print("="*60)

    for data_type, file_path in generated_files.items():
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"{data_type:20} -> {file_path} ({file_size_mb:.2f} MB)")

    print(f"\nTotal files generated: {len(generated_files)}")
    print(f"Output directory: {generator.output_dir}")
    print("\nUse test_data_manifest.json for detailed information about generated datasets.")


if __name__ == "__main__":
    main()