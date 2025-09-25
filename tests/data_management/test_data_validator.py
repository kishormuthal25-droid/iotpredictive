"""
Test Data Validator

Validates generated test data for quality, consistency, and completeness:
- Data integrity checks
- Schema validation
- Statistical validation
- Cross-reference validation
- Performance benchmarking
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import hashlib
import statistics
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results"""
    validator_name: str
    data_type: str
    passed: bool
    score: float
    issues: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    timestamp: datetime


@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    completeness: float
    consistency: float
    accuracy: float
    validity: float
    uniqueness: float
    timeliness: float
    overall_score: float


class TestDataValidator:
    """Comprehensive test data validator"""

    def __init__(self, data_directory: str = "generated_test_data"):
        self.data_dir = Path(data_directory)
        self.validation_results = []

        # Validation thresholds
        self.thresholds = {
            'completeness_min': 0.95,
            'consistency_min': 0.90,
            'accuracy_min': 0.85,
            'validity_min': 0.95,
            'uniqueness_min': 0.98,
            'timeliness_min': 0.90,
            'overall_min': 0.90
        }

        logger.info(f"TestDataValidator initialized for directory: {self.data_dir}")

    def validate_all_data(self) -> Dict[str, ValidationResult]:
        """Validate all generated test data"""
        logger.info("Starting comprehensive test data validation...")

        validation_results = {}

        # Load data manifest
        manifest_path = self.data_dir / "test_data_manifest.json"
        if not manifest_path.exists():
            logger.error(f"Data manifest not found: {manifest_path}")
            return validation_results

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Validate each data type
        for data_type, file_path in manifest['generated_files'].items():
            if data_type == 'manifest':
                continue

            logger.info(f"Validating {data_type}...")

            try:
                result = self._validate_data_file(data_type, file_path)
                validation_results[data_type] = result
                self.validation_results.append(result)

            except Exception as e:
                logger.error(f"Validation failed for {data_type}: {e}")
                validation_results[data_type] = ValidationResult(
                    validator_name="file_validator",
                    data_type=data_type,
                    passed=False,
                    score=0.0,
                    issues=[f"Validation error: {str(e)}"],
                    warnings=[],
                    statistics={},
                    timestamp=datetime.now()
                )

        # Cross-validation checks
        cross_validation_result = self._perform_cross_validation(manifest, validation_results)
        validation_results['cross_validation'] = cross_validation_result

        # Generate validation report
        self._generate_validation_report(validation_results, manifest)

        logger.info(f"Validation completed for {len(validation_results)} datasets")
        return validation_results

    def _validate_data_file(self, data_type: str, file_path: str) -> ValidationResult:
        """Validate individual data file"""

        # Load data
        if not os.path.exists(file_path):
            return ValidationResult(
                validator_name="file_validator",
                data_type=data_type,
                passed=False,
                score=0.0,
                issues=[f"File not found: {file_path}"],
                warnings=[],
                statistics={},
                timestamp=datetime.now()
            )

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return ValidationResult(
                validator_name="file_validator",
                data_type=data_type,
                passed=False,
                score=0.0,
                issues=[f"Failed to load JSON: {str(e)}"],
                warnings=[],
                statistics={},
                timestamp=datetime.now()
            )

        # Route to specific validator
        validator_map = {
            'sensor_metadata': self._validate_sensor_metadata,
            'sensor_data': self._validate_sensor_time_series,
            'anomaly_data': self._validate_anomaly_scenarios,
            'business_data': self._validate_business_scenarios,
            'performance_data': self._validate_performance_test_data,
            'failure_data': self._validate_failure_scenarios,
            'nasa_data': self._validate_nasa_like_datasets,
            'quality_data': self._validate_data_quality_test_sets
        }

        validator_func = validator_map.get(data_type, self._validate_generic_data)
        return validator_func(data, data_type, file_path)

    def _validate_sensor_metadata(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate sensor metadata"""
        issues = []
        warnings = []
        statistics = {}

        # Check if data is a list
        if not isinstance(data, list):
            issues.append("Sensor metadata should be a list of sensor objects")
            return self._create_failed_result("metadata_validator", data_type, issues, warnings, statistics)

        # Count sensors by type
        sensor_counts = {}
        sensor_types = set()
        equipment_ids = set()
        locations = set()

        required_fields = ['sensor_id', 'sensor_type', 'equipment_id', 'location', 'criticality', 'unit', 'normal_range']

        for i, sensor in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in sensor:
                    issues.append(f"Sensor {i}: Missing required field '{field}'")

            # Validate sensor_id format
            if 'sensor_id' in sensor:
                sensor_id = sensor['sensor_id']
                if not re.match(r'^(SMAP|MSL)_\d{3}$', sensor_id):
                    issues.append(f"Sensor {i}: Invalid sensor_id format: {sensor_id}")

            # Count by type
            if 'sensor_type' in sensor:
                sensor_type = sensor['sensor_type']
                sensor_types.add(sensor_type)
                sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1

            # Collect equipment IDs and locations
            if 'equipment_id' in sensor:
                equipment_ids.add(sensor['equipment_id'])
            if 'location' in sensor:
                locations.add(sensor['location'])

            # Validate normal_range
            if 'normal_range' in sensor:
                normal_range = sensor['normal_range']
                if not isinstance(normal_range, list) or len(normal_range) != 2:
                    issues.append(f"Sensor {i}: normal_range should be a list of two values")
                elif normal_range[0] >= normal_range[1]:
                    issues.append(f"Sensor {i}: normal_range min should be less than max")

            # Validate criticality
            if 'criticality' in sensor:
                if sensor['criticality'] not in ['LOW', 'MEDIUM', 'HIGH']:
                    issues.append(f"Sensor {i}: Invalid criticality level: {sensor['criticality']}")

        # Expected counts
        expected_smap = 25
        expected_msl = 55
        total_expected = expected_smap + expected_msl

        smap_count = len([s for s in data if s.get('sensor_id', '').startswith('SMAP')])
        msl_count = len([s for s in data if s.get('sensor_id', '').startswith('MSL')])

        if smap_count != expected_smap:
            issues.append(f"Expected {expected_smap} SMAP sensors, found {smap_count}")
        if msl_count != expected_msl:
            issues.append(f"Expected {expected_msl} MSL sensors, found {msl_count}")

        # Check for duplicates
        sensor_ids = [s.get('sensor_id') for s in data if 'sensor_id' in s]
        if len(sensor_ids) != len(set(sensor_ids)):
            issues.append("Duplicate sensor_ids found")

        # Statistics
        statistics = {
            'total_sensors': len(data),
            'smap_sensors': smap_count,
            'msl_sensors': msl_count,
            'sensor_types': list(sensor_types),
            'sensor_type_counts': sensor_counts,
            'unique_equipment_ids': len(equipment_ids),
            'unique_locations': len(locations)
        }

        # Calculate score
        score = self._calculate_metadata_score(data, issues, statistics)

        return ValidationResult(
            validator_name="metadata_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_sensor_time_series(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate sensor time series data"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, dict):
            issues.append("Sensor time series data should be a dictionary")
            return self._create_failed_result("timeseries_validator", data_type, issues, warnings, statistics)

        total_sensors = len(data)
        total_data_points = 0
        anomaly_counts = []
        quality_scores = []
        timestamp_issues = 0
        value_range_issues = 0

        for sensor_id, sensor_data in data.items():
            # Check required fields
            required_fields = ['sensor_id', 'sensor_type', 'timestamps', 'values', 'normal_range']
            for field in required_fields:
                if field not in sensor_data:
                    issues.append(f"Sensor {sensor_id}: Missing required field '{field}'")

            # Validate data consistency
            if 'timestamps' in sensor_data and 'values' in sensor_data:
                timestamps = sensor_data['timestamps']
                values = sensor_data['values']

                if len(timestamps) != len(values):
                    issues.append(f"Sensor {sensor_id}: Timestamp and value counts don't match")

                total_data_points += len(values)

                # Validate timestamp format and ordering
                try:
                    parsed_timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps[:10]]
                    if len(parsed_timestamps) > 1:
                        for i in range(1, len(parsed_timestamps)):
                            if parsed_timestamps[i] <= parsed_timestamps[i-1]:
                                timestamp_issues += 1
                                break
                except Exception:
                    issues.append(f"Sensor {sensor_id}: Invalid timestamp format")

                # Validate value ranges
                if 'normal_range' in sensor_data:
                    normal_range = sensor_data['normal_range']
                    if isinstance(normal_range, list) and len(normal_range) == 2:
                        min_val, max_val = normal_range
                        out_of_range = sum(1 for v in values if v < min_val * 0.5 or v > max_val * 2.0)
                        if out_of_range > len(values) * 0.1:  # More than 10% out of range
                            value_range_issues += 1
                            warnings.append(f"Sensor {sensor_id}: {out_of_range} values significantly out of range")

            # Collect anomaly statistics
            if 'anomaly_indices' in sensor_data:
                anomaly_counts.append(len(sensor_data['anomaly_indices']))

            # Collect quality scores
            if 'data_quality_score' in sensor_data:
                quality_scores.append(sensor_data['data_quality_score'])

        # Statistics
        statistics = {
            'total_sensors': total_sensors,
            'total_data_points': total_data_points,
            'avg_data_points_per_sensor': total_data_points / max(total_sensors, 1),
            'timestamp_issues': timestamp_issues,
            'value_range_issues': value_range_issues,
            'avg_anomalies_per_sensor': np.mean(anomaly_counts) if anomaly_counts else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0
        }

        # Calculate score
        score = self._calculate_timeseries_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="timeseries_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_anomaly_scenarios(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate anomaly scenarios"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, list):
            issues.append("Anomaly scenarios should be a list")
            return self._create_failed_result("anomaly_validator", data_type, issues, warnings, statistics)

        # Required fields for anomaly scenarios
        required_fields = ['scenario_id', 'sensor_id', 'anomaly_type', 'failure_mode', 'severity']

        anomaly_types = set()
        failure_modes = set()
        severity_counts = {}
        sensor_coverage = set()

        for i, scenario in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in scenario:
                    issues.append(f"Scenario {i}: Missing required field '{field}'")

            # Validate scenario_id format
            if 'scenario_id' in scenario:
                scenario_id = scenario['scenario_id']
                if not re.match(r'^(SMAP|MSL)_\d{3}_\w+_\w+$', scenario_id):
                    warnings.append(f"Scenario {i}: Unusual scenario_id format: {scenario_id}")

            # Collect statistics
            if 'anomaly_type' in scenario:
                anomaly_types.add(scenario['anomaly_type'])
            if 'failure_mode' in scenario:
                failure_modes.add(scenario['failure_mode'])
            if 'severity' in scenario:
                severity = scenario['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            if 'sensor_id' in scenario:
                sensor_coverage.add(scenario['sensor_id'])

            # Validate severity levels
            if 'severity' in scenario:
                if scenario['severity'] not in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                    issues.append(f"Scenario {i}: Invalid severity level: {scenario['severity']}")

            # Validate expected detection time
            if 'expected_detection_time' in scenario:
                detection_time = scenario['expected_detection_time']
                if not isinstance(detection_time, (int, float)) or detection_time <= 0:
                    issues.append(f"Scenario {i}: Invalid expected_detection_time: {detection_time}")

        # Statistics
        statistics = {
            'total_scenarios': len(data),
            'unique_anomaly_types': len(anomaly_types),
            'unique_failure_modes': len(failure_modes),
            'severity_distribution': severity_counts,
            'sensor_coverage': len(sensor_coverage),
            'anomaly_types': list(anomaly_types),
            'failure_modes': list(failure_modes)
        }

        # Calculate score
        score = self._calculate_anomaly_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="anomaly_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_business_scenarios(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate business scenarios"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, list):
            issues.append("Business scenarios should be a list")
            return self._create_failed_result("business_validator", data_type, issues, warnings, statistics)

        required_fields = ['scenario_id', 'equipment_id', 'scenario_type', 'priority', 'estimated_cost']

        scenario_types = set()
        priority_counts = {}
        cost_ranges = []
        equipment_coverage = set()

        for i, scenario in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in scenario:
                    issues.append(f"Scenario {i}: Missing required field '{field}'")

            # Collect statistics
            if 'scenario_type' in scenario:
                scenario_types.add(scenario['scenario_type'])
            if 'priority' in scenario:
                priority = scenario['priority']
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            if 'estimated_cost' in scenario:
                cost_ranges.append(scenario['estimated_cost'])
            if 'equipment_id' in scenario:
                equipment_coverage.add(scenario['equipment_id'])

            # Validate priority levels
            if 'priority' in scenario:
                if scenario['priority'] not in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
                    issues.append(f"Scenario {i}: Invalid priority level: {scenario['priority']}")

            # Validate cost estimates
            if 'estimated_cost' in scenario:
                cost = scenario['estimated_cost']
                if not isinstance(cost, (int, float)) or cost < 0:
                    issues.append(f"Scenario {i}: Invalid estimated_cost: {cost}")

        # Statistics
        statistics = {
            'total_scenarios': len(data),
            'unique_scenario_types': len(scenario_types),
            'priority_distribution': priority_counts,
            'cost_statistics': {
                'min_cost': min(cost_ranges) if cost_ranges else 0,
                'max_cost': max(cost_ranges) if cost_ranges else 0,
                'avg_cost': np.mean(cost_ranges) if cost_ranges else 0,
                'median_cost': np.median(cost_ranges) if cost_ranges else 0
            },
            'equipment_coverage': len(equipment_coverage),
            'scenario_types': list(scenario_types)
        }

        # Calculate score
        score = self._calculate_business_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="business_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_performance_test_data(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate performance test data"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, dict):
            issues.append("Performance test data should be a dictionary")
            return self._create_failed_result("performance_validator", data_type, issues, warnings, statistics)

        dataset_sizes = []
        dataset_types = list(data.keys())

        for dataset_name, dataset in data.items():
            if dataset_name in ['small', 'medium', 'large', 'xlarge', 'stress']:
                # Validate scalability datasets
                if 'dataset_info' not in dataset:
                    issues.append(f"Dataset {dataset_name}: Missing dataset_info")
                    continue

                info = dataset['dataset_info']
                if 'total_data_points' in info:
                    dataset_sizes.append(info['total_data_points'])

                # Check that larger datasets actually have more data
                if dataset_name == 'medium' and 'small' in data:
                    if info.get('total_data_points', 0) <= data['small']['dataset_info'].get('total_data_points', 0):
                        issues.append("Medium dataset should be larger than small dataset")

        # Statistics
        statistics = {
            'dataset_types': dataset_types,
            'scalability_datasets': len([d for d in dataset_types if d in ['small', 'medium', 'large', 'xlarge', 'stress']]),
            'dataset_size_range': {
                'min_size': min(dataset_sizes) if dataset_sizes else 0,
                'max_size': max(dataset_sizes) if dataset_sizes else 0
            },
            'specialized_datasets': [d for d in dataset_types if d not in ['small', 'medium', 'large', 'xlarge', 'stress']]
        }

        # Calculate score
        score = self._calculate_performance_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="performance_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_failure_scenarios(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate failure scenarios"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, list):
            issues.append("Failure scenarios should be a list")
            return self._create_failed_result("failure_validator", data_type, issues, warnings, statistics)

        required_fields = ['scenario_id', 'failure_type', 'impact_scope', 'expected_impact']

        failure_types = set()
        impact_scopes = set()
        recovery_times = []

        for i, scenario in enumerate(data):
            # Check required fields
            for field in required_fields:
                if field not in scenario:
                    issues.append(f"Scenario {i}: Missing required field '{field}'")

            # Collect statistics
            if 'failure_type' in scenario:
                failure_types.add(scenario['failure_type'])
            if 'impact_scope' in scenario:
                impact_scopes.add(scenario['impact_scope'])

            # Validate expected impact
            if 'expected_impact' in scenario:
                impact = scenario['expected_impact']
                if 'recovery_time_seconds' in impact:
                    recovery_time = impact['recovery_time_seconds']
                    if isinstance(recovery_time, (int, float)) and recovery_time > 0:
                        recovery_times.append(recovery_time)
                    else:
                        issues.append(f"Scenario {i}: Invalid recovery_time_seconds: {recovery_time}")

        # Statistics
        statistics = {
            'total_scenarios': len(data),
            'unique_failure_types': len(failure_types),
            'unique_impact_scopes': len(impact_scopes),
            'recovery_time_statistics': {
                'min_recovery_time': min(recovery_times) if recovery_times else 0,
                'max_recovery_time': max(recovery_times) if recovery_times else 0,
                'avg_recovery_time': np.mean(recovery_times) if recovery_times else 0
            },
            'failure_types': list(failure_types),
            'impact_scopes': list(impact_scopes)
        }

        # Calculate score
        score = self._calculate_failure_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="failure_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_nasa_like_datasets(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate NASA-like datasets"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, dict):
            issues.append("NASA-like datasets should be a dictionary")
            return self._create_failed_result("nasa_validator", data_type, issues, warnings, statistics)

        required_datasets = ['smap_synthetic', 'msl_synthetic', 'combined_mission']

        for dataset_name in required_datasets:
            if dataset_name not in data:
                issues.append(f"Missing required dataset: {dataset_name}")

        # Validate SMAP synthetic data
        if 'smap_synthetic' in data:
            smap_data = data['smap_synthetic']
            if 'mission_info' not in smap_data:
                issues.append("SMAP synthetic: Missing mission_info")
            if 'synthetic_sensors' not in smap_data:
                issues.append("SMAP synthetic: Missing synthetic_sensors")
            elif len(smap_data['synthetic_sensors']) != 25:
                warnings.append(f"SMAP synthetic: Expected 25 sensors, found {len(smap_data['synthetic_sensors'])}")

        # Validate MSL synthetic data
        if 'msl_synthetic' in data:
            msl_data = data['msl_synthetic']
            if 'mission_info' not in msl_data:
                issues.append("MSL synthetic: Missing mission_info")
            if 'synthetic_sensors' not in msl_data:
                issues.append("MSL synthetic: Missing synthetic_sensors")
            elif len(msl_data['synthetic_sensors']) != 55:
                warnings.append(f"MSL synthetic: Expected 55 sensors, found {len(msl_data['synthetic_sensors'])}")

        # Statistics
        statistics = {
            'datasets_present': list(data.keys()),
            'smap_sensors': len(data.get('smap_synthetic', {}).get('synthetic_sensors', [])),
            'msl_sensors': len(data.get('msl_synthetic', {}).get('synthetic_sensors', [])),
            'combined_mission_present': 'combined_mission' in data
        }

        # Calculate score
        score = self._calculate_nasa_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="nasa_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_data_quality_test_sets(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Validate data quality test sets"""
        issues = []
        warnings = []
        statistics = {}

        if not isinstance(data, dict):
            issues.append("Data quality test sets should be a dictionary")
            return self._create_failed_result("quality_validator", data_type, issues, warnings, statistics)

        expected_quality_levels = ['high', 'medium', 'low', 'corrupted']
        expected_issue_types = ['missing_values', 'outliers', 'inconsistent_timestamps',
                               'duplicate_records', 'schema_violations', 'encoding_errors']

        quality_levels_present = []
        issue_types_present = []

        for dataset_name, dataset in data.items():
            if dataset_name in expected_quality_levels:
                quality_levels_present.append(dataset_name)
            elif dataset_name.startswith('issue_'):
                issue_type = dataset_name.replace('issue_', '')
                issue_types_present.append(issue_type)

        # Check coverage
        missing_quality_levels = set(expected_quality_levels) - set(quality_levels_present)
        if missing_quality_levels:
            warnings.append(f"Missing quality levels: {list(missing_quality_levels)}")

        missing_issue_types = set(expected_issue_types) - set(issue_types_present)
        if missing_issue_types:
            warnings.append(f"Missing issue types: {list(missing_issue_types)}")

        # Statistics
        statistics = {
            'total_datasets': len(data),
            'quality_levels_present': quality_levels_present,
            'issue_types_present': issue_types_present,
            'coverage_quality_levels': len(quality_levels_present) / len(expected_quality_levels),
            'coverage_issue_types': len(issue_types_present) / len(expected_issue_types)
        }

        # Calculate score
        score = self._calculate_quality_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="quality_validator",
            data_type=data_type,
            passed=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _validate_generic_data(self, data: Any, data_type: str, file_path: str) -> ValidationResult:
        """Generic validation for unknown data types"""
        issues = []
        warnings = []
        statistics = {}

        # Basic JSON validation (already passed if we got here)
        file_size = os.path.getsize(file_path)

        # Basic structure checks
        if isinstance(data, dict):
            statistics['structure'] = 'dictionary'
            statistics['keys'] = list(data.keys())[:10]  # First 10 keys
        elif isinstance(data, list):
            statistics['structure'] = 'list'
            statistics['length'] = len(data)
        else:
            statistics['structure'] = type(data).__name__

        statistics['file_size_bytes'] = file_size
        statistics['file_size_mb'] = file_size / (1024 * 1024)

        # Score based on file existence and basic structure
        score = 0.8  # Basic score for valid JSON

        return ValidationResult(
            validator_name="generic_validator",
            data_type=data_type,
            passed=True,
            score=score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _perform_cross_validation(self, manifest: Dict, validation_results: Dict) -> ValidationResult:
        """Perform cross-validation between datasets"""
        issues = []
        warnings = []
        statistics = {}

        # Check sensor ID consistency between metadata and time series
        try:
            metadata_path = self.data_dir / "sensor_metadata.json"
            timeseries_path = self.data_dir / "sensor_time_series.json"

            if metadata_path.exists() and timeseries_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                with open(timeseries_path, 'r') as f:
                    timeseries = json.load(f)

                # Extract sensor IDs
                metadata_sensors = set(s['sensor_id'] for s in metadata if 'sensor_id' in s)
                timeseries_sensors = set(timeseries.keys())

                # Check consistency
                missing_in_timeseries = metadata_sensors - timeseries_sensors
                missing_in_metadata = timeseries_sensors - metadata_sensors

                if missing_in_timeseries:
                    issues.append(f"Sensors in metadata but not in timeseries: {list(missing_in_timeseries)[:5]}")
                if missing_in_metadata:
                    issues.append(f"Sensors in timeseries but not in metadata: {list(missing_in_metadata)[:5]}")

                statistics['sensor_consistency'] = {
                    'metadata_sensors': len(metadata_sensors),
                    'timeseries_sensors': len(timeseries_sensors),
                    'intersection': len(metadata_sensors & timeseries_sensors),
                    'missing_in_timeseries': len(missing_in_timeseries),
                    'missing_in_metadata': len(missing_in_metadata)
                }
        except Exception as e:
            warnings.append(f"Could not perform sensor ID cross-validation: {e}")

        # Check data volume consistency
        total_data_volume = sum(
            result.statistics.get('total_data_points', 0)
            for result in validation_results.values()
            if hasattr(result, 'statistics')
        )

        statistics['data_volume_summary'] = {
            'total_estimated_data_points': total_data_volume,
            'datasets_validated': len(validation_results),
            'validation_success_rate': sum(1 for r in validation_results.values() if r.passed) / max(len(validation_results), 1)
        }

        # Calculate cross-validation score
        cross_validation_score = self._calculate_cross_validation_score(statistics, issues, warnings)

        return ValidationResult(
            validator_name="cross_validator",
            data_type="cross_validation",
            passed=len(issues) == 0,
            score=cross_validation_score,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _calculate_metadata_score(self, data: List, issues: List, statistics: Dict) -> float:
        """Calculate metadata validation score"""
        base_score = 1.0

        # Deduct for critical issues
        critical_deductions = len([i for i in issues if 'Missing required field' in i]) * 0.1
        format_deductions = len([i for i in issues if 'Invalid' in i]) * 0.05
        count_deductions = len([i for i in issues if 'Expected' in i and 'found' in i]) * 0.1

        total_deduction = critical_deductions + format_deductions + count_deductions
        return max(0.0, base_score - total_deduction)

    def _calculate_timeseries_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate time series validation score"""
        base_score = 1.0

        # Deduct for issues
        critical_deductions = len(issues) * 0.1
        warning_deductions = len(warnings) * 0.02

        # Quality bonus
        avg_quality = statistics.get('avg_quality_score', 0.8)
        quality_bonus = (avg_quality - 0.8) * 0.2

        final_score = base_score - critical_deductions - warning_deductions + quality_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_anomaly_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate anomaly scenarios validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.1 + len(warnings) * 0.02

        # Coverage bonus
        anomaly_type_coverage = min(statistics.get('unique_anomaly_types', 0) / 5, 1.0)  # Expect at least 5 types
        failure_mode_coverage = min(statistics.get('unique_failure_modes', 0) / 10, 1.0)  # Expect at least 10 modes

        coverage_bonus = (anomaly_type_coverage + failure_mode_coverage) * 0.1

        final_score = base_score - deductions + coverage_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_business_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate business scenarios validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.1 + len(warnings) * 0.02

        # Coverage and distribution bonuses
        scenario_type_coverage = min(statistics.get('unique_scenario_types', 0) / 4, 1.0)  # Expect at least 4 types
        equipment_coverage = min(statistics.get('equipment_coverage', 0) / 20, 1.0)  # Expect reasonable coverage

        coverage_bonus = (scenario_type_coverage + equipment_coverage) * 0.1

        final_score = base_score - deductions + coverage_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_performance_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate performance test data validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.15 + len(warnings) * 0.05

        # Coverage bonus for scalability datasets
        scalability_coverage = min(statistics.get('scalability_datasets', 0) / 5, 1.0)  # Expect 5 scalability levels
        coverage_bonus = scalability_coverage * 0.1

        final_score = base_score - deductions + coverage_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_failure_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate failure scenarios validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.1 + len(warnings) * 0.02

        # Coverage bonus
        failure_type_coverage = min(statistics.get('unique_failure_types', 0) / 5, 1.0)  # Expect 5 failure types
        impact_scope_coverage = min(statistics.get('unique_impact_scopes', 0) / 4, 1.0)  # Expect 4 impact scopes

        coverage_bonus = (failure_type_coverage + impact_scope_coverage) * 0.1

        final_score = base_score - deductions + coverage_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_nasa_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate NASA-like datasets validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.15 + len(warnings) * 0.05

        # Completeness bonus
        datasets_expected = 3  # smap, msl, combined
        datasets_present = len(statistics.get('datasets_present', []))
        completeness_bonus = (datasets_present / datasets_expected) * 0.1

        final_score = base_score - deductions + completeness_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_quality_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate quality test sets validation score"""
        base_score = 1.0

        # Deduct for issues
        deductions = len(issues) * 0.1 + len(warnings) * 0.02

        # Coverage bonuses
        quality_coverage = statistics.get('coverage_quality_levels', 0)
        issue_coverage = statistics.get('coverage_issue_types', 0)
        coverage_bonus = (quality_coverage + issue_coverage) * 0.1

        final_score = base_score - deductions + coverage_bonus
        return max(0.0, min(1.0, final_score))

    def _calculate_cross_validation_score(self, statistics: Dict, issues: List, warnings: List) -> float:
        """Calculate cross-validation score"""
        base_score = 1.0

        # Deduct for consistency issues
        deductions = len(issues) * 0.2 + len(warnings) * 0.05

        # Success rate bonus
        success_rate = statistics.get('data_volume_summary', {}).get('validation_success_rate', 0.8)
        success_bonus = (success_rate - 0.8) * 0.2

        final_score = base_score - deductions + success_bonus
        return max(0.0, min(1.0, final_score))

    def _create_failed_result(self, validator_name: str, data_type: str, issues: List,
                            warnings: List, statistics: Dict) -> ValidationResult:
        """Create a failed validation result"""
        return ValidationResult(
            validator_name=validator_name,
            data_type=data_type,
            passed=False,
            score=0.0,
            issues=issues,
            warnings=warnings,
            statistics=statistics,
            timestamp=datetime.now()
        )

    def _generate_validation_report(self, validation_results: Dict, manifest: Dict):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")

        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'validator_version': '1.0.0',
                'total_datasets_validated': len(validation_results),
                'datasets_passed': len([r for r in validation_results.values() if r.passed]),
                'datasets_failed': len([r for r in validation_results.values() if not r.passed]),
                'overall_success_rate': len([r for r in validation_results.values() if r.passed]) / max(len(validation_results), 1),
                'average_validation_score': np.mean([r.score for r in validation_results.values()])
            },
            'dataset_results': {
                data_type: {
                    'validator': result.validator_name,
                    'passed': result.passed,
                    'score': result.score,
                    'issues_count': len(result.issues),
                    'warnings_count': len(result.warnings),
                    'issues': result.issues,
                    'warnings': result.warnings,
                    'statistics': result.statistics
                }
                for data_type, result in validation_results.items()
            },
            'recommendations': self._generate_recommendations(validation_results),
            'data_quality_assessment': self._assess_overall_data_quality(validation_results)
        }

        # Save validation report
        report_path = self.data_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {report_path}")

    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        failed_datasets = [data_type for data_type, result in validation_results.items() if not result.passed]
        if failed_datasets:
            recommendations.append(f"Review and fix validation issues in: {', '.join(failed_datasets)}")

        low_score_datasets = [
            data_type for data_type, result in validation_results.items()
            if result.score < 0.8
        ]
        if low_score_datasets:
            recommendations.append(f"Improve data quality for datasets with low scores: {', '.join(low_score_datasets)}")

        # Add specific recommendations based on common issues
        all_issues = []
        for result in validation_results.values():
            all_issues.extend(result.issues)

        if any("Missing required field" in issue for issue in all_issues):
            recommendations.append("Ensure all required fields are present in generated data")

        if any("Invalid" in issue for issue in all_issues):
            recommendations.append("Review data format validation and generation logic")

        if any("timestamp" in issue.lower() for issue in all_issues):
            recommendations.append("Improve timestamp generation and validation")

        if not recommendations:
            recommendations.append("All validations passed! Data quality is excellent.")

        return recommendations

    def _assess_overall_data_quality(self, validation_results: Dict) -> DataQualityMetrics:
        """Assess overall data quality metrics"""

        scores = [r.score for r in validation_results.values()]

        # Calculate individual quality dimensions
        completeness = np.mean([r.score for r in validation_results.values() if 'metadata' in r.data_type or 'timeseries' in r.data_type])
        consistency = validation_results.get('cross_validation', ValidationResult('', '', False, 0.8, [], [], {}, datetime.now())).score
        accuracy = np.mean([r.score for r in validation_results.values() if 'anomaly' in r.data_type or 'business' in r.data_type])
        validity = np.mean([r.score for r in validation_results.values() if 'failure' in r.data_type or 'nasa' in r.data_type])
        uniqueness = 0.95  # Assume high uniqueness based on ID validation
        timeliness = 0.95  # Assume high timeliness for generated data

        overall_score = np.mean(scores) if scores else 0.0

        return DataQualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            validity=validity,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score
        )


def main():
    """Main function to validate test data"""

    # Initialize validator
    validator = TestDataValidator()

    # Validate all data
    results = validator.validate_all_data()

    # Print summary
    print("\n" + "="*60)
    print("TEST DATA VALIDATION COMPLETED")
    print("="*60)

    for data_type, result in results.items():
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{data_type:20} -> {status} (Score: {result.score:.3f})")

        if result.issues:
            print(f"{'':22} Issues: {len(result.issues)}")
        if result.warnings:
            print(f"{'':22} Warnings: {len(result.warnings)}")

    overall_pass_rate = len([r for r in results.values() if r.passed]) / len(results)
    avg_score = np.mean([r.score for r in results.values()])

    print(f"\nOverall Pass Rate: {overall_pass_rate:.1%}")
    print(f"Average Score: {avg_score:.3f}")
    print(f"\nDetailed validation report saved to: validation_report.json")


if __name__ == "__main__":
    main()