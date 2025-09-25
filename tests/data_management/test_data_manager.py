"""
Test Data Manager

Centralized management of test data lifecycle:
- Test data generation orchestration
- Data validation and quality assurance
- Test data provisioning for different test suites
- Data cleanup and archival
- Test data versioning and cataloging
"""

import os
import json
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import hashlib
import tempfile

# Import our test data components
from test_data_generator import TestDataGenerator, TestDataConfig
from test_data_validator import TestDataValidator, ValidationResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestDataVersion:
    """Test data version information"""
    version: str
    creation_date: datetime
    generator_config: Dict[str, Any]
    validation_status: str
    data_quality_score: float
    file_count: int
    total_size_mb: float
    description: str


@dataclass
class TestDataRequest:
    """Test data provisioning request"""
    request_id: str
    test_suite: str
    data_types: List[str]
    quality_level: str
    volume_requirements: Dict[str, Any]
    custom_parameters: Dict[str, Any]
    deadline: datetime


class TestDataManager:
    """Centralized test data management system"""

    def __init__(self, base_directory: str = "test_data_management", archive_directory: str = "test_data_archive"):
        self.base_dir = Path(base_directory)
        self.archive_dir = Path(archive_directory)
        self.current_version_dir = self.base_dir / "current"
        self.catalog_file = self.base_dir / "data_catalog.json"

        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        self.current_version_dir.mkdir(exist_ok=True)

        # Load existing catalog
        self.data_catalog = self._load_data_catalog()

        logger.info(f"TestDataManager initialized with base directory: {self.base_dir}")

    def generate_and_validate_test_data(self, config: TestDataConfig = None) -> str:
        """Generate and validate new test data version"""
        logger.info("Starting test data generation and validation process...")

        # Use default config if none provided
        if config is None:
            config = TestDataConfig()

        # Create version identifier
        version = self._create_version_identifier()
        version_dir = self.base_dir / f"version_{version}"
        version_dir.mkdir(exist_ok=True)

        # Generate test data
        logger.info(f"Generating test data version {version}...")
        generator = TestDataGenerator(config, str(version_dir))
        generated_files = generator.generate_all_test_data()

        # Validate generated data
        logger.info(f"Validating test data version {version}...")
        validator = TestDataValidator(str(version_dir))
        validation_results = validator.validate_all_data()

        # Calculate overall validation metrics
        validation_status = "PASSED" if all(r.passed for r in validation_results.values()) else "FAILED"
        avg_score = sum(r.score for r in validation_results.values()) / len(validation_results)

        # Calculate total size
        total_size = sum(
            os.path.getsize(file_path) for file_path in generated_files.values()
            if os.path.exists(file_path)
        )

        # Create version record
        version_record = TestDataVersion(
            version=version,
            creation_date=datetime.now(),
            generator_config=asdict(config),
            validation_status=validation_status,
            data_quality_score=avg_score,
            file_count=len(generated_files),
            total_size_mb=total_size / (1024 * 1024),
            description=f"Auto-generated test data with {config.total_sensors} sensors"
        )

        # Update catalog
        self._update_data_catalog(version_record, generated_files, validation_results)

        # Set as current version if validation passed
        if validation_status == "PASSED":
            self._set_current_version(version)
            logger.info(f"Test data version {version} set as current version")
        else:
            logger.warning(f"Test data version {version} failed validation - not set as current")

        # Save version metadata
        self._save_version_metadata(version_dir, version_record, generated_files, validation_results)

        logger.info(f"Test data generation and validation completed for version {version}")
        return version

    def provision_test_data(self, request: TestDataRequest) -> str:
        """Provision test data for specific test suite"""
        logger.info(f"Provisioning test data for request {request.request_id}")

        # Create provisioning directory
        provision_dir = self.base_dir / "provisioned" / request.test_suite / request.request_id
        provision_dir.mkdir(parents=True, exist_ok=True)

        # Get current version
        current_version = self._get_current_version()
        if not current_version:
            raise ValueError("No current test data version available")

        current_version_dir = self.base_dir / f"version_{current_version}"

        # Copy requested data types
        provisioned_files = {}
        for data_type in request.data_types:
            source_file = self._find_data_file(current_version_dir, data_type)
            if source_file:
                target_file = provision_dir / f"{data_type}.json"
                shutil.copy2(source_file, target_file)
                provisioned_files[data_type] = str(target_file)
                logger.info(f"Provisioned {data_type} -> {target_file}")

        # Apply custom transformations if needed
        if request.custom_parameters:
            provisioned_files = self._apply_custom_transformations(
                provisioned_files, request.custom_parameters
            )

        # Create provisioning manifest
        manifest = {
            'request_id': request.request_id,
            'test_suite': request.test_suite,
            'provision_timestamp': datetime.now().isoformat(),
            'source_version': current_version,
            'provisioned_files': provisioned_files,
            'request_details': asdict(request)
        }

        manifest_file = provision_dir / "provisioning_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        logger.info(f"Test data provisioning completed for {request.test_suite}")
        return str(provision_dir)

    def archive_test_data_version(self, version: str) -> str:
        """Archive a test data version"""
        logger.info(f"Archiving test data version {version}")

        version_dir = self.base_dir / f"version_{version}"
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        # Create archive
        archive_file = self.archive_dir / f"test_data_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in version_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(version_dir)
                    zipf.write(file_path, arcname)

        # Update catalog with archive information
        if version in self.data_catalog['versions']:
            self.data_catalog['versions'][version]['archived'] = True
            self.data_catalog['versions'][version]['archive_file'] = str(archive_file)
            self.data_catalog['versions'][version]['archive_date'] = datetime.now().isoformat()

        self._save_data_catalog()

        logger.info(f"Version {version} archived to {archive_file}")
        return str(archive_file)

    def cleanup_old_versions(self, keep_versions: int = 3) -> List[str]:
        """Clean up old test data versions"""
        logger.info(f"Cleaning up old versions, keeping {keep_versions} most recent")

        # Get version list sorted by creation date
        versions = []
        for version, info in self.data_catalog['versions'].items():
            if isinstance(info, dict) and 'creation_date' in info:
                creation_date = datetime.fromisoformat(info['creation_date'])
                versions.append((version, creation_date))

        versions.sort(key=lambda x: x[1], reverse=True)

        # Determine versions to remove
        versions_to_remove = versions[keep_versions:]
        removed_versions = []

        for version, _ in versions_to_remove:
            try:
                # Archive before removing if not already archived
                if not self.data_catalog['versions'][version].get('archived', False):
                    self.archive_test_data_version(version)

                # Remove version directory
                version_dir = self.base_dir / f"version_{version}"
                if version_dir.exists():
                    shutil.rmtree(version_dir)

                # Mark as removed in catalog
                self.data_catalog['versions'][version]['removed'] = True
                self.data_catalog['versions'][version]['removal_date'] = datetime.now().isoformat()

                removed_versions.append(version)
                logger.info(f"Removed version {version}")

            except Exception as e:
                logger.error(f"Failed to remove version {version}: {e}")

        self._save_data_catalog()

        logger.info(f"Cleanup completed. Removed {len(removed_versions)} versions")
        return removed_versions

    def get_data_catalog(self) -> Dict[str, Any]:
        """Get the complete data catalog"""
        return self.data_catalog.copy()

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version"""
        return self.data_catalog['versions'].get(version)

    def list_available_versions(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """List available test data versions"""
        versions = []

        for version, info in self.data_catalog['versions'].items():
            if isinstance(info, dict):
                # Skip removed versions
                if info.get('removed', False):
                    continue

                # Skip archived versions if not requested
                if info.get('archived', False) and not include_archived:
                    continue

                version_info = info.copy()
                version_info['version'] = version
                versions.append(version_info)

        # Sort by creation date (newest first)
        versions.sort(
            key=lambda x: datetime.fromisoformat(x.get('creation_date', '1970-01-01')),
            reverse=True
        )

        return versions

    def get_test_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive test data statistics"""
        stats = {
            'total_versions': len(self.data_catalog['versions']),
            'active_versions': len([v for v in self.data_catalog['versions'].values()
                                  if isinstance(v, dict) and not v.get('removed', False)]),
            'archived_versions': len([v for v in self.data_catalog['versions'].values()
                                    if isinstance(v, dict) and v.get('archived', False)]),
            'current_version': self._get_current_version(),
            'total_storage_mb': 0,
            'quality_distribution': {'PASSED': 0, 'FAILED': 0},
            'generation_timeline': []
        }

        for version, info in self.data_catalog['versions'].items():
            if isinstance(info, dict) and not info.get('removed', False):
                # Storage statistics
                stats['total_storage_mb'] += info.get('total_size_mb', 0)

                # Quality statistics
                validation_status = info.get('validation_status', 'UNKNOWN')
                if validation_status in stats['quality_distribution']:
                    stats['quality_distribution'][validation_status] += 1

                # Timeline
                if 'creation_date' in info:
                    stats['generation_timeline'].append({
                        'version': version,
                        'date': info['creation_date'],
                        'quality_score': info.get('data_quality_score', 0),
                        'size_mb': info.get('total_size_mb', 0)
                    })

        # Sort timeline by date
        stats['generation_timeline'].sort(key=lambda x: x['date'])

        return stats

    def validate_existing_version(self, version: str) -> Dict[str, ValidationResult]:
        """Re-validate an existing version"""
        logger.info(f"Re-validating test data version {version}")

        version_dir = self.base_dir / f"version_{version}"
        if not version_dir.exists():
            raise ValueError(f"Version {version} not found")

        # Run validation
        validator = TestDataValidator(str(version_dir))
        validation_results = validator.validate_all_data()

        # Update catalog with new validation results
        validation_status = "PASSED" if all(r.passed for r in validation_results.values()) else "FAILED"
        avg_score = sum(r.score for r in validation_results.values()) / len(validation_results)

        if version in self.data_catalog['versions']:
            self.data_catalog['versions'][version]['validation_status'] = validation_status
            self.data_catalog['versions'][version]['data_quality_score'] = avg_score
            self.data_catalog['versions'][version]['last_validation'] = datetime.now().isoformat()

        self._save_data_catalog()

        logger.info(f"Re-validation completed for version {version}: {validation_status}")
        return validation_results

    def create_custom_test_dataset(self, config: TestDataConfig, dataset_name: str) -> str:
        """Create a custom test dataset with specific configuration"""
        logger.info(f"Creating custom test dataset: {dataset_name}")

        # Create custom dataset directory
        custom_dir = self.base_dir / "custom_datasets" / dataset_name
        custom_dir.mkdir(parents=True, exist_ok=True)

        # Generate custom data
        generator = TestDataGenerator(config, str(custom_dir))
        generated_files = generator.generate_all_test_data()

        # Validate custom data
        validator = TestDataValidator(str(custom_dir))
        validation_results = validator.validate_all_test_data()

        # Create custom dataset metadata
        custom_metadata = {
            'dataset_name': dataset_name,
            'creation_date': datetime.now().isoformat(),
            'generator_config': asdict(config),
            'generated_files': generated_files,
            'validation_results': {
                name: {
                    'passed': result.passed,
                    'score': result.score,
                    'issues_count': len(result.issues)
                }
                for name, result in validation_results.items()
            }
        }

        metadata_file = custom_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(custom_metadata, f, indent=2, default=str)

        logger.info(f"Custom dataset {dataset_name} created successfully")
        return str(custom_dir)

    def _load_data_catalog(self) -> Dict[str, Any]:
        """Load the data catalog"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load data catalog: {e}")

        # Return default catalog structure
        return {
            'catalog_version': '1.0.0',
            'created': datetime.now().isoformat(),
            'current_version': None,
            'versions': {},
            'metadata': {
                'total_versions_created': 0,
                'last_cleanup': None,
                'last_generation': None
            }
        }

    def _save_data_catalog(self):
        """Save the data catalog"""
        try:
            with open(self.catalog_file, 'w') as f:
                json.dump(self.data_catalog, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save data catalog: {e}")

    def _update_data_catalog(self, version_record: TestDataVersion,
                           generated_files: Dict[str, str],
                           validation_results: Dict[str, ValidationResult]):
        """Update the data catalog with new version"""

        version_entry = {
            'version': version_record.version,
            'creation_date': version_record.creation_date.isoformat(),
            'generator_config': version_record.generator_config,
            'validation_status': version_record.validation_status,
            'data_quality_score': version_record.data_quality_score,
            'file_count': version_record.file_count,
            'total_size_mb': version_record.total_size_mb,
            'description': version_record.description,
            'generated_files': generated_files,
            'validation_summary': {
                name: {
                    'passed': result.passed,
                    'score': result.score,
                    'issues_count': len(result.issues),
                    'warnings_count': len(result.warnings)
                }
                for name, result in validation_results.items()
            },
            'archived': False,
            'removed': False
        }

        self.data_catalog['versions'][version_record.version] = version_entry
        self.data_catalog['metadata']['total_versions_created'] += 1
        self.data_catalog['metadata']['last_generation'] = datetime.now().isoformat()

        self._save_data_catalog()

    def _create_version_identifier(self) -> str:
        """Create a unique version identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_number = self.data_catalog['metadata']['total_versions_created'] + 1
        return f"{version_number:03d}_{timestamp}"

    def _set_current_version(self, version: str):
        """Set the current active version"""
        self.data_catalog['current_version'] = version

        # Copy to current directory
        version_dir = self.base_dir / f"version_{version}"
        if version_dir.exists():
            # Remove existing current data
            if self.current_version_dir.exists():
                shutil.rmtree(self.current_version_dir)

            # Copy new current data
            shutil.copytree(version_dir, self.current_version_dir)

        self._save_data_catalog()

    def _get_current_version(self) -> Optional[str]:
        """Get the current active version"""
        return self.data_catalog.get('current_version')

    def _find_data_file(self, version_dir: Path, data_type: str) -> Optional[str]:
        """Find data file for specific data type in version directory"""
        # Mapping of data types to file names
        file_mapping = {
            'sensor_metadata': 'sensor_metadata.json',
            'sensor_data': 'sensor_time_series.json',
            'anomaly_data': 'anomaly_scenarios.json',
            'business_data': 'business_scenarios.json',
            'performance_data': 'performance_test_data.json',
            'failure_data': 'failure_scenarios.json',
            'nasa_data': 'nasa_like_datasets.json',
            'quality_data': 'data_quality_test_sets.json'
        }

        filename = file_mapping.get(data_type)
        if filename:
            file_path = version_dir / filename
            if file_path.exists():
                return str(file_path)

        return None

    def _apply_custom_transformations(self, provisioned_files: Dict[str, str],
                                    custom_parameters: Dict[str, Any]) -> Dict[str, str]:
        """Apply custom transformations to provisioned data"""
        logger.info("Applying custom transformations to provisioned data")

        # Example transformations based on parameters
        if 'filter_sensors' in custom_parameters:
            sensor_filter = custom_parameters['filter_sensors']
            self._filter_sensors(provisioned_files, sensor_filter)

        if 'data_subset' in custom_parameters:
            subset_config = custom_parameters['data_subset']
            self._create_data_subset(provisioned_files, subset_config)

        if 'quality_degradation' in custom_parameters:
            degradation_config = custom_parameters['quality_degradation']
            self._apply_quality_degradation(provisioned_files, degradation_config)

        return provisioned_files

    def _filter_sensors(self, files: Dict[str, str], sensor_filter: Dict[str, Any]):
        """Filter sensors based on criteria"""
        # Implementation would filter sensor data based on criteria
        logger.info(f"Applying sensor filter: {sensor_filter}")

    def _create_data_subset(self, files: Dict[str, str], subset_config: Dict[str, Any]):
        """Create data subset based on configuration"""
        # Implementation would create subset of data
        logger.info(f"Creating data subset: {subset_config}")

    def _apply_quality_degradation(self, files: Dict[str, str], degradation_config: Dict[str, Any]):
        """Apply quality degradation for testing robustness"""
        # Implementation would deliberately degrade data quality
        logger.info(f"Applying quality degradation: {degradation_config}")

    def _save_version_metadata(self, version_dir: Path, version_record: TestDataVersion,
                             generated_files: Dict[str, str],
                             validation_results: Dict[str, ValidationResult]):
        """Save comprehensive version metadata"""

        metadata = {
            'version_info': asdict(version_record),
            'generated_files': generated_files,
            'validation_results': {
                name: {
                    'validator_name': result.validator_name,
                    'passed': result.passed,
                    'score': result.score,
                    'issues': result.issues,
                    'warnings': result.warnings,
                    'statistics': result.statistics,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in validation_results.items()
            },
            'file_checksums': self._calculate_file_checksums(generated_files)
        }

        metadata_file = version_dir / "version_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _calculate_file_checksums(self, files: Dict[str, str]) -> Dict[str, str]:
        """Calculate checksums for generated files"""
        checksums = {}

        for data_type, file_path in files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        checksum = hashlib.md5(content).hexdigest()
                        checksums[data_type] = checksum
                except Exception as e:
                    logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
                    checksums[data_type] = "ERROR"

        return checksums


# CLI functions for easy usage
def generate_test_data():
    """CLI function to generate test data"""
    manager = TestDataManager()

    # Use default configuration
    config = TestDataConfig(
        smap_sensors=25,
        msl_sensors=55,
        data_points_per_sensor=1000,
        anomaly_injection_rate=0.05,
        data_quality_level='high'
    )

    version = manager.generate_and_validate_test_data(config)
    print(f"Generated test data version: {version}")

    # Print statistics
    stats = manager.get_test_data_statistics()
    print(f"Total versions: {stats['total_versions']}")
    print(f"Current version: {stats['current_version']}")
    print(f"Total storage: {stats['total_storage_mb']:.1f} MB")


def provision_for_test_suite(test_suite: str, data_types: List[str]):
    """CLI function to provision test data for a test suite"""
    manager = TestDataManager()

    request = TestDataRequest(
        request_id=f"{test_suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        test_suite=test_suite,
        data_types=data_types,
        quality_level='high',
        volume_requirements={},
        custom_parameters={},
        deadline=datetime.now() + timedelta(hours=1)
    )

    provision_dir = manager.provision_test_data(request)
    print(f"Test data provisioned for {test_suite} at: {provision_dir}")


def cleanup_old_data():
    """CLI function to cleanup old test data"""
    manager = TestDataManager()
    removed = manager.cleanup_old_versions(keep_versions=3)
    print(f"Cleaned up {len(removed)} old versions")


def show_catalog():
    """CLI function to show data catalog"""
    manager = TestDataManager()
    catalog = manager.get_data_catalog()

    print("\n" + "="*60)
    print("TEST DATA CATALOG")
    print("="*60)

    versions = manager.list_available_versions(include_archived=True)
    for version_info in versions:
        status = "✓" if version_info.get('validation_status') == 'PASSED' else "✗"
        archived = " [ARCHIVED]" if version_info.get('archived') else ""
        current = " [CURRENT]" if version_info['version'] == catalog.get('current_version') else ""

        print(f"{status} {version_info['version']}{current}{archived}")
        print(f"   Created: {version_info.get('creation_date', 'Unknown')}")
        print(f"   Quality Score: {version_info.get('data_quality_score', 0):.3f}")
        print(f"   Size: {version_info.get('total_size_mb', 0):.1f} MB")
        print()

    stats = manager.get_test_data_statistics()
    print(f"Total versions: {stats['total_versions']}")
    print(f"Storage used: {stats['total_storage_mb']:.1f} MB")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_data_manager.py <command>")
        print("Commands: generate, provision, cleanup, catalog")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        generate_test_data()
    elif command == "provision":
        if len(sys.argv) < 4:
            print("Usage: python test_data_manager.py provision <test_suite> <data_type1,data_type2,...>")
            sys.exit(1)
        test_suite = sys.argv[2]
        data_types = sys.argv[3].split(',')
        provision_for_test_suite(test_suite, data_types)
    elif command == "cleanup":
        cleanup_old_data()
    elif command == "catalog":
        show_catalog()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)