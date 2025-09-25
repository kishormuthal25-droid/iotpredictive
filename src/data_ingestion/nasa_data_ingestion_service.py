"""
NASA Data Ingestion Service
Ingests core NASA telemetry data from data/raw/data/data/ into the database
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import time

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_data_path
from src.data_ingestion.database_manager import DatabaseManager, TelemetryData
from src.data_ingestion.equipment_mapper import equipment_mapper

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class IngestionStats:
    """Statistics for data ingestion process"""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    total_records: int = 0
    total_anomalies: int = 0
    processing_time: float = 0.0
    start_time: datetime = None
    end_time: datetime = None


class NASADataIngestionService:
    """
    Service for ingesting NASA telemetry data from core dataset into database
    Handles SMAP/MSL data from data/raw/data/data/ directory structure
    """

    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """Initialize NASA data ingestion service"""
        self.db_manager = database_manager or DatabaseManager()
        self.core_data_path = Path("data/raw/data/data")
        self.labeled_anomalies_path = Path("data/raw/labeled_anomalies.csv")

        # Load anomaly labels
        self.anomaly_labels = self._load_anomaly_labels()

        # Thread safety
        self._lock = threading.Lock()
        self.stats = IngestionStats()

        logger.info("NASA Data Ingestion Service initialized")

    def _load_anomaly_labels(self) -> Dict[str, List[Dict[str, int]]]:
        """Load labeled anomalies from CSV file"""
        try:
            if not self.labeled_anomalies_path.exists():
                logger.warning(f"Labeled anomalies file not found: {self.labeled_anomalies_path}")
                return {}

            df = pd.read_csv(self.labeled_anomalies_path)
            anomaly_labels = {}

            for _, row in df.iterrows():
                spacecraft = row['spacecraft']
                if spacecraft not in anomaly_labels:
                    anomaly_labels[spacecraft] = []

                anomaly_labels[spacecraft].append({
                    'anomaly_sequences': row['anomaly_sequences'],
                    'start_index': row['start_index'],
                    'end_index': row['end_index']
                })

            logger.info(f"Loaded {len(df)} anomaly labels from CSV")
            return anomaly_labels

        except Exception as e:
            logger.error(f"Failed to load anomaly labels: {e}")
            return {}

    def _determine_spacecraft_and_channel(self, file_path: Path) -> Tuple[str, str]:
        """Determine spacecraft and channel from filename

        Args:
            file_path: Path to .npy file (e.g., A-1.npy, D-14.npy)

        Returns:
            Tuple of (spacecraft, channel)
        """
        filename = file_path.stem  # Remove .npy extension

        # Extract channel from filename (e.g., A-1 -> A-1, D-14 -> D-14)
        channel = filename

        # Determine spacecraft based on channel patterns
        # SMAP channels: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T
        # MSL channels: Different pattern, but we'll use channel prefix
        channel_prefix = filename.split('-')[0] if '-' in filename else filename[0]

        # For simplicity, map based on alphabetical ranges or use equipment mapper
        equipment_info = equipment_mapper.get_equipment_by_channel(channel)
        if equipment_info:
            spacecraft = "SMAP" if "SMAP" in equipment_info.get('subsystem', '') else "MSL"
        else:
            # Fallback mapping based on channel prefix
            smap_channels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
            spacecraft = "SMAP" if channel_prefix in smap_channels else "MSL"

        return spacecraft, channel

    def _process_signal_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """Process a single NASA signal file

        Args:
            file_path: Path to .npy file
            data_type: 'train', 'test', or timestamp folder name

        Returns:
            Dictionary with processing results
        """
        try:
            # Load numpy data
            signal_data = np.load(file_path)

            # Determine spacecraft and channel
            spacecraft, channel = self._determine_spacecraft_and_channel(file_path)

            # Get equipment info for metadata
            equipment_info = equipment_mapper.get_equipment_info(f"{spacecraft}_{channel}")

            # Calculate statistics
            mean_value = float(np.mean(signal_data))
            std_value = float(np.std(signal_data))
            min_value = float(np.min(signal_data))
            max_value = float(np.max(signal_data))

            # Check for anomalies in this signal
            anomaly_indices = set()
            if spacecraft in self.anomaly_labels:
                for anomaly in self.anomaly_labels[spacecraft]:
                    # Check if this channel/sequence has anomalies
                    # Note: The anomaly_sequences might need mapping to channel
                    anomaly_indices.update(range(
                        anomaly['start_index'],
                        anomaly['end_index'] + 1
                    ))

            # Create timestamp series (simulate real timestamps)
            base_time = datetime(2018, 5, 19, 15, 0, 10)
            time_delta = timedelta(seconds=1)  # 1 second intervals
            timestamps = [base_time + i * time_delta for i in range(len(signal_data))]

            # Prepare telemetry records for batch insertion
            telemetry_records = []

            # Insert data in chunks to avoid memory issues
            chunk_size = 1000
            total_chunks = (len(signal_data) + chunk_size - 1) // chunk_size

            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(signal_data))

                # Get chunk data
                chunk_data = signal_data[start_idx:end_idx].tolist()
                chunk_timestamps = timestamps[start_idx:end_idx]

                # Create telemetry record for this chunk
                telemetry_record = {
                    'timestamp': chunk_timestamps[0],  # Use first timestamp of chunk
                    'spacecraft': spacecraft,
                    'channel': channel,
                    'sequence_id': chunk_idx,
                    'data': chunk_data,
                    'data_shape': f"({len(chunk_data)},)",
                    'mean_value': float(np.mean(chunk_data)),
                    'std_value': float(np.std(chunk_data)),
                    'min_value': float(np.min(chunk_data)),
                    'max_value': float(np.max(chunk_data)),
                    'is_anomaly': any(i in anomaly_indices for i in range(start_idx, end_idx)),
                    'anomaly_score': 0.0,  # Will be calculated by ML models later
                    'event_metadata': {
                        'file_path': str(file_path),
                        'data_type': data_type,
                        'chunk_index': chunk_idx,
                        'total_chunks': total_chunks,
                        'equipment_info': equipment_info,
                        'original_shape': signal_data.shape,
                        'anomaly_count': len([i for i in range(start_idx, end_idx) if i in anomaly_indices])
                    }
                }

                telemetry_records.append(telemetry_record)

            # Insert batch into database
            self.db_manager.insert_telemetry_batch(telemetry_records)

            # Update statistics
            with self._lock:
                self.stats.total_records += len(telemetry_records)
                self.stats.total_anomalies += len(anomaly_indices)
                self.stats.processed_files += 1

            logger.info(f"Processed {file_path.name}: {len(signal_data)} points, {len(telemetry_records)} chunks, {len(anomaly_indices)} anomalies")

            return {
                'status': 'success',
                'file': str(file_path),
                'spacecraft': spacecraft,
                'channel': channel,
                'records_inserted': len(telemetry_records),
                'data_points': len(signal_data),
                'anomalies': len(anomaly_indices)
            }

        except Exception as e:
            with self._lock:
                self.stats.failed_files += 1
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                'status': 'failed',
                'file': str(file_path),
                'error': str(e)
            }

    def ingest_dataset(self, data_type: str = "all", max_workers: int = 4,
                      max_files: Optional[int] = None) -> IngestionStats:
        """Ingest NASA dataset into database

        Args:
            data_type: 'train', 'test', 'all', or specific timestamp folder
            max_workers: Number of parallel workers
            max_files: Maximum number of files to process (for testing)

        Returns:
            Ingestion statistics
        """
        self.stats = IngestionStats()
        self.stats.start_time = datetime.now()

        try:
            # Determine which directories to process
            if data_type == "all":
                target_dirs = [
                    self.core_data_path / "train",
                    self.core_data_path / "test",
                    self.core_data_path / "2018-05-19_15.00.10"
                ]
            else:
                target_dirs = [self.core_data_path / data_type]

            # Collect all .npy files
            all_files = []
            for target_dir in target_dirs:
                if target_dir.exists():
                    npy_files = list(target_dir.glob("*.npy"))
                    all_files.extend([(f, target_dir.name) for f in npy_files])
                    logger.info(f"Found {len(npy_files)} files in {target_dir}")
                else:
                    logger.warning(f"Directory not found: {target_dir}")

            # Apply max_files limit if specified
            if max_files:
                all_files = all_files[:max_files]
                logger.info(f"Limited to {len(all_files)} files for processing")

            self.stats.total_files = len(all_files)

            if not all_files:
                logger.warning("No .npy files found to process")
                return self.stats

            logger.info(f"Starting ingestion of {len(all_files)} files with {max_workers} workers")

            # Process files in parallel
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_signal_file, file_path, dir_name): (file_path, dir_name)
                    for file_path, dir_name in all_files
                }

                # Process completed tasks
                for future in as_completed(future_to_file):
                    file_path, dir_name = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Log progress
                        if len(results) % 10 == 0 or len(results) == len(all_files):
                            logger.info(f"Progress: {len(results)}/{len(all_files)} files processed")

                    except Exception as e:
                        logger.error(f"Task failed for {file_path}: {e}")
                        with self._lock:
                            self.stats.failed_files += 1

            # Finalize statistics
            self.stats.end_time = datetime.now()
            self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()

            # Log summary
            successful_results = [r for r in results if r.get('status') == 'success']
            failed_results = [r for r in results if r.get('status') == 'failed']

            logger.info(f"""
            Ingestion completed:
            - Total files: {self.stats.total_files}
            - Processed successfully: {len(successful_results)}
            - Failed: {len(failed_results)}
            - Total records inserted: {self.stats.total_records}
            - Total anomalies: {self.stats.total_anomalies}
            - Processing time: {self.stats.processing_time:.2f} seconds
            """)

            if failed_results:
                logger.warning("Failed files:")
                for result in failed_results[:10]:  # Show first 10 failures
                    logger.warning(f"  {result.get('file')}: {result.get('error')}")

            return self.stats

        except Exception as e:
            logger.error(f"Ingestion process failed: {e}")
            self.stats.end_time = datetime.now()
            self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            raise

    def verify_ingestion(self) -> Dict[str, Any]:
        """Verify the ingested data in database"""
        try:
            with self.db_manager.get_session() as session:
                # Count total records
                total_records = session.query(TelemetryData).count()

                # Count by spacecraft
                smap_records = session.query(TelemetryData).filter(
                    TelemetryData.spacecraft == 'SMAP'
                ).count()

                msl_records = session.query(TelemetryData).filter(
                    TelemetryData.spacecraft == 'MSL'
                ).count()

                # Count anomalies
                anomaly_records = session.query(TelemetryData).filter(
                    TelemetryData.is_anomaly == True
                ).count()

                # Get unique channels
                unique_channels = session.query(TelemetryData.channel).distinct().count()

                # Get date range
                date_range = session.query(
                    func.min(TelemetryData.timestamp),
                    func.max(TelemetryData.timestamp)
                ).first()

                verification_results = {
                    'total_records': total_records,
                    'smap_records': smap_records,
                    'msl_records': msl_records,
                    'anomaly_records': anomaly_records,
                    'anomaly_percentage': (anomaly_records / total_records * 100) if total_records > 0 else 0,
                    'unique_channels': unique_channels,
                    'date_range': {
                        'start': date_range[0].isoformat() if date_range[0] else None,
                        'end': date_range[1].isoformat() if date_range[1] else None
                    },
                    'verification_time': datetime.now().isoformat()
                }

                logger.info(f"Database verification completed: {verification_results}")
                return verification_results

        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return {'error': str(e)}

    def clear_existing_data(self, confirm: bool = False) -> bool:
        """Clear existing telemetry data (use with caution)

        Args:
            confirm: Must be True to actually clear data

        Returns:
            True if data was cleared, False otherwise
        """
        if not confirm:
            logger.warning("Data clear operation requires explicit confirmation")
            return False

        try:
            with self.db_manager.get_session() as session:
                # Count existing records
                existing_count = session.query(TelemetryData).count()

                if existing_count == 0:
                    logger.info("No existing data to clear")
                    return True

                # Clear telemetry data
                session.query(TelemetryData).delete()
                session.commit()

                logger.info(f"Cleared {existing_count} existing telemetry records")
                return True

        except Exception as e:
            logger.error(f"Failed to clear existing data: {e}")
            return False


# Global instance for easy access
nasa_data_ingestion_service = NASADataIngestionService()


def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="NASA Data Ingestion Service")
    parser.add_argument('--data-type', choices=['train', 'test', 'all'], default='all',
                       help='Type of data to ingest')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--clear-existing', action='store_true',
                       help='Clear existing data before ingesting')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing data without ingesting')

    args = parser.parse_args()

    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    service = NASADataIngestionService()

    try:
        if args.verify_only:
            # Only verify existing data
            results = service.verify_ingestion()
            print(json.dumps(results, indent=2))
            return

        # Clear existing data if requested
        if args.clear_existing:
            if service.clear_existing_data(confirm=True):
                print("Existing data cleared successfully")
            else:
                print("Failed to clear existing data")
                return

        # Run ingestion
        print(f"Starting NASA data ingestion...")
        stats = service.ingest_dataset(
            data_type=args.data_type,
            max_workers=args.max_workers,
            max_files=args.max_files
        )

        print(f"\nIngestion completed:")
        print(f"  Files processed: {stats.processed_files}/{stats.total_files}")
        print(f"  Records inserted: {stats.total_records}")
        print(f"  Anomalies found: {stats.total_anomalies}")
        print(f"  Processing time: {stats.processing_time:.2f} seconds")

        # Verify ingestion
        print("\nVerifying ingestion...")
        verification = service.verify_ingestion()
        print(json.dumps(verification, indent=2))

    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        raise


if __name__ == "__main__":
    main()