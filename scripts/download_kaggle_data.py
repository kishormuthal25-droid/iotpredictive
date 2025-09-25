#!/usr/bin/env python3
"""
Kaggle NASA Dataset Download Script
Downloads the NASA SMAP/MSL anomaly detection dataset from Kaggle
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import json

def setup_kaggle_api():
    """Setup and verify Kaggle API"""
    try:
        import kaggle
        print("✓ Kaggle API found")
        return True
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle
        return True
    except Exception as e:
        print(f"Error setting up Kaggle API: {e}")
        return False

def download_dataset():
    """Download the NASA dataset from Kaggle"""
    dataset_name = "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"

    print(f"Downloading dataset: {dataset_name}")
    print("Note: You need Kaggle API credentials configured")
    print("Visit: https://www.kaggle.com/settings/account -> Create New API Token")

    try:
        # Create data directories
        os.makedirs("data/raw", exist_ok=True)

        # Download using kaggle API
        import kaggle
        kaggle.api.dataset_download_files(
            dataset_name,
            path="data/raw",
            unzip=True
        )

        print("✓ Dataset downloaded successfully")
        return True

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Manual download instructions:")
        print(f"1. Go to: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. Click 'Download' button")
        print("3. Extract the ZIP file to 'data/raw/' directory")
        return False

def organize_data():
    """Organize downloaded data into expected structure"""
    raw_dir = Path("data/raw")

    # Look for the downloaded files
    files = list(raw_dir.glob("*"))
    print(f"Found files: {[f.name for f in files]}")

    # Create organized structure
    smap_dir = raw_dir / "smap"
    msl_dir = raw_dir / "msl"
    smap_dir.mkdir(exist_ok=True)
    msl_dir.mkdir(exist_ok=True)

    # Move files to appropriate directories based on naming
    for file in files:
        if file.is_file():
            filename = file.name.lower()
            if 'smap' in filename:
                dest = smap_dir / file.name
                if not dest.exists():
                    file.rename(dest)
                    print(f"Moved {file.name} to smap/")
            elif 'msl' in filename:
                dest = msl_dir / file.name
                if not dest.exists():
                    file.rename(dest)
                    print(f"Moved {file.name} to msl/")

    print("✓ Data organized into smap/ and msl/ directories")

    # Generate data statistics
    generate_data_stats()

def generate_data_stats():
    """Generate statistics about the downloaded data"""
    stats = {
        'smap': {},
        'msl': {},
        'summary': {}
    }

    smap_dir = Path("data/raw/smap")
    msl_dir = Path("data/raw/msl")

    # Count files
    smap_files = list(smap_dir.glob("*"))
    msl_files = list(msl_dir.glob("*"))

    stats['summary'] = {
        'total_files': len(smap_files) + len(msl_files),
        'smap_files': len(smap_files),
        'msl_files': len(msl_files),
        'smap_file_names': [f.name for f in smap_files],
        'msl_file_names': [f.name for f in msl_files]
    }

    # Save statistics
    with open('data/data_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("✓ Data statistics saved to data/data_statistics.json")

def main():
    print("="*50)
    print("NASA SMAP/MSL Dataset Downloader")
    print("="*50)

    # Setup Kaggle API
    if not setup_kaggle_api():
        print("Failed to setup Kaggle API")
        return False

    # Download dataset
    if download_dataset():
        organize_data()
        print("\n✓ Dataset download and organization completed!")
        print("Data location: data/raw/")
        return True
    else:
        print("\nDataset download failed. Please download manually.")
        return False

if __name__ == "__main__":
    main()