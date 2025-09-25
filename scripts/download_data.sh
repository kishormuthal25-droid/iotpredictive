#!/bin/bash

#############################################################################
# IoT Anomaly Detection System - Data Download Script
# Downloads SMAP and MSL datasets from NASA for anomaly detection
#############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/data"
RAW_DATA_DIR="${DATA_DIR}/raw"
SMAP_DIR="${RAW_DATA_DIR}/smap"
MSL_DIR="${RAW_DATA_DIR}/msl"
PROCESSED_DIR="${DATA_DIR}/processed"
MODELS_DIR="${DATA_DIR}/models"
TEMP_DIR="${DATA_DIR}/temp"

# NASA Dataset URLs (replace with actual URLs when available)
# These are example URLs - you'll need to replace with actual NASA dataset locations
SMAP_TRAIN_URL="https://s3-us-west-2.amazonaws.com/telemanom/data/train/smap_train.tar.gz"
SMAP_TEST_URL="https://s3-us-west-2.amazonaws.com/telemanom/data/test/smap_test.tar.gz"
MSL_TRAIN_URL="https://s3-us-west-2.amazonaws.com/telemanom/data/train/msl_train.tar.gz"
MSL_TEST_URL="https://s3-us-west-2.amazonaws.com/telemanom/data/test/msl_test.tar.gz"
LABELS_URL="https://s3-us-west-2.amazonaws.com/telemanom/data/labeled_anomalies.csv"

# Alternative data sources (if NASA URLs change)
GITHUB_REPO="https://github.com/khundman/telemanom"
KAGGLE_DATASET="nasa/smap-msl-anomaly-detection"

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print banner
print_banner() {
    echo ""
    print_message "$BLUE" "============================================"
    print_message "$BLUE" "   IoT Anomaly Detection Data Downloader   "
    print_message "$BLUE" "============================================"
    echo ""
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_message "$YELLOW" "Checking system requirements..."
    
    local missing_tools=()
    
    # Check for required tools
    for tool in curl wget tar gzip python3 pip3; do
        if ! command_exists "$tool"; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_message "$RED" "Error: Missing required tools: ${missing_tools[*]}"
        print_message "$YELLOW" "Please install the missing tools and try again."
        
        # Provide installation suggestions based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            print_message "$YELLOW" "On Ubuntu/Debian: sudo apt-get install ${missing_tools[*]}"
            print_message "$YELLOW" "On CentOS/RHEL: sudo yum install ${missing_tools[*]}"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            print_message "$YELLOW" "On macOS: brew install ${missing_tools[*]}"
        fi
        
        exit 1
    fi
    
    print_message "$GREEN" "✓ All requirements satisfied"
}

# Function to create directory structure
create_directories() {
    print_message "$YELLOW" "Creating directory structure..."
    
    # Create all necessary directories
    mkdir -p "$SMAP_DIR"
    mkdir -p "$MSL_DIR"
    mkdir -p "$PROCESSED_DIR"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$TEMP_DIR"
    mkdir -p "${DATA_DIR}/logs"
    mkdir -p "${DATA_DIR}/exports"
    mkdir -p "${DATA_DIR}/reports"
    
    print_message "$GREEN" "✓ Directory structure created"
}

# Function to download file with progress
download_file() {
    local url=$1
    local output_file=$2
    local description=$3
    
    print_message "$YELLOW" "Downloading ${description}..."
    
    # Check if file already exists
    if [ -f "$output_file" ]; then
        print_message "$YELLOW" "File already exists. Checking integrity..."
        
        # Verify file is not corrupted (basic check)
        if [ -s "$output_file" ]; then
            print_message "$GREEN" "✓ File appears valid, skipping download"
            return 0
        else
            print_message "$YELLOW" "File appears corrupted, re-downloading..."
            rm -f "$output_file"
        fi
    fi
    
    # Try wget first, then curl
    if command_exists wget; then
        wget --progress=bar:force \
             --tries=3 \
             --timeout=30 \
             --continue \
             -O "$output_file" \
             "$url" 2>&1 | \
        grep --line-buffered "%" | \
        sed -u -e "s,\.,,g" | \
        awk '{printf("\rDownloading: %s", $2)}'
        echo ""
    elif command_exists curl; then
        curl -L \
             --retry 3 \
             --retry-delay 5 \
             --max-time 300 \
             --progress-bar \
             -o "$output_file" \
             "$url"
    else
        print_message "$RED" "Error: Neither wget nor curl is available"
        return 1
    fi
    
    # Verify download
    if [ ! -f "$output_file" ] || [ ! -s "$output_file" ]; then
        print_message "$RED" "Error: Download failed for ${description}"
        return 1
    fi
    
    print_message "$GREEN" "✓ Downloaded ${description}"
    return 0
}

# Function to extract tar.gz files
extract_archive() {
    local archive_file=$1
    local output_dir=$2
    local description=$3
    
    print_message "$YELLOW" "Extracting ${description}..."
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Extract based on file extension
    if [[ "$archive_file" == *.tar.gz ]] || [[ "$archive_file" == *.tgz ]]; then
        tar -xzf "$archive_file" -C "$output_dir"
    elif [[ "$archive_file" == *.zip ]]; then
        unzip -q "$archive_file" -d "$output_dir"
    else
        print_message "$RED" "Error: Unknown archive format for ${archive_file}"
        return 1
    fi
    
    print_message "$GREEN" "✓ Extracted ${description}"
    return 0
}

# Function to download SMAP dataset
download_smap_data() {
    print_message "$BLUE" "\n=== Downloading SMAP Dataset ==="
    
    # Download training data
    local smap_train_file="${TEMP_DIR}/smap_train.tar.gz"
    if download_file "$SMAP_TRAIN_URL" "$smap_train_file" "SMAP training data"; then
        extract_archive "$smap_train_file" "$SMAP_DIR" "SMAP training data"
    else
        print_message "$RED" "Failed to download SMAP training data"
        return 1
    fi
    
    # Download test data
    local smap_test_file="${TEMP_DIR}/smap_test.tar.gz"
    if download_file "$SMAP_TEST_URL" "$smap_test_file" "SMAP test data"; then
        extract_archive "$smap_test_file" "$SMAP_DIR" "SMAP test data"
    else
        print_message "$RED" "Failed to download SMAP test data"
        return 1
    fi
    
    print_message "$GREEN" "✓ SMAP dataset downloaded successfully"
    return 0
}

# Function to download MSL dataset
download_msl_data() {
    print_message "$BLUE" "\n=== Downloading MSL Dataset ==="
    
    # Download training data
    local msl_train_file="${TEMP_DIR}/msl_train.tar.gz"
    if download_file "$MSL_TRAIN_URL" "$msl_train_file" "MSL training data"; then
        extract_archive "$msl_train_file" "$MSL_DIR" "MSL training data"
    else
        print_message "$RED" "Failed to download MSL training data"
        return 1
    fi
    
    # Download test data
    local msl_test_file="${TEMP_DIR}/msl_test.tar.gz"
    if download_file "$MSL_TEST_URL" "$msl_test_file" "MSL test data"; then
        extract_archive "$msl_test_file" "$MSL_DIR" "MSL test data"
    else
        print_message "$RED" "Failed to download MSL test data"
        return 1
    fi
    
    print_message "$GREEN" "✓ MSL dataset downloaded successfully"
    return 0
}

# Function to download labels
download_labels() {
    print_message "$BLUE" "\n=== Downloading Anomaly Labels ==="
    
    local labels_file="${RAW_DATA_DIR}/labeled_anomalies.csv"
    if download_file "$LABELS_URL" "$labels_file" "anomaly labels"; then
        print_message "$GREEN" "✓ Anomaly labels downloaded successfully"
        return 0
    else
        print_message "$RED" "Failed to download anomaly labels"
        return 1
    fi
}

# Function to download sample data for testing (smaller dataset)
download_sample_data() {
    print_message "$BLUE" "\n=== Creating Sample Dataset ==="
    
    # Create a Python script to generate sample data
    cat > "${TEMP_DIR}/generate_sample.py" << 'EOF'
import numpy as np
import pandas as pd
import os
import h5py

# Set seed for reproducibility
np.random.seed(42)

# Generate sample time series data
def generate_time_series(n_samples=1000, n_features=25, anomaly_rate=0.05):
    # Generate normal data
    time = np.linspace(0, 100, n_samples)
    data = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Create different patterns for different features
        if i % 3 == 0:
            data[:, i] = np.sin(time * (i+1) * 0.1) + np.random.randn(n_samples) * 0.1
        elif i % 3 == 1:
            data[:, i] = np.cos(time * (i+1) * 0.1) + np.random.randn(n_samples) * 0.1
        else:
            data[:, i] = np.random.randn(n_samples) * 0.5
    
    # Add anomalies
    labels = np.zeros(n_samples)
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Inject anomalies
        data[idx] += np.random.randn(n_features) * 3
        labels[idx] = 1
    
    return data, labels

# Create directories
os.makedirs('data/raw/smap', exist_ok=True)
os.makedirs('data/raw/msl', exist_ok=True)

# Generate SMAP sample data
print("Generating SMAP sample data...")
smap_train_data, smap_train_labels = generate_time_series(5000, 25, 0.03)
smap_test_data, smap_test_labels = generate_time_series(2000, 25, 0.05)

# Save as .npy files
np.save('data/raw/smap/train.npy', smap_train_data)
np.save('data/raw/smap/train_labels.npy', smap_train_labels)
np.save('data/raw/smap/test.npy', smap_test_data)
np.save('data/raw/smap/test_labels.npy', smap_test_labels)

# Generate MSL sample data
print("Generating MSL sample data...")
msl_train_data, msl_train_labels = generate_time_series(5000, 55, 0.04)
msl_test_data, msl_test_labels = generate_time_series(2000, 55, 0.06)

# Save as .h5 files
with h5py.File('data/raw/msl/train.h5', 'w') as f:
    f.create_dataset('data', data=msl_train_data)
    f.create_dataset('labels', data=msl_train_labels)

with h5py.File('data/raw/msl/test.h5', 'w') as f:
    f.create_dataset('data', data=msl_test_data)
    f.create_dataset('labels', data=msl_test_labels)

# Create labels CSV
print("Creating labels CSV...")
labels_df = pd.DataFrame({
    'spacecraft': ['SMAP'] * len(anomaly_indices) + ['MSL'] * len(anomaly_indices),
    'anomaly_sequences': list(anomaly_indices) + list(anomaly_indices),
    'start_index': list(anomaly_indices) + list(anomaly_indices),
    'end_index': [idx + 10 for idx in anomaly_indices] + [idx + 10 for idx in anomaly_indices]
})
labels_df.to_csv('data/raw/labeled_anomalies.csv', index=False)

print("Sample data generation completed!")
EOF
    
    # Run the Python script
    if command_exists python3; then
        cd "$BASE_DIR"
        python3 "${TEMP_DIR}/generate_sample.py"
        print_message "$GREEN" "✓ Sample dataset created successfully"
    else
        print_message "$RED" "Error: Python 3 is required to generate sample data"
        return 1
    fi
}

# Function to verify downloaded data
verify_data() {
    print_message "$BLUE" "\n=== Verifying Downloaded Data ==="
    
    local verification_passed=true
    
    # Check SMAP data
    if [ -d "$SMAP_DIR" ] && [ "$(ls -A $SMAP_DIR)" ]; then
        print_message "$GREEN" "✓ SMAP data verified"
    else
        print_message "$RED" "✗ SMAP data missing or empty"
        verification_passed=false
    fi
    
    # Check MSL data
    if [ -d "$MSL_DIR" ] && [ "$(ls -A $MSL_DIR)" ]; then
        print_message "$GREEN" "✓ MSL data verified"
    else
        print_message "$RED" "✗ MSL data missing or empty"
        verification_passed=false
    fi
    
    # Check labels file
    if [ -f "${RAW_DATA_DIR}/labeled_anomalies.csv" ]; then
        print_message "$GREEN" "✓ Labels file verified"
    else
        print_message "$RED" "✗ Labels file missing"
        verification_passed=false
    fi
    
    if [ "$verification_passed" = true ]; then
        print_message "$GREEN" "\n✓ All data verified successfully!"
        return 0
    else
        print_message "$RED" "\n✗ Data verification failed!"
        return 1
    fi
}

# Function to download from alternative sources
download_from_alternative() {
    print_message "$YELLOW" "\n=== Attempting Alternative Download Sources ==="
    
    # Try GitHub repository
    print_message "$YELLOW" "Trying GitHub repository..."
    local github_zip="${TEMP_DIR}/telemanom.zip"
    
    if download_file "${GITHUB_REPO}/archive/master.zip" "$github_zip" "GitHub repository"; then
        extract_archive "$github_zip" "$TEMP_DIR" "GitHub data"
        
        # Move data to appropriate directories
        if [ -d "${TEMP_DIR}/telemanom-master/data" ]; then
            cp -r "${TEMP_DIR}/telemanom-master/data/"* "$RAW_DATA_DIR/"
            print_message "$GREEN" "✓ Data copied from GitHub repository"
            return 0
        fi
    fi
    
    # If GitHub fails, provide Kaggle instructions
    print_message "$YELLOW" "\nAlternative: Download from Kaggle"
    print_message "$YELLOW" "1. Install Kaggle API: pip install kaggle"
    print_message "$YELLOW" "2. Setup API credentials: https://github.com/Kaggle/kaggle-api"
    print_message "$YELLOW" "3. Run: kaggle datasets download -d ${KAGGLE_DATASET}"
    print_message "$YELLOW" "4. Extract to: ${RAW_DATA_DIR}"
    
    return 1
}

# Function to setup Python environment
setup_python_env() {
    print_message "$BLUE" "\n=== Setting up Python Environment ==="
    
    # Check if virtual environment exists
    if [ ! -d "${BASE_DIR}/venv" ]; then
        print_message "$YELLOW" "Creating virtual environment..."
        python3 -m venv "${BASE_DIR}/venv"
    fi
    
    # Activate virtual environment
    source "${BASE_DIR}/venv/bin/activate"
    
    # Install required packages for data processing
    print_message "$YELLOW" "Installing required Python packages..."
    pip install --quiet --upgrade pip
    pip install --quiet numpy pandas h5py scipy scikit-learn
    
    print_message "$GREEN" "✓ Python environment ready"
}

# Function to create data statistics
generate_data_stats() {
    print_message "$BLUE" "\n=== Generating Data Statistics ==="
    
    cat > "${TEMP_DIR}/data_stats.py" << 'EOF'
import numpy as np
import pandas as pd
import h5py
import os
import json
from pathlib import Path

def get_file_stats(file_path):
    """Get statistics for a data file"""
    stats = {}
    
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        stats['shape'] = data.shape
        stats['dtype'] = str(data.dtype)
        stats['size_mb'] = data.nbytes / (1024 * 1024)
        if data.ndim <= 2:
            stats['mean'] = float(np.mean(data))
            stats['std'] = float(np.std(data))
            stats['min'] = float(np.min(data))
            stats['max'] = float(np.max(data))
    elif file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as f:
            stats['datasets'] = list(f.keys())
            for key in f.keys():
                data = f[key][:]
                stats[key] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                }
    
    return stats

# Collect statistics
stats = {
    'smap': {},
    'msl': {},
    'summary': {}
}

# SMAP statistics
smap_dir = Path('data/raw/smap')
for file in smap_dir.glob('*.npy'):
    stats['smap'][file.name] = get_file_stats(str(file))

# MSL statistics
msl_dir = Path('data/raw/msl')
for file in msl_dir.glob('*.h5'):
    stats['msl'][file.name] = get_file_stats(str(file))

# Summary
stats['summary'] = {
    'total_files': len(list(smap_dir.glob('*'))) + len(list(msl_dir.glob('*'))),
    'smap_files': len(list(smap_dir.glob('*'))),
    'msl_files': len(list(msl_dir.glob('*')))
}

# Save statistics
with open('data/data_statistics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Data statistics generated: data/data_statistics.json")
EOF
    
    cd "$BASE_DIR"
    python3 "${TEMP_DIR}/data_stats.py"
    
    print_message "$GREEN" "✓ Data statistics generated"
}

# Function to cleanup temporary files
cleanup() {
    print_message "$YELLOW" "\nCleaning up temporary files..."
    
    # Remove compressed archives after extraction
    rm -f "${TEMP_DIR}"/*.tar.gz
    rm -f "${TEMP_DIR}"/*.zip
    
    print_message "$GREEN" "✓ Cleanup completed"
}

# Main execution function
main() {
    print_banner
    
    # Parse command line arguments
    SKIP_DOWNLOAD=false
    USE_SAMPLE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --sample)
                USE_SAMPLE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-download    Skip downloading if data exists"
                echo "  --sample          Generate sample data instead of downloading"
                echo "  --help            Show this help message"
                exit 0
                ;;
            *)
                print_message "$RED" "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check requirements
    check_requirements
    
    # Create directory structure
    create_directories
    
    # Setup Python environment (optional)
    # setup_python_env
    
    if [ "$USE_SAMPLE" = true ]; then
        # Generate sample data for testing
        download_sample_data
    else
        # Download real datasets
        if [ "$SKIP_DOWNLOAD" = false ]; then
            # Try primary download sources
            download_success=true
            
            download_smap_data || download_success=false
            download_msl_data || download_success=false
            download_labels || download_success=false
            
            # If primary sources fail, try alternatives
            if [ "$download_success" = false ]; then
                print_message "$YELLOW" "\nPrimary download failed, trying alternatives..."
                download_from_alternative || download_sample_data
            fi
        else
            print_message "$YELLOW" "Skipping download as requested"
        fi
    fi
    
    # Verify downloaded data
    verify_data
    
    # Generate statistics
    generate_data_stats
    
    # Cleanup
    cleanup
    
    print_message "$BLUE" "\n============================================"
    print_message "$GREEN" "   Data download process completed!        "
    print_message "$BLUE" "============================================"
    print_message "$GREEN" "\nData location: ${DATA_DIR}"
    print_message "$GREEN" "You can now run the training scripts."
    echo ""
}

# Trap errors and cleanup on exit
trap 'print_message "$RED" "\nScript interrupted! Cleaning up..."; cleanup; exit 1' INT TERM

# Run main function
main "$@"