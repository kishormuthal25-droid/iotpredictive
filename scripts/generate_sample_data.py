#!/usr/bin/env python3
"""
Generate Sample NASA-like Telemetry Data for Testing
Creates SMAP and MSL sample datasets that match the expected format
"""

import numpy as np
import os

def generate_time_series(n_samples=1000, n_features=25, anomaly_rate=0.05):
    """Generate sample time series data with anomalies"""
    # Set seed for reproducibility
    np.random.seed(42)

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

    return data, labels, anomaly_indices

def main():
    print("Generating sample NASA SMAP/MSL telemetry data...")

    # Create directories if they don't exist
    os.makedirs('data/raw/smap', exist_ok=True)
    os.makedirs('data/raw/msl', exist_ok=True)

    # Generate SMAP sample data (25 features as per NASA SMAP)
    print("Generating SMAP sample data...")
    smap_train_data, smap_train_labels, smap_anomalies = generate_time_series(5000, 25, 0.03)
    smap_test_data, smap_test_labels, _ = generate_time_series(2000, 25, 0.05)

    # Save as .npy files
    np.save('data/raw/smap/train.npy', smap_train_data)
    np.save('data/raw/smap/train_labels.npy', smap_train_labels)
    np.save('data/raw/smap/test.npy', smap_test_data)
    np.save('data/raw/smap/test_labels.npy', smap_test_labels)

    print(f"[OK] SMAP data saved - Train: {smap_train_data.shape}, Test: {smap_test_data.shape}")

    # Generate MSL sample data (55 features as per NASA MSL)
    print("Generating MSL sample data...")
    msl_train_data, msl_train_labels, msl_anomalies = generate_time_series(5000, 55, 0.04)
    msl_test_data, msl_test_labels, _ = generate_time_series(2000, 55, 0.06)

    # Save as .npy files (using .npy instead of .h5 to avoid h5py dependency issues)
    np.save('data/raw/msl/train.npy', msl_train_data)
    np.save('data/raw/msl/train_labels.npy', msl_train_labels)
    np.save('data/raw/msl/test.npy', msl_test_data)
    np.save('data/raw/msl/test_labels.npy', msl_test_labels)

    print(f"[OK] MSL data saved - Train: {msl_train_data.shape}, Test: {msl_test_data.shape}")

    # Create a simple labels CSV
    print("Creating labels CSV...")

    # Create simple CSV content without pandas dependency
    csv_content = "spacecraft,anomaly_sequences,start_index,end_index\n"

    # Add SMAP anomalies
    for idx in smap_anomalies[:10]:  # Just first 10 for testing
        csv_content += f"SMAP,{idx},{idx},{idx+10}\n"

    # Add MSL anomalies
    for idx in msl_anomalies[:10]:  # Just first 10 for testing
        csv_content += f"MSL,{idx},{idx},{idx+10}\n"

    with open('data/raw/labeled_anomalies.csv', 'w') as f:
        f.write(csv_content)

    print("[OK] Labels CSV created")

    print("\nSample data generation completed!")
    print(f"SMAP train shape: {smap_train_data.shape}")
    print(f"MSL train shape: {msl_train_data.shape}")
    print(f"Data saved in: data/raw/")

    return True

if __name__ == "__main__":
    main()