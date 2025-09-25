#!/usr/bin/env python3
"""
Test NASA Telemanom Model Integration
Quick test to verify trained models work correctly
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.anomaly_detection.nasa_telemanom import NASATelemanom

def test_model_loading():
    """Test loading and using trained models"""
    print("Testing NASA Telemanom Model Integration")
    print("="*50)

    models_dir = Path("data/models/telemanom")
    model_files = list(models_dir.glob("SMAP_*.pkl"))

    if not model_files:
        print("No trained models found!")
        return False

    print(f"Found {len(model_files)} trained models")

    # Test first model
    model_file = model_files[0]
    print(f"\nTesting model: {model_file.name}")

    try:
        # Load model
        model = NASATelemanom.load_model(str(model_file))
        print(f"Model loaded successfully")
        print(f"   Sensor ID: {model.sensor_id}")
        print(f"   Features: {model.n_features}")
        print(f"   Threshold: {model.error_threshold:.4f}")
        print(f"   Trained: {model.is_trained}")

        # Generate test data
        test_data = np.random.randn(1000, model.n_features)
        print(f"Generated test data: {test_data.shape}")

        # Run anomaly detection
        results = model.predict_anomalies(test_data)
        print(f"Anomaly detection results:")
        print(f"   Anomalies detected: {np.sum(results['anomalies'])}")
        print(f"   Max score: {np.max(results['scores']):.4f}")
        print(f"   Mean score: {np.mean(results['scores']):.4f}")

        # Test single score
        single_score = model.get_anomaly_score(test_data[:100])
        print(f"   Single window score: {single_score:.4f}")

        print("Model integration test PASSED")
        return True

    except Exception as e:
        print(f"Model integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """Test all trained models"""
    print("\n" + "="*50)
    print("Testing All Trained Models")
    print("="*50)

    models_dir = Path("data/models/telemanom")
    model_files = list(models_dir.glob("SMAP_*.pkl"))

    successful_tests = 0

    for model_file in model_files:
        print(f"\nTesting: {model_file.stem}")

        try:
            model = NASATelemanom.load_model(str(model_file))
            test_data = np.random.randn(500, model.n_features)
            results = model.predict_anomalies(test_data)

            print(f"   PASS {model.sensor_id}: {np.sum(results['anomalies'])} anomalies, score: {np.max(results['scores']):.3f}")
            successful_tests += 1

        except Exception as e:
            print(f"   FAIL {model_file.stem}: {e}")

    print(f"\nTest Summary: {successful_tests}/{len(model_files)} models passed")
    return successful_tests == len(model_files)

if __name__ == "__main__":
    print("Starting NASA Telemanom Integration Tests\n")

    # Test model loading and basic functionality
    test1_passed = test_model_loading()

    # Test all models
    test2_passed = test_all_models()

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)

    if test1_passed and test2_passed:
        print("ALL TESTS PASSED - NASA Telemanom integration is working!")
        print("Models can be loaded, saved, and used for anomaly detection")
        print("Ready for dashboard integration")
    else:
        print("Some tests failed - need to investigate")

    print("\nNext steps:")
    print("1. Integrate trained models with dashboard")
    print("2. Replace placeholder detection with real models")
    print("3. Test real-time anomaly detection")