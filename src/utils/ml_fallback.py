"""
ML Fallback Module
Provides fallback implementations when TensorFlow is not available
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    print("[INFO] TensorFlow is available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not available, using fallback implementations")

class MockModel:
    """Mock model class for when TensorFlow is not available"""

    def __init__(self, input_shape: Tuple, **kwargs):
        self.input_shape = input_shape
        self.is_trained = False
        self.history = None

    def fit(self, X, y=None, epochs=50, batch_size=32, validation_split=0.2, **kwargs):
        """Mock training"""
        print(f"[MOCK] Training model on data shape {X.shape} for {epochs} epochs")

        # Simulate training history
        self.history = {
            'loss': np.random.exponential(0.1, epochs) + 0.01,
            'val_loss': np.random.exponential(0.12, epochs) + 0.02
        }
        self.is_trained = True
        return self

    def predict(self, X):
        """Mock prediction"""
        if not self.is_trained:
            warnings.warn("Model not trained, using random predictions")

        # Return reconstructed data (for autoencoders) or predictions
        return X + np.random.normal(0, 0.1, X.shape)

    def save(self, filepath):
        """Mock save"""
        print(f"[MOCK] Saving model to {filepath}")
        np.savez(filepath, weights=np.random.randn(100), trained=self.is_trained)

    def load_weights(self, filepath):
        """Mock load"""
        print(f"[MOCK] Loading model from {filepath}")
        self.is_trained = True

class MockAutoencoder(MockModel):
    """Mock LSTM Autoencoder"""

    def __init__(self, sequence_length: int, n_features: int, **kwargs):
        super().__init__((sequence_length, n_features))
        self.sequence_length = sequence_length
        self.n_features = n_features

    def get_reconstruction_error(self, X):
        """Calculate reconstruction error"""
        reconstructed = self.predict(X)
        error = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return error

class MockLSTMPredictor(MockModel):
    """Mock LSTM Predictor"""

    def __init__(self, sequence_length: int, n_features: int, **kwargs):
        super().__init__((sequence_length, n_features))
        self.sequence_length = sequence_length
        self.n_features = n_features

    def predict(self, X):
        """Predict next values"""
        # Simulate prediction by adding trend + noise
        last_values = X[:, -1, :]
        predictions = last_values + np.random.normal(0, 0.05, last_values.shape)
        return predictions

class MockVAE(MockModel):
    """Mock LSTM-VAE"""

    def __init__(self, sequence_length: int, n_features: int, latent_dim: int = 20, **kwargs):
        super().__init__((sequence_length, n_features))
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim

    def encode(self, X):
        """Encode to latent space"""
        batch_size = X.shape[0]
        return np.random.normal(0, 1, (batch_size, self.latent_dim))

    def decode(self, z):
        """Decode from latent space"""
        batch_size = z.shape[0]
        return np.random.normal(0, 0.1, (batch_size, self.sequence_length, self.n_features))

    def get_elbo_loss(self, X):
        """Calculate ELBO loss"""
        reconstructed = self.predict(X)
        reconstruction_loss = np.mean(np.square(X - reconstructed))
        kl_loss = np.random.exponential(0.1)
        return reconstruction_loss + 0.1 * kl_loss

def create_model(model_type: str, **kwargs):
    """Factory function to create models"""

    if TENSORFLOW_AVAILABLE:
        # Return actual TensorFlow models when available
        return None  # Will be implemented with actual TF models
    else:
        # Return mock models
        if model_type == 'lstm_autoencoder':
            return MockAutoencoder(**kwargs)
        elif model_type == 'lstm_predictor':
            return MockLSTMPredictor(**kwargs)
        elif model_type == 'lstm_vae':
            return MockVAE(**kwargs)
        else:
            return MockModel(**kwargs)

def is_tensorflow_available() -> bool:
    """Check if TensorFlow is available"""
    return TENSORFLOW_AVAILABLE

def get_mock_metrics():
    """Generate mock training metrics"""
    return {
        'accuracy': np.random.uniform(0.85, 0.95),
        'precision': np.random.uniform(0.80, 0.92),
        'recall': np.random.uniform(0.82, 0.90),
        'f1_score': np.random.uniform(0.83, 0.91),
        'roc_auc': np.random.uniform(0.88, 0.94)
    }