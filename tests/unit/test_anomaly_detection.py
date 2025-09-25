"""
Unit Tests for Anomaly Detection Module
Tests for LSTM, Autoencoder, VAE models and evaluation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any

# Import modules to test
from src.anomaly_detection.base_detector import BaseAnomalyDetector
from src.anomaly_detection.lstm_detector import LSTMDetector
from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
from src.anomaly_detection.lstm_vae import LSTMVAE
from src.anomaly_detection.model_evaluator import ModelEvaluator


class TestBaseAnomalyDetector(unittest.TestCase):
    """Test cases for BaseAnomalyDetector abstract class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a concrete implementation for testing
        class ConcreteDetector(BaseAnomalyDetector):
            def build_model(self):
                return keras.Sequential([
                    keras.layers.Dense(10, input_shape=(self.input_dim,)),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
            
            def train(self, X_train, y_train=None, **kwargs):
                self.model.compile(optimizer='adam', loss='binary_crossentropy')
                return self.model.fit(X_train, y_train or np.zeros(len(X_train)), epochs=1, verbose=0)
            
            def predict(self, X):
                return self.model.predict(X, verbose=0)
            
            def detect_anomalies(self, X, threshold=0.5):
                predictions = self.predict(X)
                return predictions > threshold
        
        self.detector = ConcreteDetector(input_dim=10, name='test_detector')
        self.sample_data = np.random.randn(100, 10)
        self.sample_labels = np.random.randint(0, 2, 100)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.input_dim, 10)
        self.assertEqual(self.detector.name, 'test_detector')
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.model, keras.Model)
    
    def test_model_building(self):
        """Test model building"""
        model = self.detector.build_model()
        self.assertIsInstance(model, keras.Model)
        self.assertEqual(model.input_shape[1], 10)
    
    def test_training(self):
        """Test model training"""
        history = self.detector.train(self.sample_data, self.sample_labels)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
    
    def test_prediction(self):
        """Test prediction"""
        self.detector.train(self.sample_data, self.sample_labels)
        predictions = self.detector.predict(self.sample_data[:10])
        
        self.assertEqual(predictions.shape[0], 10)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        self.detector.train(self.sample_data, self.sample_labels)
        anomalies = self.detector.detect_anomalies(self.sample_data[:10], threshold=0.5)
        
        self.assertEqual(anomalies.shape[0], 10)
        self.assertTrue(np.all(np.isin(anomalies, [0, 1])))
    
    def test_save_load_model(self):
        """Test saving and loading model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.detector.train(self.sample_data, self.sample_labels)
            save_path = os.path.join(tmpdir, 'model')
            self.detector.save_model(save_path)
            
            # Check files exist
            self.assertTrue(os.path.exists(f"{save_path}.h5"))
            
            # Load model
            new_detector = ConcreteDetector(input_dim=10, name='loaded')
            new_detector.load_model(save_path)
            
            # Compare predictions
            orig_pred = self.detector.predict(self.sample_data[:5])
            new_pred = new_detector.predict(self.sample_data[:5])
            np.testing.assert_array_almost_equal(orig_pred, new_pred)
    
    def test_threshold_calculation(self):
        """Test threshold calculation"""
        scores = np.random.random(100)
        threshold = self.detector.calculate_threshold(scores, percentile=95)
        
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 1)
        self.assertAlmostEqual(threshold, np.percentile(scores, 95))


class TestLSTMDetector(unittest.TestCase):
    """Test cases for LSTM Detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sequence_length = 20
        self.n_features = 5
        self.detector = LSTMDetector(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            hidden_units=[32, 16],
            dropout_rate=0.2
        )
        
        # Create sample sequential data
        self.sample_sequences = np.random.randn(50, self.sequence_length, self.n_features)
        self.sample_labels = np.random.randint(0, 2, 50)
    
    def test_initialization(self):
        """Test LSTM detector initialization"""
        self.assertEqual(self.detector.sequence_length, 20)
        self.assertEqual(self.detector.n_features, 5)
        self.assertEqual(self.detector.hidden_units, [32, 16])
        self.assertEqual(self.detector.dropout_rate, 0.2)
    
    def test_model_architecture(self):
        """Test LSTM model architecture"""
        model = self.detector.build_model()
        
        # Check model has LSTM layers
        lstm_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.LSTM)]
        self.assertGreater(len(lstm_layers), 0)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, self.sequence_length, self.n_features))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
    
    def test_data_preprocessing(self):
        """Test data preprocessing for LSTM"""
        # Test sequence creation
        raw_data = np.random.randn(100, self.n_features)
        sequences = self.detector.create_sequences(raw_data)
        
        expected_n_sequences = len(raw_data) - self.sequence_length + 1
        self.assertEqual(sequences.shape[0], expected_n_sequences)
        self.assertEqual(sequences.shape[1], self.sequence_length)
        self.assertEqual(sequences.shape[2], self.n_features)
    
    def test_training_with_validation(self):
        """Test training with validation split"""
        history = self.detector.train(
            self.sample_sequences,
            self.sample_labels,
            validation_split=0.2,
            epochs=2,
            batch_size=8,
            verbose=0
        )
        
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)
        self.assertEqual(len(history.history['loss']), 2)
    
    def test_early_stopping(self):
        """Test early stopping callback"""
        with patch.object(keras.callbacks.EarlyStopping, '__init__', return_value=None) as mock_es:
            history = self.detector.train(
                self.sample_sequences,
                self.sample_labels,
                early_stopping=True,
                patience=5,
                epochs=10,
                verbose=0
            )
            
            # Verify early stopping was configured
            mock_es.assert_called_once()
            call_kwargs = mock_es.call_args[1]
            self.assertEqual(call_kwargs.get('patience'), 5)
    
    def test_anomaly_scoring(self):
        """Test anomaly scoring"""
        self.detector.train(self.sample_sequences, self.sample_labels, epochs=1, verbose=0)
        
        scores = self.detector.calculate_anomaly_scores(self.sample_sequences[:10])
        
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.all(scores >= 0) and np.all(scores <= 1))
    
    def test_threshold_optimization(self):
        """Test threshold optimization"""
        self.detector.train(self.sample_sequences, self.sample_labels, epochs=1, verbose=0)
        
        scores = self.detector.calculate_anomaly_scores(self.sample_sequences)
        optimal_threshold = self.detector.optimize_threshold(scores, self.sample_labels)
        
        self.assertIsInstance(optimal_threshold, float)
        self.assertGreater(optimal_threshold, 0)
        self.assertLess(optimal_threshold, 1)
    
    def test_prediction_with_confidence(self):
        """Test prediction with confidence intervals"""
        self.detector.train(self.sample_sequences, self.sample_labels, epochs=1, verbose=0)
        
        predictions, confidence = self.detector.predict_with_confidence(
            self.sample_sequences[:10]
        )
        
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(confidence), 10)
        self.assertTrue(np.all(confidence >= 0) and np.all(confidence <= 1))
    
    def test_model_persistence(self):
        """Test model save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.detector.train(self.sample_sequences, self.sample_labels, epochs=1, verbose=0)
            save_path = os.path.join(tmpdir, 'lstm_model')
            self.detector.save(save_path)
            
            # Load into new detector
            new_detector = LSTMDetector(
                sequence_length=self.sequence_length,
                n_features=self.n_features
            )
            new_detector.load(save_path)
            
            # Compare predictions
            orig_scores = self.detector.calculate_anomaly_scores(self.sample_sequences[:5])
            new_scores = new_detector.calculate_anomaly_scores(self.sample_sequences[:5])
            np.testing.assert_array_almost_equal(orig_scores, new_scores, decimal=5)
    
    def test_online_learning(self):
        """Test online/incremental learning"""
        # Initial training
        self.detector.train(self.sample_sequences[:30], self.sample_labels[:30], epochs=1, verbose=0)
        initial_loss = self.detector.model.evaluate(self.sample_sequences[:30], self.sample_labels[:30], verbose=0)
        
        # Incremental training
        self.detector.update(self.sample_sequences[30:], self.sample_labels[30:], epochs=1, verbose=0)
        
        # Check model has adapted
        final_loss = self.detector.model.evaluate(self.sample_sequences, self.sample_labels, verbose=0)
        self.assertIsNotNone(final_loss)


class TestLSTMAutoencoder(unittest.TestCase):
    """Test cases for LSTM Autoencoder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sequence_length = 20
        self.n_features = 5
        self.autoencoder = LSTMAutoencoder(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            encoding_dim=16,
            latent_dim=8
        )
        
        # Create sample sequences (normal data for autoencoder)
        self.normal_sequences = np.random.randn(50, self.sequence_length, self.n_features)
        self.anomaly_sequences = np.random.randn(10, self.sequence_length, self.n_features) * 3
    
    def test_initialization(self):
        """Test autoencoder initialization"""
        self.assertEqual(self.autoencoder.sequence_length, 20)
        self.assertEqual(self.autoencoder.n_features, 5)
        self.assertEqual(self.autoencoder.encoding_dim, 16)
        self.assertEqual(self.autoencoder.latent_dim, 8)
    
    def test_encoder_decoder_architecture(self):
        """Test encoder-decoder architecture"""
        model = self.autoencoder.build_model()
        
        # Check model has encoder and decoder parts
        self.assertIsNotNone(self.autoencoder.encoder)
        self.assertIsNotNone(self.autoencoder.decoder)
        
        # Check bottleneck dimension
        bottleneck_layer = [l for l in model.layers if 'latent' in l.name.lower()]
        if bottleneck_layer:
            self.assertEqual(bottleneck_layer[0].units, self.autoencoder.latent_dim)
        
        # Check input/output shapes match (reconstruction)
        self.assertEqual(model.input_shape[1:], model.output_shape[1:])
    
    def test_reconstruction(self):
        """Test sequence reconstruction"""
        self.autoencoder.train(self.normal_sequences, epochs=2, verbose=0)
        
        reconstructed = self.autoencoder.reconstruct(self.normal_sequences[:10])
        
        self.assertEqual(reconstructed.shape, (10, self.sequence_length, self.n_features))
        
        # Reconstruction should be similar to input for normal data
        mse = np.mean((self.normal_sequences[:10] - reconstructed) ** 2)
        self.assertLess(mse, 10.0)  # Reasonable threshold
    
    def test_reconstruction_error_calculation(self):
        """Test reconstruction error calculation"""
        self.autoencoder.train(self.normal_sequences, epochs=2, verbose=0)
        
        errors = self.autoencoder.calculate_reconstruction_error(self.normal_sequences[:10])
        
        self.assertEqual(len(errors), 10)
        self.assertTrue(np.all(errors >= 0))
    
    def test_anomaly_detection_via_reconstruction(self):
        """Test anomaly detection using reconstruction error"""
        # Train on normal data
        self.autoencoder.train(self.normal_sequences, epochs=5, verbose=0)
        
        # Calculate errors for normal and anomaly data
        normal_errors = self.autoencoder.calculate_reconstruction_error(self.normal_sequences[:10])
        anomaly_errors = self.autoencoder.calculate_reconstruction_error(self.anomaly_sequences)
        
        # Anomalies should have higher reconstruction error
        self.assertGreater(np.mean(anomaly_errors), np.mean(normal_errors))
    
    def test_threshold_from_validation_data(self):
        """Test threshold calculation from validation data"""
        self.autoencoder.train(self.normal_sequences, epochs=2, verbose=0)
        
        validation_errors = self.autoencoder.calculate_reconstruction_error(self.normal_sequences)
        threshold = self.autoencoder.determine_threshold(validation_errors, contamination=0.1)
        
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
        
        # Check ~10% of validation data would be flagged as anomalies
        anomaly_count = np.sum(validation_errors > threshold)
        expected_count = int(len(validation_errors) * 0.1)
        self.assertAlmostEqual(anomaly_count, expected_count, delta=5)
    
    def test_encoding_extraction(self):
        """Test extracting latent representations"""
        self.autoencoder.train(self.normal_sequences, epochs=2, verbose=0)
        
        encodings = self.autoencoder.encode(self.normal_sequences[:10])
        
        self.assertEqual(encodings.shape[0], 10)
        self.assertEqual(encodings.shape[1], self.autoencoder.latent_dim)
    
    def test_model_compression(self):
        """Test model size after training"""
        self.autoencoder.train(self.normal_sequences, epochs=1, verbose=0)
        
        # Count parameters
        total_params = self.autoencoder.model.count_params()
        
        # Autoencoder should have reasonable number of parameters
        self.assertGreater(total_params, 1000)  # Not too simple
        self.assertLess(total_params, 1000000)  # Not too complex
    
    def test_save_load_autoencoder(self):
        """Test saving and loading autoencoder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.autoencoder.train(self.normal_sequences, epochs=1, verbose=0)
            save_path = os.path.join(tmpdir, 'autoencoder')
            self.autoencoder.save(save_path)
            
            # Load into new autoencoder
            new_autoencoder = LSTMAutoencoder(
                sequence_length=self.sequence_length,
                n_features=self.n_features,
                encoding_dim=16,
                latent_dim=8
            )
            new_autoencoder.load(save_path)
            
            # Compare reconstruction errors
            orig_errors = self.autoencoder.calculate_reconstruction_error(self.normal_sequences[:5])
            new_errors = new_autoencoder.calculate_reconstruction_error(self.normal_sequences[:5])
            np.testing.assert_array_almost_equal(orig_errors, new_errors, decimal=5)


class TestLSTMVAE(unittest.TestCase):
    """Test cases for LSTM Variational Autoencoder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sequence_length = 20
        self.n_features = 5
        self.vae = LSTMVAE(
            sequence_length=self.sequence_length,
            n_features=self.n_features,
            latent_dim=8,
            intermediate_dim=32
        )
        
        self.normal_sequences = np.random.randn(50, self.sequence_length, self.n_features)
        self.anomaly_sequences = np.random.randn(10, self.sequence_length, self.n_features) * 3
    
    def test_initialization(self):
        """Test VAE initialization"""
        self.assertEqual(self.vae.sequence_length, 20)
        self.assertEqual(self.vae.n_features, 5)
        self.assertEqual(self.vae.latent_dim, 8)
        self.assertEqual(self.vae.intermediate_dim, 32)
    
    def test_vae_architecture(self):
        """Test VAE specific architecture"""
        model = self.vae.build_model()
        
        # Check for VAE-specific layers (sampling layer)
        layer_names = [layer.name for layer in model.layers]
        
        # Should have encoder, decoder, and sampling components
        self.assertIsNotNone(self.vae.encoder)
        self.assertIsNotNone(self.vae.decoder)
        
        # Check latent space has mean and log_var
        self.assertIsNotNone(self.vae.z_mean)
        self.assertIsNotNone(self.vae.z_log_var)
    
    def test_sampling_layer(self):
        """Test reparameterization trick sampling"""
        # Create sample mean and log variance
        z_mean = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
        z_log_var = tf.constant([[-1.0, 0.0], [0.0, -1.0]], dtype=tf.float32)
        
        # Sample from distribution
        z_sampled = self.vae.sampling([z_mean, z_log_var])
        
        self.assertEqual(z_sampled.shape, (2, 2))
        # Samples should be different due to random sampling
        self.assertFalse(np.allclose(z_sampled[0], z_sampled[1]))
    
    def test_vae_loss_function(self):
        """Test VAE loss (reconstruction + KL divergence)"""
        self.vae.train(self.normal_sequences, epochs=1, verbose=0)
        
        # Get loss components
        total_loss = self.vae.model.evaluate(self.normal_sequences, self.normal_sequences, verbose=0)
        
        # VAE loss should be positive
        self.assertGreater(total_loss, 0)
    
    def test_latent_space_generation(self):
        """Test generating from latent space"""
        self.vae.train(self.normal_sequences, epochs=2, verbose=0)
        
        # Generate new samples from latent space
        n_samples = 5
        latent_samples = np.random.randn(n_samples, self.vae.latent_dim)
        generated = self.vae.generate_from_latent(latent_samples)
        
        self.assertEqual(generated.shape, (n_samples, self.sequence_length, self.n_features))
    
    def test_latent_space_interpolation(self):
        """Test interpolation in latent space"""
        self.vae.train(self.normal_sequences, epochs=2, verbose=0)
        
        # Encode two sequences
        latent1 = self.vae.encode(self.normal_sequences[0:1])[0]  # z_mean
        latent2 = self.vae.encode(self.normal_sequences[1:2])[0]
        
        # Interpolate
        alpha = 0.5
        latent_interp = alpha * latent1 + (1 - alpha) * latent2
        
        # Decode interpolated point
        generated = self.vae.generate_from_latent(latent_interp)
        
        self.assertEqual(generated.shape, (1, self.sequence_length, self.n_features))
    
    def test_anomaly_detection_elbo(self):
        """Test anomaly detection using ELBO (Evidence Lower Bound)"""
        self.vae.train(self.normal_sequences, epochs=5, verbose=0)
        
        # Calculate ELBO for normal and anomaly data
        normal_elbo = self.vae.calculate_elbo(self.normal_sequences[:10])
        anomaly_elbo = self.vae.calculate_elbo(self.anomaly_sequences)
        
        self.assertEqual(len(normal_elbo), 10)
        self.assertEqual(len(anomaly_elbo), 10)
        
        # Anomalies should have different ELBO
        # Note: Can be higher or lower depending on the nature of anomalies
        self.assertNotAlmostEqual(np.mean(normal_elbo), np.mean(anomaly_elbo), places=1)
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation from VAE"""
        self.vae.train(self.normal_sequences, epochs=2, verbose=0)
        
        # Get uncertainty estimates
        uncertainties = self.vae.estimate_uncertainty(self.normal_sequences[:10])
        
        self.assertEqual(len(uncertainties), 10)
        self.assertTrue(np.all(uncertainties >= 0))
    
    def test_save_load_vae(self):
        """Test saving and loading VAE"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            self.vae.train(self.normal_sequences, epochs=1, verbose=0)
            save_path = os.path.join(tmpdir, 'vae')
            self.vae.save(save_path)
            
            # Save should include architecture and weights
            self.assertTrue(os.path.exists(f"{save_path}_model.h5"))
            self.assertTrue(os.path.exists(f"{save_path}_config.json"))
            
            # Load into new VAE
            new_vae = LSTMVAE(
                sequence_length=self.sequence_length,
                n_features=self.n_features,
                latent_dim=8,
                intermediate_dim=32
            )
            new_vae.load(save_path)
            
            # Compare latent encodings
            orig_latent = self.vae.encode(self.normal_sequences[:5])[0]  # z_mean
            new_latent = new_vae.encode(self.normal_sequences[:5])[0]
            np.testing.assert_array_almost_equal(orig_latent, new_latent, decimal=5)


class TestModelEvaluator(unittest.TestCase):
    """Test cases for Model Evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ModelEvaluator()
        
        # Create sample data
        self.X_test = np.random.randn(100, 20, 5)
        self.y_true = np.random.randint(0, 2, 100)
        self.y_scores = np.random.random(100)
        self.y_pred = (self.y_scores > 0.5).astype(int)
    
    def test_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.results, {})
    
    def test_evaluate_single_model(self):
        """Test evaluating single model"""
        # Create mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=self.y_scores)
        mock_model.name = "test_model"
        
        metrics = self.evaluator.evaluate_model(
            mock_model,
            self.X_test,
            self.y_true,
            threshold=0.5
        )
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('auc_roc', metrics)
        self.assertIn('confusion_matrix', metrics)
    
    def test_compare_multiple_models(self):
        """Test comparing multiple models"""
        # Create mock models
        models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.random.random(100))
            mock_model.name = f"model_{i}"
            models.append(mock_model)
        
        comparison = self.evaluator.compare_models(
            models,
            self.X_test,
            self.y_true
        )
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 3)
        self.assertIn('accuracy', comparison.columns)
        self.assertIn('f1_score', comparison.columns)
    
    def test_cross_validation(self):
        """Test cross-validation evaluation"""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=self.y_scores)
        mock_model.name = "cv_model"
        mock_model.train = Mock()
        
        cv_results = self.evaluator.cross_validate(
            mock_model,
            self.X_test,
            self.y_true,
            cv_folds=3
        )
        
        self.assertIn('mean_accuracy', cv_results)
        self.assertIn('std_accuracy', cv_results)
        self.assertIn('mean_f1', cv_results)
        self.assertIn('fold_scores', cv_results)
        self.assertEqual(len(cv_results['fold_scores']), 3)
    
    def test_threshold_analysis(self):
        """Test threshold analysis"""
        thresholds, metrics = self.evaluator.analyze_thresholds(
            self.y_true,
            self.y_scores,
            thresholds=np.linspace(0.1, 0.9, 9)
        )
        
        self.assertEqual(len(thresholds), 9)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check metrics shape
        self.assertEqual(len(metrics['precision']), 9)
    
    def test_roc_curve_generation(self):
        """Test ROC curve generation"""
        fpr, tpr, thresholds, auc = self.evaluator.generate_roc_curve(
            self.y_true,
            self.y_scores
        )
        
        self.assertTrue(len(fpr) > 0)
        self.assertTrue(len(tpr) > 0)
        self.assertEqual(len(fpr), len(tpr))
        self.assertGreaterEqual(auc, 0)
        self.assertLessEqual(auc, 1)
    
    def test_precision_recall_curve(self):
        """Test precision-recall curve generation"""
        precision, recall, thresholds, auc_pr = self.evaluator.generate_pr_curve(
            self.y_true,
            self.y_scores
        )
        
        self.assertTrue(len(precision) > 0)
        self.assertTrue(len(recall) > 0)
        self.assertEqual(len(precision), len(recall))
        self.assertGreaterEqual(auc_pr, 0)
        self.assertLessEqual(auc_pr, 1)
    
    def test_confusion_matrix_analysis(self):
        """Test confusion matrix analysis"""
        cm_analysis = self.evaluator.analyze_confusion_matrix(
            self.y_true,
            self.y_pred
        )
        
        self.assertIn('confusion_matrix', cm_analysis)
        self.assertIn('true_positives', cm_analysis)
        self.assertIn('false_positives', cm_analysis)
        self.assertIn('true_negatives', cm_analysis)
        self.assertIn('false_negatives', cm_analysis)
        self.assertIn('sensitivity', cm_analysis)
        self.assertIn('specificity', cm_analysis)
    
    def test_ensemble_evaluation(self):
        """Test ensemble model evaluation"""
        # Create mock models for ensemble
        models = []
        for i in range(3):
            mock_model = Mock()
            mock_model.predict = Mock(return_value=np.random.random(100))
            models.append(mock_model)
        
        ensemble_metrics = self.evaluator.evaluate_ensemble(
            models,
            self.X_test,
            self.y_true,
            voting='soft'
        )
        
        self.assertIn('ensemble_accuracy', ensemble_metrics)
        self.assertIn('individual_accuracies', ensemble_metrics)
        self.assertEqual(len(ensemble_metrics['individual_accuracies']), 3)
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Perform evaluation
            mock_model = Mock()
            mock_model.predict = Mock(return_value=self.y_scores)
            mock_model.name = "test_model"
            
            metrics = self.evaluator.evaluate_model(
                mock_model,
                self.X_test,
                self.y_true
            )
            
            # Save results
            save_path = os.path.join(tmpdir, 'evaluation_results.json')
            self.evaluator.save_results(save_path)
            
            # Check file exists and contains results
            self.assertTrue(os.path.exists(save_path))
            
            with open(save_path, 'r') as f:
                loaded_results = json.load(f)
            
            self.assertIn('test_model', loaded_results)
    
    def test_statistical_significance(self):
        """Test statistical significance testing between models"""
        scores1 = np.random.random(100)
        scores2 = np.random.random(100)
        
        p_value, is_significant = self.evaluator.test_significance(
            scores1,
            scores2,
            alpha=0.05
        )
        
        self.assertIsInstance(p_value, float)
        self.assertIsInstance(is_significant, bool)
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)


class TestAnomalyDetectionIntegration(unittest.TestCase):
    """Integration tests for anomaly detection pipeline"""
    
    def test_end_to_end_lstm_pipeline(self):
        """Test end-to-end LSTM anomaly detection pipeline"""
        # Generate synthetic data
        np.random.seed(42)
        normal_data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)
        normal_data += np.random.normal(0, 0.1, normal_data.shape)
        
        # Add anomalies
        anomaly_indices = np.random.choice(1000, 50, replace=False)
        normal_data[anomaly_indices] += np.random.normal(0, 1, (50, 1))
        
        # Create labels
        labels = np.zeros(1000)
        labels[anomaly_indices] = 1
        
        # Create sequences
        sequence_length = 20
        detector = LSTMDetector(sequence_length=sequence_length, n_features=1)
        sequences = detector.create_sequences(normal_data)
        seq_labels = labels[sequence_length-1:]
        
        # Split data
        split_idx = int(0.8 * len(sequences))
        X_train, X_test = sequences[:split_idx], sequences[split_idx:]
        y_train, y_test = seq_labels[:split_idx], seq_labels[split_idx:]
        
        # Train model
        detector.train(X_train, y_train, epochs=5, verbose=0)
        
        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(detector, X_test, y_test)
        
        # Check metrics are reasonable
        self.assertGreater(metrics['accuracy'], 0.5)  # Better than random
        self.assertIn('confusion_matrix', metrics)
    
    def test_autoencoder_vs_vae_comparison(self):
        """Test comparing Autoencoder and VAE performance"""
        # Generate data
        sequence_length = 20
        n_features = 5
        n_samples = 100
        
        normal_data = np.random.randn(n_samples, sequence_length, n_features)
        anomaly_data = np.random.randn(20, sequence_length, n_features) * 3
        
        # Create models
        autoencoder = LSTMAutoencoder(
            sequence_length=sequence_length,
            n_features=n_features,
            encoding_dim=16,
            latent_dim=8
        )
        
        vae = LSTMVAE(
            sequence_length=sequence_length,
            n_features=n_features,
            latent_dim=8,
            intermediate_dim=32
        )
        
        # Train both models
        autoencoder.train(normal_data, epochs=5, verbose=0)
        vae.train(normal_data, epochs=5, verbose=0)
        
        # Calculate reconstruction errors
        ae_normal_errors = autoencoder.calculate_reconstruction_error(normal_data[:20])
        ae_anomaly_errors = autoencoder.calculate_reconstruction_error(anomaly_data)
        
        vae_normal_errors = vae.calculate_elbo(normal_data[:20])
        vae_anomaly_errors = vae.calculate_elbo(anomaly_data)
        
        # Both should distinguish anomalies
        self.assertNotEqual(np.mean(ae_normal_errors), np.mean(ae_anomaly_errors))
        self.assertNotEqual(np.mean(vae_normal_errors), np.mean(vae_anomaly_errors))


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability of anomaly detection models"""
    
    def test_large_dataset_handling(self):
        """Test handling large datasets"""
        # Create large dataset
        n_samples = 10000
        sequence_length = 50
        n_features = 10
        
        large_data = np.random.randn(n_samples, sequence_length, n_features)
        
        # Test LSTM detector can handle it
        detector = LSTMDetector(
            sequence_length=sequence_length,
            n_features=n_features,
            hidden_units=[32]  # Smaller model for speed
        )
        
        # Should complete without memory errors
        import time
        start_time = time.time()
        
        # Use generator for batch training
        batch_size = 32
        for i in range(0, n_samples, batch_size):
            batch = large_data[i:i+batch_size]
            if i == 0:
                detector.model.compile(optimizer='adam', loss='binary_crossentropy')
            detector.model.train_on_batch(batch, np.zeros(len(batch)))
        
        training_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust based on hardware)
        self.assertLess(training_time, 60)  # Less than 1 minute
    
    def test_memory_efficiency(self):
        """Test memory efficiency of models"""
        import tracemalloc
        
        # Start memory tracking
        tracemalloc.start()
        
        # Create and train model
        detector = LSTMDetector(sequence_length=20, n_features=5)
        data = np.random.randn(100, 20, 5)
        detector.train(data, np.zeros(100), epochs=1, verbose=0)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Convert to MB
        peak_mb = peak / 1024 / 1024
        
        # Should use reasonable amount of memory
        self.assertLess(peak_mb, 500)  # Less than 500 MB
    
    def test_inference_speed(self):
        """Test inference speed"""
        # Create model
        detector = LSTMDetector(sequence_length=20, n_features=5)
        data = np.random.randn(100, 20, 5)
        detector.train(data, np.zeros(100), epochs=1, verbose=0)
        
        # Test inference speed
        test_data = np.random.randn(1000, 20, 5)
        
        import time
        start_time = time.time()
        predictions = detector.predict(test_data)
        inference_time = time.time() - start_time
        
        # Calculate throughput
        throughput = len(test_data) / inference_time
        
        # Should achieve reasonable throughput
        self.assertGreater(throughput, 100)  # At least 100 samples/second


if __name__ == '__main__':
    unittest.main()