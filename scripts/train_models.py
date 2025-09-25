#!/usr/bin/env python3
"""
Model Training Script for IoT Anomaly Detection System
Trains and evaluates all models (LSTM, Autoencoder, VAE, Transformer)
"""

import os
import sys
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
import optuna

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data_ingestion.data_loader import DataLoader
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.preprocessing.feature_engineering import AdvancedFeatureEngineering
from src.anomaly_detection.lstm_detector import LSTMDetector
from src.anomaly_detection.lstm_autoencoder import LSTMAutoencoder
from src.anomaly_detection.lstm_vae import LSTMVAE
from src.anomaly_detection.model_evaluator import ModelEvaluator
from src.forecasting.transformer_forecaster import TransformerForecaster
from src.forecasting.lstm_forecaster import LSTMForecaster
from src.forecasting.forecast_evaluator import ForecastEvaluator
from src.utils.logger import get_logger
from src.utils.metrics import AnomalyDetectionMetrics, ForecastingMetrics
from config.settings import Settings

# Initialize logger
logger = get_logger(__name__)

class ModelTrainer:
    """Main class for training all models"""
    
    def __init__(self, config_path: str = None):
        """Initialize model trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.settings = Settings(config_path)
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = AdvancedFeatureEngineering()
        self.model_evaluator = ModelEvaluator()
        self.forecast_evaluator = ForecastEvaluator()
        
        # Initialize MLflow for experiment tracking
        mlflow.set_tracking_uri(self.settings.get('mlflow_uri', 'mlruns'))
        mlflow.set_experiment('iot_anomaly_detection')
        
        # Set random seeds for reproducibility
        self.set_seeds(42)
        
        # Results storage
        self.results = {
            'anomaly_detection': {},
            'forecasting': {},
            'metadata': {
                'training_date': datetime.now().isoformat(),
                'config': self.settings.to_dict()
            }
        }
    
    @staticmethod
    def set_seeds(seed: int = 42):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def load_data(self, dataset: str = 'smap'):
        """Load and prepare dataset
        
        Args:
            dataset: Dataset name ('smap' or 'msl')
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        logger.info(f"Loading {dataset.upper()} dataset...")
        
        data_path = self.settings.data[f'{dataset}_path']
        
        # Load training data
        if dataset == 'smap':
            X_train = self.data_loader.load_npy(f"{data_path}/train.npy")
            y_train = self.data_loader.load_npy(f"{data_path}/train_labels.npy")
            X_test = self.data_loader.load_npy(f"{data_path}/test.npy")
            y_test = self.data_loader.load_npy(f"{data_path}/test_labels.npy")
        else:  # MSL
            X_train, y_train = self.data_loader.load_h5_with_labels(
                f"{data_path}/train.h5", 'data', 'labels'
            )
            X_test, y_test = self.data_loader.load_h5_with_labels(
                f"{data_path}/test.h5", 'data', 'labels'
            )
        
        logger.info(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """Preprocess data for training
        
        Returns:
            Preprocessed data tuple
        """
        logger.info("Preprocessing data...")
        
        # Scale features
        scaler_type = self.settings.preprocessing.get('scaler', 'standard')
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Reshape for scaling if needed
        original_shape_train = X_train.shape
        original_shape_test = X_test.shape
        
        if len(X_train.shape) == 3:
            n_samples, n_timesteps, n_features = X_train.shape
            X_train = X_train.reshape(-1, n_features)
            X_test = X_test.reshape(-1, n_features)
        
        # Fit and transform
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Reshape back if needed
        if len(original_shape_train) == 3:
            X_train = X_train.reshape(original_shape_train)
            X_test = X_test.reshape(original_shape_test)
        
        # Handle imbalanced data if specified
        if self.settings.preprocessing.get('handle_imbalance', False):
            from imblearn.over_sampling import SMOTE
            # Note: SMOTE works on 2D data, might need adjustment for sequences
            pass
        
        logger.info("Preprocessing completed")
        return X_train, y_train, X_test, y_test, scaler
    
    def create_sequences(self, data, labels, sequence_length):
        """Create sequences for time series models
        
        Args:
            data: Input data
            labels: Target labels
            sequence_length: Length of sequences
        
        Returns:
            Tuple of (sequences, sequence_labels)
        """
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
            # Use the label at the end of the sequence
            if labels is not None:
                sequence_labels.append(labels[i + sequence_length - 1])
        
        return np.array(sequences), np.array(sequence_labels) if labels is not None else None
    
    def train_lstm_detector(self, X_train, y_train, X_val, y_val, hyperparams=None):
        """Train LSTM anomaly detector
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            hyperparams: Hyperparameters dictionary
        
        Returns:
            Trained model
        """
        logger.info("Training LSTM Detector...")
        
        # Default hyperparameters
        default_params = {
            'hidden_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create model
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        model = LSTMDetector(
            sequence_length=sequence_length,
            n_features=n_features,
            hidden_units=default_params['hidden_units'],
            dropout_rate=default_params['dropout_rate']
        )
        
        # Compile and train
        model.model.compile(
            optimizer=keras.optimizers.Adam(default_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            keras.callbacks.ModelCheckpoint(
                'models/lstm_detector_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_lstm_autoencoder(self, X_train, X_val, hyperparams=None):
        """Train LSTM Autoencoder
        
        Args:
            X_train: Training sequences (normal data only)
            X_val: Validation sequences
            hyperparams: Hyperparameters dictionary
        
        Returns:
            Trained model
        """
        logger.info("Training LSTM Autoencoder...")
        
        # Default hyperparameters
        default_params = {
            'encoding_dim': 32,
            'latent_dim': 16,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create model
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        model = LSTMAutoencoder(
            sequence_length=sequence_length,
            n_features=n_features,
            encoding_dim=default_params['encoding_dim'],
            latent_dim=default_params['latent_dim']
        )
        
        # Build and compile
        model.build_model()
        model.model.compile(
            optimizer=keras.optimizers.Adam(default_params['learning_rate']),
            loss='mse'
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train (reconstruction task)
        history = model.model.fit(
            X_train, X_train,  # Input and output are the same
            validation_data=(X_val, X_val),
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_lstm_vae(self, X_train, X_val, hyperparams=None):
        """Train LSTM VAE
        
        Args:
            X_train: Training sequences
            X_val: Validation sequences
            hyperparams: Hyperparameters dictionary
        
        Returns:
            Trained model
        """
        logger.info("Training LSTM VAE...")
        
        # Default hyperparameters
        default_params = {
            'latent_dim': 16,
            'intermediate_dim': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'beta': 1.0  # KL divergence weight
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create model
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        
        model = LSTMVAE(
            sequence_length=sequence_length,
            n_features=n_features,
            latent_dim=default_params['latent_dim'],
            intermediate_dim=default_params['intermediate_dim']
        )
        
        # Build and compile
        model.build_model()
        model.model.compile(
            optimizer=keras.optimizers.Adam(default_params['learning_rate']),
            loss=model.vae_loss
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train
        history = model.model.fit(
            X_train, X_train,
            validation_data=(X_val, X_val),
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_transformer_forecaster(self, X_train, y_train, X_val, y_val, hyperparams=None):
        """Train Transformer for forecasting
        
        Args:
            X_train: Training sequences
            y_train: Training targets (future values)
            X_val: Validation sequences
            y_val: Validation targets
            hyperparams: Hyperparameters dictionary
        
        Returns:
            Trained model
        """
        logger.info("Training Transformer Forecaster...")
        
        # Default hyperparameters
        default_params = {
            'd_model': 128,
            'n_heads': 8,
            'n_layers': 4,
            'ff_dim': 256,
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create model
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        forecast_horizon = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        model = TransformerForecaster(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            n_features=n_features,
            d_model=default_params['d_model'],
            n_heads=default_params['n_heads'],
            n_layers=default_params['n_layers'],
            ff_dim=default_params['ff_dim'],
            dropout_rate=default_params['dropout_rate']
        )
        
        # Build and compile
        model.build_model()
        model.model.compile(
            optimizer=keras.optimizers.Adam(default_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6
            )
        ]
        
        # Train
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def train_lstm_forecaster(self, X_train, y_train, X_val, y_val, hyperparams=None):
        """Train LSTM for forecasting
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            hyperparams: Hyperparameters dictionary
        
        Returns:
            Trained model
        """
        logger.info("Training LSTM Forecaster...")
        
        # Default hyperparameters
        default_params = {
            'hidden_units': [64, 32],
            'dropout_rate': 0.2,
            'use_bidirectional': True,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Create model
        sequence_length = X_train.shape[1]
        n_features = X_train.shape[2]
        forecast_horizon = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        model = LSTMForecaster(
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            n_features=n_features,
            hidden_units=default_params['hidden_units'],
            dropout_rate=default_params['dropout_rate'],
            use_bidirectional=default_params['use_bidirectional']
        )
        
        # Build and compile
        model.build_model()
        model.model.compile(
            optimizer=keras.optimizers.Adam(default_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Train
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=default_params['batch_size'],
            epochs=default_params['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate_anomaly_model(self, model, X_test, y_test, model_name):
        """Evaluate anomaly detection model
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            model_name: Name of the model
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        if isinstance(model, (LSTMAutoencoder, LSTMVAE)):
            # For reconstruction-based models
            reconstruction_errors = model.calculate_reconstruction_error(X_test)
            threshold = np.percentile(reconstruction_errors, 95)
            predictions = (reconstruction_errors > threshold).astype(int)
        else:
            # For classification models
            predictions = model.predict(X_test)
            predictions = (predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        metrics = AnomalyDetectionMetrics.calculate_metrics(
            y_test, predictions
        )
        
        # Generate classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.log_metrics({
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'auc_roc': metrics.auc_roc
            })
        
        return {
            'metrics': metrics.to_dict(),
            'classification_report': report
        }
    
    def evaluate_forecast_model(self, model, X_test, y_test, model_name):
        """Evaluate forecasting model
        
        Args:
            model: Trained model
            X_test: Test sequences
            y_test: Test targets
            model_name: Name of the model
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        predictions = model.predict(X_test)
        
        # Calculate metrics
        metrics = ForecastingMetrics.calculate_metrics(y_test, predictions)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            mlflow.log_metrics({
                'mse': metrics.mse,
                'rmse': metrics.rmse,
                'mae': metrics.mae,
                'mape': metrics.mape,
                'r2': metrics.r2
            })
        
        return metrics.to_dict()
    
    def hyperparameter_tuning(self, model_type, X_train, y_train, X_val, y_val, n_trials=20):
        """Perform hyperparameter tuning using Optuna
        
        Args:
            model_type: Type of model to tune
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            n_trials: Number of Optuna trials
        
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        def objective(trial):
            if model_type == 'lstm_detector':
                params = {
                    'hidden_units': [
                        trial.suggest_int('hidden_units_1', 32, 128),
                        trial.suggest_int('hidden_units_2', 16, 64)
                    ],
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'epochs': 20  # Reduced for tuning
                }
                
                model, _ = self.train_lstm_detector(
                    X_train, y_train, X_val, y_val, params
                )
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_predictions = (val_predictions > 0.5).astype(int).flatten()
                val_metrics = AnomalyDetectionMetrics.calculate_metrics(
                    y_val, val_predictions
                )
                
                return val_metrics.f1_score
            
            elif model_type == 'transformer':
                params = {
                    'd_model': trial.suggest_categorical('d_model', [64, 128, 256]),
                    'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
                    'n_layers': trial.suggest_int('n_layers', 2, 6),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'epochs': 20
                }
                
                model, _ = self.train_transformer_forecaster(
                    X_train, y_train, X_val, y_val, params
                )
                
                # Evaluate on validation set
                val_predictions = model.predict(X_val)
                val_metrics = ForecastingMetrics.calculate_metrics(
                    y_val, val_predictions
                )
                
                return -val_metrics.mse  # Minimize MSE
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Create study
        study = optuna.create_study(
            direction='maximize' if model_type == 'lstm_detector' else 'minimize'
        )
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")
        
        return study.best_params
    
    def cross_validation(self, model_class, X, y, n_splits=5):
        """Perform cross-validation
        
        Args:
            model_class: Model class to instantiate
            X: Input data
            y: Target data
            n_splits: Number of CV folds
        
        Returns:
            Cross-validation scores
        """
        logger.info(f"Performing {n_splits}-fold cross-validation...")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}...")
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx] if y is not None else None
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx] if y is not None else None
            
            # Train model
            if model_class == LSTMDetector:
                model, _ = self.train_lstm_detector(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold
                )
                # Evaluate
                val_predictions = model.predict(X_val_fold)
                val_predictions = (val_predictions > 0.5).astype(int).flatten()
                metrics = AnomalyDetectionMetrics.calculate_metrics(
                    y_val_fold, val_predictions
                )
                scores.append(metrics.f1_score)
            
            # Clear session to free memory
            keras.backend.clear_session()
        
        cv_results = {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
        
        logger.info(f"CV Results - Mean: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
        
        return cv_results
    
    def save_models(self, models, save_dir='models'):
        """Save trained models
        
        Args:
            models: Dictionary of trained models
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in models.items():
            model_path = save_path / f"{name}_{timestamp}"
            model.save(str(model_path))
            logger.info(f"Saved {name} to {model_path}")
    
    def generate_report(self, results, output_path='reports/training_report.json'):
        """Generate training report
        
        Args:
            results: Training results dictionary
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {output_path}")
        
        # Generate summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print("\nAnomaly Detection Models:")
        for model_name, metrics in results['anomaly_detection'].items():
            if 'metrics' in metrics:
                print(f"\n{model_name}:")
                print(f"  Accuracy: {metrics['metrics'].get('accuracy', 0):.4f}")
                print(f"  F1 Score: {metrics['metrics'].get('f1_score', 0):.4f}")
                print(f"  AUC-ROC: {metrics['metrics'].get('auc_roc', 0):.4f}")
        
        print("\nForecasting Models:")
        for model_name, metrics in results['forecasting'].items():
            print(f"\n{model_name}:")
            print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
            print(f"  MAE: {metrics.get('mae', 0):.4f}")
            print(f"  R²: {metrics.get('r2', 0):.4f}")
        
        print("\n" + "="*60)
    
    def train_all_models(self, dataset='smap', tune_hyperparameters=False, use_cv=False):
        """Train all models in the pipeline
        
        Args:
            dataset: Dataset to use ('smap' or 'msl')
            tune_hyperparameters: Whether to perform hyperparameter tuning
            use_cv: Whether to use cross-validation
        
        Returns:
            Training results
        """
        logger.info("Starting model training pipeline...")
        
        # Load data
        X, y, X_test, y_test = self.load_data(dataset)
        
        # Preprocess data
        X, y, X_test, y_test, scaler = self.preprocess_data(X, y, X_test, y_test)
        
        # Create sequences for time series models
        sequence_length = self.settings.models['anomaly_detection']['lstm']['sequence_length']
        X_seq, y_seq = self.create_sequences(X, y, sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, sequence_length)
        
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        
        # Filter normal data for autoencoder/VAE training
        normal_indices_train = y_train == 0
        X_train_normal = X_train[normal_indices_train]
        normal_indices_val = y_val == 0
        X_val_normal = X_val[normal_indices_val]
        
        trained_models = {}
        
        # 1. Train LSTM Detector
        logger.info("\n" + "="*40)
        logger.info("Training LSTM Detector")
        logger.info("="*40)
        
        if tune_hyperparameters:
            best_params = self.hyperparameter_tuning(
                'lstm_detector', X_train, y_train, X_val, y_val
            )
            lstm_model, lstm_history = self.train_lstm_detector(
                X_train, y_train, X_val, y_val, best_params
            )
        else:
            lstm_model, lstm_history = self.train_lstm_detector(
                X_train, y_train, X_val, y_val
            )
        
        lstm_results = self.evaluate_anomaly_model(
            lstm_model, X_test_seq, y_test_seq, 'LSTM_Detector'
        )
        self.results['anomaly_detection']['lstm_detector'] = lstm_results
        trained_models['lstm_detector'] = lstm_model
        
        # 2. Train LSTM Autoencoder
        logger.info("\n" + "="*40)
        logger.info("Training LSTM Autoencoder")
        logger.info("="*40)
        
        ae_model, ae_history = self.train_lstm_autoencoder(
            X_train_normal, X_val_normal
        )
        ae_results = self.evaluate_anomaly_model(
            ae_model, X_test_seq, y_test_seq, 'LSTM_Autoencoder'
        )
        self.results['anomaly_detection']['lstm_autoencoder'] = ae_results
        trained_models['lstm_autoencoder'] = ae_model
        
        # 3. Train LSTM VAE
        logger.info("\n" + "="*40)
        logger.info("Training LSTM VAE")
        logger.info("="*40)
        
        vae_model, vae_history = self.train_lstm_vae(
            X_train_normal, X_val_normal
        )
        vae_results = self.evaluate_anomaly_model(
            vae_model, X_test_seq, y_test_seq, 'LSTM_VAE'
        )
        self.results['anomaly_detection']['lstm_vae'] = vae_results
        trained_models['lstm_vae'] = vae_model
        
        # 4. Prepare forecasting data
        forecast_horizon = self.settings.models['forecasting']['horizon']
        X_forecast = []
        y_forecast = []
        
        for i in range(len(X) - sequence_length - forecast_horizon):
            X_forecast.append(X[i:i + sequence_length])
            y_forecast.append(X[i + sequence_length:i + sequence_length + forecast_horizon])
        
        X_forecast = np.array(X_forecast)
        y_forecast = np.array(y_forecast)
        
        # Split forecasting data
        X_fc_train, X_fc_val, y_fc_train, y_fc_val = train_test_split(
            X_forecast[:len(X_forecast)//2],
            y_forecast[:len(y_forecast)//2],
            test_size=0.2,
            random_state=42
        )
        
        X_fc_test = X_forecast[len(X_forecast)//2:]
        y_fc_test = y_forecast[len(y_forecast)//2:]
        
        # 5. Train Transformer Forecaster
        logger.info("\n" + "="*40)
        logger.info("Training Transformer Forecaster")
        logger.info("="*40)
        
        if tune_hyperparameters:
            best_params = self.hyperparameter_tuning(
                'transformer', X_fc_train, y_fc_train, X_fc_val, y_fc_val
            )
            transformer_model, transformer_history = self.train_transformer_forecaster(
                X_fc_train, y_fc_train, X_fc_val, y_fc_val, best_params
            )
        else:
            transformer_model, transformer_history = self.train_transformer_forecaster(
                X_fc_train, y_fc_train, X_fc_val, y_fc_val
            )
        
        transformer_results = self.evaluate_forecast_model(
            transformer_model, X_fc_test, y_fc_test, 'Transformer_Forecaster'
        )
        self.results['forecasting']['transformer'] = transformer_results
        trained_models['transformer_forecaster'] = transformer_model
        
        # 6. Train LSTM Forecaster
        logger.info("\n" + "="*40)
        logger.info("Training LSTM Forecaster")
        logger.info("="*40)
        
        lstm_fc_model, lstm_fc_history = self.train_lstm_forecaster(
            X_fc_train, y_fc_train, X_fc_val, y_fc_val
        )
        lstm_fc_results = self.evaluate_forecast_model(
            lstm_fc_model, X_fc_test, y_fc_test, 'LSTM_Forecaster'
        )
        self.results['forecasting']['lstm_forecaster'] = lstm_fc_results
        trained_models['lstm_forecaster'] = lstm_fc_model
        
        # 7. Cross-validation (optional)
        if use_cv:
            logger.info("\n" + "="*40)
            logger.info("Performing Cross-Validation")
            logger.info("="*40)
            
            cv_results = self.cross_validation(LSTMDetector, X_seq, y_seq)
            self.results['cross_validation'] = cv_results
        
        # Save models
        self.save_models(trained_models)
        
        # Generate report
        self.generate_report(self.results)
        
        logger.info("Model training pipeline completed!")
        
        return self.results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train IoT Anomaly Detection Models')
    parser.add_argument('--dataset', type=str, default='smap',
                       choices=['smap', 'msl'],
                       help='Dataset to use for training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--cv', action='store_true',
                       help='Use cross-validation')
    parser.add_argument('--gpu', type=str, default='0',
                       help='GPU device to use')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Configure TensorFlow
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info(f"Using GPU: {physical_devices[0]}")
        except:
            logger.warning("GPU configuration failed, using CPU")
    else:
        logger.info("No GPU found, using CPU")
    
    # Initialize trainer
    trainer = ModelTrainer(args.config)
    
    # Train models
    try:
        results = trainer.train_all_models(
            dataset=args.dataset,
            tune_hyperparameters=args.tune,
            use_cv=args.cv
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()