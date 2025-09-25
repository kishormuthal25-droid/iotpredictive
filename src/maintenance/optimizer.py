"""
Optimizer Module for IoT Anomaly Detection System
Hyperparameter optimization, model compression, and system performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import warnings
import json
import pickle
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import make_scorer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import quantize_dynamic
import joblib
from scipy.optimize import differential_evolution, minimize
from scipy.stats import uniform, loguniform
import psutil
import gc

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class OptimizationConfig:
    """Configuration for optimization tasks"""
    n_trials: int = 100
    timeout: Optional[int] = 3600  # seconds
    n_jobs: int = -1  # Parallel jobs
    sampler: str = 'tpe'  # 'tpe', 'random', 'cmaes'
    pruner: str = 'median'  # 'median', 'hyperband'
    cv_folds: int = 5
    scoring_metric: str = 'f1_score'
    direction: str = 'maximize'  # 'maximize' or 'minimize'
    
    # Model compression
    enable_quantization: bool = True
    enable_pruning: bool = True
    pruning_sparsity: float = 0.5
    
    # Resource optimization
    optimize_memory: bool = True
    optimize_batch_size: bool = True
    optimize_cache: bool = True
    
    # Early stopping
    early_stopping_rounds: int = 10
    early_stopping_tolerance: float = 0.001


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    resource_usage: Dict[str, Any]
    optimization_time: float
    model_size_reduction: Optional[float] = None
    inference_speedup: Optional[float] = None


class HyperparameterOptimizer:
    """Hyperparameter optimization for ML models"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize Hyperparameter Optimizer
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        logger.info("Initialized Hyperparameter Optimizer")
        
    def optimize_lstm_forecaster(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray) -> OptimizationResult:
        """Optimize LSTM forecaster hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            OptimizationResult with best parameters
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'lstm_units': [
                    trial.suggest_int('lstm_units_1', 32, 256, step=32),
                    trial.suggest_int('lstm_units_2', 16, 128, step=16)
                ],
                'dense_units': [
                    trial.suggest_int('dense_units_1', 16, 64, step=8)
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.0, 0.3),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
                'optimizer': trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
            }
            
            # Build and train model with suggested parameters
            model = self._build_lstm_model(X_train.shape[1:], params)
            
            # Train with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            val_loss = min(history.history['val_loss'])
            
            # Prune if not promising
            trial.report(val_loss, len(history.history['val_loss']))
            if trial.should_prune():
                raise optuna.TrialPruned()
                
            return val_loss
            
        # Create and run study
        study = optuna.create_study(
            direction='minimize',
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=1  # TensorFlow doesn't play well with multiprocessing
        )
        
        optimization_time = time.time() - start_time
        
        # Store results
        self.study = study
        self.best_params = study.best_params
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=[{
                'params': t.params,
                'value': t.value,
                'state': str(t.state)
            } for t in study.trials],
            resource_usage=self._get_resource_usage(),
            optimization_time=optimization_time
        )
        
    def optimize_isolation_forest(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray) -> OptimizationResult:
        """Optimize Isolation Forest hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels (anomalies)
            
        Returns:
            OptimizationResult with best parameters
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import f1_score
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_samples': trial.suggest_uniform('max_samples', 0.1, 1.0),
                'contamination': trial.suggest_uniform('contamination', 0.01, 0.3),
                'max_features': trial.suggest_uniform('max_features', 0.5, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            
            # Create model
            model = IsolationForest(
                **params,
                random_state=42,
                n_jobs=-1
            )
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train = X_train[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train[val_idx]
                
                # Fit and predict
                model.fit(X_fold_train)
                y_pred = model.predict(X_fold_val)
                y_pred_binary = (y_pred == -1).astype(int)
                
                # Calculate score
                score = f1_score(y_fold_val, y_pred_binary)
                scores.append(score)
                
            return np.mean(scores)
            
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=[{
                'params': t.params,
                'value': t.value,
                'state': str(t.state)
            } for t in study.trials],
            resource_usage=self._get_resource_usage(),
            optimization_time=optimization_time
        )
        
    def optimize_autoencoder(self,
                           X_train: np.ndarray,
                           X_val: np.ndarray) -> OptimizationResult:
        """Optimize Autoencoder hyperparameters
        
        Args:
            X_train: Training data
            X_val: Validation data
            
        Returns:
            OptimizationResult with best parameters
        """
        def objective(trial):
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 2, 4)
            encoder_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'encoder_dim_{i}', 
                                       16 * (2 ** (n_layers - i - 1)),
                                       128 * (2 ** (n_layers - i - 1)),
                                       step=16)
                encoder_dims.append(dim)
                
            params = {
                'encoder_dims': encoder_dims,
                'latent_dim': trial.suggest_int('latent_dim', 8, 64, step=8),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            
            # Build and train model
            model = self._build_autoencoder(X_train.shape[1:], params)
            
            # Train with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, X_train,
                validation_data=(X_val, X_val),
                epochs=50,
                batch_size=params['batch_size'],
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return best validation loss
            return min(history.history['val_loss'])
            
        # Create and run study
        study = optuna.create_study(
            direction='minimize',
            sampler=self._get_sampler(),
            pruner=self._get_pruner()
        )
        
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=1
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=[{
                'params': t.params,
                'value': t.value,
                'state': str(t.state)
            } for t in study.trials],
            resource_usage=self._get_resource_usage(),
            optimization_time=optimization_time
        )
        
    def _build_lstm_model(self, input_shape: Tuple, params: Dict) -> tf.keras.Model:
        """Build LSTM model with given parameters"""
        model = tf.keras.Sequential()
        
        # First LSTM layer
        if params['use_bidirectional']:
            model.add(layers.Bidirectional(
                layers.LSTM(params['lstm_units'][0],
                          return_sequences=len(params['lstm_units']) > 1,
                          dropout=params['dropout_rate'],
                          recurrent_dropout=params['recurrent_dropout']),
                input_shape=input_shape
            ))
        else:
            model.add(layers.LSTM(
                params['lstm_units'][0],
                return_sequences=len(params['lstm_units']) > 1,
                dropout=params['dropout_rate'],
                recurrent_dropout=params['recurrent_dropout'],
                input_shape=input_shape
            ))
            
        # Additional LSTM layers
        for units in params['lstm_units'][1:]:
            if params['use_bidirectional']:
                model.add(layers.Bidirectional(
                    layers.LSTM(units,
                              return_sequences=False,
                              dropout=params['dropout_rate'],
                              recurrent_dropout=params['recurrent_dropout'])
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=False,
                    dropout=params['dropout_rate'],
                    recurrent_dropout=params['recurrent_dropout']
                ))
                
        # Dense layers
        for units in params['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(params['dropout_rate']))
            
        # Output layer
        model.add(layers.Dense(input_shape[-1]))
        
        # Compile
        if params['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=params['learning_rate'])
            
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
        
    def _build_autoencoder(self, input_shape: Tuple, params: Dict) -> tf.keras.Model:
        """Build autoencoder model with given parameters"""
        input_layer = layers.Input(shape=input_shape)
        
        # Encoder
        x = input_layer
        for dim in params['encoder_dims']:
            x = layers.Dense(dim, activation=params['activation'])(x)
            x = layers.Dropout(params['dropout_rate'])(x)
            
        # Latent layer
        latent = layers.Dense(params['latent_dim'], activation=params['activation'])(x)
        
        # Decoder
        x = latent
        for dim in reversed(params['encoder_dims']):
            x = layers.Dense(dim, activation=params['activation'])(x)
            x = layers.Dropout(params['dropout_rate'])(x)
            
        # Output
        output = layers.Dense(input_shape[0], activation='linear')(x)
        
        # Create model
        model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _get_sampler(self):
        """Get Optuna sampler based on configuration"""
        if self.config.sampler == 'tpe':
            return TPESampler(seed=42)
        elif self.config.sampler == 'random':
            return RandomSampler(seed=42)
        elif self.config.sampler == 'cmaes':
            return CmaEsSampler(seed=42)
        else:
            return TPESampler(seed=42)
            
    def _get_pruner(self):
        """Get Optuna pruner based on configuration"""
        if self.config.pruner == 'median':
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        elif self.config.pruner == 'hyperband':
            return HyperbandPruner(
                min_resource=1,
                max_resource='auto',
                reduction_factor=3
            )
        else:
            return MedianPruner()
            
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'n_threads': threading.active_count()
        }


class ModelCompressor:
    """Model compression and optimization for deployment"""
    
    def __init__(self):
        """Initialize Model Compressor"""
        self.compression_results = {}
        logger.info("Initialized Model Compressor")
        
    def compress_tensorflow_model(self,
                                 model: tf.keras.Model,
                                 quantization: bool = True,
                                 pruning: bool = True,
                                 pruning_sparsity: float = 0.5) -> Tuple[tf.keras.Model, Dict]:
        """Compress TensorFlow model for CPU deployment
        
        Args:
            model: Original model
            quantization: Apply quantization
            pruning: Apply weight pruning
            pruning_sparsity: Target sparsity for pruning
            
        Returns:
            Compressed model and compression metrics
        """
        original_size = self._get_model_size_tf(model)
        start_time = time.time()
        
        compressed_model = model
        
        # Apply pruning
        if pruning:
            compressed_model = self._prune_tf_model(compressed_model, pruning_sparsity)
            
        # Apply quantization
        if quantization:
            compressed_model = self._quantize_tf_model(compressed_model)
            
        # Calculate metrics
        compressed_size = self._get_model_size_tf(compressed_model)
        compression_time = time.time() - start_time
        
        metrics = {
            'original_size_mb': original_size / 1024 / 1024,
            'compressed_size_mb': compressed_size / 1024 / 1024,
            'compression_ratio': original_size / compressed_size,
            'size_reduction': 1 - (compressed_size / original_size),
            'compression_time': compression_time
        }
        
        logger.info(f"Model compressed: {metrics['compression_ratio']:.2f}x reduction")
        
        return compressed_model, metrics
        
    def compress_pytorch_model(self,
                             model: nn.Module,
                             quantization: bool = True,
                             pruning: bool = True,
                             pruning_sparsity: float = 0.5) -> Tuple[nn.Module, Dict]:
        """Compress PyTorch model for CPU deployment
        
        Args:
            model: Original model
            quantization: Apply quantization
            pruning: Apply weight pruning
            pruning_sparsity: Target sparsity for pruning
            
        Returns:
            Compressed model and compression metrics
        """
        original_size = self._get_model_size_torch(model)
        start_time = time.time()
        
        compressed_model = model
        
        # Apply pruning
        if pruning:
            compressed_model = self._prune_torch_model(compressed_model, pruning_sparsity)
            
        # Apply quantization
        if quantization:
            compressed_model = self._quantize_torch_model(compressed_model)
            
        # Calculate metrics
        compressed_size = self._get_model_size_torch(compressed_model)
        compression_time = time.time() - start_time
        
        metrics = {
            'original_size_mb': original_size / 1024 / 1024,
            'compressed_size_mb': compressed_size / 1024 / 1024,
            'compression_ratio': original_size / compressed_size,
            'size_reduction': 1 - (compressed_size / original_size),
            'compression_time': compression_time
        }
        
        logger.info(f"Model compressed: {metrics['compression_ratio']:.2f}x reduction")
        
        return compressed_model, metrics
        
    def _prune_tf_model(self, model: tf.keras.Model, sparsity: float) -> tf.keras.Model:
        """Apply weight pruning to TensorFlow model"""
        import tensorflow_model_optimization as tfmot
        
        def apply_pruning(layer):
            if isinstance(layer, (layers.Dense, layers.Conv1D, layers.Conv2D)):
                return tfmot.sparsity.keras.prune_low_magnitude(
                    layer,
                    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=sparsity,
                        begin_step=0,
                        frequency=100
                    )
                )
            return layer
            
        # Clone and apply pruning
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning
        )
        
        # Copy weights
        pruned_model.set_weights(model.get_weights())
        
        return pruned_model
        
    def _quantize_tf_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Apply quantization to TensorFlow model"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Convert to TFLite
        tflite_model = converter.convert()
        
        # For now, return original model (TFLite requires different inference)
        # In production, you'd use the TFLite model
        return model
        
    def _prune_torch_model(self, model: nn.Module, sparsity: float) -> nn.Module:
        """Apply weight pruning to PyTorch model"""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')
                
        return model
        
    def _quantize_torch_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to PyTorch model"""
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        return quantized_model
        
    def _get_model_size_tf(self, model: tf.keras.Model) -> int:
        """Get TensorFlow model size in bytes"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            model.save(tmp.name)
            size = Path(tmp.name).stat().st_size
        return size
        
    def _get_model_size_torch(self, model: nn.Module) -> int:
        """Get PyTorch model size in bytes"""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            size = Path(tmp.name).stat().st_size
        return size


class BatchSizeOptimizer:
    """Optimize batch size for maximum throughput"""
    
    def __init__(self):
        """Initialize Batch Size Optimizer"""
        self.optimal_batch_sizes = {}
        logger.info("Initialized Batch Size Optimizer")
        
    def find_optimal_batch_size(self,
                               model: Any,
                               input_shape: Tuple,
                               min_batch: int = 1,
                               max_batch: int = 512,
                               target_memory_usage: float = 0.8) -> int:
        """Find optimal batch size for model
        
        Args:
            model: Model to optimize
            input_shape: Shape of single input
            min_batch: Minimum batch size
            max_batch: Maximum batch size
            target_memory_usage: Target memory usage fraction
            
        Returns:
            Optimal batch size
        """
        logger.info("Finding optimal batch size...")
        
        # Binary search for optimal batch size
        left, right = min_batch, max_batch
        optimal_batch = min_batch
        best_throughput = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test batch size
            try:
                throughput = self._test_batch_size(model, input_shape, mid)
                memory_usage = psutil.virtual_memory().percent / 100
                
                if memory_usage < target_memory_usage:
                    if throughput > best_throughput:
                        best_throughput = throughput
                        optimal_batch = mid
                    left = mid + 1
                else:
                    right = mid - 1
                    
            except (RuntimeError, MemoryError):
                # Out of memory, reduce batch size
                right = mid - 1
                
            # Clear memory
            gc.collect()
            
        logger.info(f"Optimal batch size: {optimal_batch} (throughput: {best_throughput:.2f} samples/sec)")
        
        return optimal_batch
        
    def _test_batch_size(self, model: Any, input_shape: Tuple, batch_size: int) -> float:
        """Test throughput for given batch size"""
        # Create dummy batch
        if len(input_shape) == 1:
            dummy_batch = np.random.randn(batch_size, input_shape[0])
        else:
            dummy_batch = np.random.randn(batch_size, *input_shape)
            
        # Measure inference time
        start_time = time.time()
        
        # Detect model type and run inference
        if hasattr(model, 'predict'):
            # TensorFlow/Keras model
            _ = model.predict(dummy_batch, verbose=0)
        elif hasattr(model, 'forward'):
            # PyTorch model
            with torch.no_grad():
                _ = model(torch.FloatTensor(dummy_batch))
        else:
            # Sklearn model
            _ = model.predict(dummy_batch)
            
        inference_time = time.time() - start_time
        throughput = batch_size / inference_time
        
        return throughput


class MemoryOptimizer:
    """Optimize memory usage for data processing"""
    
    def __init__(self):
        """Initialize Memory Optimizer"""
        self.optimization_stats = {}
        logger.info("Initialized Memory Optimizer")
        
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage
        
        Args:
            df: Original DataFrame
            
        Returns:
            Optimized DataFrame
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(f"Initial DataFrame memory: {initial_memory:.2f} MB")
        
        # Optimize each column
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                # Optimize numeric columns
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            else:
                # Optimize object columns
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
                    
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Final DataFrame memory: {final_memory:.2f} MB (reduced by {reduction:.1f}%)")
        
        self.optimization_stats['dataframe'] = {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'reduction_percent': reduction
        }
        
        return df
        
    def optimize_numpy_array(self, arr: np.ndarray, precision: str = 'float32') -> np.ndarray:
        """Optimize NumPy array memory usage
        
        Args:
            arr: Original array
            precision: Target precision ('float16', 'float32', 'int8', etc.)
            
        Returns:
            Optimized array
        """
        initial_memory = arr.nbytes / 1024 / 1024
        
        # Convert to specified precision
        if precision == 'float16':
            arr_optimized = arr.astype(np.float16)
        elif precision == 'float32':
            arr_optimized = arr.astype(np.float32)
        elif precision == 'int8':
            arr_optimized = arr.astype(np.int8)
        elif precision == 'int16':
            arr_optimized = arr.astype(np.int16)
        else:
            arr_optimized = arr
            
        final_memory = arr_optimized.nbytes / 1024 / 1024
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        logger.info(f"Array memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB ({reduction:.1f}%)")
        
        return arr_optimized


class InferenceOptimizer:
    """Optimize model inference for production"""
    
    def __init__(self):
        """Initialize Inference Optimizer"""
        self.optimization_cache = {}
        logger.info("Initialized Inference Optimizer")
        
    def optimize_inference_pipeline(self,
                                   model: Any,
                                   preprocess_fn: Optional[Callable] = None,
                                   postprocess_fn: Optional[Callable] = None,
                                   enable_caching: bool = True,
                                   cache_size: int = 1000) -> Callable:
        """Create optimized inference pipeline
        
        Args:
            model: Model for inference
            preprocess_fn: Preprocessing function
            postprocess_fn: Postprocessing function
            enable_caching: Enable result caching
            cache_size: Maximum cache size
            
        Returns:
            Optimized inference function
        """
        from functools import lru_cache
        
        # Create cached inference function
        if enable_caching:
            @lru_cache(maxsize=cache_size)
            def cached_inference(data_hash):
                return self._run_inference(model, data_hash)
        else:
            cached_inference = None
            
        def optimized_inference(data):
            # Preprocess
            if preprocess_fn:
                data = preprocess_fn(data)
                
            # Check cache if enabled
            if enable_caching:
                data_hash = hash(data.tobytes() if hasattr(data, 'tobytes') else str(data))
                result = cached_inference(data_hash)
            else:
                result = self._run_inference(model, data)
                
            # Postprocess
            if postprocess_fn:
                result = postprocess_fn(result)
                
            return result
            
        return optimized_inference
        
    def _run_inference(self, model: Any, data: Any) -> Any:
        """Run model inference"""
        if hasattr(model, 'predict'):
            return model.predict(data)
        elif hasattr(model, 'forward'):
            with torch.no_grad():
                return model(data).numpy()
        else:
            return model(data)
            
    def profile_inference(self, 
                         model: Any,
                         test_data: np.ndarray,
                         n_iterations: int = 100) -> Dict[str, float]:
        """Profile model inference performance
        
        Args:
            model: Model to profile
            test_data: Test data for profiling
            n_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        times = []
        memory_usage = []
        
        for _ in range(n_iterations):
            # Measure memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Measure inference time
            start_time = time.perf_counter()
            _ = self._run_inference(model, test_data)
            inference_time = time.perf_counter() - start_time
            
            # Measure memory after
            mem_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            times.append(inference_time)
            memory_usage.append(mem_after - mem_before)
            
        metrics = {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'p50_inference_time': np.percentile(times, 50),
            'p95_inference_time': np.percentile(times, 95),
            'p99_inference_time': np.percentile(times, 99),
            'mean_memory_delta_mb': np.mean(memory_usage),
            'throughput_samples_per_sec': 1.0 / np.mean(times)
        }
        
        logger.info(f"Inference profiling: {metrics['mean_inference_time']*1000:.2f}ms average, "
                   f"{metrics['throughput_samples_per_sec']:.2f} samples/sec")
        
        return metrics