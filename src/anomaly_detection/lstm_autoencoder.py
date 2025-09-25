"""
LSTM Autoencoder Module for Anomaly Detection
Reconstruction-based anomaly detection using LSTM encoder-decoder architecture
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

# Import project modules first
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import TensorFlow, fall back to mock implementation
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks, backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
        Input, RepeatVector, TimeDistributed, Bidirectional,
        Lambda, Concatenate, Conv1D, MaxPooling1D, UpSampling1D,
        GRU, Flatten, Reshape, Layer
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, Callback
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
    from tensorflow.keras.regularizers import L1L2
    TENSORFLOW_AVAILABLE = True
    print("[INFO] TensorFlow available for LSTM Autoencoder")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not available, using mock implementation")

from sklearn.preprocessing import StandardScaler
from src.anomaly_detection.base_detector import (
    BaseAnomalyDetector, ThresholdStrategy, PercentileThreshold
)
from src.utils.ml_fallback import MockAutoencoder, is_tensorflow_available
from config.settings import settings, get_config, get_data_path

# Import training tracker (with error handling for circular imports)
try:
    from src.anomaly_detection.training_tracker import training_tracker
except ImportError:
    training_tracker = None

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Set TensorFlow logging level if available
if TENSORFLOW_AVAILABLE:
    tf.get_logger().setLevel('ERROR')


@dataclass
class LSTMAutoencoderConfig:
    """Configuration for LSTM Autoencoder"""
    # Encoder architecture
    encoder_units: List[int] = None
    decoder_units: List[int] = None
    latent_dim: int = 32
    
    # Model type
    use_bidirectional: bool = False
    use_attention: bool = False
    use_conv: bool = False  # Convolutional layers
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'mse'
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    
    # Regularization
    dropout_rate: float = 0.2
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    use_batch_norm: bool = True
    noise_factor: float = 0.0  # For denoising autoencoder
    
    # Sequence parameters
    sequence_length: int = 100
    
    def __post_init__(self):
        if self.encoder_units is None:
            self.encoder_units = [128, 64, 32]
        if self.decoder_units is None:
            # Mirror of encoder by default
            self.decoder_units = self.encoder_units[::-1]


class AttentionLayer(Layer):
    """Custom attention layer for sequence-to-sequence models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


class ReconstructionCallback(Callback):
    """Custom callback for monitoring reconstruction during training"""
    
    def __init__(self, validation_data=None, n_samples=5):
        super().__init__()
        self.validation_data = validation_data
        self.n_samples = n_samples
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def on_epoch_end(self, epoch, logs=None):
        self.history['train_loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        if hasattr(self.model.optimizer, 'learning_rate'):
            self.history['lr'].append(K.get_value(self.model.optimizer.learning_rate))
        
        # Log reconstruction quality every 10 epochs
        if epoch % 10 == 0 and self.validation_data is not None:
            X_val, y_val = self.validation_data
            reconstructions = self.model.predict(X_val[:self.n_samples], verbose=0)
            mse = np.mean((X_val[:self.n_samples] - reconstructions) ** 2)
            logger.info(f"Epoch {epoch}: loss={logs.get('loss'):.4f}, "
                       f"val_loss={logs.get('val_loss'):.4f}, "
                       f"sample_mse={mse:.4f}")


class TrainingProgressCallback(Callback):
    """Keras callback to track training progress"""

    def __init__(self, equipment_id: str):
        super().__init__()
        self.equipment_id = equipment_id

    def on_epoch_end(self, epoch, logs=None):
        """Update training progress after each epoch"""
        if training_tracker and logs:
            loss = logs.get('loss', 0.0)
            val_loss = logs.get('val_loss', 0.0)
            accuracy = 1.0 / (1.0 + loss)  # Convert loss to accuracy-like metric

            training_tracker.update_epoch(
                self.equipment_id,
                epoch + 1,  # Convert to 1-based indexing
                loss,
                val_loss,
                accuracy
            )


class LSTMAutoencoder(BaseAnomalyDetector):
    """
    LSTM Autoencoder for anomaly detection
    Detects anomalies based on reconstruction error
    """
    
    def __init__(self,
                 name: str = "LSTMAutoencoder",
                 config: Optional[LSTMAutoencoderConfig] = None,
                 threshold_strategy: Optional[ThresholdStrategy] = None,
                 contamination: float = 0.1,
                 random_state: int = 42,
                 equipment_id: Optional[str] = None):
        """
        Initialize LSTM Autoencoder
        
        Args:
            name: Detector name
            config: Autoencoder configuration
            threshold_strategy: Threshold determination strategy
            contamination: Expected anomaly ratio
            random_state: Random seed
            equipment_id: Equipment identifier for progress tracking
        """
        super().__init__(
            name=name,
            threshold_strategy=threshold_strategy or PercentileThreshold(100 * (1 - contamination)),
            contamination=contamination,
            window_size=config.sequence_length if config else 100,
            random_state=random_state
        )
        
        self.config = config or LSTMAutoencoderConfig()
        self.equipment_id = equipment_id

        # Model components
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.autoencoder: Optional[Model] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Training state
        self.training_history = None
        self.best_model_path = None
        
        # Reconstruction statistics
        self.reconstruction_stats = {'mean': 0, 'std': 1}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info(f"LSTMAutoencoder initialized with latent_dim={self.config.latent_dim}")
    
    def _build_model(self):
        """Build LSTM Autoencoder architecture"""
        # Input layer
        inputs = Input(shape=(self.config.sequence_length, self.n_features), name='encoder_input')
        
        # Add noise for denoising autoencoder
        if self.config.noise_factor > 0:
            noisy_inputs = Lambda(lambda x: x + self.config.noise_factor * K.random_normal(shape=K.shape(x)))(inputs)
            x = noisy_inputs
        else:
            x = inputs
        
        # Build encoder
        encoded = self._build_encoder(x)
        
        # Build decoder
        decoded = self._build_decoder(encoded)
        
        # Create models
        self.encoder = Model(inputs, encoded, name='encoder')
        
        # For decoder, we need to handle the latent input
        latent_inputs = Input(shape=(self.config.latent_dim,), name='latent_input')
        decoder_outputs = self._build_decoder(latent_inputs)
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # Full autoencoder
        self.autoencoder = Model(inputs, decoded, name='autoencoder')
        self.model = self.autoencoder  # For compatibility with base class
        
        # Compile model
        self._compile_model()
        
        logger.info(f"Built LSTM Autoencoder with {self.autoencoder.count_params()} parameters")
    
    def _build_encoder(self, inputs):
        """Build encoder network"""
        x = inputs
        
        # Optional convolutional layers for feature extraction
        if self.config.use_conv and self.n_features > 1:
            x = Conv1D(32, 3, activation='relu', padding='same')(x)
            x = MaxPooling1D(2, padding='same')(x)
            if self.config.use_batch_norm:
                x = BatchNormalization()(x)
        
        # LSTM encoder layers
        for i, units in enumerate(self.config.encoder_units):
            return_sequences = (i < len(self.config.encoder_units) - 1)
            
            # Regularization
            if self.config.l1_reg > 0 or self.config.l2_reg > 0:
                regularizer = L1L2(l1=self.config.l1_reg, l2=self.config.l2_reg)
            else:
                regularizer = None
            
            # LSTM layer
            if self.config.use_bidirectional:
                x = Bidirectional(LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    name=f'encoder_lstm_{i}'
                ))(x)
            else:
                x = LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    name=f'encoder_lstm_{i}'
                )(x)
            
            # Batch normalization
            if self.config.use_batch_norm:
                x = BatchNormalization()(x)
            
            # Dropout
            if self.config.dropout_rate > 0:
                x = Dropout(self.config.dropout_rate)(x)
        
        # Attention mechanism
        if self.config.use_attention:
            # Apply attention to get weighted representation
            x = AttentionLayer()(x) if return_sequences else x
        
        # Latent representation
        latent = Dense(self.config.latent_dim, activation='relu', name='latent_layer')(x)
        
        return latent
    
    def _build_decoder(self, latent):
        """Build decoder network"""
        x = latent
        
        # Repeat vector for sequence generation
        x = RepeatVector(self.config.sequence_length)(x)
        
        # LSTM decoder layers
        for i, units in enumerate(self.config.decoder_units):
            return_sequences = True  # Always return sequences in decoder
            
            # Regularization
            if self.config.l1_reg > 0 or self.config.l2_reg > 0:
                regularizer = L1L2(l1=self.config.l1_reg, l2=self.config.l2_reg)
            else:
                regularizer = None
            
            # LSTM layer
            if self.config.use_bidirectional:
                x = Bidirectional(LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    name=f'decoder_lstm_{i}'
                ))(x)
            else:
                x = LSTM(
                    units,
                    activation=self.config.activation,
                    recurrent_activation=self.config.recurrent_activation,
                    return_sequences=return_sequences,
                    kernel_regularizer=regularizer,
                    name=f'decoder_lstm_{i}'
                )(x)
            
            # Batch normalization
            if self.config.use_batch_norm:
                x = BatchNormalization()(x)
            
            # Dropout
            if self.config.dropout_rate > 0:
                x = Dropout(self.config.dropout_rate)(x)
        
        # Optional convolutional decoder
        if self.config.use_conv and self.n_features > 1:
            x = Conv1D(32, 3, activation='relu', padding='same')(x)
            x = UpSampling1D(2)(x)
        
        # Output layer
        outputs = TimeDistributed(Dense(self.n_features), name='decoder_output')(x)
        
        return outputs
    
    def _compile_model(self):
        """Compile the autoencoder model"""
        # Select optimizer
        if self.config.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.config.learning_rate)
        
        # Select loss
        if self.config.loss == 'mse':
            loss = MeanSquaredError()
        elif self.config.loss == 'mae':
            loss = MeanAbsoluteError()
        else:
            loss = MeanSquaredError()
        
        # Compile with run_eagerly=True to fix TensorFlow prediction issues
        self.autoencoder.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse'],
            run_eagerly=True
        )
    
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit LSTM Autoencoder
        
        Args:
            X: Training data (should contain mostly normal samples)
            y: Optional labels (used for semi-supervised learning)
        """
        # Initialize scaler if not exists
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        # Remove anomalies if labels provided (semi-supervised)
        if y is not None:
            normal_mask = y == 0
            X_normal = X[normal_mask]
            logger.info(f"Using {len(X_normal)} normal samples for training (removed {np.sum(y==1)} anomalies)")
        else:
            X_normal = X
        
        # Prepare data
        if X_normal.ndim == 2:
            # Need to create sequences
            X_normal = X_normal.reshape(-1, self.n_features)
            X_scaled = self.scaler.fit_transform(X_normal)
            X_sequences = self._create_sequences(X_scaled)
        else:
            # Already sequenced
            original_shape = X_normal.shape
            X_flat = X_normal.reshape(-1, X_normal.shape[-1])
            X_scaled = self.scaler.fit_transform(X_flat)
            X_sequences = X_scaled.reshape(original_shape)
        
        # Split data for validation
        split_idx = int(len(X_sequences) * (1 - self.config.validation_split))
        X_train = X_sequences[:split_idx]
        X_val = X_sequences[split_idx:]
        
        # Callbacks
        callbacks_list = self._get_callbacks(X_val)
        
        # Train autoencoder to reconstruct input
        logger.info(f"Training LSTM Autoencoder on {len(X_train)} sequences")
        
        self.training_history = self.autoencoder.fit(
            X_train, X_train,  # Input and target are the same
            validation_data=(X_val, X_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate reconstruction error statistics on training data
        train_reconstructions = self.autoencoder.predict(X_train, verbose=0)
        train_errors = self._calculate_reconstruction_error(X_train, train_reconstructions)
        self.reconstruction_stats['mean'] = np.mean(train_errors)
        self.reconstruction_stats['std'] = np.std(train_errors)
        
        logger.info(f"Training complete. Final loss: {self.training_history.history['loss'][-1]:.4f}")
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences from time series data"""
        sequences = []
        
        for i in range(len(data) - self.config.sequence_length + 1):
            sequences.append(data[i:i + self.config.sequence_length])
        
        return np.array(sequences)
    
    def _get_callbacks(self, validation_data=None) -> List[Callback]:
        """Get training callbacks"""
        callbacks_list = []
        
        # Custom reconstruction callback
        if validation_data is not None:
            callbacks_list.append(ReconstructionCallback(
                validation_data=(validation_data, validation_data),
                n_samples=min(5, len(validation_data))
            ))

        # Training progress callback
        if self.equipment_id and training_tracker:
            callbacks_list.append(TrainingProgressCallback(self.equipment_id))

        # Early stopping
        callbacks_list.append(EarlyStopping(
            monitor='val_loss',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0
        ))
        
        # Reduce learning rate
        callbacks_list.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=0
        ))
        
        # Model checkpoint
        checkpoint_path = get_data_path('models') / f'{self.name}_best.h5'
        callbacks_list.append(ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        ))
        self.best_model_path = checkpoint_path
        
        return callbacks_list
    
    def _calculate_reconstruction_error(self, 
                                       original: np.ndarray,
                                       reconstructed: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error
        
        Args:
            original: Original sequences
            reconstructed: Reconstructed sequences
            
        Returns:
            Error scores per sequence
        """
        # Mean squared error per sequence
        mse = np.mean(np.square(original - reconstructed), axis=(1, 2))
        
        # Could also use other metrics
        # mae = np.mean(np.abs(original - reconstructed), axis=(1, 2))
        
        return mse
    
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on reconstruction error
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores
        """
        # Prepare data
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_sequences = self._create_sequences(X_scaled)
            
            if len(X_sequences) == 0:
                # Not enough data for sequences
                padding = self.config.sequence_length - len(X_scaled)
                X_padded = np.pad(X_scaled, ((padding, 0), (0, 0)), mode='edge')
                X_sequences = X_padded.reshape(1, self.config.sequence_length, -1)
        else:
            # Already sequenced
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X_sequences = X_scaled.reshape(original_shape)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction errors
        errors = self._calculate_reconstruction_error(X_sequences, reconstructions)
        
        # Normalize errors using training statistics
        if self.reconstruction_stats['std'] > 0:
            normalized_errors = (errors - self.reconstruction_stats['mean']) / self.reconstruction_stats['std']
        else:
            normalized_errors = errors
        
        # Convert to anomaly scores
        scores = np.abs(normalized_errors)
        
        return scores
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode sequences to latent representation
        
        Args:
            X: Input sequences
            
        Returns:
            Latent representations
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Prepare data
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_sequences = self._create_sequences(X_scaled)
        else:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X_sequences = X_scaled.reshape(original_shape)
        
        # Encode
        latent = self.encoder.predict(X_sequences, verbose=0)
        
        return latent
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode latent representations to sequences
        
        Args:
            latent: Latent representations
            
        Returns:
            Decoded sequences
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Decode
        reconstructed = self.decoder.predict(latent, verbose=0)
        
        # Inverse scale
        original_shape = reconstructed.shape
        reconstructed_flat = reconstructed.reshape(-1, self.n_features)
        reconstructed_scaled = self.scaler.inverse_transform(reconstructed_flat)
        reconstructed = reconstructed_scaled.reshape(original_shape)
        
        return reconstructed
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input sequences
        
        Args:
            X: Input sequences
            
        Returns:
            Reconstructed sequences
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Prepare data
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_sequences = self._create_sequences(X_scaled)
        else:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_flat)
            X_sequences = X_scaled.reshape(original_shape)
        
        # Reconstruct
        reconstructions = self.autoencoder.predict(X_sequences, verbose=0)
        
        # Inverse scale
        reconstructions_flat = reconstructions.reshape(-1, self.n_features)
        reconstructions_scaled = self.scaler.inverse_transform(reconstructions_flat)
        reconstructions = reconstructions_scaled.reshape(reconstructions.shape)
        
        return reconstructions
    
    def plot_reconstructions(self,
                           X: np.ndarray,
                           n_samples: int = 5,
                           save_path: Optional[Path] = None):
        """
        Plot original vs reconstructed sequences
        
        Args:
            X: Input sequences
            n_samples: Number of samples to plot
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        # Get reconstructions
        if X.ndim == 2:
            X_scaled = self.scaler.transform(X)
            X_sequences = self._create_sequences(X_scaled)[:n_samples]
        else:
            X_sequences = X[:n_samples]
        
        reconstructions = self.autoencoder.predict(X_sequences, verbose=0)
        errors = self._calculate_reconstruction_error(X_sequences, reconstructions)
        
        # Create plots
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Plot original
            axes[i, 0].plot(X_sequences[i, :, 0], 'b-', alpha=0.7, label='Original')
            axes[i, 0].set_title(f'Sample {i+1} - Original')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot reconstruction
            axes[i, 1].plot(X_sequences[i, :, 0], 'b-', alpha=0.7, label='Original')
            axes[i, 1].plot(reconstructions[i, :, 0], 'r-', alpha=0.7, label='Reconstructed')
            axes[i, 1].set_title(f'Reconstruction (Error: {errors[i]:.4f})')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Value')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # Plot difference
            diff = X_sequences[i, :, 0] - reconstructions[i, :, 0]
            axes[i, 2].plot(diff, 'g-', alpha=0.7)
            axes[i, 2].fill_between(range(len(diff)), diff, alpha=0.3)
            axes[i, 2].set_title('Reconstruction Error')
            axes[i, 2].set_xlabel('Time')
            axes[i, 2].set_ylabel('Error')
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Reconstruction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_latent_space(self, 
                         X: np.ndarray,
                         y: Optional[np.ndarray] = None,
                         save_path: Optional[Path] = None):
        """
        Visualize latent space representations
        
        Args:
            X: Input data
            y: Optional labels for coloring
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Get latent representations
        latent = self.encode(X)
        
        # Reduce to 2D if necessary
        if latent.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent)
            explained_var = pca.explained_variance_ratio_
        else:
            latent_2d = latent
            explained_var = [1.0, 0.0]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            # Color by labels
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                c=y, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter, label='Anomaly Label')
        else:
            # Color by reconstruction error
            errors = self.score_samples(X)
            scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                                c=errors, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, label='Reconstruction Error')
        
        plt.xlabel(f'Latent Dim 1 (Var: {explained_var[0]:.2%})')
        plt.ylabel(f'Latent Dim 2 (Var: {explained_var[1]:.2%})')
        plt.title(f'{self.name} - Latent Space Visualization')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving"""
        state = {
            'config': self.config,
            'reconstruction_stats': self.reconstruction_stats,
            'autoencoder_weights': self.autoencoder.get_weights() if self.autoencoder else None,
            'scaler': self.scaler
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state for loading"""
        self.config = state['config']
        self.reconstruction_stats = state['reconstruction_stats']
        self.scaler = state['scaler']
        
        if state['autoencoder_weights'] and self.autoencoder:
            self.autoencoder.set_weights(state['autoencoder_weights'])
    
    def summary(self):
        """Print model summary"""
        if self.autoencoder:
            print("\n=== Autoencoder Architecture ===")
            self.autoencoder.summary()
            print("\n=== Encoder Architecture ===")
            self.encoder.summary()
            print("\n=== Decoder Architecture ===")
            self.decoder.summary()
        else:
            print("Model not built yet")


class DenoisingLSTMAutoencoder(LSTMAutoencoder):
    """Denoising LSTM Autoencoder"""
    
    def __init__(self, noise_factor: float = 0.1, **kwargs):
        super().__init__(name="DenoisingLSTMAutoencoder", **kwargs)
        self.config.noise_factor = noise_factor


class SparseLSTMAutoencoder(LSTMAutoencoder):
    """Sparse LSTM Autoencoder with sparsity constraint"""
    
    def __init__(self, sparsity_weight: float = 0.01, **kwargs):
        super().__init__(name="SparseLSTMAutoencoder", **kwargs)
        self.sparsity_weight = sparsity_weight
    
    def _compile_model(self):
        """Compile with sparsity constraint"""
        super()._compile_model()
        
        # Add L1 regularization to latent layer for sparsity
        if self.config.l1_reg == 0:
            self.config.l1_reg = self.sparsity_weight


class ConvLSTMAutoencoder(LSTMAutoencoder):
    """Convolutional LSTM Autoencoder"""
    
    def __init__(self, **kwargs):
        super().__init__(name="ConvLSTMAutoencoder", **kwargs)
        self.config.use_conv = True


if __name__ == "__main__":
    # Test LSTM Autoencoder
    print("\n" + "="*60)
    print("Testing LSTM Autoencoder")
    print("="*60)
    
    # Create synthetic multivariate time series
    np.random.seed(42)
    n_samples = 2000
    n_features = 3
    
    # Normal data: correlated sine waves
    t = np.linspace(0, 100, n_samples)
    normal_data = np.column_stack([
        np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_samples),
        np.sin(2 * np.pi * 0.1 * t + np.pi/4) + 0.1 * np.random.randn(n_samples),
        np.sin(2 * np.pi * 0.1 * t + np.pi/2) + 0.1 * np.random.randn(n_samples)
    ])
    
    # Add anomalies
    anomaly_indices = [500, 1000, 1500]
    labels = np.zeros(n_samples)
    
    for idx in anomaly_indices:
        # Different anomaly types
        if idx == 500:
            # Amplitude anomaly
            normal_data[idx:idx+50, :] *= 3
        elif idx == 1000:
            # Phase shift anomaly
            normal_data[idx:idx+50, 1] = np.sin(2 * np.pi * 0.5 * t[idx:idx+50])
        else:
            # Random noise anomaly
            normal_data[idx:idx+50, :] = np.random.randn(50, 3) * 2
        
        labels[idx:idx+50] = 1
    
    # Test 1: Basic LSTM Autoencoder
    print("\n1. Testing Basic LSTM Autoencoder...")
    
    config = LSTMAutoencoderConfig(
        encoder_units=[64, 32],
        decoder_units=[32, 64],
        latent_dim=16,
        sequence_length=50,
        epochs=20,
        batch_size=32
    )
    
    autoencoder = LSTMAutoencoder(config=config, contamination=0.05)
    
    # Fit on normal data (first 1500 samples, excluding anomalies)
    print("   Training autoencoder on normal data...")
    train_data = normal_data[:1500]
    train_labels = labels[:1500]
    autoencoder.fit(train_data, train_labels)  # Semi-supervised
    
    # Test detection
    print("\n   Testing on remaining data...")
    test_data = normal_data[1500:]
    test_labels = labels[1500:]
    
    predictions = autoencoder.predict(test_data)
    scores = autoencoder.score_samples(test_data)
    
    print(f"   Detected {np.sum(predictions)} anomalies in test data")
    
    # Test 2: Reconstruction visualization
    print("\n2. Testing Reconstruction...")
    autoencoder.plot_reconstructions(normal_data[:500], n_samples=3)
    
    # Test 3: Latent space visualization
    print("\n3. Testing Latent Space...")
    autoencoder.plot_latent_space(normal_data[:1000], labels[:1000])
    
    # Test 4: Denoising Autoencoder
    print("\n4. Testing Denoising Autoencoder...")
    denoising_ae = DenoisingLSTMAutoencoder(
        noise_factor=0.1,
        config=config,
        contamination=0.05
    )
    denoising_ae.fit(train_data, train_labels)
    denoising_predictions = denoising_ae.predict(test_data)
    print(f"   Denoising AE detected {np.sum(denoising_predictions)} anomalies")
    
    # Test 5: Encoding and decoding
    print("\n5. Testing Encoding/Decoding...")
    sample_data = normal_data[:100]
    
    # Encode to latent space
    latent_repr = autoencoder.encode(sample_data)
    print(f"   Encoded shape: {latent_repr.shape}")
    
    # Decode back
    decoded = autoencoder.decode(latent_repr)
    print(f"   Decoded shape: {decoded.shape}")
    
    # Test 6: Model summary
    print("\n6. Model Architecture:")
    autoencoder.summary()
    
    print("\n" + "="*60)
    print("LSTM Autoencoder test complete")
    print("="*60)
