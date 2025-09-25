"""
LSTM Variational Autoencoder (VAE) Module for Anomaly Detection
Probabilistic anomaly detection using LSTM-based VAE
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, 
    Input, RepeatVector, TimeDistributed, Bidirectional,
    Lambda, Layer, Concatenate, GRU, Flatten, Reshape,
    Conv1D, MaxPooling1D, UpSampling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, Callback
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import Loss
from tensorflow.keras.regularizers import L1L2
from sklearn.preprocessing import StandardScaler

# Try to import tensorflow_probability, fallback gracefully
try:
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    logger.warning("TensorFlow Probability not available, some advanced features disabled")

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.anomaly_detection.base_detector import (
    BaseAnomalyDetector, ThresholdStrategy, PercentileThreshold
)
from config.settings import settings, get_config, get_data_path

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')


@dataclass
class LSTMVAEConfig:
    """Configuration for LSTM VAE"""
    # Encoder architecture
    encoder_units: List[int] = None
    decoder_units: List[int] = None
    latent_dim: int = 20
    
    # VAE specific
    kl_weight: float = 0.1  # Weight for KL divergence loss
    reconstruction_weight: float = 0.9  # Weight for reconstruction loss
    use_mmd: bool = False  # Use Maximum Mean Discrepancy instead of KL
    beta: float = 1.0  # Beta-VAE parameter for disentanglement
    
    # Model type
    use_bidirectional: bool = False
    use_conv: bool = False
    activation: str = 'tanh'
    recurrent_activation: str = 'sigmoid'
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    warmup_epochs: int = 10  # Epochs for KL annealing
    
    # Regularization
    dropout_rate: float = 0.2
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    use_batch_norm: bool = True
    
    # Sequence parameters
    sequence_length: int = 100
    
    # Anomaly detection
    use_reconstruction: bool = True
    use_likelihood: bool = True
    likelihood_samples: int = 10  # Number of samples for likelihood estimation
    
    def __post_init__(self):
        if self.encoder_units is None:
            self.encoder_units = [128, 64]
        if self.decoder_units is None:
            self.decoder_units = self.encoder_units[::-1]


class SamplingLayer(Layer):
    """Custom sampling layer using reparameterization trick"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample epsilon from standard normal
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Reparameterization trick: z = μ + σ * ε
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]


class VAELoss(Loss):
    """Custom VAE loss combining reconstruction and KL divergence"""
    
    def __init__(self, 
                 original_dim: int,
                 kl_weight: float = 0.1,
                 reconstruction_weight: float = 0.9,
                 beta: float = 1.0,
                 warmup_epochs: int = 10,
                 name: str = 'vae_loss'):
        super().__init__(name=name)
        self.original_dim = original_dim
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.current_epoch = K.variable(0)
    
    def call(self, y_true, y_pred):
        # Split predictions into reconstruction and latent parameters
        # Assumes y_pred contains [reconstruction, z_mean, z_log_var] concatenated
        # This is handled differently - we'll use model outputs directly
        
        # Reconstruction loss (MSE)
        reconstruction_loss = K.mean(K.square(y_true - y_pred))
        
        # This will be added separately in the model
        return reconstruction_loss


class VAECallback(Callback):
    """Custom callback for VAE training monitoring"""
    
    def __init__(self, beta_anneal_epochs: int = 10):
        super().__init__()
        self.beta_anneal_epochs = beta_anneal_epochs
        self.history = {
            'loss': [], 'val_loss': [],
            'reconstruction_loss': [], 'kl_loss': [],
            'beta': []
        }
    
    def on_epoch_begin(self, epoch, logs=None):
        # Implement KL annealing (warmup)
        if epoch < self.beta_anneal_epochs:
            beta = epoch / self.beta_anneal_epochs
        else:
            beta = 1.0
        
        # Update beta if model has this attribute
        if hasattr(self.model, 'beta'):
            K.set_value(self.model.beta, beta)
        
        self.history['beta'].append(beta)
    
    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: loss={logs.get('loss'):.4f}, "
                       f"val_loss={logs.get('val_loss'):.4f}")


class LSTMVAE(BaseAnomalyDetector):
    """
    LSTM Variational Autoencoder for anomaly detection
    Detects anomalies using probabilistic reconstruction
    """
    
    def __init__(self,
                 name: str = "LSTMVAE",
                 config: Optional[LSTMVAEConfig] = None,
                 threshold_strategy: Optional[ThresholdStrategy] = None,
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize LSTM VAE
        
        Args:
            name: Detector name
            config: VAE configuration
            threshold_strategy: Threshold determination strategy
            contamination: Expected anomaly ratio
            random_state: Random seed
        """
        super().__init__(
            name=name,
            threshold_strategy=threshold_strategy or PercentileThreshold(100 * (1 - contamination)),
            contamination=contamination,
            window_size=config.sequence_length if config else 100,
            random_state=random_state
        )
        
        self.config = config or LSTMVAEConfig()
        
        # Model components
        self.encoder: Optional[Model] = None
        self.decoder: Optional[Model] = None
        self.vae: Optional[Model] = None
        self.scaler: Optional[StandardScaler] = None
        
        # VAE specific
        self.z_mean_layer = None
        self.z_log_var_layer = None
        self.sampling_layer = None
        
        # Training state
        self.training_history = None
        self.best_model_path = None
        
        # Loss tracking
        self.reconstruction_loss = None
        self.kl_loss = None
        
        # Statistics
        self.latent_stats = {'mean': None, 'std': None}
        self.reconstruction_stats = {'mean': 0, 'std': 1}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info(f"LSTMVAE initialized with latent_dim={self.config.latent_dim}")
    
    def _build_model(self):
        """Build LSTM VAE architecture"""
        # Input layer
        encoder_inputs = Input(shape=(self.config.sequence_length, self.n_features), name='encoder_input')
        
        # Build encoder
        z_mean, z_log_var, encoder_state = self._build_encoder(encoder_inputs)
        
        # Sampling layer
        z = SamplingLayer(name='z_sampling')([z_mean, z_log_var])
        
        # Build decoder
        decoder_outputs = self._build_decoder(z)
        
        # Create models
        self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # For standalone decoder
        latent_inputs = Input(shape=(self.config.latent_dim,), name='latent_input')
        _decoder_outputs = self._build_decoder(latent_inputs)
        self.decoder = Model(latent_inputs, _decoder_outputs, name='decoder')
        
        # Full VAE model
        self.vae = Model(encoder_inputs, decoder_outputs, name='vae')
        self.model = self.vae  # For compatibility with base class
        
        # Add VAE loss
        self._add_vae_loss(z_mean, z_log_var)
        
        # Compile model
        self._compile_model()
        
        logger.info(f"Built LSTM VAE with {self.vae.count_params()} parameters")
    
    def _build_encoder(self, inputs):
        """Build encoder network"""
        x = inputs
        
        # Optional convolutional layers
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
        
        # Latent variable parameters
        z_mean = Dense(self.config.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.config.latent_dim, name='z_log_var')(x)
        
        return z_mean, z_log_var, x
    
    def _build_decoder(self, z):
        """Build decoder network"""
        x = z
        
        # Expand latent vector to sequence
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
    
    def _add_vae_loss(self, z_mean, z_log_var):
        """Add VAE loss to model"""
        # KL divergence loss
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1
        )
        kl_loss = K.mean(kl_loss)
        
        # Add KL loss to model
        self.kl_loss = kl_loss
        self.vae.add_loss(self.config.kl_weight * self.config.beta * kl_loss)
        
        # Store for monitoring
        self.vae.add_metric(kl_loss, name='kl_loss')
    
    def _compile_model(self):
        """Compile the VAE model"""
        # Select optimizer
        if self.config.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = Adam(learning_rate=self.config.learning_rate)
        
        # Compile with reconstruction loss (MSE)
        self.vae.compile(
            optimizer=optimizer,
            loss='mse',
            loss_weights=[self.config.reconstruction_weight]
        )
    
    def _fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit LSTM VAE
        
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
        
        # Train VAE
        logger.info(f"Training LSTM VAE on {len(X_train)} sequences")
        
        self.training_history = self.vae.fit(
            X_train, X_train,  # Input and target are the same
            validation_data=(X_val, X_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Calculate statistics on normal data
        self._calculate_statistics(X_train)
        
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
        
        # Custom VAE callback
        callbacks_list.append(VAECallback(beta_anneal_epochs=self.config.warmup_epochs))
        
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
    
    def _calculate_statistics(self, X_train: np.ndarray):
        """Calculate statistics on normal training data"""
        # Get latent representations
        z_mean, z_log_var, z = self.encoder.predict(X_train, verbose=0)
        
        # Store latent statistics
        self.latent_stats['mean'] = np.mean(z_mean, axis=0)
        self.latent_stats['std'] = np.std(z_mean, axis=0)
        
        # Calculate reconstruction errors
        reconstructions = self.vae.predict(X_train, verbose=0)
        errors = self._calculate_reconstruction_error(X_train, reconstructions)
        
        self.reconstruction_stats['mean'] = np.mean(errors)
        self.reconstruction_stats['std'] = np.std(errors)
    
    def _calculate_reconstruction_error(self,
                                       original: np.ndarray,
                                       reconstructed: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error"""
        # Mean squared error per sequence
        mse = np.mean(np.square(original - reconstructed), axis=(1, 2))
        return mse
    
    def _calculate_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate likelihood of samples under the learned distribution
        
        Args:
            X: Input sequences
            
        Returns:
            Negative log-likelihood scores
        """
        # Get latent parameters
        z_mean, z_log_var, _ = self.encoder.predict(X, verbose=0)
        
        # Calculate KL divergence from prior
        kl_divergence = -0.5 * np.sum(
            1 + z_log_var - np.square(z_mean) - np.exp(z_log_var),
            axis=-1
        )
        
        # Calculate reconstruction likelihood
        reconstructions = self.vae.predict(X, verbose=0)
        reconstruction_error = self._calculate_reconstruction_error(X, reconstructions)
        
        # Estimate negative log-likelihood (lower bound)
        # ELBO = reconstruction_error + kl_divergence
        nll = reconstruction_error + self.config.kl_weight * kl_divergence
        
        return nll
    
    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores
        
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
        
        # Calculate scores based on configuration
        scores = np.zeros(len(X_sequences))
        
        if self.config.use_reconstruction:
            # Reconstruction error
            reconstructions = self.vae.predict(X_sequences, verbose=0)
            rec_errors = self._calculate_reconstruction_error(X_sequences, reconstructions)
            
            # Normalize
            if self.reconstruction_stats['std'] > 0:
                rec_scores = (rec_errors - self.reconstruction_stats['mean']) / self.reconstruction_stats['std']
            else:
                rec_scores = rec_errors
            
            scores += rec_scores
        
        if self.config.use_likelihood:
            # Likelihood-based scores
            nll_scores = self._calculate_likelihood(X_sequences)
            
            # Normalize (higher NLL = more anomalous)
            nll_scores = (nll_scores - np.min(nll_scores)) / (np.max(nll_scores) - np.min(nll_scores) + 1e-10)
            
            scores += nll_scores
        
        # Average if using both
        if self.config.use_reconstruction and self.config.use_likelihood:
            scores /= 2
        
        return np.abs(scores)
    
    def encode(self, X: np.ndarray, return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Encode sequences to latent space
        
        Args:
            X: Input sequences
            return_params: Whether to return mean and log variance
            
        Returns:
            Latent representations or tuple of (z_mean, z_log_var, z)
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
        z_mean, z_log_var, z = self.encoder.predict(X_sequences, verbose=0)
        
        if return_params:
            return z_mean, z_log_var, z
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Decode latent representations to sequences
        
        Args:
            z: Latent representations
            
        Returns:
            Decoded sequences
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        # Decode
        reconstructed = self.decoder.predict(z, verbose=0)
        
        # Inverse scale
        original_shape = reconstructed.shape
        reconstructed_flat = reconstructed.reshape(-1, self.n_features)
        reconstructed_scaled = self.scaler.inverse_transform(reconstructed_flat)
        reconstructed = reconstructed_scaled.reshape(original_shape)
        
        return reconstructed
    
    def generate(self, n_samples: int = 1, z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate new sequences from latent space
        
        Args:
            n_samples: Number of samples to generate
            z: Optional latent vectors (if None, sample from prior)
            
        Returns:
            Generated sequences
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if z is None:
            # Sample from prior (standard normal)
            z = np.random.normal(0, 1, size=(n_samples, self.config.latent_dim))
        
        # Generate sequences
        generated = self.decoder.predict(z, verbose=0)
        
        # Inverse scale
        original_shape = generated.shape
        generated_flat = generated.reshape(-1, self.n_features)
        generated_scaled = self.scaler.inverse_transform(generated_flat)
        generated = generated_scaled.reshape(original_shape)
        
        return generated
    
    def plot_latent_distribution(self, 
                                X: np.ndarray,
                                y: Optional[np.ndarray] = None,
                                save_path: Optional[Path] = None):
        """
        Plot latent space distribution
        
        Args:
            X: Input data
            y: Optional labels
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Get latent representations
        z_mean, z_log_var, z = self.encode(X, return_params=True)
        
        # Reduce to 2D if necessary
        if z.shape[1] > 2:
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z)
            z_mean_2d = pca.transform(z_mean)
        else:
            z_2d = z
            z_mean_2d = z_mean
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Sampled latent space
        if y is not None:
            scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], 
                                      c=y, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter1, ax=axes[0], label='Anomaly')
        else:
            axes[0].scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6)
        axes[0].set_title('Sampled Latent Space (z)')
        axes[0].set_xlabel('Latent Dim 1')
        axes[0].set_ylabel('Latent Dim 2')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Mean latent space
        if y is not None:
            scatter2 = axes[1].scatter(z_mean_2d[:, 0], z_mean_2d[:, 1],
                                      c=y, cmap='coolwarm', alpha=0.6)
            plt.colorbar(scatter2, ax=axes[1], label='Anomaly')
        else:
            axes[1].scatter(z_mean_2d[:, 0], z_mean_2d[:, 1], alpha=0.6)
        axes[1].set_title('Mean Latent Space (μ)')
        axes[1].set_xlabel('Latent Dim 1')
        axes[1].set_ylabel('Latent Dim 2')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Variance heatmap
        variance = np.exp(z_log_var)
        im = axes[2].imshow(variance[:min(100, len(variance))].T, 
                           aspect='auto', cmap='viridis')
        axes[2].set_title('Latent Variance (First 100 samples)')
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=axes[2], label='Variance')
        
        plt.suptitle(f'{self.name} - Latent Space Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_generation_quality(self, 
                              n_samples: int = 5,
                              save_path: Optional[Path] = None):
        """
        Plot generated samples quality
        
        Args:
            n_samples: Number of samples to generate
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        # Generate samples
        generated = self.generate(n_samples)
        
        # Create plots
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Plot generated sequence (first feature)
            axes[i, 0].plot(generated[i, :, 0], 'b-', alpha=0.7)
            axes[i, 0].set_title(f'Generated Sample {i+1}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Value')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot all features if multivariate
            if self.n_features > 1:
                for j in range(min(3, self.n_features)):  # Plot first 3 features
                    axes[i, 1].plot(generated[i, :, j], alpha=0.7, label=f'Feature {j+1}')
                axes[i, 1].set_title('All Features')
                axes[i, 1].set_xlabel('Time')
                axes[i, 1].set_ylabel('Value')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
            else:
                # Plot distribution
                axes[i, 1].hist(generated[i, :, 0], bins=30, alpha=0.7, edgecolor='black')
                axes[i, 1].set_title('Value Distribution')
                axes[i, 1].set_xlabel('Value')
                axes[i, 1].set_ylabel('Frequency')
                axes[i, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.name} - Generated Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get model state for saving"""
        state = {
            'config': self.config,
            'latent_stats': self.latent_stats,
            'reconstruction_stats': self.reconstruction_stats,
            'vae_weights': self.vae.get_weights() if self.vae else None,
            'scaler': self.scaler
        }
        return state
    
    def _set_model_state(self, state: Dict[str, Any]):
        """Set model state for loading"""
        self.config = state['config']
        self.latent_stats = state['latent_stats']
        self.reconstruction_stats = state['reconstruction_stats']
        self.scaler = state['scaler']
        
        if state['vae_weights'] and self.vae:
            self.vae.set_weights(state['vae_weights'])
    
    def summary(self):
        """Print model summary"""
        if self.vae:
            print("\n=== VAE Architecture ===")
            self.vae.summary()
            print("\n=== Encoder Architecture ===")
            self.encoder.summary()
            print("\n=== Decoder Architecture ===")
            self.decoder.summary()
        else:
            print("Model not built yet")


class BetaLSTMVAE(LSTMVAE):
    """Beta-VAE with controllable disentanglement"""
    
    def __init__(self, beta: float = 4.0, **kwargs):
        super().__init__(name="BetaLSTMVAE", **kwargs)
        self.config.beta = beta


class MMDLSTMVAE(LSTMVAE):
    """VAE using Maximum Mean Discrepancy instead of KL divergence"""
    
    def __init__(self, **kwargs):
        super().__init__(name="MMDLSTMVAE", **kwargs)
        self.config.use_mmd = True
    
    def _add_vae_loss(self, z_mean, z_log_var):
        """Add MMD loss instead of KL divergence"""
        # Sample from latent distribution
        z = SamplingLayer()([z_mean, z_log_var])
        
        # Compute MMD loss
        def compute_mmd(z):
            # Sample from prior
            z_prior = K.random_normal(shape=K.shape(z))
            
            # Compute MMD using RBF kernel
            def compute_kernel(x, y):
                x_size = K.shape(x)[0]
                y_size = K.shape(y)[0]
                dim = K.shape(x)[1]
                
                x = K.expand_dims(x, 1)
                y = K.expand_dims(y, 0)
                
                distances = K.sum(K.square(x - y), axis=2)
                kernel = K.exp(-distances / (2.0 * K.cast(dim, 'float32')))
                
                return kernel
            
            xx = K.mean(compute_kernel(z, z))
            yy = K.mean(compute_kernel(z_prior, z_prior))
            xy = K.mean(compute_kernel(z, z_prior))
            
            mmd = xx + yy - 2 * xy
            return mmd
        
        mmd_loss = Lambda(compute_mmd)(z)
        mmd_loss = K.mean(mmd_loss)
        
        # Add MMD loss to model
        self.vae.add_loss(self.config.kl_weight * mmd_loss)
        self.vae.add_metric(mmd_loss, name='mmd_loss')


if __name__ == "__main__":
    # Test LSTM VAE
    print("\n" + "="*60)
    print("Testing LSTM Variational Autoencoder")
    print("="*60)
    
    # Create synthetic multivariate time series
    np.random.seed(42)
    n_samples = 2000
    n_features = 3
    
    # Normal data: correlated patterns
    t = np.linspace(0, 100, n_samples)
    normal_data = np.column_stack([
        np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_samples),
        np.cos(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_samples),
        np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(n_samples)
    ])
    
    # Add anomalies
    anomaly_indices = [500, 1000, 1500]
    labels = np.zeros(n_samples)
    
    for idx in anomaly_indices:
        if idx == 500:
            # Distribution shift anomaly
            normal_data[idx:idx+50, :] = np.random.randn(50, 3) * 3
        elif idx == 1000:
            # Pattern break anomaly
            normal_data[idx:idx+50, :] = 0
        else:
            # Extreme values
            normal_data[idx:idx+50, :] = normal_data[idx:idx+50, :] * 5
        
        labels[idx:idx+50] = 1
    
    # Test 1: Basic LSTM VAE
    print("\n1. Testing Basic LSTM VAE...")
    
    config = LSTMVAEConfig(
        encoder_units=[64, 32],
        decoder_units=[32, 64],
        latent_dim=10,
        sequence_length=50,
        kl_weight=0.1,
        epochs=20,
        batch_size=32
    )
    
    vae = LSTMVAE(config=config, contamination=0.05)
    
    # Fit on normal data
    print("   Training VAE on normal data...")
    train_data = normal_data[:1500]
    train_labels = labels[:1500]
    vae.fit(train_data, train_labels)  # Semi-supervised
    
    # Test detection
    print("\n   Testing on remaining data...")
    test_data = normal_data[1500:]
    test_labels = labels[1500:]
    
    predictions = vae.predict(test_data)
    scores = vae.score_samples(test_data)
    
    print(f"   Detected {np.sum(predictions)} anomalies in test data")
    
    # Test 2: Latent space visualization
    print("\n2. Testing Latent Space Visualization...")
    vae.plot_latent_distribution(normal_data[:1000], labels[:1000])
    
    # Test 3: Generation
    print("\n3. Testing Generation...")
    generated = vae.generate(n_samples=3)
    print(f"   Generated shape: {generated.shape}")
    vae.plot_generation_quality(n_samples=3)
    
    # Test 4: Beta-VAE
    print("\n4. Testing Beta-VAE...")
    beta_vae = BetaLSTMVAE(beta=4.0, config=config, contamination=0.05)
    beta_vae.fit(train_data, train_labels)
    beta_predictions = beta_vae.predict(test_data)
    print(f"   Beta-VAE detected {np.sum(beta_predictions)} anomalies")
    
    # Test 5: Encoding and likelihood
    print("\n5. Testing Encoding and Likelihood...")
    sample_data = normal_data[:100]
    
    # Get latent parameters
    z_mean, z_log_var, z = vae.encode(sample_data, return_params=True)
    print(f"   Latent mean shape: {z_mean.shape}")
    print(f"   Latent log variance shape: {z_log_var.shape}")
    print(f"   Sampled latent shape: {z.shape}")
    
    # Test 6: Model summary
    print("\n6. Model Architecture:")
    vae.summary()
    
    print("\n" + "="*60)
    print("LSTM VAE test complete")
    print("="*60)
