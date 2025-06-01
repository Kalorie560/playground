"""
Neural network model architecture for Spaceship Titanic competition.
Implements a flexible PyTorch neural network with configurable layers and regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceshipClassifier(nn.Module):
    """
    Neural network classifier for Spaceship Titanic competition.
    
    Features:
    - Configurable hidden layers
    - Dropout regularization
    - Batch normalization
    - Flexible activation functions
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128, 64],
                 dropout_rates: List[float] = [0.3, 0.4, 0.5],
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 output_size: int = 1):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rates: List of dropout rates for each layer
            activation: Activation function ('relu', 'leaky_relu', 'elu', 'gelu')
            use_batch_norm: Whether to use batch normalization
            output_size: Number of output classes (1 for binary classification)
        """
        super(SpaceshipClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.output_size = output_size
        
        # Ensure dropout rates match hidden layers
        if len(dropout_rates) != len(hidden_sizes):
            dropout_rates = dropout_rates * len(hidden_sizes)
            dropout_rates = dropout_rates[:len(hidden_sizes)]
            self.dropout_rates = dropout_rates
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.batch_norms.append(nn.Identity())
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def _get_activation(self):
        """Get activation function based on string name."""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        return activations.get(self.activation, F.relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        activation_fn = self._get_activation()
        
        # Forward through hidden layers
        for i, (layer, batch_norm, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            
            if self.use_batch_norm:
                x = batch_norm(x)
            
            x = activation_fn(x)
            x = dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout_rates': self.dropout_rates,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'output_size': self.output_size
        }


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in monitored quantity to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: PyTorch model
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
        
        return self.early_stop


class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    @staticmethod
    def save_model(model: nn.Module, filepath: str, additional_info: Dict[str, Any] = None):
        """Save model with additional information."""
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
        }
        
        if additional_info:
            save_dict.update(additional_info)
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(model: nn.Module, filepath: str) -> Dict[str, Any]:
        """Load model and return additional information."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        additional_info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        logger.info(f"Model loaded from {filepath}")
        
        return additional_info


def create_model(input_size: int, config: Dict[str, Any]) -> SpaceshipClassifier:
    """
    Create model from configuration.
    
    Args:
        input_size: Number of input features
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    model = SpaceshipClassifier(
        input_size=input_size,
        hidden_sizes=model_config.get('hidden_sizes', [256, 128, 64]),
        dropout_rates=model_config.get('dropout_rates', [0.3, 0.4, 0.5]),
        activation=model_config.get('activation', 'relu'),
        use_batch_norm=model_config.get('use_batch_norm', True),
        output_size=model_config.get('output_size', 1)
    )
    
    return model


def main():
    """Example usage of the model."""
    # Example configuration
    config = {
        'model': {
            'hidden_sizes': [256, 128, 64],
            'dropout_rates': [0.3, 0.4, 0.5],
            'activation': 'relu',
            'use_batch_norm': True,
            'output_size': 1
        }
    }
    
    # Create model
    input_size = 20  # Example input size
    model = create_model(input_size, config)
    
    # Print model information
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()