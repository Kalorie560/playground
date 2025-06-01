"""
Training pipeline for Spaceship Titanic deep learning solution.
Includes ClearML integration, early stopping, learning rate scheduling, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import yaml
from typing import Dict, Any, Tuple
import logging
from tqdm import tqdm
import os
import warnings

# Import our modules
from data_preprocessing import SpaceshipDataProcessor
from model import SpaceshipClassifier, EarlyStopping, ModelUtils, create_model

# ClearML imports with error handling
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    print("ClearML not available. Continuing without experiment tracking.")
    CLEARML_AVAILABLE = False
    Task = None
    Logger = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceshipTrainer:
    """Main trainer class for Spaceship Titanic competition."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.device = self.setup_device()
        self.setup_reproducibility()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None
        self.data_processor = SpaceshipDataProcessor()
        
        # ClearML setup
        self.task = None
        self.clearml_logger = None
        self.setup_clearml()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
    
    def setup_device(self) -> torch.device:
        """Setup computing device."""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)
        
        logger.info(f"Using device: {device}")
        return device
    
    def setup_reproducibility(self):
        """Setup reproducibility settings."""
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
        
        if self.config['reproducibility']['benchmark']:
            torch.backends.cudnn.benchmark = True
        
        logger.info("Reproducibility settings configured")
    
    def setup_clearml(self):
        """Setup ClearML experiment tracking."""
        if not CLEARML_AVAILABLE:
            return
        
        try:
            clearml_config = self.config['clearml']
            self.task = Task.init(
                project_name=clearml_config['project_name'],
                task_name=clearml_config['task_name'],
                tags=clearml_config.get('tags', []),
                auto_connect_frameworks=clearml_config.get('auto_connect_frameworks', True),
                auto_connect_arg_parser=clearml_config.get('auto_connect_arg_parser', True)
            )
            
            # Connect configuration
            self.task.connect(self.config)
            
            # Get logger
            self.clearml_logger = self.task.get_logger()
            
            logger.info("ClearML experiment tracking initialized")
        except Exception as e:
            logger.warning(f"ClearML setup failed: {e}. Continuing without experiment tracking.")
            self.task = None
            self.clearml_logger = None
    
    def prepare_data(self) -> Dict[str, torch.Tensor]:
        """Prepare data for training."""
        logger.info("Loading and preprocessing data...")
        
        # Load data
        train_df, test_df = self.data_processor.load_data(
            self.config['data']['train_path'],
            self.config['data']['test_path']
        )
        
        # Preprocess data
        processed_data = self.data_processor.preprocess(
            train_df, test_df, 
            validation_split=self.config['data']['validation_split']
        )
        
        # Convert to tensors
        data_tensors = {}
        for key, data in processed_data.items():
            if isinstance(data, pd.DataFrame):
                data_tensors[key] = torch.FloatTensor(data.values)
            elif isinstance(data, pd.Series):
                data_tensors[key] = torch.FloatTensor(data.values)
            else:
                data_tensors[key] = data
        
        logger.info(f"Data prepared. Features: {len(processed_data['feature_names'])}")
        
        # Log data info to ClearML
        if self.clearml_logger:
            self.clearml_logger.report_single_value("data/num_features", len(processed_data['feature_names']))
            self.clearml_logger.report_single_value("data/train_size", len(data_tensors['X_train']))
            self.clearml_logger.report_single_value("data/val_size", len(data_tensors['X_val']))
            if 'X_test' in data_tensors:
                self.clearml_logger.report_single_value("data/test_size", len(data_tensors['X_test']))
        
        return data_tensors
    
    def create_dataloaders(self, data: Dict[str, torch.Tensor]) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders."""
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        val_dataset = TensorDataset(data['X_val'], data['y_val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        return train_loader, val_loader
    
    def setup_model(self, input_size: int):
        """Setup model, optimizer, scheduler, and early stopping."""
        # Create model
        self.model = create_model(input_size, self.config)
        self.model.to(self.device)
        
        # Setup optimizer
        optimizer_config = self.config['training']
        if optimizer_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        elif optimizer_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay'],
                momentum=0.9
            )
        elif optimizer_config['optimizer'].lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=optimizer_config['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        
        # Setup scheduler
        scheduler_config = optimizer_config['scheduler']
        if scheduler_config['type'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=optimizer_config['epochs']
            )
        elif scheduler_config['type'] == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=scheduler_config['gamma']
            )
        elif scheduler_config['type'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor']
            )
        
        # Setup early stopping
        early_stop_config = optimizer_config['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            restore_best_weights=early_stop_config['restore_best_weights']
        )
        
        # Log model info
        model_info = self.model.get_model_info()
        logger.info(f"Model created with {model_info['total_parameters']} parameters")
        
        if self.clearml_logger:
            for key, value in model_info.items():
                self.clearml_logger.report_single_value(f"model/{key}", value)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model and return metrics."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device).float().unsqueeze(1)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Collect predictions and targets
                predictions = torch.sigmoid(output).cpu().numpy().flatten()
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy().flatten())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        binary_predictions = (all_predictions > 0.5).astype(int)
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy_score(all_targets, binary_predictions),
            'precision': precision_score(all_targets, binary_predictions, zero_division=0),
            'recall': recall_score(all_targets, binary_predictions, zero_division=0),
            'f1': f1_score(all_targets, binary_predictions, zero_division=0),
            'auc': roc_auc_score(all_targets, all_predictions)
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Prepare data
        data = self.prepare_data()
        train_loader, val_loader = self.create_dataloaders(data)
        
        # Setup model
        input_size = data['X_train'].shape[1]
        self.setup_model(input_size)
        
        # Training loop
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, LR: {current_lr:.2e}"
            )
            
            # ClearML logging
            if self.clearml_logger:
                self.clearml_logger.report_scalar("loss", "train", iteration=epoch, value=train_loss)
                for metric_name, metric_value in val_metrics.items():
                    self.clearml_logger.report_scalar("validation", metric_name, iteration=epoch, value=metric_value)
                self.clearml_logger.report_scalar("learning_rate", "lr", iteration=epoch, value=current_lr)
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                ModelUtils.save_model(
                    self.model,
                    self.config['output']['model_save_path'],
                    {
                        'epoch': epoch,
                        'val_metrics': val_metrics,
                        'config': self.config,
                        'feature_names': data['feature_names']
                    }
                )
            
            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Final evaluation
        final_metrics = self.validate(val_loader)
        logger.info("Training completed!")
        logger.info("Final validation metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Log final metrics to ClearML
        if self.clearml_logger:
            for metric_name, metric_value in final_metrics.items():
                self.clearml_logger.report_single_value(f"final/{metric_name}", metric_value)
        
        return training_history, final_metrics


def main():
    """Main training function."""
    try:
        trainer = SpaceshipTrainer("config.yaml")
        history, metrics = trainer.train()
        
        print("\nTraining completed successfully!")
        print("Final metrics:", metrics)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()