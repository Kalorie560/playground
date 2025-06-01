"""
Prediction module for Spaceship Titanic competition.
Generates predictions for test set and creates submission file in Kaggle format.
"""

import torch
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, Optional
import logging
import os

# Import our modules
from data_preprocessing import SpaceshipDataProcessor
from model import SpaceshipClassifier, create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpaceshipPredictor:
    """Predictor class for generating test set predictions."""
    
    def __init__(self, config_path: str = "config.yaml", model_path: str = "best_model.pth"):
        """Initialize predictor with configuration and model."""
        self.config = self.load_config(config_path)
        self.model_path = model_path
        self.device = self.setup_device()
        
        # Initialize components
        self.model = None
        self.data_processor = SpaceshipDataProcessor()
        self.feature_names = None
        
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
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file {self.model_path} not found")
            raise FileNotFoundError(f"Model file {self.model_path} not found")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config
        
        # Get feature names and input size
        if 'feature_names' in checkpoint:
            self.feature_names = checkpoint['feature_names']
            input_size = len(self.feature_names)
        else:
            # Fallback - will be determined during data processing
            input_size = None
        
        # Create model
        if input_size:
            self.model = create_model(input_size, model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        
        # Log model information
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            logger.info("Model validation metrics:")
            for metric, value in val_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    def prepare_test_data(self) -> torch.Tensor:
        """Prepare test data for prediction."""
        logger.info("Loading and preprocessing test data...")
        
        # Load data
        train_df, test_df = self.data_processor.load_data(
            self.config['data']['train_path'],
            self.config['data']['test_path']
        )
        
        # We need to process training data first to fit transformers
        processed_data = self.data_processor.preprocess(train_df, test_df)
        
        # Get test features
        X_test = processed_data['X_test']
        
        # If model wasn't loaded with input size, create it now
        if self.model is None:
            input_size = X_test.shape[1]
            self.model = create_model(input_size, self.config)
            
            # Load model weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        # Store feature names if not already stored
        if self.feature_names is None:
            self.feature_names = processed_data['feature_names']
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
        
        logger.info(f"Test data prepared. Shape: {X_test_tensor.shape}")
        return X_test_tensor, test_df
    
    def predict(self, X_test: torch.Tensor) -> np.ndarray:
        """Generate predictions for test data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info("Generating predictions...")
        
        self.model.eval()
        predictions = []
        
        # Predict in batches to handle large test sets
        batch_size = self.config['training']['batch_size']
        num_batches = (len(X_test) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_test))
                batch = X_test[start_idx:end_idx]
                
                # Get model output
                output = self.model(batch)
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                predictions.extend(probs)
        
        predictions = np.array(predictions)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray, 
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """Create submission file in Kaggle format."""
        if output_path is None:
            output_path = self.config['output']['submission_file']
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > 0.5).astype(bool)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'PassengerId': test_df['PassengerId'] if 'PassengerId' in test_df.columns else range(len(predictions)),
            'Transported': binary_predictions
        })
        
        # Save submission file
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission file saved to {output_path}")
        
        # Log prediction statistics
        logger.info(f"Prediction statistics:")
        logger.info(f"  Total predictions: {len(predictions)}")
        logger.info(f"  Transported (True): {binary_predictions.sum()} ({binary_predictions.mean():.2%})")
        logger.info(f"  Not Transported (False): {(~binary_predictions).sum()} ({(~binary_predictions).mean():.2%})")
        logger.info(f"  Average probability: {predictions.mean():.4f}")
        logger.info(f"  Probability std: {predictions.std():.4f}")
        
        return submission
    
    def save_detailed_predictions(self, test_df: pd.DataFrame, predictions: np.ndarray, 
                                output_path: str = "detailed_predictions.csv"):
        """Save detailed predictions including probabilities."""
        detailed_df = test_df.copy()
        detailed_df['Transported_Probability'] = predictions
        detailed_df['Transported_Prediction'] = (predictions > 0.5).astype(bool)
        
        detailed_df.to_csv(output_path, index=False)
        logger.info(f"Detailed predictions saved to {output_path}")
    
    def run_prediction_pipeline(self) -> pd.DataFrame:
        """Run the complete prediction pipeline."""
        try:
            # Load model
            self.load_model()
            
            # Prepare test data
            X_test, test_df = self.prepare_test_data()
            
            # Generate predictions
            predictions = self.predict(X_test)
            
            # Create submission
            submission = self.create_submission(test_df, predictions)
            
            # Save detailed predictions if requested
            if self.config['output'].get('save_predictions', False):
                self.save_detailed_predictions(test_df, predictions)
            
            return submission
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise


def main():
    """Main prediction function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions for Spaceship Titanic test set')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='best_model.pth', 
                       help='Path to trained model file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output path for submission file')
    
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = SpaceshipPredictor(args.config, args.model)
        
        # Run prediction pipeline
        submission = predictor.run_prediction_pipeline()
        
        print("\nPrediction completed successfully!")
        print(f"Submission file created with {len(submission)} predictions")
        print("\nFirst few predictions:")
        print(submission.head())
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main()