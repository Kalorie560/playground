# Spaceship Titanic Deep Learning Solution

A comprehensive deep learning solution for the Kaggle Spaceship Titanic competition, featuring modular architecture, ClearML experiment tracking, and configurable hyperparameters.

## Features

- **Modular Design**: Well-organized code structure with separate modules for data processing, model architecture, training, and prediction
- **Deep Learning**: PyTorch-based neural network with configurable architecture
- **Experiment Tracking**: ClearML integration for comprehensive experiment logging
- **Configuration Management**: YAML-based configuration system for easy hyperparameter tuning
- **Feature Engineering**: Advanced preprocessing including cabin information extraction, family size features, and spending patterns
- **Training Pipeline**: Complete training loop with early stopping, learning rate scheduling, and comprehensive metrics
- **Reproducibility**: Seed management and deterministic training for reproducible results

## Project Structure

```
├── config.yaml                 # Configuration file for all hyperparameters
├── data_preprocessing.py        # Data loading, cleaning, and feature engineering
├── model.py                    # Neural network architecture and utilities
├── train.py                    # Training pipeline with ClearML integration
├── predict.py                  # Test set prediction and submission generation
├── requirements.txt            # Python dependencies
└── README_SPACESHIP_TITANIC.md # This documentation
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Place your Kaggle competition data files in the project directory:
- `train.csv` - Training dataset
- `test.csv` - Test dataset

**Note**: If data files are not present, the solution will create sample data for demonstration purposes.

### 3. Training

```bash
python train.py
```

### 4. Prediction

```bash
python predict.py
```

## Configuration

All hyperparameters and settings are managed through `config.yaml`. Key sections include:

### Data Configuration
- File paths for train/test data
- Validation split ratio
- Random seed for reproducibility

### Model Architecture
- Hidden layer sizes: `[256, 128, 64]`
- Dropout rates: `[0.3, 0.4, 0.5]`
- Activation function: `relu`
- Batch normalization: `true`

### Training Parameters
- Batch size: `64`
- Learning rate: `0.001`
- Epochs: `100`
- Optimizer: `adam`
- Weight decay: `0.0001`

### Early Stopping & Scheduling
- Early stopping patience: `15`
- Learning rate scheduler: `step`
- Step size: `20`, Gamma: `0.5`

### ClearML Integration
- Project name: `Spaceship_Titanic`
- Task name: `Neural_Network_Classifier`
- Automatic framework connection

## Data Preprocessing Features

### Missing Value Handling
- Numerical features: Filled with median values
- Categorical features: Filled with mode or 'Unknown'
- Boolean features: Filled with `False`

### Feature Engineering
- **Cabin Information**: Extracted deck, cabin number, and side from cabin strings
- **Family Features**: Group size, solo travelers, small/large group indicators
- **Spending Patterns**: Total spending, spending indicators across all amenities
- **Age Groups**: Categorized into Child, Teen, Young Adult, Adult, Senior
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Scaling**: StandardScaler for numerical features

### Generated Features
- `Deck`, `Cabin_num`, `Side` from `Cabin`
- `Group_size`, `Is_solo`, `Is_small_group`, `Is_large_group` from `PassengerId`
- `Total_spending`, `Has_spending` from amenity spending
- `Age_group` from `Age`

## Model Architecture

### Neural Network Design
- **Input Layer**: Automatically sized based on preprocessed features
- **Hidden Layers**: Configurable sizes with default `[256, 128, 64]`
- **Regularization**: Dropout layers with configurable rates
- **Normalization**: Optional batch normalization
- **Activation**: Configurable activation functions (ReLU, LeakyReLU, ELU, GELU)
- **Output Layer**: Single neuron for binary classification

### Training Features
- **Loss Function**: BCEWithLogitsLoss for stable training
- **Optimizers**: Adam, SGD, RMSprop with configurable parameters
- **Learning Rate Scheduling**: Step, Cosine, Exponential, Plateau schedulers
- **Early Stopping**: Prevents overfitting with configurable patience
- **Model Checkpointing**: Automatic saving of best model

## ClearML Experiment Tracking

The solution integrates with ClearML for comprehensive experiment tracking:

### Quick Setup

To set up ClearML experiment tracking, run the interactive setup helper:

```bash
python setup_clearml.py
```

This script will guide you through the configuration process and test your connection.

### Manual Setup Options

#### Option 1: Using clearml-init (Recommended)
1. Create a free account at [https://app.clear.ml/](https://app.clear.ml/)
2. Go to Settings > Workspace Configuration
3. Copy the configuration
4. Run `clearml-init` and paste the configuration

#### Option 2: Environment Variables
Add to your shell profile (`.bashrc`, `.zshrc`, etc.):
```bash
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=your_access_key
export CLEARML_API_SECRET_KEY=your_secret_key
```

#### Option 3: Config File
Update `config.yaml` with your credentials:
```yaml
clearml:
  api:
    web_server: "https://app.clear.ml"
    api_server: "https://api.clear.ml"
    files_server: "https://files.clear.ml"
    access_key: "your_access_key"
    secret_key: "your_secret_key"
```

### Logged Metrics
- Training and validation loss per epoch
- Validation accuracy, precision, recall, F1-score, AUC
- Learning rate progression
- Model architecture parameters
- Data statistics (feature count, dataset sizes)
- Final model performance metrics

### Artifacts
- Model checkpoints
- Configuration files
- Training history

## Usage Examples

### Custom Configuration

Create a custom config file:

```yaml
# custom_config.yaml
model:
  hidden_sizes: [512, 256, 128]
  dropout_rates: [0.2, 0.3, 0.4]
  activation: "leaky_relu"

training:
  batch_size: 128
  learning_rate: 0.0005
  epochs: 150
```

Run training with custom config:

```bash
python train.py --config custom_config.yaml
```

### Prediction with Custom Model

```bash
python predict.py --model custom_model.pth --output custom_submission.csv
```

### Data Preprocessing Only

```python
from data_preprocessing import SpaceshipDataProcessor

processor = SpaceshipDataProcessor()
train_df, test_df = processor.load_data('train.csv', 'test.csv')
processed_data = processor.preprocess(train_df, test_df)
```

### Model Creation and Training

```python
from model import create_model
from train import SpaceshipTrainer

# Custom training
trainer = SpaceshipTrainer('config.yaml')
history, metrics = trainer.train()
```

## Performance Monitoring

### Validation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for transported passengers
- **Recall**: Sensitivity for transported passengers
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

### Training Monitoring
- Real-time loss tracking
- Learning rate scheduling
- Early stopping based on validation loss
- Automatic best model saving

## Advanced Features

### Reproducibility
- Configurable random seeds
- Deterministic training options
- Consistent data splitting

### Hardware Support
- Automatic device detection (CPU/CUDA/MPS)
- Configurable number of workers for data loading
- Memory pinning for faster GPU transfers

### Error Handling
- Graceful handling of missing data files
- ClearML connection fallbacks
- Comprehensive logging and error messages

## Output Files

### Generated Files
- `best_model.pth`: Trained model checkpoint with metadata
- `submission.csv`: Kaggle-ready submission file
- `detailed_predictions.csv`: Detailed predictions with probabilities (optional)

### Submission Format
```csv
PassengerId,Transported
0001_01,False
0002_01,True
...
```

## Tips for Optimization

### Hyperparameter Tuning
1. Adjust learning rate (start with 0.001, try 0.0001-0.01)
2. Modify network architecture (layer sizes and depths)
3. Tune dropout rates (typically 0.1-0.5)
4. Experiment with different optimizers
5. Adjust batch sizes based on available memory

### Feature Engineering
1. Create domain-specific features based on competition insights
2. Experiment with different encoding strategies
3. Consider feature selection techniques
4. Try polynomial or interaction features

### Training Strategies
1. Use cross-validation for more robust evaluation
2. Implement ensemble methods
3. Try different loss functions
4. Experiment with learning rate scheduling

## Troubleshooting

### Common Issues

**CUDA/Memory Errors**:
- Reduce batch size in config.yaml
- Set device to 'cpu' in hardware configuration

**ClearML Connection Issues**:
- Run `python setup_clearml.py` for interactive setup assistance
- Check ClearML server configuration
- Ensure proper API credentials (see ClearML Setup section below)
- Training will continue without ClearML if connection fails

**Data Loading Errors**:
- Verify train.csv and test.csv file paths
- Check file formats and column names
- Sample data will be generated if files are missing

## Requirements

- Python 3.7+
- PyTorch 2.0+
- pandas 1.5+
- scikit-learn 1.3+
- ClearML 1.14+ (optional)
- PyYAML 6.0+

## License

This solution is provided for educational and competition purposes.
