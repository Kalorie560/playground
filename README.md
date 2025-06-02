# 🚀 Spaceship Titanic Deep Learning Solution

A complete machine learning solution for predicting passenger transportation in the Kaggle Spaceship Titanic competition.

## Quick Start

### Step 1: Configure ClearML (Optional)
Edit `config.yaml` and add your ClearML API credentials:

```yaml
clearml:
  api:
    access_key: "YOUR_ACCESS_KEY_HERE"
    secret_key: "YOUR_SECRET_KEY_HERE"
```

### Step 2: Prepare Data
Place your Kaggle competition files in the project root:
```
playground/
├── train.csv     ← Your training data
├── test.csv      ← Your test data
└── ...
```

> ⚠️ **No data files?** The system will automatically generate sample data for testing.

### Step 3: Train the Model
Run the training script to generate your model:

```bash
python train.py
```

This will create `best_model.pth` with your trained neural network.

### Step 4: Use the Web Application
Launch the interactive web interface:

```bash
python run_app.py
```

Open http://localhost:8501 in your browser to input passenger features and get real-time predictions.

## Data Preprocessing

The preprocessing pipeline automatically:

1. **Handles Missing Values**
   - Numerical features: Filled with median values
   - Categorical features: Filled with mode values
   - Boolean features: Filled with False

2. **Feature Engineering**
   - Extracts cabin deck, number, and side from cabin strings
   - Creates family size and group features from PassengerId
   - Calculates total spending and spending ratios
   - Generates age groups and interaction features

3. **Encoding and Scaling**
   - Label encodes categorical variables
   - Standardizes numerical features using StandardScaler
   - Converts boolean variables to integers

## Neural Network Architecture

The model is a multi-layer feedforward neural network:

```
Input Layer (varies based on features)
    ↓
Hidden Layer 1: 512 neurons + Batch Norm + Swish + Dropout(0.2)
    ↓
Hidden Layer 2: 256 neurons + Batch Norm + Swish + Dropout(0.3)
    ↓
Hidden Layer 3: 128 neurons + Batch Norm + Swish + Dropout(0.4)
    ↓
Hidden Layer 4: 64 neurons + Batch Norm + Swish + Dropout(0.5)
    ↓
Output Layer: 1 neuron (binary classification)
```

**Key Features:**
- **Activation Function**: Swish (x * sigmoid(x)) for smooth gradients
- **Regularization**: Batch normalization and progressive dropout
- **Residual Connections**: Skip connections for better gradient flow
- **Weight Initialization**: He initialization for ReLU-family activations

## Hyperparameters

Current optimized settings in `config.yaml`:

```yaml
model:
  hidden_sizes: [512, 256, 128, 64]
  dropout_rates: [0.2, 0.3, 0.4, 0.5]
  activation: "swish"
  use_batch_norm: true
  use_residual: true

training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 150
  optimizer: "adamw"
  weight_decay: 0.0005
  
  loss:
    type: "label_smoothing"
    smoothing: 0.1
  
  scheduler:
    type: "cosine"
  
  early_stopping:
    patience: 20
    min_delta: 0.00005
```

**Training Strategy:**
- **Optimizer**: AdamW with weight decay for regularization
- **Loss Function**: Label smoothing BCE to prevent overconfidence
- **Learning Rate Scheduler**: Cosine annealing for smooth convergence
- **Early Stopping**: Prevents overfitting with patience-based monitoring

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Setup ClearML for experiment tracking
python setup_clearml.py
```

## Usage Examples

### Training with Custom Config
```bash
# Copy and modify configuration
cp config.yaml my_config.yaml
# Edit my_config.yaml with your settings
python train.py --config my_config.yaml
```

### Making Predictions
```bash
# Generate predictions on test set
python predict.py
```

### Web Application
```bash
# Launch interactive interface
python run_app.py
# Or use the shell script
./run_app.sh
```

## Files Overview

- `config.yaml` - All hyperparameters and settings
- `train.py` - Main training pipeline with ClearML integration
- `model.py` - Neural network architecture
- `data_preprocessing.py` - Data loading and feature engineering
- `predict.py` - Generate test predictions
- `app.py` - Streamlit web application
- `run_app.py` - Web app launcher

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- scikit-learn 1.3+
- ClearML 1.14+ (optional)

See `requirements.txt` for the complete list.