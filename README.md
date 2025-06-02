# 🚀 Spaceship Titanic Deep Learning Solution

> A complete machine learning solution for predicting passenger transportation in the Kaggle Spaceship Titanic competition

## 📊 Overview

This project provides an end-to-end solution for the Kaggle Spaceship Titanic competition, where the goal is to predict which passengers were transported to an alternate dimension during the spaceship's collision with a spacetime anomaly.

### 🎯 What This Project Does

- **Predicts passenger transportation** using advanced deep learning techniques
- **Provides an interactive web interface** for real-time predictions
- **Tracks experiments** with comprehensive logging and metrics
- **Offers multiple usage modes** - web app, command line, or programmatic API

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Smart AI Model** | PyTorch neural network with configurable architecture |
| 🌐 **Web Interface** | User-friendly Streamlit app with Japanese support |
| 📈 **Experiment Tracking** | ClearML integration for monitoring model performance |
| ⚙️ **Easy Configuration** | YAML-based settings for all hyperparameters |
| 🔧 **Feature Engineering** | Advanced data preprocessing and feature extraction |
| 📊 **Comprehensive Metrics** | Detailed performance evaluation and visualization |
| 🎲 **Reproducible Results** | Seed management for consistent outcomes |

## 📁 Project Structure

```
├── config.yaml                 # Configuration file for all hyperparameters
├── data_preprocessing.py        # Data loading, cleaning, and feature engineering
├── model.py                    # Neural network architecture and utilities
├── train.py                    # Training pipeline with ClearML integration
├── predict.py                  # Test set prediction and submission generation
├── app.py                      # Streamlit web application for interactive predictions
├── run_app.py                  # Python script to launch the web application
├── run_app.sh                  # Bash script to launch the web application
├── setup_clearml.py            # Interactive ClearML setup helper
├── requirements.txt            # Python dependencies
├── README.md                   # Main documentation (English)
├── README_ja.md                # Japanese documentation
└── WEB_APP_README.md           # Detailed web application documentation
```

## 🚀 Quick Start Guide

### Step 1: Setup Environment

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd playground

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Add your Kaggle competition files to the project root:
```
📁 playground/
├── train.csv     ← Your training data
├── test.csv      ← Your test data
└── ...
```

> 💡 **No data?** The system will automatically generate sample data for testing!

### Step 3: Choose Your Adventure

#### 🎮 Option A: Interactive Web App (Recommended for Beginners)

Perfect for exploring predictions with a visual interface:

```bash
python run_app.py
```

Then open http://localhost:8501 in your browser

#### 🖥️ Option B: Command Line (For Data Scientists)

Train your own model and generate predictions:

```bash
# Train the model
python train.py

# Make predictions
python predict.py
```

#### 🔧 Option C: Custom Configuration

Experiment with different settings:

```bash
# Edit config.yaml to your liking, then:
python train.py --config config.yaml
```

## 🌐 Interactive Web Application

Experience the power of AI prediction through an intuitive web interface!

### 🎨 What You'll Get

| Feature | Benefit |
|---------|---------|
| 🖱️ **Point & Click Interface** | No coding required - just fill out the form |
| 🌏 **Bilingual Support** | Full Japanese and English language support |
| ⚡ **Instant Results** | Get predictions in real-time as you type |
| 📊 **Visual Feedback** | See probability scores and confidence metrics |
| 🛡️ **Smart Validation** | Automatic error checking and data formatting |

### 📝 Input Information

Simply enter passenger details:

- 🏠 **Background**: Home planet, age, VIP status, cryo-sleep state
- 🏨 **Accommodation**: Cabin deck, number, and side preference  
- 🎯 **Destination**: Where they're headed in the galaxy
- 💰 **Spending**: Money spent on ship amenities and services

### 📈 Prediction Results

The AI will show you:
- 🎯 **Transportation probability** (0-100%)
- 📊 **Confidence breakdown** with detailed metrics
- 💵 **Spending analysis** and patterns
- 📋 **Input summary** for verification

## ⚙️ Configuration Made Simple

Everything is controlled through `config.yaml` - no code changes needed!

### 🎛️ Quick Settings

```yaml
# Model tweaks
model:
  hidden_sizes: [256, 128, 64]  # Network size
  dropout_rates: [0.3, 0.4, 0.5]  # Prevent overfitting
  activation: "relu"  # Activation function

# Training behavior  
training:
  batch_size: 64  # How much data per step
  learning_rate: 0.001  # How fast to learn
  epochs: 100  # How long to train
```

### 🔧 Advanced Options

- **Data paths**: Point to your train/test CSV files
- **Validation split**: How much data to use for testing (default: 20%)
- **Reproducibility**: Set random seeds for consistent results
- **Hardware**: Auto-detect GPU/CPU or force specific device
- **Experiment tracking**: Connect to ClearML for monitoring

## 🔧 How the AI Works

### 🧹 Smart Data Processing

The system automatically cleans and prepares your data:

- **Fills missing values** intelligently (median for numbers, mode for categories)
- **Creates new features** from existing data (cabin deck, family size, spending patterns)
- **Encodes categories** for machine learning compatibility
- **Scales numbers** to work well with neural networks

### 🧠 Neural Network Architecture

Think of it as a digital brain with multiple layers:

```
Input → Hidden Layer 1 (256 neurons) → Hidden Layer 2 (128 neurons) → Hidden Layer 3 (64 neurons) → Prediction
```

**What makes it smart:**
- 🎯 **Dropout prevention**: Avoids memorizing and focuses on patterns
- 📊 **Batch normalization**: Keeps learning stable and fast
- ⚡ **ReLU activation**: Efficient mathematical function for decisions
- 🎛️ **Configurable size**: Adjust complexity based on your needs

### 🎓 Training Process

The AI learns through:
- **Loss calculation**: Measures how wrong predictions are
- **Backpropagation**: Adjusts neurons to reduce errors  
- **Early stopping**: Prevents overtraining when improvement plateaus
- **Learning rate scheduling**: Starts fast, then slows down for precision

## 📊 Experiment Tracking (Optional)

Want to monitor your model's training like a pro? ClearML integration is included!

### 🚀 Easy Setup

```bash
# Let the script guide you through setup
python setup_clearml.py
```

**What you get:**
- 📈 Real-time training graphs
- 📊 Performance metrics tracking  
- 🔄 Experiment comparison tools
- 💾 Automatic model saving
- 📋 Complete training logs

### 🎯 What Gets Tracked

| Metric Type | Examples |
|-------------|----------|
| 📈 **Training Progress** | Loss curves, accuracy over time |
| 🎯 **Performance** | Precision, recall, F1-score, AUC |
| ⚙️ **Configuration** | All hyperparameters and settings |
| 📊 **Data Stats** | Dataset sizes, feature distributions |
| 🏆 **Final Results** | Best model performance and metrics |

> 💡 **Don't want tracking?** No problem! The system works perfectly without ClearML.

## 📋 Common Usage Patterns

### 🎯 Experimenting with Settings

Want to try different model configurations? Easy!

```bash
# Create your own config file
cp config.yaml my_experiment.yaml

# Edit my_experiment.yaml with your changes, then:
python train.py --config my_experiment.yaml
```

### 🔄 Programmatic Usage

Integrate the AI into your own Python code:

```python
# Quick prediction pipeline
from data_preprocessing import SpaceshipDataProcessor
from model import create_model

# Process your data
processor = SpaceshipDataProcessor()
train_df, test_df = processor.load_data('train.csv', 'test.csv')
X_train, X_test, y_train = processor.preprocess(train_df, test_df)

# Train and predict
from train import SpaceshipTrainer
trainer = SpaceshipTrainer('config.yaml')
model, metrics = trainer.train()
```

### 🎮 Web App Integration

```python
# Use the web predictor in your own app
from app import SpaceshipWebPredictor

predictor = SpaceshipWebPredictor()
result = predictor.predict({
    'home_planet': 'Earth',
    'age': 25,
    'vip': False,
    # ... add other passenger details
})
print(f"Transportation probability: {result}%")
```

## 📈 Understanding Your Results

### 📊 Key Metrics Explained

| Metric | What It Means | Good Score |
|--------|---------------|------------|
| **Accuracy** | How often the model is correct overall | > 80% |
| **Precision** | Of predicted transportations, how many were right | > 75% |
| **Recall** | Of actual transportations, how many were caught | > 75% |
| **F1-Score** | Balanced measure of precision and recall | > 75% |
| **AUC** | How well the model separates the two classes | > 0.85 |

### 📁 Output Files

After training and prediction, you'll get:

- 📦 `best_model.pth` - Your trained AI model  
- 📄 `submission.csv` - Ready for Kaggle submission
- 📊 `detailed_predictions.csv` - Probabilities for analysis

## 💡 Tips for Better Results

### 🎛️ Try These Settings

```yaml
# For better accuracy
model:
  hidden_sizes: [512, 256, 128]  # Bigger network
  dropout_rates: [0.2, 0.3, 0.4]  # Less dropout

# For faster training  
training:
  batch_size: 128  # Bigger batches
  learning_rate: 0.003  # Faster learning
```

### 🚀 Advanced Techniques

1. **Cross-validation**: Train multiple models for robustness
2. **Ensemble methods**: Combine multiple model predictions  
3. **Feature engineering**: Create new features from existing data
4. **Hyperparameter search**: Systematically try different settings

## 🐛 Troubleshooting

Having issues? Here are quick fixes for common problems:

### 💾 Memory Problems

```bash
# Out of memory errors?
# Edit config.yaml and reduce:
training:
  batch_size: 32  # Make this smaller
```

Or force CPU usage:
```yaml
hardware:
  device: "cpu"  # Use CPU instead of GPU
```

### 🌐 Web App Won't Start

```bash
# Port already in use?
streamlit run app.py --server.port 8502

# Missing dependencies?
pip install -r requirements.txt
```

### 📁 Data File Issues

- **Missing train.csv/test.csv?** No worries! Sample data will be generated automatically
- **Wrong file format?** Make sure your CSV files have the expected columns
- **File path errors?** Place your CSV files in the project root directory

### 🔗 ClearML Connection Problems

```bash
# Run the setup helper
python setup_clearml.py

# Or just skip it - training works without ClearML too!
```

### ❓ Still Stuck?

1. Check the console output for detailed error messages
2. Make sure all dependencies are installed correctly
3. Try running with sample data first to test the setup
4. Check file permissions and paths

## 📋 System Requirements

| Component | Version | Required |
|-----------|---------|----------|
| Python | 3.7+ | ✅ Yes |
| PyTorch | 2.0+ | ✅ Yes |
| Streamlit | 1.28+ | ✅ Yes |
| pandas | 1.5+ | ✅ Yes |
| scikit-learn | 1.3+ | ✅ Yes |
| ClearML | 1.14+ | ❌ Optional |

## 📄 License & Acknowledgments

📚 **Educational Use**: This project is designed for learning and Kaggle competition participation.

🙏 **Thanks to**:
- Kaggle for the Spaceship Titanic competition
- PyTorch, Streamlit, and the open-source ML community
- ClearML for excellent experiment tracking tools

---

**🚀 Ready to predict the future? Let's go!**