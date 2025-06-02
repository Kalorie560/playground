#!/bin/bash

# Spaceship Titanic Web Application Launcher
# 宇宙船タイタニック ウェブアプリケーション起動スクリプト

echo "🚀 Starting Spaceship Titanic Web Application..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📋 Installing/updating requirements..."
    pip install -r requirements.txt
fi

# Run the Streamlit app
echo "🌐 Launching web application..."
streamlit run app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

echo "👋 Application stopped."