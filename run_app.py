#!/usr/bin/env python3
"""
Script to run the Spaceship Titanic Web Application
宇宙船タイタニック ウェブアプリケーション実行スクリプト
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    try:
        import streamlit
        import torch
        import pandas
        import numpy
        import yaml
        import sklearn
        print("✅ All required packages are available!")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def run_streamlit_app():
    """Run the Streamlit application."""
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        return False
    
    print("🚀 Starting Spaceship Titanic Web Application...")
    print("📝 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("=" * 60)
    print("🚀 Spaceship Titanic Web Application Launcher")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Please run this script from the project root directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()