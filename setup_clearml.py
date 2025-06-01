#!/usr/bin/env python3
"""
ClearML Setup Helper Script

This script helps users set up ClearML configuration for the Spaceship Titanic project.
It provides multiple setup options and validates the configuration.
"""

import os
import sys
import yaml
from typing import Dict, Any


def check_clearml_installation():
    """Check if ClearML is installed."""
    try:
        import clearml
        print(f"✓ ClearML is installed (version: {clearml.__version__})")
        return True
    except ImportError:
        print("✗ ClearML is not installed. Please install it with: pip install clearml")
        return False


def check_existing_config():
    """Check for existing ClearML configuration."""
    config_paths = [
        os.path.expanduser('~/clearml.conf'),
        '/opt/clearml/clearml.conf',
        'clearml.conf'
    ]
    
    env_vars = ['CLEARML_WEB_HOST', 'CLEARML_API_HOST', 'CLEARML_FILES_HOST', 
               'CLEARML_API_ACCESS_KEY', 'CLEARML_API_SECRET_KEY']
    
    has_conf_file = any(os.path.exists(path) for path in config_paths)
    has_env_vars = any(os.getenv(var) for var in env_vars)
    
    if has_conf_file:
        existing_path = next(path for path in config_paths if os.path.exists(path))
        print(f"✓ Found existing clearml.conf at: {existing_path}")
        return True
    
    if has_env_vars:
        print("✓ Found ClearML environment variables")
        return True
    
    print("✗ No existing ClearML configuration found")
    return False


def setup_method_1():
    """Setup using clearml-init command."""
    print("\n" + "="*60)
    print("METHOD 1: Using clearml-init (Recommended)")
    print("="*60)
    print("1. Create a free account at https://app.clear.ml/")
    print("2. Go to Settings > Workspace Configuration")
    print("3. Copy the configuration")
    print("4. Run the following command and paste the configuration:")
    print("\n   clearml-init\n")
    print("This will create a clearml.conf file in your home directory.")


def setup_method_2():
    """Setup using environment variables."""
    print("\n" + "="*60)
    print("METHOD 2: Using Environment Variables")
    print("="*60)
    print("Add these lines to your ~/.bashrc or ~/.zshrc:")
    print()
    print("export CLEARML_WEB_HOST=https://app.clear.ml")
    print("export CLEARML_API_HOST=https://api.clear.ml")
    print("export CLEARML_FILES_HOST=https://files.clear.ml")
    print("export CLEARML_API_ACCESS_KEY=your_access_key_here")
    print("export CLEARML_API_SECRET_KEY=your_secret_key_here")
    print()
    print("Get your access key and secret key from:")
    print("https://app.clear.ml/settings/workspace-configuration")


def setup_method_3():
    """Setup using config.yaml file."""
    print("\n" + "="*60)
    print("METHOD 3: Using config.yaml")
    print("="*60)
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("✗ config.yaml not found in current directory")
        return
    
    print("Edit your config.yaml file and uncomment/update the API section:")
    print()
    print("clearml:")
    print("  # ... other settings ...")
    print("  api:")
    print("    web_server: \"https://app.clear.ml\"")
    print("    api_server: \"https://api.clear.ml\"")
    print("    files_server: \"https://files.clear.ml\"")
    print("    access_key: \"your_access_key_here\"")
    print("    secret_key: \"your_secret_key_here\"")
    print()
    print("Get your credentials from:")
    print("https://app.clear.ml/settings/workspace-configuration")
    
    # Check if API section is already configured
    api_config = config.get('clearml', {}).get('api', {})
    if api_config.get('access_key') and api_config.get('secret_key'):
        print("\n✓ API credentials are already configured in config.yaml")
    else:
        print("\n✗ API credentials need to be added to config.yaml")


def test_clearml_connection():
    """Test ClearML connection."""
    print("\n" + "="*60)
    print("TESTING CLEARML CONNECTION")
    print("="*60)
    
    try:
        from clearml import Task
        
        # Try to create a test task
        test_task = Task.init(
            project_name="test_connection",
            task_name="connection_test",
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False
        )
        
        print("✓ ClearML connection successful!")
        print(f"✓ Connected to: {test_task.get_logger().get_default_upload_destination()}")
        
        # Close the test task
        test_task.close()
        
        return True
        
    except Exception as e:
        print(f"✗ ClearML connection failed: {e}")
        return False


def main():
    """Main setup function."""
    print("ClearML Setup Helper for Spaceship Titanic Project")
    print("="*50)
    
    # Check if ClearML is installed
    if not check_clearml_installation():
        sys.exit(1)
    
    # Check for existing configuration
    has_config = check_existing_config()
    
    if has_config:
        print("\nTesting existing configuration...")
        if test_clearml_connection():
            print("\n🎉 ClearML is already configured and working!")
            return
        else:
            print("\n⚠️  Configuration found but connection failed.")
    
    # Show setup methods
    print("\nChoose a setup method:")
    setup_method_1()
    setup_method_2()
    setup_method_3()
    
    print("\n" + "="*60)
    print("SELF-HOSTED CLEARML SERVER")
    print("="*60)
    print("If you want to use your own ClearML server:")
    print("1. Follow the installation guide: https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server")
    print("2. Update the server URLs in your configuration")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Choose one of the setup methods above")
    print("2. Configure your credentials")
    print("3. Run this script again to test the connection")
    print("4. Run 'python train.py' to start training with ClearML tracking")


if __name__ == "__main__":
    main()