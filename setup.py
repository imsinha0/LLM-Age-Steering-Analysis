#!/usr/bin/env python3
"""
Setup script for age-aware-lm-analysis project.
This script helps set up the project environment and download necessary data.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed", 
        "results/probes",
        "results/analysis",
        "experiments/notebooks",
        "experiments/scripts",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up age-aware-lm-analysis project...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Failed to install dependencies. Please install manually:")
        print("   pip install -r requirements.txt")
        return False
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🎯 CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available. Experiments will run on CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed. Please install dependencies first.")
    
    print("\n✅ Setup completed!")
    print("\n📚 Next steps:")
    print("1. Download your datasets to data/raw/")
    print("2. Run experiments from the experiments/notebooks/ directory")
    print("3. Check the README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
