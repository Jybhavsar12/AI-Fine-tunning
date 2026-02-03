#!/bin/bash

# Setup script for the fine-tuning environment

echo "Setting up AI Fine-Tuning Environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To get started quickly, run:"
echo "  python examples/quick_start.py"
echo ""

