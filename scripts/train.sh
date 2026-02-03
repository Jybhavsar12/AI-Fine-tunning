#!/bin/bash

# Training script with common configurations

# Default values
CONFIG="config/training_config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/train.sh [--config path/to/config.yaml]"
            exit 1
            ;;
    esac
done

echo "Starting training with config: $CONFIG"
echo ""

# Run training
python src/finetune.py --config "$CONFIG"

echo ""
echo "Training completed!"
echo "Model saved to outputs/final_model"

