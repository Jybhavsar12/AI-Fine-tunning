"""
Quick start example for fine-tuning a small model.
This example uses a smaller model for faster training.
"""

import sys

sys.path.append("src")

from data_preparation import DatasetPreparator
from finetune import FineTuner


def create_sample_data():
    """Create sample training data."""
    print("Creating sample datasets...")
    preparator = DatasetPreparator()

    # Create training and evaluation datasets
    preparator.create_sample_dataset("data/train.jsonl", num_samples=100)
    preparator.create_sample_dataset("data/eval.jsonl", num_samples=20)

    print("Sample datasets created!")


def run_quick_finetune():
    """Run a quick fine-tuning example."""
    print("\n" + "=" * 60)
    print("Quick Start Fine-Tuning Example")
    print("=" * 60 + "\n")

    # Step 1: Create sample data
    create_sample_data()

    # Step 2: Run fine-tuning
    print("\nStarting fine-tuning...")
    print("Note: This will download the base model on first run.")
    print("For quick testing, we recommend using a smaller model like:")
    print("  - microsoft/phi-2 (2.7B parameters)")
    print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)")
    print("\nUsing config from: config/training_config.yaml")
    print("You can modify the config file to change the base model.\n")

    finetuner = FineTuner(config_path="config/training_config.yaml")
    finetuner.train()

    print("\n" + "=" * 60)
    print("Fine-tuning completed!")
    print("=" * 60)
    print("\nNext steps:")
    print(
        "1. Test your model with: python src/inference.py --model_path outputs/final_model"
    )
    print(
        "2. Evaluate your model with: python src/evaluate.py --model_path outputs/final_model --dataset data/eval.jsonl"
    )
    print("3. Chat with your model interactively by running inference without --prompt")


if __name__ == "__main__":
    run_quick_finetune()
