"""
Minimal example that works on ANY hardware (even CPU-only).
Uses the smallest possible model and minimal data.
"""

import json
import sys
from pathlib import Path

sys.path.append("src")


def create_tiny_dataset():
    """Create a very small dataset for testing."""

    # Just 10 examples - enough to test the system
    examples = [
        {
            "instruction": "Greet the user",
            "input": "",
            "output": "Hello! How can I help you today?",
        },
        {
            "instruction": "Say goodbye",
            "input": "",
            "output": "Goodbye! Have a great day!",
        },
        {
            "instruction": "Explain what AI is",
            "input": "",
            "output": "AI stands for Artificial Intelligence. It's technology that allows computers to learn and make decisions.",
        },
        {
            "instruction": "Tell a joke",
            "input": "",
            "output": "Why did the programmer quit? Because they didn't get arrays!",
        },
        {"instruction": "Count to five", "input": "", "output": "1, 2, 3, 4, 5"},
        {
            "instruction": "What is Python?",
            "input": "",
            "output": "Python is a popular programming language known for its simplicity and readability.",
        },
        {"instruction": "Translate to Spanish", "input": "Hello", "output": "Hola"},
        {"instruction": "What is 2+2?", "input": "", "output": "2+2 equals 4"},
        {"instruction": "Name a color", "input": "", "output": "Blue"},
        {
            "instruction": "What day comes after Monday?",
            "input": "",
            "output": "Tuesday comes after Monday",
        },
    ]

    # Create data directory
    Path("data").mkdir(exist_ok=True)

    # Save training data (8 examples)
    with open("data/tiny_train.jsonl", "w") as f:
        for example in examples[:8]:
            f.write(json.dumps(example) + "\n")

    # Save eval data (2 examples)
    with open("data/tiny_eval.jsonl", "w") as f:
        for example in examples[8:]:
            f.write(json.dumps(example) + "\n")

    print("✓ Created tiny dataset:")
    print(f"  - Training: 8 examples (data/tiny_train.jsonl)")
    print(f"  - Evaluation: 2 examples (data/tiny_eval.jsonl)")
    print()


def check_system():
    """Check what hardware is available."""
    import torch

    print("System Check:")
    print("-" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")

        if gpu_memory < 4:
            print("  → Recommendation: Use CPU config (very limited GPU)")
            config = "cpu_only_config.yaml"
        elif gpu_memory < 6:
            print("  → Recommendation: Use minimal config")
            config = "minimal_config.yaml"
        else:
            print("  → Recommendation: Use quick test config")
            config = "quick_test_config.yaml"
    else:
        print("✗ No GPU detected")
        print("  → Recommendation: Use CPU config")
        config = "cpu_only_config.yaml"

    print()
    return config


def main():
    """Run minimal example."""
    print("=" * 60)
    print("MINIMAL FINE-TUNING EXAMPLE")
    print("Works on ANY hardware (even CPU-only)")
    print("=" * 60)
    print()

    # Check system
    recommended_config = check_system()

    # Create tiny dataset
    create_tiny_dataset()

    # Instructions
    print("Next Steps:")
    print("-" * 50)
    print()
    print("1. Train the model:")
    print(f"   python src/finetune.py --config config/{recommended_config}")
    print()
    print("2. Test the model:")
    print("   python src/inference.py --model_path outputs/*/final_model")
    print()
    print("Expected:")
    print("  - Training time: 10-30 minutes (depending on hardware)")
    print("  - Storage needed: 2-5 GB")
    print("  - Model will learn basic patterns from 8 examples")
    print()
    print("Tips:")
    print("  - Start with this tiny dataset to test the system")
    print("  - If it works, gradually add more examples")
    print("  - Monitor memory usage during training")
    print()


if __name__ == "__main__":
    main()
