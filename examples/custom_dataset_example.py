"""
Example of preparing a custom dataset for fine-tuning.
"""

import json
import sys
sys.path.append('src')

from data_preparation import DatasetPreparator


def create_custom_dataset():
    """Create a custom dataset for a specific task."""
    
    # Example: Creating a dataset for a coding assistant
    coding_examples = [
        {
            "instruction": "Write a Python function to reverse a string",
            "input": "",
            "output": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "instruction": "Explain what this code does",
            "input": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "output": "This is a recursive function that calculates the factorial of a number n. It returns 1 if n is 0 or 1, otherwise it multiplies n by the factorial of (n-1)."
        },
        {
            "instruction": "Fix the bug in this code",
            "input": "def add_numbers(a, b):\n    return a + b + c",
            "output": "def add_numbers(a, b):\n    return a + b\n\n# The bug was that 'c' was not defined. The function should only add 'a' and 'b'."
        },
        {
            "instruction": "Write a function to check if a number is prime",
            "input": "",
            "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "instruction": "Convert this for loop to a list comprehension",
            "input": "result = []\nfor i in range(10):\n    if i % 2 == 0:\n        result.append(i * 2)",
            "output": "result = [i * 2 for i in range(10) if i % 2 == 0]"
        }
    ]
    
    # Save training data
    with open('data/custom_train.jsonl', 'w') as f:
        for example in coding_examples * 20:  # Repeat for more training data
            f.write(json.dumps(example) + '\n')
    
    # Save evaluation data
    with open('data/custom_eval.jsonl', 'w') as f:
        for example in coding_examples[:3]:  # Use subset for evaluation
            f.write(json.dumps(example) + '\n')
    
    print("Custom dataset created!")
    print("Training samples: data/custom_train.jsonl")
    print("Evaluation samples: data/custom_eval.jsonl")
    
    # Test different prompt formats
    preparator = DatasetPreparator(prompt_format="alpaca")
    dataset = preparator.load_from_jsonl('data/custom_train.jsonl')
    formatted = preparator.format_dataset(dataset)
    
    print("\nSample formatted prompt (Alpaca style):")
    print("-" * 60)
    print(formatted[0]['text'])
    print("-" * 60)


if __name__ == "__main__":
    create_custom_dataset()

