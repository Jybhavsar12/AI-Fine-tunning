"""
Dataset preparation utilities for fine-tuning LLMs.
Supports multiple prompt formats and data sources.
"""

import json
import pandas as pd
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset
from pathlib import Path


class PromptTemplate:
    """Handles different prompt formatting styles."""
    
    @staticmethod
    def alpaca(instruction: str, input_text: str = "", output: str = "") -> str:
        """Alpaca-style prompt format."""
        if input_text:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        return prompt
    
    @staticmethod
    def chatml(instruction: str, input_text: str = "", output: str = "") -> str:
        """ChatML format (used by many modern models)."""
        user_message = f"{instruction}\n{input_text}".strip()
        return f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    @staticmethod
    def llama2(instruction: str, input_text: str = "", output: str = "") -> str:
        """Llama-2 chat format."""
        user_message = f"{instruction}\n{input_text}".strip()
        return f"[INST] {user_message} [/INST] {output}"
    
    @staticmethod
    def custom(instruction: str, input_text: str = "", output: str = "", 
               template: str = "{instruction}\n{input}\n{output}") -> str:
        """Custom template format."""
        return template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )


class DatasetPreparator:
    """Prepare datasets for fine-tuning."""
    
    def __init__(self, prompt_format: str = "alpaca", max_length: int = 2048):
        self.prompt_format = prompt_format
        self.max_length = max_length
        self.template_fn = getattr(PromptTemplate, prompt_format, PromptTemplate.alpaca)
    
    def load_from_jsonl(self, file_path: str) -> Dataset:
        """Load dataset from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data)
    
    def load_from_json(self, file_path: str) -> Dataset:
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Dataset.from_list(data)
    
    def load_from_csv(self, file_path: str) -> Dataset:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace Hub."""
        return load_dataset(dataset_name, split=split)
    
    def format_dataset(self, dataset: Dataset, 
                      instruction_col: str = "instruction",
                      input_col: str = "input",
                      output_col: str = "output") -> Dataset:
        """Format dataset with the specified prompt template."""
        
        def format_example(example):
            instruction = example.get(instruction_col, "")
            input_text = example.get(input_col, "")
            output = example.get(output_col, "")
            
            formatted_text = self.template_fn(instruction, input_text, output)
            return {"text": formatted_text}
        
        return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def create_sample_dataset(self, output_path: str, num_samples: int = 100):
        """Create a sample dataset for testing."""
        samples = []
        
        # Example tasks
        tasks = [
            {
                "instruction": "Summarize the following text.",
                "input": "Artificial intelligence is transforming how we work and live. Machine learning models can now perform tasks that once required human intelligence.",
                "output": "AI and ML are changing work and daily life by automating tasks previously requiring human intelligence."
            },
            {
                "instruction": "Translate the following English text to French.",
                "input": "Hello, how are you today?",
                "output": "Bonjour, comment allez-vous aujourd'hui?"
            },
            {
                "instruction": "Write a Python function to calculate factorial.",
                "input": "",
                "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)"
            },
            {
                "instruction": "Explain what is machine learning in simple terms.",
                "input": "",
                "output": "Machine learning is a way for computers to learn from examples and improve at tasks without being explicitly programmed for every scenario."
            }
        ]

        # Repeat and vary samples
        import random
        for i in range(num_samples):
            sample = random.choice(tasks).copy()
            samples.append(sample)

        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        print(f"Created sample dataset with {num_samples} examples at {output_path}")
        return output_path


def main():
    """Example usage of dataset preparation."""
    preparator = DatasetPreparator(prompt_format="alpaca")

    # Create sample datasets
    preparator.create_sample_dataset("data/train.jsonl", num_samples=100)
    preparator.create_sample_dataset("data/eval.jsonl", num_samples=20)

    # Load and format
    train_dataset = preparator.load_from_jsonl("data/train.jsonl")
    formatted_dataset = preparator.format_dataset(train_dataset)

    print(f"Dataset size: {len(formatted_dataset)}")
    print(f"Sample:\n{formatted_dataset[0]['text']}")


if __name__ == "__main__":
    main()

