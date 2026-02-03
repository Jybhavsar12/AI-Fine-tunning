"""
Evaluation utilities for fine-tuned models.
"""

import json
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm

import evaluate
from data_preparation import DatasetPreparator
from inference import ModelInference


class ModelEvaluator:
    """Evaluate fine-tuned models on various metrics."""

    def __init__(self, model_path: str, base_model: str = None):
        """Initialize evaluator with model."""
        self.inference = ModelInference(model_path, base_model)
        self.inference.load_model()

        # Load metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")

    def evaluate_on_dataset(
        self, dataset_path: str, num_samples: int = None, prompt_format: str = "alpaca"
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            dataset_path: Path to evaluation dataset (JSONL)
            num_samples: Number of samples to evaluate (None for all)
            prompt_format: Prompt template format

        Returns:
            Dictionary with evaluation metrics
        """
        # Load dataset
        preparator = DatasetPreparator(prompt_format=prompt_format)
        dataset = preparator.load_from_jsonl(dataset_path)

        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        predictions = []
        references = []

        print(f"Evaluating on {len(dataset)} samples...")

        for example in tqdm(dataset):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected_output = example.get("output", "")

            # Create prompt (without the response)
            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

            # Generate prediction
            prediction = self.inference.generate(
                prompt,
                max_new_tokens=256,
                temperature=0.1,  # Lower temperature for evaluation
                do_sample=False,  # Greedy decoding for consistency
            )

            predictions.append(prediction)
            references.append(expected_output)

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, references)

        return metrics

    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate evaluation metrics."""

        # ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions, references=references, use_stemmer=True
        )

        # BLEU score
        # Format references for BLEU (needs list of lists)
        references_bleu = [[ref] for ref in references]
        bleu_score = self.bleu.compute(
            predictions=predictions, references=references_bleu
        )

        # Exact match
        exact_matches = sum(
            1 for p, r in zip(predictions, references) if p.strip() == r.strip()
        )
        exact_match_rate = exact_matches / len(predictions) if predictions else 0

        metrics = {
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "bleu": bleu_score["bleu"],
            "exact_match_rate": exact_match_rate,
            "num_samples": len(predictions),
        }

        return metrics

    def print_metrics(self, metrics: Dict):
        """Pretty print evaluation metrics."""
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Number of samples: {metrics['num_samples']}")
        print(f"ROUGE-1: {metrics['rouge1']:.4f}")
        print(f"ROUGE-2: {metrics['rouge2']:.4f}")
        print(f"ROUGE-L: {metrics['rougeL']:.4f}")
        print(f"BLEU: {metrics['bleu']:.4f}")
        print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f}")
        print("=" * 50 + "\n")

    def save_metrics(self, metrics: Dict, output_path: str):
        """Save metrics to JSON file."""
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")


def main():
    """Main entry point for evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (for LoRA adapters)",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to evaluation dataset (JSONL)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for metrics",
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.base_model)
    metrics = evaluator.evaluate_on_dataset(args.dataset, args.num_samples)

    # Print and save results
    evaluator.print_metrics(metrics)
    evaluator.save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
