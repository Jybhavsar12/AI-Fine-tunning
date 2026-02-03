"""
Main fine-tuning script with support for LoRA/QLoRA and full fine-tuning.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

from data_preparation import DatasetPreparator


class FineTuner:
    """Fine-tuning orchestrator for LLMs."""

    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize fine-tuner with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization if specified."""
        model_name = self.config["model"]["name"]

        print(f"Loading model: {model_name}")

        # Configure quantization
        quantization_config = None
        if self.config["model"].get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config["model"].get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config["model"].get("trust_remote_code", True),
            padding_side="right",
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=self.config["model"].get("trust_remote_code", True),
            torch_dtype=(
                torch.bfloat16
                if self.config["training"].get("bf16", True)
                else torch.float16
            ),
        )

        # Prepare model for k-bit training if using quantization
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        print(f"Model loaded successfully")
        return self.model, self.tokenizer

    def setup_lora(self):
        """Configure and apply LoRA to the model."""
        if not self.config["lora"].get("enabled", True):
            print("LoRA is disabled, using full fine-tuning")
            return self.model

        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=self.config["lora"]["task_type"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        return self.model

    def prepare_datasets(self):
        """Load and prepare training and evaluation datasets."""
        dataset_config = self.config["dataset"]
        preparator = DatasetPreparator(
            prompt_format=dataset_config.get("prompt_template", "alpaca"),
            max_length=dataset_config.get("max_length", 2048),
        )

        # Load datasets
        if dataset_config["name"] == "custom":
            train_dataset = preparator.load_from_jsonl(dataset_config["train_file"])
            eval_dataset = preparator.load_from_jsonl(dataset_config["eval_file"])
        else:
            # Load from HuggingFace
            train_dataset = preparator.load_from_huggingface(
                dataset_config["name"], split="train"
            )
            eval_dataset = preparator.load_from_huggingface(
                dataset_config["name"], split="validation"
            )

        # Format datasets
        self.train_dataset = preparator.format_dataset(train_dataset)
        self.eval_dataset = preparator.format_dataset(eval_dataset)

        # Tokenize
        def tokenize_function(examples):
            outputs = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=dataset_config.get("max_length", 2048),
                padding=False,
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
        )

        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names,
        )

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Evaluation samples: {len(self.eval_dataset)}")

        return self.train_dataset, self.eval_dataset

    def get_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        train_config = self.config["training"]

        return TrainingArguments(
            output_dir=train_config["output_dir"],
            num_train_epochs=train_config["num_train_epochs"],
            per_device_train_batch_size=train_config["per_device_train_batch_size"],
            per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
            learning_rate=train_config["learning_rate"],
            lr_scheduler_type=train_config["lr_scheduler_type"],
            warmup_ratio=train_config["warmup_ratio"],
            weight_decay=train_config["weight_decay"],
            max_grad_norm=train_config["max_grad_norm"],
            optim=train_config["optim"],
            fp16=train_config.get("fp16", False),
            bf16=train_config.get("bf16", True),
            logging_steps=train_config["logging_steps"],
            save_steps=train_config["save_steps"],
            eval_steps=train_config["eval_steps"],
            save_total_limit=train_config["save_total_limit"],
            evaluation_strategy=train_config["evaluation_strategy"],
            group_by_length=train_config.get("group_by_length", True),
            report_to=train_config.get("report_to", ["tensorboard"]),
            seed=train_config.get("seed", 42),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def train(self):
        """Execute the fine-tuning process."""
        print("Starting fine-tuning process...")

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Setup LoRA
        self.setup_lora()

        # Prepare datasets
        self.prepare_datasets()

        # Get training arguments
        training_args = self.get_training_arguments()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print("Training started...")
        trainer.train()

        # Save final model
        output_dir = self.config["training"]["output_dir"]
        final_model_path = f"{output_dir}/final_model"

        print(f"Saving final model to {final_model_path}")
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)

        print("Training completed successfully!")
        return trainer


def main():
    """Main entry point for fine-tuning."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLMs with LoRA/QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    # Initialize and run fine-tuning
    finetuner = FineTuner(config_path=args.config)
    finetuner.train()


if __name__ == "__main__":
    main()
