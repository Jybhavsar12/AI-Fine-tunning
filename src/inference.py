"""
Inference script for testing fine-tuned models.
"""

import torch
import yaml
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel


class ModelInference:
    """Handle inference with fine-tuned models."""
    
    def __init__(self, 
                 model_path: str,
                 base_model: Optional[str] = None,
                 config_path: str = "config/training_config.yaml"):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to fine-tuned model (LoRA adapter or full model)
            base_model: Base model name (required if using LoRA adapter)
            config_path: Path to training config for inference parameters
        """
        self.model_path = model_path
        self.base_model = base_model
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Check if this is a LoRA adapter or full model
        if self.base_model:
            # Load base model and apply LoRA adapter
            print(f"Loading base model: {self.base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.model = self.model.merge_and_unload()  # Merge LoRA weights
        else:
            # Load full fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        
        self.model.eval()
        print("Model loaded successfully")
        
        return self.model, self.tokenizer
    
    def generate(self, 
                 prompt: str,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 do_sample: Optional[bool] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        if self.model is None:
            self.load_model()
        
        # Use config defaults if not specified
        inf_config = self.config.get('inference', {})
        max_new_tokens = max_new_tokens or inf_config.get('max_new_tokens', 512)
        temperature = temperature or inf_config.get('temperature', 0.7)
        top_p = top_p or inf_config.get('top_p', 0.9)
        top_k = top_k or inf_config.get('top_k', 50)
        repetition_penalty = repetition_penalty or inf_config.get('repetition_penalty', 1.1)
        do_sample = do_sample if do_sample is not None else inf_config.get('do_sample', True)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(self):
        """Interactive chat mode."""
        if self.model is None:
            self.load_model()
        
        print("\n" + "="*50)
        print("Interactive Chat Mode")
        print("Type 'quit' or 'exit' to end the conversation")
        print("="*50 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Format prompt (adjust based on your training format)
            prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
            
            response = self.generate(prompt)
            print(f"\nAssistant: {response}\n")


def main():
    """Main entry point for inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model or LoRA adapter"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required if using LoRA adapter)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for single generation (if not provided, enters chat mode)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )

    args = parser.parse_args()

    # Initialize inference
    inference = ModelInference(
        model_path=args.model_path,
        base_model=args.base_model,
        config_path=args.config
    )

    if args.prompt:
        # Single generation
        response = inference.generate(args.prompt)
        print(f"Response: {response}")
    else:
        # Interactive chat
        inference.chat()


if __name__ == "__main__":
    main()

