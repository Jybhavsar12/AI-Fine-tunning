"""
Compare different fine-tuned models or configurations.
"""

import sys
import json
sys.path.append('src')

from inference import ModelInference
from evaluate import ModelEvaluator


def compare_models(model_paths, test_prompts, base_model=None):
    """
    Compare multiple models on the same prompts.
    
    Args:
        model_paths: List of paths to fine-tuned models
        test_prompts: List of prompts to test
        base_model: Base model name (if using LoRA adapters)
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"Testing model: {model_path}")
        print('='*60)
        
        inference = ModelInference(model_path, base_model)
        inference.load_model()
        
        model_results = []
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = inference.generate(prompt, max_new_tokens=256)
            print(f"Response: {response}\n")
            
            model_results.append({
                'prompt': prompt,
                'response': response
            })
        
        results[model_path] = model_results
    
    return results


def main():
    """Example model comparison."""
    
    # Example test prompts
    test_prompts = [
        "### Instruction:\nExplain what is machine learning.\n\n### Response:\n",
        "### Instruction:\nWrite a Python function to calculate fibonacci numbers.\n\n### Response:\n",
        "### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n",
    ]
    
    # Models to compare (update these paths)
    model_paths = [
        "outputs/checkpoint-100",
        "outputs/checkpoint-200",
        "outputs/final_model",
    ]
    
    print("Model Comparison Tool")
    print("=" * 60)
    print(f"Testing {len(model_paths)} models on {len(test_prompts)} prompts")
    
    # Run comparison
    results = compare_models(model_paths, test_prompts)
    
    # Save results
    output_file = "model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for model_path, model_results in results.items():
        print(f"\nModel: {model_path}")
        print(f"  - Tested on {len(model_results)} prompts")
        avg_length = sum(len(r['response']) for r in model_results) / len(model_results)
        print(f"  - Average response length: {avg_length:.0f} characters")


if __name__ == "__main__":
    main()

