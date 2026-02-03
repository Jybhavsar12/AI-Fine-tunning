# 🎓 Getting Started Guide

This guide will walk you through your first fine-tuning project step-by-step.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- Basic understanding of machine learning concepts

## Step-by-Step Tutorial

### Step 1: Environment Setup (5 minutes)

```bash
# Clone or navigate to the project directory
cd AI-Fine-tunning

# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate the virtual environment
source venv/bin/activate
```

### Step 2: Understanding the Configuration (10 minutes)

Open `config/training_config.yaml` and review the key settings:

**Model Selection:**
```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # Change this to your preferred model
  load_in_4bit: true                 # Use QLoRA for memory efficiency
```

**For beginners, we recommend starting with a smaller model:**
```yaml
model:
  name: "microsoft/phi-2"  # Only 2.7B parameters, faster to train
```

**LoRA Configuration:**
```yaml
lora:
  r: 16              # Rank: higher = more capacity but slower
  lora_alpha: 32     # Scaling factor (usually 2x the rank)
  lora_dropout: 0.05 # Regularization
```

**Training Parameters:**
```yaml
training:
  num_train_epochs: 3                    # Number of passes through data
  per_device_train_batch_size: 4         # Samples per GPU
  gradient_accumulation_steps: 4         # Effective batch size = 4 * 4 = 16
  learning_rate: 2.0e-4                  # How fast the model learns
```

### Step 3: Prepare Your Dataset (15 minutes)

#### Option A: Use Sample Data (Quickest)

```bash
python src/data_preparation.py
```

This creates sample datasets in `data/train.jsonl` and `data/eval.jsonl`.

#### Option B: Create Custom Data

Create a file `data/my_data.jsonl` with your examples:

```json
{"instruction": "Translate to Spanish", "input": "Hello, how are you?", "output": "Hola, ¿cómo estás?"}
{"instruction": "Summarize this text", "input": "Long article about AI...", "output": "Brief summary..."}
{"instruction": "Write a poem about", "input": "the ocean", "output": "Waves crash upon the shore..."}
```

**Important:** Each line must be valid JSON!

Update your config to point to your data:
```yaml
dataset:
  train_file: "data/my_data.jsonl"
  eval_file: "data/my_eval.jsonl"
```

### Step 4: Start Training (30-120 minutes depending on model size)

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Start training
python src/finetune.py --config config/training_config.yaml
```

**What to expect:**
1. Model download (first time only, can be several GB)
2. Dataset loading and tokenization
3. Training progress with loss metrics
4. Periodic checkpoints saved to `outputs/`

**Monitor training:**
```bash
# In another terminal, start TensorBoard
tensorboard --logdir outputs/runs
```

Then open http://localhost:6006 in your browser.

### Step 5: Test Your Model (5 minutes)

#### Interactive Chat

```bash
python src/inference.py --model_path outputs/final_model
```

Type your prompts and see the model's responses!

#### Single Prompt Test

```bash
python src/inference.py \
  --model_path outputs/final_model \
  --prompt "Explain what is machine learning in simple terms"
```

### Step 6: Evaluate Performance (10 minutes)

```bash
python src/evaluate.py \
  --model_path outputs/final_model \
  --dataset data/eval.jsonl \
  --output evaluation_results.json
```

This will show metrics like ROUGE and BLEU scores.

## Common Workflows

### Workflow 1: Quick Experimentation

```bash
# 1. Create sample data
python src/data_preparation.py

# 2. Edit config to use a small model
# Change model.name to "microsoft/phi-2"

# 3. Train for 1 epoch (quick test)
# Change num_train_epochs to 1

# 4. Run training
python src/finetune.py

# 5. Test immediately
python src/inference.py --model_path outputs/final_model
```

### Workflow 2: Production Fine-Tuning

```bash
# 1. Prepare high-quality dataset (100+ examples)
python examples/custom_dataset_example.py

# 2. Configure for your use case
# - Choose appropriate base model
# - Set optimal LoRA parameters
# - Configure training epochs

# 3. Train with monitoring
python src/finetune.py --config config/training_config.yaml

# 4. Evaluate thoroughly
python src/evaluate.py --model_path outputs/final_model --dataset data/eval.jsonl

# 5. Iterate based on results
# - Adjust learning rate if loss is unstable
# - Add more data if performance is poor
# - Increase LoRA rank if underfitting
```

## Tips for Success

### 1. Data Quality Matters Most

 **Good:**
- Clear, consistent instructions
- Diverse examples covering your use case
- Correct, high-quality outputs
- 50-100+ examples minimum

 **Bad:**
- Inconsistent formatting
- Duplicate or near-duplicate examples
- Errors in the expected outputs
- Too few examples (< 20)

### 2. Start Small, Scale Up

1. Test with `microsoft/phi-2` (2.7B)
2. If results are good, try `Llama-2-7b` (7B)
3. For best quality, use `Llama-2-13b` (13B) or larger

### 3. Monitor Your Training

Watch for these signs:

- **Loss decreasing steadily**: Good!
- **Loss stuck/not decreasing**: Try higher learning rate
- **Loss jumping around**: Try lower learning rate
- **Eval loss increasing**: You're overfitting, stop early

### 4. Hyperparameter Tuning

If results aren't good, try adjusting:

```yaml
# For better quality (slower training):
lora:
  r: 32  # Increase from 16

# For faster convergence:
training:
  learning_rate: 3.0e-4  # Increase from 2.0e-4

# For more stable training:
training:
  learning_rate: 1.0e-4  # Decrease from 2.0e-4
```

## Next Steps

Once you're comfortable with the basics:

1. **Experiment with different models** - Try Mistral, Gemma, or other architectures
2. **Try different prompt formats** - ChatML, Llama-2 format, or custom templates
3. **Use real datasets** - Load from HuggingFace Hub or your own data
4. **Advanced techniques** - Experiment with different LoRA configurations
5. **Deploy your model** - Use the inference script in your applications

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review example scripts in `examples/`
- Look at the configuration file comments
- Check training logs in `outputs/`

Happy fine-tuning! 

