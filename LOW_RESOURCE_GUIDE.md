# Low-Resource Fine-Tuning Guide

This guide shows you how to use this framework on **limited hardware** - even without a GPU!

## Hardware Tiers

### Tier 1: CPU Only (No GPU)
- **RAM**: 8GB minimum
- **Storage**: 5GB free
- **Model**: DistilGPT2 (82M parameters)
- **Training Time**: 1-2 hours for 50 examples

### Tier 2: Basic GPU (4GB VRAM)
- **GPU**: GTX 1050 Ti, GTX 1650, or similar
- **RAM**: 8GB
- **Storage**: 10GB free
- **Model**: TinyLlama (1.1B parameters)
- **Training Time**: 20-30 minutes for 100 examples

### Tier 3: Mid-Range GPU (6-8GB VRAM)
- **GPU**: RTX 3060, RTX 2060, or similar
- **RAM**: 16GB
- **Storage**: 15GB free
- **Model**: Phi-2 (2.7B parameters)
- **Training Time**: 30-45 minutes for 100 examples

## Quick Start for Low-Resource Systems

### Option A: CPU-Only Training

```bash
# 1. Setup (lightweight installation)
python3 -m venv venv
source venv/bin/activate
pip install torch transformers datasets peft accelerate pyyaml tqdm

# 2. Create minimal dataset (10-20 examples)
python src/data_preparation.py

# 3. Train with CPU config
python src/finetune.py --config config/cpu_only_config.yaml

# 4. Test
python src/inference.py --model_path outputs/cpu/final_model
```

**Expected:**
- Model download: ~300MB (DistilGPT2)
- Training: 30-60 minutes
- Total storage: ~2GB

### Option B: Minimal GPU Training

```bash
# 1. Setup
source venv/bin/activate

# 2. Create small dataset
python src/data_preparation.py

# 3. Train with minimal config
python src/finetune.py --config config/minimal_config.yaml

# 4. Test
python src/inference.py --model_path outputs/minimal/final_model
```

**Expected:**
- Model download: ~2GB (TinyLlama)
- Training: 15-30 minutes
- Total storage: ~5GB

## Storage Optimization Tips

### 1. Use Smaller Models

Instead of 7B models, use:

| Model | Size | Storage | VRAM | Quality |
|-------|------|---------|------|---------|
| distilgpt2 | 82M | 300MB | CPU/1GB | Basic |
| TinyLlama-1.1B | 1.1B | 2GB | 4GB | Good |
| Phi-2 | 2.7B | 5GB | 6GB | Very Good |
| Llama-2-7B | 7B | 14GB | 8GB+ | Excellent |

### 2. Reduce Checkpoint Storage

In your config:
```yaml
training:
  save_total_limit: 1  # Keep only 1 checkpoint
  save_steps: 1000     # Save less frequently
```

### 3. Skip Evaluation During Training

```yaml
training:
  evaluation_strategy: "no"  # Don't evaluate during training
```

### 4. Use Shorter Sequences

```yaml
dataset:
  max_length: 256  # Instead of 2048
```

### 5. Clean Up After Training

```bash
# Remove intermediate checkpoints
rm -rf outputs/checkpoint-*

# Keep only final model
ls outputs/final_model/
```

## Performance Optimization Tips

### 1. Reduce Batch Size

```yaml
training:
  per_device_train_batch_size: 1  # Smallest possible
  gradient_accumulation_steps: 8  # Simulate larger batch
```

### 2. Use Fewer Training Examples

Start with just 20-50 high-quality examples instead of hundreds.

### 3. Train for Fewer Epochs

```yaml
training:
  num_train_epochs: 1  # Just one pass through data
```

### 4. Disable Monitoring

```yaml
training:
  report_to: []  # No TensorBoard/W&B
  logging_steps: 50  # Log less frequently
```

### 5. Use Gradient Checkpointing

Add to your training script (for very low memory):
```python
model.gradient_checkpointing_enable()
```

## Minimal Dataset Example

Create `data/tiny_train.jsonl` with just 10-20 examples:

```json
{"instruction": "Greet the user", "output": "Hello! How can I help you today?"}
{"instruction": "Say goodbye", "output": "Goodbye! Have a great day!"}
{"instruction": "Explain AI", "output": "AI is artificial intelligence, computer systems that can learn."}
```

This is enough to see if the system works!

## Cloud Alternatives (Free Tier)

If local resources are too limited, use free cloud options:

### Google Colab (Free)
- **GPU**: Tesla T4 (15GB VRAM)
- **RAM**: 12GB
- **Storage**: 100GB
- **Time Limit**: 12 hours
- **Cost**: FREE

### Kaggle Notebooks (Free)
- **GPU**: Tesla P100 (16GB VRAM)
- **RAM**: 13GB
- **Storage**: 73GB
- **Time Limit**: 9 hours/week
- **Cost**: FREE

### How to use with Colab:

```python
# In Colab notebook
!git clone <your-repo>
!cd AI-Fine-tunning && pip install -r requirements.txt
!python src/finetune.py --config config/quick_test_config.yaml
```

## Recommended Workflow for Limited Resources

### Step 1: Start Tiny
```bash
# Use CPU config with 10 examples
python src/finetune.py --config config/cpu_only_config.yaml
```

### Step 2: Test Immediately
```bash
python src/inference.py --model_path outputs/cpu/final_model \
  --prompt "Test prompt"
```

### Step 3: If It Works, Scale Up Gradually
- Try minimal_config.yaml (if you have 4GB GPU)
- Add more training examples (20 → 50 → 100)
- Increase epochs (1 → 2 → 3)

### Step 4: Clean Up Between Runs
```bash
rm -rf outputs/checkpoint-*
rm -rf outputs/cpu
```

## What to Expect with Small Models

### DistilGPT2 (82M)
- **Good for**: Simple tasks, pattern learning
- **Not good for**: Complex reasoning, long context
- **Example use**: Chatbot responses, simple classification

### TinyLlama (1.1B)
- **Good for**: Most basic tasks, decent quality
- **Not good for**: Very complex tasks
- **Example use**: Q&A, summarization, simple coding

### Phi-2 (2.7B)
- **Good for**: Most tasks, surprisingly capable
- **Not good for**: Extremely specialized domains
- **Example use**: Coding, reasoning, general chat

## Troubleshooting Low-Resource Issues

### "Out of Memory" Error

1. Reduce batch size to 1
2. Reduce max_length to 128 or 256
3. Use smaller model
4. Enable gradient checkpointing
5. Close other applications

### "Disk Space Full" Error

1. Delete old checkpoints: `rm -rf outputs/checkpoint-*`
2. Use `save_total_limit: 1`
3. Clear pip cache: `pip cache purge`
4. Use smaller model

### Training Too Slow

1. Use fewer examples (20-50)
2. Reduce max_length
3. Train for 1 epoch only
4. Use CPU config if GPU is too slow
5. Consider Google Colab (free GPU)

## Summary: Minimum Requirements

**Absolute Minimum (CPU Only):**
- 8GB RAM
- 5GB free storage
- 1-2 hours training time
- Use: `config/cpu_only_config.yaml`

**Recommended Minimum (with GPU):**
- 4GB VRAM GPU
- 8GB RAM
- 10GB free storage
- 20-30 minutes training time
- Use: `config/minimal_config.yaml`

**Sweet Spot (Budget GPU):**
- 6-8GB VRAM GPU
- 16GB RAM
- 15GB free storage
- 30-45 minutes training time
- Use: `config/quick_test_config.yaml`

---

**Bottom Line:** You can absolutely use this framework on modest hardware! Start with the CPU or minimal config, and you'll be fine-tuning in minutes.

