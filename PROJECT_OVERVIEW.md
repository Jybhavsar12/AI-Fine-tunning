#  Project Overview

## What is This?

This is a **production-ready fine-tuning framework** for Large Language Models (LLMs). It allows you to customize pre-trained models like Llama-2, Mistral, or Phi-2 for your specific use cases.

## What Can You Do With This?

### 1. **Customize AI Models for Your Needs**
- Train a coding assistant
- Build a domain-specific chatbot
- Create a text summarization tool
- Develop a translation system
- Fine-tune for question answering

### 2. **Efficient Training Methods**
- **QLoRA**: Train 7B models on consumer GPUs (8GB VRAM)
- **LoRA**: Parameter-efficient fine-tuning
- **Full Fine-Tuning**: For maximum customization

### 3. **Easy to Use**
- Simple YAML configuration
- Pre-built examples
- Interactive testing
- Comprehensive evaluation

## Quick Start (5 Minutes)

```bash
# 1. Setup
./scripts/setup.sh
source venv/bin/activate

# 2. Create sample data
python src/data_preparation.py

# 3. Train (uses default config)
python src/finetune.py

# 4. Test your model
python src/inference.py --model_path outputs/final_model
```

## Project Structure

```
AI-Fine-tunning/
│
├── 📁 config/                    # Configuration files
│   ├── training_config.yaml      # Default configuration
│   ├── quick_test_config.yaml    # Fast testing setup
│   └── production_config.yaml    # Production-ready setup
│
├── 📁 src/                       # Core source code
│   ├── data_preparation.py       # Dataset utilities
│   ├── finetune.py              # Main training script
│   ├── inference.py             # Model testing & chat
│   └── evaluate.py              # Performance metrics
│
├── 📁 examples/                  # Example scripts
│   ├── quick_start.py           # Beginner-friendly example
│   ├── custom_dataset_example.py # Custom data tutorial
│   └── model_comparison.py      # Compare models
│
├── 📁 scripts/                   # Utility scripts
│   ├── setup.sh                 # Environment setup
│   └── train.sh                 # Training launcher
│
├── 📁 data/                      # Your datasets (created on first run)
├── 📁 outputs/                   # Trained models (created during training)
│
├── 📄 README.md                  # Full documentation
├── 📄 GETTING_STARTED.md         # Step-by-step tutorial
└── 📄 requirements.txt           # Python dependencies
```

## Key Features

###  **Performance**
- QLoRA: Train 7B models on 8GB VRAM
- Mixed precision training (BF16/FP16)
- Gradient accumulation for larger effective batch sizes
- Automatic multi-GPU support

###  **Flexibility**
- Multiple prompt formats (Alpaca, ChatML, Llama-2, Custom)
- Support for JSONL, JSON, CSV, and HuggingFace datasets
- Configurable LoRA parameters
- Easy model switching

### **Monitoring**
- TensorBoard integration
- Weights & Biases support
- Real-time loss tracking
- Comprehensive evaluation metrics

###  **Developer-Friendly**
- Clean, modular code
- Extensive documentation
- Example scripts
- Interactive chat mode

## Typical Workflow

```
1. Prepare Data
   ↓
2. Configure Training
   ↓
3. Fine-Tune Model
   ↓
4. Evaluate Performance
   ↓
5. Test Interactively
   ↓
6. Deploy or Iterate
```

## Configuration Files

### `training_config.yaml` (Default)
- Balanced settings for Llama-2 7B
- Good starting point for most use cases
- 3 epochs, LoRA rank 16

### `quick_test_config.yaml`
- Uses smaller Phi-2 model (2.7B)
- 1 epoch for rapid testing
- Lower LoRA rank for speed

### `production_config.yaml`
- Optimized for best quality
- Higher LoRA rank (32)
- More training epochs (5)
- W&B integration enabled

## Common Use Cases

### 1. **Code Assistant**
```python
# Train on code examples
{"instruction": "Write a function to...", "output": "def function()..."}
```

### 2. **Customer Support Bot**
```python
# Train on support conversations
{"instruction": "How do I reset my password?", "output": "To reset..."}
```

### 3. **Content Summarization**
```python
# Train on article-summary pairs
{"instruction": "Summarize:", "input": "Long text...", "output": "Summary..."}
```

### 4. **Domain Expert**
```python
# Train on domain-specific Q&A
{"instruction": "Explain medical term:", "input": "Hypertension", "output": "High blood pressure..."}
```

## System Requirements

### Minimum (Quick Testing)
- GPU: 8GB VRAM (RTX 3060, RTX 4060)
- RAM: 16GB
- Storage: 20GB free
- Model: Phi-2 (2.7B) with QLoRA

### Recommended (Production)
- GPU: 16GB+ VRAM (RTX 4080, A4000)
- RAM: 32GB
- Storage: 50GB free
- Model: Llama-2 7B with QLoRA

### Optimal (Best Quality)
- GPU: 24GB+ VRAM (RTX 4090, A5000)
- RAM: 64GB
- Storage: 100GB free
- Model: Llama-2 13B or Mixtral

## Next Steps

1. **Read the Documentation**
   - [README.md](README.md) - Full documentation
   - [GETTING_STARTED.md](GETTING_STARTED.md) - Step-by-step tutorial

2. **Run Examples**
   - `examples/quick_start.py` - Complete workflow
   - `examples/custom_dataset_example.py` - Custom data

3. **Experiment**
   - Try different models
   - Adjust LoRA parameters
   - Test various prompt formats

4. **Deploy**
   - Use `inference.py` in your applications
   - Export models for production
   - Share your fine-tuned models

## Support & Resources

- **Documentation**: See README.md and GETTING_STARTED.md
- **Examples**: Check the `examples/` directory
- **Configuration**: Review `config/` files with comments
- **Code**: All source code is in `src/` with docstrings

## License

MIT License - Free to use for personal and commercial projects.

---

**Ready to start?** Run `./scripts/setup.sh` and follow [GETTING_STARTED.md](GETTING_STARTED.md)!

