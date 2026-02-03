# START HERE - Choose Your Path

Welcome! This framework works on **ANY hardware** - from CPU-only laptops to high-end GPUs.

## Step 1: What Hardware Do You Have?

### Path A: No GPU or Very Limited Resources

**Your Hardware:**
- No GPU, or GPU with less than 4GB VRAM
- 8GB RAM
- 5-10GB free storage

**What You'll Use:**
- Model: DistilGPT2 (82M parameters) or TinyLlama (1.1B)
- Config: `cpu_only_config.yaml` or `minimal_config.yaml`
- Training Time: 30-90 minutes
- Quality: Basic to Good

**Quick Start:**
```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install torch transformers datasets peft accelerate pyyaml tqdm

# 2. Check your system
python examples/minimal_example.py

# 3. Create tiny dataset
# (This creates just 10 examples - perfect for testing!)

# 4. Train
python src/finetune.py --config config/cpu_only_config.yaml

# 5. Test
python src/inference.py --model_path outputs/cpu/final_model
```

**Read:** [LOW_RESOURCE_GUIDE.md](LOW_RESOURCE_GUIDE.md)

---

### Path B: Budget GPU (4-6GB VRAM)

**Your Hardware:**
- GPU: GTX 1650, RTX 3050, or similar (4-6GB VRAM)
- 8-16GB RAM
- 10-15GB free storage

**What You'll Use:**
- Model: TinyLlama (1.1B) or Phi-2 (2.7B)
- Config: `minimal_config.yaml` or `quick_test_config.yaml`
- Training Time: 15-40 minutes
- Quality: Good to Very Good

**Quick Start:**
```bash
# 1. Setup
./scripts/setup.sh
source venv/bin/activate

# 2. Check your system
python examples/minimal_example.py

# 3. Create sample data
python src/data_preparation.py

# 4. Train
python src/finetune.py --config config/minimal_config.yaml

# 5. Test
python src/inference.py --model_path outputs/minimal/final_model
```

**Read:** [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md)

---

### Path C: Mid-Range GPU (6-10GB VRAM)

**Your Hardware:**
- GPU: RTX 3060, RTX 2060, RTX 4060 (6-10GB VRAM)
- 16GB RAM
- 15-20GB free storage

**What You'll Use:**
- Model: Phi-2 (2.7B) or Llama-2-7B (7B)
- Config: `quick_test_config.yaml` or `training_config.yaml`
- Training Time: 20-60 minutes
- Quality: Very Good to Excellent

**Quick Start:**
```bash
# 1. Setup
./scripts/setup.sh
source venv/bin/activate

# 2. Create sample data
python examples/quick_start.py

# This will:
# - Create sample dataset
# - Train the model
# - Show you how to test it
```

**Read:** [GETTING_STARTED.md](GETTING_STARTED.md)

---

### Path D: High-End GPU (10GB+ VRAM)

**Your Hardware:**
- GPU: RTX 3080, RTX 4070, RTX 4080, RTX 4090 (10GB+ VRAM)
- 32GB+ RAM
- 30GB+ free storage

**What You'll Use:**
- Model: Llama-2-7B or Llama-2-13B
- Config: `training_config.yaml` or `production_config.yaml`
- Training Time: 30-120 minutes
- Quality: Excellent to Best

**Quick Start:**
```bash
# 1. Setup
./scripts/setup.sh
source venv/bin/activate

# 2. Create or prepare your dataset
python examples/custom_dataset_example.py

# 3. Train
./scripts/train.sh --config config/training_config.yaml

# 4. Evaluate
python src/evaluate.py --model_path outputs/final_model --dataset data/eval.jsonl

# 5. Chat
python src/inference.py --model_path outputs/final_model
```

**Read:** [README.md](README.md) (full documentation)

---

### Path E: No Local Hardware (Use Free Cloud)

**Use Google Colab or Kaggle:**
- Free GPU access (Tesla T4 or P100)
- 12-16GB VRAM
- No installation needed

**Quick Start:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run:
```python
!git clone <your-repo-url>
%cd AI-Fine-tunning
!pip install -r requirements.txt
!python src/data_preparation.py
!python src/finetune.py --config config/quick_test_config.yaml
```

**Read:** [LOW_RESOURCE_GUIDE.md](LOW_RESOURCE_GUIDE.md) (Cloud section)

---

## Step 2: What Do You Want to Build?

- **Chatbot**: Use Alpaca format, train on conversation examples
- **Code Assistant**: Use examples from `custom_dataset_example.py`
- **Summarizer**: Train on text-summary pairs
- **Q&A System**: Train on question-answer pairs
- **Translator**: Train on translation pairs

## Step 3: Key Files to Know

| File | Purpose |
|------|---------|
| `START_HERE.md` | This file - choose your path |
| `HARDWARE_REQUIREMENTS.md` | Detailed hardware comparison |
| `LOW_RESOURCE_GUIDE.md` | Guide for limited hardware |
| `GETTING_STARTED.md` | Step-by-step tutorial |
| `README.md` | Complete documentation |
| `config/*.yaml` | Configuration files |
| `examples/*.py` | Working examples |

## Step 4: Run Your First Training

**Absolute Minimum (works on ANY computer):**
```bash
python examples/minimal_example.py  # Creates tiny dataset
python src/finetune.py --config config/cpu_only_config.yaml
```

**This will work even on a laptop with no GPU!**

## Need Help?

1. **Check your hardware**: `python examples/minimal_example.py`
2. **Read the guide for your hardware tier**: See paths above
3. **Start with tiny dataset**: Just 10-20 examples to test
4. **Monitor memory**: Watch RAM/VRAM usage during training
5. **Start small, scale up**: Begin with smallest model, then upgrade

## Summary

- **No GPU?** → Use `cpu_only_config.yaml` with DistilGPT2
- **4GB GPU?** → Use `minimal_config.yaml` with TinyLlama
- **6-8GB GPU?** → Use `quick_test_config.yaml` with Phi-2
- **8GB+ GPU?** → Use `training_config.yaml` with Llama-2-7B
- **No hardware?** → Use Google Colab (free!)

**You can start fine-tuning in the next 10 minutes, regardless of your hardware!**

