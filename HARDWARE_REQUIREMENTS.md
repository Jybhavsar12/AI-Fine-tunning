# Hardware Requirements & Model Comparison

## Quick Reference Table

| Config | Model | Parameters | Storage | RAM | VRAM | Training Time* | Quality |
|--------|-------|------------|---------|-----|------|----------------|---------|
| **cpu_only_config.yaml** | DistilGPT2 | 82M | 2GB | 8GB | None | 60-90 min | Basic |
| **minimal_config.yaml** | TinyLlama | 1.1B | 5GB | 8GB | 4GB | 15-30 min | Good |
| **quick_test_config.yaml** | Phi-2 | 2.7B | 8GB | 16GB | 6GB | 20-40 min | Very Good |
| **training_config.yaml** | Llama-2-7B | 7B | 15GB | 16GB | 8GB | 30-60 min | Excellent |
| **production_config.yaml** | Llama-2-7B | 7B | 20GB | 32GB | 12GB | 60-120 min | Best |

*Training time for 100 examples, 1-3 epochs

## Detailed Breakdown

### CPU-Only Setup (No GPU Required)

**Hardware:**
- CPU: Any modern processor (4+ cores recommended)
- RAM: 8GB minimum, 16GB recommended
- Storage: 5GB free space
- GPU: Not required

**Configuration:** `config/cpu_only_config.yaml`

**Model:** DistilGPT2 (82 million parameters)

**What to Expect:**
- Download size: ~300MB
- Training time: 1-2 hours for 50 examples
- Quality: Basic but functional
- Good for: Testing, simple tasks, learning

**Pros:**
- Works on ANY computer
- No GPU needed
- Minimal storage
- Free to run

**Cons:**
- Slower training
- Lower quality outputs
- Limited context length

**Best For:**
- Testing the framework
- Learning how fine-tuning works
- Very simple tasks (greetings, basic Q&A)
- Systems without GPU

---

### Minimal GPU Setup

**Hardware:**
- GPU: 4GB VRAM (GTX 1050 Ti, GTX 1650, RTX 3050 4GB)
- RAM: 8GB
- Storage: 10GB free space

**Configuration:** `config/minimal_config.yaml`

**Model:** TinyLlama-1.1B (1.1 billion parameters)

**What to Expect:**
- Download size: ~2GB
- Training time: 20-30 minutes for 100 examples
- Quality: Good for most basic tasks
- Good for: Chatbots, simple Q&A, basic coding help

**Pros:**
- Works on budget GPUs
- Reasonable quality
- Fast training
- Low storage needs

**Cons:**
- Limited reasoning ability
- Shorter context window
- Not great for complex tasks

**Best For:**
- Budget hardware
- Quick experiments
- Simple chatbots
- Basic task automation

---

### Mid-Range Setup (Recommended)

**Hardware:**
- GPU: 6-8GB VRAM (RTX 3060, RTX 2060, RTX 4060)
- RAM: 16GB
- Storage: 15GB free space

**Configuration:** `config/quick_test_config.yaml`

**Model:** Phi-2 (2.7 billion parameters)

**What to Expect:**
- Download size: ~5GB
- Training time: 30-45 minutes for 100 examples
- Quality: Very good, surprisingly capable
- Good for: Coding, reasoning, general chat, Q&A

**Pros:**
- Excellent quality-to-size ratio
- Fast training
- Good reasoning abilities
- Works on consumer GPUs

**Cons:**
- Still limited vs larger models
- Needs decent GPU

**Best For:**
- Most users
- Production use cases
- Coding assistants
- General-purpose chatbots

---

### High-End Setup

**Hardware:**
- GPU: 8-12GB VRAM (RTX 3080, RTX 4070, RTX 4080)
- RAM: 16-32GB
- Storage: 20GB free space

**Configuration:** `config/training_config.yaml`

**Model:** Llama-2-7B (7 billion parameters)

**What to Expect:**
- Download size: ~14GB
- Training time: 30-60 minutes for 100 examples
- Quality: Excellent
- Good for: Complex reasoning, coding, specialized domains

**Pros:**
- High quality outputs
- Good reasoning
- Large context window
- Industry-standard model

**Cons:**
- Needs good GPU
- Larger storage
- Longer download time

**Best For:**
- Professional use
- Complex tasks
- High-quality requirements
- Production deployments

---

### Professional Setup

**Hardware:**
- GPU: 16-24GB VRAM (RTX 4090, A5000, A6000)
- RAM: 32-64GB
- Storage: 50GB free space

**Configuration:** `config/production_config.yaml`

**Model:** Llama-2-7B or Llama-2-13B

**What to Expect:**
- Download size: 14-26GB
- Training time: 1-3 hours for 500+ examples
- Quality: Best possible
- Good for: Any task

**Pros:**
- Maximum quality
- Can train larger models
- More training epochs
- Better fine-tuning

**Cons:**
- Expensive hardware
- Large storage needs
- Longer training

**Best For:**
- Production systems
- Research
- Maximum quality requirements
- Large-scale deployments

---

## Free Cloud Alternatives

If you don't have the hardware, use free cloud GPUs:

### Google Colab (Free Tier)

**Specs:**
- GPU: Tesla T4 (15GB VRAM)
- RAM: 12GB
- Storage: 100GB
- Time Limit: 12 hours per session

**How to Use:**
1. Go to colab.research.google.com
2. Upload your code or clone from GitHub
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run your training

**Recommended Config:** `quick_test_config.yaml` or `training_config.yaml`

### Kaggle Notebooks (Free)

**Specs:**
- GPU: Tesla P100 (16GB VRAM)
- RAM: 13GB
- Storage: 73GB
- Time Limit: 30 hours/week

**How to Use:**
1. Go to kaggle.com
2. Create new notebook
3. Enable GPU in settings
4. Upload your code and run

**Recommended Config:** `training_config.yaml`

---

## Which Config Should You Use?

### Decision Tree:

```
Do you have a GPU?
│
├─ NO → Use cpu_only_config.yaml
│
└─ YES → How much VRAM?
    │
    ├─ < 4GB → Use cpu_only_config.yaml (GPU too small)
    │
    ├─ 4-6GB → Use minimal_config.yaml
    │
    ├─ 6-10GB → Use quick_test_config.yaml
    │
    └─ 10GB+ → Use training_config.yaml
```

### By Use Case:

- **Just testing/learning**: cpu_only_config.yaml
- **Budget hardware**: minimal_config.yaml
- **Most users**: quick_test_config.yaml
- **Production/quality**: training_config.yaml
- **Maximum quality**: production_config.yaml

---

## Storage Breakdown

What takes up space:

1. **Base Model Download**: 300MB - 14GB (one-time)
2. **Training Checkpoints**: 2-10GB (can be deleted)
3. **Final Model**: Same as base model size
4. **Dependencies**: ~2GB (PyTorch, etc.)

**Total for minimal setup**: ~5GB
**Total for standard setup**: ~20GB
**Total for production**: ~50GB

---

## How to Check Your Hardware

Run this to see what you have:

```bash
python examples/minimal_example.py
```

This will:
- Detect your GPU (if any)
- Show available memory
- Recommend the best config for your system

