#  AI Fine-Tuning Framework

A comprehensive, production-ready framework for fine-tuning Large Language Models (LLMs) with support for LoRA, QLoRA, and full fine-tuning.

##  Features

-  **Multiple Fine-Tuning Methods**: Full fine-tuning, LoRA, and QLoRA (4-bit/8-bit)
-  **Easy Configuration**: YAML-based configuration for all training parameters
-  **Multiple Prompt Formats**: Alpaca, ChatML, Llama-2, and custom templates
-  **Flexible Dataset Support**: JSONL, JSON, CSV, and HuggingFace datasets
-  **Comprehensive Evaluation**: ROUGE, BLEU, and exact match metrics
-  **Interactive Chat**: Test your models with an interactive chat interface
-  **Experiment Tracking**: Built-in support for Weights & Biases and TensorBoard
-  **Optimized Training**: Gradient accumulation, mixed precision, and distributed training

##  Project Structure

```
AI-Fine-tunning/
├── config/
│   └── training_config.yaml      # Training configuration
├── src/
│   ├── data_preparation.py       # Dataset utilities
│   ├── finetune.py              # Main training script
│   ├── inference.py             # Inference and chat
│   └── evaluate.py              # Evaluation metrics
├── examples/
│   ├── quick_start.py           # Quick start example
│   └── custom_dataset_example.py # Custom dataset example
├── scripts/
│   ├── setup.sh                 # Environment setup
│   └── train.sh                 # Training script
├── data/                        # Your datasets go here
├── outputs/                     # Trained models
└── requirements.txt             # Python dependencies
```

##  Quick Start

**Limited Hardware?** See [LOW_RESOURCE_GUIDE.md](LOW_RESOURCE_GUIDE.md) and [HARDWARE_REQUIREMENTS.md](HARDWARE_REQUIREMENTS.md) for CPU-only and minimal GPU options!

### 1. Setup Environment

```bash
# Run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Check your hardware:**
```bash
python examples/minimal_example.py  # Recommends best config for your system
```

### 2. Prepare Your Data

Create a dataset in JSONL format with the following structure:

```json
{"instruction": "Your task description", "input": "Optional context", "output": "Expected response"}
```

Or use the built-in sample data generator:

```bash
python src/data_preparation.py
```

### 3. Configure Training

Edit `config/training_config.yaml` to set:
- Base model (e.g., `meta-llama/Llama-2-7b-hf`, `mistralai/Mistral-7B-v0.1`)
- LoRA parameters
- Training hyperparameters
- Dataset paths

### 4. Start Fine-Tuning

```bash
# Using the training script
chmod +x scripts/train.sh
./scripts/train.sh

# Or directly
python src/finetune.py --config config/training_config.yaml
```

### 5. Test Your Model

```bash
# Interactive chat
python src/inference.py --model_path outputs/final_model

# Single prompt
python src/inference.py --model_path outputs/final_model --prompt "Explain quantum computing"

# Evaluate on test set
python src/evaluate.py --model_path outputs/final_model --dataset data/eval.jsonl
```

## 📚 Detailed Usage

### Dataset Preparation

The framework supports multiple dataset formats and prompt templates:

```python
from data_preparation import DatasetPreparator

# Initialize with your preferred prompt format
preparator = DatasetPreparator(prompt_format="alpaca", max_length=2048)

# Load from various sources
dataset = preparator.load_from_jsonl("data/train.jsonl")
dataset = preparator.load_from_huggingface("tatsu-lab/alpaca")

# Format with template
formatted = preparator.format_dataset(dataset)
```

### Supported Prompt Formats

1. **Alpaca**: Standard instruction-following format
2. **ChatML**: Modern chat format (`<|im_start|>user...`)
3. **Llama-2**: Llama-2 chat format (`[INST]...`)
4. **Custom**: Define your own template

### Training Configuration

Key configuration options in `training_config.yaml`:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"
  load_in_4bit: true  # Enable QLoRA

lora:
  enabled: true
  r: 16              # LoRA rank
  lora_alpha: 32
  target_modules:    # Which layers to apply LoRA
    - q_proj
    - v_proj

training:
  num_train_epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

### Fine-Tuning Methods

#### 1. QLoRA (Recommended for Consumer GPUs)

Most memory-efficient method using 4-bit quantization:

```yaml
model:
  load_in_4bit: true

lora:
  enabled: true
  r: 16
```

**Memory Requirements**: ~6-8GB VRAM for 7B models

#### 2. LoRA

Efficient fine-tuning without quantization:

```yaml
model:
  load_in_4bit: false

lora:
  enabled: true
  r: 16
```

**Memory Requirements**: ~16-20GB VRAM for 7B models

#### 3. Full Fine-Tuning

Train all model parameters:

```yaml
lora:
  enabled: false
```

**Memory Requirements**: ~40-60GB VRAM for 7B models

##  Use Cases & Examples

### 1. Code Assistant

```python
# examples/custom_dataset_example.py
python examples/custom_dataset_example.py
```

### 2. Question Answering

Create a dataset with Q&A pairs:

```json
{"instruction": "Answer the question", "input": "What is photosynthesis?", "output": "Photosynthesis is..."}
```

### 3. Text Summarization

```json
{"instruction": "Summarize the text", "input": "Long article...", "output": "Brief summary..."}
```

### 4. Language Translation

```json
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

##  Evaluation Metrics

The framework provides comprehensive evaluation:

- **ROUGE**: Measures overlap with reference text
- **BLEU**: Translation quality metric
- **Exact Match**: Percentage of perfect matches

```bash
python src/evaluate.py \
  --model_path outputs/final_model \
  --dataset data/eval.jsonl \
  --num_samples 100 \
  --output results.json
```

##  Advanced Features

### Custom Prompt Templates

Define your own prompt format:

```python
from data_preparation import PromptTemplate

custom_template = "{instruction}\n\nContext: {input}\n\nAnswer: {output}"
formatted = PromptTemplate.custom(instruction, input_text, output, template=custom_template)
```

### Experiment Tracking

Enable Weights & Biases:

```yaml
training:
  report_to: ["wandb"]
```

Then login:
```bash
wandb login
```

### Multi-GPU Training

The framework automatically uses all available GPUs with `device_map="auto"`.

## 💡 Best Practices

1. **Start Small**: Test with a small model (phi-2, TinyLlama) before scaling up
2. **Quality over Quantity**: 100 high-quality examples > 10,000 low-quality ones
3. **Validate Your Data**: Always check formatted prompts before training
4. **Monitor Training**: Use TensorBoard or W&B to track loss
5. **Evaluate Regularly**: Run evaluation on a held-out test set
6. **Experiment with LoRA Rank**: Try r=8, 16, 32, 64 for different tasks
7. **Adjust Learning Rate**: Start with 2e-4 for LoRA, 5e-6 for full fine-tuning

## 🐛 Troubleshooting

### Out of Memory (OOM)

- Enable 4-bit quantization (`load_in_4bit: true`)
- Reduce batch size
- Increase gradient accumulation steps
- Reduce max sequence length

### Poor Performance

- Increase training epochs
- Add more diverse training data
- Adjust learning rate
- Try different LoRA ranks
- Check data quality and formatting

### Slow Training

- Enable mixed precision (`bf16: true`)
- Use gradient checkpointing
- Increase batch size if memory allows
- Use flash attention (if supported)

##  Recommended Models

### Small Models (Good for Testing)
- `microsoft/phi-2` (2.7B)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B)

### Medium Models (Balanced)
- `meta-llama/Llama-2-7b-hf` (7B)
- `mistralai/Mistral-7B-v0.1` (7B)
- `google/gemma-7b` (7B)

### Large Models (Best Performance)
- `meta-llama/Llama-2-13b-hf` (13B)
- `mistralai/Mixtral-8x7B-v0.1` (47B with MoE)

##  Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

##  License

This project is open source and available under the MIT License.

##  Acknowledgments

Built with:
- [Transformers](https://github.com/huggingface/transformers) by HuggingFace
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- [TRL](https://github.com/huggingface/trl) for training utilities

##  Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [HuggingFace Documentation](https://huggingface.co/docs)

---

**Happy Fine-Tuning! 🎉**

For questions or issues, please open an issue on GitHub.


