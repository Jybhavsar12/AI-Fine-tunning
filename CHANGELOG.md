# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of AI Fine-Tuning Framework
- Support for LoRA and QLoRA fine-tuning
- Multiple model configurations (CPU-only, minimal, quick test, production)
- Comprehensive documentation suite
- CI/CD pipeline with GitHub Actions
- Docker support for containerized training
- Example scripts for quick start
- Evaluation and inference tools

### Features
- **Data Preparation**: Support for JSONL, JSON, CSV, and HuggingFace datasets
- **Prompt Templates**: Alpaca, ChatML, Llama-2, and custom formats
- **Model Support**: DistilGPT2, TinyLlama, Phi-2, Llama-2, Mistral, and more
- **Hardware Flexibility**: Works on CPU-only, 4GB GPU, to high-end setups
- **Monitoring**: TensorBoard and Weights & Biases integration
- **Evaluation**: ROUGE, BLEU, and exact match metrics

### Documentation
- README.md - Complete framework documentation
- START_HERE.md - Quick start guide with hardware paths
- GETTING_STARTED.md - Step-by-step tutorial
- LOW_RESOURCE_GUIDE.md - Guide for limited hardware
- HARDWARE_REQUIREMENTS.md - Detailed hardware comparison
- PROJECT_OVERVIEW.md - High-level project overview

### Configuration Files
- training_config.yaml - Default configuration for Llama-2 7B
- minimal_config.yaml - For 4GB GPU systems
- cpu_only_config.yaml - For CPU-only training
- quick_test_config.yaml - For rapid testing with Phi-2
- production_config.yaml - Production-optimized settings

### Examples
- quick_start.py - Complete workflow example
- custom_dataset_example.py - Custom dataset creation
- minimal_example.py - Hardware detection and minimal setup
- model_comparison.py - Compare multiple models

### CI/CD
- Automated testing on Python 3.8, 3.9, 3.10, 3.11
- Code linting with flake8, black, isort
- Security scanning with bandit
- Documentation validation
- Docker image building
- Automated releases

## [1.0.0] - 2026-02-03

### Added
- Initial public release

