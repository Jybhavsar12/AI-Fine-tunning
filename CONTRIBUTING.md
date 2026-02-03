# Contributing to AI Fine-Tuning Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (OS, Python version, GPU, etc.)
- Relevant logs or error messages

### Suggesting Features

Feature requests are welcome! Please:
- Check if the feature already exists or is planned
- Clearly describe the use case
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Fine-tunning.git
   cd AI-Fine-tunning
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run linting
   black src/ examples/
   isort src/ examples/
   flake8 src/
   
   # Test data preparation
   python src/data_preparation.py
   
   # Validate configs
   python -c "import yaml; yaml.safe_load(open('config/training_config.yaml'))"
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add support for Gemma models
fix: resolve memory leak in data preparation
docs: update hardware requirements guide
```

## Code Style

- **Python**: Follow PEP 8
- **Line Length**: Max 127 characters
- **Formatting**: Use `black` and `isort`
- **Type Hints**: Use where appropriate
- **Docstrings**: Use Google style

Example:
```python
def format_dataset(self, dataset: Dataset, 
                  instruction_col: str = "instruction") -> Dataset:
    """
    Format dataset with the specified prompt template.
    
    Args:
        dataset: Input dataset to format
        instruction_col: Column name for instructions
        
    Returns:
        Formatted dataset with 'text' column
    """
    pass
```

## Testing

- Add tests for new features
- Ensure existing tests pass
- Test on different hardware if possible
- Validate configuration files

## Documentation

When adding features:
- Update relevant documentation files
- Add examples if applicable
- Update CHANGELOG.md
- Add docstrings to new functions/classes

## Development Setup

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/AI-Fine-tunning.git
cd AI-Fine-tunning
python3 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install black isort flake8 mypy pytest

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Project Structure

```
src/
├── data_preparation.py  # Dataset utilities
├── finetune.py         # Main training script
├── inference.py        # Inference and chat
└── evaluate.py         # Evaluation metrics

config/                 # Configuration files
examples/              # Example scripts
tests/                 # Unit tests (add here!)
```

## Questions?

- Open an issue for questions
- Check existing documentation
- Review closed issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉

