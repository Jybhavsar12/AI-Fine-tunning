"""
Unit tests for data preparation module.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preparation import DatasetPreparator, PromptTemplate


class TestPromptTemplate:
    """Test prompt template formatting."""
    
    def test_alpaca_format_with_input(self):
        """Test Alpaca format with input."""
        result = PromptTemplate.alpaca(
            instruction="Translate to Spanish",
            input_text="Hello",
            output="Hola"
        )
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert "Translate to Spanish" in result
        assert "Hello" in result
        assert "Hola" in result
    
    def test_alpaca_format_without_input(self):
        """Test Alpaca format without input."""
        result = PromptTemplate.alpaca(
            instruction="Say hello",
            output="Hello!"
        )
        assert "### Instruction:" in result
        assert "### Response:" in result
        assert "### Input:" not in result
    
    def test_chatml_format(self):
        """Test ChatML format."""
        result = PromptTemplate.chatml(
            instruction="Test instruction",
            output="Test output"
        )
        assert "<|im_start|>user" in result
        assert "<|im_start|>assistant" in result
        assert "<|im_end|>" in result
    
    def test_llama2_format(self):
        """Test Llama-2 format."""
        result = PromptTemplate.llama2(
            instruction="Test instruction",
            output="Test output"
        )
        assert "[INST]" in result
        assert "[/INST]" in result


class TestDatasetPreparator:
    """Test dataset preparation functionality."""
    
    def test_initialization(self):
        """Test DatasetPreparator initialization."""
        preparator = DatasetPreparator(prompt_format="alpaca", max_length=512)
        assert preparator.prompt_format == "alpaca"
        assert preparator.max_length == 512
    
    def test_load_from_jsonl(self):
        """Test loading from JSONL file."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"instruction": "Test", "output": "Result"}) + '\n')
            f.write(json.dumps({"instruction": "Test2", "output": "Result2"}) + '\n')
            temp_file = f.name
        
        try:
            preparator = DatasetPreparator()
            dataset = preparator.load_from_jsonl(temp_file)
            assert len(dataset) == 2
            assert dataset[0]['instruction'] == "Test"
        finally:
            os.unlink(temp_file)
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.jsonl")
            preparator = DatasetPreparator()
            preparator.create_sample_dataset(output_path, num_samples=10)
            
            assert os.path.exists(output_path)
            
            # Verify content
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 10
                
                # Check first line is valid JSON
                data = json.loads(lines[0])
                assert 'instruction' in data
                assert 'output' in data


def test_prompt_templates_exist():
    """Test that all prompt template methods exist."""
    assert hasattr(PromptTemplate, 'alpaca')
    assert hasattr(PromptTemplate, 'chatml')
    assert hasattr(PromptTemplate, 'llama2')
    assert hasattr(PromptTemplate, 'custom')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

