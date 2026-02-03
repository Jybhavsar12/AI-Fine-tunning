# Multi-stage Dockerfile for AI Fine-Tuning Framework

# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs checkpoints logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Expose port for TensorBoard
EXPOSE 6006

CMD ["/bin/bash"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY examples/ /app/examples/

# Create directories
RUN mkdir -p data outputs checkpoints logs

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Default command
CMD ["python", "src/finetune.py", "--config", "config/training_config.yaml"]

