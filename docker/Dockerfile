# Enso Atlas - Production Dockerfile
# Optimized for NVIDIA DGX Spark deployment

FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    # OpenSlide dependencies
    openslide-tools \
    libopenslide0 \
    libopenslide-dev \
    # Image processing
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # General utilities
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install GPU-specific packages
RUN pip install --no-cache-dir \
    faiss-gpu \
    accelerate \
    bitsandbytes

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/cache /app/models

# Set environment variables
ENV ENSO_CONFIG=/app/config/default.yaml
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "-m", "enso_atlas.ui.app"]
