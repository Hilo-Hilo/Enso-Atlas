# Enso Atlas - Production Dockerfile
# Optimized for NVIDIA DGX Spark deployment (ARM64 + CUDA)

FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install PyTorch for ARM64 with CUDA support
# Using pip index for aarch64 builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages
RUN pip install --no-cache-dir \
    faiss-cpu \
    accelerate \
    openslide-python \
    gradio

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
ENV PYTHONPATH=/app/src

# Expose Gradio port (temporary - will be replaced with React frontend)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "-m", "enso_atlas.ui.demo_app"]
