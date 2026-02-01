# Enso Atlas - Production Dockerfile
# Optimized for NVIDIA DGX Spark deployment (ARM64 + CUDA)
# Reference: DeepSeek-OCR-Web Dockerfile (working ARM64 setup)
#
# Build: docker build --network=host -t enso-atlas .
# Run:   docker run --gpus all -p 7860:7860 -v ./models:/app/models enso-atlas

# NVIDIA PyTorch container with CUDA support (ARM64 compatible)
# Same base as DeepSeek OCR which works on DGX Spark
FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Print base image PyTorch build info
RUN python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopenslide0 \
    libopenslide-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

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

# Expose port (Gradio temporary, will be React frontend)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command
CMD ["python", "-m", "enso_atlas.ui.demo_app"]
