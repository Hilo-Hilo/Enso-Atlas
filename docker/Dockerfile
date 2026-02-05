# Enso Atlas - Production Dockerfile
# Optimized for NVIDIA DGX Spark deployment (ARM64 + CUDA)
# Using TF 25.02 for Blackwell GPU support (sm_121)
#
# Build: docker build --network=host -t enso-atlas .
# Run:   docker run --gpus all --ipc=host -p 7860:7860 -v ./models:/app/models enso-atlas

# NVIDIA TensorFlow 25.02 container - has Blackwell GPU support
# TF 2.17.0 with CUDA kernels for sm_121 (GB10)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Print base image TensorFlow build info
RUN python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')" || true

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

# Install Python dependencies (includes PyTorch for TransMIL)
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support for TransMIL model
# Using pip index for ARM64 compatibility
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install additional packages
RUN pip install --no-cache-dir \
    faiss-cpu \
    accelerate \
    openslide-python \
    gradio \
    fastapi \
    uvicorn[standard] \
    pydantic

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY transmil.py ./

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/cache /app/models

# Set environment variables
ENV ENSO_CONFIG=/app/config/default.yaml
ENV TRANSFORMERS_CACHE=/app/cache/huggingface
ENV HF_HOME=/app/cache/huggingface
ENV PYTHONPATH=/app/src:/app

# Expose ports (Gradio UI and FastAPI)
EXPOSE 7860 8000

# Copy startup script
COPY scripts/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Health check (check API endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - runs both UI and API
CMD ["/app/start.sh"]
