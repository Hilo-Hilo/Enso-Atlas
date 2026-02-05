# Enso Atlas - Production Dockerfile
# Optimized for NVIDIA DGX Spark deployment (ARM64 + CUDA)
# Using PyTorch base for proper CUDA support (MedGemma/Transformers needs PyTorch)
# TensorFlow for Path Foundation runs on CPU (acceptable - embeddings are pre-computed)
#
# Build: docker build --network=host -t enso-atlas .
# Run:   docker run --gpus all --ipc=host -p 7860:7860 -v ./models:/app/models enso-atlas

# NVIDIA PyTorch container - proper ARM64 + CUDA support for Blackwell
FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Print base image PyTorch build info
RUN python -c "import torch; print(f'PyTorch: {torch.__version__} CUDA: {torch.cuda.is_available()}')"

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

# Install TensorFlow for Path Foundation (CPU mode is acceptable)
# Embeddings are pre-computed, TF is only used for on-demand embedding
RUN pip install --no-cache-dir tensorflow

# Install additional packages
RUN pip install --no-cache-dir \
    faiss-cpu \
    accelerate \
    openslide-python \
    gradio \
    fastapi \
    uvicorn[standard] \
    pydantic

# Fix OpenCV compatibility issue (cv2.dnn.DictValue AttributeError)
# The NVIDIA base image has a newer OpenCV that breaks cv2.typing imports
RUN pip install opencv-python-headless==4.8.1.78 --force-reinstall

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
