#!/bin/bash
# Enso Atlas startup script
# Runs both the FastAPI backend and Gradio UI

set -e

echo "============================================"
echo "       Starting Enso Atlas v0.1.0"
echo "============================================"
echo ""

# Check CUDA
CUDA_STATUS=$(python -c 'import torch; print("YES" if torch.cuda.is_available() else "NO")')
echo "CUDA available: $CUDA_STATUS"
if [ "$CUDA_STATUS" = "YES" ]; then
    CUDA_DEVICE=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
    echo "CUDA device: $CUDA_DEVICE"
fi
echo ""

# Generate demo data if needed
if [ ! -d "/app/data/demo/embeddings" ]; then
    echo "Generating demo data..."
    python /app/scripts/generate_demo_data.py \
        --output /app/data/demo \
        --num-slides 10 \
        --patches-per-slide 500 \
        --train-model \
        --model-output /app/models/demo_clam.pt
    echo ""
fi

# Start FastAPI in background
echo "Starting FastAPI API server on port 8000..."
cd /app
uvicorn enso_atlas.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 1
done

# Check API health
curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || true
echo ""

# Start Gradio UI in foreground
echo "Starting Gradio UI on port 7860..."
python -m enso_atlas.ui.demo_app --port 7860

# If Gradio exits, kill API
kill $API_PID 2>/dev/null || true
