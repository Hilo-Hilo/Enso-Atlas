#!/bin/bash
# Enso Atlas startup script
# Runs the FastAPI backend (Gradio currently disabled)

set -euo pipefail

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

# Generate demo data if needed (best-effort)
DEMO_EMBEDDINGS_DIR="/app/data/demo/embeddings"
DEMO_GENERATOR_SCRIPT="/app/scripts/generate_demo_data.py"

if [ ! -d "$DEMO_EMBEDDINGS_DIR" ]; then
    if [ -f "$DEMO_GENERATOR_SCRIPT" ]; then
        echo "Demo embeddings not found. Generating demo data..."
        if ! python "$DEMO_GENERATOR_SCRIPT" \
            --output /app/data/demo \
            --num-slides 10 \
            --patches-per-slide 500 \
            --train-model \
            --model-output /app/models/demo_clam.pt; then
            echo "Warning: demo data generation failed; continuing startup without demo data."
        fi
    else
        echo "Demo data generator not found at $DEMO_GENERATOR_SCRIPT; skipping demo data generation."
    fi
    echo ""
fi

# Start FastAPI in background
echo "Starting FastAPI API server on port 8000..."
cd /app
uvicorn enso_atlas.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
HEALTH_URL="http://localhost:8000/api/health"
echo "Waiting for API to start (${HEALTH_URL})..."
for i in {1..30}; do
    if curl -fsS "$HEALTH_URL" > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 1
done

# Check API health
curl -fsS "$HEALTH_URL" | python -m json.tool 2>/dev/null || true
echo ""

# Gradio UI disabled due to cv2.dnn.DictValue compatibility issue with NVIDIA container
# The API server provides all functionality needed for the hackathon
echo "Gradio UI disabled - API server running on port 8000"
echo "Use API endpoints directly or access via frontend"
echo ""

# Keep container running - wait for API process
wait $API_PID
