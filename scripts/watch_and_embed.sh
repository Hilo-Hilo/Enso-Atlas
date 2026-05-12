#!/bin/bash
# Watch for download completion and start embedding generation

TARGET_COUNT=286
DATA_DIR="$HOME/med-gemma-hackathon/data/ovarian_bev"
LOG="$HOME/med-gemma-hackathon/watch_embed.log"

echo "[$(date)] Starting watch for $TARGET_COUNT slides..." | tee -a "$LOG"

while true; do
    COUNT=$(find "$DATA_DIR/slides" -name "*.svs" -size +0 2>/dev/null | wc -l)
    echo "[$(date)] Downloaded: $COUNT / $TARGET_COUNT" | tee -a "$LOG"
    
    if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
        echo "[$(date)] Download complete! Starting embedding generation..." | tee -a "$LOG"
        
        cd ~/med-gemma-hackathon
        
        # Generate embeddings
        docker run --gpus all --ipc=host --rm \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/scripts:/app/scripts \
            -v $(pwd)/src:/app/src \
            medgemma-embed python /app/scripts/generate_embeddings_batch.py \
                --input /app/data/ovarian_bev/slides \
                --output /app/data/ovarian_bev/embeddings \
                --batch-size 64 \
                --device cuda 2>&1 | tee -a embed_bev.log
        
        echo "[$(date)] Embeddings complete! Starting training..." | tee -a "$LOG"
        
        # Train model
        docker run --gpus all --ipc=host --rm \
            -v $(pwd)/data:/app/data \
            -v $(pwd)/scripts:/app/scripts \
            -v $(pwd)/src:/app/src \
            -v $(pwd)/outputs:/app/outputs \
            medgemma-embed python /app/scripts/train_bevacizumab.py \
                --data_dir /app/data/ovarian_bev \
                --output_dir /app/outputs/bevacizumab 2>&1 | tee -a train_bev.log
        
        echo "[$(date)] Training complete!" | tee -a "$LOG"
        break
    fi
    
    sleep 30
done
