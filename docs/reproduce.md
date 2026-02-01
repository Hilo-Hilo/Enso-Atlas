# Reproduce on NVIDIA DGX Spark (Docker)

## Prerequisites (on the DGX)

```bash
# Sanity checks
nvidia-smi
docker --version
docker compose version
```

Verify the NVIDIA Container Toolkit is working (GPU visible inside containers):

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Build

From the repo root:

```bash
make docker-build
```

(Equivalent: `docker compose -f docker/docker-compose.yaml build`.)

## Download models (required for offline mode)

The default Compose config runs in offline mode (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`). Populate the model volume once, then run offline.

```bash
# If needed for gated models
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"

docker compose -f docker/docker-compose.yaml run --rm \
  -e TRANSFORMERS_OFFLINE=0 \
  -e HF_HUB_OFFLINE=0 \
  -e HF_TOKEN="$HF_TOKEN" \
  enso-atlas python /app/scripts/download_models.py --cache-dir /app/models --models all
```

## Run

```bash
make docker-up
```

Check status / logs:

```bash
docker compose -f docker/docker-compose.yaml ps
docker logs -f enso-atlas
```

## Verify GPU access

```bash
docker exec -it enso-atlas python -c "import torch; print('cuda:', torch.cuda.is_available()); print('gpus:', torch.cuda.device_count())"
```

## Open the UI

- `http://<DGX_HOSTNAME_OR_IP>:7860`

## Stop

```bash
make docker-down
```

## Troubleshooting

### GPU not visible in container

1. Check nvidia-container-toolkit is installed:
   ```bash
   docker info | grep -i nvidia
   ```

2. Test with a minimal CUDA container:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

### Model download fails

- Ensure HF_TOKEN is set and the account has accepted model terms
- Check network connectivity
- Try downloading outside Docker first to verify token works

### Build fails on Python/faiss-gpu

The Dockerfile uses Python 3.10 (jammy default). If you see wheel issues:
- Consider using `faiss-cpu` instead of `faiss-gpu`
- Or switch to a PyTorch base image (see Dockerfile comments)
