# Enso Atlas Isolated Dev Stack Runbook (DGX Perf Lab)

This runbook defines how to run an **isolated development stack** for performance testing on DGX without touching production services.

## Scope & Safety

- Uses `docker/docker-compose.dev.yaml` only.
- Uses separate container names, ports, and volumes from the primary stack.
- Targets a dedicated dev Postgres database (`atlas-dev-db`, `enso_atlas_dev`).
- **No schema migrations against production DB are allowed without explicit approval.**

## Files

- Compose file: `docker/docker-compose.dev.yaml`
- Startup wrapper: `scripts/dev-stack-up.sh`
- Shutdown wrapper: `scripts/dev-stack-down.sh`
- Optional env template: `docker/.env.dev.example`

## 1) Prepare host (DGX)

From repo root:

```bash
cp docker/.env.dev.example docker/.env.dev
# Edit docker/.env.dev if you need custom paths/credentials/ports
```

Recommended directories (if using defaults):

```bash
mkdir -p data-dev outputs-dev .cache/huggingface
```

## 2) Start isolated dev stack

```bash
./scripts/dev-stack-up.sh
```

This starts a compose project named `enso-atlas-dev`.

### Port map (dev stack)

- API health endpoint: `http://localhost:18003/api/health`
- UI mapping: `17862 -> 7860`
- Postgres mapping: `15433 -> 5432`

## 3) Health checks

### Basic container status

```bash
docker compose --project-name enso-atlas-dev -f docker/docker-compose.dev.yaml ps
```

### API health

```bash
curl -fsS http://localhost:18003/api/health
```

### DB readiness

```bash
docker exec atlas-dev-db pg_isready -U enso_dev -d enso_atlas_dev
```

## 4) Stop stack

Standard stop (keeps volumes/data):

```bash
./scripts/dev-stack-down.sh
```

Stop and remove dev volumes (destructive to dev data only):

```bash
./scripts/dev-stack-down.sh --purge-volumes
```

## 5) Rollback / Recovery

If a dev stack update causes issues:

1. Stop dev stack:
   ```bash
   ./scripts/dev-stack-down.sh
   ```
2. Revert code/compose changes to last known-good commit:
   ```bash
   git checkout <known-good-commit> -- docker/docker-compose.dev.yaml scripts/dev-stack-up.sh scripts/dev-stack-down.sh docs/dev-stack-runbook.md docker/.env.dev.example
   ```
3. Start again:
   ```bash
   ./scripts/dev-stack-up.sh
   ```

## 6) Guardrails checklist

- [ ] Using `docker/docker-compose.dev.yaml` (not prod compose)
- [ ] Dev container names (`enso-atlas-dev`, `atlas-dev-db`)
- [ ] Dev-only ports (`17862`, `18003`, `15433`)
- [ ] Dev DB URL points to `atlas-dev-db`
- [ ] No schema migrations against production DB without explicit approval
