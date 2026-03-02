#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.dev.yaml"
PROJECT_NAME="enso-atlas-dev"
DEFAULT_ENV_FILE="$ROOT_DIR/docker/.env.dev"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: $COMPOSE_FILE not found." >&2
  exit 1
fi

compose_cmd=(docker compose --project-name "$PROJECT_NAME" -f "$COMPOSE_FILE")
if [[ -f "$DEFAULT_ENV_FILE" ]]; then
  compose_cmd+=(--env-file "$DEFAULT_ENV_FILE")
fi

echo "[dev-stack-up] Starting isolated dev stack: $PROJECT_NAME"
echo "[dev-stack-up] Compose file: $COMPOSE_FILE"
echo "[dev-stack-up] Safety: no schema migrations against production DB without explicit approval."

"${compose_cmd[@]}" config >/dev/null
"${compose_cmd[@]}" up -d --build
"${compose_cmd[@]}" ps

echo
echo "Dev API expected at: http://localhost:18003/api/health"
echo "Dev UI port mapping: localhost:17862 -> container:7860"
echo "Dev Postgres mapping: localhost:15433 -> container:5432"
