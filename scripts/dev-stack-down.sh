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

extra_args=(--remove-orphans)
if [[ "${1:-}" == "--purge-volumes" ]]; then
  echo "[dev-stack-down] --purge-volumes enabled: removing dev volumes for $PROJECT_NAME"
  extra_args+=(--volumes)
fi

echo "[dev-stack-down] Stopping isolated dev stack: $PROJECT_NAME"
echo "[dev-stack-down] Safety: this script targets only docker/docker-compose.dev.yaml"
"${compose_cmd[@]}" down "${extra_args[@]}"
