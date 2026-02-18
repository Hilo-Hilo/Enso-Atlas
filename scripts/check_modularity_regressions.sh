#!/usr/bin/env bash
set -euo pipefail

if command -v pytest >/dev/null 2>&1; then
  PYTEST_CMD=(pytest)
elif [ -x ".venv/bin/python" ]; then
  PYTEST_CMD=(.venv/bin/python -m pytest)
else
  PYTEST_CMD=(python3 -m pytest)
fi

"${PYTEST_CMD[@]}" \
  tests/test_model_scope.py \
  tests/test_frontend_heatmap_proxy_scoping.py \
  tests/test_heatmap_grid_alignment.py \
  -q
