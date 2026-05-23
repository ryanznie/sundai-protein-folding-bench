#!/usr/bin/env bash
set -euo pipefail

cd /app

if [ -d "/opt/sundai/ml-simplefold" ]; then
  if /app/.venv/bin/python -c "import simplefold" >/dev/null 2>&1; then
    echo "[api] simplefold already available in container environment"
  else
    echo "[api] Installing mounted ml-simplefold runtime into container environment"
    rm -rf /tmp/ml-simplefold-src
    mkdir -p /tmp/ml-simplefold-src
    cp /opt/sundai/ml-simplefold/pyproject.toml /tmp/ml-simplefold-src/pyproject.toml
    cp /opt/sundai/ml-simplefold/README.md /tmp/ml-simplefold-src/README.md
    cp /opt/sundai/ml-simplefold/LICENSE /tmp/ml-simplefold-src/LICENSE
    cp -R /opt/sundai/ml-simplefold/src /tmp/ml-simplefold-src/src
    chmod -R u+w /tmp/ml-simplefold-src
    uv pip install --python /app/.venv/bin/python /tmp/ml-simplefold-src
  fi
else
  echo "[api] Mounted ml-simplefold runtime not found at /opt/sundai/ml-simplefold" >&2
fi

exec uv run uvicorn service.app:app --host 0.0.0.0 --port 8000
