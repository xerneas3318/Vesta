#!/usr/bin/env bash
set -euo pipefail

PORT=33263

echo "Starting unified UI on port ${PORT}"
echo "Open: http://127.0.0.1:${PORT}"

uv run flask --app main:app run --host 0.0.0.0 --port "${PORT}"
