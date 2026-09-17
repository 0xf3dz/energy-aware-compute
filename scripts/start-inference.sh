#!/usr/bin/env bash
# Check the local llama.cpp server, and start one when no server answers.
#
# The server needs --metrics for the telemetry of this project. A server that a
# LaunchAgent already manages must be left alone.
set -euo pipefail

MODEL="${MODEL:-/Users/f3dz/dev/local-inference/models/Qwen3.5-9B-Q4_K_M.gguf}"
LLAMA="${LLAMA:-/Users/f3dz/dev/local-inference/runtime/llama-b10964/llama-server}"
ALIAS="${ALIAS:-qwen3.5-9b}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8082}"
CTX="${CTX_SIZE:-8192}"

if curl -fsS -m 3 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "llama-server answers on http://${HOST}:${PORT}"
  curl -fsS "http://${HOST}:${PORT}/metrics" | grep -c '^llamacpp:' | \
    sed 's/^/metrics available: /' || echo "warning: GET /metrics is not available; start the server with --metrics"
  exit 0
fi

if [ ! -x "$LLAMA" ]; then
  echo "llama-server not found at $LLAMA" >&2
  exit 1
fi

echo "Starting llama-server on http://${HOST}:${PORT}"
exec "$LLAMA" \
  --model "$MODEL" --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  --ctx-size "$CTX" --parallel 1 --n-gpu-layers 999 \
  --jinja --no-webui --metrics --slots
