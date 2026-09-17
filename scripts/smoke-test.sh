#!/usr/bin/env bash
# Offline check: no boat, no VRM, no weather API, no Qwen.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

.venv/bin/ruff check src tests scripts 2>/dev/null || echo "ruff not installed; skipping lint"
.venv/bin/python -m pytest -q
.venv/bin/python -m api.cli smoke
echo "smoke test passed"
