#!/usr/bin/env bash
# Create the virtual environment, install the package, and apply migrations.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3.12}"

cd "$HERE"
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e '.[dev]'

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Fill in the VRM token and the position."
fi

echo "Installed. Apply the schema with:"
echo "  .venv/bin/python -m db.migrate"
