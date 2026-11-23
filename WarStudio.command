#!/bin/bash
# Double-click to start WarStudio (opens http://127.0.0.1:8765 in your browser).
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up..."
  bash setup_mac.sh || exit 1
fi
uv run python run.py gui
