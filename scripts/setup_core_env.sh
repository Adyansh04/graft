#!/usr/bin/env bash
# Sets up the core pipeline environment (CLI, config, dataset
# postprocessing, YOLO training) — NOT the Isaac Sim runtime, which lives
# in a separate venv for dependency isolation. See
# scripts/setup_isaac_sim_env.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync
echo "Core env ready: .venv/ (Python $(cat .python-version 2>/dev/null || echo '3.12'))"
echo "Run commands with: uv run <command>"
