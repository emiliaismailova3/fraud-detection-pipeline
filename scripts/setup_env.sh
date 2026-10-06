#!/usr/bin/env bash
# Create the Python virtual environment and install all libraries.
# Usage (from the repo root):  bash scripts/setup_env.sh
set -euo pipefail

VENV="${VENV:-$HOME/.venvs/fraud}"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip
# CPU-only PyTorch: ~200 MB instead of ~2 GB of CUDA libraries
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -c "import torch, xgboost, lightgbm, pandas, fastapi; print('environment OK')"
