#!/usr/bin/env bash
# Assemble a Hugging Face Docker Space from this repo.
# HF builds the Dockerfile at the Space root and reads settings from README.md,
# so we copy only what the inference image needs into one flat folder.
# Usage (from the repo root):  bash scripts/build_hf_space.sh
set -euo pipefail

OUT=build/hf_space
rm -rf "$OUT"
mkdir -p "$OUT/src/api" "$OUT/models"

cp deploy/hf_space/README.md "$OUT/README.md"
cp docker/Dockerfile "$OUT/Dockerfile"
cp requirements-api.txt "$OUT/"
cp src/__init__.py src/config.py src/preprocess.py "$OUT/src/"
cp -r src/api/. "$OUT/src/api/"
cp models/preprocessor.joblib models/lightgbm.txt models/api_config.json "$OUT/models/"
find "$OUT" -name "__pycache__" -prune -exec rm -rf {} +

echo "Space folder ready: $OUT"
find "$OUT" -type f | sort
