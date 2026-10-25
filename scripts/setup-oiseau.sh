#!/usr/bin/env bash
# Non-root setup for the Intel N150 mini-PC.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

python3 -m venv bench-env
source bench-env/bin/activate
pip install --upgrade pip
pip install torch torchvision opencv-python-headless numpy
pip install openvino ai-edge-litert scipy

# BirdNET V2.4 FP32 TFLite, the model BirdNET-Go bundles
mkdir -p "$REPO/models"
curl -L -o "$REPO/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite" \
  https://github.com/tphakala/birdnet-go/raw/main/internal/classifier/data/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
