#!/usr/bin/env bash
# Non-root setup for the Orange Pi 5 (RK3588). Root steps: docs/PREREQUISITES.md
set -euo pipefail

cd "$HOME"
if [ -d rknn-toolkit2 ]; then
  git -C rknn-toolkit2 pull --ff-only
else
  git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1
fi

python3 -m venv rknn-env
source rknn-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install opencv-python-headless numpy
pip install torch torchvision ai-edge-litert scipy

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
pip install "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp"${PY_VER}"-*.whl

# Pre-compiled model and test image, into the repo's assets directory
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$REPO/assets"
cp "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/resnet18_for_rk3588.rknn "$REPO/assets/"
cp "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/space_shuttle_224.jpg "$REPO/assets/"

# BirdNET V2.4 FP32 TFLite, the model BirdNET-Go bundles
mkdir -p "$REPO/models"
curl -L -o "$REPO/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite" \
  https://github.com/tphakala/birdnet-go/raw/main/internal/classifier/data/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite
