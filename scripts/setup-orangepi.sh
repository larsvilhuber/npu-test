#!/usr/bin/env bash
# Non-root setup for the Orange Pi 5 (RK3588). Root steps: docs/PREREQUISITES.md
set -euo pipefail

cd "$HOME"
git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1

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
