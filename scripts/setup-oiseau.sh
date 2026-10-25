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
