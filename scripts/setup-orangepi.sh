#!/usr/bin/env bash
# Non-root setup for the Orange Pi 5 (RK3588). Root steps: docs/PREREQUISITES.md
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)

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
pip install torch torchvision ai-edge-litert scipy openvino

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
pip install "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp"${PY_VER}"-*.whl

# Pre-compiled model and test image, into the repo's assets directory
mkdir -p "$REPO/assets"
cp "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/resnet18_for_rk3588.rknn "$REPO/assets/"
cp "$HOME"/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/space_shuttle_224.jpg "$REPO/assets/"

# BirdNET V2.4 FP32 TFLite, the model BirdNET-Go bundles
mkdir -p "$REPO/models"
curl -L -o "$REPO/models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite" \
  https://github.com/tphakala/birdnet-go/raw/main/internal/classifier/data/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite

# YOLOv5s: Rockchip's NPU-friendly ONNX, anchors, sample image, calibration images
cd "$HOME"
if [ -d rknn_model_zoo ]; then
  git -C rknn_model_zoo pull --ff-only
else
  git clone --depth 1 --filter=blob:none --sparse https://github.com/airockchip/rknn_model_zoo.git
  git -C rknn_model_zoo sparse-checkout set datasets/COCO examples/yolov5
fi
ZOO="$HOME/rknn_model_zoo/examples/yolov5"
curl -L -o "$REPO/models/yolov5s_relu.onnx" \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5s_relu.onnx
cp "$ZOO/model/anchors_yolov5.txt" "$REPO/models/"
cp "$ZOO/model/bus.jpg" "$REPO/assets/"

# Full RKNN-Toolkit2 (ARM64 build, ONNX/PyTorch only) in its own venv: it pins
# numpy and torch versions that conflict with rknn-env. torch itself is not needed.
python3 -m venv rknn-build-env
source rknn-build-env/bin/activate
pip install --upgrade pip
pip install -r <(grep -v '^torch' "$HOME/rknn-toolkit2/rknn-toolkit2/packages/arm64/arm64_requirements_cp${PY_VER}.txt")
pip install --no-deps "$HOME"/rknn-toolkit2/rknn-toolkit2/packages/arm64/rknn_toolkit2-*-cp"${PY_VER}"-*.whl

# INT8 (calibrated on the model zoo's 20 COCO images) and FP16 builds for RK3588
cd "$ZOO/python"
python3 convert.py "$REPO/models/yolov5s_relu.onnx" rk3588 i8 "$REPO/assets/yolov5s_relu_i8.rknn"
python3 convert.py "$REPO/models/yolov5s_relu.onnx" rk3588 fp "$REPO/assets/yolov5s_relu_fp16.rknn"
