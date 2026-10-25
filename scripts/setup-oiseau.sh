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

# YOLOv5s: Rockchip's NPU-friendly ONNX (the same file the NPU model is built from)
mkdir -p "$REPO/assets"
curl -L -o "$REPO/models/yolov5s_relu.onnx" \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5s_relu.onnx
curl -L -o "$REPO/models/anchors_yolov5.txt" https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/examples/yolov5/model/anchors_yolov5.txt
curl -L -o "$REPO/assets/bus.jpg" https://raw.githubusercontent.com/airockchip/rknn_model_zoo/main/examples/yolov5/model/bus.jpg
