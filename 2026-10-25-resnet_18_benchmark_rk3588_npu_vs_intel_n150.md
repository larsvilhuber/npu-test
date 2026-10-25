---
title: "Benchmarking Edge AI: Orange Pi 5 (RK3588 NPU) vs. Intel N150 CPU"
date: 2026-10-25
categories: [Edge Computing, Machine Learning]
tags: [rk3588, npu, rknn, intel-n150, pytorch, benchmark]
---

# Edge AI Showdown: Pi-like device with NPU vs. Intel N150 CPU: 10x speedup

I never know what the purpose is of **NPU** (neural-processing unit). Turns out, I have an Orange Pi 5 sitting around, which actually does have one. Compared to a small computer running a low-power Intel CPU, that NPU can accelerate certain tasks, at lower power. But note that I did not have an "AI CPU"  at hand, to compare what that would look like on an Intel chip that also has an NPU.

Now I need to figure out where I need this... (updating my [Birdnet Go](https://github.com/tphakala/birdnet-go) [node](https://app.birdweather.com/) on [Birdnet](https://birdnet.cornell.edu/)?)

> In a head-to-head [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) inference benchmark, the **Orange Pi 5 (RK3588 NPU)** executed image classification **10.0x faster** than an **Intel N150 CPU** running stock PyTorch, while leaving the primary CPU cores virtually idle.

## Benchmark Results Summary

| System | Target Compute Device | Runtime Engine | Avg Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **Orange Pi 5** | RK3588 NPU (3 Cores) | RKNN Lite 2 (v2.5.0) | **4.39 ms** | **227.8 FPS** |
| **Intel N150** | Intel N150 CPU | PyTorch (FP32) | **43.98 ms** | **22.7 FPS** |

*Note: The RK3588 result was measured under default dynamic frequency scaling. Locking the NPU frequency to performance mode could further reduce latency to ~1.5 ms (~660 FPS).*

---

## Reproducible Setup & Execution Guide - What worked!

The following instructions detail how to set up, build, and run the benchmarks on both systems using relative paths within activated virtual environments.

---

### Part 1: Orange Pi 5 (RK3588 NPU) Benchmark

{% details System level details %}

Ensure your user account belongs to the `render` and `video` groups to access `/dev/rknpu` and `/dev/dri/renderD128` without root privilege:

```bash
sudo usermod -aG render,video $USER
# Re-login or run: exec su - $USER
```
{% enddetails %}

####  Runtime Library Installation

Download the official RKNPU runtime library and copy it to system path:

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git --depth 1
sudo cp rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp312-*/rknnlite/api/libs/librknnrt.so /usr/lib/
sudo ldconfig
```

#### Python Setup

```bash
python3 -m venv rknn-env
source rknn-env/bin/activate

pip install --upgrade pip setuptools wheel
pip install opencv-python-headless numpy

# Install matching RKNN-Lite2 package (e.g., Python 3.12 / cp312)
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
pip install ~/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp${PY_VER}-*.whl
```

#### Copy Benchmark Assets

Copy the pre-compiled model and test image into your workspace directory:

```bash
cp ~/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/resnet18_for_rk3588.rknn ./
cp ~/rknn-toolkit2/rknn-toolkit-lite2/examples/resnet18/space_shuttle_224.jpg ./
```

#### Benchmark Script (`test-rknn.py`)

```python
# test-rknn.py
import time
import cv2
import numpy as np
from rknnlite.api import RKNNLite

MODEL_PATH = './resnet18_for_rk3588.rknn'
IMAGE_PATH = './space_shuttle_224.jpg'

rknn = RKNNLite()

# Load model
if rknn.load_rknn(MODEL_PATH) != 0:
    print(f'Error loading {MODEL_PATH}')
    exit(1)

# Initialize runtime across all 3 NPU cores
if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
    print('Error initializing NPU runtime')
    exit(1)

# Preprocess image (224x224 RGB)
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)

# Warmup pass
rknn.inference(inputs=[img])

# Benchmark 100 iterations
start = time.time()
for _ in range(100):
    _ = rknn.inference(inputs=[img])
end = time.time()

avg_latency = ((end - start) / 100) * 1000
print(f'Orange Pi 5 RK3588 NPU Avg Latency: {avg_latency:.2f} ms')

rknn.release()
```

which can then be run with

```bash
python3 test-rknn.py
```

---

### Part 2: Intel N150 (CPU) Benchmark

#### 1. Python Virtual Environment Setup

Create the workspace and setup a Python virtual environment:

```bash
python3 -m venv bench-env
source bench-env/bin/activate

pip install --upgrade pip
pip install torch torchvision opencv-python-headless numpy
```

#### Benchmark Script (`test-intel.py`)


```python
# test-intel.py
import time
import cv2
import numpy as np
import torch
import torchvision.models as models

# 1. Load standard ResNet-18 pretrained weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. Prepare sample input tensor (1, 3, 224, 224)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# 3. Warmup CPU context
with torch.no_grad():
    for _ in range(10):
        _ = model(img_tensor)

# 4. Benchmark 100 iterations
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(img_tensor)
end = time.time()

avg_ms = ((end - start) / 100) * 1000
print(f"Intel N150 CPU Average Inference Latency: {avg_ms:.2f} ms")
```

which can then be run with

```bash
python3 test-intel.py
```

---

## Key Takeaways

1. **Hardware Acceleration Matters:** Dedicated ASICs like the RK3588 NPU are an order-of-magnitude  faster for  vision tasks.
2. **Resource Allocation:** Offloading inference to the NPU leaves all CPU cores free. The weaker CPU on the Orange Pi does not matter. ([Orange Pi CPU is not THAT much slower, though](https://www.cpubenchmark.net/compare/6304vs4906/Intel-N150-vs-Rockchip-RK3588)).
3. **Optimizations:** I did not compare the CPU on the Orange Pi to its own NPU, and the N150 stack was not customized further (Intel OpenVINO or ONNX Runtime).