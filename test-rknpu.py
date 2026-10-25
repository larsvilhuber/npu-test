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
