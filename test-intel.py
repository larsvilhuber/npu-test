import time
import cv2
import numpy as np
import torch
import torchvision.models as models

# 1. Load standard ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# 2. Prepare sample input (224x224x3)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# 3. Warmup CPU
with torch.no_grad():
    for _ in range(10):
        _ = model(img_tensor)

# 4. Measure 100 iterations
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(img_tensor)
end = time.time()

avg_ms = ((end - start) / 100) * 1000
print(f"Intel N150 CPU Average Inference Latency: {avg_ms:.2f} ms")
