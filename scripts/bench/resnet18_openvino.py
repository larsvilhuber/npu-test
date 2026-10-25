import argparse
import numpy as np
import openvino as ov
import torch
import torchvision.models as models

from common import add_args, timed_loop

p = add_args(argparse.ArgumentParser())
p.add_argument('--device', default='CPU', help='OpenVINO device: CPU or GPU')
args = p.parse_args()

x = torch.from_numpy(np.random.rand(1, 3, 224, 224).astype(np.float32))
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
compiled = ov.compile_model(ov.convert_model(model, example_input=x), args.device,
                            {'PERFORMANCE_HINT': 'LATENCY'})
xn = x.numpy()

timed_loop(lambda: compiled(xn), args, 'openvino-fp32', args.device.lower(), 0, 1)
