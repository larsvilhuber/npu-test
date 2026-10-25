import argparse
import numpy as np
import torch
import torchvision.models as models

from common import add_args, timed_loop

p = add_args(argparse.ArgumentParser())
p.add_argument('--threads', type=int, required=True)
p.add_argument('--batch', type=int, default=1)
args = p.parse_args()

torch.set_num_threads(args.threads)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()

img = np.random.randint(0, 255, (args.batch, 224, 224, 3), dtype=np.uint8)
x = torch.from_numpy(img).permute(0, 3, 1, 2).float() / 255.0


def call():
    with torch.no_grad():
        model(x)


timed_loop(call, args, 'torch-fp32', 'cpu', args.threads, args.batch)
