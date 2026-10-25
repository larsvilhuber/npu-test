import argparse
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

from common import add_args, timed_loop

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets')

p = add_args(argparse.ArgumentParser())
p.add_argument('--core-mask', default='auto',
               help='auto, 0, 1, 2, 0_1 or 0_1_2 (NPU cores)')
args = p.parse_args()

rknn = RKNNLite()
if rknn.load_rknn(os.path.join(ASSETS, 'resnet18_for_rk3588.rknn')) != 0:
    raise SystemExit('Error loading model')
mask = getattr(RKNNLite, 'NPU_CORE_' + args.core_mask.upper())
if rknn.init_runtime(core_mask=mask) != 0:
    raise SystemExit('Error initializing NPU runtime')

img = cv2.imread(os.path.join(ASSETS, 'space_shuttle_224.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)

timed_loop(lambda: rknn.inference(inputs=[img]), args,
           'rknn-int8', f'npu-core-{args.core_mask}', 0, 1)
rknn.release()
