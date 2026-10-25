import argparse
import os

import numpy as np
from rknnlite.api import RKNNLite

from common import add_args, timed_loop
from yolo_common import REPO, dump_detections, letterbox, load_anchors, load_image, post_process

p = add_args(argparse.ArgumentParser())
p.add_argument('--model', required=True, help='file in assets/, e.g. yolov5s_relu_i8.rknn')
p.add_argument('--core-mask', default='auto', help='auto, 0, 1, 2, 0_1 or 0_1_2')
p.add_argument('--stage', choices=['infer', 'e2e'], default='infer',
               help='infer: model call only; e2e: letterbox + model + decode + NMS')
args = p.parse_args()

rknn = RKNNLite()
if rknn.load_rknn(os.path.join(REPO, 'assets', args.model)) != 0:
    raise SystemExit('Error loading model')
if rknn.init_runtime(core_mask=getattr(RKNNLite, 'NPU_CORE_' + args.core_mask.upper())) != 0:
    raise SystemExit('Error initializing NPU runtime')

anchors, bgr = load_anchors(), load_image()
rgb, geom = letterbox(bgr)
dump_detections(args.out, args.test, *post_process(rknn.inference(inputs=[rgb[None]]), anchors), geom)


def e2e():
    img, _ = letterbox(bgr)
    post_process(rknn.inference(inputs=[img[None]]), anchors)


call = (lambda: rknn.inference(inputs=[rgb[None]])) if args.stage == 'infer' else e2e
timed_loop(call, args, 'yolov5s-' + args.model.split('_')[-1].split('.')[0],
           f'npu-core-{args.core_mask}', 0, 1)
rknn.release()
