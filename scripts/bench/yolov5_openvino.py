import argparse
import os

import numpy as np
import openvino as ov

from common import add_args, timed_loop
from yolo_common import REPO, dump_detections, letterbox, load_anchors, load_image, post_process

p = add_args(argparse.ArgumentParser())
p.add_argument('--device', default='CPU', help='OpenVINO device: CPU or GPU')
p.add_argument('--stage', choices=['infer', 'e2e'], default='infer',
               help='infer: model call only; e2e: letterbox + model + decode + NMS')
args = p.parse_args()

core = ov.Core()
compiled = core.compile_model(core.read_model(os.path.join(REPO, 'models', 'yolov5s_relu.onnx')),
                              args.device, {'PERFORMANCE_HINT': 'LATENCY'})


def run(rgb):
    x = (rgb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
    out = compiled(x)
    return [out[o] for o in compiled.outputs]


anchors, bgr = load_anchors(), load_image()
rgb, geom = letterbox(bgr)
dump_detections(args.out, args.test, *post_process(run(rgb), anchors), geom)


def e2e():
    img, _ = letterbox(bgr)
    post_process(run(img), anchors)


x0 = (rgb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
call = (lambda: compiled(x0)) if args.stage == 'infer' else e2e
timed_loop(call, args, 'yolov5s-openvino-fp32', args.device.lower(), 0, 1)
