"""YOLOv5 pre/post-processing shared by the NPU and OpenVINO benchmarks.

The model is Rockchip's NPU-friendly yolov5s_relu.onnx (rknn_model_zoo): three
raw output heads (1, 255, H, W) that already include the sigmoid; box decoding
and NMS happen here, on the CPU, as they would in a video pipeline.
"""
import csv
import fcntl
import os

import cv2
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45


def load_anchors():
    with open(os.path.join(REPO, 'models', 'anchors_yolov5.txt')) as f:
        return np.array([float(v) for v in f]).reshape(3, -1, 2)


def load_image():
    return cv2.imread(os.path.join(REPO, 'assets', 'bus.jpg'))


def letterbox(bgr):
    """Resize keeping aspect ratio, pad to SIZE x SIZE, return RGB uint8 + geometry."""
    h, w = bgr.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (r, left, top)


def _nms(boxes, scores):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter)
        order = rest[iou <= NMS_THRESH]
    return keep


def post_process(heads, anchors):
    """heads: three arrays (1, 255, H, W). Returns boxes (x1,y1,x2,y2 in the
    640x640 letterboxed frame), class ids, scores."""
    all_boxes, all_scores, all_cls = [], [], []
    for out, anc in zip(heads, anchors):
        out = np.asarray(out, dtype=np.float32)
        gh, gw = out.shape[-2:]
        p = out.reshape(len(anc), -1, gh, gw)             # (3, 85, H, W)
        col, row = np.meshgrid(np.arange(gw), np.arange(gh))
        stride = SIZE // gh
        xy = (p[:, :2] * 2 - 0.5 + np.stack([col, row])) * stride
        wh = (p[:, 2:4] * 2) ** 2 * anc.reshape(-1, 2, 1, 1)
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
        flat = lambda a: a.transpose(0, 2, 3, 1).reshape(-1, a.shape[1])
        all_boxes.append(flat(xyxy))
        all_scores.append(flat(p[:, 4:5]))
        all_cls.append(flat(p[:, 5:]))
    boxes, obj, cls = map(np.concatenate, (all_boxes, all_scores, all_cls))
    cmax = cls.max(axis=1)
    score = obj[:, 0] * cmax
    sel = score >= OBJ_THRESH
    boxes, score, cid = boxes[sel], score[sel], cls[sel].argmax(axis=1)
    kb, ks, kc = [], [], []
    for c in np.unique(cid):
        m = cid == c
        keep = _nms(boxes[m], score[m])
        kb.append(boxes[m][keep]); ks.append(score[m][keep]); kc.append(np.full(len(keep), c))
    if not kb:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    return np.concatenate(kb), np.concatenate(kc), np.concatenate(ks)


def dump_detections(out_csv, test, boxes, classes, scores, geom):
    """Append this run's detections on the sample image, in original-image pixels."""
    r, left, top = geom
    path = os.path.join(os.path.dirname(out_csv), 'detections.csv')
    with open(path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['test', 'class_id', 'score', 'x1', 'y1', 'x2', 'y2'])
        for b, c, s in zip(boxes, classes, scores):
            x1, y1, x2, y2 = ((b[0] - left) / r, (b[1] - top) / r,
                              (b[2] - left) / r, (b[3] - top) / r)
            w.writerow([test, int(c), f'{s:.4f}', f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}'])
        fcntl.flock(f, fcntl.LOCK_UN)
