"""ROS-independent image validation and reproducible ROI preprocessing."""
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROI_PATH = Path(__file__).with_name('cam_main_roi.json')
MAIN_ROI_PATH = Path(__file__).with_name('cam_main_view_roi.json')
CAMERA_KEYS = ('cam_main', 'cam_wrist')


def decode_rgb(msg):
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f'Unsupported/ambiguous encoding: {msg.encoding!r}')
    h, w, stride = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if h <= 0 or w <= 0 or stride < w * 3 or data.size != h * stride:
        raise ValueError(f'Invalid image layout: {w}x{h}, step={stride}, bytes={data.size}')
    rgb = data.reshape(h, stride)[:, :w * 3].reshape(h, w, 3)
    if msg.encoding == 'bgr8':
        rgb = rgb[:, :, ::-1]
    return rgb.copy()


def validate_roi(spec):
    if spec.get('version') != 1 or spec.get('camera') != 'cam_main':
        raise ValueError('ROI must use version=1, camera=cam_main')
    if spec.get('preprocess') != 'letterbox_rgb_v1':
        raise ValueError('Unsupported ROI preprocessing')
    values = spec['source_size'] + spec['xyxy']
    if len(values) != 6 or any(type(v) is not int for v in values):
        raise ValueError('ROI size and bounds must be integers')
    w, h, x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI is empty or outside the source image')
    return spec


def load_roi(path=None):
    path = Path(path) if path else DEFAULT_ROI_PATH
    if not path.exists():
        return None
    return validate_roi(json.loads(path.read_text()))


def crop_roi(rgb, spec, size):
    validate_roi(spec)
    if list(rgb.shape[1::-1]) != spec['source_size']:
        raise ValueError('cam_main resolution changed; select the ROI again')
    x0, y0, x1, y1 = spec['xyxy']
    crop = rgb[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = size / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=(
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
    out = np.zeros((size, size, 3), np.uint8)
    x, y = (size - nw) // 2, (size - nh) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def load_main_roi():
    return load_roi(MAIN_ROI_PATH)
