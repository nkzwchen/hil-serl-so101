#!/usr/bin/env python3
"""抓拍相机原始分辨率的图片（**不动机械臂**）。

为什么抓原始分辨率
------------------
录制/训练用的是缩到 224×224 之后的图，看不出相机本身的对焦、摩尔纹、
残帧这类问题。这里直接订 `.../image_raw`，存 **424×240（腕部）/
640×480（主）** 的原始帧。

同时给出三个客观指标：
  · 拉普拉斯方差 —— 清晰度（越大越锐；但摩尔纹也会把它抬高，要配合看）
  · 整行全 0 比例 —— 残帧检测（>10% 就是缓冲区没填满）
  · 帧间平均绝对差 —— 静止时应该只有传感器噪声（~2/255）；远大于就是解码抖动

用法
----
    python grab_stills.py --cam cam_wrist --n 6
    python grab_stills.py --cam both --n 6
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

ENC_PX = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, 'yuv422': 2, 'yuv422_yuy2': 2}


def to_rgb(msg):
    """sensor_msgs/Image → (H, W, 3) uint8 RGB。校验 step 与声明是否自洽。"""
    buf = np.frombuffer(msg.data, np.uint8)
    px = ENC_PX.get(msg.encoding)
    if px is None or not msg.step:
        raise RuntimeError(f'不认识的编码 {msg.encoding}')
    need = msg.step * msg.height
    if len(buf) < need:
        raise RuntimeError(f'数据不足：{len(buf)} < step*height={need}')
    if msg.step != msg.width * px:
        # 行距与声明宽度不一致 → 图像会**斜切**，必须报出来而不是默默 reshape
        print(f'  ⚠️ step={msg.step} != width*px={msg.width * px} → 可能斜切！')
    img = buf[:need].reshape(msg.height, msg.step)[:, :msg.width * px]
    img = img.reshape(msg.height, msg.width, px)
    if msg.encoding == 'bgr8':
        img = img[:, :, ::-1]
    if px == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img)


def partial_rows(img):
    """整行全 0 的行占比。残帧的特征。"""
    return float((np.asarray(img).max(axis=(1, 2)) == 0).mean())


def sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', default='cam_wrist', choices=['cam_main', 'cam_wrist', 'both'])
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--interval', type=float, default=0.4, help='帧间隔秒')
    ap.add_argument('--out', default='wrist_captures')
    ap.add_argument('--timeout', type=float, default=60.0)
    args = ap.parse_args()

    cams = ['cam_main', 'cam_wrist'] if args.cam == 'both' else [args.cam]
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime('%Y-%m-%d_%H%M%S')

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Grab(Node):
        def __init__(self):
            super().__init__('grab_stills')
            self.latest = {}
            for c in cams:
                self.create_subscription(
                    Image, f'/follower/{c}/image_raw',
                    lambda m, c=c: self.latest.__setitem__(c, m), 10)

    rclpy.init()
    node = Grab()
    frames = {c: [] for c in cams}
    t0 = time.time()
    next_t = t0
    print('=' * 84)
    print(f'抓拍  {" + ".join(cams)}  ×  {args.n} 张（原始分辨率，不动机械臂）')
    print('=' * 84)
    while len(frames[cams[0]]) < args.n and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if args.interval and time.time() < next_t:
            continue
        if len(node.latest) < len(cams):
            continue
        next_t = time.time() + args.interval
        for c in cams:
            try:
                frames[c].append(to_rgb(node.latest[c]))
            except Exception as e:
                print(f'  ❌ {c}: {e}')

    rows = []
    for c in cams:
        fs = frames[c]
        if not fs:
            print(f'  ❌ {c}: 一帧都没拿到')
            continue
        d = [float(np.abs(fs[i + 1].astype(np.int16) - fs[i].astype(np.int16)).mean())
             for i in range(len(fs) - 1)] or [0.0]
        print(f'\n### {c}   {len(fs)} 张  {fs[0].shape[1]}x{fs[0].shape[0]}')
        print(f"  {'#':>3}{'平均亮度':>10}{'拉普拉斯方差':>14}{'全0行%':>10}"
              f"{'与上一张差':>12}   文件")
        for i, f in enumerate(fs):
            fn = f'{args.out}/{stamp}_{c}_{i + 1:02d}.png'
            cv2.imwrite(fn, f[:, :, ::-1])          # cv2 写文件要 BGR
            print(f'  {i + 1:>3}{f.mean():>10.1f}{sharpness(f):>14.1f}'
                  f'{partial_rows(f) * 100:>10.1f}{(d[i - 1] if i else float("nan")):>12.2f}'
                  f'   {os.path.basename(fn)}')
            rows.append((c, i, f))
        print(f'  → 帧间差 中位 {np.median(d):.2f}/255（静止时应当只有~2/255 的噪声）')

    # 拼一张总览图（每种相机一行）
    tiles = []
    for c in cams:
        fs = frames[c]
        if not fs:
            continue
        want = fs[0].shape[:2]
        row = [cv2.resize(f, (want[1], want[0])) for f in fs]
        tiles.append(np.concatenate(row, axis=1))
    if tiles:
        W = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles]
        grid = np.concatenate(tiles, axis=0)
        for j, c in enumerate(cams):
            cv2.putText(grid, c, (8, j * tiles[0].shape[0] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        mp = f'{args.out}/{stamp}_montage.png'
        cv2.imwrite(mp, grid[:, :, ::-1])
        print(f'\n  🖼  总览图：{mp}   {grid.shape[1]}x{grid.shape[0]}')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
