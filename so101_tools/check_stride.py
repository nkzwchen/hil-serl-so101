#!/usr/bin/env python3
"""订阅一个 image_raw 话题，判断图像**有没有被行距补齐弄斜切**。

为什么要单独一个脚本
--------------------
usb_cam 0.8.1 用 `pixel_format:=mjpeg2rgb` 发布时，内部行距会被对齐补齐
（424*3 = 1272 → 1280），而 `Image.step` 仍声明 width*3 ⇒ 消费方 reshape 出斜图。
判据：
  · `align N`：按"N 字节对齐"重新解释同一段字节，若某个 N 让画面明显变清晰，
    说明真实行距就是那个对齐值（= 发布方撒谎了）。
  · 正常图像的逐行最佳水平位移 ≈ 0；被斜切则是固定非零值（如 -3px = -8 字节/3）。

用法
----
    python check_stride.py --topic /probe640/image_raw
    python check_stride.py --topic /follower/cam_wrist/image_raw   # 对着整套 launch 查
"""

import argparse
import sys
import time

import cv2
import numpy as np

ALIGNS = (16, 32, 64, 128, 256)


def shear_per_row(img, rows=50, max_shift=40):
    """相邻两行对齐所需水平位移的中位数。真实图像≈0；斜切则恒定非零。"""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    out = []
    for i in range(0, H - 1, max(1, H // rows)):
        a, b = g[i], g[i + 1]
        best, bd = 0, 1e18
        for s in range(-max_shift, max_shift + 1):
            if s >= 0:
                if s >= W:
                    continue
                d = np.abs(a[s:] - b[:W - s]).mean()
            else:
                d = np.abs(a[:W + s] - b[-s:]).mean()
            if d < bd:
                best, bd = s, d
        out.append(best)
    return float(np.median(out))


def var(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                               cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', required=True)
    ap.add_argument('--timeout', type=float, default=20.0)
    ap.add_argument('--save', default=None)
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class G(Node):
        def __init__(self):
            super().__init__('check_stride')
            self.m = None
            self.create_subscription(Image, args.topic, self.cb, 10)

        def cb(self, m):
            if self.m is None:
                self.m = m

    rclpy.init()
    n = G()
    t0 = time.time()
    while n.m is None and time.time() - t0 < args.timeout:
        rclpy.spin_once(n, timeout_sec=0.3)
    m = n.m
    if m is None:
        print(f'  ❌ {args.topic}: {args.timeout:.0f}s 内没收到图像')
        return 1

    px = 3 if m.encoding in ('rgb8', 'bgr8') else 1
    buf = np.frombuffer(m.data, np.uint8)
    img = buf[:m.step * m.height].reshape(m.height, m.step)[:, :m.width * px]
    img = img.reshape(m.height, m.width, px)
    s = shear_per_row(img)
    print(f'  话题     {args.topic}')
    print(f'  编码     {m.encoding}   {m.width}x{m.height}   '
          f'step={m.step}  (width*{px} = {m.width * px}, '
          f'{"一致" if m.step == m.width * px else "不一致 ⚠️"})')
    print(f'  逐行位移 {s:+.1f} px     拉普拉斯方差 {var(img):.0f}')
    bad = False
    for A in ALIGNS:
        st = ((m.width * px + A - 1) // A) * A
        if st == m.width * px:
            print(f'    align {A:>3}: 本来就不需要补齐 ✓（这个宽度是安全的）')
            continue
        rows = min(m.height, len(buf) // st)
        if rows < 10:
            continue
        im2 = buf[:rows * st].reshape(rows, st)[:, :m.width * px]
        im2 = im2.reshape(rows, m.width, px)
        v2 = var(im2)
        flag = ''
        if v2 < var(img) * 0.5:
            flag = '   ← 明显更清晰：真实行距就是这个对齐值 ⇒ 图像被斜切了'
            bad = True
        print(f'    align {A:>3}: stride {st:>5} 解出 {rows:>3} 行  方差 {v2:>8.0f}{flag}')
    if args.save:
        cv2.imwrite(args.save, img[:, :, ::-1])
        print(f'  存图     {args.save}')
    print(f'\n  判定     {"❌ 被斜切（需要改宽度）" if bad or abs(s) > 0.5 else "✅ 正常"}')
    n.destroy_node()
    rclpy.shutdown()
    return 0 if not (bad or abs(s) > 0.5) else 2


if __name__ == '__main__':
    sys.exit(main())
