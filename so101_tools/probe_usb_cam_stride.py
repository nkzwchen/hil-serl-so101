#!/usr/bin/env python3
"""测 usb_cam 在**哪个宽度**上会把图像发布歪（行距 stride 与声明不一致）。

背景
----
usb_cam 0.8.1 以 `pixel_format:=mjpeg2rgb` 发布时，内部解码/转换用的行距
**可能被补齐**（例如 424×3=1272 → 对齐到 1280），但 ROS `Image.step` 仍按
`width*3` 声明。消费方按 `step` reshape 就会得到**斜切**的图像 ——
画面看起来是斜条纹 + 彩虹伪彩，实测：
    cam_wrist(424x240) 拉普拉斯方差 4293  ← 歪的
    同一台设备直接 V4L2 抓     866  ← 正常
    cam_main(640x480)   正常（640*3=1920 本来就是 64 的倍数）

本脚本逐个宽度起一个**独立**的 usb_cam 节点（不碰机械臂、不用整套 launch），
订阅它，然后判断：
  ① 逐行最佳水平位移 ≈ 0 ?（非 0 = 斜切）
  ② 按「向上对齐到 A 字节」重新解释能否变清晰（A = 32 / 64 / 128）

用法
----
    python probe_usb_cam_stride.py                      # 测 424x240 / 352x288 / 640x360
    python probe_usb_cam_stride.py --sizes 424x240 640x480
"""

import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np

DEV = '/dev/v4l/by-id/usb-JoyandAI_JYU2C-2083_JYU2C-2083-2602030-video-index0'
ENC_PX = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}
TOPIC = '/stride_probe/image_raw'


def launch_node(w, h):
    cmd = ['ros2', 'run', 'usb_cam', 'usb_cam_node_exe', '--ros-args',
           '-p', f'video_device:={DEV}',
           '-p', 'pixel_format:=mjpeg2rgb',
           '-p', f'image_width:={w}', '-p', f'image_height:={h}',
           '-p', 'framerate:=30.0',
           '-p', 'io_method:=mmap',
           '-r', f'image_raw:={TOPIC}']
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def grab_one(timeout=25.0):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class G(Node):
        def __init__(self):
            super().__init__('stride_grab')
            self.msg = None
            self.create_subscription(Image, TOPIC, self.cb, 10)

        def cb(self, m):
            if self.msg is None:
                self.msg = m

    node = G()
    t0 = time.time()
    while node.msg is None and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.3)
    m = node.msg
    node.destroy_node()
    return m


def shear_per_row(img, rows=50, max_shift=40):
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
    return np.median(out), np.mean(out)


def try_aligns(buf, w, h, aligns=(16, 32, 64, 128, 256)):
    """按"行距向上对齐到 A 字节"重新解释，返回各 A 下的清晰度。"""
    res = {}
    for A in aligns:
        stride = ((w * 3 + A - 1) // A) * A
        rows = min(h, len(buf) // stride)
        if rows < 10:
            continue
        img = buf[:rows * stride].reshape(rows, stride)[:, :w * 3].reshape(rows, w, 3)
        res[A] = (cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var(),
                  stride, rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', nargs='*', default=['424x240', '352x288', '640x360'])
    args = ap.parse_args()

    import rclpy
    rclpy.init()
    print('=' * 92)
    print(f'usb_cam 0.8.1 stride 探针   device={DEV}')
    print('=' * 92)
    print(f"  {'请求':<10}{'声明 step':>10}{'逐行位移px':>12}{'方差(原样)':>12}"
          f"{'最清晰的重新解释':>26}{'判定':>12}")
    for sz in args.sizes:
        w, h = (int(v) for v in sz.split('x'))
        p = launch_node(w, h)
        try:
            m = grab_one()
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        time.sleep(1.0)
        if m is None:
            print(f'  {sz:<10} 拿不到图像')
            continue
        px = ENC_PX.get(m.encoding, 3)
        buf = np.frombuffer(m.data, np.uint8)
        if m.step != m.width * px:
            print(f'  {sz:<10} ⚠️ step={m.step} != width*px={m.width * px}')
        img = buf[:m.step * m.height].reshape(m.height, m.step)[:, :m.width * px]
        img = img.reshape(m.height, m.width, px)
        var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        med, mean = shear_per_row(img)
        al = try_aligns(buf, m.width, m.height)
        best = min(al.items(), key=lambda kv: kv[1][0]) if al else None
        ok = abs(med) < 0.5
        btxt = (f'A={best[0]} → stride {best[1]} 方差 {best[1] and best[1][0]:.0f}'
                if best else 'n/a')
        print(f'  {sz:<10}{m.step:>10}{med:>12.1f}{var:>12.0f}{btxt:>26}'
              f'{"✅ 正常" if ok else "❌ 被斜切":>12}')
        # 存一张原样图，便于人眼对比
        os.makedirs('wrist_captures', exist_ok=True)
        cv2.imwrite(f'wrist_captures/strideprobe_{sz}_asis.png', img[:, :, ::-1])
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
