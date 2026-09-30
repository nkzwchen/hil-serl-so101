#!/usr/bin/env python3
"""采集从臂关节状态，输出稳定位姿（均值 + 标准差）。

用途：用 leader 遥操作把从臂摆到理想位置并**静止保持**，然后运行本脚本，
      取最后 N 秒的均值作为候选 START_JOINT_POSE。

★ 为什么要看标准差，而不是只取均值：
    如果操作时手在飘，"平均值"会得到一个**谁都没真正摆过**的位姿——
    既可能不可达，也可能不在 IK 良态区，反而比随手取一帧更糟。
    标准差就是"这个位姿是否真实稳定"的判据：
        过小  → 位姿稳定，均值可信
        过大  → 手在动，均值无意义，需要重新保持静止

用法：
    # 默认采 20s，取最后 10s
    python sample_pose.py

    # 自定义：采 30s，取最后 15s
    python sample_pose.py --duration 30 --tail 15

    # 顺便导出成可直接粘进 config.py 的字面量
    python sample_pose.py --emit-config
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# 与 env.py 的 JOINT_NAMES 严格一致
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# 稳定判据（弧度）。STS3215 编码器分辨率约 2π/4096 ≈ 0.0015 rad，
# 所以 std 远小于此说明是真实静止，而不是传感器噪声。
STD_OK = 0.010      # < 0.01 rad 视为稳定
STD_WARN = 0.030    # > 0.03 rad 视为手在飘


class PoseSampler(Node):
    def __init__(self, topic: str):
        super().__init__('pose_sampler')
        self.samples = []           # list[(t, np.ndarray)]
        self.create_subscription(JointState, topic, self._cb, 50)

    def _cb(self, msg: JointState):
        if not msg.name:
            return
        try:
            pos = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES], dtype=float)
        except ValueError:
            # 名字对不上时退化为原顺序
            pos = np.array(msg.position[:len(JOINT_NAMES)], dtype=float)
        if pos.shape[0] != len(JOINT_NAMES):
            return
        self.samples.append((time.time(), pos))

    def collect(self, duration: float):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/follower/joint_states')
    ap.add_argument('--duration', type=float, default=20.0, help='总采集时长（秒）')
    ap.add_argument('--tail', type=float, default=10.0, help='取最后多少秒做统计')
    ap.add_argument('--emit-config', action='store_true', help='额外输出可粘贴的 config 字面量')
    args = ap.parse_args()

    rclpy.init()
    node = PoseSampler(args.topic)
    print(f'采集 {args.topic}，持续 {args.duration:.0f}s ... （请保持机械臂静止）')
    node.collect(args.duration)

    if not node.samples:
        print('❌ 没收到任何 joint_states。launch 在跑吗？话题名对吗？')
        node.destroy_node(); rclpy.shutdown(); return 1

    ts = np.array([t for t, _ in node.samples])
    ps = np.stack([p for _, p in node.samples])

    t_end = ts[-1]
    mask = ts >= (t_end - args.tail)
    sel = ps[mask]
    mean = sel.mean(axis=0)
    std = sel.std(axis=0)

    print(f'\n总样本 {len(ps)} 帧（约 {len(ps)/max(ts[-1]-ts[0],1e-6):.1f} Hz），'
          f'统计窗口内 {len(sel)} 帧 / {args.tail:.0f}s')
    print()
    print(f'{"关节":<16}{"均值 (rad)":>14}{"标准差 (rad)":>16}   判定')
    print('-' * 62)
    for i, name in enumerate(JOINT_NAMES):
        s = std[i]
        flag = '✅ 稳定' if s < STD_OK else ('⚠️ 偏大' if s < STD_WARN else '❌ 在动')
        print(f'{name:<16}{mean[i]:>14.6f}{s:>16.6f}   {flag}')

    worst = float(std.max())
    print()
    if worst < STD_OK:
        print(f'✅ 位姿稳定（最大标准差 {worst:.6f} rad < {STD_OK}），均值可用作候选起点')
    elif worst < STD_WARN:
        print(f'⚠️ 有点飘（最大标准差 {worst:.6f} rad）。建议保持静止重采一次')
    else:
        print(f'❌ 明显在动（最大标准差 {worst:.6f} rad）。均值不可信，请扶稳后重采')

    print()
    print('均值（按 JOINT_NAMES 顺序）：')
    print('  [' + ', '.join(f'{v:.6f}' for v in mean) + ']')

    if args.emit_config:
        print()
        print('可粘贴进 config.py：')
        print('    START_JOINT_POSE = [')
        for i, name in enumerate(JOINT_NAMES):
            print(f'        {mean[i]:+.6f},   # {name}')
        print('    ]')

    print()
    print('提醒：这只是**候选**起点。是否可用要由 V2(IK 精度) / V3(工作空间扫描) 判定。')
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
