#!/usr/bin/env python3
"""量化"开/关力矩时机械臂会不会跳"。

背景（2026-09-29 真机事故）
--------------------------
Feetech 舵机 `Torque_Enable = 1` 的瞬间会**立刻冲向 `Goal_Position` 寄存器里
的旧值**，而这个寄存器在力矩关闭期间不会被更新。于是"开力矩"这个动作本身就会
让机械臂猛冲向上一次运行留下的目标。

现场表现（主臂）：**突然跳起来**（冲向旧的抬高目标）→ 归位脚本第一条指令是
"开力矩之前"测到的下垂位置 → 又把它拽回去 → 第一帧就落后插值参考 34.8°
→ 直接触发跟踪中止。而且日志里**没有任何**写失败/过热报错，因为舵机没出错。

根治在 bridge：`_sync_goal_to_present()` 在 enable_torque() **之前**把
`Goal_Position` 写成当前实测位置。本脚本就是验证它到底生效没有。

判据
----
**两个阶段都必须纹丝不动**（< 1°），因为有两种互为镜像的错误：

| 阶段 | 情形 | 做错了会怎样 |
|---|---|---|
| A | 舵机**松着** → 开力矩 | 不写 `Goal_Position` → 冲向上次运行留下的旧目标 → **向上猛跳** |
| B | 舵机**已带力矩** → 再开一次 | 无条件把 `Goal_Position` 抹成实测位置 → 抹掉顶重力的静差偏置（比例环无积分项）→ **向下垂几度** |

阶段 B 是 2026-09-29 修 A 的时候**自己引入**的：主臂当场下垂 4.0°。
只测 A 会以为修好了，其实引入了 B。所以这个脚本两个都测。

用法
----
    python test_torque_jump.py leader      # 测主臂（默认）
    python test_torque_jump.py follower    # 测从臂
    python test_torque_jump.py leader --hold 3.0
"""

import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']


class Probe(Node):
    def __init__(self, arm: str):
        super().__init__(f'{arm}_torque_jump_probe')
        self.arm = arm
        self.pos = None
        self.stamp = 0.0
        self._stop = threading.Event()
        self.create_subscription(JointState, f'/{arm}/joint_states', self._cb, 10)
        self.torque_cli = self.create_client(SetBool, f'/{arm}/set_torque')
        # ★ 必须用后台 spin 线程：主线程里 spin_once 会漏消息（踩过坑）
        self._th = threading.Thread(target=self._spin, daemon=True)
        self._th.start()

    def _spin(self):
        while not self._stop.is_set() and rclpy.ok():
            try:
                rclpy.spin_once(self, timeout_sec=0.02)
            except Exception:
                break

    def _cb(self, msg: JointState):
        try:
            self.pos = np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES])
            self.stamp = time.time()
        except (ValueError, IndexError):
            pass

    def shutdown(self):
        self._stop.set()
        try:
            self._th.join(timeout=1.0)
        except Exception:
            pass

    def wait_states(self, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.pos is not None and time.time() - self.stamp < 0.3:
                return True
            time.sleep(0.05)
        return False

    def set_torque(self, on: bool):
        if not self.torque_cli.wait_for_service(timeout_sec=5.0):
            print(f'  ❌ 服务 /{self.arm}/set_torque 不可用')
            return False
        req = SetBool.Request()
        req.data = on
        fut = self.torque_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10.0:
            time.sleep(0.02)
        if not fut.done():
            print('  ❌ 服务调用超时')
            return False
        res = fut.result()
        print(f'  /{self.arm}/set_torque {on} -> success={res.success} '
              f'({res.message})')
        return bool(res.success)

    def sample(self, dur: float):
        """采样 dur 秒，返回位置序列 (n, 6)。"""
        out = []
        t0 = time.time()
        while time.time() - t0 < dur:
            if self.pos is not None:
                out.append(self.pos.copy())
            time.sleep(0.01)
        return np.array(out) if out else np.zeros((0, 6))


def one_phase(node, title, pre_off: bool, hold: float):
    """跑一个阶段，返回 (全过程最大偏移°, 最终偏移°, before, after, traj)。

    pre_off=True  → 先松力矩，再开（考察"会不会冲向旧 Goal_Position"）
    pre_off=False → 直接再开一次（考察"会不会把重力偏置抹掉导致下垂"）
    """
    print()
    print('=' * 68)
    print(f'  {title}')
    print('=' * 68)
    if pre_off:
        print('  先松力矩，等 0.6s 让手臂稳定 ...')
        if not node.set_torque(False):
            return None
        time.sleep(0.6)

    if node.pos is None:
        print('  ❌ 读不到关节状态')
        return None
    before = node.pos.copy()
    torque_state = '松着' if pre_off else '已带力矩'
    print(f'  开力矩前（{torque_state}）：')
    for n, v in zip(JOINT_NAMES, before):
        print(f'    {n:<16}{v:+.4f} rad ({math.degrees(v):+7.2f}°)')

    print(f'  开力矩，观察 {hold:.1f}s ...')
    if not node.set_torque(True):
        return None
    traj = node.sample(hold)
    if len(traj) == 0:
        print('  ❌ 开力矩后收不到关节状态')
        return None
    after = traj[-1]

    dev = np.abs(traj - before).max(axis=0)
    final = np.abs(after - before)
    print()
    print(f"    {'关节':<16}{'开力矩前':>11}{'稳定后':>11}"
          f"{'最终偏移':>11}{'过程中最大':>12}")
    for i, n in enumerate(JOINT_NAMES):
        print(f'    {n:<16}{math.degrees(before[i]):>10.2f}°'
              f'{math.degrees(after[i]):>10.2f}°'
              f'{math.degrees(final[i]):>10.2f}°'
              f'{math.degrees(dev[i]):>11.2f}°')
    return float(dev.max()), float(final.max()), before, after, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', nargs='?', default='leader', choices=['leader', 'follower'])
    ap.add_argument('--hold', type=float, default=2.0,
                    help='每个阶段开力矩后观察多少秒（默认 2.0）')
    ap.add_argument('--keep', action='store_true',
                    help='测完保持力矩 ON（默认恢复成测量前的状态）')
    args = ap.parse_args()

    rclpy.init()
    node = Probe(args.arm)
    rc = 0
    try:
        if not node.wait_states():
            print(f'❌ 拿不到 /{args.arm}/joint_states，launch 在跑吗？')
            return 2

        print(f'  目标：{args.arm}')
        print('  判据：两个阶段都应该**纹丝不动**（< 1°）')

        # 阶段 A：松着 → 开力矩（旧 bug 会让它向上猛跳）
        ra = one_phase(node, '阶段 A：松着 → 开力矩', pre_off=True, hold=args.hold)
        # 阶段 B：已带力矩 → 再开一次（同步写错会让它向下垂）
        rb = one_phase(node, '阶段 B：已带力矩 → 再开一次', pre_off=False, hold=args.hold)
        if ra is None or rb is None:
            return 2

        print()
        print('=' * 68)
        print('  结论')
        print('=' * 68)
        for tag, r, hint in (('A 松着→开', ra,
                              '↑ 大说明 bridge 没重编/重启，开力矩冲向了旧 Goal_Position'),
                             ('B 带力矩→再开', rb,
                              '↓ 大说明同步了本来带力矩的电机，抹掉了顶重力的静差偏置')):
            worst = math.degrees(r[0])
            mark = '✅' if worst < 1.0 else ('⚠️ ' if worst < 5.0 else '❌')
            print(f'  {mark} 阶段 {tag:<14} 最大偏移 = {worst:6.2f}°'
                  + ('' if worst < 1.0 else f'   ← {hint}'))
            if worst >= 5.0:
                rc = 1
        print('  两个阶段都 ✅ 才算真的修好；只修好一个说明只解决了一半的问题。')

        if not args.keep:
            print()
            print('  恢复力矩（收尾）...')
            node.set_torque(True)
        return rc
    finally:
        node.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    sys.exit(main())
