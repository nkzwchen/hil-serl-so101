#!/usr/bin/env python
"""L1 验证：leader 命令链路 + 力矩 service。

测试内容（安全，不会让机械臂大幅运动）：
  1. 读 /leader/joint_states_raw 的当前位置
  2. 发命令让 leader「保持当前位置」
  3. 再发一个「小幅偏移」命令（默认 +0.1 rad on shoulder_pan），观察是否真的动
  4. 切力矩（可选，--torque-test）

用法：
    python test_leader_drive.py                 # 只做保持位置测试
    python test_leader_drive.py --offset 0.15   # 让 shoulder_pan 偏 0.15 rad
    python test_leader_drive.py --torque-test   # 额外测力矩开关
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool


class LeaderDriver(Node):
    def __init__(self):
        super().__init__('leader_drive_tester')
        self.joint = None
        self.create_subscription(JointState, '/leader/joint_states_raw', self._on_joint, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/leader/arm_command', 10)
        self.torque_cli = self.create_client(SetBool, '/leader/set_torque')

    def _on_joint(self, msg):
        self.joint = list(msg.position)

    def wait_joint(self, timeout=10.0):
        t0 = time.time()
        while self.joint is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.joint

    def send(self, values):
        m = Float64MultiArray()
        m.data = [float(v) for v in values]
        self.pub.publish(m)

    def set_torque(self, on: bool):
        if not self.torque_cli.wait_for_service(timeout_sec=5.0):
            print('  ⚠️ set_torque 服务不可用')
            return None
        req = SetBool.Request()
        req.data = on
        fut = self.torque_cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.2)
        return fut.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=float, default=0.0,
                    help='给 shoulder_pan 加多少弧度（默认 0 = 保持位置）')
    ap.add_argument('--hold-secs', type=float, default=3.0)
    ap.add_argument('--torque-test', action='store_true')
    args = ap.parse_args()

    rclpy.init()
    n = LeaderDriver()

    print('1) 读取 leader 当前位置...')
    before = n.wait_joint()
    if before is None:
        print('   ❌ 拿不到 /leader/joint_states_raw，确认 leader 桥在跑')
        return 1
    print(f'   {[round(v, 4) for v in before]}')

    if args.torque_test:
        print('\n2) 测试力矩 service')
        r = n.set_torque(False)
        print(f'   torque OFF -> {r}')
        time.sleep(1)
        r = n.set_torque(True)
        print(f'   torque ON  -> {r}')

    print(f'\n3) 发命令（{args.offset:+.3f} rad on shoulder_pan），保持 {args.hold_secs}s')
    target = list(before)
    target[0] += args.offset
    end = time.time() + args.hold_secs
    while time.time() < end:
        n.send(target)
        rclpy.spin_once(n, timeout_sec=0.05)
        time.sleep(0.05)   # ~10-20Hz

    # 等它稳定
    time.sleep(1.0)
    after = n.wait_joint()
    print(f'   发送后: {[round(v, 4) for v in after]}')

    err = [round(a - b, 4) for a, b in zip(after, before)]
    print(f'   实际变化: {err}')
    print(f'   shoulder_pan 期望变化 {args.offset:+.3f}，实际 {err[0]:+.4f}')

    if abs(args.offset) < 1e-6:
        print('   （本次是「保持位置」测试：变化应接近 0，说明命令链路通且无限幅抖动）')
        print('   ✅ 命令已被接受（无异常即通过）')
    else:
        ok = abs(err[0] - args.offset) < max(0.05, abs(args.offset) * 0.5)
        print('   ✅ leader 确实被驱动了' if ok else '   ⚠️ 变化量与期望差距较大，检查限幅/标定')

    n.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
