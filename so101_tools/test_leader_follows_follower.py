#!/usr/bin/env python
"""测试「leader 跟随 follower」—— 模式 A 的核心能力。

做什么：
  1. 同时启动 follower 桥 和 leader 桥
  2. 起一个中继：持续读 /follower/joint_states，转发到 /leader/arm_command
  3. 于是 leader 臂会实时跟随 follower 臂

怎么验证：
  · 用手轻轻推动 **follower** 臂 → **leader** 臂应该跟着动
  · 观察打印的跟随误差，以及有没有抖动/滞后

安全：
  · follower 侧力矩默认是开的（它会被命令驱动），但它此时没有别人在命令它，
    所以它是「松」的，你可以用手推
  · leader 侧会被电机驱动，请留出空间
  · Ctrl-C 退出，脚本会自动把 leader 力矩恢复 ON 并停掉两个桥

用法：
  python test_leader_follows_follower.py
  python test_leader_follows_follower.py --rate 10      # 中继频率（默认 10Hz）
  python test_leader_follows_follower.py --no-bridges   # 桥已在跑，只起中继
"""

import argparse
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool

# ─── 路径自解析（自动生成，勿手改：见 pack_so101_tools.py）───────────────
# 让脚本在两种布局下都能跑：
#   A) pi_project 布局： <P>/脚本.py              + <P>/hil-serl/examples/...
#   B) 仓库布局：       <repo>/so101_tools/脚本.py + <repo>/examples/...
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _resolve_dirs():
    for _b in (_HERE, _os.path.dirname(_HERE)):
        if _os.path.isdir(_os.path.join(_b, 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'examples')
        if _os.path.isdir(_os.path.join(_b, 'hil-serl', 'examples', 'experiments')):
            return _b, _os.path.join(_b, 'hil-serl', 'examples')
    _up = _os.path.dirname(_HERE)
    return _up, _os.path.join(_up, 'examples')


_ROOT, _EX = _resolve_dirs()
_EXPERIMENTS_DIR = _os.path.join(_EX, 'experiments')
_SO101_DIR = _os.path.join(_EXPERIMENTS_DIR, 'so101')
_sys.path.insert(0, _SO101_DIR)
_sys.path.insert(0, _EXPERIMENTS_DIR)
_sys.path.insert(0, _EX)
_sys.path.insert(0, _ROOT)
# ─────────────────────────────────────────────────────────────────────────
ROS_WS = '_ROOT/ros2_ws'
PARAMS_DIR = f'{ROS_WS}/install/so101_ros2_bridge/share/so101_ros2_bridge/config'
LOGDIR = _ROOT
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def _bridge_cmd(robot_type: str) -> str:
    """生成启动某个桥的 shell 命令（含 conda 环境）。"""
    params = f'{PARAMS_DIR}/so101_{robot_type}_params.yaml'
    return (
        'unset PYTHONPATH; '
        'source /home/nkzwchen/miniforge3/etc/profile.d/conda.sh; '
        'conda activate /home/nkzwchen/miniforge3/envs/hil-serl; '
        f'source {ROS_WS}/install/local_setup.bash; '
        'export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=0; '
        f'exec ros2 run so101_ros2_bridge {robot_type}_ros2_node '
        f'--ros-args -r __ns:=/{robot_type} '
        f'-r __node:=so101_{robot_type}_interface '
        f'-p type:={robot_type} '
        f'--params-file {params}'
    )


class Relay(Node):
    """读 follower 位置 → 发给 leader。

    ⚠️ 观测源用 /follower/joint_states_raw（桥直发），而不是 /follower/joint_states。
       原因：/follower/joint_states 由 ros2_control 的 joint_state_broadcaster 发布，
       而本测试**不启动 follower 的 ros2_control**（不需要控制器，也不需要力矩）。
       桥自己发的 raw 话题在任何情况下都有，更适合这个隔离测试。
       正式训练时 SO101Env 会用 /follower/joint_states（因为那时 ros2_control 在跑）。
    """

    def __init__(self, rate_hz: float):
        super().__init__('leader_follows_follower_relay')
        self.follower = None          # 最近一次 follower 关节角（弧度）
        self.leader = None            # 最近一次 leader 关节角
        self.n_sent = 0

        self.create_subscription(JointState, '/follower/joint_states_raw', self._on_follower, 10)
        self.create_subscription(JointState, '/leader/joint_states_raw', self._on_leader, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/leader/arm_command', 10)
        self.torque_cli = self.create_client(SetBool, '/leader/set_torque')
        self.follower_torque_cli = self.create_client(SetBool, '/follower/set_torque')

        self.dt = 1.0 / rate_hz
        self.timer = self.create_timer(self.dt, self._tick)

    def _reorder(self, msg: JointState):
        if not msg.name:
            return list(msg.position)
        try:
            return [msg.position[msg.name.index(j)] for j in JOINTS]
        except ValueError:
            # name 不全，退化为原顺序
            return list(msg.position)

    def _on_follower(self, msg):
        self.follower = self._reorder(msg)

    def _on_leader(self, msg):
        self.leader = self._reorder(msg)

    def _tick(self):
        if self.follower is None:
            return
        m = Float64MultiArray()
        m.data = [float(v) for v in self.follower]
        self.pub.publish(m)
        self.n_sent += 1

    def set_leader_torque(self, on: bool):
        return self._call_torque(self.torque_cli, on, '/leader/set_torque')

    def set_follower_torque(self, on: bool):
        return self._call_torque(self.follower_torque_cli, on, '/follower/set_torque')

    def _call_torque(self, cli, on: bool, name: str):
        if not cli.wait_for_service(timeout_sec=5.0):
            print(f'  ⚠️ {name} 服务不可用')
            return None
        req = SetBool.Request()
        req.data = on
        fut = cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 10:
            rclpy.spin_once(self, timeout_sec=0.2)
        return fut.result()


def start_bridges():
    procs = []
    for rtype in ('follower', 'leader'):
        log = open(f'{LOGDIR}/l2_{rtype}.log', 'w')
        p = subprocess.Popen(['bash', '-c', _bridge_cmd(rtype)],
                             stdout=log, stderr=subprocess.STDOUT)
        procs.append((rtype, p, log))
        print(f'  启动 {rtype} 桥 pid={p.pid}  日志: {LOGDIR}/l2_{rtype}.log')
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=10.0, help='中继频率 Hz')
    ap.add_argument('--no-bridges', action='store_true', help='桥已在跑，只起中继')
    ap.add_argument('--seconds', type=float, default=0.0,
                    help='自动运行多少秒后退出（0=一直跑到 Ctrl-C）')
    args = ap.parse_args()

    procs = []
    if not args.no_bridges:
        print('############ 启动两个臂的桥 ############')
        procs = start_bridges()
        print('  等待桥就绪（约 20s）...')
        time.sleep(20)
        for rtype, p, _ in procs:
            logf = f'{LOGDIR}/l2_{rtype}.log'
            try:
                txt = open(logf).read()
            except OSError:
                txt = ''
            ok = 'initialized' in txt.lower()
            print(f'  {rtype} 桥: {"✅ 就绪" if ok else "⚠️ 未见就绪日志"}')
            if not ok:
                print('     日志尾部:')
                print('     ' + '\n     '.join(txt.strip().splitlines()[-6:]))
    else:
        print('############ 跳过启动桥（--no-bridges）############')

    print()
    print('############ 启动中继：follower → leader ############')
    rclpy.init()
    n = Relay(args.rate)

    # leader 力矩必须 ON（否则它不会跟随）
    r = n.set_leader_torque(True)
    print(f'  leader   力矩 ON  -> {r}')

    # ★ follower 力矩必须 OFF，否则它是「硬」的，手推不动
    #   这是之前测试失败的原因：舵机连接后默认保持位置（有力矩），
    #   所以"用手推动 follower"根本推不动。
    r = n.set_follower_torque(False)
    print(f'  follower 力矩 OFF -> {r}')

    print()
    print('=' * 72)
    print(' 现在用手推动【 follower 臂 】（ttyACM1，编号较小的插口那条）')
    print(' 预期：【 leader 臂 】（ttyACM0）跟着动')
    print()
    print(' 已把 follower 力矩关掉 → 它现在是「松」的，可以用手推动')
    print(' leader 力矩是开的 → 它会被驱动去跟随')
    print()
    print(' 【不要】在此期间跑 control.sh 或阶段 A 的 launch，')
    print('  否则会有别的进程同时控制 follower。')
    print()
    print(f' 中继频率: {args.rate} Hz   按 Ctrl-C 结束')
    print('=' * 72)
    print()

    t0 = time.time()
    last_report = t0
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.05)
            now = time.time()
            if now - last_report >= 2.0:
                last_report = now
                if n.follower is not None and n.leader is not None:
                    err = [l - f for l, f in zip(n.leader, n.follower)]
                    ae = sum(abs(e) for e in err[:5]) / 5.0
                    print(f'  [{now - t0:5.1f}s] 已发 {n.n_sent:4d} 条 | '
                          f'平均跟随误差 {ae:.4f} rad ({ae * 57.3:.2f}°) | '
                          f'各关节误差 {[round(e, 3) for e in err[:5]]}')
                else:
                    got_f = n.follower is not None
                    got_l = n.leader is not None
                    print(f'  [{now - t0:5.1f}s] 等数据... follower={got_f} leader={got_l}')
            if args.seconds > 0 and now - t0 >= args.seconds:
                print(f'  已运行 {args.seconds}s，自动退出')
                break
    except KeyboardInterrupt:
        print('\n  已中断')
    finally:
        print()
        print('############ 收尾 ############')
        print('  1) 恢复力矩')
        try:
            # leader 恢复 ON：避免松着下垂磕碰
            n.set_leader_torque(True)
            print('     leader   -> ON  ✅')
        except Exception as e:
            print(f'     leader   ⚠️ {e}')
        try:
            # follower 保持 OFF（测试结束后它没有被任何东西控制，
            # 关着更安全，方便你用手把它摆回原位）
            n.set_follower_torque(False)
            print('     follower -> OFF ✅（保持松，便于手动摆回）')
        except Exception as e:
            print(f'     follower ⚠️ {e}')

        # 先停桥，再关 rclpy（顺序反了会让 service 调用失败）
        if procs:
            print('  2) 停止两个桥')
            for rtype, p, log in procs:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                log.close()
                print(f'     已停止 {rtype} 桥')
            # 兜底：杀残留
            subprocess.run(
                ['bash', '-c',
                 "ps -eo pid,args | grep -E '[l]eader_ros2_node|[f]ollower_ros2_node' | "
                 "awk '{print $1}' | xargs -r kill -9"],
                check=False,
            )
        else:
            print('  2) 未由本脚本启动桥，跳过停止（如需要：pkill -f ros2_node）')
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
