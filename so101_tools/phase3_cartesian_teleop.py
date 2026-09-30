#!/usr/bin/env python3
"""阶段 A：3-DOF 笛卡尔主臂→从臂遥操作（与 policy 推理走**完全相同**的路径）。

为什么必须先做这个：
    录制 demo 时，**录下来的 action 必须和策略推理输出的量是同一个东西**。
    env.py 里的警告说得很重：
        `info.pop("intervene_action")` 会把两种量纲的动作混进同一个 buffer，
        critic 学出来就是坏的。
    所以不能"先录关节运动、事后再转换"，必须让主臂驱动从臂时就走策略那条
    IK 路径，并把路径上的**归一化 3 维动作原样记下**。

本脚本建立的链路（与之后 policy 推理逐字一致）：
    主臂 6 关节角
      → FK (so101_kin)                    → 主臂 TCP 位置 p_L  （基座系）
      → Δp = p_L(t) − p_L(t−1)            → 3 维平移增量
      → a = clip(Δp / ACTION_SCALE, −1, 1) → ★ 归一化动作（与 policy 同量纲）
      → Δp_cmd = a × ACTION_SCALE
      → p_target = 从臂管尖当前位置 + Δp_cmd
      → IK（位置主任务 + 管子方向对齐）    → 从臂 5 关节目标
      → /follower/arm_controller/joint_trajectory

两种模式（空格切换）：
    CART  笛卡尔：上面那条链 —— **录制阶段用这个**
    JOINT 关节映射：从臂关节角 = 主臂关节角（逐字直传）—— **回程阶段用这个**
          （已核对 leader_teleop_component.cpp：确实是原样直传，
            无缩放/偏移/符号变换；gripper 单独走 action）

用法：
    python phase3_cartesian_teleop.py --preview          # 只打印配置，不动作
    python phase3_cartesian_teleop.py --duration 30      # 试 30 秒手感
    python phase3_cartesian_teleop.py                    # 一直跑，按 q 退出

    # 调手感：尺度越小越精细、但人手容易推太快被削顶
    python phase3_cartesian_teleop.py --action-scale 0.005
    python phase3_cartesian_teleop.py --action-scale 0.020
"""

import argparse
import math
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import os

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
from so101_kin import SO101Kin  # noqa: E402
# 起点 / 限位 / 主臂抬高偏置的**唯一真相源**（不要在这里另写一份）
from so101_home_pose import (  # noqa: E402
    ARM_JOINTS, CONTROLLED_IDX, CONTROLLED_JOINTS, FROZEN_JOINTS, JOINT_NAMES,
    N_ARM, Q_HIGH, Q_LOW, START_JOINT_POSE, hold_frozen,
)

# ★ 冻结的维度（唯一真相源在 so101_home_pose.FROZEN_JOINTS）：
#   · 第 1 维 shoulder_pan ：本任务插孔位置固定，不需要侧向旋转（2026-09-29 用户要求）
#   · 第 5 维 wrist_roll   ：管子轴对称，绕自身轴转对插入任务无意义
#   · 第 6 维 gripper      ：本任务固定夹着试管，不参与控制（也不在这条 5 关节轨迹里）
#   实际冻结动作由 `hold_frozen(q_t, 实测值)` 完成 —— 用**实测值**而不是下面这两个
#   常量，这样残差/松弛永远不会变成跳变。两个常量只作初值和打印用。
FIXED_WRIST_ROLL = float(START_JOINT_POSE[4])
FIXED_GRIPPER = float(START_JOINT_POSE[5])

TRACK_ABORT = 0.60       # rad ≈ 34°：从臂完全跟不上就中止（异常保护）


class Key:
    """WSL2 下唯一可用的键盘方案（termios + select）。正常退出时还原终端。"""

    def __init__(self):
        self.ok = False
        self.fd = None
        self.old = None
        if not sys.stdin.isatty():
            return
        try:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.ok = True
        except Exception:
            self.ok = False

    def read(self):
        if not self.ok:
            return ''
        try:
            r, _, _ = select.select([self.fd], [], [], 0)
            if r:
                return sys.stdin.read(1)
        except Exception:
            pass
        return ''

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.ok and self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            except Exception:
                pass
        return False


class CartesianTeleop(Node):
    def __init__(self, action_scale=0.010, deadband=0.002, rate=10.0,
                 orient_weight=0.5):
        super().__init__('cartesian_teleop')
        self.action_scale = float(action_scale)
        self.deadband = float(deadband)
        self.dt = 1.0 / float(rate)
        self.orient_weight = float(orient_weight)

        self.leader_q = None
        self.follower_q = None
        # ★ 每一步读数的**时间戳**，用于新鲜度检查。
        #   踩坑（2026-09）：原先主循环里用 rclpy.spin_once(timeout_sec=0.0)，
        #   每个周期只处理 1 条消息；而两路 joint_states 各 50Hz 合计 100 条/秒，
        #   主循环只有 10Hz → 队列必然积压 → 读到的是几百毫秒前的旧值。
        #   后果（实测症状）：
        #     · Δp 被放大数倍（单步 87mm！人手不可能这么快）→ 削顶 25.8%
        #     · IK 基准 p_f 是旧位置 → 目标指向已过时的点 → 跟踪误差累积到 34.6°
        #     · 管子被带歪到 14°
        #   修法：**独立线程持续 spin**（env.py 就是这么做的），主循环只读缓存。
        self.leader_t = 0.0
        self.follower_t = 0.0
        self.create_subscription(JointState, '/leader/joint_states', self._cb_l, 10)
        self.create_subscription(JointState, '/follower/joint_states', self._cb_f, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/follower/arm_controller/joint_trajectory', 10)
        self._leader_torque = self.create_client(SetBool, '/leader/set_torque')

        self._stop_evt = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        # ★ 差分映射基准：进入 direct 模式时记录一次 从臂−主臂，之后永久使用。
        #   q_t = leader[:4] + offset[:4]
        #   进入瞬间 q_t 恰好等于从臂当前角 → 绝不跳变；之后从臂只跟随主臂的
        #   **变化量**，所以主臂可以有自己的归位偏置（桌子略高 → 抬高），
        #   从臂不会跟着走。详见 _arm_direct()。
        #   （max_relative_target 被设为 0.0 后限幅关闭，唯一的"防跳变"就靠这里。）
        self._offset = np.zeros(N_ARM)

        self.kin = SO101Kin(tcp_offset=0.0)   # 管尖偏移不影响方向，先按工具系原点
        q5 = START_JOINT_POSE[:N_ARM]
        _, self.R_ref = self.kin.fk(q5)        # 姿态参考：保持管子竖直
        self.q_posture = q5.copy()             # 零空间：把构型钉在起点附近

        # 统计
        self.n_steps = 0
        self.n_ik_fail = 0
        self.n_clip = 0
        self.n_moved = 0
        self.max_abs_dp_req = np.zeros(3)
        self.max_abs_a = np.zeros(3)
        self.max_dp_applied = np.zeros(3)
        self.max_track = 0.0

    # ------------------------------------------------------------ 回调
    @staticmethod
    def _reorder(msg):
        if not msg.name:
            return None
        try:
            return np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES],
                            dtype=float)
        except ValueError:
            return None

    def _cb_l(self, m):
        p = self._reorder(m)
        if p is not None:
            self.leader_q = p
            self.leader_t = time.time()

    def _cb_f(self, m):
        p = self._reorder(m)
        if p is not None:
            self.follower_q = p
            self.follower_t = time.time()

    def _spin_loop(self):
        """后台持续 spin —— 保证缓存值始终是最新的。"""
        while not self._stop_evt.is_set():
            try:
                rclpy.spin_once(self, timeout_sec=0.02)
            except Exception:
                break

    def shutdown(self):
        """停掉后台 spin 线程。

        ⚠️ **必须在 destroy_node() 之前调用**：否则线程还在 spin 一个已销毁的
        节点 → C++ terminate → `Aborted (core dumped)`。功能虽已完成，但退出会崩。
        """
        try:
            self._stop_evt.set()
        except Exception:
            pass
        t = getattr(self, '_spin_thread', None)
        if t is not None:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass

    def wait_states(self, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.leader_q is not None and self.follower_q is not None:
                return True
            time.sleep(0.05)
        return False

    def age(self):
        """返回 (leader 数据年龄, follower 数据年龄)，秒。"""
        now = time.time()
        return (now - self.leader_t if self.leader_t else 9.9,
                now - self.follower_t if self.follower_t else 9.9)

    def _arm_direct(self):
        """进入 direct 模式时调用：记录两臂的**基准差**，之后一直用它做差分映射。

        映射定义（4 自由度）：
            从臂目标 = 主臂当前角 + 基准差
            基准差   = 从臂进入时角 − 主臂进入时角     ← 进入瞬间记录一次，永久有效

        为什么必须是差分、而不是"从臂直接用主臂的绝对角"：
            · 进入瞬间 `从臂目标 = 主臂 + (从臂 − 主臂) = 从臂当前角` → **绝不跳变**
            · 之后从臂只跟随主臂的**变化量**，两臂各自停在自己的"家"上
            · ★ 主臂允许有**自己的归位偏置**（`phase1_home.py --leader-lift`
              把主臂抬高，因为主臂那张桌子略高）。若用绝对角映射，
              **主臂抬高多少、从臂就跟着抬多少** → 从臂会离开 Phase 0 已验证的
              插管起点。差分映射从根上消除这个耦合。
            · 从臂 `max_relative_target = 0.0`（限幅关闭），"不跳变"必须由这里保证。

        踩坑（2026-09）：原先这里是"1 秒内把权重从 1 衰减到 0"，即**从差分过渡到
        绝对**。两臂归位到同一姿态时 offset≈0，两种写法一样，所以一直没暴露；
        一旦给主臂加归位偏置，衰减到 0 就等价于"把偏置当人工输入"，从臂会被拖走。
        """
        if self.leader_q is not None and self.follower_q is not None:
            self._offset = self.follower_q[:N_ARM] - self.leader_q[:N_ARM]
            d4 = np.degrees(self._offset[:4])
            gap = float(np.abs(self._offset[:4]).max())
            print('  差分映射基准（从臂 = 主臂 + 该偏置，主臂怎么归位都不影响从臂）：')
            print('    ' + '  '.join(f'{n}={v:+.2f}°'
                                     for n, v in zip(ARM_JOINTS[:4], d4)))
            if gap > 0.05:
                print(f'    ℹ️  两臂当前相差 {math.degrees(gap):.1f}°：'
                      f'从臂**保持原位不动**，只跟随主臂接下来的动作')

    # ------------------------------------------------------------ 下发
    def pub_traj(self, q5):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(ARM_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q5]
        pt.time_from_start.nanosec = 3_000_000      # 3ms：立刻到该点，对发布抖动免疫
        msg.points = [pt]
        self.traj_pub.publish(msg)

    def set_leader_torque(self, on: bool) -> bool:
        if not self._leader_torque.wait_for_service(timeout_sec=5.0):
            print('  ⚠️  /leader/set_torque 不可用')
            return False
        req = SetBool.Request()
        req.data = on
        fut = self._leader_torque.call_async(req)
        # ★ 不能在这里再调 rclpy.spin_once：后台 spin 线程已经在 spin 同一个节点，
        #   两个线程同时 spin 同一个节点是未定义行为。这里只等 future。
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 8.0:
            time.sleep(0.02)
        ok = fut.done() and fut.result() is not None and fut.result().success
        print(f"  /leader/set_torque -> {'ON' if on else 'OFF'}  {'✅' if ok else '❌'}")
        return ok

    # ------------------------------------------------------------ 主循环
    def run(self, duration=None, start_mode='cart'):
        if not self.wait_states():
            print('❌ 拿不到 joint_states（launch 在跑吗？需要 follower + leader 桥）')
            return False

        print()
        print('  ⚠️  下一步会**松开主臂的力矩**（松手后主臂会因重力下垂）。')
        print('      请先【用手扶住主臂】，再按 Enter。')
        try:
            input('      准备好后按 Enter ... ')
        except KeyboardInterrupt:
            print('\n已取消。')
            return False

        # 录制/回程阶段主臂必须**松开**（人手自由推动）
        self.set_leader_torque(False)
        time.sleep(0.3)

        p_L_prev = self.kin.fk(self.leader_q[:N_ARM])[0].copy()
        mode = start_mode
        mode_switches = 0
        self.n_stale = 0
        if mode == 'direct':
            self._arm_direct()

        print()
        print('=' * 72)
        print('  模式：**DIRECT 直接关节映射**（无任何变换）')
        print(f'    跟随   : {" / ".join(CONTROLLED_JOINTS)}'
              f'   ← 只剩 {len(CONTROLLED_IDX)} 个自由度')
        print(f'    冻结   : {" / ".join(FROZEN_JOINTS)}'
              f'   ← 指令 = 实测值，永远不发改变它们的指令')
        print('             （wrist_roll 管子轴对称转了没意义；gripper 固定夹着试管；'
              'shoulder_pan 插孔位置固定）')
        print('             ⚠️ 冻结 pan 的代价：末端被限制在一个竖直平面内，'
              '只有 2 个平移自由度')
        print()
        print('  【空格】切换 DIRECT(直接映射) ↔ CART(笛卡尔，实测不好用)   【q】退出')
        print(f'  当前模式：{"DIRECT 直接关节映射" if mode=="direct" else "CART 笛卡尔"}')
        print('=' * 72)
        print()

        t0 = time.time()
        last_report = t0
        with Key() as kb:
            while True:
                loop_t = time.time()
                if duration is not None and loop_t - t0 > duration:
                    print(f'\n  到达设定的 {duration:.0f}s，退出。')
                    break

                ch = kb.read()
                if ch in ('q', 'Q', '\x1b'):
                    print('\n  收到退出键。')
                    break
                if ch == ' ':
                    mode = 'direct' if mode == 'cart' else 'cart'
                    mode_switches += 1
                    # 切换时重置 leader 参考点，避免累积位移造成跳变
                    p_L_prev = self.kin.fk(self.leader_q[:N_ARM])[0].copy()
                    print(f'  ⇄ 切到 {"CART 笛卡尔" if mode=="cart" else "DIRECT 直接映射"}')
                    if mode == 'direct':
                        self._arm_direct()

                # 后台 spin 线程负责收数据；主循环只读缓存
                if self.leader_q is None or self.follower_q is None:
                    time.sleep(self.dt)
                    continue

                # ★ 新鲜度检查：数据太旧就跳过本步，绝不拿旧值算 Δp / IK 基准。
                #   这是上面那个"单步 Δp=87mm"假象的直接对策。
                a_l, a_f = self.age()
                if a_l > 0.25 or a_f > 0.25:
                    self.n_stale += 1
                    p_L_prev = self.kin.fk(self.leader_q[:N_ARM])[0].copy()
                    time.sleep(self.dt)
                    continue

                self.n_steps += 1
                lq5 = self.leader_q[:N_ARM]
                fq5 = self.follower_q[:N_ARM]

                if mode == 'direct':
                    # ---- ★ 直接关节映射（4 自由度，无任何变换）----
                    # 只跟随 pan / lift / elbow / wrist_flex 这 4 个关节；
                    # wrist_roll 与 gripper **冻结在保存的起点值**：
                    #   · wrist_roll：管子是轴对称的，绕自身轴转对插入任务没有意义，
                    #                 放开只会引入噪声
                    #   · gripper   ：本任务夹爪固定夹着试管，不参与控制
                    #     （gripper 不在这条 5 关节轨迹里，控制器本来就不碰它）
                    #
                    # ★ 参与控制的**只有第 2/3/4 维**：shoulder_lift / elbow_flex /
                    #   wrist_flex。第 1 维 shoulder_pan、第 5 维 wrist_roll、
                    #   第 6 维 gripper 全部**冻结**（见 so101_home_pose.FROZEN_JOINTS）。
                    #   冻结做法：把这几维的指令替换成**实测值**（= 不命令它），
                    #   而不是写死一个目标角 —— 写死会把残差变成一次跳变。
                    #   ⚠️ 代价：冻结 pan 后末端被限制在一个竖直平面内，
                    #      只有 2 个平移自由度（实测 s3/s1 ≈ 1.5e-15）。
                    q_t = lq5[:4] + self._offset[:4]      # ★ 差分映射，永不变基准
                    # 差分映射下从臂 = 从臂起点 + 主臂变化量，人手把主臂推到头时
                    # 从臂可能超限，显式夹一下（控制器本身也会夹，这是第二道保险）
                    q_t = np.clip(q_t, Q_LOW[:4], Q_HIGH[:4])
                    q_t = np.concatenate([q_t, [FIXED_WRIST_ROLL]])
                    q_t = hold_frozen(q_t, fq5)           # ★ 冻结维度 = 实测值
                    self.pub_traj(q_t)
                    if self.follower_q is not None:
                        self.max_track = max(
                            self.max_track,
                            float(np.abs(self.follower_q[:N_ARM] - q_t).max()))
                else:
                    # ---- 笛卡尔（保留备查：实测在本硬件上不好用）----
                    p_L = self.kin.fk(lq5)[0]
                    dp = p_L - p_L_prev
                    p_L_prev = p_L.copy()
                    self.max_abs_dp_req = np.maximum(self.max_abs_dp_req, np.abs(dp))

                    if float(np.linalg.norm(dp)) < self.deadband:
                        a = np.zeros(3)                 # 死区：主臂静止噪声不算干预
                    else:
                        raw = dp / self.action_scale
                        a = np.clip(raw, -1.0, 1.0)
                        if np.any(np.abs(raw) > 1.0):
                            self.n_clip += 1            # 削顶：说明尺度偏小 / 人手太快
                    self.max_abs_a = np.maximum(self.max_abs_a, np.abs(a))

                    dp_cmd = a * self.action_scale
                    self.max_dp_applied = np.maximum(self.max_dp_applied, np.abs(dp_cmd))
                    if np.any(dp_cmd != 0.0):
                        self.n_moved += 1

                    p_f = self.kin.fk(fq5)[0]
                    p_target = p_f + dp_cmd
                    q_t, info = self.kin.ik_solve(
                        fq5, p_target, R_ref=self.R_ref, q_posture=self.q_posture,
                        tol=1e-3, max_iter=30, damping=0.02,
                        orient_weight=self.orient_weight)
                    if info['converged']:
                        # ★ 冻结同样要过 —— IK 会把 pan/wrist_roll 也解出来，
                        #   不过这一道，按空格切到 CART 就能把第 1 维带走。
                        q_t = hold_frozen(q_t, fq5)
                        self.pub_traj(q_t)
                    else:
                        self.n_ik_fail += 1             # 不收敛就**保持不动**，绝不发坏目标

                    if self.follower_q is not None:
                        self.max_track = max(
                            self.max_track,
                            float(np.abs(self.follower_q[:N_ARM] - q_t).max()))

                # 每 3 秒简报一次
                now = time.time()
                if now - last_report > 3.0:
                    last_report = now
                    pa = self.kin.fk(self.follower_q[:N_ARM])[0]
                    da = self.kin.diagnose(self.follower_q[:N_ARM])
                    al, af = self.age()
                    print(f'  [{mode:<5}] 管尖=[{pa[0]:+.3f},{pa[1]:+.3f},{pa[2]:+.3f}] '
                          f'竖直={da["tube_angle"]:5.2f}° '
                          f'削顶={self.n_clip:4d}/{self.n_steps} '
                          f'IK失败={self.n_ik_fail:3d} 陈旧={self.n_stale:3d} '
                          f'数据龄={al*1000:4.0f}/{af*1000:4.0f}ms '
                          f'跟踪误差={math.degrees(self.max_track):5.1f}°')
                    if self.max_track > TRACK_ABORT:
                        print('\n❌ 从臂跟随误差过大，中止。')
                        break

                sleep = self.dt - (time.time() - loop_t)
                if sleep > 0:
                    time.sleep(sleep)

        self.report()
        return True

    # ------------------------------------------------------------ 报告
    def report(self):
        print()
        print('=' * 72)
        print('  本次统计')
        print('=' * 72)
        print(f'  总步数            : {self.n_steps}')
        print(f'  数据陈旧而跳过的步数: {self.n_stale}   （>0 说明 spin 又跟不上，或 launch 有问题）')
        print(f'  实际发出动作的步数 : {self.n_moved}')
        print(f'  IK 不收敛（保持不动）: {self.n_ik_fail}')
        print(f'  **削顶步数**       : {self.n_clip}  ({100.0*self.n_clip/max(self.n_steps,1):.1f}%)')
        print()
        print('  削顶率是调 ACTION_SCALE 的关键指标：')
        print('    > 20%  → 尺度偏小，人手经常被削顶（从臂跟不上手）')
        print('    1~20%  → 合理')
        print('    ~ 0%   → 尺度偏大，精度有余但可能太"肉"')
        print()
        ax = ['x', 'y', 'z']
        print(f"  {'轴':<4}{'人手Δp峰值':>14}{'归一化动作峰值':>16}{'实际下发Δp峰值':>16}")
        for i in range(3):
            print(f'  {ax[i]:<4}{self.max_abs_dp_req[i]*1000:>12.2f}mm'
                  f'{self.max_abs_a[i]:>16.3f}{self.max_dp_applied[i]*1000:>14.2f}mm')
        print()
        print(f'  从臂最大跟随误差 : {math.degrees(self.max_track):.2f}°')
        print(f'  CART 模式削顶次数: {self.n_clip}   （若为 0 且你感觉"不够跟手"，'
              f'说明是舵机动力学而非尺度问题）')
        print('=' * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--action-scale', type=float, default=0.010,
                    help='每步最大位移（米）。默认 0.010 = 1cm/步 @10Hz = 10cm/s')
    ap.add_argument('--deadband', type=float, default=0.002,
                    help='主臂静止死区（米），默认 2mm')
    ap.add_argument('--rate', type=float, default=10.0, help='控制频率，默认 10Hz（=ENV_HZ）')
    ap.add_argument('--orient-weight', type=float, default=0.5, help='IK 管子方向权重')
    ap.add_argument('--duration', type=float, default=None, help='跑多少秒后自动退出')
    ap.add_argument('--start-mode', choices=['direct', 'cart'], default='direct')
    ap.add_argument('--preview', action='store_true', help='只打印配置并退出，不动作')
    args = ap.parse_args()

    if args.preview:
        print()
        print('=' * 72)
        print('  主臂→从臂 遥操作 —— 配置预览（不发布任何指令）')
        print('=' * 72)
        print('  模式 direct（默认，推荐）：**直接关节映射，无任何变换**')
        print(f'    跟随     : {", ".join(ARM_JOINTS[:4])}')
        print(f'    冻结     : wrist_roll = {FIXED_WRIST_ROLL:+.6f} rad')
        print(f'               gripper    = {FIXED_GRIPPER:+.6f} rad')
        print(f'    频率     : {args.rate:.0f} Hz   起始模式: {args.start_mode}')
        print()
        print('  模式 cart（备查，实测在本硬件上不好用）：')
        print(f'    ACTION_SCALE = {args.action_scale*1000:.1f} mm/步'
              f'  → 速度上限 {args.action_scale*args.rate*100:.1f} cm/s')
        print(f'    死区 = {args.deadband*1000:.1f} mm   IK 方向权重 = {args.orient_weight}')
        print('=' * 72)
        return 0

    rclpy.init()
    node = CartesianTeleop(action_scale=args.action_scale, deadband=args.deadband,
                           rate=args.rate, orient_weight=args.orient_weight)
    ok = False
    try:
        ok = node.run(duration=args.duration, start_mode=args.start_mode)
    except KeyboardInterrupt:
        print('\n中断（Ctrl-C）。从臂保持在当前位置。')
        try:
            node.report()
        except Exception:
            pass
    finally:
        # ★ 必须先停后台 spin 线程，再销毁节点。
        #   否则线程仍在 spin 一个已销毁的节点 → C++ terminate → Aborted (core dumped)。
        try:
            node.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.try_shutdown()
        except Exception:
            pass
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
