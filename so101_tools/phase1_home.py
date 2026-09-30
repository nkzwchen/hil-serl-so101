#!/usr/bin/env python3
"""Phase 1：双臂自动归位（需求 1+2）。

需求（来自使用者的原始描述）：
    1. 给从臂找一个初始位置，每次自动生成轨迹，将**从臂和主臂同时**慢慢移动到此位置
    2. 然后此时**力矩同时控制**从臂和主臂
    3. （Phase 3 才做）按空格恢复主臂操作时，只要求 x,y,z 三个方向的增量

本脚本实现 1+2：
    · 用 min-jerk 插值生成平滑轨迹（首末速度、加速度都为 0，无冲击）
    · follower 与 leader **各自从自己的当前位置**插值到自己的目标，
      所以两臂即使初始相距很远，也不会出现"甩过去"的猛跳
    · 到位后双臂力矩都保持 ON：follower 保持位置，leader 跟随

★ 两臂的**目标不是同一个姿态**（2026-09 需求）：
    主臂那张桌子比从臂这边略高，所以主臂要抬得高一些，否则夹爪贴着自己的
    桌面 → 堵转 → 舵机过热保护（真机事故，见 安全边界-主臂.md §0）。
    默认让主臂比从臂多抬 2.5cm —— 见 `--leader-lift`（填 0 关闭）。
    只影响主臂目标，从臂目标一个数都不动。
    姿态/限位/偏置的定义全部在共享模块 so101_home_pose.py，本文件不另写一份。

★ 这一步顺带修掉一个现存隐患：
    env.py 进入 mode A 时会调 _drive_leader(cur)，把 follower 的当前位置
    **直接甩**给 leader。如果两臂当时差得远，leader 会猛跳（只靠
    max_relative_target=20° 兜底）。归位流程让两臂都平滑收敛到各自的目标。

★ 已验证的接口约定（读源码确认，不是猜的）：
    /follower/joint_states              6 关节，**弧度**（gripper 由 (norm/100)*π 转换）
    /leader/joint_states                6 关节，**弧度**（同上）
    /follower/arm_controller/joint_trajectory
                                        JointTrajectory，**只含 5 个臂关节**，不含 gripper
    /leader/arm_command                 Float64MultiArray，**必须 6 个值**，单位弧度
    /follower/set_torque, /leader/set_torque
                                        std_srvs/SetBool；leader ON=跟随 OFF=人手自由

⚠️ 夹爪处理：本任务夹爪固定夹着试管，归位时**不动夹爪**。
   follower 不碰 gripper action；leader 只把它当前的 gripper 值原样回发（保持不动）。

用法：
    python phase1_home.py --preview      # 只打印轨迹计划，不动机械臂
    python phase1_home.py               # 执行归位（会提示确认）
    python phase1_home.py --duration 6  # 更慢（默认 4 秒）
    python phase1_home.py --leader-lift 5   # 想抬更多（默认 2.5cm）
    python phase1_home.py --leader-lift 0   # 两臂归到同一姿态（关闭主臂抬高）
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
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ---- 起点 / 限位 / 主臂抬高偏置：全部来自共享模块，**不要在这里另写一份** ----
# 单一真相源：hil-serl/examples/experiments/so101/so101_home_pose.py
# （主臂和从臂的家姿态**不一样**：主臂桌面略高，所以主臂要抬得高一些。
#   任何"读两臂姿态"的代码都必须区分绝对语义 / 差分语义，详见该模块 docstring。）
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
if _SO101_DIR not in sys.path:
    sys.path.insert(0, _SO101_DIR)
from so101_keys import NonBlockingKey  # noqa: E402
from so101_home_pose import (  # noqa: E402
    ARM_JOINTS, JOINT_NAMES, LEADER_LIFT_DEFAULT_CM,
    N_ARM, Q_HIGH, Q_LOW, START_JOINT_POSE,
    leader_lift_offset, min_jerk, save_home_cmd, tool_z,
)

# 到位判据
SETTLE_TOL = 0.05          # rad，约 2.9°
SETTLE_TIMEOUT = 6.0       # s（已不用于等待收敛，仅作参考）
# ★ 迭代修正（层 1）参数
SETTLE_HOLD = 1.0          # 每轮轨迹走完后的"稳定期"（秒），之后才测残差
CORRECTION_TOL = 0.010     # rad ≈ 0.57°：残差小于它就停止修正
CORRECTION_MAX = 0.35      # rad ≈ 20°：指令相对目标的**累计**偏移上限（防跑飞）
FOLLOWER_TRACK_WARN = 0.09   # rad ≈ 5°：超了就说明跟得不够好
FOLLOWER_TRACK_ABORT = 0.35  # rad ≈ 20°：完全跟不上，有异常
LEADER_TRACK_ABORT = 0.60    # rad ≈ 34°：放宽（原本 26°）。超了通常是人手按住主臂
# （min_jerk 现在从共享模块 so101_home_pose 导入，不再在此重复定义）


class ArmHomer(Node):
    def __init__(self):
        super().__init__('arm_homer')
        self.follower_pos = None
        self.leader_pos = None
        # ★ 数据时间戳，用于新鲜度检查（见 home() 里的说明）
        self.follower_t = 0.0
        self.leader_t = 0.0
        # ★ 50Hz 全量轨迹记录（用于分析"停顿"和速度尖峰 —— 主循环只有 20Hz，看不见）
        self.trace = []
        self._tracing = False
        # ★ 精确的"停顿"证据：每次下发指令的时刻。
        #   位置阈值（step<0.0015rad）等价于速度<4.3°/s，**慢速运动会被误判成停顿**
        #   （实测：4.8° 走 8s 峰值仅 1.1°/s，整条主轨迹都被判成"静止"）。
        #   直接记录"代码有没有在发指令"才是与速度无关的精确判据。
        self.pub_times = []
        self.create_subscription(JointState, '/follower/joint_states', self._cb_f, 10)
        self.create_subscription(JointState, '/leader/joint_states', self._cb_l, 10)
        self.traj_pub = self.create_publisher(
            JointTrajectory, '/follower/arm_controller/joint_trajectory', 10)
        self.leader_pub = self.create_publisher(
            Float64MultiArray, '/leader/arm_command', 10)
        # ★ 力矩 service client 必须**只建一次**并复用。
        #   踩坑：原先在 _set_torque() 里每次 create_client，新建的 client 需要
        #   重新走 DDS 发现，3 秒超时内 future 常常没完成 → 误报 `-> ON ❌`，
        #   结果是 leader 力矩根本没打开、整条 leader 动作全部失效。
        from std_srvs.srv import SetBool
        self._torque_cli = {
            '/follower/set_torque': self.create_client(SetBool, '/follower/set_torque'),
            '/leader/set_torque': self.create_client(SetBool, '/leader/set_torque'),
        }
        # ★★ 后台 spin 线程（2026-09 修）★★
        #   原先主循环里用 rclpy.spin_once(timeout_sec=0.0)，每圈只处理 1 条消息；
        #   而两路 joint_states 各 50Hz 合计 100 条/秒，主循环只有 20Hz
        #   → **队列必然积压** → 读到的是几百毫秒前的旧值。
        #   双重危害：
        #     1) 跟踪误差统计与"到位判据"全部基于旧数据 → 报出来的数字不可信
        #     2) 每圈变慢 → 更容易落后下面那个"绝对时间表" → 触发爆发式补发
        #        （这正是"主臂动得好、从臂动得差"的根因链）
        #   改为独立线程持续 spin，主循环只读缓存 + 检查新鲜度。
        self._stop_evt = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    @staticmethod
    def _reorder(msg: JointState):
        if not msg.name:
            return None
        try:
            return np.array([msg.position[msg.name.index(j)] for j in JOINT_NAMES], dtype=float)
        except ValueError:
            return None

    def _cb_f(self, msg):
        p = self._reorder(msg)
        if p is not None:
            self.follower_pos = p
            self.follower_t = time.time()
            if self._tracing:
                self.trace.append((self.follower_t, p[:N_ARM].copy()))

    def _cb_l(self, msg):
        p = self._reorder(msg)
        if p is not None:
            self.leader_pos = p
            self.leader_t = time.time()

    def _spin_loop(self):
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

    def age(self):
        """返回 (leader 数据年龄, follower 数据年龄)，秒。"""
        now = time.time()
        return (now - self.leader_t if self.leader_t else 9.9,
                now - self.follower_t if self.follower_t else 9.9)

    # ---------------------------------------------------------------- 等待
    def wait_states(self, timeout=6.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.follower_pos is not None and self.leader_pos is not None:
                return True
            time.sleep(0.05)
        return False

    # ---------------------------------------------------------------- 下发
    def pub_follower(self, q5, dt_sec=None):
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(ARM_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q5]
        # ★ time_from_start 必须用 3ms，与 env.py / LeaderTeleopComponent 保持一致。
        #
        # 踩坑记录（2026-09）：我一开始"自作聪明"用了 tick 周期（20Hz → 50ms），
        # 结果归位过程中机械臂**一顿一顿地晃**。原因：
        #   本脚本用 Python 的 time.sleep 配速，发布间隔必然抖动。
        #   time_from_start=50ms 的语义是"用 50ms 平滑走到这个点"——若某个点迟到，
        #   控制器已经把上一段插值走完、停在原地等下一个点，到了再走 → 一顿一顿。
        #   而 3ms 的语义是"立刻到这一点"：控制器永远追最新目标，
        #   **对发布抖动天然免疫**，所以原实现用 3ms 是有道理的。
        #
        # 现象佐证：那次运行日志里 140 次 `Relative goal position magnitude had to be
        # clamped to be safe` **全部集中在 elbow_flex**（present≈48.9° 而目标 4.97°，
        # 相差 44°）—— 正是"跟不上指令"的表现。
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 3_000_000
        msg.points = [pt]
        self.traj_pub.publish(msg)
        if self._tracing:
            self.pub_times.append(time.time())

    def pub_leader(self, q5, gripper_rad):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in q5] + [float(gripper_rad)]
        self.leader_pub.publish(msg)

    # ---------------------------------------------------------------- 归位
    # ------------------------------------------------------------ 测量 / 单轮
    def _measure(self, n=12, dt=0.02):
        """稳定测量：取 n 次采样均值（机械臂应处于静止保持状态）。"""
        fs, ls = [], []
        for _ in range(n):
            if self.follower_pos is not None:
                fs.append(self.follower_pos[:N_ARM].copy())
            if self.leader_pos is not None:
                ls.append(self.leader_pos[:N_ARM].copy())
            time.sleep(dt)
        f = np.mean(fs, axis=0) if fs else None
        l = np.mean(ls, axis=0) if ls else None
        return f, l

    def _measure_still(self, timeout=3.0, still_tol=0.0015, n=12):
        """★ 先等机械臂**真正静止**，再取均值。

        为什么必须这样（2026-09 实测教训）：
            迭代修正每轮只走很小一段（常常 < 1°），若在机械臂仍在收敛爬行时就测
            残差，测到的值会偏，于是"差多少补多少"就补错了。症状是**残差非单调**：
                第 0 轮后 4.23° → 第 1 轮后 0.26°（很好）
                → 第 2 轮后 0.91°（反而变差）→ 第 3 轮后 1.06°
            第 1 轮本来已经到位，后两轮却把它推离了。
        """
        prev = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            f, _ = self._measure(n=3, dt=0.02)
            if f is None:
                time.sleep(0.05)
                continue
            if prev is not None and float(np.abs(f - prev).max()) < still_tol:
                break
            prev = f
        return self._measure(n=n)

    # ------------------------------------------------------------ 单轮
    def _pass(self, qf_from, ql_from, qf_to, ql_to, dur, rate, lg, kb, st,
              drive_leader=True, label='main'):
        """走一段 min-jerk 轨迹 + 短稳定保持。返回 True 表示被中止。

        st 是跨轮累计的统计 dict。
        """
        dt = 1.0 / rate
        n_steps = max(int(dur * rate), 2)
        # ★ prev 必须从**实测位置**初始化，不能用指令 qf_from！
        #   踩坑（2026-09）：修正段报出 4.96° 的"跳变"，数值正好等于该轮指令偏移量 ——
        #   因为指令已带偏置、而实测在下垂位置，第一帧把两者的差当成了跳变。**纯测量假象。**
        if self.follower_pos is not None:
            prev_f = self.follower_pos[:N_ARM].copy()
        else:
            prev_f = qf_from.copy()
        prev_t = time.time()
        jump_local = 0.0      # 本段最大**瞬时速度**（deg/s），与循环耗时无关
        for k in range(n_steps + 1):
            loop_t = time.time()
            s = min_jerk(k / n_steps)
            qf = qf_from + (qf_to - qf_from) * s
            ql = ql_from + (ql_to - ql_from) * s

            # 新鲜度检查：数据太旧就跳过本步，避免按节拍"爆发式补发"
            a_l, a_f = self.age()
            if a_l < 0.25 and a_f < 0.25:
                self.pub_follower(qf, dt)
                if drive_leader:
                    self.pub_leader(ql, lg)
            else:
                self.n_stale += 1

            # ★ 只有【确实在驱动主臂】时才检查它的跟随误差。
            #   Bug 修复（2026-09）：原先漏了 drive_leader 条件，于是
            #   `--no-leader` 时也拿主臂（松着、停在完全不同的位姿）跟指令比，
            #   必然算出上百度的"误差"，直接把整个归位误判为异常而中止。
            #   主臂没被驱动 → 它没有"跟随"的义务 → 不该检查。
            if drive_leader and self.leader_pos is not None:
                lerr_v = np.abs(self.leader_pos[:N_ARM] - ql)
                st['max_l_per'] = np.maximum(st['max_l_per'], lerr_v)
                err = float(lerr_v.max())
                st['max_l'] = max(st['max_l'], err)
                if err > LEADER_TRACK_ABORT:
                    print(f'\n❌ leader 跟随误差过大（{math.degrees(err):.1f}°）—— '
                          f'可能有人按住主臂或被挡住，已中止。')
                    return True

            if self.follower_pos is not None:
                fp = self.follower_pos[:N_ARM]
                ferr_v = np.abs(fp - qf)
                st['max_f_per'] = np.maximum(st['max_f_per'], ferr_v)
                ferr = float(ferr_v.max())
                st['max_f'] = max(st['max_f'], ferr)
                # ★ 用**速度**（位移/实际耗时）而不是每帧位移：
                #   每帧位移会被循环耗时污染（某帧慢了就显示成大跳变），速度才干净。
                now_t = time.time()
                dt_act = max(now_t - prev_t, 1e-3)
                vel = float(np.abs(fp - prev_f).max()) / dt_act
                st['max_jump'] = max(st['max_jump'], vel)
                jump_local = max(jump_local, vel)
                prev_f = fp.copy()
                if ferr > FOLLOWER_TRACK_ABORT:
                    print(f'\n❌ follower 跟随误差过大（{math.degrees(ferr):.1f}°）—— '
                          f'可能被挡住或舵机异常，已中止。')
                    return True

            if kb.pressed():
                print('\n⏹  收到中止键，停止下发指令（机械臂保持在当前位置）。')
                return True

            # ★ 相对睡眠（绝不用绝对时间表，否则落后时会爆发式补发）
            sleep = dt - (time.time() - loop_t)
            if sleep > 0:
                time.sleep(sleep)

        # ---- 短稳定期 ----
        # ⚠️ 这里**不能**等"收敛到目标"：受重力下垂的关节（肘）永远到不了指令值，
        #    等下去只会白耗满 6 秒。改为固定短稳定期，之后由迭代修正去补。
        t0 = time.time()
        while time.time() - t0 < SETTLE_HOLD:
            self.pub_follower(qf_to, dt)
            if drive_leader:
                self.pub_leader(ql_to, lg)
            time.sleep(dt)
        st.setdefault('jumps', {})[label] = jump_local
        return False

    # ---------------------------------------------------------------- 归位
    def home(self, target6, duration=4.0, rate=20.0, preview=False,
             corrections=3, corr_dur=1.5, drive_leader=True, leader_offset=None):
        """双臂归位 + **迭代修正**（层 1 优化）。

        为什么要迭代修正：
            舵机是比例位置环且 `I_Coefficient = 0`（无积分项），要顶住重力就必须存在
            `e = τ_重力 / Kp` 的稳态偏移。所以**开环插值走完，机械臂就是停在目标附近
            而不是目标上**（实测肘关节差 3~4°，力臂最长所以最明显）。

            但下垂是**静态、可重复**的：同一个指令每次停的位置几乎一样。
            于是不需要建模、不需要改舵机增益，**差多少补多少**即可：

                ① 走完主轨迹，稳定后测残差 e = 目标 − 实测
                ② 新指令 = 旧指令 + e     （把下垂量补进指令）
                ③ 再走一段短修正，回到 ①
                重复若干轮，误差指数收敛。

            安全边界：单轮修正量有上限，指令本身也夹在关节限位内，避免跑飞。

        leader_offset:
            **主臂专用**的 5 关节角度偏置（rad）。默认 None = 两臂归到同一姿态。
            主臂桌面略高时用 `--leader-lift` 生成它，让主臂停得高一点。
            只影响主臂目标；从臂目标仍是 target6。
        """
        if not self.wait_states():
            print('❌ 拿不到 /follower/joint_states 或 /leader/joint_states。'
                  'launch 在跑吗？（需要 follower + leader 桥）')
            return False

        trg5 = target6[:N_ARM]
        # ★ 主臂独立目标。偏置经过 min/max 限位夹取，不会把指令送出工作空间。
        off_l = (np.zeros(N_ARM) if leader_offset is None
                 else np.asarray(leader_offset, dtype=float).reshape(-1)[:N_ARM])
        trg_l = np.clip(trg5 + off_l, Q_LOW, Q_HIGH)
        f0 = self.follower_pos[:N_ARM].copy()
        l0 = self.leader_pos[:N_ARM].copy()
        lg = float(self.leader_pos[5])          # leader gripper 原样保持

        print()
        print('=' * 70)
        print('双臂归位计划（含迭代修正）')
        print('=' * 70)
        print(f'  主轨迹 {duration:.1f}s @ {rate:.0f}Hz，min-jerk 插值')
        print(f'  修正轮数 ≤ {corrections}，每轮 {corr_dur:.1f}s，'
              f'容差 {math.degrees(CORRECTION_TOL):.2f}°，单轮修正上限 '
              f'{math.degrees(CORRECTION_MAX):.0f}°')
        print()
        has_off = bool(np.any(off_l))
        if has_off:
            print(f"  {'关节':<16}{'follower起点':>14}{'leader起点':>13}"
                  f"{'从臂目标':>12}{'主臂目标':>12}{'主臂偏置':>12}")
            print('  ' + '-' * 78)
            for i, n in enumerate(ARM_JOINTS):
                print(f'  {n:<16}{f0[i]:>14.4f}{l0[i]:>13.4f}{trg5[i]:>12.4f}'
                      f'{trg_l[i]:>12.4f}{math.degrees(off_l[i]):>11.2f}°')
        else:
            print(f"  {'关节':<16}{'follower 起点':>16}{'leader 起点':>16}{'目标':>14}")
            print('  ' + '-' * 62)
            for i, n in enumerate(ARM_JOINTS):
                print(f'  {n:<16}{f0[i]:>16.4f}{l0[i]:>16.4f}{trg5[i]:>14.4f}')
        print()
        print(f'  follower 需移动最大关节量 = {math.degrees(np.abs(trg5 - f0).max()):.1f}°')
        print(f'  leader   需移动最大关节量 = {math.degrees(np.abs(trg_l - l0).max()):.1f}°')
        if has_off:
            # ★ 把"主臂更高了"这件事**当场量出来**，不靠感觉。
            #   注意：主臂只是人手握的把手，"管子是否竖直"对它没有意义，
            #   所以这里只关心夹爪高度（撞桌风险），不管管子倾角。
            z_ref, z_l = tool_z(trg5), tool_z(trg_l)
            if not math.isnan(z_ref) and not math.isnan(z_l):
                print(f'  主臂夹爪高度：与从臂同角度时 {z_ref*100:.1f}cm → '
                      f'加偏置后 {z_l*100:.1f}cm '
                      f'（{"抬高" if z_l > z_ref else "压低"} '
                      f'{abs(z_l-z_ref)*100:.1f}cm）')
            # 沿**真实路径**（l0 → trg_l 的逐关节 min-jerk）采样，看全程会不会低头
            path = [l0 + min_jerk(s / 20.0) * (trg_l - l0) for s in range(21)]
            zs = [tool_z(q) for q in path]
            if not any(math.isnan(z) for z in zs):
                print(f'  主臂路径上夹爪高度   = {min(zs)*100:.1f} ~ {max(zs)*100:.1f}cm '
                      f'（min-jerk 21 点采样；全程 ≥ {min(zs)*100:.1f}cm 才不会撞桌）')
        print('=' * 70)

        if preview:
            print('  --preview：只打印计划，不发布任何指令。')
            return True

        print()
        print('  ⚠️  leader 即将被电机驱动。请确保【手离开主臂】，并确认周围无障碍。')
        print('  ⚠️  归位过程中按【空格】或 Ctrl-C 可随时中止（会停在当前位置并保持力矩）。')
        try:
            input('  按 Enter 开始，Ctrl-C 取消 ... ')
        except KeyboardInterrupt:
            print('\n已取消。')
            return False

        if drive_leader:
            # ★ 力矩打不开就**立刻中止**，绝不继续下发指令。
            #   踩坑（2026-09，真机事故）：原先只打印 ❌ 就继续跑，于是给一个
            #   使不上力的臂持续下发指令 —— 既不安全，又白费。
            #   实测原因：舵机**过热保护**，写入被拒绝：
            #       RuntimeError: Failed to write 'Torque_Enable' on id_=3 with '1'
            #       [RxPacketError] Overheat error!
            #   过热通常是"被顶住还持续对抗"造成的（见迭代修正的卡住保护）。
            if not self._set_torque('/leader/set_torque', True):
                print()
                print('  ❌ 主臂力矩无法打开 —— 立即中止（不发送任何指令）。')
                print('     最常见原因：舵机【过热保护】，表现为')
                print('       RxPacketError: Overheat error!')
                print('     处理：停止驱动该臂、等它冷却（实测约 1~2 分钟自行恢复）；')
                print('           急用可先加 --no-leader 只归位从臂。')
                return False
            time.sleep(0.45)
            # ★★ 开力矩后必须重新读一次主臂位置，并用它当轨迹起点 ★★
            #   理由一：l0 是"按 Enter 之前"测的，中间人手可能碰过主臂。
            #   理由二：开力矩这个动作**本身就可能让主臂动**，有两种已知机制：
            #     ① 舵机本来松着 —— `Torque_Enable=1` 时会立刻冲向 `Goal_Position`
            #        寄存器里的旧值（该寄存器在力矩关闭期间不更新）→ **向上猛跳**。
            #        现场："主臂突然跳起来，又回去了"，第一帧就落后参考 34.8°。
            #     ② 舵机本来就有力矩，而 bridge 无条件把 `Goal_Position` 抹成实测
            #        位置 → 抹掉了顶重力的静差偏置（比例环无积分项）→ **向下垂几度**。
            #        现场：2026-09-29 实测下垂 4.0°。
            #   两种机制都已在 bridge 侧处理（只对确实松着的电机做同步）。
            #   这里留一道保险：正常应**纹丝不动**。
            if self.leader_pos is not None:
                l0_new = self.leader_pos[:N_ARM].copy()
                delta = l0_new - l0
                jumped = float(np.abs(delta).max())
                if math.degrees(jumped) > 3.0:
                    # 用 shoulder_lift（力臂最长、负重最明显）的方向辅助判断成因
                    d_lift = math.degrees(delta[1])
                    why = ('向上猛跳 → 像机制①：舵机原本松着，冲向了旧 Goal_Position'
                           '（bridge 没重编/没重启？）'
                           if d_lift > 0 else
                           '向下垂 → 像机制②：Goal_Position 被抹平，顶重力的静差偏置没了'
                           '（bridge 无条件同步了带力矩的电机？）')
                    print()
                    print(f'  ⚠️⚠️  开力矩后主臂移动了 {math.degrees(jumped):.1f}° —— '
                          f'正常应该纹丝不动！')
                    print('         逐关节位移：'
                          + '  '.join(f'{n}={math.degrees(d):+.2f}°'
                                      for n, d in zip(ARM_JOINTS, delta)))
                    print(f'         判断：{why}')
                    print('         处置：确认 bridge 是**最新版**且已重启 ——')
                    print('           cd ~/pi_project/ros2_ws && colcon build '
                          '--packages-select so101_ros2_bridge')
                    print('           然后重启 launch，再跑本脚本。')
                    print('         本次已中止，不发送任何轨迹指令。')
                    return False
                if jumped > 1e-6:
                    print(f'  ℹ️  开力矩后主臂位置变化 {math.degrees(jumped):.2f}°（在容差内）')
                l0 = l0_new
                lg = float(self.leader_pos[5])
        else:
            print('  (--no-leader: 本次不驱动主臂，只移动从臂)')

        self.n_stale = 0
        self._best_f = None      # 卡住保护的逐关节最佳残差基准
        self._best_l = None
        self.trace = []
        self.pub_times = []
        self._tracing = True
        st = {
            'max_f': 0.0, 'max_l': 0.0, 'max_jump': 0.0,
            'max_f_per': np.zeros(N_ARM), 'max_l_per': np.zeros(N_ARM),
        }
        # 实际下发的指令（会随迭代修正偏移）；初始 = 目标
        cmd_f = trg5.copy()
        cmd_l = trg_l.copy()       # ★ 主臂用带偏置的目标
        aborted = False
        hist = []          # [(轮次, follower 残差, leader 残差)]

        with NonBlockingKey() as kb:
            # ---- 第 0 轮：主轨迹 ----
            aborted = self._pass(f0, l0, cmd_f, cmd_l, duration, rate, lg, kb, st,
                                 drive_leader=drive_leader, label='main')

            # ---- 迭代修正 ----
            it = 0
            while not aborted and it < corrections:
                fm, lm = self._measure_still()
                if fm is None or lm is None:
                    break
                e_f = trg5 - fm
                e_l = trg_l - lm       # ★ 主臂的目标是 trg_l，不是 trg5
                w_f, w_l = float(np.abs(e_f).max()), float(np.abs(e_l).max())
                hist.append((it, w_f, w_l))
                print(f'  第 {it} 轮后残差：follower {math.degrees(w_f):5.2f}° / '
                      f'leader {math.degrees(w_l):5.2f}°')
                # ★ --no-leader 时不看主臂（否则主臂那个必然很大的误差会让
                #   "提前停止"永远不触发，白跑满 3 轮修正）
                worst = max(w_f, w_l) if drive_leader else w_f
                if worst < CORRECTION_TOL:
                    print('  ✅ 已在容差内，停止修正')
                    break
                # ★★ 卡住保护（防止把舵机顶到过热）★★
                #   真机事故（2026-09）：主臂某关节被顶住 → 迭代修正持续加大它的
                #   指令偏移（日志里涨到 10°、13°）→ 舵机堵转发热 → **过热保护** →
                #   力矩再也打不开（RxPacketError: Overheat error），主臂彻底动不了。
                #   对策：某关节残差**不降反升**就放弃对它修正，不再加大偏移。
                e_f_abs, e_l_abs = np.abs(e_f), np.abs(e_l)
                if getattr(self, '_best_f', None) is None:
                    self._best_f, self._best_l = e_f_abs.copy(), e_l_abs.copy()
                else:
                    stuck_f = e_f_abs > self._best_f + 0.003
                    stuck_l = e_l_abs > self._best_l + 0.003
                    if stuck_f.any():
                        print(f'  ⚠️ 这些关节残差不降反升，放弃对其修正（防过热）: '
                              f'{[ARM_JOINTS[i] for i in np.where(stuck_f)[0]]}')
                        e_f = np.where(stuck_f, 0.0, e_f)
                    if stuck_l.any():
                        print(f'  ⚠️ 主臂这些关节残差不降反升，放弃修正: '
                              f'{[ARM_JOINTS[i] for i in np.where(stuck_l)[0]]}')
                        e_l = np.where(stuck_l, 0.0, e_l)
                    self._best_f = np.minimum(self._best_f, e_f_abs)
                    self._best_l = np.minimum(self._best_l, e_l_abs)

                it += 1
                # ★★ 关键修正：修正轮必须从「上一条指令」起步，不是「实测位置」★★
                #   踩坑（2026-09）：原先从实测位置起步，于是产生锯齿：
                #     主轨迹结束时 指令 = cmd_old（目标+偏置），机械臂停在 = 目标
                #     修正轮一开场把指令写回「实测位置(≈目标)」
                #     → **指令突跳回退了整个偏置量（4°）**
                #     → 舵机卸力、机械臂往回坠 → 轨迹再把它推出去
                #   每轮边界都来一次「退回—再前进」，这就是顿挫的来源。
                #   证据：某次运行主轨迹只动了 0.2°，却测出 2.46° 的单步跳变 ——
                #         主轨迹没动，跳变只能来自修正轮。
                #   改为从 cmd_old 起步后，指令序列连续，不再有轮边界突跳。
                prev_f, prev_l = cmd_f.copy(), cmd_l.copy()
                cmd_f = np.clip(cmd_f + e_f, Q_LOW, Q_HIGH)
                cmd_l = np.clip(cmd_l + e_l, Q_LOW, Q_HIGH)
                # 安全边界：指令相对目标的偏移不超上限，避免跑飞
                cmd_f = np.clip(cmd_f, trg5 - CORRECTION_MAX, trg5 + CORRECTION_MAX)
                cmd_l = np.clip(cmd_l, trg_l - CORRECTION_MAX, trg_l + CORRECTION_MAX)
                print(f'  第 {it} 轮修正：指令偏移 follower '
                      f'{math.degrees(np.abs(cmd_f - trg5).max()):.2f}° / leader '
                      f'{math.degrees(np.abs(cmd_l - trg_l).max()):.2f}°')
                aborted = self._pass(
                    prev_f, prev_l, cmd_f, cmd_l, corr_dur, rate, lg, kb, st,
                    drive_leader=drive_leader, label=f'corr{it}')

        # ---- 结果 ----
        self._tracing = False
        fm, lm = self._measure_still(n=8)
        ef = float(np.abs(trg5 - fm).max()) if fm is not None else float('nan')
        el = float(np.abs(trg_l - lm).max()) if lm is not None else float('nan')
        print()
        print('=' * 70)
        print(f'  归位{"中止" if aborted else "完成"}')
        pre = (f'（迭代前 {math.degrees(hist[0][1]):.3f}°）' if hist else '（未跑到修正轮）')
        print(f'  follower 最终误差 = {math.degrees(ef):7.3f}°   {pre}')
        if drive_leader:
            pre_l = (f'（迭代前 {math.degrees(hist[0][2]):.3f}°）' if hist
                     else '（未跑到修正轮）')
            print(f'  leader   最终误差 = {math.degrees(el):7.3f}°   {pre_l}')
        else:
            print(f'  leader   最终误差 = {math.degrees(el):7.3f}°   （--no-leader：'
                  f'主臂未被驱动，此数字无意义）')
        if len(hist) >= 2:
            print('  迭代收敛过程：')
            for it_, wf, wl in hist:
                print(f'    第 {it_} 轮后: follower {math.degrees(wf):5.2f}° / '
                      f'leader {math.degrees(wl):5.2f}°')
        print(f'  过程中 follower 最大跟随误差 = {math.degrees(st["max_f"]):7.3f}°  '
              f'← ★ 主要判据（目标 < {math.degrees(FOLLOWER_TRACK_WARN):.0f}°）')
        print(f'  过程中 leader   最大跟随误差 = {math.degrees(st["max_l"]):7.3f}°  '
              f'（次要，放宽到 {math.degrees(LEADER_TRACK_ABORT):.0f}°）')
        # 理论 min-jerk 峰值速度 = 1.875 × 最大关节位移 / 时长
        theo = 1.875 * float(np.abs(trg5 - f0).max()) / max(duration, 1e-6)
        move_deg = math.degrees(float(np.abs(trg5 - f0).max()))
        if move_deg > 5.0:
            extra = (f'（主轨迹理论 min-jerk 峰值 {math.degrees(theo):.2f} °/s，'
                     f'比值 {st["max_jump"]/max(theo,1e-9):.2f}×）')
        else:
            # 位移太小时理论峰值趋近 0，比值没有意义（会显示成几十倍）
            extra = f'（本次位移仅 {move_deg:.1f}°，理论峰值比值无意义）'
        print(f'  过程中最大瞬时速度        = {math.degrees(st["max_jump"]):7.2f} °/s  {extra}')
        if st.get('jumps'):
            seg = '  '.join(f'{k}={math.degrees(v):.1f}°/s' for k, v in st['jumps'].items())
            print(f'    └ 逐段归因(峰值速度)：{seg}')
        print(f'  数据陈旧而跳过下发的步数  = {self.n_stale}')
        print()
        print('  ── 逐关节误差分解 ──')
        fin_f = np.abs(trg5 - fm) if fm is not None else np.full(N_ARM, np.nan)
        fin_l = np.abs(trg_l - lm) if lm is not None else np.full(N_ARM, np.nan)
        print(f"  {'关节':<16}{'follower最大':>14}{'follower最终':>14}"
              f"{'leader最大':>13}{'leader最终':>13}")
        for i, n in enumerate(ARM_JOINTS):
            print(f'  {n:<16}{math.degrees(st["max_f_per"][i]):>13.2f}°'
                  f'{math.degrees(fin_f[i]):>13.2f}°'
                  f'{math.degrees(st["max_l_per"][i]):>12.2f}°'
                  f'{math.degrees(fin_l[i]):>12.2f}°')
        print('=' * 70)
        self._analyze_trace()

        # ★ 成功判据同样要区分：--no-leader 时主臂没被驱动，不能拿它的误差否决
        ok = (not aborted) and ef < SETTLE_TOL and              (el < SETTLE_TOL if drive_leader else True)
        if ok:
            # ★ 把收敛后的**指令**存档，供 env.reset() 做重力静差补偿。
            #   为什么必须存：舵机是比例位置环、无积分项，发"名义角"时手臂会停在
            #   名义角**下方**（实测低 3.65cm、管子歪 8.75°）。归位脚本的迭代修正
            #   已经把这个静差补进 cmd_f 了，丢掉就白跑了。
            try:
                p = save_home_cmd(cmd_f, trg5, leader_cmd=cmd_l, extra={
                    'follower_final_err_deg': math.degrees(ef),
                    'leader_final_err_deg': math.degrees(el) if el == el else None,
                    'leader_driven': bool(drive_leader),
                    'leader_offset_deg': [float(math.degrees(v)) for v in off_l],
                })
                print(f'  💾 归位指令已存档（reset 的重力补偿用）: {p}')
                print('     follower 补偿偏置 = '
                      + ' '.join(f'{math.degrees(v):+.2f}°'
                                 for v in (cmd_f - trg5)))
            except Exception as e:                       # noqa: BLE001
                print(f'  ⚠️  存档归位指令失败（{type(e).__name__}: {e}）—— '
                      f'env.reset() 的重力补偿会退化为 0')
            if drive_leader:
                print('  ✅ 双臂已到位，力矩保持 ON —— 需求 1+2 达成')
            else:
                print('  ✅ 从臂已到位（本次未驱动主臂）')
            return True
        if aborted:
            print('  ⚠️  被中止')
        elif ef >= SETTLE_TOL:
            print(f'  ⚠️  从臂未完全到位：{math.degrees(ef):.2f}°')
        else:
            print(f'  ⚠️  主臂未完全到位：{math.degrees(el):.2f}°')
        return False

    # ------------------------------------------------------------ 轨迹分析
    def _analyze_trace(self):
        """分析 50Hz 全量轨迹：区分「走走停停」和「速度尖峰」两种顿挫。

        为什么要单独做：主循环只有 20Hz 且只统计最大值，**停顿根本不会体现在
        最大值里**；而 50Hz 的全量记录能看出机械臂何时静止、何时又动起来。
        """
        if len(self.trace) < 20:
            print('  (轨迹样本不足，跳过分析)')
            return
        ts = np.array([t for t, _ in self.trace])
        ps = np.stack([q for _, q in self.trace])
        dur = ts[-1] - ts[0]
        rate = len(ts) / max(dur, 1e-6)

        # ★ 先重采样到**均匀 50Hz 网格**再求速度。
        #   直接对原始时间戳求差会有假象：/joint_states 的消息偶尔成串到达
        #   （两条间隔 1ms），此时 位移/极小dt 会算出几十甚至上百 °/s 的假尖峰。
        #   重采样后每条样本的 dt 恒为 20ms，速度才可比。
        GRID = 0.02
        t_u = np.arange(ts[0], ts[-1], GRID)
        if len(t_u) < 4:
            print('  (轨迹太短，跳过分析)')
            return
        ps_u = np.stack([np.interp(t_u, ts, ps[:, i]) for i in range(N_ARM)], axis=1)
        step = np.abs(np.diff(ps_u, axis=0)).max(axis=1)
        vel = step / GRID                            # rad/s

        # 静止判据：相邻两帧位移 < 0.0015 rad（约 0.09°，接近编码器分辨率）
        # 静止判据：相邻两帧位移 < 0.0015 rad（约 0.09°，接近编码器分辨率）
        still = step < 0.0015
        # 数「停顿段」：静止连续 >= 0.3s 算一次停顿；同时累计停顿时长与最长停顿
        n_stop, run, longest = 0, 0, 0
        stop_frames = int(0.3 / GRID)
        for st_ in still:
            run = run + 1 if st_ else 0
            longest = max(longest, run)
            if run == stop_frames:
                n_stop += 1
        moving = ~still
        total_pause = float(still.sum()) * GRID
        print()
        print('  ── 50Hz 轨迹分析（区分「走停」与「速度尖峰」）──')
        print(f'    原始采样 {len(ts)} 帧 / {dur:.1f}s ({rate:.1f}Hz)，'
              f'重采样到均匀 {1/GRID:.0f}Hz 后 {len(t_u)} 帧')
        # ★ 与速度阈值无关的精确判据：代码有没有在发指令
        if len(self.pub_times) > 2:
            pt = np.array(self.pub_times)
            gaps = np.diff(pt)
            sil = gaps[gaps > 0.15]          # >150ms 未下发 = 真静默
            print(f'    指令静默(>150ms)次数  = {len(sil)}   ← ★ 与速度无关的精确判据')
            print(f'    静默总时长 / 最长一次 = {sil.sum():.1f}s / '
                  f'{(sil.max() if len(sil) else 0.0):.1f}s')
            print(f'    下发间隔 中位/最大    = {np.median(gaps)*1000:.0f}ms / '
                  f'{gaps.max()*1000:.0f}ms')
        print(f'    运动帧占比            = {100.0*moving.sum()/len(still):.1f}%')
        print(f'    停顿次数(位置判据,仅供参考) = {n_stop}'
              f'   ← ⚠️ 与速度阈值耦合，慢速运动会被误判，别跨位移量级比较')
        print(f'    停顿总时长 / 最长一次 = {total_pause:.1f}s / {longest*GRID:.1f}s')
        if moving.any():
            print(f'    运动段峰值速度        = {math.degrees(vel[moving].max()):.2f} °/s')
            print(f'    运动段平均速度        = {math.degrees(vel[moving].mean()):.2f} °/s')
            # 速度"抖动": 运动段内相邻速度变化的 95 分位
            v = vel[moving]
            if len(v) > 3:
                jit = np.abs(np.diff(v))
                print(f'    速度变化 95 分位      = {math.degrees(np.percentile(jit,95)):.2f} °/s '
                      f'（越小越顺；接近峰值说明有顿挫）')

    # ---------------------------------------------------------------- 力矩
    def _set_torque(self, srv_name, on: bool):
        cli = self._torque_cli[srv_name]
        if not cli.wait_for_service(timeout_sec=5.0):
            print(f'  ⚠️  服务 {srv_name} 不可用')
            return False
        req = SetBool.Request()
        req.data = on
        fut = cli.call_async(req)
        # ★ 不能再调 rclpy.spin_once：后台 spin 线程已在 spin 同一节点。
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 8.0:
            time.sleep(0.02)
        if not fut.done():
            print(f'  {srv_name} -> 调用超时（8s 未返回）❌')
            return False
        ok = fut.result() is not None and fut.result().success
        print(f'  {srv_name} -> {"ON" if on else "OFF"}  {"✅" if ok else "❌"}')
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=4.0, help='归位时长（秒）')
    ap.add_argument('--rate', type=float, default=20.0, help='下发频率（Hz）')
    ap.add_argument('--preview', action='store_true', help='只打印计划，不动作')
    ap.add_argument('--corrections', type=int, default=3,
                    help='迭代修正的最大轮数（层1优化）；0 = 关闭修正，退回纯开环')
    ap.add_argument('--corr-dur', type=float, default=1.5,
                    help='每轮修正的轨迹时长（秒）')
    ap.add_argument('--no-leader', action='store_true',
                    help='诊断用：不驱动主臂，只移动从臂')
    ap.add_argument('--force-leader', action='store_true',
                    help='配合 --offset 使用：强制也驱动主臂（危险，未验证主臂是否会撞桌）')
    ap.add_argument(
        '--leader-lift', type=float, default=LEADER_LIFT_DEFAULT_CM, metavar='CM',
        help=f'★ 主臂专用：让主臂比从臂**多抬几厘米**（正数=抬高；'
             f'默认 {LEADER_LIFT_DEFAULT_CM:g}cm，因为主臂那张桌子略高）。'
             f'只改主臂目标，从臂目标一个数都不动；填 0 可关闭。'
             f'内部只动 shoulder_lift（抬升效率最高、几乎纯竖直），'
             f'并用正运动学二分求解，不写死系数。')

    ap.add_argument(
        '--offset', default='',
        help='压力测试用：在起点上叠加的角度偏移（度，逗号分隔 5 个臂关节）。'
             '已离线验证安全的预设：0,55,-65,0,0 （65° 移动，管尖最低 0.041m 不撞桌）')
    args = ap.parse_args()

    rclpy.init()
    node = ArmHomer()
    target = START_JOINT_POSE.copy()
    # ★ --offset 的偏移只在【从臂】上离线验证过安全（管尖不撞桌），
    #   **从未验证主臂**。主臂撞桌会被顶住 → 舵机过热保护 → 力矩再也打不开。
    #   所以 --offset 时默认不驱动主臂，除非显式 --force-leader。
    leader_off = args.no_leader
    if args.offset and not args.no_leader and not args.force_leader:
        print('  ⚠️  --offset 的偏移只在从臂上验证过安全性；为避免把主臂顶到桌上'
              '（会导致舵机过热保护），本次自动不驱动主臂。')
        print('      确实要驱动主臂请另加 --force-leader。')
        leader_off = True
    if args.offset:
        off = np.array([float(v) for v in args.offset.split(',')], dtype=float)
        if off.shape[0] != N_ARM:
            print(f'❌ --offset 需要 {N_ARM} 个逗号分隔的数，收到 {off.shape[0]} 个')
            node.shutdown()          # 先停 spin 线程，再销毁节点
            node.destroy_node()
            rclpy.try_shutdown()
            return 1
        target[:N_ARM] = target[:N_ARM] + np.radians(off)
        print(f'（压力测试模式：目标 = 起点 + {args.offset} 度）')

    # ★ 主臂专用抬高偏置（从臂目标不受影响）
    leader_offset = np.zeros(N_ARM)
    if abs(args.leader_lift) > 1e-9:
        leader_offset = leader_lift_offset(args.leader_lift / 100.0)
        if leader_off:
            print('  ℹ️  已加 --leader-lift，但本次不驱动主臂（--no-leader），'
                  '该偏置不会实际生效。')
    try:
        ok = node.home(target, duration=args.duration,
                       rate=args.rate, preview=args.preview,
                       corrections=args.corrections, corr_dur=args.corr_dur,
                       drive_leader=not leader_off, leader_offset=leader_offset)
    except KeyboardInterrupt:
        print('\n中断（Ctrl-C）。机械臂保持在当前位置。')
        ok = False
    finally:
        # ★ Ctrl-C 时 rclpy 的信号处理器可能已经 shutdown 过 context，
        #   再调 rclpy.shutdown() 会抛 RCLError("rcl_shutdown already called")。
        #   用 try_shutdown()（Humble 提供）+ try/except 兜住。
        # ★ 必须先停后台 spin 线程，再销毁节点。
        #   踩坑（2026-09）：加了后台 spin 线程后忘了停它，主流程 destroy_node()
        #   时线程仍在 spin 一个已销毁的节点 → C++ `terminate called without an
        #   active exception` → Aborted (core dumped)。
        #   现象具有迷惑性：功能其实都正常完成了，只是**退出时崩**。
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
            try:
                rclpy.shutdown()
            except Exception:
                pass
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
