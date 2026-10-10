"""两臂【归位姿态】+【主臂专用抬高偏置】—— **唯一真相源**。

为什么单独抽一个模块
--------------------
"主臂的家"和"从臂的家"**不是同一个姿态**：主臂那张桌子比从臂这边略高，
所以主臂归位时要抬得高一些，否则夹爪贴着自己的桌面 —— 历史上主臂 elbow_flex
就是这样被顶住 → 堵转 → 舵机过热保护 → 力矩再也打不开（真机事故，见
`安全边界-主臂.md` §0）。

这个"两臂家姿态不同"的事实必须被**每一个**用到两臂姿态的地方知道，否则就会出现
"一边按抬高后的姿态归位、另一边按绝对角映射，于是把偏置当成人工输入"的经典
打架。2026-09 就真的踩到了：`phase1_home.py` 给主臂加了偏置，而
`phase3_cartesian_teleop.py` 的 direct 模式会把软启动权重衰减到 0（= 从臂直接
用主臂绝对角），结果从臂会被抬离 Phase 0 已验证的插管起点。

所以：**姿态、限位、偏置都只在这个文件里定义一次**，其余模块一律 import。
新增任何"读两臂姿态"的代码，都必须先想清楚它是**绝对**语义还是**差分**语义。

约定（很重要）
--------------
    follower_pose  =  leader_pose  −  LEADER_HOME_OFFSET

即"从臂视角的虚拟主臂位置"。凡是把主臂位姿喂给从臂的地方，都要先减去偏置；
凡是让主臂去跟随从臂的地方，都要先加上偏置。偏置为 0 时全式退化成"两臂同姿态"，
与旧代码完全等价。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np


def min_jerk(s: float) -> float:
    """min-jerk 归一化位置曲线：s∈[0,1] → [0,1]，首末速度与加速度均为 0。

    用它的意义：轨迹两端速度和加速度都是 0，起停**没有冲击** ——
    归位时夹爪里还夹着试管，冲击会直接传到试管和舵机上。
    """
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ============================== 姿态与限位 ==============================
# 起点：主臂遥操作摆出 → 20s 采样 / 取最后 10s 均值（std < 0.004 rad）
# 已过 Phase 0 的 V2/V3 验证：729/729 工作空间点通过，管子全程距竖直 1.66~1.68°
START_JOINT_POSE = np.array([
    +0.330903,   # shoulder_pan
    +0.283856,   # shoulder_lift
    +0.086691,   # elbow_flex
    -0.342928,   # wrist_flex
    +0.057538,   # wrist_roll
    +0.324895,   # gripper（本任务固定，归位时不动它）
])

JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']
ARM_JOINTS = JOINT_NAMES[:5]
N_ARM = 5
N_ALL = 6

# 5 个臂关节的限位（取自 so101_description/urdf/so101_new_calib.urdf 的 <limit>）
Q_LOW = np.array([-1.920, -1.745, -1.690, -1.658, -2.744])
Q_HIGH = np.array([+1.920, +1.745, +1.690, +1.658, +2.841])


# ========================= 主臂专用抬高偏置 =========================
# 为什么选 shoulder_lift（起点处用正运动学实测）：
#     shoulder_lift −10°  →  夹爪抬高 5.9cm，水平只动 2mm（几乎纯竖直）
#     elbow_flex    −10°  →  夹爪抬高 4.9cm，水平要动 14mm
# 而且主臂只是**人手握的把手**，"管子竖直"只对从臂插管有意义 —— 所以主臂可以
# 随便改姿态，那就取抬得最多、又不把夹爪往旁边带的 shoulder_lift。
LEADER_LIFT_JOINT = 1          # shoulder_lift 在 ARM_JOINTS 里的下标
LEADER_LIFT_MAX_M = 0.30       # 一次最多抬 30cm（再往上 shoulder_lift 就绕过顶了）
LEADER_LIFT_FALLBACK = 1.70    # °/cm。仅在 so101_kin 加载失败时的兜底线性系数
LEADER_LIFT_DEFAULT_CM = 2.5   # ★ 默认抬高量：主臂桌面比从臂略高
#   2026-09-29：先按 5cm 做出来实测，使用后觉得"搞多了"，改成 2.5cm。
#   换算不是猜的（正运动学二分求解实测）：
#       2.5cm → shoulder_lift −4.18°（起点 +16.26° → +12.09°，夹爪 16.1 → 18.6cm）

_KIN = None


def _load_kin():
    """惰性加载同目录的 so101_kin（失败返回 None，不致命）。"""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from so101_kin import SO101Kin
        return SO101Kin(tcp_offset=0.0)   # 管尖伸出量未实测，用工具系原点
    except Exception as e:                 # noqa: BLE001 —— 缺 urdf 也不该让归位挂掉
        print(f'  ⚠️  加载 so101_kin 失败（{e}），退回兜底线性系数换算')
        return None


def _kin():
    global _KIN
    if _KIN is None:
        _KIN = _load_kin()
    return _KIN


def tool_z(q5) -> float:
    """给 5 个臂关节角，返回夹爪（工具系原点）相对基座的高度，米。"""
    k = _kin()
    if k is None:
        return float('nan')
    return float(k.fk(np.asarray(q5, dtype=float)[:N_ARM])[0][2])


def tube_tilt_deg(q5) -> float:
    """给 5 个臂关节角，返回管子距**竖直向下**的夹角（度）。

    这是插管任务最关键的姿态指标：Phase 0 验证的名义起点是 1.66°。
    ⚠️ `diagnose()['tube_angle']` 返回的就是**度**，别再 math.degrees() 一次。
    """
    k = _kin()
    if k is None:
        return float('nan')
    return float(k.diagnose(np.asarray(q5, dtype=float)[:N_ARM])['tube_angle'])


def leader_lift_offset(lift_m: float, verbose: bool = True) -> np.ndarray:
    """把「主臂夹爪抬高 lift_m 米」换算成 5 关节角度偏置（只动 shoulder_lift）。

    实现要点：用**正运动学二分求解**，而不是在写死的系数上乘一下。
    这样起点或 URDF 一改，换算自动跟着变；而且能当场把结果验回来。

    返回新数组，调用方可以随便改（不会污染别人）。
    """
    off = np.zeros(N_ARM, dtype=float)
    if abs(lift_m) < 1e-9:
        return off
    lim = Q_LOW[LEADER_LIFT_JOINT], Q_HIGH[LEADER_LIFT_JOINT]
    want = math.copysign(min(abs(lift_m), LEADER_LIFT_MAX_M), lift_m)
    if abs(lift_m) > LEADER_LIFT_MAX_M and verbose:
        print(f'  ⚠️  请求抬升 {abs(lift_m)*100:.1f}cm 超过上限 '
              f'{LEADER_LIFT_MAX_M*100:.0f}cm，已夹到上限')

    k = _kin()
    if k is None:
        # 兜底：起点处实测 −10° → +5.9cm。抬高要**减小** shoulder_lift，故取负号。
        off[LEADER_LIFT_JOINT] = -math.copysign(
            abs(want) * 100.0 * math.radians(LEADER_LIFT_FALLBACK), want)
        return off

    q0 = START_JOINT_POSE[:N_ARM].copy()
    z0 = tool_z(q0)

    def dz(d):
        q = q0.copy()
        q[LEADER_LIFT_JOINT] += d
        return tool_z(q) - z0

    # 起点附近 dz 关于 shoulder_lift 单调递减（已全量程扫过：
    # −70°→46.4cm、+16°→16.1cm、+50°→−3.8cm），所以二分一定收敛。
    # 区间同时受关节限位约束。
    lo = max(lim[0] - q0[LEADER_LIFT_JOINT], -1.45)
    hi = min(lim[1] - q0[LEADER_LIFT_JOINT], +0.90)
    if dz(lo) < want or dz(hi) > want:
        # 区间端点都到不了目标高度：夹到能到的最近端点
        off[LEADER_LIFT_JOINT] = lo if abs(dz(lo) - want) < abs(dz(hi) - want) else hi
        if verbose:
            print(f'  ⚠️  shoulder_lift 行程不足以抬 {want*100:.1f}cm，'
                  f'取端点 {math.degrees(off[LEADER_LIFT_JOINT]):+.1f}°')
        return off
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if dz(mid) < want:
            hi = mid
        else:
            lo = mid
    off[LEADER_LIFT_JOINT] = 0.5 * (lo + hi)

    if verbose:
        got = dz(off[LEADER_LIFT_JOINT])
        print(f'  主臂抬高换算：shoulder_lift '
              f'{math.degrees(off[LEADER_LIFT_JOINT]):+.2f}° '
              f'→ 夹爪 {z0*100:.1f}cm → {(z0+got)*100:.1f}cm'
              f'（实测 {got*100:+.1f}cm，目标 {want*100:+.1f}cm）')
        if abs(got - want) > 0.005:
            print(f'  ⚠️  换算残差 {(got-want)*1000:+.0f}mm，偏大，请检查')
    return off


def leader_home_offset(verbose: bool = False,
                       lift_cm: Optional[float] = None) -> np.ndarray:
    """默认主臂归位偏置（5 关节 rad）。`follower = leader − 这个偏置`。"""
    if lift_cm is None:
        lift_cm = LEADER_LIFT_DEFAULT_CM
    return leader_lift_offset(float(lift_cm) / 100.0, verbose=verbose)


# ===================== 参与控制的关节（冻结清单）=====================
# 本任务只有两个关节不参与控制：
#
#   第 5 维  wrist_roll     → 冻结（管子轴对称，绕自身轴转对插入没有意义）
#   第 6 维  gripper        → 冻结（夹爪固定夹着试管）
#
# 第 1~4 维（shoulder_pan / shoulder_lift / elbow_flex / wrist_flex）**都受控**。
#
# ★ 变更历史：2026-09-29 早先曾把 **shoulder_pan 也冻结**（动作 3 维）。当时接受的
#   代价是"末端被锁在一个竖直平面内、只有 2 个平移自由度、随机化只能 2 方向"。
#   同日改回受控，因为：
#     · 真正需要的是"末端能到孔那儿"，而 pan 是唯一的侧向自由度 —— 锁掉它就意味着
#       pan 被推动过（实测偏过 13~15°）时整个可达平面被转掉，管子够不到固定的孔
#     · 解冻后随机框可以回到**真正的三维 ±half_m**，不再受平面限制
#   解冻后 IK 变成"4 关节 / 5 约束（位置3+方向2）"看着超定，但实测没问题：
#   400/400 收敛、最差位置残差 2.0mm、管子倾角变化 0.10°、限位余量 ≥1.077rad。
#
# ⚠️ 动作维度由此从 3 变 **4**（受控关节数）。之前按 3 维录的 demo 与现在
#   **不兼容**，必须重录。
#
# ★★★ 冻结的唯一正确实现：把冻结维度的指令替换成**实测值** ★★★
#     也就是"根本不命令它"。**绝不要**写成一个固定的目标角 ——
#     那样归位残差、机械松弛、人手碰过，都会在下一刻变成一次跳变。
FROZEN_JOINTS = ('wrist_flex', 'wrist_roll', 'gripper')
FROZEN_IDX = tuple(JOINT_NAMES.index(j) for j in FROZEN_JOINTS)            # (3, 4, 5)
CONTROLLED_IDX = tuple(i for i in range(N_ALL) if i not in FROZEN_IDX)     # (0, 1, 2)
CONTROLLED_JOINTS = tuple(JOINT_NAMES[i] for i in CONTROLLED_IDX)

# 当前动作是前三个受控关节的相对角度（3 维）；不运行 Cartesian IK。
# 训练使用无夹爪输出的连续 SAC agent。
N_CONTROLLED = len(CONTROLLED_IDX)                                        # 3 actuated joints
N_ACTION = N_CONTROLLED                                                  # relative joint increments


def hold_frozen(q_cmd, q_cur):
    """把冻结关节的指令替换成**实测值** —— 即"不改变这些维度"。

    q_cmd / q_cur 长度可为 5（只含臂关节）或 6（含 gripper），顺序同 JOINT_NAMES。
    短的那个按公共前缀对齐，所以传 5 维的臂轨迹也没问题。

    为什么用实测值而不是固定目标角：实测值意味着"零回复力"，
    机器臂停在哪儿就是哪儿，**永远不会产生跳变**；写死一个目标角则会把
    当时的偏差变成下一条指令里的突跳。
    """
    out = np.array(q_cmd, dtype=float, copy=True)
    cur = np.asarray(q_cur, dtype=float).reshape(-1)
    n = min(out.shape[0], cur.shape[0])
    for i in FROZEN_IDX:
        if i < n:
            out[i] = cur[i]
    return out


def action_to_joint_delta(a, scale: float = 1.0):
    """3 维动作 → 5 关节增量（冻结维恒 0）。

    长度不对就**报错**，不做静默截断/补齐 —— 动作维数错位属于
    "跑起来不报错、但学出来是坏的"那类问题，必须当场炸掉。
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.shape[0] != N_CONTROLLED:
        raise ValueError(f'旧关节增量维数应为 {N_CONTROLLED}，收到 {a.shape[0]}')
    d = np.zeros(N_ARM, dtype=float)
    d[list(CONTROLLED_IDX)] = a * float(scale)
    return d


def joint_delta_to_action(d, scale: float = 1.0):
    """5 关节增量 → 3 维动作（只取受控维）。`action_to_joint_delta` 的逆。"""
    d = np.asarray(d, dtype=float).reshape(-1)
    return d[list(CONTROLLED_IDX)] / float(scale)


# ==================== 归位指令存档（gravity droop 补偿）====================
# ★ 为什么必须有这个东西（2026-09-29 实测，不是理论）：
#   舵机是**比例位置环且 I_Coefficient = 0**，要顶住重力就必须存在
#   `误差 = 重力力矩 / Kp` 的稳态静差。后果是：**发"名义角"，手臂停在名义角下方**。
#   实测（从臂夹着试管）：
#       发送名义起点 → 实际停在 z=12.40cm、管子距竖直 **8.75°**
#       （而 Phase 0 验证的名义起点是 z=16.05cm、管子 1.66°）
#   管子歪 8.75° 是不能接受的 —— 插管必须基本竖直。
#
#   归位脚本 phase1_home.py 用**迭代修正**把这个静差补掉了：它不断累加
#   `cmd += (目标 − 实测)`，收敛后 `cmd − 目标` 就是"要多推多少"。
#   所以这里把那个收敛后的指令**存下来**，让 env 直接复用：
#     · `reset()` 用它把手臂真正送到起点
#     · 模式 A / 模式 B 都用它当"重力偏置"（`hold_bias`）—— 不加它的话，
#       模式 A 的指令等于实测位置 → 比例误差恒为 0 → **没有顶重力的力** → 缓慢下垂
#
#   反过来也说明：**跑训练之前必须先跑一次归位脚本**生成这个存档。
HOME_CMD_FILE = os.environ.get(
    'SO101_HOME_CMD_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'so101_home_cmd.json'))


def save_home_cmd(follower_cmd, target, leader_cmd=None, extra=None):
    """把归位收敛后的指令存档。返回写入的路径。"""
    cmd = [float(v) for v in follower_cmd]
    tgt = [float(v) for v in target]
    d = {
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'target': tgt,
        'follower_cmd': cmd,
        'follower_offset': [a - b for a, b in zip(cmd, tgt)],
        'note': 'env 用它做重力静差补偿（reset 的起点 + 模式A/B 的 hold_bias）；'
                '先跑 phase1_home.py 生成',
    }
    if leader_cmd is not None:
        d['leader_cmd'] = [float(v) for v in leader_cmd]
    if extra:
        d['extra'] = extra
    tmp = HOME_CMD_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, HOME_CMD_FILE)     # 原子替换，避免读到半个文件
    return HOME_CMD_FILE


def load_home_cmd():
    """读归位指令存档；没有/坏了返回 None（调用方自己决定怎么告警）。"""
    try:
        with open(HOME_CMD_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        off = np.asarray(d['follower_offset'], dtype=float).reshape(-1)
        if off.shape[0] != N_ARM:
            raise ValueError(f'follower_offset 长度 {off.shape[0]} ≠ {N_ARM}')
        return d
    except Exception:
        return None


def follower_home_offset():
    """follower 的**重力静差补偿**偏置（5 关节 rad）。

    没有存档时返回全 0，并**大声告警** —— 全 0 意味着手臂会停在名义姿态下方
    约 3.6cm、管子歪到 8.75°，插管任务的前提就不成立了。
    """
    d = load_home_cmd()
    if d is None:
        print('  ⚠️⚠️  找不到归位指令存档，重力静差补偿 = 0 —— ')
        print('         手臂将停在名义起点**下方约 3.6cm、管子歪约 8.8°**。')
        print(f'         请先跑一次归位脚本生成：{HOME_CMD_FILE}')
        return np.zeros(N_ARM)
    return np.asarray(d['follower_offset'], dtype=float)


def start_plane_basis(kin=None, q=None):
    """返回 (n, h)：末端所在竖直平面的法向，与平面内**水平**方向。

    ⚠️ 2026-09-29 起 pan 已解冻，随机化**不再**受这个平面限制（改成了三维框）。
       本函数现在**只用于诊断/报告**，不再参与随机化。
    """
    k = kin if kin is not None else _kin()
    if k is None:
        raise RuntimeError('so101_kin 不可用')
    qq = START_JOINT_POSE[:N_ARM] if q is None else np.asarray(q, float)[:N_ARM]
    _, axes, _ = k._forward(qq)
    n = np.asarray(axes[CONTROLLED_IDX[1]], dtype=float)     # shoulder_lift 轴
    n = n / np.linalg.norm(n)
    h = np.cross(n, np.array([0.0, 0.0, 1.0]))               # 平面内、且水平
    h = h / np.linalg.norm(h)
    return n, h


# ==================== 起始位置随机化（只动 shoulder_pan）====================
# ★ 2026-09-29 改：**只随机化 pan 一个关节，不做 IK**。
#
# 为什么放弃"三维框 + IK"（真机现象：起点有时候手臂**非常靠上/收起来**）：
#   同一个 ±2.5cm 的末端框，IK 解在**关节空间**的跨度极大（实测 2000 次）：
#       shoulder_lift  36.2°      elbow_flex  52.6°      wrist_flex  31.5°
#   因为 pan 解冻后有 1 维零空间（pan 转 + 手臂"反绕"能到同一个末端位置），
#   IK 会沿这条零空间跑到完全不同的姿态上 —— 末端位置和管子倾角都对，
#   但手臂形状天差地别。当时的验证只查了末端位置和倾角，**没查姿态**，所以漏了。
#
# 只动 pan 的好处（全部是"精确"的，不是近似）：
#   · 管子倾角**精确不变** —— pan 绕竖直轴转，倾角大小不受影响
#   · 高度**精确不变** —— 末端在一个水平圆上移动
#   · 其余关节**一个都不动** —— 不可能跑到别的姿态分支
#   · 不需要 IK，所以快得多、也不可能不收敛
# 代价：随机化是沿水平**圆弧**（切向一个方向），不是一个水平面方块。
#   —— 用户要的就是这个："相当于只水平移动"。
#
# half_m 仍然是**米**（末端水平位移），内部按实际水平半径换算成 pan 角度。
RANDOMIZE_HALF_M = 0.01      # 末端水平 ±1cm（此前三维框是 ±2.5cm，太大了）


def randomize_joint_start_pose(rng, half_deg=1.5, q_cur=None, kin=None):
    """Sample pan/lift/elbow independently; compensate wrist to retain pitch."""
    if not math.isfinite(half_deg) or half_deg < 0:
        raise ValueError('randomize_joint_half_deg must be finite and nonnegative')
    q_nom = START_JOINT_POSE[:N_ARM].copy()
    if q_cur is not None:
        q_nom[4] = np.asarray(q_cur, dtype=float)[4]
    k = kin if kin is not None else _kin()
    if k is None:
        return q_nom, {'ok': False, 'reason': 'so101_kin 不可用'}
    delta = np.zeros(N_ARM)
    delta[:3] = np.radians(rng.uniform(-half_deg, half_deg, 3))
    delta[3] = -delta[1] - delta[2]
    q = q_nom + delta
    d0, d = k.diagnose(q_nom), k.diagnose(q)
    displacement = k.fk(q)[0] - k.fk(q_nom)[0]
    margin = float(np.min(np.minimum(q - Q_LOW[:N_ARM], Q_HIGH[:N_ARM] - q)))
    valid = bool(np.all(np.isfinite(q)) and margin >= 0 and
                 np.isfinite(d['tube_angle']) and d['tube_angle'] <= 3.0)
    info = dict(ok=valid, dx_m=float(displacement[0]), dy_m=float(displacement[1]),
                dz_m=float(displacement[2]), err_mm=0.,
                tube_angle_deg=float(d['tube_angle']),
                d_tilt_deg=float(d['tube_angle'] - d0['tube_angle']),
                margin_rad=margin, iters=0, randomization_mode='joint_uniform_v1',
                joint_half_deg=float(half_deg), joint_delta_deg=np.degrees(delta).tolist())
    if not valid:
        info['reason'] = '关节随机起点越界或倾角超过 3°'
    return q, info


def randomize_start_pose(rng, half_m: float = RANDOMIZE_HALF_M, q_cur=None,
                         kin=None, tol: float = 2e-3, max_iter: int = 300):
    """只随机化 `shoulder_pan`，让末端沿**水平圆弧**移动 ±half_m 米。

    其余关节**一点不动**（保持名义起点），所以三件事是**精确**成立的：
      · 管子距竖直的夹角不变 —— pan 绕竖直轴转，倾角大小不受影响
      · 末端高度不变 —— 末端在一个水平圆上移动
      · 不可能跑到别的姿态分支 —— 根本不做 IK

    为什么放弃"三维框 + IK"（真机现象：起点有时候手臂**非常靠上/收起来**）：
      同一个 ±2.5cm 的末端框，IK 解在**关节空间**的跨度极大（实测 2000 次）：
          shoulder_lift 36.2°   elbow_flex 52.6°   wrist_flex 31.5°
      pan 解冻后有 1 维零空间（pan 转 + 手臂"反绕"能到同一个末端位置），
      IK 会沿这条零空间跑到完全不同的姿态上。当时的验证只查了末端位置和倾角、
      **没查姿态**，所以漏掉了。只动 pan 就从根上消除了这件事。

    代价：随机化沿水平**圆弧**（切向一个方向），不是一个水平面方块 ——
    用户要的就是这个："相当于只水平移动"。

    `tol` / `max_iter` 保留只为兼容旧调用签名，现在不使用（没有迭代）。

    返回 (q5, info)；`info['dx_m'/'dy_m'/'dz_m']` 是 **FK 实测**的末端位移。
    """
    q_nom = START_JOINT_POSE[:N_ARM].copy()
    k = kin if kin is not None else _kin()
    if k is None:
        return q_nom, {'ok': False, 'reason': 'so101_kin 不可用'}

    locked = [i for i in FROZEN_IDX if i < N_ARM]      # (4,) = wrist_roll
    q = q_nom.copy()
    if q_cur is not None:                              # 冻结关节取实测值
        qc = np.asarray(q_cur, dtype=float).reshape(-1)
        for i in locked:
            if i < qc.shape[0]:
                q[i] = qc[i]

    p_nom = k.fk(q)[0]                                 # pan 未动之前的真实末端
    d0 = k.diagnose(q)
    # ★★ 末端到 pan 轴的**垂直距离**，不是 hypot(x, y) ★★
    #   pan 轴确实竖直（与竖直夹角 0.0002°），但它过的是 (x=+0.0388, y=0)，
    #   **不过基座原点**：垂直距离 R = 0.375381，而 hypot(x,y) = 0.412288。
    #   用 hypot 会把弦长多算 1.098 倍 ⇒ 实测位移只有请求值的 0.9105 倍
    #   （踩坑 2026-09-29：3000 次抽样实测最大位移 0.910cm 而不是 1.000cm，
    #     一开始还误以为是"抽样没抽到边界"，其实是公式错。）
    #   做法：让 FK 自己在名义点 ±eps 处量一次弦长，反解出半径 ——
    #   这比手算轴位置更稳，轴倾角/偏置/换 URDF 都不用改这里。
    eps = 1e-3                                         # 0.057°，纯数值探针
    qp = q.copy(); qp[0] += eps
    qm = q.copy(); qm[0] -= eps
    r_h = float(np.linalg.norm(k.fk(qp)[0] - k.fk(qm)[0])
                / (2.0 * math.sin(eps)))
    if r_h < 1e-6:
        return q, {'ok': False, 'reason': '末端几乎在 pan 轴上，pan 转不动它'}

    d_want = float(rng.uniform(-half_m, half_m))
    ratio = min(abs(d_want) / (2.0 * r_h), 1.0)
    d_pan = math.copysign(2.0 * math.asin(ratio), d_want)   # 弦长 d 对应的圆心角
    if abs(d_pan) > math.radians(45.0):                # 安全上限，正常到不了
        return q, {'ok': False,
                   'reason': f'算出的 pan 角度 {math.degrees(d_pan):.1f}° 过大'}
    q[0] += d_pan

    # ---- 复核（不再是"IK 收敛性"，而是"确实只水平移动"）----
    d = k.diagnose(q)
    p_fin = k.fk(q)[0]
    delta = p_fin - p_nom
    margin = float(d['margin_min'])
    locked_ok = all(abs(q[i] - q_nom[i]) < 1e-9 for i in locked
                    ) if q_cur is None else True
    # ⚠️ diagnose() 的 'tube_angle' 已经是「度」，别再 math.degrees() 一次
    #    （踩坑 2026-09-29：多转一次把 1.66° 显示成 95.19°）
    dtilt_deg = float(d['tube_angle'] - d0['tube_angle'])
    out = {
        # 阈值取"物理上无意义"的量级（10µm / 0.0001°）—— 实测噪声是
        # 6 纳米 / 1.0 微度，都在下面三四个数量级之外。
        # （写成 1e-12 就比浮点精度还严了，会把数值噪声判成失败。）
        'ok': bool(abs(delta[2]) < 1e-5 and abs(dtilt_deg) < 1e-4
                   and margin > 0.0 and locked_ok),
        'dx_m': float(delta[0]), 'dy_m': float(delta[1]), 'dz_m': float(delta[2]),
        'pan_deg': float(math.degrees(d_pan)),
        'want_m': d_want,                                  # 请求位移（带正负号）
        'r_pan_m': r_h,                                    # 末端到 pan 轴的垂直距离
        'err_mm': float(np.linalg.norm(delta)) * 1000.0,   # 实际水平位移
        'tube_angle_deg': float(d['tube_angle']),
        'd_tilt_deg': dtilt_deg,                           # 恒为 0
        'margin_rad': margin,
        'iters': 0,
        'locked_fixed': bool(locked_ok),
        'p_target': p_fin, 'p_fin': p_fin,
    }
    if not out['ok']:
        out['reason'] = (f'未通过复核（dz={out["dz_m"]*1000:.3f}mm 应为 0, '
                         f'倾角变化={dtilt_deg:.3e}° 应为 0, '
                         f'限位余量={margin:.3f}rad, 冻结关节未动={locked_ok}）')
    return q, out
