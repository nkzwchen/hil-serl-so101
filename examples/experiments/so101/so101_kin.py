#!/usr/bin/env python3
"""SO101 运动学：URDF → 自实现 FK / 雅可比 / 基座系微分 IK。

设计要点（对应 Phase 0 方案）：

1. **全链在基座系（base_link）**，不做任何坐标变换。
   动作 (dx,dy,dz) 直接是 base 系平移增量。

2. **工具系用 URDF 里已有的 `gripper_frame_link`**（纯坐标系，无 visual/collision）。
   URDF 中 gripper_frame_joint: gripper_link -> gripper_frame_link,
   xyz=(-0.0079,-0.0002,-0.0981) rpy=(0, π, 0)。

   ═══ ⚠️ 重要：管子沿的是 `+X_tool`，不是 `+Z_tool`（2026-09 实测确认）═══
   只看 URDF 会得出错误结论：gripper_frame_joint 的偏移是沿 gripper_link 的 -Z，
   而 rpy=(0,π,0) 使工具系 +Z = gripper_link 的 -Z，所以**纯几何推导**会认为
   工具轴（插入方向）是 `+Z_tool`。

   但实机实测推翻了这个推导：用 leader 遥操作把管子摆成**竖直朝下**后，读关节角算 FK：
       X_tool = [-0.029, +0.001, -1.000]  → 距竖直仅  1.7°   ← 与"管子朝下"完全吻合
       Z_tool = [+0.945, -0.325, -0.028]  → 距竖直   88.4°   （水平）
   即：**URDF 标称的工具坐标系取向与物理夹持方向并不一致**，
   管子实际沿 `+X_tool` 伸出。这类"标称坐标系 ≠ 实际夹持方向"的偏差不会报错，
   但会让 tcp_pose 观测、插入方向判据、防撞桌下限全部错位，属于最难查的一类 bug。

   ⇒ 因此本模块把管子轴向做成参数 `tube_axis`（工具系内），**默认 (1,0,0)**，
     并且**改机械臂/夹爪/URDF 后必须重新实测确认**。

3. **臂只有 5 个转动关节**，而我们要控制 3 维位置 → 5-3 = 2 个冗余自由度。
   所以 IK 用**阻尼最小二乘 + 零空间正则**，不能用朴素 J^-1；
   否则相邻目标点可能解到不同 IK 分支，真机上表现为"莫名抽搐"。

4. `gripper` 转动关节在 `gripper_link` 处**分支出去**驱动活动夹爪，
   不在到工具系的链上，因此不影响管尖位置 —— 这是"夹爪固定"能成立的前提。

═══════════════════════════════════════════════════════════════════════════
⚠️ 为什么不直接用 PyKDL（2026-09 实测踩坑记录，别再走回头路）
═══════════════════════════════════════════════════════════════════════════
PyKDL 的 `Joint(name, origin: Vector, axis: Vector, RotAxis)` 构造出的
Joint 本身是**正确**的：`Joint.pose(q)` 会把原点固定在 origin、绕轴旋转。

但 `Segment(joint, f_tip)` 的行为与 KDL 文档不符。最小复现（origin=(1,0,0),
axis=z, f_tip=Identity）：

    Joint.pose(0.0).p   = (1, 0, 0)          ← 正确
    Segment.pose(0.0).p = (0, 0, 0)          ← 原点被丢掉！
    Segment.pose(0.5).p = (1-cos.5, -sin.5, 0) = (I - R)·origin

即 Segment 把关节原点当成了"绕父系原点旋转"的支点，与 KDL 文档语义不符。
后果是 FK 全错（q=0 时管尖算在 (0,0,0)），且误差随链长累积。

因此本模块**完全不用 PyKDL**，FK 与雅可比都按 URDF 语义自实现，
并用**数值微分**独立校验雅可比（见 jacobian_numeric）。这比"双实现互校"
更强，因为它直接验证了导数本身。

另一个必须避免的坑：URDF 的 `rpy` 是**固定轴 X-Y-Z（外旋）**，
即 R = Rz(y)·Ry(p)·Rx(r)。`_rpy_to_R` 按此实现。
（已实测确认 KDL 的 Rotation.RPY 与该约定一致，误差 5.6e-17；
  问题只出在 Segment，不在 RPY。）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from urdf_parser_py.urdf import URDF

ROOT_LINK = 'base_link'
TOOL_LINK = 'gripper_frame_link'          # URDF 自带工具参考系
DEFAULT_URDF = '/home/nkzwchen/pi_project/so101_ros2/so101_description/urdf/so101_new_calib.urdf'


# ----------------------------------------------------------------------------
# 基础旋转工具
# ----------------------------------------------------------------------------
def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF 固定轴 X-Y-Z（外旋）：R = Rz(y)·Ry(p)·Rx(r)。"""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_rot(axis: Sequence[float], theta: float) -> np.ndarray:
    """绕任意轴旋转 theta（Rodrigues）。axis 无需归一化。"""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _rot_error(R_cur: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """姿态误差的旋转向量近似：0.5·Σ (a_i × a_i^ref)。"""
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(R_cur[:, i], R_ref[:, i])
    return 0.5 * e


# ----------------------------------------------------------------------------
class SO101Kin:
    """SO101 运动学。q5 = 5 个臂关节角（弧度），顺序 = root→tool 的链顺序。"""

    def __init__(self, urdf_path: str = DEFAULT_URDF, tcp_offset: float = 0.0,
                 tube_axis: Sequence[float] = (1.0, 0.0, 0.0)):
        self.urdf_path = urdf_path
        self.tcp_offset = float(tcp_offset)     # 管尖相对工具系原点的伸出量（米）
        # ★ 管子沿工具系的哪个轴伸出。实测 = +X_tool（见模块头注释），
        #   不要照 URDF 推导改成 +Z —— 那与实机不符。
        _a = np.asarray(tube_axis, dtype=float)
        self.tube_axis = _a / np.linalg.norm(_a)

        robot = URDF.from_xml_string(open(urdf_path, 'rb').read())

        # ---- 取 root→tool 的关节链 ----
        # ★ 必须从 TOOL_LINK 往回走（用 child→joint 映射），不能用 parent→joint：
        #   gripper_link 有**两个**子关节（gripper 和 gripper_frame_joint），
        #   用 parent→joint 建字典会被覆盖，链就走错了。
        by_child = {j.child: j for j in robot.joints}
        chain_joints = []
        cur = TOOL_LINK
        while cur != ROOT_LINK:
            if cur not in by_child:
                raise RuntimeError(f'URDF 中找不到通向 {ROOT_LINK} 的链，断在 {cur}')
            j = by_child[cur]
            chain_joints.append(j)
            cur = j.parent
        chain_joints.reverse()

        self.chain_joints = chain_joints
        self.arm_joints = [j for j in chain_joints if j.type == 'revolute']
        self.arm_joint_names = [j.name for j in self.arm_joints]
        self.n = len(self.arm_joints)
        if self.n != 5:
            raise RuntimeError(f'预期 5 个臂关节，实际 {self.n}: {self.arm_joint_names}')

        self.q_low = np.array([j.limit.lower for j in self.arm_joints], dtype=float)
        self.q_high = np.array([j.limit.upper for j in self.arm_joints], dtype=float)

        # 把每段预解析成 (xyz, rpy, axis)；固定关节 axis=None
        self.segments: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
        for j in chain_joints:
            xyz = list(j.origin.xyz) if (j.origin and j.origin.xyz) else [0.0, 0.0, 0.0]
            rpy = list(j.origin.rpy) if (j.origin and j.origin.rpy) else [0.0, 0.0, 0.0]
            if j.type == 'revolute':
                axis = np.asarray(j.axis if j.axis else [0.0, 0.0, 1.0], dtype=float)
                axis = axis / np.linalg.norm(axis)
            else:
                axis = None
            self.segments.append((np.asarray(xyz, float), np.asarray(rpy, float), axis))

    # ------------------------------------------------------------- 前向遍历
    def _forward(self, q: Sequence[float]):
        """一次前向遍历。返回 (T_tip(4x4), axes(基座系各轴), origins(基座系各关节原点))。

        关键顺序：先把父→关节系的变换累乘进 T，**此时**记录关节轴与原点
        （因为转轴固定在父系里、且穿过关节原点），**然后**才施加关节自身的旋转。
        """
        T = np.eye(4)
        axes, origins = [], []
        qi = 0
        for xyz, rpy, axis in self.segments:
            Tj = np.eye(4)
            Tj[:3, :3] = _rpy_to_R(*rpy)
            Tj[:3, 3] = xyz
            T = T @ Tj
            if axis is not None:
                axes.append(T[:3, :3] @ axis)           # 轴方向（基座系）
                origins.append(T[:3, 3].copy())         # 轴上一点（基座系）
                Rj = np.eye(4)
                Rj[:3, :3] = _axis_rot(axis, float(q[qi]))
                T = T @ Rj
                qi += 1
        # 试管尖端：沿工具系的 tube_axis（实测为 +X）伸出
        Tt = np.eye(4)
        Tt[:3, 3] = self.tube_axis * self.tcp_offset
        T = T @ Tt
        return T, axes, origins

    # ------------------------------------------------------------------ FK
    def fk(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """正运动学，返回 (p_tip(3,), R(3,3))，均在 base 系。"""
        T, _, _ = self._forward(q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def fk_mat(self, q: Sequence[float]) -> np.ndarray:
        """正运动学，返回 4x4 齐次矩阵。"""
        return self._forward(q)[0]

    # ------------------------------------------------------------- Jacobian
    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        """6x5 几何雅可比（前 3 行线速度、后 3 行角速度），表达在 base 系。"""
        T, axes, origins = self._forward(q)
        p_tip = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, (z, p) in enumerate(zip(axes, origins)):
            J[:3, i] = np.cross(z, p_tip - p)
            J[3:, i] = z
        return J

    def jacobian_numeric(self, q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """数值微分求雅可比 —— 独立校验 jacobian() 用。"""
        q = np.asarray(q, dtype=float)
        R0 = self.fk_mat(q)[:3, :3]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            dq = np.zeros(self.n)
            dq[i] = eps
            Tp = self.fk_mat(q + dq)
            Tm = self.fk_mat(q - dq)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
            dR = (Tp[:3, :3] - Tm[:3, :3]) / (2 * eps)
            W = dR @ R0.T                      # 反对称矩阵，ω 由其非对角元取出
            J[3:, i] = [W[2, 1], W[0, 2], W[1, 0]]
        return J

    # ------------------------------------------------------------------ IK
    def tube_dir_error(self, R_cur: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
        """管子**方向**对齐误差（旋转向量，模 ≈ sin(夹角)）。

        ★ 为什么不用完整 3D 姿态误差：
          试管是**轴对称**的，绕自身轴自转对任务毫无影响。
          所以真正的姿态任务只有 **2 个自由度**（把方向对齐），
          而完整的 3D 旋转误差是 3 个约束 —— 3(位置)+3(姿态)=6 压在 5 个
          关节上，是**自己给自己造超定**。

          改成方向对齐后：3(位置) + 2(方向) = 5 = 自由度数，恰好方阵，
          位置和方向可以同时精确满足（绕管自转留给零空间自由处理）。

        误差向量取 d_cur × d_ref：它同时垂直于两者，
        正是"把 d_cur 转到 d_ref"所需的旋转轴，且**不含 d_cur 分量**，
        因此不会去驱动绕管自转。
        """
        d_cur = R_cur @ self.tube_axis
        d_ref = R_ref @ self.tube_axis
        return np.cross(d_cur, d_ref)

    def ik_step(self, q, p_target, R_ref, q_posture=None, orient_weight: float = 0.5,
                damping: float = 0.05, max_dq: float = 0.15, null_gain: float = 0.3,
                locked_idx=None) -> np.ndarray:
        """一步阻尼最小二乘微分 IK。

        位置是主任务（权重 1），管子**方向对齐**是次任务（orient_weight）。

        locked_idx:
            要**锁死不动**的关节下标（如冻结的 shoulder_pan / wrist_roll）。
            实现上做两件事，缺一不可：
              ① 把 J 中这些列清零 → 任务步 dq 的对应分量为 0
              ② 最后把 dq 中这些分量**再显式清零** → 防止零空间项把它们推走
            为什么 ② 必需：J 的锁定列为 0 时，
                N = I − pinv(Jw)·Jw  在锁定列上恰好是 e_i，
            于是 `N @ (q_posture − q)` 在锁定关节上正好等于 `q_posture − q`
            —— 也就是说**零空间项会照样把它们往 q_posture 拉**。
            只做 ① 的话，锁定关节会缓慢漂移，而且看起来"莫名其妙"。

        ═══ 两个实测教训（2026-09），都极具迷惑性 ═══
        ① 姿态任务必须只约束**方向**（2 自由度），不能用完整 3D 旋转误差。
           见 tube_dir_error() 的说明。用完整 3D 误差时系统超定，
           最小二乘的妥协方式是**牺牲位置精度**：
               orient_weight=0.2（用 3D 误差）→ 位置误差 6.5mm，扫描 85% 不收敛
        ② 权重也不能过大。即使改成方向对齐，权重给太大仍会挤占位置精度。
           实测（同一可达目标，迭代上限 2000）：
               权重 0.000 → 0.028mm / 4 次
               权重 0.005 → 0.033mm / 4 次
               权重 0.020 → 0.099mm / 5 次
               权重 0.050 → 0.507mm / 2000 次  ⚠️ 收敛变慢
               权重 0.200 → 6.549mm / 2000 次  ❌ 稳态残差

        保持构型不跳分支主要靠零空间 q_posture 项，不靠这个权重。
        """
        q = np.asarray(q, dtype=float)
        p_cur, R_cur = self.fk(q)
        J = self.jacobian(q)
        if locked_idx is not None and len(locked_idx):
            J = J.copy()
            J[:, list(locked_idx)] = 0.0          # ① 锁列

        e = np.concatenate([np.asarray(p_target, float) - p_cur,
                            orient_weight * self.tube_dir_error(R_cur, R_ref)])
        Jw = np.vstack([J[:3, :], orient_weight * J[3:, :]])

        A = Jw @ Jw.T + (damping ** 2) * np.eye(6)
        Ainv = np.linalg.inv(A)
        dq = Jw.T @ (Ainv @ e)

        if q_posture is not None and null_gain > 0:
            # ★★ 零空间投影必须用**精确伪逆**，不能用上面那个带阻尼的 Ainv ★★
            #   `N = I - Jw^T Ainv Jw` 中 Ainv 含阻尼项 λ²I，所以 N **不是真正的
            #   零空间投影**，姿态项会"漏"进任务空间，形成**稳态残差**。
            #   症状极具迷惑性：条件数只有 8.8、限位余量 1.17 rad 都很正常，
            #   但 IK 要跑几千次迭代还停在 1.39mm，真机表现为
            #   **越靠近工作空间边缘，policy 越"够不着"目标点**。
            #
            #   实测（目标 = 起点 + (2.5,-2.5,0)cm，tol=1e-5，迭代上限 5000）：
            #       带阻尼投影 null_gain=0.30 → 残差 1.391mm，5000 次未收敛
            #       带阻尼投影 null_gain=0.15 → 残差 0.723mm
            #       带阻尼投影 null_gain=0.05 → 残差 0.248mm
            #       **精确伪逆** null_gain=0.30 → 残差 0.008mm，16 次收敛 ✅
            #
            #   注意：任务步 dq 仍用带阻尼的 Ainv（数值稳健），
            #         只有**投影**改用精确伪逆，两者分工不同。
            Jpinv = np.linalg.pinv(Jw)
            N = np.eye(self.n) - Jpinv @ Jw
            dq = dq + N @ (null_gain * (np.asarray(q_posture, float) - q))

        if locked_idx is not None and len(locked_idx):
            dq[list(locked_idx)] = 0.0            # ② 零空间项也不许动它们
        dq = np.clip(dq, -max_dq, max_dq)
        return np.clip(q + dq, self.q_low, self.q_high)

    def ik_solve(self, q0, p_target, R_ref=None, q_posture=None, tol: float = 1e-3,
                 max_iter: int = 200, **kw) -> Tuple[np.ndarray, dict]:
        """迭代到收敛。返回 (q, info)，info['converged'] 表示位置误差是否达标。

        ⚠️ 不可达时必须**安全失败**：返回 converged=False，绝不返回 NaN。
        """
        q = np.asarray(q0, dtype=float).copy()
        if R_ref is None:
            _, R_ref = self.fk(q)
        if q_posture is None:
            q_posture = q.copy()

        p_target = np.asarray(p_target, float)
        best_q, best_err, it = q.copy(), np.inf, 0
        for it in range(max_iter):
            p, _ = self.fk(q)
            err = float(np.linalg.norm(p - p_target))
            if err < best_err:
                best_err, best_q = err, q.copy()
            if err < tol:
                break
            q = self.ik_step(q, p_target, R_ref, q_posture, **kw)

        p_fin, _ = self.fk(best_q)
        final_err = float(np.linalg.norm(p_fin - p_target))
        return best_q, {
            'converged': final_err < tol,
            'err': final_err,
            'iters': it + 1,
            'q': best_q,
        }

    # -------------------------------------------------------------- 诊断
    def diagnose(self, q) -> dict:
        """给定构型，返回 IK 良态性指标。V3 工作空间扫描用。"""
        q = np.asarray(q, dtype=float)
        p, R = self.fk(q)
        Jp = self.jacobian(q)[:3, :]                    # 位置雅可比 3x5
        sv = np.linalg.svd(Jp, compute_uv=False)
        margin = np.minimum(q - self.q_low, self.q_high - q)
        tube_dir = R @ self.tube_axis               # 管子轴向在 base 系的方向
        # 与"竖直向下"(0,0,-1) 的夹角：0° = 正朝下插，90° = 水平
        tube_angle = float(np.degrees(np.arccos(np.clip(-tube_dir[2], -1.0, 1.0))))
        return {
            'p': p,
            'R': R,
            'cond': float(sv[0] / max(sv[-1], 1e-12)),
            'sigma_min': float(sv[-1]),
            'margin_min': float(margin.min()),
            'margin': margin,
            'tool_z': R[:, 2].copy(),               # 保留：工具系 Z 轴（非管子方向）
            'tube_dir': tube_dir,                   # ★ 管子实际轴向
            'tube_angle': tube_angle,               # ★ 与竖直向下夹角（度）
        }
