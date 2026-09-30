#!/usr/bin/env python3
"""V1-c：把自实现的 FK 与**官方 ROS 管线**（robot_state_publisher + TF）对账。

为什么要这一步：
    雅可比"解析 vs 数值微分"一致，只能证明**导数自洽**，
    不能证明 FK 绝对正确 —— 一个系统性写错的 FK 也能有自洽的导数。
    所以必须拿官方 URDF→TF 管线当独立参照。

⚠️ 踩过的坑：不要用 `ros2 topic pub` 发 joint_states。
   它发出的 header.stamp = 0，robot_state_publisher 会拿这个当 TF 时间戳，
   tf2 视为远古数据直接丢弃，表现为"joint_states 明明在流、/tf 却是空的"。
   必须自己打时间戳。

用法（需先另起 robot_state_publisher）：
    ros2 launch so101_description rsp.launch.py \
        model:=<.../so101_new_calib.urdf.xacro> type:=follower mode:=real
    python phase0_fk_tf_check.py
"""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
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

NS = 'follower'
BASE = f'{NS}/base_link'
TOOL = f'{NS}/gripper_frame_link'
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

# 测试构型：故意用非零值。全零有可能"碰巧对上"，掩盖符号/顺序错误。
TEST_Q = [
    np.array([0.3, 0.5, -0.6, 0.4, 0.2]),
    np.array([-0.8, 0.2, 0.9, -0.3, 1.5]),
    np.zeros(5),
]


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    rclpy.init()
    node = Node('fk_tf_check')
    pub = node.create_publisher(JointState, f'/{NS}/joint_states', 10)
    buf = Buffer()
    TransformListener(buf, node)

    kin = SO101Kin(tcp_offset=0.0)

    # ---- warm-up：先等 TF 树建起来 ----
    # 否则第一个构型会因为 "source_frame does not exist" 而假失败。
    warm = JointState()
    warm.name = JOINTS
    warm.position = [0.0] * 6
    for _ in range(120):                          # 最多约 6 秒
        warm.header.stamp = node.get_clock().now().to_msg()
        pub.publish(warm)
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            buf.lookup_transform(BASE, TOOL, rclpy.time.Time())
            break
        except Exception:
            continue
    else:
        print('❌ warm-up 超时：TF 树没建起来。rsp 在跑吗？帧名对吗？')
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f'TF 参照: {BASE} -> {TOOL}')
    print()
    worst_p = worst_R = 0.0
    for qi, q in enumerate(TEST_Q):
        msg = JointState()
        msg.name = JOINTS
        msg.position = [float(v) for v in q] + [0.0]

        # ★ 两阶段查询，避免"读到上一个构型缓存"的假失败/假通过：
        #   阶段1：持续发 1.5s（每帧都打新时间戳）并 spin，
        #          确保 tf buffer 已被**本构型**覆盖；
        #   阶段2：取最新，并**校验时间戳新鲜度**——
        #          若 TF 时间戳早于本阶段起点，说明读到的是旧值，判为失败。
        #   （用 Time() 直接查最新、或用精确 stamp 查，都踩过坑：
        #     前者读到旧构型；后者在 listener 尚未收到时超时。）
        t_begin = node.get_clock().now()
        for _ in range(30):
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)

        got = None
        try:
            got = buf.lookup_transform(BASE, TOOL, rclpy.time.Time())
        except Exception as exc:
            print(f'  构型 {qi}: ❌ 查询失败 {exc}')

        if got is not None:
            t_tf = rclpy.time.Time.from_msg(got.header.stamp)
            if t_tf < t_begin:
                print(f'  构型 {qi}: ❌ 读到过期 TF（时间戳早于本次发布），判为失败')
                got = None

        if got is None:
            print(f'  构型 {qi}: ❌ 拿不到 TF（rsp 是否在跑？）')
            continue

        t = got.transform.translation
        r = got.transform.rotation
        p_tf = np.array([t.x, t.y, t.z])
        R_tf = quat_to_R(r.x, r.y, r.z, r.w)
        p_my, R_my = kin.fk(q)

        ep = float(np.abs(p_tf - p_my).max())
        eR = float(np.abs(R_tf - R_my).max())
        worst_p, worst_R = max(worst_p, ep), max(worst_R, eR)
        print(f'  构型 {qi}  q={np.round(q,2)}')
        print(f'    TF     p={np.round(p_tf,6)}')
        print(f'    自实现 p={np.round(p_my,6)}   位置最大差={ep:.2e}  姿态最大差={eR:.2e}')

    print()
    print(f'  最坏: 位置 {worst_p:.3e} m, 姿态 {worst_R:.3e}')
    ok = worst_p < 1e-6 and worst_R < 1e-6
    print('  判定:', '✅ 通过 —— 自实现 FK 与官方管线一致' if ok else '❌ 失败 —— FK 与官方管线不符')
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
