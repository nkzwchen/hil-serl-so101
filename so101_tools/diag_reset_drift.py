#!/usr/bin/env python3
"""诊断：reset 之后从臂为什么在**累积下沉**。

只做观测，不改任何控制逻辑。分三段量：
    A. reset 前（上一轮结束后的位置）
    B. reset 的斜坡刚走完（dur 之后）
    C. 再等 3 秒（看它还在不在动 → 是"没走完"还是"静差/蠕变"）

如果 A→A 逐轮下沉，说明是"每轮结束的位置比上一轮低"；如果 B→C 还在下沉，
说明是舵机在指令不变的情况下继续下垂（静差/蠕变）。

用法：
    python diag_reset_drift.py [--n 4] [--randomize 0|1]
"""

import argparse
import math
import sys
import time

import numpy as np
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
from so101_home_pose import ARM_JOINTS, N_ARM, START_JOINT_POSE  # noqa: E402
from so101_kin import SO101Kin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--randomize', type=int, default=0,
                    help='0 = 关掉随机化（隔离变量），1 = 开着')
    ap.add_argument('--settle', type=float, default=3.0)
    args = ap.parse_args()

    from experiments.so101.env import SO101Env

    kin = SO101Kin(tcp_offset=0.0)
    q_nom = START_JOINT_POSE[:N_ARM]
    p_nom = kin.fk(q_nom)[0]

    env = SO101Env(hz=10.0, action_scale=0.05, max_episode_steps=100,
                   randomize_start=bool(args.randomize))
    try:
        # 等第一帧关节数据（刚构造完还没订阅到）
        t0 = time.time()
        while env.joint_pos is None and time.time() - t0 < 8.0:
            time.sleep(0.05)
        if env.joint_pos is None:
            print('❌ 拿不到 /follower/joint_states')
            return 2

        def snap(tag):
            q = env.joint_pos[:N_ARM].copy()
            p = kin.fk(q)[0]
            return q, p, float(p[2])

        print('=' * 84)
        print(f'reset 下沉诊断   randomize={args.randomize}   settle={args.settle}s')
        print('=' * 84)
        print(f'  名义起点命令 = {np.round(q_nom, 4)}')
        print(f'  名义起点末端 z = {p_nom[2]*100:.2f}cm')
        print()
        hdr = f"  {'轮':>2} {'阶段':<10}" + ''.join(f'{n[:9]:>10}' for n in ARM_JOINTS[:5]) + f"{'z(cm)':>9}"
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))

        prev_end, prev_end_z = None, 0.0
        for i in range(args.n):
            qA, pA, zA = snap('A')
            print(f"  {i:>2} {'A reset前':<10}"
                  + ''.join(f'{math.degrees(v):>10.2f}' for v in qA) + f'{zA*100:>9.2f}')
            if prev_end is not None:
                d = (qA - prev_end)
                print(f"  {i:>2} {'  (比上轮末)':<10}"
                      + ''.join(f'{math.degrees(v):>+10.2f}' for v in d)
                      + f'{(zA-prev_end_z)*100:>+9.2f}')

            env.reset()
            qB, pB, zB = snap('B')
            print(f"  {i:>2} {'B 斜坡后':<10}"
                  + ''.join(f'{math.degrees(v):>10.2f}' for v in qB) + f'{zB*100:>9.2f}')

            time.sleep(args.settle)
            qC, pC, zC = snap('C')
            print(f"  {i:>2} {f'C +{args.settle:g}s后':<10}"
                  + ''.join(f'{math.degrees(v):>10.2f}' for v in qC) + f'{zC*100:>9.2f}')
            print(f"  {i:>2} {'  B→C 变化':<10}"
                  + ''.join(f'{math.degrees(v):>+10.2f}' for v in (qC - qB))
                  + f'{(zC-zB)*100:>+9.2f}')
            print()
            prev_end, prev_end_z = qC.copy(), zC

        qE, _, zE = snap('end')
        print(f'  末态相对名义起点：z 差 = {(zE-p_nom[2])*100:+.2f}cm，'
              f'逐关节差 = ' + ', '.join(
                  f'{n}={math.degrees(v):+.2f}°' for n, v in zip(ARM_JOINTS[:5], qE - q_nom)))
    finally:
        try:
            env.close()
        except Exception as e:
            print('  close 警告:', e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
