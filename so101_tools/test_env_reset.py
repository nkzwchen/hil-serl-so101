#!/usr/bin/env python3
"""真机验证 env.reset()：起始位置随机化 + 平滑斜坡。

会真的动从臂（水平弧 ±1cm），请确认周围无障碍。

量三件事：
  ① 随机化是否真的生效（每次起点不同、且落在 ±1cm 水平弧内）
  ② **实际达到**的偏移 vs **请求**的偏移 —— 舵机是比例位置环无积分项，
     有重力静差，所以实际位置会比指令低几度，从而让实际偏移与请求偏移有差别。
     这个差别必须量出来，否则"随机化 ±1cm"只是指令上的说法。
  ③ 被冻结的关节（wrist_roll / gripper）有没有被碰到

用法：
    python test_env_reset.py [--n 4] [--seed 0] [--half 0.01]
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
from so101_home_pose import (  # noqa: E402
    FROZEN_IDX, N_ARM, RANDOMIZE_HALF_M, START_JOINT_POSE,
)
from so101_kin import SO101Kin  # noqa: E402
from record_demo_so101 import START_INFO_KEYS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=4, help='reset 次数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--half', type=float, default=RANDOMIZE_HALF_M,
                    help='水平弧半长（米），默认跟 so101_home_pose 一致')
    args = ap.parse_args()

    from experiments.so101.env import SO101Env

    kin = SO101Kin(tcp_offset=0.0)
    q_nom = START_JOINT_POSE[:N_ARM]
    p_nom = kin.fk(q_nom)[0]

    env = SO101Env(hz=10.0, action_scale=0.05, max_episode_steps=100,
                   randomize_start=True, randomize_half_m=args.half)
    rows = []
    try:
        print('=' * 78)
        print(f'env.reset() 真机验证   n={args.n}  seed={args.seed}  '
              f'half=±{args.half*100:.1f}cm')
        print('=' * 78)
        print(f'  名义起点末端 = [{p_nom[0]:+.4f}, {p_nom[1]:+.4f}, {p_nom[2]:+.4f}]')
        print()
        print(f"  {'#':>2}  {'请求(dx,dy,dz)cm':>24}  {'实际(dx,dy,dz)cm':>24}  "
              f"{'位置差mm':>9}  {'管子倾角':>9}  {'冻结漂移':>9}")
        print('  ' + '-' * 74)

        for i in range(args.n):
            obs, info = env.reset(seed=(args.seed + i if args.seed is not None else None))
            # 再等一会儿让舵机静差稳定（斜坡之后还有个沉降）
            time.sleep(0.8)
            q_act = env.joint_pos[:N_ARM].copy()
            p_act = kin.fk(q_act)[0]

            st = info.get('start', {})
            # ★ 契约核对：录制器就是读这些键（全用 .get()，键名错了会**静默**把
            #   元数据写成 0）。真机 env 必须真的提供它们 —— 桩环境满足不算数。
            missing = [k for k in START_INFO_KEYS if k not in st]
            assert not missing, (
                f'真机 env 的 start_info 缺 {missing}；实际键 = {sorted(st)}。'
                f' 录制器的元数据会静默变成 0。')
            if st.get('randomized'):
                req = (st['dx_m'] * 100, st['dy_m'] * 100, st['dz_m'] * 100)
            else:
                req = (0.0, 0.0, 0.0)
            # 实际偏移 = 实际末端位置 − 名义起点（三维）
            d = (p_act - p_nom) * 100
            act = (float(d[0]), float(d[1]), float(d[2]))
            # 位置差：请求的目标点 vs 实际末端点
            p_req = p_nom + np.asarray(req, float) / 100.0
            gap = float(np.linalg.norm(p_act - p_req)) * 1000
            tilt = kin.diagnose(q_act)['tube_angle']
            fz = max(abs(q_act[j] - q_nom[j]) for j in FROZEN_IDX if j < N_ARM)
            # 冻结关节的"漂移"是 P-only 位置环的重力静差：每次 reset 都该是同
            # 一个常数值（随机化根本没碰它们）。真要查的是它**逐次是否一致**。
            rows.append((req, act, gap, tilt, math.degrees(fz)))
            print(f"  {i:>2}  ({req[0]:+6.2f},{req[1]:+6.2f},{req[2]:+6.2f})  "
                  f"({act[0]:+6.2f},{act[1]:+6.2f},{act[2]:+6.2f})  {gap:>9.2f}  "
                  f"{tilt:>8.2f}°  {math.degrees(fz):>8.3f}°")

        print()
        reqs = np.array([r[0] for r in rows])
        acts = np.array([r[1] for r in rows])
        gaps = np.array([r[2] for r in rows])
        print('  ── summary ──')
        for i, ax in enumerate('xyz'):
            print(f'  d{ax}: requested [{reqs[:,i].min():+.2f},{reqs[:,i].max():+.2f}]cm  '
                  f'achieved [{acts[:,i].min():+.2f},{acts[:,i].max():+.2f}]cm')
        # 随机化是 pan-only，所以"水平弧"是应有的形状：水平位移有序、
        # 竖直位移只是重力静差（不是随机化的方向分量）。
        rh = np.linalg.norm(reqs[:, :2], axis=1)      # reqs 已经是 cm
        ah = np.linalg.norm(acts[:, :2], axis=1)
        print(f'  水平位移: requested {rh.min():.2f}~{rh.max():.2f}cm  '
              f'achieved {ah.min():.2f}~{ah.max():.2f}cm  (弧长上限 {args.half*100:.1f})')
        print(f'  竖直位移: achieved {acts[:,2].min():+.2f}~{acts[:,2].max():+.2f}cm  '
              f'(应为重力静差量级，不是随机化)')
        print(f'  请求→实际 位置偏差 = 均值 {gaps.mean():.2f}mm  最大 {gaps.max():.2f}mm')
        tilts = [r[3] for r in rows]
        print(f'  管子距竖直 = {min(tilts):.2f}° ~ {max(tilts):.2f}°  '
              f'(跨度 {max(tilts)-min(tilts):.2f}°，来自重力静差)')
        fzs = [r[4] for r in rows]
        print(f'  冻结关节 idx {[j for j in FROZEN_IDX if j < N_ARM]}: '
              f'漂移 {min(fzs):.3f}~{max(fzs):.3f}°  '
              f'（重力静差常数；**逐次跨度 {max(fzs)-min(fzs):.3f}° 才说明没被碰**）')
        print()
        ok = (gaps.max() < 15.0 and ah.max() <= args.half*100 + 0.5
              and (max(fzs) - min(fzs)) < 0.05)
        print(f'  {"✅ reset 工作正常" if ok else "⚠️ 见上面的偏差数字"}')
        print('  说明：请求→实际的偏差主要是**舵机重力静差**（比例位置环无积分项）。'
              '它不影响训练 ——')
        print('        policy 的观测是**实测**关节角，静差只是让起点整体略偏，'
              '随机化的散布照旧。')
    finally:
        try:
            env.close()
        except Exception as e:
            print('  close 警告:', e)
    return 0


if __name__ == '__main__':
    sys.exit(main())
