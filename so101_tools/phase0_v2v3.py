#!/usr/bin/env python3
"""Phase 0 的 V2（IK 精度）+ V3（工作空间扫描）。

V2：验证微分 IK 的精度与安全性
    · 单步 0.5cm  —— 等于 ACTION_SCALE，policy 每步的真实位移
    · 连续 10 步累积 5cm —— 模拟 policy 走直线，检查累积漂移
    · 一次跳 5cm   —— 压力测试
    · 超出臂展     —— 必须**安全失败**，不能返回 NaN 或乱解

V3：以候选起点为中心扫描工作空间，四项判据
    · IK 收敛精度 < 1mm
    · 关节限位余量 > 0.1 rad
    · 位置雅可比条件数 < COND_MAX（远离奇异）
    · **分支连续性** —— 相邻网格点的 |Δq| 要小

★ 为什么第 4 项最重要：
    SO101 只有 5 个臂关节，控制 3 维位置 ⇒ 有 **2 个冗余自由度**，
    IK 的零空间是 2 维。若零空间正则没做好，相邻目标点可能解到**完全不同的
    构型**上，真机表现为"扫过某个区域时机械臂突然抽一下"。
    这类问题在真机上极难定位，但在离线扫描里一眼可见。

用法：
    python phase0_v2v3.py
"""

import sys

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
from so101_kin import SO101Kin  # noqa: E402

# ---- 候选起点（leader 遥操作摆出，采样 20s/取最后 10s 均值，std<0.004 rad）----
Q_START = np.array([0.330903, 0.283856, 0.086691, -0.342928, 0.057538])

# ---- 设计参数 ----
ACTION_SCALE = 0.005      # m/步（10Hz → 5cm/s）
BOX_HALF = 0.025          # ±2.5cm ⇒ 5cm×5cm 随机框
DEPTH = 0.080             # 向下插入深度 8cm
GRID = 9                  # 每轴格点数（9^3 = 729）

# ---- 判据 ----
TOL = 1e-3                # IK 位置误差 < 1mm
MARGIN_MIN = 0.10         # 关节限位余量 > 0.1 rad
COND_MAX = 50.0           # 位置雅可比条件数上限
DQ_MAX = 0.30             # 相邻网格点 |Δq| 上限（分支连续性）
TUBE_ANGLE_MAX = 10.0     # 管子相对竖直向下的最大偏角（度）—— 插入对准所需


def v2(kin):
    print('=' * 74)
    print('V2  IK 精度与安全性')
    print('=' * 74)
    p0, R0 = kin.fk(Q_START)
    print(f'  起点工具系原点 = [{p0[0]:+.4f}, {p0[1]:+.4f}, {p0[2]:+.4f}] m')
    d0 = kin.diagnose(Q_START)
    print(f'  起点管子轴向   = [{d0["tube_dir"][0]:+.3f}, {d0["tube_dir"][1]:+.3f}, '
          f'{d0["tube_dir"][2]:+.3f}]  → 距竖直 {d0["tube_angle"]:.1f}°')
    print()

    ok = True

    # ① 单步（= ACTION_SCALE）
    q = Q_START.copy()
    tgt = p0 + np.array([0.0, 0.0, -ACTION_SCALE])
    q, info = kin.ik_solve(q, tgt, R_ref=R0, q_posture=Q_START, tol=TOL,
                           max_iter=400, damping=0.02)
    print(f"① 单步 {ACTION_SCALE*1000:.1f}mm : 误差={info['err']*1000:.4f} mm  "
          f"迭代={info['iters']}  {'✅' if info['converged'] else '❌'}")
    ok &= info['converged']

    # ② 连续 10 步累积 5cm（模拟 policy 走直线）
    q = Q_START.copy()
    worst = 0.0
    for k in range(1, 11):
        tgt = p0 + np.array([0.0, 0.0, -0.05 * k / 10])
        q, info = kin.ik_solve(q, tgt, R_ref=R0, q_posture=Q_START, tol=TOL,
                               max_iter=400, damping=0.02)
        worst = max(worst, info['err'])
    print(f'② 连续 10 步累积 50mm : 最差单步误差={worst*1000:.4f} mm  '
          f"{'✅' if worst < TOL else '❌'}")
    ok &= worst < TOL

    # ③ 一次跳 5cm
    tgt = p0 + np.array([0.0, 0.0, -0.05])
    q, info = kin.ik_solve(Q_START, tgt, R_ref=R0, q_posture=Q_START, tol=TOL,
                           max_iter=800, damping=0.02)
    print(f"③ 一次跳 50mm        : 误差={info['err']*1000:.4f} mm  迭代={info['iters']}  "
          f"{'✅' if info['converged'] else '❌'}")
    ok &= info['converged']

    # ④ 不可达目标 —— 必须安全失败
    far = p0 + np.array([1.0, 0.0, 0.0])          # 前方 1m，远超臂展
    q, info = kin.ik_solve(Q_START, far, R_ref=R0, q_posture=Q_START, tol=TOL,
                           max_iter=400, damping=0.02)
    safe = (not info['converged']) and np.isfinite(info['q']).all()
    print(f"④ 不可达目标(前方1m)  : converged={info['converged']}  "
          f"解有限={np.isfinite(info['q']).all()}  "
          f"{'✅ 安全失败' if safe else '❌ 危险：未安全失败'}")
    ok &= safe

    print()
    print(f"V2 判定：{'✅ 全部通过' if ok else '❌ 存在失败项'}")
    return ok


def v3(kin):
    print()
    print('=' * 74)
    print('V3  工作空间扫描')
    print('=' * 74)
    p0, R0 = kin.fk(Q_START)

    xs = np.linspace(-BOX_HALF, BOX_HALF, GRID)
    ys = np.linspace(-BOX_HALF, BOX_HALF, GRID)
    zs = np.linspace(0.0, -DEPTH, GRID)          # 从起点向下插
    print(f'  体积: x,y ∈ ±{BOX_HALF*100:.1f}cm, z ∈ [0, -{DEPTH*100:.0f}]cm'
          f'  共 {GRID**3} 点')

    n_ok = 0
    fails = {'不收敛': 0, '限位余量': 0, '条件数': 0, '管子倾斜': 0, '分支跳变': 0}
    conds, margins, angles = [], [], []
    q_prev = Q_START.copy()
    prev_grid = None
    max_dq = 0.0
    worst_err = 0.0

    # 蛇形遍历，保证相邻点空间上相邻（用于分支连续性判定）
    for iz, dz in enumerate(zs):
        for iy in range(GRID):
            yy = ys[iy] if iz % 2 == 0 else ys[GRID - 1 - iy]   # 蛇形
            for ix in range(GRID):
                xx = xs[ix] if (iy + iz) % 2 == 0 else xs[GRID - 1 - ix]
                tgt = p0 + np.array([xx, yy, dz])
                # 从**上一个点的解**出发 —— 模拟真实控制环的连续跟踪
                q, info = kin.ik_solve(q_prev, tgt, R_ref=R0, q_posture=Q_START,
                                       tol=TOL, max_iter=400, damping=0.02)
                worst_err = max(worst_err, info['err'])
                d = kin.diagnose(info['q'])
                conds.append(d['cond'])
                margins.append(d['margin_min'])
                angles.append(d['tube_angle'])

                bad = None
                if not info['converged']:
                    bad = '不收敛'
                elif d['margin_min'] < MARGIN_MIN:
                    bad = '限位余量'
                elif d['cond'] > COND_MAX:
                    bad = '条件数'
                elif d['tube_angle'] > TUBE_ANGLE_MAX:
                    bad = '管子倾斜'
                elif prev_grid is not None:
                    dq = float(np.abs(info['q'] - q_prev).max())
                    max_dq = max(max_dq, dq)
                    if dq > DQ_MAX:
                        bad = '分支跳变'
                if bad:
                    fails[bad] += 1
                else:
                    n_ok += 1
                q_prev = info['q']
                prev_grid = (xx, yy, dz)

    total = GRID ** 3
    print()
    print(f'  通过            : {n_ok}/{total}  ({100.0*n_ok/total:.1f}%)')
    print(f'  最差 IK 误差    : {worst_err*1000:.4f} mm')
    print(f'  最大相邻 |Δq|   : {max_dq:.4f} rad   (上限 {DQ_MAX})')
    print(f'  条件数          : 中位 {np.median(conds):.1f} / 最大 {max(conds):.1f}  (上限 {COND_MAX})')
    print(f'  最小限位余量    : {min(margins):.4f} rad  (下限 {MARGIN_MIN})')
    print(f'  管子距竖直向下  : 中位 {np.median(angles):.2f}° / 最大 {max(angles):.2f}°  '
          f'(上限 {TUBE_ANGLE_MAX}°)')
    if any(fails.values()):
        print()
        print('  失败分类：')
        for k, v in fails.items():
            if v:
                print(f'    {k:<10} {v} 点')
    print()
    ok = n_ok == total
    print(f"V3 判定：{'✅ 100% 通过' if ok else '❌ 存在失败区域，需调整起点或缩小随机框'}")
    return ok


def main():
    kin = SO101Kin(tcp_offset=0.0)      # 管尖伸出量未实测，先按工具系原点
    d = kin.diagnose(Q_START)
    print()
    print('候选起点诊断：')
    print(f"  工具系原点  = [{d['p'][0]:+.4f}, {d['p'][1]:+.4f}, {d['p'][2]:+.4f}] m")
    print(f"  管子轴向    = [{d['tube_dir'][0]:+.4f}, {d['tube_dir'][1]:+.4f}, "
          f"{d['tube_dir'][2]:+.4f}]  → 距竖直向下 {d['tube_angle']:.2f}°")
    print(f"  条件数      = {d['cond']:.2f}")
    print(f"  最小限位余量= {d['margin_min']:.4f} rad")
    print()
    r2 = v2(kin)
    r3 = v3(kin)
    print()
    print('=' * 74)
    print(f'Phase 0 结论：V2 {"✅" if r2 else "❌"}   V3 {"✅" if r3 else "❌"}')
    print('=' * 74)
    return 0 if (r2 and r3) else 1


if __name__ == '__main__':
    sys.exit(main())
