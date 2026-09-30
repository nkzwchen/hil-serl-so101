#!/usr/bin/env python3
"""Verify start-pose randomization: pan-only horizontal arc (no IK).

What changed and why
--------------------
The randomizer used to sample a 3D box and solve IK for each sample. On hardware
the arm sometimes started **much higher** than the nominal pose. Root cause: with
pan unfrozen the IK has a 1-D null space (rotate pan *and* unwind the arm to reach
the same tip), so solutions spanned, over 2000 samples:

    shoulder_lift 36.2 deg   elbow_flex 52.6 deg   wrist_flex 31.5 deg

while the tip position (+/-2.5cm) and tube tilt (0.05 deg) both looked correct.
The old check verified position and tilt but **not posture**, so it passed.

`randomize_start_pose` now rotates only `shoulder_pan`, so:

    tip moves on a HORIZONTAL CIRCLE  ->  dz == 0 exactly
    pan is about a vertical axis      ->  tube tilt unchanged exactly
    no IK at all                      ->  posture unchanged exactly (span == 0)

Note this trades a 2D box for one horizontal direction ("just slide it
horizontally") -- that is the user's explicit choice.

Criteria (all must hold)
------------------------
1. chord identity : |achieved horizontal displacement| == |requested| (< 1e-6 m)
2. height         : dz == 0 (< 1e-5 m)
3. tube tilt      : change from nominal == 0 (< 1e-4 deg)
4. posture        : shoulder_lift / elbow_flex / wrist_flex span == 0 exactly
5. locked joints  : wrist_roll, gripper do not move
6. joint limits   : margin > 0 everywhere

Check 1 is the one that catches a wrong radius: an earlier version measured the
radius to the vertical axis through the **base origin** (`hypot(x, y)` = 0.4123)
while the pan axis actually passes through `(x=+0.0388, y=0)`, giving a true
radius of 0.3754. The chord came out 1.098x too large, so the real displacement
was only 0.910x the request -- and a bounds-only check ("is it within +/-1cm?")
happily passes that. Compare request to achievement, not to a range.

Usage
-----
    python phase0_randomize_check.py
    python phase0_randomize_check.py --half 0.02 --n-random 5000
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
    CONTROLLED_JOINTS, FROZEN_IDX, FROZEN_JOINTS, N_ACTION, N_ARM,
    RANDOMIZE_HALF_M, START_JOINT_POSE, randomize_start_pose,
)
from so101_kin import SO101Kin  # noqa: E402

TOL_DZ_M = 1e-5         # 10 um; measured float noise is ~25 nm
TOL_CHORD_M = 1e-6      # 1 um; the closed form is exact to float precision
TOL_TILT_DEG = 1e-4     # measured float noise is ~4 u deg
TOL_JOINT_DEG = 1e-9    # posture must be bit-identical, not merely close


class _Fixed:
    """Stand-in for a numpy Generator whose `uniform` always returns one value.

    Lets section (2) drive the exact requested displacement, so achieved and
    requested can be compared 1:1 instead of only checked against a range.
    """

    def __init__(self, value):
        self.value = float(value)

    def uniform(self, lo, hi):      # noqa: ARG002
        return self.value


def pan_axis(kin, q):
    """Return (axis_dir, point_on_axis) of shoulder_pan in the base frame."""
    _, axes, origins = kin._forward(q)
    return np.asarray(axes[0], float), np.asarray(origins[0], float)


def perp_radius(kin, q, q_nom):
    """Distance from the tip to the pan axis line (the true swing radius)."""
    a, o = pan_axis(kin, q_nom)
    v = kin.fk(q)[0] - o
    return float(np.linalg.norm(v - np.dot(v, a) * a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--half', type=float, default=RANDOMIZE_HALF_M,
                    help='half range in metres (default 0.01 => +/-1cm arc)')
    ap.add_argument('--n-random', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    kin = SO101Kin(tcp_offset=0.0)
    q_nom = START_JOINT_POSE[:N_ARM].copy()
    p_nom = kin.fk(q_nom)[0]
    d_nom = kin.diagnose(q_nom)
    H = args.half
    fails = []

    def check(name, cond, detail=''):
        print(f'    [{"PASS" if cond else "FAIL"}] {name}{"  " + detail if detail else ""}')
        if not cond:
            fails.append(name)

    print('=' * 78)
    print('start-pose randomization check  (pan-only horizontal arc, no IK)')
    print('=' * 78)
    print(f'  action dim    : {N_ACTION}  -> {CONTROLLED_JOINTS}')
    print(f'  frozen joints : {FROZEN_JOINTS}   (idx {FROZEN_IDX})')
    print(f'  randomization : shoulder_pan only, +/-{H*100:.2f}cm horizontal arc')
    print(f'  nominal start : tip [{p_nom[0]:+.4f}, {p_nom[1]:+.4f}, {p_nom[2]:+.4f}]'
          f'  tube {d_nom["tube_angle"]:.3f} deg from vertical')

    # ------------------------------------------------------------------ (1)
    print()
    print('(1) pan axis geometry  (this is what the radius bug got wrong)')
    a0, o0 = pan_axis(kin, q_nom)
    tilt_deg = math.degrees(math.acos(min(1.0, abs(float(a0[2])))))
    r_true = perp_radius(kin, q_nom, q_nom)
    r_wrong = float(math.hypot(p_nom[0], p_nom[1]))
    print(f'    axis dir      : [{a0[0]:+.6f}, {a0[1]:+.6f}, {a0[2]:+.6f}]'
          f'   -> {tilt_deg:.4f} deg off vertical')
    print(f'    axis origin   : [{o0[0]:+.5f}, {o0[1]:+.5f}, {o0[2]:+.5f}]')
    print(f'    R to axis     : {r_true:.6f} m     <- the swing radius that matters')
    print(f'    hypot(x, y)   : {r_wrong:.6f} m     <- WRONG radius, {r_wrong/r_true:.4f}x too big')
    print(f'    => using hypot() shrinks every displacement by {r_true/r_wrong:.4f}x '
          f'(this was the 0.910cm-vs-1.000cm bug)')
    _, info_nom = randomize_start_pose(np.random.default_rng(0), half_m=H,
                                       q_cur=q_nom, kin=kin)
    check('randomizer measures the true perpendicular radius',
          abs(info_nom['r_pan_m'] - r_true) < 1e-9,
          f'probe {info_nom["r_pan_m"]:.6f} vs geometry {r_true:.6f}')
    check('pan axis is vertical (< 0.01 deg)', tilt_deg < 0.01, f'{tilt_deg:.4f} deg')

    # ------------------------------------------------------------------ (2)
    print()
    print('(2) chord identity: requested displacement == achieved displacement')
    print(f'    {"want(cm)":>10}{"achieved(cm)":>14}{"ratio":>9}{"pan(deg)":>11}{"dz(nm)":>10}')
    worst_ratio = 0.0
    for want_cm in (-0.0, -0.25, -0.5, -0.9, -1.0, 0.1, 0.5, 0.75, 0.9999, 1.0):
        q, info = randomize_start_pose(_Fixed(want_cm / 100.0), half_m=H,
                                       q_cur=q_nom, kin=kin)
        got = float(np.linalg.norm(kin.fk(q)[0] - p_nom))
        ratio = got / abs(want_cm / 100.0) if want_cm else 1.0
        worst_ratio = max(worst_ratio, abs(ratio - 1.0))
        print(f'    {want_cm:>10.4f}{got*100:>14.4f}{ratio:>9.6f}'
              f'{info["pan_deg"]:>11.4f}{info["dz_m"]*1e9:>10.2f}')
    check('achieved == requested (< 1um)', worst_ratio * H < TOL_CHORD_M,
          f'worst relative error {worst_ratio:.3e}')

    # ------------------------------------------------------------------ (3)
    print()
    print(f'(3) real randomize_start_pose, {args.n_random} samples')
    rng = np.random.default_rng(args.seed)
    ok_n, bad = 0, []
    hor, dz, dt, pand, qs = [], [], [], [], []
    t0 = time.time()
    for _ in range(args.n_random):
        q, info = randomize_start_pose(rng, half_m=H, q_cur=q_nom, kin=kin)
        if not info['ok']:
            bad.append(info.get('reason'))
            continue
        ok_n += 1
        hor.append(float(np.linalg.norm(kin.fk(q)[0] - p_nom)))
        dz.append(info['dz_m'])
        dt.append(info['d_tilt_deg'])
        pand.append(info['pan_deg'])
        qs.append(q.copy())
    dt_all = time.time() - t0
    qq = np.array(qs)
    print(f'    ok {ok_n}/{args.n_random} = {100*ok_n/args.n_random:.1f}%   '
          f'{dt_all:.2f}s ({dt_all/args.n_random*1000:.2f} ms/sample)')
    for b in bad[:3]:
        print(f'    FAIL {b}')

    check('every sample accepted', ok_n == args.n_random)
    check(f'|dz| < {TOL_DZ_M*1e9:.0f} nm',
          max(abs(x) for x in dz) < TOL_DZ_M,
          f'max {max(abs(x) for x in dz)*1e9:.2f} nm')
    check('tube tilt change == 0',
          max(abs(x) for x in dt) < TOL_TILT_DEG,
          f'max {max(abs(x) for x in dt)*1e6:.3f} u deg')
    spans = {}
    for i in range(N_ARM):
        if i == 0:
            continue
        spans[i] = float(qq[:, i].max() - qq[:, i].min())
    worst_span = max(spans.values())
    check('posture unchanged (non-pan joints)',
          worst_span < math.radians(TOL_JOINT_DEG),
          'spans: ' + '  '.join(f'{kin.arm_joint_names[i]}={math.degrees(v):.2e}deg'
                                for i, v in spans.items()))
    locked_arm = [i for i in FROZEN_IDX if i < N_ARM]
    locked_dev = float(np.abs(qq[:, locked_arm] - q_nom[locked_arm]).max())
    check('locked joints do not move', locked_dev < 1e-12,
          f'idx {locked_arm} ({[kin.arm_joint_names[i] for i in locked_arm]}) '
          f'max deviation {locked_dev:.2e} rad')
    check('displacement stays within +/-half',
          max(hor) <= H + 1e-9,
          f'{min(hor)*100:.4f} .. {max(hor)*100:.4f} cm  (request cap {H*100:.2f})')
    check('pan stays in joint limits',
          all(abs(x) < 45.0 for x in pand),
          f'pan {min(pand):+.3f} .. {max(pand):+.3f} deg')

    print()
    print('=' * 78)
    final = not fails
    print('  OK - randomization is safe to enable' if final
          else f'  FAILED - {len(fails)} check(s): {", ".join(fails)}')
    print('=' * 78)
    return 0 if final else 1


if __name__ == '__main__':
    sys.exit(main())
