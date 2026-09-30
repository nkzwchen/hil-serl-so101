#!/usr/bin/env python3
"""体检 `recordings/` 里的录制数据：视频到底存了没、动作到底存了没、能不能用。

检查项
------
图 像
  · 两个相机是否**每一帧**都在、形状/类型对不对
  · 是否全 0（黑图）、是否整段一模一样（相机卡死 —— 踩过的真实故障）
  · 段内**有多少帧是真变化的**（相机活着 ≠ 画面在动）
  · 亮度分布是否正常（过曝/欠曝）
动 作
  · 形状/类型/取值域；是否全 0；是否在变化（不是卡住）
  · `action × action_scale` 与实测关节位移是否对得上 ——
    这是 RLPD 的命根子：`(obs, action, next_obs)` 必须满足"施加 action 得到 next_obs"
  · 接管比例
结 构
  · 七个字段齐不齐；transition 链 `next_obs[k] == obs[k+1]` 自不自洽
  · 观测布局是嵌套还是规范（嵌套 = 训练用不了）

用法
----
    python audit_recordings.py                       # 全部
    python audit_recordings.py --session v2          # 只查某个会话
    python audit_recordings.py --fast                # 只查前 8 帧图像（快）
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

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
from so101_home_pose import N_ARM  # noqa: E402

ACTION_SCALE = 0.05
CONTROLLED = [0, 1, 2, 3]          # shoulder_pan/lift, elbow_flex, wrist_flex


def unpack_state(o):
    """嵌套或规范，都归一成 (joint_pos6, joint_vel6)。"""
    if isinstance(o.get('state'), dict):
        return (np.asarray(o['state']['joint_pos'], np.float32),
                np.asarray(o['state']['joint_vel'], np.float32))
    s = np.asarray(o['state'], np.float32).reshape(-1)
    return s[:6], s[6:12]


def unpack_images(o):
    if isinstance(o.get('images'), dict):
        return {k: np.asarray(v) for k, v in o['images'].items()}
    return {k: np.asarray(v) for k, v in o.items() if k != 'state'}


def audit_episode(path, fast=False):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    tr = d['transitions']
    meta = d.get('meta', {})
    n = len(tr)
    r = {'path': path, 'n': n, 'meta': meta, 'problems': [], 'warn': []}

    # ---------- 结构 ----------
    need = {'observations', 'next_observations', 'actions',
            'rewards', 'masks', 'dones', 'infos'}
    miss = need - set(tr[0])
    if miss:
        r['problems'].append(f'缺字段 {sorted(miss)}')

    o0 = tr[0]['observations']
    r['layout'] = '嵌套(训练用不了)' if isinstance(o0.get('state'), dict) else '规范'

    # ---------- 动作 ----------
    A = np.asarray([t['actions'] for t in tr], np.float32)
    r['act_shape'] = tuple(A.shape)
    if A.shape[1:] != (4,):
        r['problems'].append(f'动作形状 {A.shape[1:]} != (4,)')
    r['act_absmax'] = float(np.abs(A).max()) if A.size else 0.0
    r['act_allzero'] = bool(A.size and not A.any())
    # 每一维是否在变
    r['act_std_per_dim'] = [float(A[:, i].std()) for i in range(A.shape[1])] \
        if A.ndim == 2 and A.shape[1] == 4 else []
    r['act_range'] = [float(A[:, i].min()) for i in range(4)] if A.ndim == 2 and A.shape[1] == 4 else []
    r['act_range_hi'] = [float(A[:, i].max()) for i in range(4)] if A.ndim == 2 and A.shape[1] == 4 else []
    if A.size and np.abs(A).max() > 1.0 + 1e-6:
        r['problems'].append(f'动作超出 [-1,1]：|max|={np.abs(A).max():.3f}')
    if r['act_allzero']:
        r['problems'].append('动作全是 0')

    # ---------- 动作 ↔ 实测位移 ----------
    if n >= 2:
        cur = np.asarray([unpack_state(t['observations'])[0] for t in tr], np.float32)
        nxt = np.asarray([unpack_state(t['next_observations'])[0] for t in tr], np.float32)
        dq = (nxt - cur)[:, CONTROLLED]                # 实测位移（rad）
        pred = A[:, CONTROLLED] * ACTION_SCALE          # 期望位移
        err = np.abs(dq - pred)
        r['consist_med_deg'] = float(np.degrees(np.median(err)))
        r['consist_max_deg'] = float(np.degrees(err.max()))
        # 相关性：动作能不能解释运动
        if pred.std() > 1e-9 and dq.std() > 1e-9:
            r['corr'] = float(np.corrcoef(pred.ravel(), dq.ravel())[0, 1])
        else:
            r['corr'] = float('nan')
        # 实测是否真的在动。
        # ⚠️ 口径：先对**每个时间步**取受控维上的最大位移，再对步取统计量。
        #    直接对 (T,4) 展平取中位会被"只有一维在动"淹没成 0，看着像没动。
        step_move = np.degrees(np.abs(dq).max(axis=1))
        r['motion_med_deg'] = float(np.median(step_move))
        r['motion_max_deg'] = float(np.degrees(np.abs(dq).max()))
        r['move_frac'] = float((step_move > 0.05).mean())      # 有效运动步占比
        r['act_sat_frac'] = float((np.abs(A) > 0.999).mean())  # 动作触顶比例

    # ---------- 图像 ----------
    idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    if fast:
        idxs = sorted(set(idxs + [min(3, n - 1)]))
    frames = {}
    for i in idxs:
        for kn, which in (('imgs', 'observations'), ('nims', 'next_observations')):
            im = unpack_images(tr[i][which])
            for k, v in im.items():
                frames.setdefault(k, []).append((i, which, v))
    r['cams'] = sorted({k for k in frames})
    r['cam_shapes'] = {}
    for k, lst in frames.items():
        v = np.asarray(lst[0][2])
        if v.ndim == 4:          # 规范布局带 obs_horizon 前导维
            v = v[0]
        lst[0] = (lst[0][0], lst[0][1], v)
        r['cam_shapes'][k] = (tuple(v.shape), str(v.dtype))
        if v.ndim != 3 or v.shape[2] != 3:
            r['problems'].append(f'{k} 形状异常 {v.shape}')
        if v.dtype != np.uint8:
            r['problems'].append(f'{k} dtype {v.dtype} != uint8')
        if not v.any():
            r['problems'].append(f'{k} 有全 0 帧（黑图）')
        m = float(v.mean())
        if m < 3:
            r['warn'].append(f'{k} 平均亮度很低 {m:.1f}')
        if m > 250:
            r['warn'].append(f'{k} 平均亮度接近饱和 {m:.1f}')

    # 整段是否"冻住"：抽 6 帧比一比（用主相机）
    frozen = {}
    for k, lst in frames.items():
        if len(lst) >= 2:
            a = np.asarray(lst[0][2]);  a = a[0] if a.ndim == 4 else a
            same = all(a.shape == b[2].shape and np.array_equal(a, b[2]) for b in lst[1:])
            frozen[k] = bool(same)
    r['frozen'] = frozen
    for k, v in frozen.items():
        if v and n > 2:
            r['problems'].append(f'{k} 抽样的 {len(frames[k])} 帧完全相同 → 相机可能卡死')

    # 逐帧差异：整段连续帧的平均绝对差。相机"活着"不等于画面有信息 ——
    # 真实的故障是画面几乎不动（对着白墙/卡死），这种数据策略学不到东西。
    for kn, which in (('observations', 'observations'),
                      ('next_observations', 'next_observations')):
        ims = [unpack_images(t[which]) for t in tr]
        for k in (ims[0].keys() if ims else []):
            arr = [np.asarray(x[k], np.int16) for x in ims if k in x]
            if len(arr) < 2:
                continue
            diffs = [float(np.abs(arr[i + 1] - arr[i]).mean())
                     for i in range(len(arr) - 1)]
            uniq = len({a.tobytes() for a in arr})
            key = f'{k}'
            r.setdefault('frame_diff', {})[key] = float(np.median(diffs))
            r.setdefault('uniq_frames', {})[key] = uniq
            r.setdefault('n_frames', {})[key] = len(arr)
            if np.median(diffs) < 0.5 and n > 5:
                r['warn'].append(f'{k}: 帧间差异极小（{np.median(diffs):.2f}/255）'
                                 f' → 画面几乎不动，视频信息量可疑')

    # ---------- 链自洽 ----------
    if n >= 2:
        bad = 0
        for k in range(n - 1):
            if bool(tr[k]['dones']):
                continue
            a, b = unpack_state(tr[k]['next_observations'])[0], unpack_state(tr[k + 1]['observations'])[0]
            if not np.array_equal(a, b):
                bad += 1
        r['chain_bad'] = bad
        if bad:
            r['problems'].append(f'{bad}/{n - 1} 处 next_obs[k] != obs[k+1]')

    r['intervened'] = int(sum(bool((t.get('infos') or {}).get('intervening')) for t in tr))
    r['dones_last'] = bool(tr[-1]['dones'])
    r['masks_last'] = float(tr[-1]['masks'])
    if not r['dones_last']:
        r['warn'].append('最后一条 dones 不是 True')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='recordings')
    ap.add_argument('--session', default=None, help='只查名字里含这个串的会话')
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()

    sess_dirs = sorted(d for d in glob.glob(os.path.join(args.root, '*')) if os.path.isdir(d))
    if args.session:
        sess_dirs = [d for d in sess_dirs if args.session in d]
    grand = []
    for sd in sess_dirs:
        files = sorted(glob.glob(os.path.join(sd, 'ep_*.pkl')))
        print('=' * 100)
        print(f'{sd}   {len(files)} 段')
        print('=' * 100)
        print(f"{'ep':>5}{'步':>5}{'布局':>14}{'相机':>22}"
              f"{'动作max':>8}{'触顶%':>7}{'运动中位°':>10}{'运动步%':>8}"
              f"{'帧差/255':>10}{'唯一帧':>8}{'动作↔位移°':>11}{'相关':>7}")
        rows = []
        for p in files:
            r = audit_episode(p, fast=args.fast)
            rows.append(r)
            grand.append(r)
            cams = r.get('cam_shapes', {})
            cam_txt = ','.join(f'{k.replace("cam_","")}{v[0][0]}' for k, v in cams.items()) or '无'
            fd = r.get('frame_diff', {})
            uf = r.get('uniq_frames', {})
            nf = r.get('n_frames', {})
            fd_txt = ','.join(f'{fd[k]:.2f}' for k in sorted(fd))
            uf_txt = ','.join(f'{uf[k]}/{nf[k]}' for k in sorted(uf))
            print(f"{os.path.basename(p)[3:7]:>5}{r['n']:>5}{r['layout']:>14}{cam_txt:>22}"
                  f"{r.get('act_absmax', 0):>8.3f}{r.get('act_sat_frac', 0)*100:>7.1f}"
                  f"{r.get('motion_med_deg', float('nan')):>10.3f}"
                  f"{r.get('move_frac', float('nan'))*100:>8.0f}"
                  f"{fd_txt:>10}{uf_txt:>8}"
                  f"{r.get('consist_med_deg', float('nan')):>11.3f}"
                  f"{r.get('corr', float('nan')):>7.3f}")
            for pr in r['problems']:
                print(f'        ❌ {pr}')
            for w in r['warn']:
                print(f'        ⚠️  {w}')
        # 会话小结
        n_ok = sum(1 for r in rows if not r['problems'])
        n1 = sum(1 for r in rows if r['n'] <= 2)
        print(f'\n  —— {os.path.basename(sd.rstrip("/"))}: {n_ok}/{len(rows)} 段无问题；'
              f'{n1} 段 ≤2 步；'
              f'布局={sorted({r["layout"] for r in rows})}')
        dots = [r for r in rows if r['n'] > 2]
        if dots:
            print(f'     正常段：动过 {np.median([r.get("motion_med_deg", 0) for r in dots]):.3f}°/步(中位)，'
                  f'动作↔位移偏差中位 {np.median([r.get("consist_med_deg", 0) for r in dots]):.3f}°，'
                  f'相关 {np.nanmedian([r.get("corr", np.nan) for r in dots]):.3f}')
        print()

    print('=' * 100)
    print('总计')
    print('=' * 100)
    print(f'  段数            : {len(grand)}')
    print(f'  有问题的段      : {sum(1 for r in grand if r["problems"])}')
    print(f'  布局            : {sorted({r["layout"] for r in grand})}')
    lay = {r['layout'] for r in grand}
    if '嵌套(训练用不了)' in lay:
        print('  ⚠️ 存在**嵌套**观测 → 这些段训练侧用不了（需要转换）')
    good = [r for r in grand if r['n'] > 2]
    if good:
        print(f'  正常段(n>2)     : {len(good)}  步数中位 {np.median([r["n"] for r in good]):.0f}')
        print(f'  动作↔位移偏差   : 中位 {np.median([r["consist_med_deg"] for r in good]):.3f}°  '
              f'最大 {max(r["consist_max_deg"] for r in good):.3f}°')
        print(f'  动作-位移相关   : 中位 {np.nanmedian([r["corr"] for r in good]):.3f}')
        print(f'  实际运动        : 中位 {np.median([r["motion_med_deg"] for r in good]):.3f}°/步  '
              f'峰值 {max(r["motion_max_deg"] for r in good):.3f}°')
    return 0


if __name__ == '__main__':
    sys.exit(main())
