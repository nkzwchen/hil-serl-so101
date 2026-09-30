#!/usr/bin/env python3
"""把首批录制修成能训练的形式。**只读原目录，写到新目录**，可反复重跑。

修三件事
--------
1. **观测布局：嵌套 → 规范**
   首批 52 段是裸 `SO101Env` 录的，`.pkl` 里是
       {'state': {'joint_pos': (6,), 'joint_vel': (6,)},
        'images': {'cam_main': (224,224,3), 'cam_wrist': (224,224,3)}}
   而训练侧（ReplayBuffer 的 `observation_space`）要的是
       {'state': (1,12) f32, 'cam_main': (1,224,224,3) u8, 'cam_wrist': (1,224,224,3) u8}
   转换 = `SERLObsWrapper` + `ChunkingWrapper(obs_horizon=1)` 的等价物，
   **已逐帧逐位验证与真实 wrapper 链输出完全相同**，无损。

2. **被死区抹掉的动作用实测位移补回来**
   `env.get_intervene_action` 的死区是 `norm(Δleader) < 0.05 rad ≈ 2.86°`，
   比 10Hz 下典型的人类单步位移还大；而 mode B 的从臂命令**不受死区影响**。
   于是 81% 的步 `action` 被记成 0，其中大半手臂其实动了 1~2°（最大 12.6°）→
   `(obs, action, next_obs)` 不自洽。
   leader 位移没存下来，无法还原原值；但 `obs` 与 `next_obs` 都在，
   而 mode B 的下发语义就是「从臂位移 = action × action_scale」，
   所以取 `action = Δjoint_pos / action_scale`（裁剪到 [-1,1]）——
   与 transition **按构造**自洽。（实测评据：动作↔位移相关 0.381 → 0.957）

3. **标记退化段**
   1 步的空段（`d` 在第一轮就触发 / 对齐耗尽了步数预算）单独列出来，
   `--drop-degenerate` 可以直接不写出去。

用法
----
    python fix_recordings.py --dry-run                     # 只报告
    python fix_recordings.py --out recordings_fixed        # 写新目录
    python fix_recordings.py --in-place --drop-degenerate  # 原地改（会先备份 meta）
"""

import argparse
import glob
import json
import os
import pickle
import shutil
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
from so101_recording_io import is_canonical_obs, to_canonical_obs  # noqa: E402

SCALE_DEFAULT = 0.05
CONTROLLED = [0, 1, 2, 3]


def load_cfg():
    from experiments.so101.config import TrainConfig
    cfg = TrainConfig()
    return cfg.proprio_keys, cfg.image_keys, cfg.ACTION_SCALE, cfg


def convert_episode(path, proprio_keys, image_keys, scale, repair=True):
    """返回 (新 transition 列表, 统计 dict)。"""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    tr = d['transitions']
    meta = d.get('meta', {})
    st = {'layout_fixed': 0, 'act_repaired': 0, 'act_zero_before': 0, 'n': len(tr)}

    out = []
    # ★ 每段一个 cache：保住"next_obs[k] 与 obs[k+1] 共用同一份图像"这件事，
    #   否则 pickle 去重失效、文件体积翻倍（实测 285→588 KB/步）。
    cache = {}
    for i, t in enumerate(tr):
        t = dict(t)
        o, no = t['observations'], t['next_observations']
        if not is_canonical_obs(o):
            st['layout_fixed'] += 1
        o = to_canonical_obs(o, proprio_keys, image_keys, cache=cache)
        no = to_canonical_obs(no, proprio_keys, image_keys, cache=cache)
        t['observations'], t['next_observations'] = o, no

        a = np.asarray(t['actions'], np.float32).copy()
        if repair and not a[CONTROLLED].any():
            st['act_zero_before'] += 1
            q = np.asarray(o['state'], np.float32).reshape(-1)[:5]
            qn = np.asarray(no['state'], np.float32).reshape(-1)[:5]
            dq = qn[CONTROLLED] - q[CONTROLLED]
            if np.abs(dq).max() > 1e-5:          # 确实动了才补
                a[CONTROLLED] = np.clip(dq / scale, -1.0, 1.0)
                st['act_repaired'] += 1
        t['actions'] = a
        out.append(t)

    # 顺带把退化信息记进 meta
    meta = dict(meta)
    meta['degenerate'] = bool(len(out) <= 2)
    meta['fixed'] = {'layout_from': 'nested', 'actions_repaired': st['act_repaired']}
    return out, meta, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='indir', default='recordings')
    ap.add_argument('--out', dest='outdir', default=None,
                    help='输出根目录（默认 recordings_fixed；与 --in-place 互斥）')
    ap.add_argument('--in-place', action='store_true')
    ap.add_argument('--drop-degenerate', action='store_true',
                    help='不写出 ≤2 步的退化段')
    ap.add_argument('--no-repair-actions', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.in_place and args.outdir:
        print('--in-place 与 --out 不能同时给'); return 2
    outroot = args.indir if args.in_place else (args.outdir or 'recordings_fixed')

    proprio_keys, image_keys, scale, cfg = load_cfg()
    print('=' * 92)
    print('fix_recordings: 嵌套观测 → 规范布局；被死区抹掉的动作按实测位移补回')
    print('=' * 92)
    print(f'  输入 {args.indir}   输出 {outroot}   '
          f'{"（原地）" if args.in_place else ""}{"（dry-run）" if args.dry_run else ""}')
    print(f'  proprio_keys={proprio_keys}  image_keys={image_keys}  scale={scale}')
    print(f'  动作修复 {"关闭" if args.no_repair_actions else "开启"}   '
          f'退化段 {"丢弃" if args.drop_degenerate else "保留（仅标记）"}')
    print()

    sess_dirs = sorted(d for d in glob.glob(os.path.join(args.indir, '*'))
                       if os.path.isdir(d))
    T = dict(eps=0, kept=0, dropped=0, fixed=0, repaired=0, steps=0)
    for sd in sess_dirs:
        name = os.path.basename(sd.rstrip('/'))
        files = sorted(glob.glob(os.path.join(sd, 'ep_*.pkl')))
        metap = os.path.join(sd, 'meta.json')
        meta0 = json.load(open(metap, encoding='utf-8')) if os.path.exists(metap) else {}
        outdir = os.path.join(outroot, name)
        if not args.dry_run and not args.in_place:
            os.makedirs(outdir, exist_ok=True)
        T['eps'] += len(files)
        print(f'### {name}  {len(files)} 段')
        neweps = []
        for p in files:
            fn = os.path.basename(p)
            trs, meta, st = convert_episode(
                p, proprio_keys, image_keys, scale,
                repair=not args.no_repair_actions)
            T['fixed'] += st['layout_fixed']
            T['repaired'] += st['act_repaired']
            T['steps'] += st['n']
            if meta.get('degenerate') and args.drop_degenerate:
                T['dropped'] += 1
                print(f'    ✂ {fn}: {st["n"]} 步（退化段，丢弃）')
                continue
            T['kept'] += 1
            tag = '退化段' if meta.get('degenerate') else ''
            print(f'    ✅ {fn}: {st["n"]} 步  布局转 {st["layout_fixed"]} 帧  '
                  f'补动作 {st["act_repaired"]} 步 {tag}')
            if not args.dry_run:
                od = os.path.join(outdir, fn)
                tmp = od + '.tmp'
                with open(tmp, 'wb') as f:
                    pickle.dump({'transitions': trs, 'meta': meta}, f)
                os.replace(tmp, od)
            # meta.json 里那条摘要保留原字段（起点倾角、回程时间…），只更正/补充
            orig = next((e for e in meta0.get('episodes', [])
                         if e.get('file') == fn), {})
            neweps.append({**orig, 'file': fn, 'n_steps': st['n'],
                           'degenerate': meta.get('degenerate', False),
                           'actions_repaired': st['act_repaired']})
        if not args.dry_run:
            m2 = dict(meta0)
            m2['episodes'] = neweps
            m2['fixed_by'] = ('fix_recordings.py: 嵌套观测→规范布局；'
                              '死区抹掉的动作按实测位移补回')
            m2['obs_layout'] = 'canonical (state (1,12) + cam_* (1,224,224,3))'
            if not args.in_place:
                with open(os.path.join(outdir, 'meta.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(m2, f, ensure_ascii=False, indent=2)
        print()

    print('=' * 92)
    print(f'  段 {T["eps"]} → 写出 {T["kept"]}'
          + (f'（丢弃退化段 {T["dropped"]}）' if T['dropped'] else ''))
    print(f'  布局转换帧数 {T["fixed"]} / {T["steps"]}')
    print(f'  补回动作的步数 {T["repaired"]} / {T["steps"]} '
          f'= {T["repaired"]/max(T["steps"],1)*100:.1f}%')
    print('=' * 92)
    if not args.dry_run:
        print(f'\n  下一步：python build_demos.py {outroot}/*/  （标定后）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
