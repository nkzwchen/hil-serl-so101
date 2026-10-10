#!/usr/bin/env python3
"""Turn labelled recordings into a HIL-SERL demo pkl (`--demo_path` format).

Pipeline position
-----------------
    record_demo_so101.py   ->  recordings/<session>/ep_XXXX.pkl
    label_demos.py         ->  recordings/<session>/ep_XXXX.labels.json
    build_demos.py         ->  demo_data/so101_<N>demos_<ts>.pkl   <-- this script

Output format (verified against examples/record_demos.py)
--------------------------------------------------------
A plain `pickle.dump` of a **flat list of transitions**, each:
    dict(observations=..., actions=..., next_observations=...,
         rewards=..., masks=..., dones=..., infos=...)
`train_rlpd.py --demo_path` loads exactly this and calls
`demo_buffer.insert(transition)` per element. `include_grasp_penalty=False` for
`single-arm-fixed-gripper`, so no `grasp_penalty` field is needed.

Sparse reward and the off-by-one that matters
---------------------------------------------
`label_demos.py` shows `observations[i]` as frame i. If you mark success at frame
k, then obs_k IS the successful state -- and the transition that PRODUCED it is
k-1: `(obs_{k-1}, a_{k-1}, obs_k, r, done)`. So the sparse reward+done goes on
transition `k-1`, and the recording is truncated after it (in a real episode the
env would have ended there). Frame 0 is therefore rejected: it has no preceding
transition to reward.

Validation
----------
Every emitted transition is checked against the REAL `observation_space` /
`action_space` (from `TrainConfig().get_environment(fake_env=True)`), key by key,
shape by shape. That is the check that catches the "shapes do not match, crash at
the first ReplayBuffer insert" class of bug -- before training, not during it.

Usage
-----
    python build_demos.py recordings/2026-09-28_224840_test/
    python build_demos.py recordings/*/ep_*.pkl --out demo_data
    python build_demos.py recordings/sess/ --dry-run      # validate only
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
for _directory in (_REPO / 'examples/experiments/so101', _REPO / 'examples/experiments', _REPO / 'examples'):
    sys.path.insert(0, str(_directory))


# --------------------------------------------------------------------- loading
# 读盘（格式分发）与标签读取都放在 so101_recording_io —— 单一来源。
# 以前这里自己实现了一遍，结果 label_demos 认 4 种格式、这里只认 1 种，
# 多段文件标完却 build 不了（KeyError: 'transitions'）。
from so101_recording_io import labels_path, load_labels, load_raw  # noqa: E402


def collect_pairs(paths):
    """Expand the CLI arguments into sorted recording pkl paths."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += [os.path.join(p, f) for f in sorted(os.listdir(p))
                    if f.endswith('.pkl') and not f.endswith('.labels.json')]
        else:
            g = sorted(glob.glob(p))
            out += g if g else [p]
    return [p for p in out if os.path.exists(p)]


# ------------------------------------------------------------------ validation
def build_space_checker():
    """Return f(transition) -> list of problems, using the REAL declared spaces."""
    from experiments.so101.config import TrainConfig
    fake = TrainConfig().get_environment(fake_env=True)
    obs_sp, act_sp = fake.observation_space, fake.action_space

    def check(t):
        bad = []
        for fld in ('observations', 'next_observations'):
            o = t.get(fld)
            if not isinstance(o, dict):
                bad.append(f'{fld} 不是 dict')
                continue
            for k, sp in obs_sp.spaces.items():
                if k not in o:
                    bad.append(f'{fld} 缺 key {k!r}')
                    continue
                a = np.asarray(o[k])
                if tuple(a.shape) != tuple(sp.shape):
                    bad.append(f'{fld}[{k!r}] shape {tuple(a.shape)} != {tuple(sp.shape)}')
                elif a.dtype != sp.dtype:
                    bad.append(f'{fld}[{k!r}] dtype {a.dtype} != {sp.dtype}')
            extra = set(o) - set(obs_sp.spaces)
            if extra:
                bad.append(f'{fld} 多出 key {sorted(extra)}')
        a = np.asarray(t.get('actions', []))
        if tuple(a.shape) != tuple(act_sp.shape):
            bad.append(f'actions shape {tuple(a.shape)} != {tuple(act_sp.shape)}')
        for k in ('rewards', 'masks', 'dones'):
            if k not in t:
                bad.append(f'缺 {k}')
        return bad

    return check, obs_sp, act_sp


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recordings', nargs='+',
                    help='session directory, pkl file(s), or glob(s)')
    ap.add_argument('--out', default='demo_data', help='output directory')
    ap.add_argument('--dry-run', action='store_true', help='validate + report only')
    ap.add_argument('--all-failed', action='store_true',
                    help='如果**一段成功都没有**（全标成 failure/未标），就把所有段'
                         '完整上传（reward 全 0，每段末尾 done）。RLPD 的离线 prior '
                         'data 不要求带成功奖励，但这样上传的数据**学不到"成功"这个概念**，'
                         '只当行为先验用。')
    ap.add_argument('--allow-frame0', action='store_true',
                    help='accept a success marked on frame 0 (rewards obs_0 with no '
                         'preceding action -- usually a labelling mistake)')
    ap.add_argument('--allow-provisional', action='store_true',
                    help='显式允许待复核粗标签，仅用于流程验证')
    args = ap.parse_args()
    if args.all_failed:
        ap.error('--all-failed 已禁用：不能把未标注或未复核数据自动变成失败奖励')

    pairs = collect_pairs(args.recordings)
    if not pairs:
        print('ERROR: no recording files found')
        return 2
    print('=' * 78)
    print(f'build_demos: {len(pairs)} 个录制文件')
    print('=' * 78)

    check, obs_sp, act_sp = build_space_checker()
    print('  目标观测空间: ' + ', '.join(f'{k}{tuple(v.shape)} {v.dtype}'
                                        for k, v in obs_sp.spaces.items()))
    print(f'  目标动作空间: {act_sp}')
    print()

    from experiments.so101.config import TrainConfig
    config = TrainConfig()
    expected_roi = config.roi_spec
    expected_main_roi = config.main_roi_spec
    kept, skipped, all_tr, all_raw = [], [], [], []
    for rec in pairs:
        fname = os.path.basename(rec)
        # 一个文件里可能有多段（见 so101_recording_io 的格式表），逐段取标签。
        try:
            raw_eps, rmeta = load_raw(rec)
        except Exception as e:
            skipped.append((fname, f'读不出来：{e}'))
            continue
        if rmeta.get('recording', {}).get('has_gaps'):
            skipped.append((fname, '保存队列有缺失步；须先人工复核并切分连续轨迹，禁止直接训练'))
            continue
        if (rmeta.get('action_semantics') != config.ACTION_SEMANTICS or
                rmeta.get('observation_semantics') != config.OBSERVATION_SEMANTICS or
                rmeta.get('action_scale_rad') != config.ACTION_SCALE or
                rmeta.get('action_joints') != config.ACTION_JOINTS or
                rmeta.get('env_hz') != config.ENV_HZ or
                rmeta.get('control_backend') != config.CONTROL_BACKEND or
                rmeta.get('pose_alignment') != config.POSE_ALIGNMENT):
            skipped.append((fname, '动作/观测版本、缩放、频率、执行后端或姿态映射 与当前配置不一致；禁止混用'))
            continue
        if (rmeta.get('roi_spec') != expected_roi or
                rmeta.get('main_roi_spec') != expected_main_roi):
            skipped.append((fname, 'ROI 配置与当前训练配置不一致，拒绝混用'))
            continue
        labs = load_labels(rec)
        multi = len(raw_eps) > 1
        for ei, trs in enumerate(raw_eps):
            name = f'{fname}#{ei}' if multi else fname
            # ★ 先无条件登记：--all-failed 兜底要用到"所有段"，不只是成功段
            all_raw.append((name, trs))
            lab = labs.get(ei)
            if lab is None:
                skipped.append((name, '没有标签（还没标）'))
                continue
            if lab.get('provisional', False) and not args.allow_provisional:
                skipped.append((name, '粗标签待人工复核；流程验证可显式使用 --allow-provisional'))
                continue
            if lab.get('status') != 'success':
                skipped.append((name, f"标签是 {lab.get('status')}"))
                continue
            j = lab.get('frame')
            if j is None:
                skipped.append((name, '标签是 success 但没有 frame'))
                continue
            j = int(j)
            if j < 1 and not args.allow_frame0:
                skipped.append((name, f'成功帧 = {j}：它前面没有 transition 可给奖励 '
                                      f'（请标 ≥ 1 的帧，或用 --allow-frame0）'))
                continue
            if j > len(trs):
                skipped.append((name, f'成功帧 {j} 超出该段 {len(trs)} 步'))
                continue

            seg = [dict(t) for t in trs[:max(j, 1)]]
            # 稀疏奖励：落在 k-1 上（它的 next_observation 才是第 k 帧那个成功状态）
            for i, t in enumerate(seg):
                last = (i == len(seg) - 1)
                t['rewards'] = 1.0 if last else 0.0
                t['masks'] = 0.0 if last else 1.0
                t['dones'] = bool(last)
                if 'infos' not in t or not isinstance(t['infos'], dict):
                    t['infos'] = {}
                t['infos'] = dict(t['infos'])
                t['infos']['reward_annotation'] = dict(lab)
                t['infos'].pop('intervene_action', None)

            bad = []
            for i, t in enumerate(seg):
                bad += [f'step{i}: {b}' for b in check(t)]
            if bad:
                skipped.append((name, '形状/格式校验失败：' + '; '.join(bad[:3])))
                continue

            kept.append(name)
            all_tr += seg
            tilt = rmeta.get('start_info', {}).get('act_tilt_deg')
            print(f'  ✅ {name}: 取前 {len(seg)}/{len(trs)} 步，成功帧 {j} '
                  f'(奖励落在 transition {j - 1})'
                  + (f'，起点管子倾角 {tilt}°' if tilt is not None else ''))

    print()
    if skipped:
        print(f'  跳过 {len(skipped)} 个：')
        for n, why in skipped:
            print(f'    · {n}: {why}')
    print()
    if not all_tr and args.all_failed and all_raw:
        # ── 兜底：一段成功都没有 → 按需求「全上传」────────────────────────
        #    每段完整收进来，reward 全 0，只在段末 done/mask=0。
        #    ⚠️ 这样上传的数据里**没有任何成功样本**，RLPD 只能拿它当行为先验，
        #       学不到"成功"是什么。等有成功段时应当重跑（不要混用）。
        print('  ⚠️ 没有任何成功段 → 按 --all-failed 全量上传（reward 全 0）')
        all_tr = []
        for name, trs in all_raw:
            seg = [dict(t) for t in trs]
            for i, t in enumerate(seg):
                last = (i == len(seg) - 1)
                t['rewards'] = 0.0
                t['masks'] = 0.0 if last else 1.0
                t['dones'] = bool(last)
                if not isinstance(t.get('infos'), dict):
                    t['infos'] = {}
                t['infos'].pop('intervene_action', None)
            bad = []
            for i, t in enumerate(seg):
                bad += [f'step{i}: {b}' for b in check(t)]
            if bad:
                print(f'    · {name}: 校验失败，跳过（{bad[:2]}）')
                continue
            all_tr += seg
            kept.append(name + '(全)')
        skipped = []
        print(f'  全量收录 {len(kept)} 段')

    if not all_tr:
        print('❌ 一条 demo 都没有产出。')
        print('   常见原因：还没标定（label_demos.py）、全标成了失败、'
              '或成功帧标在了第 0 帧。')
        print('   粗标签仅供流程验证：显式添加 --allow-provisional；正式数据请先人工复核。')
        return 1

    n_steps = len(all_tr)
    print(f'  合计 {len(kept)} 段 / {n_steps} 个 transition')
    if args.dry_run:
        print('  --dry-run：不写文件。')
        return 0

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(
        args.out, f'so101_{len(kept)}demos_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pkl')
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(all_tr, f)
    os.replace(tmp, out)
    print(f'\n  💾 {out}')
    print(f'     {os.path.getsize(out)/1e6:.1f} MB')
    print(f'\n  用法：train_rlpd.py ... --demo_path {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
