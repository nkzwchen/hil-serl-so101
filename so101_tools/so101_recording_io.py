#!/usr/bin/env python3
"""Recording file readers — the ONE place that knows the on-disk formats.

Why this module exists
----------------------
`label_demos.py` and `build_demos.py` each used to implement their own format
dispatch. They drifted: the labeler learned 4 layouts, `build_demos.py` kept
handling only `{'transitions': [...]}` — so a multi-episode file could be
labelled but then failed to build, with a bare `KeyError: 'transitions'`.
Two readers for one file format is a bug waiting to happen; there is now one.

Supported layouts
-----------------
  0) {'transitions': [...], 'meta': {...}}          -- record_demo_so101.py，
                                                       **一段一个文件**（真实路径）
  1) {'episodes': [{'transitions': [...]}, ...], 'meta': {...}}   -- 多段容器
  2) [[t, t, ...], [t, t, ...]]                     -- 每段一个内层 list
  3) [t, t, ...]                                    -- 官方扁平列表，按 dones 切段

In every case the return value is

    (episodes, meta)

where `episodes` is a list of RAW transition lists (one list per episode) and
`meta` is whatever the producer recorded at file level ({} if none).
Raw, not derived: the labeler wants decoded frames, the builder wants the exact
transitions it will write into the demo buffer. Let each one derive its own.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple


def load_raw(path: str) -> Tuple[List[list], Dict[str, Any]]:
    """Load a recording file; return (episodes_of_raw_transitions, meta)."""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # ★ record_demo_so101.py 的产出：**一段一个文件**，形如
    #     {'transitions': [...], 'meta': {...}}
    #   必须在这里认出来，否则录制和标定两步根本接不上（踩过）。
    if isinstance(data, dict) and 'transitions' in data:
        return [list(data['transitions'])], dict(data.get('meta') or {})

    if isinstance(data, dict) and 'episodes' in data:
        meta = dict(data.get('meta') or {})
        out = []
        for e in data['episodes']:
            trs = e['transitions'] if isinstance(e, dict) and 'transitions' in e else e
            out.append(list(trs))
        return out, meta

    if isinstance(data, list) and data and isinstance(data[0], list):
        return [list(e) for e in data], {}

    if isinstance(data, list) and data and isinstance(data[0], dict):
        # 官方扁平列表：按 dones 切段
        segs, cur = [], []
        for t in data:
            cur.append(t)
            if bool(t.get('dones', False)):
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        return segs, {'sliced_by': 'dones'}

    raise ValueError(
        f'认不出的录制文件格式：{path}\n'
        f'  顶层类型 {type(data).__name__}'
        + (f'，keys={sorted(data)}' if isinstance(data, dict) else ''))


def labels_path(rec_path: str) -> str:
    """`recordings/ep_0000.pkl` -> `recordings/ep_0000.labels.json`."""
    import os
    return os.path.splitext(rec_path)[0] + '.labels.json'

def load_labels(rec_path: str) -> Dict[int, Dict[str, Any]]:
    """Return {episode_index: {'status':..., 'frame':...}} (empty if unlabelled)."""
    import json
    import os
    lp = labels_path(rec_path)
    if not os.path.exists(lp):
        return {}
    try:
        with open(lp, encoding='utf-8') as f:
            d = json.load(f)
        return {int(k): v for k, v in (d.get('episodes') or {}).items()}
    except Exception:
        return {}


# ------------------------------------------------------- observation layout
# 观测有**两种**形态，记录在这里免得再搞混（2026-09-29 真实录制全部踩中）：
#
#   原始（SO101Env._get_obs / 裸 env.observation_space）—— 嵌套：
#       {'state': {'joint_pos': (6,), 'joint_vel': (6,)},
#        'images': {'cam_main': (224,224,3), 'cam_wrist': (224,224,3)}}
#
#   规范（SERLObsWrapper + ChunkingWrapper(obs_horizon=1) 之后）—— 扁平、带前导维：
#       {'state': (1,12) float32, 'cam_main': (1,224,224,3) uint8, ...}
#
# **训练要用的是规范形态**（ReplayBuffer 的 observation_space 就是它）。
# 录制器必须走 TrainConfig().get_environment() 拿到包装后的 env ——
# 用裸 SO101Env 录出来的 pkl 会被 build_demos 拒绝、也会让 learner 的
# demo_buffer.insert() KeyError。

def is_canonical_obs(obs) -> bool:
    """True if `obs` already looks like the flat, obs_horizon-stacked layout."""
    if not isinstance(obs, dict) or 'state' not in obs:
        return False
    return 'images' not in obs and not isinstance(obs['state'], dict)


def to_canonical_obs(obs, proprio_keys, image_keys, cache=None):
    """嵌套观测 → 规范观测（`SERLObsWrapper` + `ChunkingWrapper` 的等价物）。

    已在真机录制上**逐帧逐位**验证与真实 wrapper 链输出完全相同（68 帧全等），
    所以这是无损转换，不是近似。

    `state` = 按 `proprio_keys` 顺序拼接（Dict space 的 flatten 就是按 key 顺序
    拼接），再各加一维 `obs_horizon` 前导维（`post_stack_obs` 的 `v[None]`）。

    ⚠️ `cache`：录制数据里 `next_observations[k]` 与 `observations[k+1]` 的**图像是
    同一个 ndarray 对象**，pickle 靠这一点去重（实测 285KB/步）。转换时若每处各建
    一份视图，去重就没了 → 文件**体积翻倍**（588KB/步）。传入同一个 dict 当 cache
    即可按源数组 id 复用同一份视图，保住体积。
    """
    import numpy as np
    if isinstance(obs, dict) and 'images' in obs and isinstance(obs.get('state'), dict):
        st = np.concatenate([np.asarray(obs['state'][k], np.float32)
                             for k in proprio_keys])
        out = {'state': st[None]}
        for k in image_keys:
            src = obs['images'][k]
            if cache is None:
                out[k] = np.asarray(src)[None]
            else:
                ck = id(src)
                if ck not in cache:
                    # 把 src 一起存住，避免 id() 被回收复用
                    cache[ck] = (src, np.asarray(src)[None])
                out[k] = cache[ck][1]
        return out
    return obs          # 已经是规范形态（或未知形态）：原样返回
