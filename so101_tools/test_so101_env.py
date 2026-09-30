#!/usr/bin/env python
"""SO101Env 端到端验证（需要 so101_rl_env.launch.py 已在跑）。

验证项：
  1. 能否拿到 /follower/joint_states（观测）
  2. SERLObsWrapper 拍平后的观测结构是否正确
  3. 能否发命令让 follower 动（小幅度、安全）
  4. leader 是否跟随 follower（模式 A）
  5. 空格接管开关是否工作（不会真的按空格，只检查属性存在）

用法：
    python test_so101_env.py            # 只读观测 + 极小动作测试
    python test_so101_env.py --move     # 允许发 ±0.03rad 的小动作
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--move', action='store_true', help='允许发小幅度动作')
    ap.add_argument('--steps', type=int, default=20)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')

    from experiments.so101.config import TrainConfig

    c = TrainConfig()
    print('=== 构造 SO101Env（真实 ROS2 客户端）===')
    env = c.get_environment(fake_env=False)
    print('  观测空间 keys:', sorted(env.observation_space.keys()))
    print('  state shape :', env.observation_space['state'].shape)
    print('  action shape:', env.action_space.shape)

    print()
    print('=== reset ===')
    obs, info = env.reset()
    print('  obs keys:', sorted(obs.keys()))
    print('  state   :', np.asarray(obs['state']).shape)
    for k in obs:
        if k != 'state':
            a = np.asarray(obs[k])
            print(f'  {k}: shape={a.shape} dtype={a.dtype} mean={a.mean():.1f}')
    print('  info    :', info)

    print()
    print('=== 读 3 帧观测（看数据是否在变）===')
    base = env.unwrapped if hasattr(env, 'unwrapped') else env
    for i in range(3):
        # 从 wrapper 链里找到真实 env
        e = env
        while hasattr(e, 'env'):
            e = e.env
        jp = e.joint_pos
        lp = e.leader_pos
        print(f'  [{i}] follower={None if jp is None else np.round(jp,3)}')
        print(f'      leader  ={None if lp is None else np.round(lp,3)}')
        time.sleep(0.5)

    if args.move:
        print()
        print(f'=== 发 {args.steps} 步小动作（±0.03 rad/步，安全幅度）===')
        e = env
        while hasattr(e, 'env'):
            e = e.env
        # 临时把 action_scale 调小，避免大幅运动
        old_scale = e.action_scale
        e.action_scale = 0.03
        try:
            for i in range(args.steps):
                # 只在 shoulder_pan 上给 +0.3 的归一化动作（≈ 0.009 rad）
                a = np.zeros(6, dtype=np.float32)
                a[0] = 0.3 * (1 if (i // 5) % 2 == 0 else -1)
                obs, rew, done, trunc, info = env.step(a)
                jp = e.joint_pos
                lp = e.leader_pos
                if i % 5 == 0:
                    print(f'  [{i:2d}] follower_pan={jp[0]:+.4f} leader_pan={lp[0]:+.4f} '
                          f'rew={rew} intervening={info.get("intervening")}')
                if done:
                    print(f'  episode 结束于第 {i} 步')
                    break
        finally:
            e.action_scale = old_scale

    print()
    print('=== 接管机制检查 ===')
    e = env
    while hasattr(e, 'env'):
        e = e.env
    print('  is_intervening:', e.is_intervening)
    print('  空格监听器    :', '已启用' if e._space_listener is not None else '不可用')
    print('  get_intervene_action():', e.get_intervene_action())

    env.close()
    print()
    print('✅ 验证完成')
    return 0


if __name__ == '__main__':
    sys.exit(main())
