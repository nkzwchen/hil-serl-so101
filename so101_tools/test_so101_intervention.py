#!/usr/bin/env python
"""验证 SO101Env 的接管状态机（用 SERL_SPACE_FILE 模拟空格）。

为什么用文件模拟：
  WSL2 无头环境下 pynput 不可用，而 stdin 在非交互/管道场景也不是 TTY。
  所以 env 提供了一个后备：SERL_SPACE_FILE 指定的文件**存在** == 空格按下。
  这样接管逻辑可以被自动化验证，不需要真人按键盘。

注意：本测试假设 so101_rl_env.launch.py 已在跑（无相机即可）。
"""

import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

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

FLAG = os.environ.get('SERL_SPACE_FILE', '/tmp/space_flag')
os.environ['SERL_SPACE_FILE'] = FLAG


def set_space(down: bool):
    if down:
        open(FLAG, 'w').close()
    elif os.path.exists(FLAG):
        os.remove(FLAG)


def main():
    from experiments.so101.config import TrainConfig

    c = TrainConfig()
    env = c.get_environment(fake_env=False)
    e = env
    while hasattr(e, 'env'):
        e = e.env

    print('=== 键盘后端 ===')
    print(f'  _kb_mode : {e._kb_mode}   (期望 stdin 或 None)')
    print(f'  模拟文件 : {FLAG}')

    print()
    print('=== 初始状态 ===')
    obs, info = env.reset()
    set_space(False)
    time.sleep(0.3)
    print(f'  is_intervening={e.is_intervening}  mode={e._last_mode}')

    print()
    print('=== 状态机测试：松开 → 按下 → 按住 → 松开 ===')
    seq = [False, True, True, True, False, False]
    for i, press in enumerate(seq):
        set_space(press)
        time.sleep(0.3)                      # 等文件轮询线程更新
        a = np.zeros(6, dtype=np.float32)
        a[0] = 0.1                           # 很小的动作
        obs, rew, done, trunc, info = env.step(a)
        has_iv = 'intervene_action' in info
        iv = info.get('intervene_action')
        iv_str = 'None' if iv is None else np.round(iv, 3).tolist()
        print(f'  [{i}] 空格={"按下" if press else "松开":4} | '
              f'is_intervening={str(e.is_intervening):5} | '
              f'mode={str(e._last_mode):6} | '
              f'info.intervening={str(info.get("intervening")):5} | '
              f'intervene_action={iv_str}')
        if done:
            print('       (episode 结束，reset)')
            obs, info = env.reset()

    print()
    print('=== 结果判定 ===')
    ok = True
    # 按下时应进入 human 模式
    set_space(True)
    time.sleep(0.3)
    env.step(np.zeros(6, dtype=np.float32))
    if e._last_mode != 'human' or not e.is_intervening:
        print('  ❌ 按下空格后未进入 human 模式')
        ok = False
    else:
        print('  ✅ 按下空格 → human 模式（leader 力矩应已 OFF）')
    # 松开后应回到 policy
    set_space(False)
    time.sleep(0.3)
    env.step(np.zeros(6, dtype=np.float32))
    if e._last_mode != 'policy' or e.is_intervening:
        print('  ❌ 松开空格后未回到 policy 模式')
        ok = False
    else:
        print('  ✅ 松开空格 → policy 模式（leader 力矩应已 ON）')

    env.close()
    set_space(False)
    print()
    print('✅ 通过' if ok else '⚠️ 有问题，见上')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
