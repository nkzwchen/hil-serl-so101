#!/usr/bin/env python
"""干净验证 SERL_SPACE_FILE 后备路径（无 TTY 环境）。

背景：在带 PTY 的终端里 sys.stdin.isatty() 为真，会走 stdin 分支。
本脚本用 setsid 脱离控制终端，强制走「文件模拟」后备路径，
以验证该路径本身可用（自动化测试/无头环境会用到）。
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

FLAG = '/tmp/space_flag_clean'
os.environ['SERL_SPACE_FILE'] = FLAG
if os.path.exists(FLAG):
    os.remove(FLAG)


def main():
    print(f'stdin 是 TTY: {sys.stdin.isatty()}   (期望 False)')
    if sys.stdin.isatty():
        print('  ⚠️ 仍有 TTY，本测试无法隔离文件分支')
        return 2

    from experiments.so101.config import TrainConfig

    c = TrainConfig()
    env = c.get_environment(fake_env=False)
    e = env
    while hasattr(e, 'env'):
        e = e.env

    print(f'键盘后端: {e._kb_mode}   (期望 stdin —— 因为 fileno 可用但非 TTY)')
    print(f'模拟文件: {FLAG}')
    print()

    obs, info = env.reset()
    results = []
    for press in (False, True, True, False):
        if press:
            open(FLAG, 'w').close()
        elif os.path.exists(FLAG):
            os.remove(FLAG)
        time.sleep(0.35)
        a = np.zeros(6, dtype=np.float32)
        obs, rew, done, trunc, info = env.step(a)
        has_iv = 'intervene_action' in info
        results.append((press, e.is_intervening, e._last_mode, has_iv))
        print(f'  空格={"按下" if press else "松开":4} -> is_intervening={str(e.is_intervening):5} '
              f'mode={str(e._last_mode):6} intervene_action={"有" if has_iv else "无"}')

    env.close()
    if os.path.exists(FLAG):
        os.remove(FLAG)

    # 判定
    ok = (
        results[0][1] is False and results[0][2] == 'policy' and results[0][3] is False
        and results[1][1] is True and results[1][2] == 'human'
        and results[3][1] is False and results[3][2] == 'policy'
    )
    print()
    print('✅ 文件后备路径可用' if ok else '❌ 判定不通过')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
