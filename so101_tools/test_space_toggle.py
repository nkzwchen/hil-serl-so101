#!/usr/bin/env python
"""验证 SO101Env 的「空格 toggle 接管」——

已确认采用 toggle 模式：**按一下进入接管，再按一下退出**。

用文件模拟（文件从"不存在→存在"算一次按键按下）来验证。
测试用 setsid 脱离 TTY，确保走文件后备路径。

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

FLAG = '/tmp/space_toggle_flag'
os.environ['SERL_SPACE_FILE'] = FLAG
os.environ.setdefault('SERL_SPACE_MODE', 'toggle')

if os.path.exists(FLAG):
    os.remove(FLAG)


def press():
    """模拟一次「按下空格」：文件从无到有。"""
    open(FLAG, 'w').close()
    time.sleep(0.25)          # 等轮询线程检测到边缘
    os.remove(FLAG)           # 复位（release，toggle 模式下不影响状态）
    time.sleep(0.25)


def main():
    from experiments.so101.config import TrainConfig

    print(f'stdin 是 TTY : {sys.stdin.isatty()}   (期望 False)')
    print(f'SERL_SPACE_MODE: {os.environ.get("SERL_SPACE_MODE")}')

    c = TrainConfig()
    env = c.get_environment(fake_env=False)
    e = env
    while hasattr(e, 'env'):
        e = e.env

    print(f'键盘后端 : {e._kb_mode}')
    print(f'空格模式 : {e.space_mode}')
    print()

    obs, info = env.reset()
    a = np.zeros(6, dtype=np.float32)

    def step_and_report(tag):
        obs, rew, done, trunc, info = env.step(a)
        iv = info.get('intervene_action')
        print(f'  {tag} | intervening={str(e.is_intervening):5} '
              f'| mode={str(e._last_mode):6} '
              f'| intervene_action={"有" if iv is not None else "无"}')
        if done:
            env.reset()
        return e.is_intervening

    print('=== toggle 序列：初始 → 按1次 → 按2次 → 按3次 ===')
    r = []
    r.append(step_and_report('初始      '))
    press(); r.append(step_and_report('按第1次   '))
    press(); r.append(step_and_report('按第2次   '))
    press(); r.append(step_and_report('按第3次   '))
    press(); r.append(step_and_report('按第4次   '))

    env.close()
    if os.path.exists(FLAG):
        os.remove(FLAG)

    print()
    print('=== 判定（期望：False → True → False → True → False）===')
    expect = [False, True, False, True, False]
    ok = r == expect
    print(f'  期望 {expect}')
    print(f'  实际 {r}')
    print()
    print('✅ toggle 模式正确（按一下进入、再按一下退出）' if ok else '❌ 与期望不符')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
